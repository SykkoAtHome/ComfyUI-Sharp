"""MergeGaussians node for ComfyUI-Sharp.

Merges multiple PLY files (from panorama samples) into a single unified Gaussian scene.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

log = logging.getLogger("sharp")

_PLY_TYPE_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def _read_binary_ply_layout(path: Path) -> dict:
    """Read enough PLY metadata to concatenate compatible binary payloads."""
    with path.open("rb") as file:
        header_bytes = bytearray()
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_bytes.extend(line)
            if line.rstrip(b"\r\n") == b"end_header":
                break

    header = header_bytes.decode("ascii")
    lines = header.splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Fast merge requires binary little-endian PLY: {path}")

    vertex_line_index = None
    vertex_count = None
    record_size = 0
    in_vertex_element = False
    for index, line in enumerate(lines):
        if line.startswith("element "):
            parts = line.split()
            in_vertex_element = len(parts) == 3 and parts[1] == "vertex"
            if in_vertex_element:
                vertex_line_index = index
                vertex_count = int(parts[2])
            elif vertex_line_index is not None:
                raise ValueError(f"Fast merge does not support extra PLY elements: {path}")
        elif in_vertex_element and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError(f"Fast merge requires scalar vertex properties: {path}")
            try:
                record_size += _PLY_TYPE_SIZES[parts[1]]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported PLY property type {parts[1]}: {path}"
                ) from exc

    if vertex_line_index is None or vertex_count is None or record_size == 0:
        raise ValueError(f"PLY has no supported vertex element: {path}")

    payload_size = path.stat().st_size - len(header_bytes)
    expected_payload_size = vertex_count * record_size
    if payload_size != expected_payload_size:
        raise ValueError(
            f"Unexpected PLY payload size in {path}: "
            f"{payload_size} != {expected_payload_size}"
        )

    normalized_lines = list(lines)
    normalized_lines[vertex_line_index] = "element vertex {vertex_count}"
    return {
        "header_lines": lines,
        "normalized_header": "\n".join(normalized_lines),
        "vertex_line_index": vertex_line_index,
        "vertex_count": vertex_count,
        "header_size": len(header_bytes),
        "payload_size": payload_size,
    }


def merge_binary_ply_files(
    ply_files: list[Path],
    output_path: Path,
    progress_callback=None,
) -> int:
    """Merge compatible binary PLY files by streaming their vertex payloads."""
    layouts = [_read_binary_ply_layout(path) for path in ply_files]
    normalized_header = layouts[0]["normalized_header"]
    for path, layout in zip(ply_files[1:], layouts[1:]):
        if layout["normalized_header"] != normalized_header:
            raise ValueError(f"Incompatible PLY schema for fast merge: {path}")

    total_vertices = sum(layout["vertex_count"] for layout in layouts)
    total_bytes = sum(layout["payload_size"] for layout in layouts)
    output_lines = list(layouts[0]["header_lines"])
    output_lines[layouts[0]["vertex_line_index"]] = f"element vertex {total_vertices}"
    output_header = ("\n".join(output_lines) + "\n").encode("ascii")

    copied_bytes = 0
    chunk_size = 8 * 1024 * 1024
    with output_path.open("wb") as output_file:
        output_file.write(output_header)
        for path, layout in zip(ply_files, layouts):
            with path.open("rb") as input_file:
                input_file.seek(layout["header_size"])
                remaining = layout["payload_size"]
                while remaining:
                    chunk = input_file.read(min(chunk_size, remaining))
                    if not chunk:
                        raise ValueError(f"Unexpected end of PLY payload: {path}")
                    output_file.write(chunk)
                    remaining -= len(chunk)
                    copied_bytes += len(chunk)
                    if progress_callback is not None:
                        progress_callback(copied_bytes, total_bytes)

    return total_vertices


def load_ply_simple(path: str) -> dict:
    """Load Gaussian data from PLY file.

    Returns dict with arrays: positions, colors, scales, rotations, opacities
    """
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=-1)

    # Colors (SH coefficients - we take DC term)
    colors = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=-1)

    # Scales
    scales = np.stack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ], axis=-1)

    # Rotations (quaternion)
    rotations = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=-1)

    # Opacity
    opacities = vertex['opacity']

    return {
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
    }


def save_merged_ply(
    positions: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    output_path: str,
):
    """Save merged Gaussians to PLY file."""

    num_gaussians = len(positions)

    # Create structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    elements = np.empty(num_gaussians, dtype=dtype)
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['f_dc_0'] = colors[:, 0]
    elements['f_dc_1'] = colors[:, 1]
    elements['f_dc_2'] = colors[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]

    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


class MergeGaussians(io.ComfyNode):
    """Merge multiple Gaussian PLY files into a single scene.

    Used after running SHARP on panorama samples to combine all views.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MergeGaussians",
            display_name="Merge Gaussians (PLY Files)",
            category="SHARP",
            description="Merge multiple Gaussian PLY files into a single unified scene.",
            is_output_node=True,
            inputs=[
                io.String.Input("ply_folder",
                                tooltip="Path to folder containing PLY files to merge"),
                io.String.Input("output_prefix", default="merged", optional=True,
                                tooltip="Prefix for output merged PLY file"),
                io.Float.Input("max_depth", default=0.0, min=0.0, max=1000.0, step=0.1,
                               optional=True,
                               tooltip="Filter out Gaussians beyond this depth (0 = no filter)"),
                io.Float.Input("min_opacity", default=0.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Filter out Gaussians with opacity below this (0 = no filter)"),
            ],
            outputs=[
                io.String.Output(display_name="ply_path"),
                io.Int.Output(display_name="num_gaussians"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        ply_folder: str,
        output_prefix: str = "merged",
        max_depth: float = 0.0,
        min_opacity: float = 0.0,
    ):
        """Merge all PLY files in folder."""

        # Find all PLY files
        ply_folder = Path(ply_folder)
        if not ply_folder.exists():
            raise ValueError(f"PLY folder does not exist: {ply_folder}")

        ply_files = sorted(ply_folder.glob("*.ply"))
        if not ply_files:
            raise ValueError(f"No PLY files found in: {ply_folder}")

        log.info(f"Found {len(ply_files)} PLY files to merge")

        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = ply_folder.parent / output_filename

        def set_status(text: str):
            if cls.hidden.unique_id:
                from server import PromptServer
                PromptServer.instance.send_progress_text(text, cls.hidden.unique_id)

        # SHARP exports compatible binary PLY files. Without filters their
        # vertex payloads can be streamed directly, avoiding a large parse,
        # concatenate, and structured-array allocation.
        if max_depth <= 0 and min_opacity <= 0:
            import comfy.utils
            total_payload = sum(path.stat().st_size for path in ply_files)
            pbar = comfy.utils.ProgressBar(
                total_payload,
                node_id=cls.hidden.unique_id,
            )
            set_status(f"Merging {len(ply_files)} PLY files")
            try:
                num_gaussians = merge_binary_ply_files(
                    ply_files,
                    output_path,
                    progress_callback=pbar.update_absolute,
                )
            except ValueError as exc:
                log.info("Fast PLY merge unavailable (%s); using parsed merge", exc)
            else:
                pbar.update_absolute(total_payload)
                log.info(
                    f"Done! Merged {len(ply_files)} files into "
                    f"{num_gaussians:,} Gaussians"
                )
                set_status(
                    f"Done: {num_gaussians:,} Gaussians from {len(ply_files)} files"
                )
                return io.NodeOutput(str(output_path), num_gaussians)

        # Load all PLY files
        import comfy.utils
        parsed_pbar = comfy.utils.ProgressBar(
            len(ply_files) + 1,
            node_id=cls.hidden.unique_id,
        )
        all_positions = []
        all_colors = []
        all_scales = []
        all_rotations = []
        all_opacities = []

        for i, ply_path in enumerate(ply_files):
            log.info(f"Loading {ply_path.name} ({i+1}/{len(ply_files)})")
            set_status(f"Loading PLY {i + 1}/{len(ply_files)}")
            data = load_ply_simple(str(ply_path))

            positions = data['positions']
            colors = data['colors']
            scales = data['scales']
            rotations = data['rotations']
            opacities = data['opacities']

            # Apply filters
            mask = np.ones(len(positions), dtype=bool)

            if max_depth > 0:
                # Filter by depth (distance from origin)
                depths = np.linalg.norm(positions, axis=-1)
                mask &= depths <= max_depth

            if min_opacity > 0:
                mask &= opacities >= min_opacity

            filtered_count = (~mask).sum()
            if filtered_count > 0:
                log.info(f"  Filtered out {filtered_count:,} Gaussians")

            all_positions.append(positions[mask])
            all_colors.append(colors[mask])
            all_scales.append(scales[mask])
            all_rotations.append(rotations[mask])
            all_opacities.append(opacities[mask])
            parsed_pbar.update_absolute(i + 1)

        # Concatenate all
        merged_positions = np.concatenate(all_positions, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_scales = np.concatenate(all_scales, axis=0)
        merged_rotations = np.concatenate(all_rotations, axis=0)
        merged_opacities = np.concatenate(all_opacities, axis=0)

        num_gaussians = len(merged_positions)
        log.info(f"Total Gaussians after merge: {num_gaussians:,}")

        # Save merged PLY
        log.info(f"Saving to {output_path}")
        set_status(f"Saving {num_gaussians:,} Gaussians")
        save_merged_ply(
            merged_positions,
            merged_colors,
            merged_scales,
            merged_rotations,
            merged_opacities,
            str(output_path),
        )
        parsed_pbar.update_absolute(len(ply_files) + 1)

        log.info(f"Done! Merged {len(ply_files)} files into {num_gaussians:,} Gaussians")
        set_status(
            f"Done: {num_gaussians:,} Gaussians from {len(ply_files)} files"
        )

        return io.NodeOutput(str(output_path), num_gaussians)


NODE_CLASS_MAPPINGS = {
    "MergeGaussians": MergeGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeGaussians": "Merge Gaussians (PLY Files)",
}
