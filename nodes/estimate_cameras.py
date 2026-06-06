"""Camera estimation adapters for multi-image SHARP workflows."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from comfy_api.latest import io

from .load_model import _comfy_tqdm
from .utils.image import convert_focallength

log = logging.getLogger("sharp")

try:
    import folder_paths

    VGGT_MODELS_DIR = os.path.join(folder_paths.models_dir, "vggt")
    os.makedirs(VGGT_MODELS_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("vggt", VGGT_MODELS_DIR)
except ImportError:
    VGGT_MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "vggt",
    )

_VGGT_MODEL = None
_VGGT_MODEL_ID = None
DEFAULT_VGGT_MODEL_ID = "facebook/VGGT-1B"


def _as_extrinsics_matrix(value: Any, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.shape == (3, 4):
        matrix = torch.cat([matrix, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape 3x4 or 4x4, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _as_intrinsics_matrix(value: Any, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.shape == (3, 3):
        homogeneous = torch.eye(4, dtype=torch.float32)
        homogeneous[:3, :3] = matrix
        matrix = homogeneous
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape 3x3 or 4x4, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    return matrix


def _build_intrinsics(
    batch_size: int,
    width: int,
    height: int,
    focal_length_mm: float,
    fov_degrees: float,
) -> torch.Tensor | None:
    if focal_length_mm > 0:
        focal_px = float(convert_focallength(width, height, focal_length_mm))
    elif fov_degrees > 0:
        focal_px = (width * 0.5) / math.tan(math.radians(fov_degrees) * 0.5)
    else:
        return None

    intrinsics = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = focal_px
    intrinsics[:, 1, 1] = focal_px
    intrinsics[:, 0, 2] = width * 0.5
    intrinsics[:, 1, 2] = height * 0.5
    return intrinsics


def _parse_camera_json(
    camera_json: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not camera_json.strip():
        raise ValueError("camera_json is required when method is 'json'")

    try:
        data = json.loads(camera_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid camera JSON: {exc}") from exc

    default_convention = data.get("convention", "world_to_camera")
    extrinsics_values = []
    intrinsics_values = []

    if "cameras" in data:
        cameras = data["cameras"]
        if len(cameras) != batch_size:
            raise ValueError(
                f"Camera JSON contains {len(cameras)} cameras, but image batch has {batch_size}"
            )
        shared_intrinsics = data.get("intrinsics", data.get("intrinsic"))
        for index, camera in enumerate(cameras):
            convention = camera.get("convention", default_convention)
            extrinsics_value = camera.get(
                "extrinsics",
                camera.get("extrinsic", camera.get("world_to_camera", camera.get("w2c"))),
            )
            if extrinsics_value is None:
                extrinsics_value = camera.get("camera_to_world", camera.get("c2w"))
                convention = "camera_to_world"
            if extrinsics_value is None:
                raise ValueError(f"Camera {index} has no extrinsics matrix")

            extrinsics = _as_extrinsics_matrix(extrinsics_value, f"camera[{index}].extrinsics")
            if convention in {"camera_to_world", "c2w"}:
                extrinsics = torch.linalg.inv(extrinsics)
            elif convention not in {"world_to_camera", "w2c"}:
                raise ValueError(f"Unsupported camera convention: {convention}")
            extrinsics_values.append(extrinsics)

            intrinsics_value = camera.get(
                "intrinsics", camera.get("intrinsic", shared_intrinsics)
            )
            if (
                intrinsics_value is shared_intrinsics
                and shared_intrinsics is not None
                and torch.as_tensor(shared_intrinsics).ndim == 3
            ):
                if len(shared_intrinsics) != batch_size:
                    raise ValueError(
                        f"Camera JSON contains {len(shared_intrinsics)} intrinsics, "
                        f"but image batch has {batch_size}"
                    )
                intrinsics_value = shared_intrinsics[index]
            if intrinsics_value is not None:
                intrinsics_values.append(
                    _as_intrinsics_matrix(intrinsics_value, f"camera[{index}].intrinsics")
                )
    else:
        raw_extrinsics = data.get("extrinsics")
        if raw_extrinsics is None:
            raise ValueError("Camera JSON must contain 'cameras' or 'extrinsics'")
        if len(raw_extrinsics) != batch_size:
            raise ValueError(
                f"Camera JSON contains {len(raw_extrinsics)} extrinsics, "
                f"but image batch has {batch_size}"
            )
        for index, value in enumerate(raw_extrinsics):
            extrinsics = _as_extrinsics_matrix(value, f"extrinsics[{index}]")
            if default_convention in {"camera_to_world", "c2w"}:
                extrinsics = torch.linalg.inv(extrinsics)
            elif default_convention not in {"world_to_camera", "w2c"}:
                raise ValueError(f"Unsupported camera convention: {default_convention}")
            extrinsics_values.append(extrinsics)

        raw_intrinsics = data.get("intrinsics", data.get("intrinsic"))
        if raw_intrinsics is not None:
            candidate = torch.as_tensor(raw_intrinsics)
            if candidate.ndim == 2:
                intrinsics_values = [
                    _as_intrinsics_matrix(raw_intrinsics, "intrinsics")
                ] * batch_size
            else:
                if len(raw_intrinsics) != batch_size:
                    raise ValueError(
                        f"Camera JSON contains {len(raw_intrinsics)} intrinsics, "
                        f"but image batch has {batch_size}"
                    )
                intrinsics_values = [
                    _as_intrinsics_matrix(value, f"intrinsics[{index}]")
                    for index, value in enumerate(raw_intrinsics)
                ]

    if intrinsics_values and len(intrinsics_values) != batch_size:
        raise ValueError("Intrinsics must be provided for every camera or omitted entirely")

    extrinsics = torch.stack(extrinsics_values)
    intrinsics = torch.stack(intrinsics_values) if intrinsics_values else None
    return extrinsics, intrinsics


def _normalize_extrinsics(
    extrinsics: torch.Tensor,
    normalize_scene: bool,
    camera_scale: float,
) -> torch.Tensor:
    result = extrinsics.clone()
    if normalize_scene:
        result = result @ torch.linalg.inv(result[0])
        camera_to_world = torch.linalg.inv(result)
        distances = torch.linalg.vector_norm(camera_to_world[:, :3, 3], dim=-1)
        nonzero_distances = distances[distances > 1e-6]
        if len(nonzero_distances) > 0:
            baseline = torch.median(nonzero_distances)
            result[:, :3, 3] /= baseline
        else:
            log.warning("All estimated cameras share one center; scene scale was not normalized")

    result[:, :3, 3] *= camera_scale
    return result


def _preprocess_for_vggt(
    images: torch.Tensor,
) -> tuple[torch.Tensor, float, float, float, float]:
    _, height, width, _ = images.shape
    target_size = 518
    scale = target_size / max(height, width)
    resized_height = max(14, round(height * scale / 14) * 14)
    resized_width = max(14, round(width * scale / 14) * 14)
    resized_height = min(target_size, resized_height)
    resized_width = min(target_size, resized_width)

    chw = images[..., :3].movedim(-1, 1).float()
    resized = F.interpolate(
        chw,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    pad_top = (target_size - resized_height) // 2
    pad_left = (target_size - resized_width) // 2
    padded = F.pad(
        resized,
        (
            pad_left,
            target_size - resized_width - pad_left,
            pad_top,
            target_size - resized_height - pad_top,
        ),
        value=1.0,
    )
    scale_x = resized_width / width
    scale_y = resized_height / height
    return padded, scale_x, scale_y, float(pad_left), float(pad_top)


def _get_vggt_model(model_id: str):
    global _VGGT_MODEL, _VGGT_MODEL_ID
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise RuntimeError(
            "VGGT method is optional and is not installed. Install the official package "
            "with: python -m pip install "
            "\"git+https://github.com/facebookresearch/vggt.git\" "
            "or use method='json'."
        ) from exc

    if _VGGT_MODEL is None or _VGGT_MODEL_ID != model_id:
        model_dir = os.path.join(VGGT_MODELS_DIR, model_id.replace("/", "--"))
        model_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(model_path):
            os.makedirs(model_dir, exist_ok=True)
            log.info(
                "Downloading VGGT checkpoint to %s: %s (about 5 GB)",
                model_dir,
                model_id,
            )
            download_kwargs = {
                "repo_id": model_id,
                "filename": "model.safetensors",
                "local_dir": model_dir,
            }
            if "tqdm_class" in inspect.signature(hf_hub_download).parameters:
                download_kwargs["tqdm_class"] = _comfy_tqdm()
            model_path = hf_hub_download(**download_kwargs)
        else:
            log.info("Using local VGGT checkpoint: %s", model_path)

        log.info("Initializing VGGT architecture without allocating weights")
        original_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            # VGGT calls .item() on this initialization-only tensor.
            kwargs["device"] = "cpu"
            return original_linspace(*args, **kwargs)

        torch.linspace = _cpu_linspace
        try:
            with torch.device("meta"):
                model = VGGT()
        finally:
            torch.linspace = original_linspace

        log.info("Loading VGGT checkpoint from %s", model_path)
        state_dict = load_safetensors(model_path, device="cpu")
        incompatible = model.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
        del state_dict
        if incompatible.missing_keys or incompatible.unexpected_keys:
            log.warning(
                "VGGT checkpoint mismatch: missing=%d, unexpected=%d",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

        for name, buffer in list(model.named_buffers()):
            if buffer.device.type == "meta":
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent._buffers[parts[-1]] = torch.zeros_like(buffer, device="cpu")

        _VGGT_MODEL = model.eval()
        _VGGT_MODEL_ID = model_id
    return _VGGT_MODEL


def _estimate_with_vggt(
    images: torch.Tensor,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.shape[0] < 2:
        raise ValueError("VGGT camera estimation requires at least two images")

    model_management = None
    try:
        import comfy.model_management as model_management
        device = model_management.get_torch_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared, scale_x, scale_y, pad_left, pad_top = _preprocess_for_vggt(images)

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        inference_dtype = torch.bfloat16 if major >= 8 else torch.float16
        autocast = torch.autocast(device_type="cuda", dtype=inference_dtype)
    else:
        inference_dtype = torch.float32
        autocast = nullcontext()

    model = _get_vggt_model(model_id)
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    log.info("Moving VGGT camera model to %s with %s", device, inference_dtype)
    model.to(device=device, dtype=inference_dtype)
    prepared = prepared.to(device=device, dtype=inference_dtype)

    scene_images = None
    aggregated_tokens = None
    pose_encoding = None
    try:
        with torch.inference_mode(), autocast:
            scene_images = prepared.unsqueeze(0)
            aggregated_tokens, _ = model.aggregator(scene_images)
            pose_encoding = model.camera_head(aggregated_tokens)[-1].float()
        extrinsics_3x4, intrinsics_3x3 = pose_encoding_to_extri_intri(
            pose_encoding,
            prepared.shape[-2:],
        )
        extrinsics_3x4 = extrinsics_3x4[0].cpu()
        intrinsics_3x3 = intrinsics_3x3[0].cpu()
    finally:
        model.cpu()
        del prepared, scene_images, aggregated_tokens, pose_encoding
        if model_management is not None:
            model_management.soft_empty_cache()

    batch_size = images.shape[0]
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
    extrinsics[:, :3, :4] = extrinsics_3x4

    intrinsics = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
    intrinsics[:, :3, :3] = intrinsics_3x3
    intrinsics[:, 0, 0] /= scale_x
    intrinsics[:, 1, 1] /= scale_y
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / scale_x
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / scale_y
    return extrinsics, intrinsics


def _cameras_to_json(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    method: str,
    normalized: bool,
    camera_scale: float,
) -> str:
    cameras = []
    for index in range(extrinsics.shape[0]):
        world_to_camera = extrinsics[index]
        camera_to_world = torch.linalg.inv(world_to_camera)
        rotation = camera_to_world[:3, :3]
        cameras.append(
            {
                "index": index,
                "position": camera_to_world[:3, 3].tolist(),
                "forward": rotation[:, 2].tolist(),
                "up": (-rotation[:, 1]).tolist(),
                "right": rotation[:, 0].tolist(),
                "world_to_camera": world_to_camera.tolist(),
                "camera_to_world": camera_to_world.tolist(),
                "intrinsics": intrinsics[index].tolist(),
            }
        )
    return json.dumps(
        {
            "convention": "OpenCV world_to_camera; x=right, y=down, z=forward",
            "method": method,
            "normalized_scene": normalized,
            "camera_scale": camera_scale,
            "cameras": cameras,
        },
        indent=2,
    )


class EstimateCamerasFromImages(io.ComfyNode):
    """Estimate or import cameras for an image batch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EstimateCamerasFromImages",
            display_name="Estimate Cameras from Images",
            category="SHARP",
            description=(
                "Estimate OpenCV world-to-camera matrices for a multi-image batch. "
                "VGGT is optional; JSON import works without extra dependencies."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["vggt", "json"], default="vggt"),
                io.Float.Input(
                    "focal_length_mm",
                    default=0.0,
                    min=0.0,
                    max=500.0,
                    step=0.1,
                    optional=True,
                    tooltip="35mm-equivalent focal length override. 0 uses method intrinsics.",
                ),
                io.Float.Input(
                    "fov_degrees",
                    default=0.0,
                    min=0.0,
                    max=179.0,
                    step=0.1,
                    optional=True,
                    tooltip="Horizontal FOV override. Used only when focal_length_mm is 0.",
                ),
                io.Boolean.Input(
                    "normalize_scene",
                    default=True,
                    optional=True,
                    tooltip="Make camera 0 the world origin and normalize median baseline to 1.",
                ),
                io.Float.Input(
                    "camera_scale",
                    default=1.0,
                    min=0.001,
                    max=1000.0,
                    step=0.01,
                    optional=True,
                    tooltip="Scale camera translations after optional scene normalization.",
                ),
                io.String.Input(
                    "camera_json",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Camera data for method=json. Supports 3x4/4x4 extrinsics and 3x3/4x4 intrinsics.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.String.Output(display_name="cameras_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        method: str = "vggt",
        focal_length_mm: float = 0.0,
        fov_degrees: float = 0.0,
        normalize_scene: bool = True,
        camera_scale: float = 1.0,
        camera_json: str = "",
    ):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE [N,H,W,C], got {tuple(image.shape)}")

        batch_size, height, width, _ = image.shape
        if method == "vggt":
            extrinsics, intrinsics = _estimate_with_vggt(
                image,
                DEFAULT_VGGT_MODEL_ID,
            )
        elif method == "json":
            extrinsics, intrinsics = _parse_camera_json(camera_json, batch_size)
        else:
            raise ValueError(f"Unsupported camera estimation method: {method}")

        override_intrinsics = _build_intrinsics(
            batch_size,
            width,
            height,
            focal_length_mm,
            fov_degrees,
        )
        if override_intrinsics is not None:
            intrinsics = override_intrinsics
        elif intrinsics is None:
            raise ValueError(
                "The selected method did not provide intrinsics. "
                "Set focal_length_mm or fov_degrees."
            )

        extrinsics = _normalize_extrinsics(
            extrinsics.float(),
            normalize_scene=normalize_scene,
            camera_scale=camera_scale,
        )
        intrinsics = intrinsics.float()

        log.info(
            "Camera adapter: method=%s, images=%d, normalize=%s, scale=%.4f",
            method,
            batch_size,
            normalize_scene,
            camera_scale,
        )
        output_json = _cameras_to_json(
            extrinsics,
            intrinsics,
            method,
            normalize_scene,
            camera_scale,
        )
        return io.NodeOutput(image, extrinsics, intrinsics, output_json)


NODE_CLASS_MAPPINGS = {
    "EstimateCamerasFromImages": EstimateCamerasFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EstimateCamerasFromImages": "Estimate Cameras from Images",
}
