import struct

from nodes.merge_gaussians import _read_binary_ply_layout, merge_binary_ply_files


PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _write_test_ply(path, records):
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(records)}\n"
    for name in PLY_PROPERTIES:
        header += f"property float {name}\n"
    header += "end_header\n"

    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        for record in records:
            file.write(struct.pack("<14f", *record))


def test_binary_ply_merge_preserves_payload(tmp_path):
    records_a = [tuple(float(i) for i in range(14))]
    records_b = [
        tuple(float(i + 14) for i in range(14)),
        tuple(float(i + 28) for i in range(14)),
    ]
    path_a = tmp_path / "a.ply"
    path_b = tmp_path / "b.ply"
    output_path = tmp_path / "merged.ply"
    _write_test_ply(path_a, records_a)
    _write_test_ply(path_b, records_b)

    progress = []
    count = merge_binary_ply_files(
        [path_a, path_b],
        output_path,
        progress_callback=lambda value, total: progress.append((value, total)),
    )

    output_layout = _read_binary_ply_layout(output_path)
    input_layouts = [_read_binary_ply_layout(path) for path in (path_a, path_b)]
    expected_payload = b"".join(
        path.read_bytes()[layout["header_size"] :]
        for path, layout in zip((path_a, path_b), input_layouts)
    )
    actual_payload = output_path.read_bytes()[output_layout["header_size"] :]

    assert count == 3
    assert output_layout["vertex_count"] == 3
    assert actual_payload == expected_payload
    assert progress[-1][0] == progress[-1][1] == len(expected_payload)
