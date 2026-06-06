# Multi-image cameras

## Pipeline

Use two `ImageBatch` nodes to combine three same-size room photographs, then
connect:

```text
images -> EstimateCamerasFromImages -> SharpPredict -> MergeGaussians
```

Connect `images`, `extrinsics`, and `intrinsics` from the camera node to the
matching `SharpPredict` inputs. A batch prediction creates a folder containing
one world-space PLY per image. Pass that folder path to `MergeGaussians`.

`SharpPredict.save_folder_ply` is optional. When empty, files are written under
the ComfyUI temporary directory. When set, it must point to an existing
directory. Batch runs create a timestamped subdirectory there; the
`save_folder_ply` output returns the exact directory containing the current
run's PLY files.

The final outputs are:

- one merged Gaussian Splat PLY,
- `cameras_json` with position, forward, up, right, world-to-camera,
  camera-to-world, and intrinsics for every input image.

The SykkoTools `Sharp Gaussian Render` viewer can use these directly:

```text
MergeGaussians.ply_path -> Sharp Gaussian Render.ply_path
EstimateCamerasFromImages.cameras_json -> Sharp Gaussian Render.cameras_json
```

Set `camera_index` to choose which input camera initializes the viewer.

## Camera convention

The SHARP adapter uses OpenCV cameras:

- extrinsics are world-to-camera matrices,
- camera axes are +X right, +Y down, +Z forward,
- intrinsics are in pixels,
- matrices accepted from JSON may be 3x4/3x3 or homogeneous 4x4/4x4.

`SamplePanorama`, `EstimateCamerasFromImages`, and `SharpPredict` use this same
contract.

## Backends

### VGGT

The optional `vggt` method estimates a separate pose and intrinsics matrix for
each image. Input images are resized and padded for VGGT; predicted intrinsics
are mapped back to the original ComfyUI image size.

VGGT is loaded lazily and moved back to CPU after estimation so SHARP can use
the GPU. Install the official package separately:

```bash
python -m pip install "git+https://github.com/facebookresearch/vggt.git"
```

Run this command in the Python environment that executes the ComfyUI-Sharp
nodes. This may be an isolated `comfy-env` environment when isolation is
enabled.

VGGT checkpoints are stored under:

```text
ComfyUI/models/vggt/<repository-name>/model.safetensors
```

### JSON

The `json` method adds no dependency. It accepts either a `cameras` list:

```json
{
  "convention": "world_to_camera",
  "cameras": [
    {
      "extrinsics": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ],
      "intrinsics": [
        [1200, 0, 768],
        [0, 1200, 512],
        [0, 0, 1]
      ]
    }
  ]
}
```

or top-level arrays:

```json
{
  "convention": "camera_to_world",
  "extrinsics": [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  ],
  "intrinsics": [
    [[1200, 0, 768], [0, 1200, 512], [0, 0, 1]]
  ]
}
```

The number and order of cameras must match the image batch. Shared intrinsics
may be supplied as one matrix. Intrinsics may also be omitted when
`focal_length_mm` or `fov_degrees` is set on the node.

## Scene scale

With `normalize_scene=true`, camera 0 becomes the world origin and the median
non-zero distance from camera 0 is normalized to one. `camera_scale` then
multiplies all camera translations. Disable normalization to preserve imported
translation units.

Camera estimation recovers relative scene scale. SHARP predicts each image
independently, so correct camera poses do not guarantee perfect depth agreement
in weakly overlapping, reflective, textureless, or moving regions. Use
`camera_scale` to tune pose baseline against SHARP depth and capture images with
substantial overlap.
