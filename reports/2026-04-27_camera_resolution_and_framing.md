# 2026-04-27 — Camera resolution + framing rework

**Type:** env-cfg change (NOT a training run). Invalidates all prior datasets and checkpoints that consumed `wrist_cam` / `third_person_cam` observations, since both stream shapes and the third-person extrinsics have changed.

## Hypothesis

The third-person camera in the prior config (`pos=(1.15,0.10,0.50)`, `focal_length=24` → FOV ~47°, 256×256) failed to keep the cube in-frame in 3 of 5 random spawns ([reports/camera_samples/2026-04-27_current_scene/](camera_samples/2026-04-27_current_scene/)). Combined with a 128×128 wrist stream that gives SmolVLA ~30–50 px on the cube at grasp distance, both image streams were under-resolved relative to the policy's 512×512 padded input. Two changes:

1. Bump native resolutions: **wrist 128→256, third-person 256→512**. SmolVLA pads inputs to 512×512 internally, so 512 native eliminates upscaling on third-person and halves it on wrist.
2. Reposition third-person to **`pos=(1.5,-0.10,0.80)`, `focal_length=18` (FOV ~60°)**, aimed at the centroid of `[spawn box ∪ target marker]` = `(0.55,-0.10,0.025)`.

## Config

| Stream | Field | Before | After |
|---|---|---|---|
| `wrist_cam` | `height`, `width` | 128, 128 | 256, 256 |
| `third_person_cam` | `height`, `width` | 256, 256 | 512, 512 |
| `third_person_cam` | `pos` | `(1.15, 0.10, 0.50)` | `(1.5, -0.10, 0.80)` |
| `third_person_cam` | `rot (wxyz)` | `(-0.2926, 0.64373, 0.64373, -0.2926)` | `(-0.30326, 0.63877, 0.63877, -0.30326)` |
| `third_person_cam` | `focal_length` | 24.0 mm | 18.0 mm |

Quaternion computed via `/tmp/cam_lookat.py` (ROS convention, world-up = +Z).

Geometry of new third-person view (ground-plane intersection at z=0.025):
- Bottom-of-frame ray meets table at X≈1.21, top-of-frame ray well past table.
- View-axis distance to centroid: 1.226 m.
- Horizontal half-extent at view-axis depth: ±0.715 m about Y=-0.10 → covers Y∈[-0.82, 0.62] (target Y=0.20 ✓, spawn Y∈[-0.40,0.00] ✓).

## Code touch-points (all on branch `vla-pipeline-v1`)

- `envs/yaskawa_pick_cube_cfg.py`: both camera CameraCfgs.
- `scripts/data/isaaclab_to_lerobot.py`: docstring shapes + `observation.images.{wrist,third_person}` feature shapes.
- `scripts/data/render_demo.py`: side-by-side hstack made dimension-agnostic.
- `scripts/train/run_vla_closed_loop.py`: GIF hstack made dimension-agnostic.

## Result

- Render shapes verified: `wrist=(256,256,3)`, `third=(512,512,3)`. Outputs in [reports/camera_samples/2026-04-27_hires/](camera_samples/2026-04-27_hires/) (resolution-only) and [reports/camera_samples/2026-04-27_repositioned/](camera_samples/2026-04-27_repositioned/) (final pose).
- 8 randomized cube poses spanning X∈[0.266, 0.830], Y∈[-0.328, -0.024]: cube + target marker + robot arm all in-frame in every sample. Worst-case (sample_06, cube at X=0.266 — the far -X spawn-box edge) places the cube directly under the arm but still fully visible.

## Lesson

Don't trust a camera pose just because the on-axis target is centered. `target_color_stability_probe` and `compare_train_eval_obs` only sampled cube poses around the spawn-box mean; they would have missed the corner-cropping that `save_camera_samples` with N=5 caught accidentally. **Going forward, camera-pose sanity checks should sweep the corners of the spawn box, not the center.**

## Next step

Regenerate the scripted demo dataset under the new camera config, then re-run the v5 SmolVLA training. The 4× pixel volume on third-person and 4× on wrist will increase converter time and disk footprint roughly proportionally — re-measure during the regen so the orchestrate scripts know the new per-demo cost. Tracked under bd issues that the user will create at the start of that run.
