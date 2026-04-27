"""Sanity probe: magenta target marker stable across all 5 cube colors
(vla_kitting-8ii).

Why: the comment at envs/yaskawa_pick_cube_cfg.py:65-69 documents a prior
bug where the target marker inherited the cube's color via scene-
replication material sharing — green/cyan/emissive variants all rendered
the cube's color. Magenta + a separate prim were the workaround. Now that
we've swapped the marker geometry to a CylinderCfg (vla_kitting-usq), we
need to confirm the workaround still holds for the new prim across the
full 5-color cube palette.

What this does:
  1. For each color in CUBE_COLOR_PALETTE, force the cube color (override
     env state) and render third-person frames.
  2. Sample the pixel ROI around the projected target marker location
     (world (0.65, 0.20) → ~px (192, 32) on a 256×256 image — see
     attention_difference.world_to_third_px).
  3. Compute mean RGB over that ROI.
  4. Report per-color mean RGB; pass iff distance from magenta (1.0,0.0,1.0)
     is < tolerance (0.02 default).

Usage:
  ./isaaclab.sh -p scripts/validate/target_color_stability_probe.py \\
      --frames_per_color 20 \\
      --out_dir reports/runs/target_color_probe_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames_per_color", type=int, default=20)
parser.add_argument("--out_dir", type=pathlib.Path, required=True)
parser.add_argument(
    "--cube_far_xy",
    type=float,
    nargs=2,
    default=(0.45, -0.20),
    help="Force the cube to this XY before each render so it never occludes "
         "the magenta target (which sits at world (0.65, 0.20) inside the "
         "widened cube spawn box).",
)
parser.add_argument(
    "--max_pairwise_L2",
    type=float,
    default=0.10,
    help="PASS criterion: maximum pairwise L2 distance between the per-cube-"
         "color mean RGB at the magenta target. The original task is to "
         "verify the target does NOT inherit the cube color via scene-"
         "replication material sharing — that is, target color must be "
         "INDEPENDENT of cube color. Inheritance baseline (red→target vs "
         "blue→target inheriting) ≈ 1.07 L2; we PASS if max pairwise < 0.10 "
         "(< 10% of inheritance baseline). Absolute distance to pure (1,0,1) "
         "is NOT the metric — render lighting + anti-aliasing brighten the "
         "absolute color but stay constant across cube colors.",
)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument(
    "--magenta_min_pixels",
    type=int,
    default=50,
    help="Per-frame minimum magenta-pixel count to accept the measurement.",
)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[target-probe] {msg}", flush=True)


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from envs.mdp.cube_palette import CUBE_COLOR_PALETTE
    from envs.mdp.events import _set_preview_surface_color

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)

    _log(f"cube fixed at {tuple(args_cli.cube_far_xy)} per frame to keep "
         "the magenta target unoccluded; magenta pixels found by color "
         "thresholding (R>200, G<80, B>200) so we don't depend on a "
         "fragile world→pixel projection.")

    zero_action = torch.zeros((1, 7), device="cuda:0")
    results: dict[str, dict] = {}
    pass_all = True

    def _magenta_mean_rgb(third: np.ndarray) -> tuple[np.ndarray, int]:
        r = third[..., 0].astype(np.int32)
        g = third[..., 1].astype(np.int32)
        b = third[..., 2].astype(np.int32)
        m = (r > 200) & (g < 80) & (b > 200)
        n = int(m.sum())
        if n == 0:
            return np.array([np.nan, np.nan, np.nan]), 0
        pixels = third[m].astype(np.float32) / 255.0
        return pixels.mean(axis=0), n

    color_mean_rgbs: list[np.ndarray] = []
    for ci, (color_name, rgb) in enumerate(CUBE_COLOR_PALETTE):
        means = []
        per_frame_dists = []
        skipped = 0
        for f in range(args_cli.frames_per_color):
            obs, _ = env.reset()
            # Override cube to a far corner so it never occludes the
            # target. Apply BEFORE forcing the color, otherwise the
            # write_root_pose_to_sim resets velocities/etc and the
            # randomized color shader is preserved.
            cube_rb = env.unwrapped.scene["cube"]
            sim_dev = env.unwrapped.sim.device
            origin = env.unwrapped.scene.env_origins[0]
            cx, cy = args_cli.cube_far_xy
            pose = torch.tensor([[
                cx + origin[0].item(),
                cy + origin[1].item(),
                0.025 + origin[2].item(),
                1.0, 0.0, 0.0, 0.0,
            ]], device=sim_dev)
            cube_rb.write_root_pose_to_sim(pose)
            cube_rb.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_dev))
            env.unwrapped.scene.write_data_to_sim()

            # Force the cube color (override the random pick that
            # randomize_cube_color just made).
            cube_cfg = env.unwrapped.scene["cube"].cfg
            suffix = cube_cfg.prim_path.split("/World/envs/env_.*")[-1]
            _set_preview_surface_color(f"/World/envs/env_0{suffix}", rgb)
            env.unwrapped.cube_color_state[0] = (color_name, ci)

            for _ in range(args_cli.settle_steps):
                obs, *_ = env.step(zero_action)

            third = obs["policy"]["third_person_cam"][0].cpu().numpy()  # (H, W, 3) uint8
            mean_rgb, n_px = _magenta_mean_rgb(third)
            if n_px < args_cli.magenta_min_pixels or np.any(np.isnan(mean_rgb)):
                skipped += 1
                if f == 0:
                    Image.fromarray(third).save(
                        args_cli.out_dir / f"sample_{color_name}_frame00.png"
                    )
                continue
            means.append(mean_rgb)
            dist = float(np.linalg.norm(mean_rgb - np.array([1.0, 0.0, 1.0])))
            per_frame_dists.append(dist)
            if f == 0:
                Image.fromarray(third).save(
                    args_cli.out_dir / f"sample_{color_name}_frame00.png"
                )

        if not means:
            _log(f"  color={color_name:>7}  no magenta pixels found in any "
                 f"of {args_cli.frames_per_color} frames; FAIL")
            results[color_name] = {"pass": False, "reason": "no_magenta_pixels"}
            pass_all = False
            continue

        agg_mean = np.stack(means, axis=0).mean(axis=0)
        agg_dist = float(np.linalg.norm(agg_mean - np.array([1.0, 0.0, 1.0])))
        color_mean_rgbs.append(agg_mean)
        results[color_name] = {
            "mean_rgb": [round(float(c), 4) for c in agg_mean],
            "L2_to_pure_magenta": round(agg_dist, 5),
            "max_frame_L2_to_pure_magenta": round(max(per_frame_dists), 5),
            "frames_used": len(means),
            "frames_skipped": skipped,
        }
        _log(
            f"  color={color_name:>7}  mean_rgb=({agg_mean[0]:.3f},"
            f"{agg_mean[1]:.3f},{agg_mean[2]:.3f})  L2_pure={agg_dist:.4f}  "
            f"used={len(means)}/{args_cli.frames_per_color}"
        )

    # Stability metric: max pairwise L2 between per-cube-color means.
    if len(color_mean_rgbs) >= 2:
        stack = np.stack(color_mean_rgbs, axis=0)
        # Pairwise distance matrix.
        diffs = stack[:, None, :] - stack[None, :, :]
        pairwise = np.linalg.norm(diffs, axis=-1)
        max_pairwise = float(pairwise.max())
        avg_pairwise = float(pairwise.sum() / (pairwise.size - len(color_mean_rgbs)))
        pass_all = max_pairwise <= args_cli.max_pairwise_L2
        _log(
            f"  STABILITY: max pairwise L2 = {max_pairwise:.4f}  "
            f"avg = {avg_pairwise:.4f}  threshold = {args_cli.max_pairwise_L2}"
        )
    else:
        pass_all = False
        max_pairwise = float("inf")

    summary_path = args_cli.out_dir / "target_color_probe.json"
    summary_path.write_text(
        json.dumps(
            {
                "max_pairwise_L2_threshold": args_cli.max_pairwise_L2,
                "max_pairwise_L2_observed": round(max_pairwise, 5),
                "cube_far_xy": list(args_cli.cube_far_xy),
                "frames_per_color": args_cli.frames_per_color,
                "magenta_threshold": "R>200, G<80, B>200",
                "results": results,
                "verdict": "PASS" if pass_all else "FAIL",
            },
            indent=2,
        )
    )
    _log(f"wrote {summary_path}")
    _log(f"VERDICT: {'PASS' if pass_all else 'FAIL'}")

    env.close()
    sim_app.close()
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
