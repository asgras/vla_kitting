"""Render N=9 third-person + wrist camera frames at a fixed 3×3 cube XY
grid, plus the manifest the attention-difference probe consumes.

This is the data-capture side of vla_kitting-uxt. Outputs (under --out_dir):
  - frame_NN_third.png (256×256 third-person camera)
  - frame_NN_wrist.png (128×128 wrist camera)
  - cube_xys.json: list of {npz: 'frame_NN_data.npz', cube_xy: [x, y]}

After this runs, feed each frame through scripts/validate/attention_overlay.py
to produce frame_NN_data.npz, then aggregate via attention_difference.py:

    GRID_DIR=reports/runs/attn_diff_$(date +%Y%m%d)
    ./isaaclab.sh -p scripts/validate/render_cube_grid_for_attn_diff.py \\
        --out_dir $GRID_DIR
    for i in 0 1 2 3 4 5 6 7 8; do
        TAG=$(printf 'frame_%02d' $i)
        ./isaaclab.sh -p scripts/validate/attention_overlay.py \\
            --checkpoint <ckpt> \\
            --third_png "$GRID_DIR/${TAG}_third.png" \\
            --wrist_png "$GRID_DIR/${TAG}_wrist.png" \\
            --tag "$TAG" \\
            --out_dir "$GRID_DIR"
    done
    python scripts/validate/attention_difference.py \\
        --cube_xys "$GRID_DIR/cube_xys.json" \\
        --out_dir "$GRID_DIR"

The cube grid covers X ∈ {0.45, 0.55, 0.65}, Y ∈ {-0.15, 0.0, 0.15} —
nine evenly-spaced points inside the widened cube box but conservative
enough that the 2F-85 finger pads can reach all 9 (gate G1 cube spans).
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
parser.add_argument("--out_dir", type=pathlib.Path, required=True)
parser.add_argument("--settle_steps", type=int, default=20,
                    help="physics steps after each cube_xy override before capturing.")
parser.add_argument(
    "--xs",
    type=float,
    nargs="+",
    default=[0.45, 0.55, 0.65],
)
parser.add_argument(
    "--ys",
    type=float,
    nargs="+",
    default=[-0.15, 0.0, 0.15],
)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[grid] {msg}", flush=True)


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    _log(f"env ready; rendering {len(args_cli.xs) * len(args_cli.ys)} frames "
         f"to {args_cli.out_dir}")

    frames_manifest = []
    idx = 0
    zero_action = torch.zeros((1, 7), device="cuda:0")
    for cx in args_cli.xs:
        for cy in args_cli.ys:
            obs, _ = env.reset()
            sim_dev = env.unwrapped.sim.device
            origin = env.unwrapped.scene.env_origins[0]
            pose = torch.tensor([[
                cx + origin[0].item(),
                cy + origin[1].item(),
                0.025 + origin[2].item(),
                1.0, 0.0, 0.0, 0.0,
            ]], device=sim_dev)
            cube_rb = env.unwrapped.scene["cube"]
            cube_rb.write_root_pose_to_sim(pose)
            cube_rb.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_dev))
            env.unwrapped.scene.write_data_to_sim()
            for _ in range(args_cli.settle_steps):
                obs, *_ = env.step(zero_action)

            wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype(np.uint8)
            third = obs["policy"]["third_person_cam"][0].cpu().numpy().astype(np.uint8)
            tag = f"frame_{idx:02d}"
            Image.fromarray(wrist).save(args_cli.out_dir / f"{tag}_wrist.png")
            Image.fromarray(third).save(args_cli.out_dir / f"{tag}_third.png")
            frames_manifest.append({
                "npz": f"{tag}_data.npz",
                "cube_xy": [round(float(cx), 4), round(float(cy), 4)],
                "third_png": f"{tag}_third.png",
                "wrist_png": f"{tag}_wrist.png",
            })
            _log(f"  {tag}: cube=({cx:+.3f}, {cy:+.3f})")
            idx += 1

    manifest_path = args_cli.out_dir / "cube_xys.json"
    manifest_path.write_text(json.dumps({"frames": frames_manifest}, indent=2))
    _log(f"wrote manifest {manifest_path} ({len(frames_manifest)} frames)")

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
