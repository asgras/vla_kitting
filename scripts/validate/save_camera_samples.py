"""Render and save one wrist + one third-person camera frame to PNG for each
of N random cube positions. Used to eyeball whether the VLA will actually see
the cube and the green target marker.

Usage:
    ./isaaclab.sh -p scripts/validate/save_camera_samples.py --samples 5 \\
        --out_dir reports/camera_samples/
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=5)
parser.add_argument("--out_dir", type=str,
                    default=str(REPO / "reports" / "camera_samples"))
parser.add_argument("--settle_steps", type=int, default=20,
                    help="physics steps after reset before capturing (lets lighting/color apply).")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg):
    print(f"[cam_samples] {msg}", flush=True)


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import envs  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"

    out_dir = pathlib.Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    _log(f"env ready; saving {args_cli.samples} samples to {out_dir}")

    for i in range(args_cli.samples):
        obs, _ = env.reset()
        # Run a few no-op steps so the randomized color/light actually takes effect
        # (Replicator / USD material writes don't hit the render until the next frame).
        zero_action = torch.zeros((1, 7), device="cuda:0")
        for _ in range(args_cli.settle_steps):
            obs, *_ = env.step(zero_action)

        cube_pos = obs["policy"]["cube_pos"][0].tolist()
        wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype(np.uint8)
        third = obs["policy"]["third_person_cam"][0].cpu().numpy().astype(np.uint8)

        wrist_path = out_dir / f"sample_{i:02d}_wrist.png"
        third_path = out_dir / f"sample_{i:02d}_third.png"
        Image.fromarray(wrist).save(wrist_path)
        Image.fromarray(third).save(third_path)
        _log(
            f"  sample {i}: cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}) "
            f"wrist={wrist.shape} third={third.shape} "
            f"-> {wrist_path.name}, {third_path.name}"
        )

    env.close()
    sim_app.close()
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
