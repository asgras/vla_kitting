"""Probe where every gripper body ends up during the scripted pick's grasp
configuration — tool0 top-down at z=0.13 to 0.17, gripper open and closed.

Prints the MIN world-Z of every gripper body (from body_pos_w, which gives the
link origin). PhysX contacts use the collision mesh attached to each body; we
also need the mesh extents to determine real collision footprint, but the link
origin already tells us the base height of each body chain.

Usage: ./isaaclab.sh -p scripts/validate/gripper_descent_probe.py
"""
from __future__ import annotations
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def main() -> int:
    import torch
    import gymnasium as gym
    import envs  # noqa
    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    names = list(robot.body_names)
    device = env.unwrapped.device

    # Drive arm down to tool0_z=0.13, holding top-down. We'll command repeatedly
    # via IK-rel until convergence, then snapshot every body's z. Then do the
    # same at tool0_z=0.16 and tool0_z=0.20.
    # Target orientation (top-down, yaw=0): (w, x, y, z) = (0, 1, 0, 0).
    q_target = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    from envs.quat_utils import quat_err_axis_angle

    obs_ref = [obs]

    def drive_to(xy, z, gripper, steps=400):
        target_pos = torch.tensor([xy[0], xy[1], z], device=device)
        for _ in range(steps):
            ee = obs_ref[0]["policy"]["ee_pose"][0]
            pos_err = target_pos - ee[:3]
            pos_delta = torch.clamp(pos_err * 10.0, -1.0, 1.0)
            rot_err = quat_err_axis_angle(ee[3:7], q_target)
            rot_delta = torch.clamp(rot_err * 3.0, -1.0, 1.0)
            a = torch.zeros((1, 7), device=device)
            a[0, :3] = pos_delta; a[0, 3:6] = rot_delta; a[0, 6] = gripper
            new_obs, *_ = env.step(a)
            obs_ref[0] = new_obs

    def snapshot(label):
        pos = robot.data.body_pos_w[0]  # (N, 3)
        print(f"\n=== {label} ===", flush=True)
        # Sort by world z
        rows = []
        for i, n in enumerate(names):
            rows.append((pos[i, 2].item(), n, pos[i, 0].item(), pos[i, 1].item()))
        rows.sort()
        for z, n, x, y in rows[:10]:
            print(f"  {n:<42s}  x={x:+.4f}  y={y:+.4f}  z={z:+.4f}", flush=True)

    # tool0 XY from env (robot base at origin, cube near 0.55) — pick (0.55, 0.0)
    print("[probe] driving to tool0 @ (0.55, 0.0, 0.20) open", flush=True)
    drive_to((0.55, 0.0), 0.20, gripper=+1.0, steps=300)
    snapshot("TOOL0_Z=0.20 GRIPPER OPEN (lowest 10 bodies)")

    print("[probe] driving to tool0 @ (0.55, 0.0, 0.16) open", flush=True)
    drive_to((0.55, 0.0), 0.16, gripper=+1.0, steps=200)
    snapshot("TOOL0_Z=0.16 GRIPPER OPEN")

    print("[probe] closing gripper at tool0_z=0.16", flush=True)
    drive_to((0.55, 0.0), 0.16, gripper=-1.0, steps=300)
    snapshot("TOOL0_Z=0.16 GRIPPER CLOSED")

    print("[probe] driving to tool0 @ (0.55, 0.0, 0.13) open", flush=True)
    drive_to((0.55, 0.0), 0.13, gripper=+1.0, steps=200)
    snapshot("TOOL0_Z=0.13 GRIPPER OPEN")

    print("[probe] closing at tool0_z=0.13", flush=True)
    drive_to((0.55, 0.0), 0.13, gripper=-1.0, steps=300)
    snapshot("TOOL0_Z=0.13 GRIPPER CLOSED (attempted)")

    env.close(); sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
