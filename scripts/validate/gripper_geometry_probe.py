"""Measure fingertip Z-offset from tool0 at various knuckle positions.

Runs the env headlessly, holds the arm at a known pose, and for a sweep of
gripper knuckle targets reads world-frame positions of the relevant bodies
directly from the articulation. From these we derive the grasp_h that puts
the fingertip contact patches around the cube (z = 0.0..0.05, center 0.025).

Usage:
    ./isaaclab.sh -p scripts/validate/gripper_geometry_probe.py
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _p(msg=""):
    print(msg, flush=True)


def main() -> int:
    import torch
    import gymnasium as gym

    _p("[probe] importing envs")
    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    _p(f"[probe] parsing env cfg for {TASK}")
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    _p("[probe] creating env")
    env = gym.make(TASK, cfg=env_cfg)
    _p("[probe] env created, resetting")
    obs, _ = env.reset()
    _p("[probe] env reset done")

    robot = env.unwrapped.scene["robot"]
    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)

    _p("=" * 70)
    _p(f"[probe] articulation body count: {len(body_names)}")
    _p(f"[probe] all bodies: {body_names}")
    _p("=" * 70)

    # Pick the bodies we care about.
    want = [
        "tool0",
        "robotiq_85_left_finger_tip_link",
        "robotiq_85_right_finger_tip_link",
        "robotiq_85_left_inner_finger_pad",
        "robotiq_85_right_inner_finger_pad",
        "robotiq_85_left_finger_link",
        "robotiq_85_right_finger_link",
    ]
    have = [b for b in want if b in body_names]
    missing = [b for b in want if b not in body_names]
    _p(f"[probe] bodies present: {have}")
    _p(f"[probe] bodies missing: {missing}")

    knuckle_joint = "robotiq_85_left_knuckle_joint"
    if knuckle_joint not in joint_names:
        _p(f"[probe] ERROR: {knuckle_joint} not in joint list")
        env.close()
        sim_app.close()
        return 1
    knuckle_idx = joint_names.index(knuckle_joint)

    # Drive knuckle to several positions via direct joint_pos write. Step physics
    # enough times for the pose to settle at each setpoint.
    def snapshot(label):
        body_pos = robot.data.body_pos_w[0]  # (num_bodies, 3) world frame
        tool0_idx = body_names.index("tool0")
        tool0_pos = body_pos[tool0_idx]
        _p(f"\n[probe] === {label} ===")
        _p(f"[probe] tool0 world xyz: ({tool0_pos[0]:.4f}, {tool0_pos[1]:.4f}, {tool0_pos[2]:.4f})")
        knuckle_q = robot.data.joint_pos[0, knuckle_idx].item()
        _p(f"[probe] left_knuckle joint_pos: {knuckle_q:.4f} rad")
        _p(f"[probe] all gripper joint positions:")
        for j, jname in enumerate(joint_names):
            if "robotiq" in jname:
                q = robot.data.joint_pos[0, j].item()
                tgt = robot.data.joint_pos_target[0, j].item() if hasattr(robot.data, "joint_pos_target") else float("nan")
                _p(f"[probe]   {jname}: pos={q:+.4f}  target={tgt:+.4f}")
        for name in have:
            if name == "tool0":
                continue
            idx = body_names.index(name)
            p = body_pos[idx]
            dz = (p[2] - tool0_pos[2]).item()
            dx = (p[0] - tool0_pos[0]).item()
            dy = (p[1] - tool0_pos[1]).item()
            _p(f"[probe]   {name}: world_z={p[2]:.4f}  rel_to_tool0 (dx,dy,dz)=({dx:+.4f},{dy:+.4f},{dz:+.4f})")
        return tool0_pos[2].item()

    # Hold the arm at init_state and drive the knuckle through a sweep.
    # Use an IK-Rel action of zero so the arm just floats at start pose; use
    # BinaryJoint gripper command to switch open/close.
    device = env.unwrapped.device
    act_open = torch.zeros((1, 7), device=device)
    act_open[0, 6] = 1.0  # +1.0 = OPEN per BinaryJointPositionAction
    act_close = torch.zeros((1, 7), device=device)
    act_close[0, 6] = -1.0  # -1.0 = CLOSE

    # Settle fully open
    for _ in range(120):
        obs, _, _, _, _ = env.step(act_open)
    tool0_z_open = snapshot("GRIPPER OPEN (knuckle target 0.0)")

    # Settle fully closed
    for _ in range(180):
        obs, _, _, _, _ = env.step(act_close)
    tool0_z_closed = snapshot("GRIPPER CLOSED (knuckle target 0.78)")

    # Back to open for final
    for _ in range(120):
        obs, _, _, _, _ = env.step(act_open)
    snapshot("GRIPPER OPEN AGAIN (sanity)")

    # Compute derived grasp_h recommendations assuming tool0 is at grasp_h and
    # table top is at z=0, cube is 0.05 tall (spans z=0..0.05, center 0.025).
    _p("\n" + "=" * 70)
    _p("[probe] RECOMMENDATIONS")
    _p("=" * 70)
    body_pos = robot.data.body_pos_w[0]
    tool0_idx = body_names.index("tool0")
    tool0_z = body_pos[tool0_idx, 2].item()
    for name in ("robotiq_85_left_finger_tip_link", "robotiq_85_left_inner_finger_pad"):
        if name not in body_names:
            continue
        idx = body_names.index(name)
        fingertip_z = body_pos[idx, 2].item()
        z_offset = tool0_z - fingertip_z  # how far above the fingertip tool0 is
        _p(f"[probe] Using {name}:")
        _p(f"[probe]   z_offset (tool0 - fingertip) = {z_offset:.4f} m (at OPEN)")
        _p(f"[probe]   For fingertip at cube center (z=0.025): grasp_h = {0.025 + z_offset:.4f}")
        _p(f"[probe]   For fingertip at cube top   (z=0.050): grasp_h = {0.050 + z_offset:.4f}")
        _p(f"[probe]   For fingertip 5mm above table (z=0.005): grasp_h = {0.005 + z_offset:.4f}")

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
