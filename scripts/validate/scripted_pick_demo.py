"""Phase 7 sanity test: script a successful pick-and-place trajectory using
privileged state (ground truth cube pose). Writes a genuine HDF5 dataset in the
same format that human teleop via record_demos.py would produce.

Proves end-to-end: env stepping + IK controller + physical grasp + recorder
manager → HDF5 all work together before the user burns time teleoping a broken
system.

Usage:
    ./isaaclab.sh -p scripts/validate/scripted_pick_demo.py \\
        --num_demos 5 \\
        --dataset_file datasets/teleop/cube_scripted.hdf5
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--dataset_file", type=str,
                    default=str(REPO / "datasets/teleop/cube_scripted.hdf5"))
parser.add_argument("--max_steps_per_demo", type=int, default=500)
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg):
    print(f"[scripted_pick] {msg}", flush=True)


def _quat_from_downward_xy_yaw(yaw: float) -> "torch.Tensor":
    """Build a quaternion that points the EE -Z axis down (toward the table),
    with rotation about world Z by `yaw` for approach direction."""
    import math
    import torch
    # Base orientation: rotate 180° around X so tool0's +Z becomes world -Z.
    # Then rotate by yaw around world Z.
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cr, sr = math.cos(math.pi / 2), math.sin(math.pi / 2)  # 180° / 2 = 90°
    # quat = qz(yaw) * qx(pi)
    # qx(pi) = (0, 1, 0, 0) in (w,x,y,z)
    # qz(yaw) = (cy, 0, 0, sy)
    # product (w1,x1,y1,z1) * (w2,x2,y2,z2):
    #   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    #   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    #   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    #   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w1, x1, y1, z1 = cy, 0, 0, sy
    w2, x2, y2, z2 = 0, 1, 0, 0
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.tensor([w, x, y, z])


def _pose_delta(current: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Compute a pose delta in the IK-rel action space: 3D position delta + 3D axis-angle rot delta.

    Inputs are (7,) tensors: (x, y, z, qw, qx, qy, qz).
    Output is (6,) tensor: position_delta + axis_angle_rot_delta.

    For this scripted demo we hold orientation roughly constant by just using
    position deltas, leaving rotation delta at zero.
    """
    import torch
    pos_delta = target[:3] - current[:3]
    rot_delta = torch.zeros(3, device=current.device)
    return torch.cat([pos_delta, rot_delta])


def script_trajectory_waypoints(cube_pos_w: "torch.Tensor") -> list[dict]:
    """Generate the 6-phase scripted trajectory waypoints given privileged cube pose.

    Returns a list of {ee_pos, gripper, steps} dicts.
    """
    import torch
    cx, cy, cz = cube_pos_w.tolist()

    # Heights below are for the TOOL0 frame (wrist, top of gripper).
    # The Robotiq 2F-85 fingertips extend ~17cm below tool0. So tool0 at z=0.20
    # puts fingertips at z=0.03 — around the cube center.
    tx, ty = 0.65, 0.20
    approach_h = 0.25  # hover well above
    grasp_h = 0.12     # fingertips straddle the cube (tool0 -> fingertip ~11cm from URDF)
    lift_h = 0.28      # lifts cube 15+ cm above table top

    # BinaryJointPositionAction: actions < 0 => CLOSE, actions >= 0 => OPEN.
    # So gripper = +1.0 means OPEN, -1.0 means CLOSE.
    return [
        # A: go above cube (open)
        {"ee_pos": torch.tensor([cx, cy, approach_h]), "gripper": +1.0, "steps": 80},
        # B: descend to grasp height (open)
        {"ee_pos": torch.tensor([cx, cy, grasp_h]),    "gripper": +1.0, "steps": 60},
        # C: close gripper in place (longer hold so it can finish closing)
        {"ee_pos": torch.tensor([cx, cy, grasp_h]),    "gripper": -1.0, "steps": 80},
        # D: lift (closed)
        {"ee_pos": torch.tensor([cx, cy, lift_h]),     "gripper": -1.0, "steps": 50},
        # E: transport to target (closed)
        {"ee_pos": torch.tensor([tx, ty, lift_h]),     "gripper": -1.0, "steps": 120},
        # F: hold at target (closed) — let success detector fire
        {"ee_pos": torch.tensor([tx, ty, lift_h]),     "gripper": -1.0, "steps": 80},
    ]


def main() -> int:
    import torch
    import gymnasium as gym

    _log("importing envs (registers task)")
    import envs  # noqa: F401

    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"

    dataset_path = pathlib.Path(args_cli.dataset_file)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        dataset_path.unlink()

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    env_cfg.recorders.dataset_filename = dataset_path.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    _log(f"creating env, output → {dataset_path}")
    env = gym.make(TASK, cfg=env_cfg)
    _log(f"env ready. action_dim={env.action_space.shape[-1]}")

    # Get success term reference (same as record_demos.py)
    success_term = None
    if hasattr(env.unwrapped, "termination_manager") and hasattr(env.unwrapped.termination_manager, "get_term_cfg"):
        try:
            success_term = env.unwrapped.termination_manager.get_term_cfg("success")
        except Exception:
            pass

    demo_target = args_cli.num_demos
    successful = 0
    attempted = 0

    while successful < demo_target and attempted < demo_target * 3:
        attempted += 1
        _log(f"attempt {attempted}: successes so far {successful}/{demo_target}")
        obs, _ = env.reset()

        # Read privileged cube pose
        cube_pos = obs["policy"]["cube_pos"][0].clone()  # (3,)
        _log(f"  cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

        waypoints = script_trajectory_waypoints(cube_pos)

        # Starting EE pose for diagnostics
        start_ee = obs["policy"]["ee_pose"][0]
        _log(f"  start EE pos: ({start_ee[0]:.3f}, {start_ee[1]:.3f}, {start_ee[2]:.3f})")

        total_steps = 0
        success_step_count = 0
        success_reached = False

        for phase_idx, wp in enumerate(waypoints):
            target_pos = wp["ee_pos"].to("cuda:0")
            gripper_cmd = wp["gripper"]
            phase_steps = wp["steps"]

            phase_start_ee = obs["policy"]["ee_pose"][0]
            _log(f"  phase {phase_idx}: EE ({phase_start_ee[0]:.2f},{phase_start_ee[1]:.2f},{phase_start_ee[2]:.2f}) -> "
                 f"target ({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}) grip={gripper_cmd}")

            for step in range(phase_steps):
                if total_steps >= args_cli.max_steps_per_demo:
                    break
                # Current EE position from policy obs (ee_pose is (N, 7))
                current_ee = obs["policy"]["ee_pose"][0]  # (7,)
                cur_pos = current_ee[:3]
                # Simple proportional controller in position, zero rotation delta
                pos_err = target_pos - cur_pos
                # Feed raw error directly; env scale=0.1 caps per-step motion to ~10 cm
                # (the processed action = raw * 0.1). Clamp raw to [-1, 1] to avoid IK blowup.
                pos_delta = torch.clamp(pos_err * 10.0, -1.0, 1.0)
                rot_delta = torch.zeros(3, device="cuda:0")
                action = torch.zeros((1, 7), device="cuda:0")
                action[0, :3] = pos_delta
                action[0, 3:6] = rot_delta
                action[0, 6] = gripper_cmd

                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1

                # Diagnostic: at phase midpoint, report current vs target
                if step == phase_steps // 2:
                    cur = obs["policy"]["ee_pose"][0, :3]
                    cube_now = obs["policy"]["cube_pos"][0]
                    grip_closed = obs["policy"]["gripper_closed"][0]
                    _log(f"    phase {phase_idx} midway: EE ({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f}) "
                         f"cube ({cube_now[0]:.3f},{cube_now[1]:.3f},{cube_now[2]:.3f}) "
                         f"grip_closed={grip_closed[0]:.1f}")

                # Track success (same logic as record_demos.py's process_success_condition)
                if success_term is not None:
                    try:
                        is_success = bool(success_term.func(env.unwrapped, **success_term.params)[0])
                    except Exception:
                        is_success = bool(terminated[0])
                else:
                    is_success = bool(terminated[0])

                if is_success:
                    success_step_count += 1
                    if success_step_count >= 10:  # 10 steps of sustained success
                        success_reached = True
                        break
                else:
                    success_step_count = 0

                if truncated[0].item():
                    break

            if success_reached or total_steps >= args_cli.max_steps_per_demo:
                break

        _log(f"  phases done in {total_steps} steps, success={success_reached}")

        # Record via recorder manager
        if success_reached:
            env.unwrapped.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.unwrapped.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
            )
            env.unwrapped.recorder_manager.export_episodes([0])
            successful += 1

        env.unwrapped.recorder_manager.reset()

    _log(f"total: {successful}/{demo_target} successful demos in {attempted} attempts")

    env.close()
    sim_app.close()

    ok = successful >= demo_target
    _log(f"dataset: {dataset_path} ({dataset_path.stat().st_size if dataset_path.exists() else 0} bytes)")
    _log(f"result: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
