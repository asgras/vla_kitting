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
parser.add_argument("--max_steps_per_demo", type=int, default=800)
parser.add_argument("--gui", action="store_true", default=False,
                    help="Run with Isaac Sim viewport visible (DCV display). Default is headless.")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=not args_cli.gui, enable_cameras=True)
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


def _quat_mul(a, b):
    """Hamilton product of two quats in (w, x, y, z) form."""
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _quat_conj(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _quat_err_axis_angle(q_cur, q_des):
    """Axis-angle (3,) representing the rotation from q_cur to q_des."""
    import torch
    q_err = _quat_mul(q_des, _quat_conj(q_cur))
    # Shortest-path: flip if w < 0
    if q_err[..., 0].item() < 0:
        q_err = -q_err
    w = q_err[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=1e-8))
    axis = q_err[..., 1:] / sin_half
    return axis * angle


def script_trajectory_waypoints(cube_pos_w: "torch.Tensor") -> list[dict]:
    """Generate the scripted trajectory waypoints given privileged cube pose.

    Returns a list of {ee_pos, gripper, steps} dicts. All heights are for the
    TOOL0 frame. Fingertip is +9.8 cm along tool0's local +Z axis (from URDF
    chain tool0 -> base -> knuckle -> finger -> finger_tip at knuckle=0), so
    when tool0 is oriented top-down the fingertip sits 9.8 cm below tool0 in
    world Z. Table top is at z=0, cube spans z=[0, 0.05] (center 0.025).

    Phase design: keep REORIENT and DESCEND strictly separate so the arm is
    already fully top-down and XY-aligned before any vertical motion starts.
    Otherwise the tilted fingertips sweep through the cube during phase 0.
    """
    import torch
    cx, cy, _ = cube_pos_w.tolist()

    # Place target well away from the pick column so the success region is
    # unambiguous.
    tx, ty = 0.65, 0.20

    # The inner-knuckle collision mesh (STL bbox 5cm long in local +x and +z)
    # is what actually grips the cube — the outer finger_tip_link converges to
    # x≈±0.027 at full close which is just outside the cube's ±0.025 half-
    # width. At full close (θ=0.8) the mesh spans base-frame z=[0.053, 0.126],
    # so when tool0 is top-down the mesh spans world z = [tool0_z - 0.126,
    # tool0_z - 0.053]. For the mesh to wrap cube sides (z=[0, 0.05]) we want
    # tool0_z ≈ 0.13. But if we try to close the gripper while tool0 sits at
    # 0.13, the rotating mesh drops below z=0 and the table blocks rotation —
    # knuckle stalls at ≈0 rad (observed empirically).
    #
    # Fix: descend to grasp_h with gripper open, then issue close and lift
    # commands simultaneously. As tool0 rises, the mesh clears the table so
    # rotation can progress, and the fingers catch the cube on the way up.
    #
    #   approach_h = 0.35  -> fingertip at z≈0.25 (clear of cube)
    #   grasp_h    = 0.130 -> outer fingertip at z=0.032 (beside cube lower
    #                          half), inner-knuckle origin at z=0.069
    #   lift_h     = 0.30  -> fingertip at z=0.20 (15 cm above table)
    approach_h = 0.35
    grasp_h = 0.130
    lift_h = 0.30

    # BinaryJointPositionAction: actions < 0 => CLOSE, >= 0 => OPEN. So
    # gripper = +1.0 is OPEN, -1.0 is CLOSE.
    return [
        # A: rise/rotate to top-down above cube — the long phase that also
        #    soaks up the ~45° home tilt. No descent below approach_h here.
        {"ee_pos": torch.tensor([cx, cy, approach_h]), "gripper": +1.0, "steps": 140},
        # B: settle at hover (kills residual XY drift before descent).
        {"ee_pos": torch.tensor([cx, cy, approach_h]), "gripper": +1.0, "steps": 30},
        # C: pure-vertical descent with gripper open — fingertips come to rest
        #    beside the cube's lower half.
        {"ee_pos": torch.tensor([cx, cy, grasp_h]),    "gripper": +1.0, "steps": 80},
        # D: slow rise to a "close height" where the inner-knuckle mesh is
        #    just clear of the table (so close can rotate) but still in the
        #    upper half of the cube (so the mesh actually pinches it). Start
        #    commanding close here — the close will catch up as tool0 rises.
        {"ee_pos": torch.tensor([cx, cy, 0.16]),       "gripper": -1.0, "steps": 120},
        # E: hold at close height while the close command finalizes its grip.
        {"ee_pos": torch.tensor([cx, cy, 0.16]),       "gripper": -1.0, "steps": 80},
        # F: lift with cube in hand.
        {"ee_pos": torch.tensor([cx, cy, lift_h]),     "gripper": -1.0, "steps": 70},
        # G: transport to the place target (stay closed).
        {"ee_pos": torch.tensor([tx, ty, lift_h]),     "gripper": -1.0, "steps": 120},
        # H: hold at target so the success detector can fire.
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

        # Target orientation: tool0 +Z pointing in world -Z (gripper fingers down).
        # Built as the Hamilton product qz(yaw=0) * qx(π) → (0, 1, 0, 0) in (w,x,y,z).
        # This is the canonical top-down pose used by kitting_ws for HC10DT picks.
        q_target = _quat_from_downward_xy_yaw(0.0).to("cuda:0")

        # Starting EE pose for diagnostics — report orientation error up-front so
        # you can see how far the arm has to rotate to reach top-down.
        start_ee = obs["policy"]["ee_pose"][0]
        start_ang_err = float(torch.linalg.vector_norm(
            _quat_err_axis_angle(start_ee[3:7], q_target)
        ))
        _log(f"  start EE pos: ({start_ee[0]:.3f}, {start_ee[1]:.3f}, {start_ee[2]:.3f}) "
             f"top-down ang_err={start_ang_err:.2f}rad")

        total_steps = 0
        success_step_count = 0
        success_reached = False

        for phase_idx, wp in enumerate(waypoints):
            target_pos = wp["ee_pos"].to("cuda:0")
            gripper_cmd = wp["gripper"]
            phase_steps = wp["steps"]

            phase_start_ee = obs["policy"]["ee_pose"][0]
            # Probe the actual left-knuckle joint angle AND the world-Z of the
            # left inner-knuckle link (the body that carries the inner pad
            # mesh in this URDF — the real gripping surface). If the inner
            # pad drops near z=0 during close, the table blocks further
            # closing.
            robot = env.unwrapped.scene["robot"]
            knuckle_idx = robot.joint_names.index("robotiq_85_left_knuckle_joint")
            knuckle_q = float(robot.data.joint_pos[0, knuckle_idx])
            inner_idx = robot.body_names.index("robotiq_85_left_inner_knuckle_link")
            inner_z = float(robot.data.body_pos_w[0, inner_idx, 2])
            fingertip_idx = robot.body_names.index("robotiq_85_left_finger_tip_link")
            fingertip_z = float(robot.data.body_pos_w[0, fingertip_idx, 2])
            _log(f"  phase {phase_idx}: EE ({phase_start_ee[0]:.2f},{phase_start_ee[1]:.2f},{phase_start_ee[2]:.2f}) -> "
                 f"target ({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}) "
                 f"grip={gripper_cmd} knuckle={knuckle_q:.2f}rad "
                 f"fingertip_z={fingertip_z:.3f} inner_z={inner_z:.3f}")

            for step in range(phase_steps):
                if total_steps >= args_cli.max_steps_per_demo:
                    break
                # Current EE pose from policy obs (ee_pose is (N, 7) as x,y,z,qw,qx,qy,qz)
                current_ee = obs["policy"]["ee_pose"][0]  # (7,)
                cur_pos = current_ee[:3]
                cur_quat = current_ee[3:7]

                # Position: P-controller, saturates at 10 cm position error.
                pos_err = target_pos - cur_pos
                pos_delta = torch.clamp(pos_err * 10.0, -1.0, 1.0)

                # Orientation: drive the EE toward the top-down target every step.
                # apply_delta_pose interprets action[3:6] as a world-frame axis-angle
                # delta and premultiplies: q_new = q_delta * q_cur. So the correct
                # command is axis_angle(q_target * q_cur^-1). Previously this was
                # hard-zeroed, which is why the arm approached at its ~45° home tilt.
                rot_err = _quat_err_axis_angle(cur_quat, q_target)
                rot_delta = torch.clamp(rot_err * 3.0, -1.0, 1.0)

                action = torch.zeros((1, 7), device="cuda:0")
                action[0, :3] = pos_delta
                action[0, 3:6] = rot_delta
                action[0, 6] = gripper_cmd

                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1

                # Diagnostic: at phase midpoint, report current vs target
                if step == phase_steps // 2:
                    cur = obs["policy"]["ee_pose"][0, :3]
                    cur_q = obs["policy"]["ee_pose"][0, 3:7]
                    ang_err = float(torch.linalg.vector_norm(_quat_err_axis_angle(cur_q, q_target)))
                    cube_now = obs["policy"]["cube_pos"][0]
                    grip_closed = obs["policy"]["gripper_closed"][0]
                    _log(f"    phase {phase_idx} midway: EE ({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f}) "
                         f"ang_err={ang_err:.2f}rad "
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
