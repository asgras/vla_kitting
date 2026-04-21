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
parser.add_argument("--max_steps_per_demo", type=int, default=1600)
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


# ----- Named pick/place locations (world frame) -----
#
# Heights are for the TOOL0 frame. On the RIA 2F-85 the grip surface
# (left_inner_finger_pad) sits ~0.14 m below tool0 when the arm is fully
# top-down. Cube spans z=[0, 0.05] (center 0.025). So to pinch the cube on
# its midline we want pad_z ≈ 0.025 → tool0_z ≈ 0.165.
PICK_APPROACH_Z = 0.40    # tool0 hover height (pad ~0.26 above table)
PICK_GRASP_Z    = 0.17    # tool0 at grasp (pad ~0.03, cube-center height)
# Transport height. 0.50 puts the HC10DT near full arm extension and the
# differential-IK controller hits a singularity mid-swing (observed: EE
# snapped 17 cm backward at phase 6 ~70%, flinging the cube). 0.40 keeps
# the arm more bent — pad still sits 26 cm above the table (well above the
# 10 cm success threshold).
LIFT_Z          = 0.40
PLACE_XY        = (0.65, 0.20)  # success target
PLACE_APPROACH_Z = 0.40

# Asymmetric-mimic-chain compensation. The PhysxMimicJointAPI on the RIA
# 2F-85 closes one finger ~18 mm ahead of the other when driven from rest,
# so closing on a centered cube consistently kicks it by -0.018 m in the
# tool0-Y direction. Before the chase: only the trailing pad made contact,
# giving a marginal grip that failed mid-transport (see phase 6 logs pre-
# compensation). Pre-offsetting the EE target by +0.018 m in Y means the
# faster pad has less travel, both pads reach the cube at the same time,
# and the cube ends up centered. Only applied during descent + close, not
# transport (where the EE goes to the fixed place target).
GRIP_BIAS_Y = 0.018


def script_trajectory_waypoints(cube_pos_w: "torch.Tensor") -> list[dict]:
    """Generate the scripted pick-and-place trajectory.

    Waypoints are consumed by an interpolating controller that ramps the EE
    linearly between phase start and `ee_pos` over `steps` frames — giving
    smooth, near-constant-velocity motion instead of saturated P-controller
    snaps. `track` controls whether pick_xy is updated from the live cube
    pose during that phase (needed for approach/descent alignment, MUST be
    False once the gripper has contacted the cube).

    Phase budget (total 1190 @ 60 Hz ≈ 20 s per demo):
      A approach+reorient    140  TRACK   above cube, rotate to top-down
      B hover settle          30  TRACK   kill residual drift
      C slow descent         200  TRACK   straight down to grasp
      D settle at grasp       40  --      let cube rest under open pads
      E close gripper        180  --      pinch, EE frozen
      F lift straight up     180  --      pure +Z to LIFT_Z
      G transport to place   300  --      smooth XY swing above target
      H hold at place        120  --      success detector needs 10 sustained
    """
    import torch
    cx, cy, _ = cube_pos_w.tolist()
    tx, ty = PLACE_XY

    # Bias-compensated grasp Y (see GRIP_BIAS_Y comment).
    gy = cy + GRIP_BIAS_Y

    # BinaryJointPositionAction: actions < 0 => CLOSE, >= 0 => OPEN. So
    # gripper = +1.0 is OPEN, -1.0 is CLOSE.
    return [
        # A: rise/rotate to top-down above cube.
        {"ee_pos": torch.tensor([cx, gy, PICK_APPROACH_Z]), "gripper": +1.0, "steps": 140, "track": True},
        # B: hover-settle above cube (bias-compensated).
        {"ee_pos": torch.tensor([cx, gy, PICK_APPROACH_Z]), "gripper": +1.0, "steps": 30,  "track": True},
        # C: slow pure-vertical descent to bias-compensated grasp.
        {"ee_pos": torch.tensor([cx, gy, PICK_GRASP_Z]),    "gripper": +1.0, "steps": 200, "track": True},
        # D: settle with pads straddling the cube. Tracking OFF — freezes
        #    the EE so the cube can rest, and we commit to grasp xy now.
        {"ee_pos": torch.tensor([cx, gy, PICK_GRASP_Z]),    "gripper": +1.0, "steps": 40,  "track": False},
        # E: close gripper in place. Tracking OFF — if the close nudges the
        #    cube, we do NOT chase it (chasing was the old feedback loop).
        {"ee_pos": torch.tensor([cx, gy, PICK_GRASP_Z]),    "gripper": -1.0, "steps": 180, "track": False},
        # F: lift straight up to LIFT_Z. Tracking OFF.
        {"ee_pos": torch.tensor([cx, gy, LIFT_Z]),          "gripper": -1.0, "steps": 180, "track": False},
        # G1-G3: transport split into three short segments. A single Cartesian
        #    line from (pick, 0.40) to (place, 0.40) pushes the HC10DT through
        #    an IK singularity mid-swing — observed EE jumping 45 cm backward
        #    in one decile even at LIFT_Z=0.40. Broken into 3 × 200-step legs
        #    the diff-IK controller stays in the same joint branch the whole
        #    way. Intermediate points bias the Y motion first, then X, then
        #    final approach — keeping the base-joint rotation monotonic.
        {"ee_pos": torch.tensor([cx + (tx - cx) * 0.33, cy + (ty - cy) * 0.33 + GRIP_BIAS_Y, LIFT_Z]),
         "gripper": -1.0, "steps": 200, "track": False},
        {"ee_pos": torch.tensor([cx + (tx - cx) * 0.66, cy + (ty - cy) * 0.66 + GRIP_BIAS_Y, LIFT_Z]),
         "gripper": -1.0, "steps": 200, "track": False},
        {"ee_pos": torch.tensor([tx, ty, LIFT_Z]),          "gripper": -1.0, "steps": 200, "track": False},
        # H: hold at place target so success detector fires.
        {"ee_pos": torch.tensor([tx, ty, LIFT_Z]),          "gripper": -1.0, "steps": 120, "track": False},
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

        # Read privileged cube pose (initial position, used for transport target).
        cube_pos = obs["policy"]["cube_pos"][0].clone()  # (3,)
        _log(f"  cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

        waypoints = script_trajectory_waypoints(cube_pos)
        # Each waypoint owns its own `track` flag (see script_trajectory_waypoints
        # docstring). Approach/hover/descent track the live cube XY so the gripper
        # aligns on the cube's actual position; settle/close/lift/transport/hold
        # DON'T track — once the pads are around the cube we commit to the grasp
        # pose, because per-step tracking during close created a feedback loop
        # where a cube-kick would swing the EE, which would kick the cube further.

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
            phase_end_pos = wp["ee_pos"].to("cuda:0").clone()
            track = wp.get("track", False)
            gripper_cmd = wp["gripper"]
            phase_steps = wp["steps"]

            # Lock in the EE's position at the start of this phase; linear
            # interpolation from here to phase_end_pos gives smooth constant-
            # velocity motion regardless of P-gain.
            phase_start_pos = obs["policy"]["ee_pose"][0, :3].clone()

            # If tracking, re-read cube XY once at phase start (per-step update
            # happens in the inner loop). Apply the grip-bias Y compensation
            # so the EE lines up for a symmetric close.
            if track:
                live_cube = obs["policy"]["cube_pos"][0]
                phase_end_pos[0] = live_cube[0]
                phase_end_pos[1] = live_cube[1] + GRIP_BIAS_Y

            phase_start_ee = obs["policy"]["ee_pose"][0]
            # Diagnostics on the RIA 2F-85 — finger_joint angle, and world-Z
            # of the left inner finger pad (the actual contact body).
            robot = env.unwrapped.scene["robot"]
            finger_idx = robot.joint_names.index("finger_joint")
            finger_q = float(robot.data.joint_pos[0, finger_idx])
            def _safe_z(name: str) -> float:
                try:
                    return float(robot.data.body_pos_w[0, robot.body_names.index(name), 2])
                except ValueError:
                    return float("nan")
            pad_z = _safe_z("left_inner_finger_pad")
            fingertip_z = _safe_z("left_inner_finger")
            _log(f"  phase {phase_idx}: EE ({phase_start_ee[0]:.2f},{phase_start_ee[1]:.2f},{phase_start_ee[2]:.2f}) -> "
                 f"end ({phase_end_pos[0]:.2f},{phase_end_pos[1]:.2f},{phase_end_pos[2]:.2f}) "
                 f"grip={gripper_cmd} track={track} steps={phase_steps} "
                 f"finger_q={finger_q:.2f}rad fingertip_z={fingertip_z:.3f} pad_z={pad_z:.3f}")

            for step in range(phase_steps):
                if total_steps >= args_cli.max_steps_per_demo:
                    break
                # Current EE pose from policy obs (ee_pose is (N, 7) as x,y,z,qw,qx,qy,qz)
                current_ee = obs["policy"]["ee_pose"][0]  # (7,)
                cur_pos = current_ee[:3]
                cur_quat = current_ee[3:7]

                # Per-step cube tracking (approach/descent only): live-update
                # the phase endpoint XY so the descending gripper follows any
                # cube nudge. Disabled for settle/close/lift/transport (once
                # pads are committed around the cube).
                if track:
                    live_cube = obs["policy"]["cube_pos"][0]
                    phase_end_pos[0] = live_cube[0]
                    phase_end_pos[1] = live_cube[1] + GRIP_BIAS_Y

                # Smoothstep interpolation of the target across the phase:
                # u = 3t² − 2t³, yielding zero velocity at both endpoints.
                # Linear interpolation caused an impulsive deceleration at
                # the end of transport that shook the cube loose; smoothstep
                # ramps acceleration in and out so both the start and stop
                # are gentle.
                if phase_steps > 1:
                    t = min(1.0, step / (phase_steps - 1))
                else:
                    t = 1.0
                u = t * t * (3.0 - 2.0 * t)
                target_pos = phase_start_pos + (phase_end_pos - phase_start_pos) * u

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

                # env auto-resets the sub-env when `terminated` OR `truncated` fires,
                # so by this point `obs` reflects the post-reset state (cube back
                # on the table, EE at home). Our only non-time-out termination is
                # `cube_lifted_over_target`, so `terminated=True` means the demo
                # just succeeded — record it and exit the phase loop immediately,
                # BEFORE the (already-reset) post-reset cube pose makes the inline
                # success check below fail.
                if bool(terminated[0]) and not bool(truncated[0]):
                    _log(f"    ++ phase {phase_idx} step {step}: success (terminated=True) "
                         f"total_steps={total_steps}")
                    success_reached = True
                    break

                # Diagnostic at phase midpoint: report EE pose, cube pose, finger_q.
                # Reduced from full-decile logging once the transport was debugged.
                if step == phase_steps // 2:
                    cur = obs["policy"]["ee_pose"][0, :3]
                    cube_now = obs["policy"]["cube_pos"][0]
                    finger_q = float(robot.data.joint_pos[0, finger_idx])
                    _log(f"    phase {phase_idx} mid: EE ({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f}) "
                         f"cube ({cube_now[0]:.3f},{cube_now[1]:.3f},{cube_now[2]:.3f}) "
                         f"finger_q={finger_q:.2f}")

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
