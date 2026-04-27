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
parser.add_argument("--max_steps_per_demo", type=int, default=1300)
parser.add_argument("--gui", action="store_true", default=False,
                    help="Run with Isaac Sim viewport visible (DCV display). Default is headless.")
parser.add_argument("--cube_xy", nargs="+", default=None,
                    help="List of forced cube XY positions as 'x,y' strings (world frame). "
                         "When supplied, each is tested once with the cube force-placed via "
                         "write_root_pose_to_sim (yaw=0); overrides --num_demos and disables "
                         "retry budget. Used for corner-coverage sanity checks of the spawn "
                         "region — see bd vla_kitting-v67.")
parser.add_argument("--overwrite", action="store_true", default=False,
                    help="If --dataset_file already exists, delete it before recording. "
                         "Without this flag, an existing file aborts the run to prevent "
                         "accidentally clobbering a prior dataset.")
parser.add_argument("--no_strict", action="store_true", default=False,
                    help="Disable strict mode. By default the script re-raises exceptions "
                         "in success detection and HDF5 color-metadata stamping — silent "
                         "fallbacks here have invalidated training runs in the past. Pass "
                         "this flag to restore the old soft-fail behaviour.")
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


from envs.quat_utils import quat_err_axis_angle


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
# the arm more bent — pad still sits 26 cm above the table.
LIFT_Z          = 0.40
PLACE_XY        = (0.65, 0.20)  # output location (cube is RELEASED here)
# Placement descent height. tool0_z=0.18 puts the pad at z≈0.04 — close to
# resting-cube top (z=0.05) so the release is a gentle ~1-2 cm drop instead
# of a fling. Higher than PICK_GRASP_Z (0.17) because we want clearance once
# the fingers open.
PLACE_Z         = 0.18

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


def script_trajectory_waypoints(
    cube_pos_w: "torch.Tensor",
    cube_yaw: float = 0.0,
) -> list[dict]:
    """Generate the scripted pick-and-place trajectory.

    Waypoints are consumed by an interpolating controller that ramps the EE
    linearly between phase start and `ee_pos` over `steps` frames — giving
    smooth, near-constant-velocity motion instead of saturated P-controller
    snaps. `track` controls whether pick_xy is updated from the live cube
    pose during that phase (needed for approach/descent alignment, MUST be
    False once the gripper has contacted the cube).

    `cube_yaw` (rad about world Z) is the cube's orientation at episode start.
    The grip-bias offset is rotated into world frame using cube_yaw so the
    EE always lines up along the gripper's local Y axis, no matter how the
    cube is rotated. The caller is responsible for building `q_target` from
    the same yaw so the gripper rotates to match the cube before descent.

    Phases (total ~1140 @ 30 Hz ≈ 38 s per demo):
      A  approach+reorient   200  TRACK   above cube, rotate to top-down
      B  hover settle         15  TRACK   kill residual drift
      C  slow descent        200  TRACK   straight down to grasp
      D  settle at grasp      30  --      let cube rest under open pads (lengthened)
      E  close gripper       120  --      pinch, EE frozen (lengthened)
      F  lift straight up     90  --      pure +Z to LIFT_Z
      G1 transport leg 1     100  --      first 1/3 of Cartesian path
      G2 transport leg 2     100  --      second 1/3
      G3 transport leg 3     100  --      last 1/3, arriving over PLACE_XY
      H  hover above place    30  --      settle at LIFT_Z over output
      I  descend to place     75  --      pads lower toward table
      J  release              40  --      open gripper, cube drops
      K  retreat up           40  --      lift EE clear, let cube settle

    Step counts derived from 60 Hz baseline (÷2 for 30 Hz; env decimation=4).
    D and E additionally lengthened from their ÷2 baseline (D 20→30, E 90→120)
    to give the policy a longer grasp-commit window (Hole C — close transition
    being statistically invisible in loss). See reports/recovery_plan_2026-04-24.md.
    """
    import math
    import torch
    cx, cy, _ = cube_pos_w.tolist()
    tx, ty = PLACE_XY

    # Rotate the GRIP_BIAS_Y offset (originally in tool0's local Y) into world
    # frame using the cube's yaw. tool0's local Y axis after a yaw rotation
    # about world Z is (-sin(yaw), cos(yaw), 0).
    bias_x = -math.sin(cube_yaw) * GRIP_BIAS_Y
    bias_y = math.cos(cube_yaw) * GRIP_BIAS_Y
    gx = cx + bias_x
    gy = cy + bias_y

    # BinaryJointPositionAction: actions < 0 => CLOSE, >= 0 => OPEN. So
    # gripper = +1.0 is OPEN, -1.0 is CLOSE.
    return [
        # A: rise/rotate to top-down above cube. Lengthened 70→200 (vla_kitting-2hp)
        # to give the policy more frames where the cube is centrally visible to
        # the third-person camera before grasp; smoothstep auto-rescales velocity.
        {"ee_pos": torch.tensor([gx, gy, PICK_APPROACH_Z]), "gripper": +1.0, "steps": 200, "track": True},
        # B: hover-settle above cube (bias-compensated).
        {"ee_pos": torch.tensor([gx, gy, PICK_APPROACH_Z]), "gripper": +1.0, "steps": 15,  "track": True},
        # C: slow pure-vertical descent to bias-compensated grasp. Lengthened
        # 100→200 (vla_kitting-2hp) for the same reason as Phase A.
        {"ee_pos": torch.tensor([gx, gy, PICK_GRASP_Z]),    "gripper": +1.0, "steps": 200, "track": True},
        # D: settle with pads straddling the cube. Tracking OFF — freezes
        #    the EE so the cube can rest, and we commit to grasp xy now.
        {"ee_pos": torch.tensor([gx, gy, PICK_GRASP_Z]),    "gripper": +1.0, "steps": 30,  "track": False},
        # E: close gripper in place. Tracking OFF — if the close nudges the
        #    cube, we do NOT chase it (chasing was the old feedback loop).
        {"ee_pos": torch.tensor([gx, gy, PICK_GRASP_Z]),    "gripper": -1.0, "steps": 120, "track": False},
        # F: lift straight up to LIFT_Z. Tracking OFF.
        {"ee_pos": torch.tensor([gx, gy, LIFT_Z]),          "gripper": -1.0, "steps": 90,  "track": False},
        # G1-G3: transport split into three short segments. A single Cartesian
        #    line from (pick, 0.40) to (place, 0.40) pushes the HC10DT through
        #    an IK singularity mid-swing — observed EE jumping 45 cm backward
        #    in one decile even at LIFT_Z=0.40. Broken into 3 short legs the
        #    diff-IK controller stays in the same joint branch the whole way.
        # Bias is in the gripper local Y; once we're transporting the cube
        # the bias is "frozen in" to the trajectory so it doesn't snap.
        {"ee_pos": torch.tensor([gx + (tx - cx) * 0.33, gy + (ty - cy) * 0.33, LIFT_Z]),
         "gripper": -1.0, "steps": 100, "track": False},
        {"ee_pos": torch.tensor([gx + (tx - cx) * 0.66, gy + (ty - cy) * 0.66, LIFT_Z]),
         "gripper": -1.0, "steps": 100, "track": False},
        {"ee_pos": torch.tensor([tx, ty, LIFT_Z]),          "gripper": -1.0, "steps": 100, "track": False},
        # H: settle above the output location before descending.
        {"ee_pos": torch.tensor([tx, ty, LIFT_Z]),          "gripper": -1.0, "steps": 30,  "track": False},
        # I: descend toward the placement height (pad ends ~4 cm over table).
        {"ee_pos": torch.tensor([tx, ty, PLACE_Z]),         "gripper": -1.0, "steps": 75,  "track": False},
        # J: RELEASE — open the fingers, cube drops the last 1-2 cm and settles.
        {"ee_pos": torch.tensor([tx, ty, PLACE_Z]),         "gripper": +1.0, "steps": 40,  "track": False},
        # K: retreat back up, giving the cube clearance to settle fully. The
        #    success term (cube_placed_at_target) fires once cube is near-stationary
        #    at z<0.05 within xy_tolerance of the target.
        {"ee_pos": torch.tensor([tx, ty, LIFT_Z]),          "gripper": +1.0, "steps": 40,  "track": False},
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
        if not args_cli.overwrite:
            _log(f"ERROR: {dataset_path} already exists. Pass --overwrite to "
                 "delete it, or point --dataset_file at a fresh path.")
            return 2
        _log(f"--overwrite given; removing existing {dataset_path}")
        dataset_path.unlink()

    strict = not args_cli.no_strict

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    # Phases A (200) + B (15) + C (200) + D (30) + E (120) + F (90) + G1-G3 (300)
    # + H (30) + I (75) + J (40) + K (40) = 1140 steps @ 30 Hz ≈ 38 s. Default
    # env episode_length_s=30 truncates at 900 steps — no demo would finish.
    # Override here at the script level (instead of editing the env cfg, which
    # would invalidate prior baselines per CLAUDE.md). vla_kitting-2hp.
    env_cfg.episode_length_s = 45.0
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

    # If --cube_xy is supplied, force each position once (no retries, every
    # position attempted exactly once). Otherwise fall back to random sampling
    # with the original 3x retry budget.
    forced_positions: list[tuple[float, float]] | None = None
    if args_cli.cube_xy:
        forced_positions = [
            tuple(float(v) for v in s.split(",")) for s in args_cli.cube_xy
        ]
        demo_target = len(forced_positions)
        iteration_plan: list[tuple[float, float] | None] = list(forced_positions)
    else:
        demo_target = args_cli.num_demos
        iteration_plan = [None] * (demo_target * 3)

    successful = 0
    attempted = 0
    per_position_outcomes: list[tuple[tuple[float, float] | None, bool, int]] = []

    for forced_xy in iteration_plan:
        # Random mode: stop once we have enough successes. Forced mode: always
        # exhaust every position so the per-corner pass/fail report is complete.
        if forced_positions is None and successful >= demo_target:
            break
        attempted += 1
        tag = (
            f"forced=({forced_xy[0]:+.3f},{forced_xy[1]:+.3f})"
            if forced_xy is not None else "random"
        )
        _log(f"attempt {attempted} [{tag}]: successes so far {successful}/{demo_target}")
        obs, _ = env.reset()

        # Force cube to override XY (yaw = 0) before the trajectory begins.
        if forced_xy is not None:
            cube_rb = env.unwrapped.scene["cube"]
            sim_dev = env.unwrapped.sim.device
            origin = env.unwrapped.scene.env_origins[0]
            fx, fy = forced_xy
            pose = torch.tensor([[
                fx + origin[0].item(),
                fy + origin[1].item(),
                0.025 + origin[2].item(),
                1.0, 0.0, 0.0, 0.0,
            ]], device=sim_dev)
            cube_rb.write_root_pose_to_sim(pose)
            cube_rb.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_dev))
            env.unwrapped.scene.write_data_to_sim()
            # Step once with a zero action so the policy obs picks up the new
            # cube pose before the controller reads cube_pos below.
            zero_act = torch.zeros(
                (1, env.action_space.shape[-1]), device=sim_dev
            )
            obs, _, _, _, _ = env.step(zero_act)

        # Read privileged cube pose (initial position + yaw, used for grasp
        # alignment). The cube observation only includes XYZ; pull the
        # quaternion straight from the rigid-body asset. Yaw randomization
        # in env config rotates only about world Z, so the cube quat is a
        # pure-Z rotation: (cos(yaw/2), 0, 0, sin(yaw/2)) → yaw = 2*atan2(z, w).
        import math
        cube_pos = obs["policy"]["cube_pos"][0].clone()  # (3,)
        cube_quat = env.unwrapped.scene["cube"].data.root_quat_w[0]  # (w,x,y,z)
        cube_yaw = 2.0 * math.atan2(float(cube_quat[3]), float(cube_quat[0]))
        # The cube is 4-fold symmetric about Z, so any equivalent yaw mod π/2
        # is fine. Wrap into [-π/4, π/4) to keep the gripper inside its
        # comfortable rotation range and out of IK-singularity territory.
        while cube_yaw > math.pi / 4:
            cube_yaw -= math.pi / 2
        while cube_yaw < -math.pi / 4:
            cube_yaw += math.pi / 2
        ep_color = (getattr(env.unwrapped, "cube_color_state", {}) or {}).get(0, ("", -1))
        _log(f"  cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) "
             f"yaw={cube_yaw:.3f}rad color={ep_color[0]!r}")

        waypoints = script_trajectory_waypoints(cube_pos, cube_yaw=cube_yaw)
        # Cache the rotated grip-bias for use inside the per-step tracking
        # loop below (track=True phases re-read live cube XY but want the
        # same yaw-rotated bias).
        bias_x = -math.sin(cube_yaw) * GRIP_BIAS_Y
        bias_y = math.cos(cube_yaw) * GRIP_BIAS_Y
        # Each waypoint owns its own `track` flag (see script_trajectory_waypoints
        # docstring). Approach/hover/descent track the live cube XY so the gripper
        # aligns on the cube's actual position; settle/close/lift/transport/hold
        # DON'T track — once the pads are around the cube we commit to the grasp
        # pose, because per-step tracking during close created a feedback loop
        # where a cube-kick would swing the EE, which would kick the cube further.

        # Target orientation: tool0 +Z pointing in world -Z (gripper fingers
        # down) AND rotated by cube_yaw about world Z so the pads align with
        # one of the cube's faces. Built as qz(cube_yaw) * qx(π).
        q_target = _quat_from_downward_xy_yaw(cube_yaw).to("cuda:0")

        # Starting EE pose for diagnostics — report orientation error up-front so
        # you can see how far the arm has to rotate to reach top-down.
        start_ee = obs["policy"]["ee_pose"][0]
        start_ang_err = float(torch.linalg.vector_norm(
            quat_err_axis_angle(start_ee[3:7], q_target)
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
            # happens in the inner loop). Apply the yaw-rotated grip-bias so
            # the EE lines up for a symmetric close along the gripper's
            # local Y axis (which is rotated by cube_yaw about world Z).
            if track:
                live_cube = obs["policy"]["cube_pos"][0]
                phase_end_pos[0] = live_cube[0] + bias_x
                phase_end_pos[1] = live_cube[1] + bias_y

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
                # pads are committed around the cube). Uses the yaw-rotated
                # grip-bias so the offset stays along the gripper's local Y.
                if track:
                    live_cube = obs["policy"]["cube_pos"][0]
                    phase_end_pos[0] = live_cube[0] + bias_x
                    phase_end_pos[1] = live_cube[1] + bias_y

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

                # Position: P-controller, gain dropped to 2.0 in v4 to break
                # the saturation pathology of the prior runs. At gain 10, any
                # |pos_err| > 10 cm saturates action to ±1, making the first
                # action of every demo nearly identical regardless of cube
                # position (since home-to-target is always ~30-50 cm). At
                # gain 2, saturation only kicks in at |pos_err| > 50 cm —
                # rare in steady-state — so action varies linearly with
                # error and per-cube-position trajectories become more
                # distinguishable in the training data.
                pos_err = target_pos - cur_pos
                pos_delta = torch.clamp(pos_err * 2.0, -1.0, 1.0)

                # Orientation: drive the EE toward the top-down target every step.
                # apply_delta_pose interprets action[3:6] as a world-frame axis-angle
                # delta and premultiplies: q_new = q_delta * q_cur. So the correct
                # command is axis_angle(q_target * q_cur^-1). Previously this was
                # hard-zeroed, which is why the arm approached at its ~45° home tilt.
                rot_err = quat_err_axis_angle(cur_quat, q_target)
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

                # Track success (same logic as record_demos.py's process_success_condition).
                # In strict mode (default) any failure in the success term re-raises so
                # we don't silently train on `terminated[0]` while the real metric is
                # broken. Pass --no_strict to restore the soft-fail behaviour.
                if success_term is not None:
                    try:
                        is_success = bool(success_term.func(env.unwrapped, **success_term.params)[0])
                    except Exception as e:
                        if strict:
                            _log(f"  ERROR: success_term raised {type(e).__name__}: {e}; "
                                 "aborting (pass --no_strict to suppress)")
                            raise
                        _log(f"  WARN: success_term raised {type(e).__name__}: {e}; "
                             "falling back to terminated[0]")
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
        per_position_outcomes.append((forced_xy, success_reached, total_steps))

        # Record via recorder manager
        if success_reached:
            # The env auto-resets on termination INSIDE env.step(), so by the
            # time we get here `env.cube_color_state[0]` is the NEXT episode's
            # color — we have to use the snapshot captured at this episode's
            # reset (`ep_color`, line 289) for the demo we're about to export.
            ep_color_name, ep_color_idx = ep_color
            env.unwrapped.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.unwrapped.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
            )
            env.unwrapped.recorder_manager.export_episodes([0])
            successful += 1
            # Stamp per-episode color metadata on the HDF5 demo group so the
            # LeRobot converter can emit a per-episode prompt string. The
            # recorder writes placeholder demo_K entries (num_samples=0) for
            # post-reset stubs interleaved with the real demos, so
            # `demo_{successful - 1}` does NOT generally point at the latest
            # real demo. Find the highest-numbered demo with num_samples > 0
            # and no "task" attr already stamped, and stamp that.
            try:
                import h5py
                from envs.mdp.cube_palette import format_task_with_color
                with h5py.File(str(dataset_path), "a") as fh:
                    if "data" in fh:
                        keys = sorted(
                            fh["data"].keys(),
                            key=lambda k: int(k.split("_")[1]),
                        )
                        target_key = None
                        for k in reversed(keys):
                            d = fh["data"][k]
                            if int(d.attrs.get("num_samples", 0)) <= 0:
                                continue
                            if "task" in d.attrs:
                                continue
                            target_key = k
                            break
                        if target_key is not None:
                            d = fh["data"][target_key]
                            if ep_color_name:
                                d.attrs["cube_color"] = ep_color_name
                                d.attrs["cube_color_idx"] = int(ep_color_idx)
                            d.attrs["task"] = format_task_with_color(ep_color_name)
                        else:
                            msg = "no eligible demo group found for color stamp"
                            if strict:
                                _log(f"  ERROR: {msg}; aborting (pass --no_strict to suppress)")
                                raise RuntimeError(msg)
                            _log(f"  WARN: {msg}")
            except Exception as e:
                # The downstream LeRobot converter relies on per-episode `task`
                # attrs to emit colour-aware prompts; a silent failure here
                # silently trains the policy on placeholder prompts. Strict
                # mode (default) re-raises so the dataset isn't half-stamped.
                if strict:
                    _log(f"  ERROR: failed to stamp color attrs ({type(e).__name__}: {e}); "
                         "aborting (pass --no_strict to suppress)")
                    raise
                _log(f"  WARN: failed to stamp color attrs: {e}")

        env.unwrapped.recorder_manager.reset()

    _log(f"total: {successful}/{demo_target} successful demos in {attempted} attempts")

    if forced_positions is not None:
        _log("=" * 70)
        _log("Per-position sanity check results:")
        _log(f"  {'#':>3}  {'cube_xy':>20}  {'success':>8}  {'steps':>6}")
        for i, (xy, ok_i, n) in enumerate(per_position_outcomes):
            xy_str = f"({xy[0]:+.3f}, {xy[1]:+.3f})" if xy is not None else "random"
            _log(f"  {i:>3}  {xy_str:>20}  {str(ok_i):>8}  {n:>6}")
        _log(f"Forced-position summary: "
             f"{successful}/{len(per_position_outcomes)} succeeded")
        _log("=" * 70)

    env.close()
    sim_app.close()

    ok = successful >= demo_target
    _log(f"dataset: {dataset_path} ({dataset_path.stat().st_size if dataset_path.exists() else 0} bytes)")
    _log(f"result: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
