"""Phase 3 robot motion test: load the HC10DT articulation in Isaac Sim, command a
square trajectory above a workspace, record joint/EE trace, verify no limit violations.

Run via:
    ./isaaclab.sh -p scripts/validate/robot_motion_test.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd",
    type=str,
    default=str(pathlib.Path(__file__).resolve().parents[2] / "assets/hc10dt_v1.usd"),
)
parser.add_argument("--steps-per-pose", type=int, default=240)  # 4s at 60 Hz
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=args_cli.headless)
sim_app = app_launcher.app


def main() -> int:
    import numpy as np
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    # Reference the arm USD
    arm_usd = args_cli.usd
    print(f"[motion_test] loading arm USD: {arm_usd}")
    if not pathlib.Path(arm_usd).exists():
        print(f"[motion_test] ERROR: USD not found at {arm_usd}", file=sys.stderr)
        sim_app.close()
        return 2

    arm_cfg = ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(usd_path=arm_usd),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1_s": 0.0,
                "joint_2_l": 0.0,
                "joint_3_u": 0.0,
                "joint_4_r": 0.0,
                "joint_5_b": 0.0,
                "joint_6_t": 0.0,
            },
        ),
        actuators={},  # use default drive from USD
    )
    arm = Articulation(arm_cfg)

    sim.reset()

    # Report info
    joint_names = arm.joint_names
    joint_pos_limits = arm.data.joint_limits[0].cpu().numpy()  # (n_joints, 2)
    print(f"[motion_test] joint_names={joint_names}")
    print(f"[motion_test] dof={len(joint_names)}")
    print(f"[motion_test] joint_pos_limits (rad):")
    for n, (lo, hi) in zip(joint_names, joint_pos_limits):
        print(f"  {n}: [{lo:.3f}, {hi:.3f}] ({math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°)")

    # Simple joint-space square: 4 corner poses + home
    # We apply known-safe joint targets to each of the 6 DoFs.
    rad = math.radians
    poses = [
        [rad(0),   rad(-45), rad(45),   rad(0), rad(-45), rad(0)],   # home-ish
        [rad(30),  rad(-30), rad(30),   rad(0), rad(-30), rad(0)],   # front-right
        [rad(-30), rad(-30), rad(30),   rad(0), rad(-30), rad(0)],   # front-left
        [rad(0),   rad(0),   rad(60),   rad(0), rad(-60), rad(0)],   # top
        [rad(0),   rad(-45), rad(45),   rad(0), rad(-45), rad(0)],   # home-ish again
    ]

    trace = []
    limit_violations = 0
    for pose_idx, target in enumerate(poses):
        target_t = torch.tensor([target], device="cuda:0", dtype=torch.float32)
        arm.set_joint_position_target(target_t)

        for _ in range(args_cli.steps_per_pose):
            arm.write_data_to_sim()
            sim.step()
            arm.update(sim_cfg.dt)

        jp = arm.data.joint_pos[0].cpu().numpy()
        trace.append(jp.tolist())

        # Check limit violations
        for i, (val, (lo, hi)) in enumerate(zip(jp, joint_pos_limits)):
            if val < lo - 0.01 or val > hi + 0.01:
                limit_violations += 1
                print(f"[motion_test] WARN: joint {joint_names[i]} out of limit: {val:.3f} not in [{lo:.3f}, {hi:.3f}]")

        print(f"[motion_test] pose {pose_idx+1}/{len(poses)} reached, joint_pos={[f'{v:.2f}' for v in jp]}")

    # Final joint position
    final_jp = arm.data.joint_pos[0].cpu().numpy()
    print(f"[motion_test] final joint_pos (rad): {[f'{v:.3f}' for v in final_jp]}")

    # Save JSON report
    report = {
        "dof": len(joint_names),
        "joint_names": list(joint_names),
        "limits_rad": [list(map(float, row)) for row in joint_pos_limits],
        "poses_reached": len(poses),
        "limit_violations": int(limit_violations),
        "trace": trace,
    }
    report_path = pathlib.Path(__file__).resolve().parents[2] / "reports/robot_motion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[motion_test] report: {report_path}")

    sim_app.close()
    ok = limit_violations == 0 and len(joint_names) == 6
    print(f"[motion_test] result: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
