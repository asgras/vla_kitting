"""Phase 4 gripper smoke: load the combined HC10DT + Robotiq articulation into Isaac Lab,
drive the gripper open/close, report that joints respond without physics explosions.

Run via:
    ./isaaclab.sh -p scripts/validate/gripper_smoke.py
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd",
    type=str,
    default=str(pathlib.Path(__file__).resolve().parents[2] / "assets/hc10dt_with_gripper_v1.usd"),
)
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=args_cli.headless)
sim_app = app_launcher.app


def _log(msg):
    print(f"[gripper_smoke] {msg}", flush=True)


def main() -> int:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext

    usd_path = pathlib.Path(args_cli.usd)
    if not usd_path.exists():
        _log(f"ERROR: USD not found at {usd_path}")
        sim_app.close()
        return 2

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    _log(f"loading USD: {usd_path}")
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={},  # let defaults from USD apply
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"],
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["robotiq_85_.*"],
                stiffness=500.0,
                damping=20.0,
                effort_limit_sim=80.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    _log("articulation loaded")

    joint_names = list(robot.joint_names)
    _log(f"joint_names ({len(joint_names)}): {joint_names}")

    # Identify the gripper drive joint
    LEFT_KNUCKLE = "robotiq_85_left_knuckle_joint"
    if LEFT_KNUCKLE not in joint_names:
        _log(f"ERROR: {LEFT_KNUCKLE} not in joint list")
        sim_app.close()
        return 3
    knuckle_idx = joint_names.index(LEFT_KNUCKLE)

    # Joint limits for the knuckle (from USD): [0, 0.8 rad ≈ 45.8°]
    limits = robot.data.joint_limits[0].cpu().numpy()
    lo, hi = limits[knuckle_idx]
    _log(f"knuckle range: [{lo:.3f}, {hi:.3f}] rad ({math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°)")

    # Script: open → close → open
    def drive(target_rad: float, steps: int = 180):
        target = robot.data.joint_pos[0].clone()
        target[knuckle_idx] = target_rad
        robot.set_joint_position_target(target.unsqueeze(0))
        for _ in range(steps):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg.dt)

    _log("open → close → open")
    drive(float(lo) + 0.01)  # open
    open_pos = robot.data.joint_pos[0, knuckle_idx].item()
    _log(f"after open: knuckle={open_pos:.3f} rad")

    drive(float(hi) - 0.01)  # close
    closed_pos = robot.data.joint_pos[0, knuckle_idx].item()
    _log(f"after close: knuckle={closed_pos:.3f} rad")

    drive(float(lo) + 0.01)  # open again
    reopen_pos = robot.data.joint_pos[0, knuckle_idx].item()
    _log(f"after reopen: knuckle={reopen_pos:.3f} rad")

    # Success: gripper moved by >=0.3 rad between open and close
    moved = abs(closed_pos - open_pos)
    _log(f"range moved: {moved:.3f} rad ({math.degrees(moved):.1f}°)")
    ok = moved >= 0.3

    # Check arm DOF count too — we expect 6 arm + 6 gripper = 12 revolute
    ok = ok and len(joint_names) >= 12

    _log(f"result: {'OK' if ok else 'FAIL'}")
    sim_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
