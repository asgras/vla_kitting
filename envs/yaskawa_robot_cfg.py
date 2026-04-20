"""Asset configuration for the HC10DT + Robotiq 2F-85 unified articulation."""
from __future__ import annotations

import pathlib

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
HC10DT_ROBOTIQ_USD = str(_ASSETS / "hc10dt_with_gripper_v1.usd")


HC10DT_ROBOTIQ_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=HC10DT_ROBOTIQ_USD,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_1_s": 0.0,
            "joint_2_l": -0.785,   # -45°  upper-arm lifted
            "joint_3_u": 0.785,    # +45°  forearm horizontal
            "joint_4_r": 0.0,
            "joint_5_b": -0.785,    # -45°  (gripper points ~45° from vertical — tilt is small enough with corrected mimic)
            "joint_6_t": 0.0,
            # All 6 Robotiq mimic joints explicitly at 0 — the PhysX mimic
            # constraint isn't enforcing the kinematic tie in our USD, so we
            # drive all joints directly via the action (see yaskawa_pick_cube_cfg).
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "robotiq_85_left_inner_knuckle_joint": 0.0,
            "robotiq_85_right_inner_knuckle_joint": 0.0,
            "robotiq_85_left_finger_tip_joint": 0.0,
            "robotiq_85_right_finger_tip_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            # Phase 7 teleop test showed stiffness=1200 wasn't enough to track IK targets
            # against gravity — the EE was sagging ~90 mm/s in -Z while the IK commanded
            # +X, and visible motion was negligible. 8000 gives plenty of margin.
            stiffness=8000.0,
            damping=500.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_.*"],
            stiffness=2000.0,
            damping=50.0,
            effort_limit_sim=80.0,
        ),
    },
)
