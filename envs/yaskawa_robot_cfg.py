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
            "joint_2_l": -0.785,   # -45°
            "joint_3_u": 0.785,    # +45°
            "joint_4_r": 0.0,
            "joint_5_b": -0.785,   # -45° — end-effector pointing down
            "joint_6_t": 0.0,
            "robotiq_85_left_knuckle_joint": 0.0,  # open
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=1200.0,
            damping=80.0,
            effort_limit_sim=400.0,
            velocity_limit_sim=2.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_.*"],
            stiffness=500.0,
            damping=20.0,
            effort_limit_sim=80.0,
        ),
    },
)
