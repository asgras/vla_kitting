"""HC10DT arm + canonical ros-industrial-attic Robotiq 2F-85, built from URDF.

The gripper topology matches Isaac Lab's UR10e_ROBOTIQ_2F_85_CFG (10 bodies per
side including inner_finger_pad). scripts/assembly/urdf_to_usd.py has applied
PhysxMimicJointAPI to the 5 mimic joints, so only finger_joint is driven and
the solver enforces the closed-loop kinematics.
"""
from __future__ import annotations

import pathlib

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
HC10DT_ROBOTIQ_USD = str(_ASSETS / "hc10dt_with_ria_gripper.usd")


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
            "joint_2_l": -0.785,
            "joint_3_u": 0.785,
            "joint_4_r": 0.0,
            "joint_5_b": -0.785,
            "joint_6_t": 0.0,
            # Open configuration.
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=8000.0,
            damping=500.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
        ),
        # Three-group split copied from Isaac Lab's canonical
        # UR10e_ROBOTIQ_2F_85_CFG:
        #   gripper_drive  — finger_joint (primary close drive)
        #   gripper_finger — left/right_inner_finger_joint (low PD to keep
        #                    the parallel-linkage end symmetric)
        #   gripper_passive— other mimics (knuckles): zero PD, their
        #                    position is enforced by the URDF mimic
        #                    constraint preserved in the USD.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            stiffness=500.0,
            damping=10.0,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
