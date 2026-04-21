"""HC10DT arm + ros-industrial-attic Robotiq 2F-85, built from URDF.

Drive gains are mirrored from Isaac Lab's canonical UR10e_ROBOTIQ_2F_85_CFG
(IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py).
That config is battle-tested for stacking cubes in the Isaac-Stack-Cube-*
tasks, so if gravity/grip fails here it's probably something else (asset,
contact materials) rather than a gain issue.

Tried swapping to NVIDIA's canonical Robotiq USD via a reference composition
(assets/hc10dt_with_nvidia_gripper.usd) but the composition hit PhysX errors
"Rigid Body missing xformstack reset when child of another enabled rigid
body" and broken internal joint body0/body1 paths. Isaac Lab's UR10e combines
arm + gripper via USD **variants** baked into the arm USD, not via runtime
references — a different composition pattern than our URDF → USD → reference
pipeline. Rewriting that is out of scope here.

The gripper joint names (from our URDF converted in scripts/assembly/urdf_to_usd.py
with --gripper=ria) are:
  finger_joint                 — primary drive (Isaac Lab BinaryJointPositionAction)
  right_outer_knuckle_joint    — mimic
  left_inner_knuckle_joint     — mimic (!= NVIDIA's inner_finger_knuckle)
  right_inner_knuckle_joint    — mimic (!= NVIDIA's inner_finger_knuckle)
  left_inner_finger_joint      — mimic
  right_inner_finger_joint     — mimic
"""
from __future__ import annotations

import pathlib

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
HC10DT_ROBOTIQ_USD = str(_ASSETS / "hc10dt_with_ria_gripper.usd")


HC10DT_ROBOTIQ_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=HC10DT_ROBOTIQ_USD,
        activate_contact_sensors=True,
        # Mirrors the working UR10e2F85GearAssemblyEnvCfg at
        # isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/
        # config/ur_10e/joint_pos_env_cfg.py (a known-working Robotiq 2F-85
        # pick task). disable_gravity=True is the key reference-proven
        # setting that keeps the fingers from sagging closed under their
        # own weight during arm motion. contact_offset 0.005 + rest_offset
        # 0 tightens contact detection for the pad-cube grasp.
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0,
        ),
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
        # Values verbatim from the working UR10e2F85GearAssemblyEnvCfg at
        # isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/
        # config/ur_10e/joint_pos_env_cfg.py. That task is Isaac Lab's
        # reference-proven Robotiq 2F-85 manipulation config.
        #
        # Critical difference from canonical UR10e_ROBOTIQ_2F_85_CFG:
        # gripper_finger stiffness jumps 0.2 → 10.0 (50×) with matching
        # damping. This is what keeps the pads parallel during close
        # under contact load — without it our pads were independent enough
        # that the cube popped out laterally.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
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
