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
        # Gain-tuning attempts on this URDF-derived Robotiq:
        #   11.25/0.1/10  — canonical Isaac Lab UR10e_ROBOTIQ_2F_85_CFG;
        #                   too soft for our setup, fingers drifted to q=0.78
        #                   during arm swing despite OPEN command
        #   400/8/20      — intermediate; still finger_q=0.53 under swing
        #   5000/100/50   — holds OPEN cleanly but cube still didn't lift
        #                   (contact physics, not drive gains, is limiting)
        # Keeping the 5000/100/50 combo here because it at least gets the
        # cube correctly aligned between the pads; the lift failure is
        # believed to be in pad collision geometry or Robotiq mimic-chain
        # physics, which would need asset-level work to fix.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.0,
            stiffness=5000.0,
            damping=100.0,
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
