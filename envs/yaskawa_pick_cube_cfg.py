"""Isaac Lab env config for HC10DT + Robotiq cube pick-place (IK-Rel action space)."""
from __future__ import annotations

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from . import mdp
from .yaskawa_robot_cfg import HC10DT_ROBOTIQ_CFG


# -------------------------------------------------------------------- scene ---
@configclass
class PickCubeSceneCfg(InteractiveSceneCfg):
    # Environment ground plane (below workspace)
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=GroundPlaneCfg(),
    )
    # Dome light (warmed up slightly)
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.9), intensity=2500.0),
    )
    # HC10DT + Robotiq
    robot = HC10DT_ROBOTIQ_CFG.copy()

    # Table: 1.2 x 0.8 x 0.04 box centered at (0.6, 0, -0.02)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, -0.02)),
        spawn=CuboidCfg(
            size=(1.2, 0.8, 0.04),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.18)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Wrist camera attached to the tool0 flange. The URDF importer nests bodies under
    # /Robot/root_joint/<link_name>/ rather than /Robot/<link_name>/.
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/tool0/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.02, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.10), rot=(0.0, 0.707, 0.707, 0.0), convention="ros"
        ),
    )

    # Third-person overhead camera fixed in the env
    third_person_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.3, 0.0, 1.0), rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )

    # The cube to pick
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8, restitution=0.0,
            ),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
    )


# ------------------------------------------------------------------ actions ---
@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],
        body_name="tool0",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        # With scale=0.1, an action of 1.0 commands a 10 cm delta per control step
        # (sim runs at 60 Hz after decimation=2 / dt=1/120). The scripted controller
        # then effectively caps step size to 1-2 cm for stability.
        scale=0.1,
    )
    # Only finger_joint is driven; the 5 mimic joints follow via the USD's
    # PhysxMimicJointAPI. Close target ≈ 0.79 rad (the 2F-85's "closed"
    # position per its URDF limit of 0.8 rad).
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.79},
    )


# ------------------------------------------------------------- observations ---
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pose = ObsTerm(func=mdp.ee_pose_world)
        gripper_closed = ObsTerm(func=mdp.gripper_is_closed)
        cube_pos = ObsTerm(func=mdp.cube_position_in_world_frame)
        actions = ObsTerm(func=mdp.last_action)
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        third_person_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("third_person_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ----------------------------------------------------------------- events ---
@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # Cube default spawn at (0.55, 0, 0.025). reset_root_state_uniform samples a DELTA
    # from this default within the ranges below, so we stay within arm reach.
    randomize_cube_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ---------------------------------------------------------- terminations ---
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.cube_lifted_over_target)


# ----------------------------------------------------------------- env cfg ---
@configclass
class YaskawaPickCubeIkRelEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickCubeSceneCfg = PickCubeSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # No rewards needed (imitation learning). rewards field must exist on the manager based cfg.
    rewards = None
    curriculum = None
    commands = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Fabric ON: required for the RTX viewport to show physics updates (the
        # Fabric Scene Delegate reads transforms from Fabric, not USD). The missing
        # wp.transform_compose in the bundled omni.warp.core-1.7.1 is worked around
        # by symlinking the extension's warp subdir to the newer pip warp (1.12.1)
        # at /opt/IsaacSim/extscache/omni.warp.core-1.7.1+lx64/warp. Backup at warp.bak.
        self.sim.use_fabric = True
        self.viewer.eye = (1.5, 1.0, 1.2)
        self.viewer.lookat = (0.55, 0.0, 0.05)

        # Keyboard teleop. pos/rot_sensitivity are the per-press delta that multiplies
        # the IK action scale (0.1) each control step. 0.3 * 0.1 = 3 cm of commanded
        # target motion per step → ~0.9 m/s when holding — strong enough to dominate
        # the small free-run IK target drift seen in the current env.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.3,
                    rot_sensitivity=0.5,
                    sim_device=self.sim.device,
                ),
            }
        )
