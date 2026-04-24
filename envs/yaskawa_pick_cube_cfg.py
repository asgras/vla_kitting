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

    # Target marker: bright-green 10×10 cm square placed flush with the table
    # surface at PLACE_XY=(0.65, 0.20). Visual-only (no collision, no rigid
    # body) so the cube rests directly on the table without climbing the mat.
    # The VLA needs this pixel cue to ground "the output location" — without
    # it the policy would have to memorize world coordinates.
    # Target marker: 20×20 cm, 1 cm thick, pure magenta. Magenta is the one
    # color empirically confirmed to render correctly on this static asset
    # path (green/cyan/emissive variants rendered in the cube's color due to
    # scene-replication material sharing). Outside the cube-color palette,
    # so the VLA can unambiguously ground "place it on the target".
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.20, 0.005)),
        spawn=CuboidCfg(
            size=(0.20, 0.20, 0.010),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 1.0),
            ),
        ),
    )

    # Wrist camera attached to the tool0 flange. The URDF importer nests bodies under
    # /Robot/root_joint/<link_name>/ rather than /Robot/<link_name>/.
    #
    # Mount: 8 cm along tool0's +X (outside the gripping plane), 5 cm along
    # tool0's +Z, aimed back at the pad center (0, 0, 0.14). This positions
    # the camera on the "open face" of the Robotiq 2F-85, looking past the
    # inner knuckles and fingers at the grasp point. Earlier axial mounts
    # (0, 0, 0.04) had the inner knuckles meeting at tool0's centerline
    # (Δ≈±1.3 cm in Y at +Z=0.061) blocking the middle of the frame — giving
    # the dreaded "two bright bars on a black background" view. Measured
    # with /tmp/probe_tool0_geom.py.
    # Quaternion computed via /tmp/cam_lookat.py with world_up=+X so that
    # image "up" aligns with tool0's +X axis.
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/tool0/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.02, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.05),
            rot=(0.66095, -0.25129, -0.25129, 0.66095),
            convention="ros",
        ),
    )

    # Third-person overhead camera fixed in the env. Re-aimed at the workspace
    # center (0.60, 0.10, 0.02) from (1.15, 0.1, 0.5) — the prior pose at
    # (1.3, 0, 1.0) with a shallow 30° tilt overshot the table by ~45 cm and
    # centered on the robot body. Rotation computed by /tmp/cam_lookat.py so
    # the target marker at (0.65, 0.20) sits squarely in-frame.
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
            pos=(1.15, 0.10, 0.50),
            rot=(-0.2926, 0.64373, 0.64373, -0.2926),
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
            # High friction models the real 2F-85's textured rubber pads
            # against a painted/plastic cube (μ ~1.5). Without this, the
            # cube could slip out of a marginal pinch during transport even
            # at low accelerations. friction_combine_mode="max" ensures the
            # contact actually uses this high value (PhysX default is
            # "average", which halves it against any low-μ robot material).
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.5,
                restitution=0.0,
                friction_combine_mode="max",
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
        # With scale=0.1, an action of 1.0 commands a 10 cm delta per control step.
        scale=0.1,
    )
    # Only finger_joint is driven; the 5 mimic joints follow via the USD's
    # PhysxMimicJointAPI.
    #
    # close target 0.79 (full close) tried first — too aggressive, the large
    # residual position error (~0.45 rad) after cube contact combined with
    # the mimic-chain asymmetry pushed the cube out of the passive-side pad.
    # Dropped to 0.45 — too low: cube holds fingers open at q≈0.50, so
    # position error goes NEGATIVE and the PD drive pushes fingers back open
    # at the -50 N·m effort cap, releasing the cube. Need target > q@contact.
    # Current 0.65 — cube holds fingers at q≈0.50 (contact), positive error
    # 0.15 rad * 5000 = 750 N·m saturates at +50 N·m gripping force, which
    # is the realistic continuous grasp force for a 2F-85 at pad contact.
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        # 0.5 rad is just past the 50 mm-cube contact angle (~0.43 rad
        # with full 85 mm open span). Lower than prior 0.65 to reduce
        # post-contact squeeze force that was launching the cube during
        # close. Matches the reference gear_large=0.45 pattern.
        close_command_expr={"finger_joint": 0.5},
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

    # Cube default spawn at (0.55, 0, 0.025). reset_root_state_uniform samples
    # a DELTA from this default within the ranges below. Widened from the
    # original (±0.08, ±0.10) to give Mimic a more diverse seed distribution —
    # still well inside arm reach and inside the camera framing.
    randomize_cube_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),   # sampled cube X ∈ [0.45, 0.65]
                "y": (-0.13, 0.13),   # sampled cube Y ∈ [-0.13, 0.13]
                "z": (0.0, 0.0),
                # Yaw randomization still disabled. With ±0.5 rad yaw, the
                # 50 mm cube's 70.7 mm diagonal clips the 2F-85 finger knuckles
                # during descent and bumps the cube sideways. Our scripted
                # controller reads cube_pos but not cube_rot, so it can't
                # align the gripper yaw. Re-enable once we teach the scripted
                # grasp to read cube_rot (or move to teleop).
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # Visual randomization so Mimic data isn't a sea of identical red-cube frames.
    randomize_cube_color = EventTerm(
        func=mdp.randomize_cube_color,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    randomize_light = EventTerm(
        func=mdp.randomize_dome_light_intensity,
        mode="reset",
        params={"prim_path": "/World/Light", "intensity_range": (1800.0, 3200.0)},
    )


# ---------------------------------------------------------- terminations ---
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success = cube released and settled at the output location, not just
    # transported above it. This means the scripted demo (and, later, the
    # learned policy) must actually PLACE the cube before the episode ends.
    success = DoneTerm(func=mdp.cube_placed_at_target)


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
