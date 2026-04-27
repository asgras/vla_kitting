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
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg
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

    # Table: 1.5 x 1.0 x 0.04 box centered at (0.6, 0, -0.02). Widened from
    # 1.2 x 0.8 (vla_kitting-8tf) so the widened cube box (cube X ∈
    # [0.40, 0.70], Y ∈ [-0.22, 0.22]) plus the 10 cm magenta target circle at
    # (0.65, 0.20) plus a 5 cm safety margin all fit clearly within the table
    # surface. Resulting table X spans [-0.15, 1.35], Y spans [-0.50, 0.50].
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, -0.02)),
        spawn=CuboidCfg(
            size=(1.5, 1.0, 0.04),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.18)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Target marker: 10 cm diameter, 1 cm thick magenta cylinder (a flat disk)
    # placed flush with the table surface at PLACE_XY=(0.65, 0.20). Visual-
    # only (no collision, no rigid body) so the cube rests directly on the
    # table without climbing the mat. The VLA needs this pixel cue to
    # ground "the output location" — without it the policy would have to
    # memorize world coordinates.
    #
    # Was a 20×20 cm magenta square (CuboidCfg). Swapped to a 10 cm circle
    # (vla_kitting-usq) so the marker is (a) more salient as a grounding
    # cue, (b) tighter precision target for the place phase, and (c) a
    # different visual signature that breaks any patch-aligned attention
    # bias the prior square may have introduced.
    #
    # Magenta retained — it's the one color empirically confirmed to render
    # correctly on this static asset path (green/cyan/emissive variants
    # rendered in the cube's color due to scene-replication material
    # sharing). Outside the cube-color palette, so the VLA can
    # unambiguously ground "place it on the magenta circle".
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.20, 0.005)),
        spawn=CylinderCfg(
            radius=0.05,
            height=0.010,
            axis="Z",
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
        # 128 -> 256: SmolVLA pads inputs to 512x512 internally, so 256 native
        # pixels means a single 2x upsample instead of 4x. The cube at grasp
        # distance (~5-15 cm from the lens) was 30-50 px wide at 128 — fine
        # for "is there a cube" but lossy for finger-cube alignment cues.
        height=256,
        width=256,
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

    # Third-person overhead camera fixed in the env. Re-positioned 2026-04-27
    # to cover the entire spawn box + target marker without cropping. The
    # prior pose at (1.15, 0.10, 0.50) with focal_length=24 (FOV ~47°) cropped
    # the cube on 3 of 5 random poses (samples in
    # reports/camera_samples/2026-04-27_current_scene). New aim point is the
    # centroid of [spawn box ∪ target] = (0.55, -0.10, 0.025); camera pulled
    # back to (1.5, -0.10, 0.80) and widened to focal_length=18 (FOV ~60°).
    # Coverage on the table plane: X ∈ [-3.4, 1.21] (well past spawn X[0.25,0.85])
    # and Y ∈ [-0.82, 0.62] at workspace depth (well past spawn Y[-0.40,0.00]
    # and target Y=0.20). Quaternion computed via /tmp/cam_lookat.py with
    # world_up=+Z so image "up" points away from the robot base.
    third_person_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_cam",
        update_period=0.0,
        # 256 -> 512: matches SmolVLA's resize_imgs_with_padding=(512,512), so
        # the policy sees native pixels with no upscaling at all. The cube at
        # ~1.2 m from the camera was ~6-12 px across at 256 — borderline
        # for color reading, marginal for shape.
        height=512,
        width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, -0.10, 0.80),
            rot=(-0.30326, 0.63877, 0.63877, -0.30326),
            convention="ros",
        ),
    )

    # The cube to pick
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.20, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
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
        # Static-per-episode palette index, set at reset by
        # randomize_cube_color. Lets the recorder + LeRobot converter +
        # closed-loop eval all derive the per-episode color word from
        # one source of truth (envs.mdp.cube_palette).
        cube_color_idx = ObsTerm(func=mdp.cube_color_idx)
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

    # Cube default spawn at (0.55, -0.20, 0.025). reset_root_state_uniform
    # samples a DELTA from this default within the ranges below.
    # Region rotated 90° (long axis along X) and shifted toward the -Y table
    # edge so the spawn rectangle no longer overlaps the (0.65, 0.20) target
    # disk. Resulting absolute spawn box: X ∈ [0.25, 0.85], Y ∈ [-0.40, 0.00],
    # 0.10 m clear of the -Y table edge (table Y ∈ [-0.50, 0.50]) and 0.12 m
    # clear of the 8 cm target tolerance circle.
    # Yaw remains ±0.5 rad — the scripted controller in
    # scripts/validate/scripted_pick_demo.py reads the cube quat at episode
    # start and rotates both the gripper target orientation and the
    # GRIP_BIAS_Y offset into world frame. The 50 mm cube diagonal across
    # ±0.5 rad stays inside the 85 mm 2F-85 open span.
    randomize_cube_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.30, 0.30),   # sampled cube X ∈ [0.25, 0.85]
                "y": (-0.20, 0.20),   # sampled cube Y ∈ [-0.40, 0.00]
                "z": (0.0, 0.0),
                "yaw": (-0.5, 0.5),   # cube yaw ∈ [-0.5, 0.5] rad
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
        # 120 Hz physics / decimation 4 → 30 Hz policy rate. 30 Hz matches
        # SmolVLA's pretrain distribution (SO-100 at 30 Hz) — neither the
        # prior 60 Hz (decimation=2) nor the shelved 15 Hz (decimation=8)
        # attempts match pretrain action statistics. v3 restart per
        # reports/runs/vision_grounded_wide_15hz_2026-04-24/run_diary.md.
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Fabric ON: required for the RTX viewport to show physics updates (the
        # Fabric Scene Delegate reads transforms from Fabric, not USD). The missing
        # wp.transform_compose in the bundled omni.warp.core-1.7.1 is worked around
        # by symlinking the extension's warp subdir to the newer pip warp (1.12.1)
        # at /opt/IsaacSim/extscache/omni.warp.core-1.7.1+lx64/warp. Backup at warp.bak.
        self.sim.use_fabric = True
        # DLSS is the IsaacLab default antialiasing_mode. It reconstructs each
        # frame from a lower-resolution input plus motion vectors and a
        # multi-frame history. When the cube teleports during reset
        # (write_root_pose_to_sim), DLSS has no motion vector for the
        # instantaneous jump, so prior-frame samples of the cube at its
        # previous spawn pose bleed into the new frame as faint "ghost cubes."
        # That ghost trail was visible in cam_check/sample_*.png on
        # 2026-04-25 — multiple cube-shaped objects per frame, with the
        # current-reset cube the brightest and stale-reset cubes fading. The
        # eval policy was therefore being fed images with phantom cubes the
        # training distribution (recorded HDF5) never contains, plausibly the
        # cause of the v3/v3.1/v3.2 visual mode collapse. FXAA is non-temporal
        # so no history bleeds across resets. Off would also work but loses
        # all anti-aliasing on cube/gripper edges. See
        # reports/2026-04-26_eval_ghost_cube_investigation.md for the
        # diagnosis.
        self.sim.render.antialiasing_mode = "FXAA"
        self.viewer.eye = (1.5, 1.0, 1.2)
        self.viewer.lookat = (0.55, -0.20, 0.05)

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
