"""MimicEnvCfg for the cube pick-place task. Layers Mimic-specific subtask
definitions + eef/subtask observation groups on top of the existing
YaskawaPickCubeIkRelEnvCfg.
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .yaskawa_pick_cube_cfg import YaskawaPickCubeIkRelEnvCfg


@configclass
class _SubtaskTermsCfg(ObsGroup):
    """Binary subtask signals, one per boundary in the pick-place sequence.
    Each rises 0→1 at the moment the subtask completes so Mimic's
    annotate_demos tool can carve demos into reusable segments.
    """

    approach_done = ObsTerm(
        func=mdp.ee_above_cube,
        params={
            "xy_tolerance": 0.05,
            "min_height_above_cube": 0.05,
            "robot_cfg": SceneEntityCfg("robot"),
            "body_name": "tool0",
            "object_cfg": SceneEntityCfg("cube"),
        },
    )
    grasp_done = ObsTerm(
        func=mdp.cube_gripped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "knuckle_joint": "finger_joint",
            "closed_threshold": 0.35,
            "min_lift_height": 0.05,
            "object_cfg": SceneEntityCfg("cube"),
        },
    )
    transport_done = ObsTerm(
        func=mdp.cube_above_target_xy,
        params={
            "target_xy": (0.65, 0.20),
            "xy_tolerance": 0.10,
            "min_height": 0.08,
            "object_cfg": SceneEntityCfg("cube"),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _MimicPolicyCfg(ObsGroup):
    """Policy obs for Mimic: reuses the parent env's scalar obs plus adds
    eef_pos / eef_quat / gripper_pos that Mimic reads directly.
    """
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    eef_pos = ObsTerm(func=mdp.ee_pos_world)
    eef_quat = ObsTerm(func=mdp.ee_quat_world)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    gripper_closed = ObsTerm(func=mdp.gripper_is_closed)
    cube_pos = ObsTerm(func=mdp.cube_position_in_world_frame)
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


@configclass
class YaskawaPickCubeIkRelMimicEnvCfg(YaskawaPickCubeIkRelEnvCfg, MimicEnvCfg):
    """Cfg entry point for Isaac Lab Mimic's data generator.

    Inherits the full pick-place env (scene, actions, events, terminations)
    and mixes in MimicEnvCfg's datagen_config + subtask_configs API. The
    observation groups are overridden in __post_init__ because we need
    `eef_pos` / `eef_quat` split out plus a `subtask_terms` group.
    """

    def __post_init__(self):
        super().__post_init__()

        # Swap observations to Mimic-compatible groups.
        self.observations.policy = _MimicPolicyCfg()
        # ObsGroup added dynamically — manager picks this up via attribute scan.
        self.observations.subtask_terms = _SubtaskTermsCfg()

        # Datagen defaults — mirror the Franka reference, tuned for
        # num_envs=1 generation (we'll bump num_envs on the CLI for parallel
        # trials rather than hard-coding here).
        self.datagen_config.name = "demo_src_yaskawa_pick_cube"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        # Raised from 50 so the generator doesn't stop early when success rate
        # dips mid-run (observed smoke: run stopped after ~22 failures at 33%).
        self.datagen_config.max_num_failures = 500
        self.datagen_config.seed = 1

        # Subtask sequence: approach → grasp → transport → place.
        # All subtasks manipulate the cube, so object_ref="cube" for each.
        # The final "place" subtask has subtask_term_signal=None — Mimic
        # doesn't need a boundary after the last subtask.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="approach_done",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach the cube",
                next_subtask_description="Close the gripper to grasp the cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="grasp_done",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the cube",
                next_subtask_description="Transport cube above the target marker",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="transport_done",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Transport cube above the magenta circle",
                next_subtask_description="Release cube onto the magenta circle",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal=None,  # final segment — no boundary after
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lower the cube and release",
            )
        )
        # Keyed by eef_name — we have one ("yaskawa"); matched by Mimic's
        # action/pose dict-unpacking.
        self.subtask_configs = {"yaskawa": subtask_configs}
