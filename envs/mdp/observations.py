"""Task-specific observation functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_position_in_world_frame(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w


def ee_pose_world(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "tool0",
) -> torch.Tensor:
    """Returns (N, 7) EE pose: xyz + quat (w, x, y, z)."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    pos = robot.data.body_pos_w[:, body_idx]
    quat = robot.data.body_quat_w[:, body_idx]
    return torch.cat([pos, quat], dim=-1)


def gripper_is_closed(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knuckle_joint: str = "finger_joint",
    threshold: float = 0.4,
) -> torch.Tensor:
    """Binary indicator: 1 if the gripper's drive joint is past `threshold`."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.joint_names.index(knuckle_joint)
    pos = robot.data.joint_pos[:, idx]
    return (pos > threshold).float().unsqueeze(-1)


# --- Mimic-style scalar observations (eef/gripper split) --------------------
# ManagerBasedRLMimicEnv.get_robot_eef_pose() reads eef_pos + eef_quat out of
# obs_buf["policy"]. We split the 7D ee_pose into two terms so the Mimic
# wrapper can consume them without knowing our custom layout.

def ee_pos_world(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "tool0",
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_idx]


def ee_quat_world(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "tool0",
) -> torch.Tensor:
    """(N, 4) quaternion in (w, x, y, z) order — matches isaaclab.utils.math conventions."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    return robot.data.body_quat_w[:, body_idx]


def gripper_pos(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knuckle_joint: str = "finger_joint",
) -> torch.Tensor:
    """Raw finger_joint angle (rad), clamped to its limits by the sim. Shape (N, 1)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.joint_names.index(knuckle_joint)
    return robot.data.joint_pos[:, idx].unsqueeze(-1)


# --- Subtask termination signals for Mimic annotation -----------------------
# Each signal is a binary (N,) float indicator that flips 0→1 at the moment
# the corresponding subtask completes. Mimic's annotate_demos tool scans a
# demo and records the first rising edge per signal — those are the segment
# boundaries it later splices across episodes.

def cube_above_target_xy(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.65, 0.20),
    xy_tolerance: float = 0.10,
    min_height: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Subtask 3 signal: cube has been transported above the output target,
    still held high. Fires during the transport hold / just before descent."""
    cube: RigidObject = env.scene[object_cfg.name]
    pos = cube.data.root_pos_w
    tx, ty = target_xy
    dx = pos[:, 0] - tx
    dy = pos[:, 1] - ty
    xy_ok = torch.sqrt(dx * dx + dy * dy) < xy_tolerance
    height_ok = pos[:, 2] > min_height
    return (xy_ok & height_ok).float()


def cube_gripped(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knuckle_joint: str = "finger_joint",
    closed_threshold: float = 0.35,
    min_lift_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Subtask 2 signal: gripper is closed AND cube is lifted above the
    table. Fires during the lift phase."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_idx = robot.joint_names.index(knuckle_joint)
    fingers_closed = robot.data.joint_pos[:, finger_idx] > closed_threshold

    cube: RigidObject = env.scene[object_cfg.name]
    lifted = cube.data.root_pos_w[:, 2] > min_lift_height
    return (fingers_closed & lifted).float()


def ee_above_cube(
    env: "ManagerBasedRLEnv",
    xy_tolerance: float = 0.05,
    min_height_above_cube: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "tool0",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Subtask 1 signal: EE is hovering above the cube with open gripper.
    Fires at the end of the approach phase — EE's XY matches cube's XY, EE
    Z is above cube top."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    ee_pos = robot.data.body_pos_w[:, body_idx]

    cube: RigidObject = env.scene[object_cfg.name]
    cube_pos = cube.data.root_pos_w

    dx = ee_pos[:, 0] - cube_pos[:, 0]
    dy = ee_pos[:, 1] - cube_pos[:, 1]
    xy_ok = torch.sqrt(dx * dx + dy * dy) < xy_tolerance
    height_ok = ee_pos[:, 2] > (cube_pos[:, 2] + min_height_above_cube)
    return (xy_ok & height_ok).float()
