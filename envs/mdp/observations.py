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
    knuckle_joint: str = "robotiq_85_left_knuckle_joint",
    threshold: float = 0.4,
) -> torch.Tensor:
    """Binary indicator: 1 if gripper knuckle position is past threshold (closing)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.joint_names.index(knuckle_joint)
    pos = robot.data.joint_pos[:, idx]
    return (pos > threshold).float().unsqueeze(-1)
