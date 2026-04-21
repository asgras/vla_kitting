"""Task-specific termination predicates."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_lifted_over_target(
    env: "ManagerBasedRLEnv",
    min_height: float = 0.10,
    target_xy: tuple[float, float] = (0.65, 0.20),
    xy_tolerance: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Success when cube is > min_height above the table (z=0 plane) AND within xy_tolerance of target."""
    cube: RigidObject = env.scene[object_cfg.name]
    pos = cube.data.root_pos_w  # (N, 3)
    height_ok = pos[:, 2] > min_height
    tx, ty = target_xy
    dx = pos[:, 0] - tx
    dy = pos[:, 1] - ty
    xy_ok = torch.sqrt(dx * dx + dy * dy) < xy_tolerance
    return height_ok & xy_ok


def cube_placed_at_target(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.65, 0.20),
    xy_tolerance: float = 0.08,
    max_resting_height: float = 0.03,
    max_speed: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Success when cube is *placed* at the target: xy within tolerance, cube
    resting on the table (z < max_resting_height), and nearly stationary (lin
    speed < max_speed). The 0.03 height cutoff (vs the cube's 0.025 resting
    center) rejects a cube still pinched in the gripper at PLACE_Z = 0.18 —
    there the pad sits at z≈0.04 and the cube center at z≈0.034, so a 0.05
    threshold would fire before the gripper has actually released. This way
    success only fires after the fingers open and the cube settles.
    """
    cube: RigidObject = env.scene[object_cfg.name]
    pos = cube.data.root_pos_w
    lin_vel = cube.data.root_lin_vel_w
    tx, ty = target_xy
    dx = pos[:, 0] - tx
    dy = pos[:, 1] - ty
    xy_ok = torch.sqrt(dx * dx + dy * dy) < xy_tolerance
    resting_ok = pos[:, 2] < max_resting_height
    slow_ok = torch.linalg.vector_norm(lin_vel, dim=-1) < max_speed
    return xy_ok & resting_ok & slow_ok
