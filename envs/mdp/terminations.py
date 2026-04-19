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
