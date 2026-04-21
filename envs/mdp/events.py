"""Task-specific reset-time randomization events.

Color and light intensity randomization run each reset so scripted/Mimic
demos produce visually varied frames — critical for the VLA not to overfit
to a single red-cube / fixed-lighting rendering.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Palette of distinct, pickable cube colors. Kept small so the policy sees
# each one plenty of times within a 750-demo Mimic budget. Values are RGB in
# [0, 1]; chosen to contrast with the green target marker and brown table.
_CUBE_COLOR_PALETTE: list[tuple[float, float, float]] = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.35, 0.90),  # blue
    (0.95, 0.80, 0.10),  # yellow
    (0.90, 0.45, 0.10),  # orange
    (0.70, 0.15, 0.80),  # purple
]


def _set_preview_surface_color(prim_path: str, rgb: tuple[float, float, float]) -> None:
    """Walk the entire subtree at prim_path and set every UsdPreviewSurface
    shader's diffuseColor input to the given rgb. Isaac Lab's CuboidCfg
    buries the shader three levels deep under ``.../geometry/material/Shader``,
    so a recursive walk is required.
    """
    from pxr import UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return

    def _walk(prim):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            diffuse = shader.GetInput("diffuseColor")
            if diffuse is not None:
                diffuse.Set(tuple(rgb))
        for child in prim.GetAllChildren():
            _walk(child)

    _walk(root)


def randomize_cube_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    palette: list[tuple[float, float, float]] | None = None,
) -> None:
    """Pick a color from the palette at each reset and apply it to the cube's
    PreviewSurface material.

    Isaac Lab stores `asset.cfg.prim_path` with `{ENV_REGEX_NS}` already
    expanded to the regex form `/World/envs/env_.*/Cube`. We only run
    num_envs=1 for scripted/Mimic generation, so resolve per-env paths by
    iterating env_ids and rebuilding `/World/envs/env_<i>/Cube` directly.
    """
    colors = palette if palette is not None else _CUBE_COLOR_PALETTE
    asset = env.scene[asset_cfg.name]
    # Strip the regex stem to recover the suffix (e.g. "Cube" from
    # "/World/envs/env_.*/Cube") so this function also works if the cube is
    # ever parented under a sub-scope.
    suffix = asset.cfg.prim_path.split("/World/envs/env_.*")[-1]
    rgb = random.choice(colors)
    for idx in env_ids.tolist():
        prim_path = f"/World/envs/env_{idx}{suffix}"
        _set_preview_surface_color(prim_path, rgb)


def randomize_dome_light_intensity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    prim_path: str = "/World/Light",
    intensity_range: tuple[float, float] = (1800.0, 3200.0),
) -> None:
    """Sample a new dome-light intensity each reset. Range brackets ±30% of
    our default 2500 so the scene never goes pitch-dark or blown-out.
    """
    import omni.usd
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    light = UsdLux.DomeLight(prim)
    if not light:
        return
    lo, hi = intensity_range
    light.GetIntensityAttr().Set(random.uniform(lo, hi))
