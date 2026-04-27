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

from .cube_palette import CUBE_COLOR_PALETTE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Backward-compatibility alias for the old RGB-only palette. New code should
# import CUBE_COLOR_PALETTE (with names) from .cube_palette.
_CUBE_COLOR_PALETTE: list[tuple[float, float, float]] = [
    rgb for _, rgb in CUBE_COLOR_PALETTE
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
    palette: list[tuple[str, tuple[float, float, float]]] | None = None,
) -> None:
    """Pick a color from the palette at each reset and apply it to the cube's
    PreviewSurface material. Records the chosen palette index per env onto
    `env.cube_color_state[env_idx] = (color_name, palette_idx)` so the
    `cube_color_idx` observation, the LeRobot conversion (per-episode prompt
    string) and the closed-loop eval can all read the same source of truth.

    Each env in `env_ids` picks an INDEPENDENT color so multi-env data
    collection produces a distribution across the palette (single-env
    pipelines are unaffected — they just see one color per reset).

    Isaac Lab stores `asset.cfg.prim_path` with `{ENV_REGEX_NS}` already
    expanded to the regex form `/World/envs/env_.*/Cube`. We only run
    num_envs=1 for scripted/Mimic generation, so resolve per-env paths by
    iterating env_ids and rebuilding `/World/envs/env_<i>/Cube` directly.
    """
    named_palette: list[tuple[str, tuple[float, float, float]]] = (
        palette if palette is not None else CUBE_COLOR_PALETTE
    )
    asset = env.scene[asset_cfg.name]
    # Strip the regex stem to recover the suffix (e.g. "Cube" from
    # "/World/envs/env_.*/Cube") so this function also works if the cube is
    # ever parented under a sub-scope.
    suffix = asset.cfg.prim_path.split("/World/envs/env_.*")[-1]
    if not hasattr(env, "cube_color_state"):
        env.cube_color_state = {}
    for idx in env_ids.tolist():
        chosen_idx = random.randrange(len(named_palette))
        name, rgb = named_palette[chosen_idx]
        env.cube_color_state[idx] = (name, chosen_idx)
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
