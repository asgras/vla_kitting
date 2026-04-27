"""Single source of truth for cube-color palette + name → RGB mapping.

Imported by `events.randomize_cube_color` (writes the chosen index to env
state), `observations.cube_color_idx` (exposes the index per step), and the
data-conversion / eval scripts (look up the human-readable color word from
the index when formatting per-episode prompts).

Keep this file dependency-free (no Isaac imports) so it can be imported from
both the Isaac Lab runtime and plain `python` data-conversion scripts.
"""
from __future__ import annotations

# Order matters: the integer index of each entry is what gets stored in
# `env.cube_color_state` and (transitively) in /obs/cube_color_idx in the
# recorded HDF5 / LeRobot dataset. Reordering this list will silently
# invalidate prior datasets — append new colors instead.
CUBE_COLOR_PALETTE: list[tuple[str, tuple[float, float, float]]] = [
    ("red", (0.85, 0.15, 0.15)),
    ("blue", (0.15, 0.35, 0.90)),
    ("yellow", (0.95, 0.80, 0.10)),
    ("orange", (0.90, 0.45, 0.10)),
    ("purple", (0.70, 0.15, 0.80)),
]


def color_name_for_idx(idx: int) -> str:
    """Reverse-lookup a color name. Returns "" if idx is out of range so
    callers can fall back to a default prompt without raising."""
    if 0 <= idx < len(CUBE_COLOR_PALETTE):
        return CUBE_COLOR_PALETTE[idx][0]
    return ""


def format_task_with_color(color_name: str | None) -> str:
    """Standard per-episode prompt string. If color_name is empty/None,
    falls back to the unspecific phrasing (older datasets)."""
    if color_name:
        return f"pick up the {color_name} cube and place it on the magenta circle"
    return "pick up the cube and place it on the magenta circle"
