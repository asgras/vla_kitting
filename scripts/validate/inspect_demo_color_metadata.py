"""Audit per-episode color metadata in a scripted/Mimic HDF5.

Checks (in order of precedence — same as isaaclab_to_lerobot._resolve_episode_task):
  1. demo.attrs["task"] (preferred)
  2. demo.attrs["cube_color"]
  3. obs/cube_color_idx[0,0]

Reports the per-episode resolved prompt + the count distribution. Use this
before running isaaclab_to_lerobot to confirm the colors are balanced and
that the metadata stamping pipeline didn't silently drop anything.

Usage:
    /opt/IsaacSim/python.sh scripts/validate/inspect_demo_color_metadata.py \\
        --hdf5 /tmp/yaw_30/cube_scripted_yaw30.hdf5
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np


_PALETTE_NAMES = ["red", "blue", "yellow", "orange", "purple"]
_DEFAULT = "pick up the cube and place it on the magenta circle"


def _format(color: str | None) -> str:
    if color:
        return f"pick up the {color} cube and place it on the magenta circle"
    return _DEFAULT


def _resolve(d: h5py.Group) -> tuple[str, str]:
    """Return (source, resolved_task)."""
    attrs = d.attrs
    if "task" in attrs:
        v = attrs["task"]
        return ("attrs[task]", v.decode() if isinstance(v, bytes) else str(v))
    if "cube_color" in attrs:
        v = attrs["cube_color"]
        return ("attrs[cube_color]", _format(v.decode() if isinstance(v, bytes) else str(v)))
    obs = d.get("obs")
    if obs is not None and "cube_color_idx" in obs:
        idx_arr = obs["cube_color_idx"][...]
        if idx_arr.size > 0:
            idx = int(np.asarray(idx_arr).flat[0])
            if 0 <= idx < len(_PALETTE_NAMES):
                return ("obs[cube_color_idx]", _format(_PALETTE_NAMES[idx]))
    return ("default", _DEFAULT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=pathlib.Path, required=True)
    args = ap.parse_args()

    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    real_demos = 0
    placeholder_demos = 0
    with h5py.File(str(args.hdf5), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            d = f["data"][k]
            ns = int(d.attrs.get("num_samples", 0))
            if ns <= 0:
                placeholder_demos += 1
                continue
            real_demos += 1
            src, task = _resolve(d)
            counts[task] = counts.get(task, 0) + 1
            sources[src] = sources.get(src, 0) + 1
            extra = []
            if "cube_color" in d.attrs:
                extra.append(f"color={d.attrs['cube_color']!r}")
            if "obs" in d and "cube_color_idx" in d["obs"]:
                extra.append(f"obs_idx={int(d['obs']['cube_color_idx'][0, 0])}")
            print(f"  {k:>10}  ns={ns:>4}  src={src:<22}  {' '.join(extra)}")

    print(f"\n[inspect] total: {real_demos} real demos, {placeholder_demos} placeholder stubs")
    print("[inspect] resolution source distribution:")
    for s, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {s}")
    print("[inspect] per-episode prompt distribution:")
    for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>4}  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
