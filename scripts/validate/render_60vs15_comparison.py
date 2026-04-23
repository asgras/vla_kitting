"""Render a mimic demo two ways to visualize the 60 Hz → 15 Hz downsampling:
  1. 60 Hz "smooth" version (every 2nd frame, played at 30 fps for display — real time).
  2. 15 Hz "training" version (every 4th frame, played at 15 fps — real time).
  3. Side-by-side composite (left pane: 60 Hz, right pane: 15 Hz with held frames).

No Isaac Sim required — just replays captured pixels.

Usage:
    .venv/bin/python scripts/validate/render_60vs15_comparison.py --demo 0
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import imageio
import numpy as np
from PIL import Image


def stitch_wrist_third(wrist: np.ndarray, third: np.ndarray) -> np.ndarray:
    """Return (T, 256, 512, 3) array: wrist upscaled to 256 on left, third on right."""
    T = wrist.shape[0]
    out = np.zeros((T, 256, 512, 3), dtype=np.uint8)
    for i in range(T):
        up = np.array(Image.fromarray(wrist[i]).resize((256, 256), Image.NEAREST))
        out[i, :, :256] = up
        out[i, :, 256:] = third[i]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/vla_kitting/datasets/mimic/cube_mimic_all.hdf5"),
    )
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/vla_kitting/reports"),
    )
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        d = f["data"][f"demo_{args.demo}"]
        wrist = d["obs"]["wrist_cam"][...]
        third = d["obs"]["third_person_cam"][...]

    T = third.shape[0]
    print(f"loaded {T} frames from demo_{args.demo}")
    stitched = stitch_wrist_third(wrist, third)

    tag = f"demo_{args.demo:03d}"

    # 60 Hz reference — every 2nd frame at 30 fps = real-time, smooth.
    # (Saves file size vs every-frame-at-60fps while still looking fluid.)
    out60 = args.out_dir / f"{tag}_60hz_smooth.gif"
    frames_60 = stitched[::2]
    imageio.mimsave(str(out60), list(frames_60), fps=30, loop=0)
    print(f"wrote {out60} ({out60.stat().st_size / 1e6:.1f} MB, {frames_60.shape[0]} frames)")

    # 15 Hz training view — every 4th frame at 15 fps = real-time, what the
    # model actually sees during training.
    out15 = args.out_dir / f"{tag}_15hz_training.gif"
    frames_15 = stitched[::4]
    imageio.mimsave(str(out15), list(frames_15), fps=15, loop=0)
    print(f"wrote {out15} ({out15.stat().st_size / 1e6:.1f} MB, {frames_15.shape[0]} frames)")

    # Side-by-side composite: left pane advances smoothly at 30 fps, right
    # pane also updates at 30 fps but each 15 Hz frame is held for 2 display
    # ticks, so the "choppiness" is directly visible.
    right_pane = np.repeat(frames_15, 2, axis=0)[: frames_60.shape[0]]
    composite = np.concatenate([frames_60, right_pane], axis=2)
    out_sbs = args.out_dir / f"{tag}_60vs15_sidebyside.gif"
    imageio.mimsave(str(out_sbs), list(composite), fps=30, loop=0)
    print(f"wrote {out_sbs} ({out_sbs.stat().st_size / 1e6:.1f} MB, {composite.shape[0]} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
