"""Render one episode from a demo HDF5 as an MP4 by stitching the recorded
camera frames. No Isaac Sim required — just replays already-captured pixels.

Usage (run with the repo .venv, which has imageio + imageio-ffmpeg):
    .venv/bin/python scripts/data/render_demo.py \\
        --input datasets/mimic/cube_mimic_smoke.hdf5 \\
        --demo 0 \\
        --out reports/demo_0.mp4

Defaults: demo 0, third_person_cam, 30 fps. Pass --cam wrist_cam for the
gripper's-eye view or --side_by_side to stitch both cameras horizontally.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np


def _log(msg: str) -> None:
    print(f"[render_demo] {msg}", flush=True)


def render(
    input_path: pathlib.Path,
    demo_idx: int,
    out_path: pathlib.Path,
    cam: str,
    side_by_side: bool,
    fps: int,
    stride: int = 1,
) -> int:
    with h5py.File(input_path, "r") as f:
        demos = sorted(
            f["data"].keys(),
            key=lambda x: int(x.split("_")[1]) if x.startswith("demo_") else -1,
        )
        if demo_idx >= len(demos):
            _log(f"demo index {demo_idx} out of range ({len(demos)} demos)")
            return 1
        demo = f["data"][demos[demo_idx]]
        obs = demo["obs"]

        if side_by_side:
            if "wrist_cam" not in obs or "third_person_cam" not in obs:
                _log("side_by_side requested but one of wrist_cam/third_person_cam missing")
                return 1
            wrist = obs["wrist_cam"][...]       # (T, 128, 128, 3)
            third = obs["third_person_cam"][...]  # (T, 256, 256, 3)
            # Upscale wrist to match third's height (256) for a clean hstack.
            from PIL import Image
            T = wrist.shape[0]
            frames = np.zeros((T, 256, 256 + 256, 3), dtype=np.uint8)
            for t in range(T):
                up = np.array(
                    Image.fromarray(wrist[t]).resize((256, 256), Image.NEAREST),
                    dtype=np.uint8,
                )
                frames[t, :, :256] = up
                frames[t, :, 256:] = third[t]
        else:
            if cam not in obs:
                _log(f"camera '{cam}' not in obs (have {list(obs.keys())})")
                return 1
            frames = obs[cam][...]

    if stride > 1:
        frames = frames[::stride]
        _log(f"subsampled every {stride}th frame ({frames.shape[0]} of {T})")
    T, H, W, C = frames.shape

    # GIF branch: imageio's pillow writer handles these in 2 lines.
    if out_path.suffix.lower() == ".gif":
        import imageio
        imageio.mimsave(
            str(out_path), list(frames), fps=fps, loop=0,
        )
        _log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return 0

    # Pad width/height up to the nearest multiple of 16 so the H.264 encoder
    # doesn't emit a non-standard slice size — old/embedded decoders reject
    # anything that isn't a macroblock-aligned frame (our 512x256 was fine but
    # 384x256 side-by-side with just one cam is not, for example).
    pad_w = (16 - W % 16) % 16
    pad_h = (16 - H % 16) % 16
    if pad_w or pad_h:
        _log(f"padding frames {W}x{H} -> {W + pad_w}x{H + pad_h} for codec compat")
        frames = np.pad(
            frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )
        H, W = frames.shape[1], frames.shape[2]
    _log(f"{demos[demo_idx]}: {T} frames at {W}x{H} -> {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the ffmpeg binary bundled with imageio-ffmpeg directly — gives us
    # explicit control over the H.264 profile + faststart flag.
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",   # max compatibility (no B-frames, no CABAC)
        "-level:v", "3.1",          # widely-supported level
        "-preset", "veryfast",
        "-crf", "22",
        "-movflags", "+faststart",  # moov atom at the front for preview tools
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    if rc != 0:
        _log(f"ffmpeg exited with code {rc}")
        return rc
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cam", type=str, default="third_person_cam",
                    choices=["third_person_cam", "wrist_cam"])
    ap.add_argument("--side_by_side", action="store_true", default=False,
                    help="Stitch wrist (upscaled) + third-person horizontally.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1,
                    help="Keep 1 of every N frames (GIFs get big fast; try 4).")
    args = ap.parse_args()
    return render(args.input, args.demo, args.out, args.cam, args.side_by_side,
                  args.fps, args.stride)


if __name__ == "__main__":
    raise SystemExit(main())
