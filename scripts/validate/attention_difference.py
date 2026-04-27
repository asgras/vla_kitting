"""Attention-DIFFERENCE cube-localization probe (vla_kitting-uxt).

Consumes the per-frame attention npz files written by attention_overlay.py
(schema: third_heat, wrist_heat, third_np, wrist_np, action) for a set of
N >= 9 frames captured at known cube XY positions. Produces:

  1. residual_third[i] = third_heat[i] - mean_i(third_heat[i])
  2. residual_wrist[i] same for wrist heat
  3. A summary plot: per-cube-position residual third-person heatmaps
     overlaid on the source frame, with the cube's projected XY pixel
     marked.
  4. Verdict: do residual peaks track cube XY?
     - For each frame, find argmax of residual_third (the pixel where
       attention is *most above the cross-frame baseline*).
     - Compare to the cube's projected pixel (computed from world XY via
       a hard-coded approximate third-person camera homography — the
       camera pose at envs/yaskawa_pick_cube_cfg.py:127 is fixed). The
       camera looks at (0.60, 0.10, 0.02) from (1.15, 0.10, 0.50) on a
       256x256 image; we use a coarse linear approximation that suffices
       for 'is the peak in the right ROI?' triage.
     - VERDICT: 'tracks' if median(|argmax - projected| in pixel) < 40 px
       (~16% of frame width), else 'invariant'.

This is invalidation-resistant by construction: the prior peak/uniform
diagnostic was contaminated by SmolVLM's register-token positional bias
(see reports/2026-04-26_attention_diagnostic_invalidated.md). Subtracting
the per-key MEAN across the 9-frame set removes any positional component
that's identical across frames — only content-driven attention survives.

Usage:
  # Step 1 (sim-bound, separate): capture 9 frames at cube XY grid using
  # an existing scene-render script (e.g. save_camera_samples.py with
  # --cube_xy override) into reports/runs/attn_diff/frame_NN_third.png +
  # frame_NN_wrist.png with corresponding cube_xys.json.
  #
  # Step 2 (sim-bound): run attention_overlay.py for each frame, writing
  # frame_NN_data.npz into the same dir.
  #
  # Step 3 (this script): aggregate.

  python scripts/validate/attention_difference.py \\
      --npz_glob 'reports/runs/attn_diff/frame_*_data.npz' \\
      --cube_xys reports/runs/attn_diff/cube_xys.json \\
      --out_dir reports/runs/attn_diff/

cube_xys.json format:
  {"frames": [
      {"npz": "frame_00_data.npz", "cube_xy": [0.45, -0.15]},
      {"npz": "frame_01_data.npz", "cube_xy": [0.45,  0.00]},
      ...
  ]}
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np


# Empirical third-person camera projection (envs/yaskawa_pick_cube_cfg.py:116-131:
# camera at world (1.15, 0.10, 0.50) looking at (0.60, 0.10, 0.02) on a
# 256×256 image). Two anchor points were measured from rendered frames
# under reports/runs/attn_diff_2026-04-27/:
#   world (0.45, -0.15)  →  img (X=94,  Y=36)   [frame_00 cube position]
#   world (0.65, +0.20)  →  img (X=170, Y=147)  [magenta target marker]
# Linear fit through both points (close enough for ROI-level triage on a
# 256-pixel image; note world X increases AWAY from camera → image Y
# decreases, so the relation is +slope despite intuition):
#   img_X = 217 * world_Y + 126.55  (Y is the side-to-side axis)
#   img_Y = 555 * world_X - 213.75  (X is the depth axis)
# The prior version had the world-X → image-Y axis flipped (cube at far X
# was placed near image bottom instead of near image top), which falsely
# inflated the residual-argmax error in the attention-difference probe.
_THIRD_W = 256
_THIRD_H = 256


def world_to_third_px(cube_xy: tuple[float, float]) -> tuple[int, int]:
    """Approximate (cube_x_world, cube_y_world) → (px_x, px_y) on the
    third-person 256×256 image. See comment above for the empirical fit."""
    cx, cy = cube_xy
    px_x = int(round(217.0 * cy + 126.55))
    px_y = int(round(555.0 * cx - 213.75))
    px_x = max(0, min(_THIRD_W - 1, px_x))
    px_y = max(0, min(_THIRD_H - 1, px_y))
    return px_x, px_y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cube_xys",
        type=pathlib.Path,
        required=True,
        help="JSON manifest mapping each npz to its cube_xy.",
    )
    ap.add_argument(
        "--npz_glob",
        type=str,
        default=None,
        help="Optional override of npz files to consume; if unset, the "
             "manifest's 'frames[*].npz' paths are used (relative to the "
             "manifest's parent dir).",
    )
    ap.add_argument(
        "--out_dir",
        type=pathlib.Path,
        required=True,
    )
    ap.add_argument(
        "--track_threshold_px",
        type=float,
        default=40.0,
        help="Median |argmax - projected_px| below this → VERDICT='tracks'.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.cube_xys.read_text())
    frames = manifest["frames"]

    # Resolve npz paths.
    base = args.cube_xys.parent
    if args.npz_glob:
        npz_paths = sorted(pathlib.Path(p) for p in glob.glob(args.npz_glob))
        # Match by filename order to manifest.
        if len(npz_paths) != len(frames):
            print(
                f"[attn-diff] WARN: glob found {len(npz_paths)} npz files "
                f"but manifest has {len(frames)} frames; mismatched"
            )
    else:
        npz_paths = [base / f["npz"] for f in frames]

    third_heats = []
    wrist_heats = []
    third_imgs = []
    cube_xys = []
    for f, p in zip(frames, npz_paths):
        if not p.exists():
            print(f"[attn-diff] missing {p}; skipping")
            continue
        d = np.load(str(p))
        third_heats.append(d["third_heat"].astype(np.float32))
        wrist_heats.append(d["wrist_heat"].astype(np.float32))
        third_imgs.append(d["third_np"])
        cube_xys.append(tuple(f["cube_xy"]))

    n = len(third_heats)
    if n < 4:
        print(f"[attn-diff] need at least 4 frames; got {n}")
        return 1
    print(f"[attn-diff] aggregating {n} frames")

    third_stack = np.stack(third_heats, axis=0)  # (N, H, W)
    wrist_stack = np.stack(wrist_heats, axis=0)
    third_mean = third_stack.mean(axis=0)
    wrist_mean = wrist_stack.mean(axis=0)
    third_resid = third_stack - third_mean[None]
    wrist_resid = wrist_stack - wrist_mean[None]

    # Per-frame argmax of the residual.
    abs_errs = []
    rows = []
    for i, (xy, frame_resid) in enumerate(zip(cube_xys, third_resid)):
        flat = frame_resid.argmax()
        py, px = np.unravel_index(flat, frame_resid.shape)
        gt_px, gt_py = world_to_third_px(xy)
        err = float(np.hypot(px - gt_px, py - gt_py))
        abs_errs.append(err)
        rows.append((i, xy, (int(px), int(py)), (gt_px, gt_py), err))
        print(
            f"  frame {i:>2}: cube_xy={xy} resid_argmax=({px},{py}) "
            f"projected=({gt_px},{gt_py})  err={err:.1f}px"
        )

    median_err = float(np.median(abs_errs))
    verdict = "TRACKS" if median_err < args.track_threshold_px else "INVARIANT"
    print(f"[attn-diff] median residual-argmax err = {median_err:.1f}px → VERDICT: {verdict}")
    if verdict == "TRACKS":
        print("[attn-diff] vision is being USED — proceed with retraining "
              "(register-token positional bias has been removed by the "
              "across-frame mean subtraction).")
    else:
        print("[attn-diff] vision is NOT being used as a localization "
              "signal — peaks don't track cube position. Retraining is "
              "unlikely to help; consider auxiliary cube-localization "
              "loss or vision-tower fine-tune (recovery plan §5).")

    # Plot grid: third image with residual overlay + projected/argmax markers.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = min(n, 3)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
        if rows_n * cols == 1:
            axes = np.array([[axes]])
        elif rows_n == 1 or cols == 1:
            axes = np.atleast_2d(axes)
        for i in range(rows_n * cols):
            ax = axes[i // cols, i % cols]
            ax.set_xticks([]); ax.set_yticks([])
            if i >= n:
                ax.axis("off")
                continue
            ax.imshow(third_imgs[i])
            r = third_resid[i]
            # Symmetric colormap centered at zero.
            mx = float(np.max(np.abs(r))) + 1e-6
            ax.imshow(r, cmap="RdBu_r", alpha=0.45, vmin=-mx, vmax=mx)
            _, xy, (px, py), (gt_px, gt_py), err = rows[i]
            ax.plot(gt_px, gt_py, marker="o", mfc="none", mec="lime", ms=14, mew=2)
            ax.plot(px, py, marker="x", mec="black", ms=12, mew=2)
            ax.set_title(f"cube=({xy[0]:.2f},{xy[1]:.2f})  err={err:.0f}px", fontsize=9)
        fig.suptitle(f"Attention residual (per-frame heat − mean across N={n}); "
                     f"○ = projected cube  × = argmax  median err {median_err:.0f}px → {verdict}",
                     fontsize=10)
        fig.tight_layout()
        out = args.out_dir / "attn_difference_grid.png"
        fig.savefig(str(out), dpi=120)
        print(f"[attn-diff] wrote {out}")
    except Exception as e:
        print(f"[attn-diff] plotting failed: {e}")

    # Dump aggregate residual data for downstream consumers.
    np.savez_compressed(
        str(args.out_dir / "attn_difference_data.npz"),
        third_resid=third_resid,
        wrist_resid=wrist_resid,
        third_mean=third_mean,
        wrist_mean=wrist_mean,
        cube_xys=np.asarray(cube_xys),
        abs_errs=np.asarray(abs_errs),
        median_err=median_err,
        verdict=verdict,
    )
    return 0 if verdict == "TRACKS" else 2  # nonzero exit signals "investigate"


if __name__ == "__main__":
    raise SystemExit(main())
