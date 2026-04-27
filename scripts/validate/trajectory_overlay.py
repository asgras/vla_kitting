"""Trajectory-overlay test for visual conditioning (vla_kitting-vd0).

Loads per-step EE/cube traces from one or more --action_log_csv files
emitted by run_vla_closed_loop.py and overlays them on a single
top-down (XY) plot, color-coded by cube start position. The shape of the
overlay distinguishes:

  - "FAN" — every trajectory bends toward its corresponding cube → policy
    is using vision (or cube_pos channel) to navigate. The bottleneck is
    PRECISION (e.g. landing offset, gripper timing).
  - "COLLAPSED" — all trajectories trace approximately the same XY path
    regardless of cube → policy ignores the cube observation, runs a
    stereotyped trajectory. Bottleneck is VISUAL MODE COLLAPSE.

Inputs:
  --action_log_csv: one or more CSVs (output of run_vla_closed_loop.py
    with --action_log_csv set). Schema (per row):
      ep,step,a0,a1,a2,a3,a4,a5,a6,cube_x,cube_y,cube_z,ee_x,ee_y,ee_z

Verdict thresholds (printed):
  - For each episode, compute mean(EE_xy) - cube_xy_start. If trajectories
    diverge with cube position (variance of the (mean_ee - cube) offset
    > 0.05 m typical), call it FAN. If variance < 0.01 m, COLLAPSED.

Usage:
  python scripts/validate/trajectory_overlay.py \\
      --action_log_csv reports/runs/.../v4_action_log.csv \\
      --out_png reports/runs/.../trajectory_overlay.png

  # Or compare two checkpoints side by side:
  python scripts/validate/trajectory_overlay.py \\
      --action_log_csv ckpt_a.csv --action_log_csv ckpt_b.csv \\
      --labels run_a run_b \\
      --out_png comparison.png
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np


def _read_csv(path: pathlib.Path) -> dict[int, dict]:
    """Group rows by episode index. Returns {ep: {ee_xy: (T, 2),
    cube_xy: (T, 2), cube0: (2,)}}."""
    by_ep: dict[int, list[dict]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["ep"])
            by_ep.setdefault(ep, []).append(row)
    out: dict[int, dict] = {}
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: int(r["step"]))
        ee_xy = np.array([[float(r["ee_x"]), float(r["ee_y"])] for r in rows], dtype=np.float32)
        cube_xy = np.array([[float(r["cube_x"]), float(r["cube_y"])] for r in rows], dtype=np.float32)
        out[ep] = {
            "ee_xy": ee_xy,
            "cube_xy": cube_xy,
            "cube0": cube_xy[0],
            "n": len(rows),
        }
    return out


def _verdict(per_ep: dict[int, dict]) -> tuple[str, dict]:
    """Compute fan-vs-collapsed verdict from per-episode traces.

    Heuristic: average EE position over the first half of each episode
    (when the policy is supposed to be approaching the cube). If those
    averages cluster regardless of cube position (low variance), the
    policy is going to the same place every episode → COLLAPSED. If they
    correlate with cube position (their (mean - cube) offset is roughly
    consistent across episodes — i.e. policy approaches each cube from
    a similar relative angle), FAN.
    """
    if not per_ep:
        return ("UNKNOWN", {})
    means = []
    cubes = []
    for ep, d in sorted(per_ep.items()):
        half = d["ee_xy"].shape[0] // 2 or 1
        m = d["ee_xy"][:half].mean(axis=0)
        means.append(m)
        cubes.append(d["cube0"])
    means = np.stack(means)
    cubes = np.stack(cubes)

    # Spread of the early-trajectory mean position.
    mean_var = float(means.var(axis=0).sum())  # scalar (m^2)
    # Correlation between EE-mean and cube position (per axis).
    rho_x = float(np.corrcoef(means[:, 0], cubes[:, 0])[0, 1]) if means.shape[0] > 1 else 0.0
    rho_y = float(np.corrcoef(means[:, 1], cubes[:, 1])[0, 1]) if means.shape[0] > 1 else 0.0

    stats = {
        "mean_var_m2": mean_var,
        "corr_ee_cube_x": rho_x,
        "corr_ee_cube_y": rho_y,
        "n_episodes": int(means.shape[0]),
    }
    # Verdict: a healthy fan has mean_var > 0.005 m^2 (each axis std > ~5cm
    # given uniform cube distribution) AND positive correlation between
    # EE-xy mean and cube-xy (the policy moves toward the cube).
    if mean_var < 0.0005 or (abs(rho_x) < 0.2 and abs(rho_y) < 0.2):
        verdict = "COLLAPSED"
    elif mean_var < 0.005:
        verdict = "WEAK_FAN"
    else:
        verdict = "FAN"
    return verdict, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action_log_csv", action="append", required=True,
                    help="repeatable; one or more action-log CSVs.")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out_png", type=pathlib.Path, required=True)
    ap.add_argument("--cube_box_x", type=float, nargs=2, default=(0.40, 0.70))
    ap.add_argument("--cube_box_y", type=float, nargs=2, default=(-0.22, 0.22))
    args = ap.parse_args()

    paths = [pathlib.Path(p) for p in args.action_log_csv]
    labels = args.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        ap.error("--labels length must match number of --action_log_csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5),
                             squeeze=False, sharex=True, sharey=True)
    summaries = []
    for ax, path, label in zip(axes[0], paths, labels):
        per_ep = _read_csv(path)
        verdict, stats = _verdict(per_ep)
        summaries.append((label, verdict, stats))
        # Plot cube-box.
        ax.add_patch(plt.Rectangle(
            (args.cube_box_y[0], args.cube_box_x[0]),
            args.cube_box_y[1] - args.cube_box_y[0],
            args.cube_box_x[1] - args.cube_box_x[0],
            fill=False, edgecolor="grey", lw=1, ls="--",
        ))
        # Plot target marker (magenta circle at world (0.65, 0.20), radius 0.05).
        ax.add_patch(plt.Circle((0.20, 0.65), 0.05, color="m", alpha=0.4))
        # Color each trajectory by its episode index (proxy for cube_xy).
        cmap = plt.get_cmap("viridis")
        eps = sorted(per_ep.keys())
        for j, ep in enumerate(eps):
            d = per_ep[ep]
            color = cmap(j / max(1, len(eps) - 1))
            # Plot EE x vs y with axes swapped so world-Y is horizontal,
            # world-X is vertical (matches a top-down render).
            ax.plot(d["ee_xy"][:, 1], d["ee_xy"][:, 0], color=color, lw=0.6, alpha=0.7)
            ax.scatter([d["cube0"][1]], [d["cube0"][0]],
                       color=color, marker="s", s=18, ec="black", lw=0.4)
        ax.set_xlabel("world Y (m)"); ax.set_ylabel("world X (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{label}  →  {verdict}\n"
            f"mean_var={stats['mean_var_m2']:.4f} m²  "
            f"corr_X={stats['corr_ee_cube_x']:.2f}  corr_Y={stats['corr_ee_cube_y']:.2f}",
            fontsize=9,
        )
        # Pad axis to cube box plus some margin so the marker is visible.
        ax.set_xlim(args.cube_box_y[0] - 0.05, args.cube_box_y[1] + 0.05)
        ax.set_ylim(args.cube_box_x[0] - 0.05, args.cube_box_x[1] + 0.10)

    fig.suptitle("EE-xy trajectory overlay (cube as ■, magenta circle as place target)",
                 fontsize=11)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out_png), dpi=130)
    print(f"[traj-overlay] wrote {args.out_png}")
    print("[traj-overlay] verdicts:")
    for label, verdict, stats in summaries:
        print(f"  {label:>20}  {verdict:>10}  mean_var={stats['mean_var_m2']:.4f}m²  "
              f"corr=({stats['corr_ee_cube_x']:+.2f},{stats['corr_ee_cube_y']:+.2f})  "
              f"n={stats['n_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
