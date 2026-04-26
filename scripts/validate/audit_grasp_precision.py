"""Audit grasp-pose precision across the v3 LeRobot dataset.

For each demo, find the frame where action[6] transitions from open (+1) to
close (-1) — the grasp moment per scripted_pick_demo.py phase E. Read the EE
position at that frame and compare to:
  - the cube starting position (parsed from the scripted gen log)
  - the expected EE position at grasp = (cube_x, cube_y + GRIP_BIAS_Y=0.018,
    PICK_GRASP_Z=0.17)

Reports the offset distribution: if (ee_x, ee_y) at grasp time is consistently
within a few mm of (cube_x, cube_y + 0.018), the scripted controller is
precise. If offsets are large/spread, the controller drifts during descent
and demos teach the policy imprecise targets.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import statistics
import sys


GRIP_BIAS_Y = 0.018  # from scripted_pick_demo.py
PICK_GRASP_Z = 0.17  # from scripted_pick_demo.py


def parse_log(log_path: pathlib.Path) -> list[dict]:
    """Walk the scripted gen log; return [{cube: (x, y, z), steps: N, success: bool}]
    per attempt, in order."""
    cube_re = re.compile(r"cube at \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")
    done_re = re.compile(r"phases done in (\d+) steps, success=(True|False)")
    attempt_re = re.compile(r"attempt (\d+): successes so far")

    attempts: list[dict] = []
    cur_cube = None
    for line in log_path.open():
        m = attempt_re.search(line)
        if m:
            cur_cube = None  # reset
            continue
        m = cube_re.search(line)
        if m:
            cur_cube = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            continue
        m = done_re.search(line)
        if m and cur_cube is not None:
            attempts.append({
                "cube": cur_cube,
                "steps": int(m.group(1)),
                "success": m.group(2) == "True",
            })
            cur_cube = None
    return attempts


def find_grasp_frame(actions) -> int | None:
    """First frame where action[6] crosses from positive (open) to negative (close)."""
    import numpy as np
    g = actions[:, 6]
    # find indices where prev > 0 and current < 0
    prev = g[:-1]
    curr = g[1:]
    mask = (prev > 0) & (curr < 0)
    idx = int(np.argmax(mask)) + 1
    if not bool(mask.any()):
        return None
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_dataset", type=pathlib.Path,
                    default=pathlib.Path("/home/ubuntu/vla_kitting/datasets/lerobot/cube_pick_v1"))
    ap.add_argument("--scripted_log", type=pathlib.Path,
                    default=pathlib.Path("/home/ubuntu/vla_kitting/reports/runs/vision_grounded_30hz_2026-04-24/logs/scripted_gen.log"))
    args = ap.parse_args()

    sys.path.insert(0, "/home/ubuntu/code/lerobot/src")
    import numpy as np

    log_attempts = parse_log(args.scripted_log)
    print(f"[audit] log: {len(log_attempts)} total attempts")
    successful_with_real_steps = [a for a in log_attempts if a["success"] and a["steps"] >= 50]
    print(f"[audit] log: {len(successful_with_real_steps)} successful with steps >= 50 (matches LeRobot filter)")

    # Read parquet files directly — bulk-load only the columns we need.
    import pyarrow.parquet as pq
    data_dir = args.lerobot_dataset / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("file-*.parquet"))
    print(f"[audit] found {len(parquet_files)} parquet shards")

    # Load only action, observation.ee_pose, episode_index columns for speed.
    cols = ["action", "observation.ee_pose", "episode_index", "frame_index"]
    tables = []
    for pf in parquet_files:
        tables.append(pq.read_table(pf, columns=cols))
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    print(f"[audit] loaded {table.num_rows} frames into memory")

    # Convert to flat numpy arrays.
    def col_to_np(col):
        # action and ee_pose are list[float32]; need to stack
        try:
            return np.stack(col.to_pylist()).astype(np.float32)
        except Exception:
            return np.asarray(col.to_pylist())
    actions = col_to_np(table.column("action"))
    ee_poses = col_to_np(table.column("observation.ee_pose"))
    episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    print(f"[audit] actions shape={actions.shape}, ee_poses shape={ee_poses.shape}")

    # Build per-episode frame ranges.
    ep_bounds: dict[int, tuple[int, int]] = {}
    cur_ep = -1
    cur_start = 0
    for i, e in enumerate(episode_index):
        e = int(e)
        if e != cur_ep:
            if cur_ep >= 0:
                ep_bounds[cur_ep] = (cur_start, i)
            cur_ep = e
            cur_start = i
    ep_bounds[cur_ep] = (cur_start, len(episode_index))
    print(f"[audit] indexed {len(ep_bounds)} episodes")

    # Pull each episode's frames, find grasp moment, compute offsets vs cube_start.
    offsets_x: list[float] = []
    offsets_y: list[float] = []
    offsets_z: list[float] = []
    bias_y_observed: list[float] = []
    grasp_frames: list[int] = []

    n = min(len(ep_bounds), len(successful_with_real_steps))
    for ep_idx in range(n):
        from_, to_ = ep_bounds[ep_idx]
        ep_actions = actions[from_:to_]
        gframe_local = find_grasp_frame(ep_actions)
        if gframe_local is None:
            continue
        ee_pose = ee_poses[from_ + gframe_local]
        ee_x, ee_y, ee_z = float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2])

        cube = successful_with_real_steps[ep_idx]["cube"]
        cube_x, cube_y, cube_z = cube

        offsets_x.append(ee_x - cube_x)
        offsets_y.append(ee_y - cube_y)
        offsets_z.append(ee_z - PICK_GRASP_Z)
        bias_y_observed.append(ee_y - cube_y)
        grasp_frames.append(gframe_local)

    print(f"\n[audit] {len(offsets_x)} grasp moments analyzed")
    print(f"[audit] grasp frame index: mean={statistics.mean(grasp_frames):.1f}, "
          f"stdev={statistics.stdev(grasp_frames):.1f}, "
          f"range=[{min(grasp_frames)},{max(grasp_frames)}]")

    print(f"\n[audit] Offset (ee - cube) at grasp moment:")
    for label, vals in [("x", offsets_x), ("y", offsets_y), ("z", offsets_z)]:
        m = statistics.mean(vals)
        sd = statistics.stdev(vals)
        lo, hi = min(vals), max(vals)
        print(f"  Δ{label}: mean={m * 1000:+7.2f} mm, stdev={sd * 1000:6.2f} mm, "
              f"range=[{lo * 1000:+7.2f}, {hi * 1000:+7.2f}] mm")
    print(f"  (Δy mean expected ~ +18mm = GRIP_BIAS_Y)")

    print(f"\n[audit] Δy distribution (expected centered on +18mm):")
    n_lo = sum(1 for v in offsets_y if v < 0.005)  # below 5mm
    n_mid = sum(1 for v in offsets_y if 0.005 <= v < 0.030)
    n_hi = sum(1 for v in offsets_y if v >= 0.030)
    print(f"  Δy <  5mm:     {n_lo:4d} demos  (gripper slightly inside cube y)")
    print(f"  Δy 5-30mm:     {n_mid:4d} demos  (within reasonable bias band)")
    print(f"  Δy ≥ 30mm:     {n_hi:4d} demos  (gripper way past cube y)")

    print(f"\n[audit] |Δx| distribution (expected near 0):")
    for thresh in [0.005, 0.010, 0.020, 0.040]:
        n_off = sum(1 for v in offsets_x if abs(v) >= thresh)
        print(f"  |Δx| ≥ {thresh*1000:>3.0f}mm: {n_off:4d} / {len(offsets_x)} demos "
              f"({100*n_off/len(offsets_x):4.1f}%)")

    # Per-demo print for first 10 outliers (largest |Δx| or |Δy - 0.018|)
    by_xerr = sorted(range(len(offsets_x)), key=lambda i: -abs(offsets_x[i]))
    by_yerr = sorted(range(len(offsets_y)), key=lambda i: -abs(offsets_y[i] - 0.018))
    print(f"\n[audit] Top-5 X-axis outliers:")
    for i in by_xerr[:5]:
        cube = successful_with_real_steps[i]["cube"]
        print(f"  demo {i}: cube=({cube[0]:.3f},{cube[1]:.3f}) "
              f"Δx={offsets_x[i]*1000:+.1f}mm Δy={offsets_y[i]*1000:+.1f}mm")
    print(f"\n[audit] Top-5 Y-axis bias outliers (|Δy - 18mm|):")
    for i in by_yerr[:5]:
        cube = successful_with_real_steps[i]["cube"]
        print(f"  demo {i}: cube=({cube[0]:.3f},{cube[1]:.3f}) "
              f"Δx={offsets_x[i]*1000:+.1f}mm Δy={offsets_y[i]*1000:+.1f}mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
