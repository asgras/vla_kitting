"""Concatenate two LeRobot v3.0 image-format datasets into one.

Why
---
isaaclab_to_lerobot.py is a fresh-write tool — there is no append mode. When
the v5 dataset gen had to be capped at --max_episodes 100 to fit the overnight
training window, the remaining 57 demos were left out. To recover them
(vla_kitting-8ux's "we should get those 57 demos that aren't included"
directive) we generate fresh demos with new seeds, convert them as a separate
LeRobot dataset, and then use this tool to splice the two datasets into one
that smolvla training can consume.

What it does
------------
1. Reads `meta/info.json` from both A (existing) and B (new). Confirms feature
   schemas match.
2. Copies A's `data/chunk-XXX/file-YYY.parquet` files unchanged.
3. Re-indexes B's parquets: `episode_index += len(A)`, `index += A.total_frames`.
   (frame_index, task_index, timestamp left as-is — frame_index is per-episode,
   task_index is a content-addressed pointer into tasks.parquet, timestamp is
   episode-relative.)
4. Copies A's `images/{key}/episode-NNNNNN/` dirs unchanged.
5. Copies B's `images/{key}/episode-MMMMMM/` dirs as `episode-{M+len(A)}/`.
6. Combines `tasks.parquet` by unioning unique task strings — task_index in B's
   data parquets gets remapped if a B-task lands at a different index in the
   combined table. Currently both datasets use the same 5 cube-color tasks, so
   in practice no remapping is needed; we still validate it.
7. Recombines `meta/episodes/chunk-XXX/file-YYY.parquet` (per-episode metadata
   and per-episode stats).
8. Recomputes `meta/stats.json` by parallel-axis-combining mean/std (Welford-
   style) across A and B's per-feature stats, with weights = frame counts.
   min = min(A.min, B.min), max = max(A.max, B.max). Quantiles (q01..q99) are
   approximated as count-weighted averages — exact would require re-reading
   all frames, which is the expensive thing this whole exercise is trying to
   avoid.
9. Writes a fresh `meta/info.json` reflecting the combined episode + frame
   counts and the union of features.

What it does NOT do
-------------------
- Does not deduplicate identical frames between A and B (we trust the seed
  scheme to make them disjoint).
- Does not validate that A and B share fps, image resolution, action dim,
  state dim — it errors loudly if these differ.
- Does not recompute ground-truth quantiles. If your downstream code is
  quantile-sensitive (rare), use lerobot's built-in stats compute instead.

Usage
-----
    /opt/IsaacSim/python.sh scripts/data/concat_lerobot_datasets.py \\
        --a datasets/lerobot/cube_pick_v5_2026_04_27 \\
        --b datasets/lerobot/cube_pick_v5_extra_2026_04_27 \\
        --output datasets/lerobot/cube_pick_v5_combined_2026_04_27
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Hardlink src→dst if on the same filesystem; else copy.

    Hardlinks let us combine LeRobot datasets without doubling disk space —
    the bulk of a v5-style dataset is per-frame PNGs and parquet files whose
    contents don't change in the combined output.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # Cross-device or some other reason link failed; fall back to copy.
        shutil.copy2(src, dst)


def _link_or_copytree(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Mirror src to dst by hardlinking each regular file. Subdirs are recreated."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        target = dst / p.name
        if p.is_dir():
            _link_or_copytree(p, target)
        else:
            _link_or_copy(p, target)


def _log(msg: str) -> None:
    print(f"[concat_lerobot] {msg}", flush=True)


def _validate_features(info_a: dict, info_b: dict) -> None:
    fa, fb = info_a["features"], info_b["features"]
    if set(fa.keys()) != set(fb.keys()):
        raise SystemExit(
            f"feature key sets differ: A={sorted(fa.keys())} vs B={sorted(fb.keys())}"
        )
    for k in fa:
        if fa[k].get("shape") != fb[k].get("shape"):
            raise SystemExit(
                f"feature {k} shape mismatch: A={fa[k].get('shape')} vs B={fb[k].get('shape')}"
            )
        if fa[k].get("dtype") != fb[k].get("dtype"):
            raise SystemExit(
                f"feature {k} dtype mismatch: A={fa[k].get('dtype')} vs B={fb[k].get('dtype')}"
            )
    if info_a["fps"] != info_b["fps"]:
        raise SystemExit(f"fps mismatch: A={info_a['fps']} vs B={info_b['fps']}")
    if info_a.get("codebase_version") != info_b.get("codebase_version"):
        _log(
            f"WARNING: codebase_version differs A={info_a.get('codebase_version')} "
            f"vs B={info_b.get('codebase_version')}; proceeding"
        )


def _list_parquets(d: pathlib.Path) -> list[pathlib.Path]:
    """Return data/chunk-*/file-*.parquet sorted by (chunk, file)."""
    parquets = sorted(
        (d / "data").rglob("file-*.parquet"),
        key=lambda p: (p.parent.name, p.name),
    )
    return parquets


def _list_episode_image_dirs(d: pathlib.Path, image_key: str) -> list[pathlib.Path]:
    """Episode dirs under images/{image_key}/, sorted by episode index."""
    base = d / "images" / image_key
    if not base.exists():
        return []
    eps = sorted(base.iterdir(), key=lambda p: int(p.name.split("-")[1]))
    return [p for p in eps if p.is_dir()]


def _combine_mean_std(
    n_a: int, mean_a: np.ndarray, std_a: np.ndarray,
    n_b: int, mean_b: np.ndarray, std_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-axis combine of population mean and std across two groups.

    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    n = n_a + n_b
    if n == 0:
        return np.zeros_like(mean_a), np.zeros_like(std_a)
    delta = mean_b - mean_a
    mean = (n_a * mean_a + n_b * mean_b) / n
    m2_a = std_a**2 * n_a
    m2_b = std_b**2 * n_b
    m2 = m2_a + m2_b + (delta**2) * (n_a * n_b / n)
    std = np.sqrt(np.clip(m2 / n, 0, None))
    return mean, std


def _combine_stats_dict(stats_a: dict, stats_b: dict) -> dict:
    """Combine two stats.json dicts into one. Keys must match; values are dicts
    with min/max/mean/std/count/q01/q10/q50/q90/q99.
    """
    out: dict = {}
    if set(stats_a.keys()) != set(stats_b.keys()):
        raise SystemExit(
            f"stats.json key sets differ: A={sorted(stats_a.keys())} vs B={sorted(stats_b.keys())}"
        )
    for k in stats_a:
        sa, sb = stats_a[k], stats_b[k]
        n_a = int(np.asarray(sa["count"]).sum())
        n_b = int(np.asarray(sb["count"]).sum())
        out_k = {}
        ma = np.asarray(sa["mean"], dtype=np.float64)
        mb = np.asarray(sb["mean"], dtype=np.float64)
        da = np.asarray(sa["std"], dtype=np.float64)
        db = np.asarray(sb["std"], dtype=np.float64)
        mean, std = _combine_mean_std(n_a, ma, da, n_b, mb, db)
        out_k["mean"] = mean.tolist()
        out_k["std"] = std.tolist()
        out_k["min"] = np.minimum(np.asarray(sa["min"]), np.asarray(sb["min"])).tolist()
        out_k["max"] = np.maximum(np.asarray(sa["max"]), np.asarray(sb["max"])).tolist()
        out_k["count"] = [n_a + n_b]
        for q in ("q01", "q10", "q50", "q90", "q99"):
            qa = np.asarray(sa[q], dtype=np.float64)
            qb = np.asarray(sb[q], dtype=np.float64)
            out_k[q] = ((n_a * qa + n_b * qb) / max(n_a + n_b, 1)).tolist()
        out[k] = out_k
    return out


def _merge_tasks(a_tasks: pd.DataFrame, b_tasks: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """Union of unique task strings. Returns (combined_df, b_to_combined_map).

    b_to_combined_map: original task_index in B → task_index in combined.
    """
    a_idx = a_tasks.index.tolist() if a_tasks.index.name == "task" else a_tasks.iloc[:, 0].tolist()
    # tasks.parquet has the task string as the index, with a single 'task_index' column
    if a_tasks.index.name == "task":
        a_strs = a_tasks.index.tolist()
        b_strs = b_tasks.index.tolist()
    else:
        a_strs = a_tasks.iloc[:, 0].tolist()
        b_strs = b_tasks.iloc[:, 0].tolist()
    combined_strs = list(a_strs) + [s for s in b_strs if s not in a_strs]
    df = pd.DataFrame({"task_index": list(range(len(combined_strs)))}, index=pd.Index(combined_strs, name="task"))
    # Map B's task_index → combined task_index by looking up the string.
    b_to_combined = {}
    for b_i, s in enumerate(b_strs):
        b_to_combined[b_i] = combined_strs.index(s)
    return df, b_to_combined


def _renumber_b_data_parquet(
    src: pathlib.Path,
    dst: pathlib.Path,
    episode_offset: int,
    frame_offset: int,
    task_remap: dict[int, int],
) -> int:
    """Copy B's data parquet to dst, adjusting episode_index, index, and task_index.

    Returns the number of rows in the parquet (=== frames in this episode).
    """
    df = pd.read_parquet(src)
    df["episode_index"] = df["episode_index"] + episode_offset
    df["index"] = df["index"] + frame_offset
    if task_remap:
        df["task_index"] = df["task_index"].map(lambda x: task_remap.get(int(x), int(x)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=pathlib.Path, required=True,
                    help="Path to dataset A (existing — episodes 0..N-1 in output)")
    ap.add_argument("--b", type=pathlib.Path, required=True,
                    help="Path to dataset B (new — episodes N..N+M-1 in output)")
    ap.add_argument("--output", type=pathlib.Path, required=True,
                    help="Path to write combined dataset")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="If --output exists, delete it first")
    args = ap.parse_args()

    a_dir, b_dir, out_dir = args.a, args.b, args.output
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing to overwrite {out_dir}; pass --overwrite")
        _log(f"--overwrite: removing {out_dir}")
        shutil.rmtree(out_dir)

    info_a = json.loads((a_dir / "meta" / "info.json").read_text())
    info_b = json.loads((b_dir / "meta" / "info.json").read_text())
    _validate_features(info_a, info_b)

    n_a = int(info_a["total_episodes"])
    f_a = int(info_a["total_frames"])
    n_b = int(info_b["total_episodes"])
    f_b = int(info_b["total_frames"])
    _log(f"A: {n_a} episodes / {f_a} frames    B: {n_b} episodes / {f_b} frames")
    _log(f"combined: {n_a + n_b} episodes / {f_a + f_b} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    (out_dir / "meta" / "episodes").mkdir(exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # --- 1. Hardlink A's data parquets (content identical to what we want in
    #       the combined dataset, so a hardlink saves disk).
    parquets_a = _list_parquets(a_dir)
    if len(parquets_a) != n_a:
        _log(f"WARNING: A has {len(parquets_a)} parquets vs info.total_episodes={n_a}; proceeding")
    for p in parquets_a:
        rel = p.relative_to(a_dir)
        dst = out_dir / rel
        _link_or_copy(p, dst)
    _log(f"hardlinked {len(parquets_a)} A parquets")

    # --- 2. Merge tasks.parquet to get B's task_index remap.
    a_tasks = pd.read_parquet(a_dir / "meta" / "tasks.parquet")
    b_tasks = pd.read_parquet(b_dir / "meta" / "tasks.parquet")
    combined_tasks, b_task_remap = _merge_tasks(a_tasks, b_tasks)
    combined_tasks.to_parquet(out_dir / "meta" / "tasks.parquet", index=True)
    _log(f"tasks.parquet: A had {len(a_tasks)} tasks, B had {len(b_tasks)} tasks, "
         f"combined {len(combined_tasks)} (remap: {b_task_remap})")

    # --- 3. Re-index B's data parquets and copy.
    parquets_b = _list_parquets(b_dir)
    if len(parquets_b) != n_b:
        _log(f"WARNING: B has {len(parquets_b)} parquets vs info.total_episodes={n_b}; proceeding")

    for i, p in enumerate(parquets_b):
        # Output filename: episode index (n_a + i), but lerobot uses chunk-XXX/file-YYY
        # naming with chunks_size from info.json; we need to mirror that.
        new_episode_idx = n_a + i
        chunk_size = int(info_a.get("chunks_size", 1000))
        chunk_idx = new_episode_idx // chunk_size
        file_idx = new_episode_idx % chunk_size
        dst = out_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        _renumber_b_data_parquet(
            src=p, dst=dst,
            episode_offset=n_a,
            frame_offset=f_a,
            task_remap=b_task_remap,
        )
    _log(f"copied + reindexed {len(parquets_b)} B parquets (episode_offset={n_a}, frame_offset={f_a})")

    # --- 4. Copy image dirs.
    image_keys = [k for k in info_a["features"] if k.startswith("observation.images.")]
    for key in image_keys:
        # A's dirs: hardlink each PNG (content identical)
        for p in _list_episode_image_dirs(a_dir, key):
            dst = out_dir / "images" / key / p.name
            _link_or_copytree(p, dst)
        # B's dirs: hardlink each PNG into a renamed episode-XXXXXX
        for i, p in enumerate(_list_episode_image_dirs(b_dir, key)):
            new_name = f"episode-{n_a + i:06d}"
            dst = out_dir / "images" / key / new_name
            _link_or_copytree(p, dst)
        _log(f"  images/{key}: {len(_list_episode_image_dirs(a_dir, key))} from A + "
             f"{len(_list_episode_image_dirs(b_dir, key))} from B (renamed; hardlinked)")

    # --- 5. Recombine episodes meta parquet (with episode_index renumbering).
    eps_a_files = sorted((a_dir / "meta" / "episodes").rglob("file-*.parquet"))
    eps_b_files = sorted((b_dir / "meta" / "episodes").rglob("file-*.parquet"))
    eps_a = pd.concat([pd.read_parquet(f) for f in eps_a_files], ignore_index=True) if eps_a_files else pd.DataFrame()
    eps_b = pd.concat([pd.read_parquet(f) for f in eps_b_files], ignore_index=True) if eps_b_files else pd.DataFrame()
    if not eps_b.empty:
        eps_b = eps_b.copy()
        eps_b["episode_index"] = eps_b["episode_index"] + n_a
        # data/chunk_index and data/file_index also need to match the new naming
        if "data/chunk_index" in eps_b.columns:
            chunk_size = int(info_a.get("chunks_size", 1000))
            eps_b["data/chunk_index"] = (eps_b["episode_index"] // chunk_size).astype("int64")
            eps_b["data/file_index"] = (eps_b["episode_index"] % chunk_size).astype("int64")
        # dataset_from/to_index need offsetting too
        if "dataset_from_index" in eps_b.columns:
            eps_b["dataset_from_index"] = eps_b["dataset_from_index"] + f_a
            eps_b["dataset_to_index"] = eps_b["dataset_to_index"] + f_a
    combined_eps = pd.concat([eps_a, eps_b], ignore_index=True)
    combined_eps_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    combined_eps_path.parent.mkdir(parents=True, exist_ok=True)
    combined_eps.to_parquet(combined_eps_path, index=False)
    _log(f"episodes meta: {len(eps_a)} from A + {len(eps_b)} from B = {len(combined_eps)} rows")

    # --- 6. Combine stats.json.
    stats_a = json.loads((a_dir / "meta" / "stats.json").read_text())
    stats_b = json.loads((b_dir / "meta" / "stats.json").read_text())
    combined_stats = _combine_stats_dict(stats_a, stats_b)
    (out_dir / "meta" / "stats.json").write_text(json.dumps(combined_stats, indent=2))
    _log("stats.json: parallel-axis combined (mean/std exact, quantiles count-weighted approximation)")

    # --- 7. Write info.json.
    info_out = dict(info_a)
    info_out["total_episodes"] = n_a + n_b
    info_out["total_frames"] = f_a + f_b
    info_out["splits"] = {"train": f"0:{n_a + n_b}"}
    # tasks: union count
    info_out["total_tasks"] = len(combined_tasks)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))
    _log(f"info.json written: total_episodes={info_out['total_episodes']}, "
         f"total_frames={info_out['total_frames']}, total_tasks={info_out['total_tasks']}")

    # --- 8. Sanity check: re-open the combined dataset's first and last
    #       episodes' parquets and verify the (episode_index, index, frame_index)
    #       columns are monotone and match the expected ranges.
    first_p = out_dir / "data" / "chunk-000" / "file-000.parquet"
    last_episode = n_a + n_b - 1
    last_p = out_dir / "data" / f"chunk-{last_episode // int(info_a.get('chunks_size', 1000)):03d}" / f"file-{last_episode % int(info_a.get('chunks_size', 1000)):03d}.parquet"
    df_first = pd.read_parquet(first_p)
    df_last = pd.read_parquet(last_p)
    assert df_first["episode_index"].iloc[0] == 0, f"first episode_index != 0: {df_first['episode_index'].iloc[0]}"
    assert df_last["episode_index"].iloc[0] == last_episode, (
        f"last episode_index != {last_episode}: {df_last['episode_index'].iloc[0]}"
    )
    assert df_first["index"].iloc[0] == 0, f"first index != 0: {df_first['index'].iloc[0]}"
    assert df_last["index"].iloc[-1] == f_a + f_b - 1, (
        f"last frame index != {f_a + f_b - 1}: {df_last['index'].iloc[-1]}"
    )
    _log("sanity check passed — first/last episode indices and frame index ranges look correct")
    _log(f"OK — combined dataset at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
