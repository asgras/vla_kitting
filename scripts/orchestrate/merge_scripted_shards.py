"""Merge K shard HDF5s produced by parallel_scripted_demo_gen.sh into one
master HDF5 with sequentially numbered demo groups.

Why
---
scripts/orchestrate/parallel_scripted_demo_gen.sh fans 750 demos across N
Isaac Sim processes (one shard each). Each shard writes its own HDF5
(`shard_<i>.hdf5`) under `datasets/teleop/parallel_<run_id>/`. To plug into
the existing isaaclab_to_lerobot.py pipeline we need a single combined HDF5
that looks just like a serial run's output: top-level "data" group with
sequential demo_<K> children, the same `env_args` attr, and per-demo attrs
preserved (cube_color, cube_color_idx, num_samples, success, task).

What this script does
---------------------
1. Walks shards in numerical order (shard_0, shard_1, ...).
2. For each demo group, copies it under a new sequential key (demo_0, demo_1,
   ...) into the merged file, preserving group attrs and child datasets/groups
   verbatim (no resampling, no decoding).
3. Replicates `data.attrs["env_args"]` from shard 0 (they're identical across
   shards by construction — same task, same env cfg).
4. Sets `data.attrs["total"]` to the sum of all shard `data.attrs["total"]`.
5. Sanity-checks: real-demo count (num_samples>0 & success=True) of merged
   == sum of per-shard real-demo counts.

What this script does NOT do
----------------------------
- Filter out placeholder (num_samples=0) groups. Why: isaaclab_to_lerobot.py
  already filters those (see scripts/data/isaaclab_to_lerobot.py:258-263), and
  preserving them keeps the merged file structurally identical to a serial-run
  HDF5 in case downstream tools rely on that.
- Re-stamp per-episode color attrs. They're already stamped by
  scripted_pick_demo.py post-export; this script only renames keys.

Usage
-----
    /opt/IsaacSim/python.sh scripts/orchestrate/merge_scripted_shards.py \\
        --shard_dir datasets/teleop/parallel_<run_id> \\
        --output    datasets/teleop/parallel_<run_id>/merged.hdf5

    # Then convert as usual:
    /opt/IsaacSim/python.sh scripts/data/isaaclab_to_lerobot.py \\
        --input  datasets/teleop/parallel_<run_id>/merged.hdf5 \\
        --output datasets/lerobot/cube_pick_v5_<run_id> \\
        --repo_id vla_kitting/cube_pick_v5 \\
        --task   "pick up the cube and place it on the magenta circle" \\
        --fps 15 --stride 1 --drop_cube_pos
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import h5py


SHARD_RE = re.compile(r"^shard_(\d+)\.hdf5$")


def _log(msg: str) -> None:
    print(f"[merge_scripted_shards] {msg}", flush=True)


def _enumerate_shards(shard_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return shard files in numerical order. Errors out if there are gaps."""
    items: list[tuple[int, pathlib.Path]] = []
    for p in shard_dir.iterdir():
        m = SHARD_RE.match(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    if not items:
        raise SystemExit(f"no shard_*.hdf5 files found under {shard_dir}")
    indices = [i for i, _ in items]
    expected = list(range(min(indices), max(indices) + 1))
    if indices != expected:
        raise SystemExit(
            f"shard indices have gaps: found {indices}, expected {expected}. "
            "Either complete the missing shards or pass --allow_gaps to skip "
            "(not yet implemented — bail loudly to avoid silent dataset loss)."
        )
    return [p for _, p in items]


def _count_real_demos(data_grp: h5py.Group) -> int:
    """Real demos = success=True AND num_samples > 0. Mirrors the filter
    isaaclab_to_lerobot.py applies (placeholder entries skipped)."""
    n = 0
    for k in data_grp.keys():
        d = data_grp[k]
        if int(d.attrs.get("num_samples", 0)) <= 0:
            continue
        if not bool(d.attrs.get("success", False)):
            continue
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=pathlib.Path, required=True,
                    help="Directory containing shard_<i>.hdf5 files.")
    ap.add_argument("--output", type=pathlib.Path, required=True,
                    help="Path to write the merged HDF5.")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="If --output exists, delete it first.")
    ap.add_argument("--delete_after_copy", action="store_true", default=False,
                    help="Delete each shard_<i>.hdf5 immediately after its "
                         "demos have been copied into the merged file. Cuts "
                         "peak disk usage roughly in half during merge — "
                         "useful for v5 production where 4 shards × 13 GB + "
                         "merged 52 GB ≈ 100 GB peak does not fit. The "
                         "sanity-check pass at the end still runs against the "
                         "merged file, so per-shard verifications are kept "
                         "intact via the in-memory counts collected up front.")
    args = ap.parse_args()

    shard_dir: pathlib.Path = args.shard_dir
    out_path: pathlib.Path = args.output

    if not shard_dir.is_dir():
        raise SystemExit(f"--shard_dir does not exist: {shard_dir}")

    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite existing {out_path}; pass --overwrite "
                "or delete it first."
            )
        _log(f"--overwrite: removing {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shards = _enumerate_shards(shard_dir)
    _log(f"found {len(shards)} shards under {shard_dir}")

    # First pass — gather per-shard counts (real demos + total groups + total
    # samples) so we can sanity-check the merged file at the end.
    per_shard_real: list[int] = []
    per_shard_groups: list[int] = []
    per_shard_total: list[int] = []
    env_args_payload: str | None = None
    for sp in shards:
        with h5py.File(sp, "r") as f:
            if "data" not in f:
                raise SystemExit(f"shard {sp} missing /data group")
            data = f["data"]
            per_shard_real.append(_count_real_demos(data))
            per_shard_groups.append(len(list(data.keys())))
            per_shard_total.append(int(data.attrs.get("total", 0)))
            if env_args_payload is None and "env_args" in data.attrs:
                v = data.attrs["env_args"]
                env_args_payload = v.decode() if isinstance(v, bytes) else str(v)

    expected_real = sum(per_shard_real)
    _log("per-shard summary:")
    for i, sp in enumerate(shards):
        _log(f"  {sp.name}: groups={per_shard_groups[i]:>4}  real_demos="
             f"{per_shard_real[i]:>4}  total_samples={per_shard_total[i]}")
    _log(f"expected merged real-demo count: {expected_real}")

    if expected_real == 0:
        raise SystemExit(
            "no real demos across any shard — refusing to write an empty "
            "merged file. Did the shard runs all crash? Inspect the per-shard "
            "logs."
        )

    # Second pass — copy demo groups into output, renumbering sequentially.
    next_key_idx = 0
    total_samples_combined = 0
    with h5py.File(out_path, "w") as out:
        out_data = out.create_group("data")
        if env_args_payload is not None:
            out_data.attrs["env_args"] = env_args_payload

        for sp in shards:
            with h5py.File(sp, "r") as f:
                src_data = f["data"]
                src_keys = sorted(
                    src_data.keys(), key=lambda k: int(k.split("_")[1])
                )
                copied = 0
                for k in src_keys:
                    new_key = f"demo_{next_key_idx}"
                    # h5py.copy preserves attrs and child datasets verbatim.
                    src_data.copy(k, out_data, name=new_key)
                    next_key_idx += 1
                    copied += 1
                    ns = int(out_data[new_key].attrs.get("num_samples", 0))
                    if ns > 0:
                        total_samples_combined += ns
                _log(f"copied {copied} groups from {sp.name}")

            # Delete the source shard now that its demos are in the merged file.
            # Critical for v5 production disk math — see --delete_after_copy
            # help text. Done OUTSIDE the with-block so the file handle is
            # released first.
            if args.delete_after_copy:
                sp.unlink()
                _log(f"  deleted source {sp.name} (--delete_after_copy)")

        out_data.attrs["total"] = total_samples_combined

    # Sanity check — re-open merged and re-count real demos.
    with h5py.File(out_path, "r") as f:
        merged_real = _count_real_demos(f["data"])
        merged_groups = len(list(f["data"].keys()))
        merged_total = int(f["data"].attrs.get("total", 0))

    _log(f"merged file: {out_path}")
    _log(f"  groups          = {merged_groups} "
         f"(sum of shards: {sum(per_shard_groups)})")
    _log(f"  real demos      = {merged_real} (expected {expected_real})")
    _log(f"  total samples   = {merged_total}")

    ok = (merged_real == expected_real) and (merged_groups == sum(per_shard_groups))
    if not ok:
        _log("FAIL — merged counts do not match per-shard sums. Investigate "
             "before passing this to isaaclab_to_lerobot.py.")
        return 1

    _log("OK — sanity check passed.")
    summary = {
        "output": str(out_path),
        "shards": [str(p) for p in shards],
        "per_shard_real": per_shard_real,
        "per_shard_groups": per_shard_groups,
        "merged_real": merged_real,
        "merged_groups": merged_groups,
        "merged_total_samples": merged_total,
    }
    _log(f"summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
