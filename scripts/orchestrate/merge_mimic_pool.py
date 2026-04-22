"""Merge all datasets/mimic/pool/batch_*.hdf5 into a single master HDF5.

Keeps only demos marked `success=True` — failures from `generation_keep_failed`
aren't useful for training. Renumbers demos sequentially so the output has
demo_0..demo_N-1. Preserves `/data.env_args` from the first non-empty batch.

Usage:
    /opt/IsaacSim/python.sh scripts/orchestrate/merge_mimic_pool.py \
        --pool datasets/mimic/pool \
        --output datasets/mimic/cube_mimic_all.hdf5
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py

# Drop demos shorter than this many action steps — they're leftovers from
# runs that timed out before reaching the success phase. Same threshold as
# scripts/data/clean_demos.py.
MIN_DEMO_LEN = 100


def _log(msg: str) -> None:
    print(f"[merge] {msg}", flush=True)


def _demo_ordering(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (ValueError, IndexError):
        return -1


def merge(pool_dir: pathlib.Path, out_path: pathlib.Path) -> int:
    """Concatenate successful demos from all batch_*.hdf5 into one file."""
    batches = sorted(pool_dir.glob("batch_*.hdf5"))
    if not batches:
        _log(f"no batches in {pool_dir}; nothing to do")
        return 1

    # Work atomically: write to .tmp then rename.
    tmp = out_path.with_suffix(".tmp.hdf5")
    if tmp.exists():
        tmp.unlink()

    total_kept = 0
    total_dropped = 0
    env_args_set = False

    with h5py.File(tmp, "w") as fdst:
        fdst.create_group("data")
        for b in batches:
            try:
                with h5py.File(b, "r") as fsrc:
                    src_data = fsrc["data"]
                    if not env_args_set:
                        for k, v in fsrc.attrs.items():
                            fdst.attrs[k] = v
                        for k, v in src_data.attrs.items():
                            fdst["data"].attrs[k] = v
                        env_args_set = True

                    for key in sorted(src_data.keys(), key=_demo_ordering):
                        g = src_data[key]
                        # Only keep successful demos
                        success = bool(g.attrs.get("success", False))
                        if not success:
                            total_dropped += 1
                            continue
                        if "actions" not in g or g["actions"].shape[0] < MIN_DEMO_LEN:
                            total_dropped += 1
                            continue
                        new_name = f"demo_{total_kept}"
                        fsrc.copy(f"data/{key}", fdst["data"], name=new_name)
                        for ak, av in g.attrs.items():
                            fdst["data"][new_name].attrs[ak] = av
                        total_kept += 1
            except (OSError, KeyError) as exc:
                _log(f"  skipping {b.name}: {exc}")
                continue

        fdst["data"].attrs["total_demos"] = total_kept

    tmp.replace(out_path)
    _log(f"merged {len(batches)} batch files → {out_path} "
         f"(kept {total_kept}, dropped {total_dropped})")

    # Print machine-readable stats for the orchestrator.
    print(json.dumps({
        "batches": len(batches),
        "total_demos": total_kept,
        "dropped": total_dropped,
        "out": str(out_path),
    }), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    return merge(args.pool, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
