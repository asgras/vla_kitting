"""Filter a scripted-pick HDF5 to only valid demos.

Problems this fixes in a raw scripted dataset:
  * Prior scripted-pick runs used to leave stale entries in the HDF5 (the
    recorder's `EXPORT_SUCCEEDED_ONLY` mode still creates per-demo groups for
    failed attempts — they end up without a top-level `actions` dataset).
  * Very short demos (< MIN_LEN steps) are leftovers that timed out before
    reaching the success phase; they have an `actions` array but no useful
    trajectory.
  * `/data.env_args` and `/data.total` attrs must be copied or the upstream
    Isaac Lab Mimic annotator throws "can't locate attribute 'env_args'".

Usage:
    /opt/IsaacSim/python.sh scripts/data/clean_demos.py \\
        --input datasets/teleop/cube_scripted.hdf5 \\
        --output datasets/teleop/cube_scripted_clean.hdf5
"""
from __future__ import annotations

import argparse
import pathlib

import h5py


# Drop demos shorter than this many action steps — they're leftovers from
# runs that timed out before reaching the success phase.
MIN_LEN = 100


def clean(src: pathlib.Path, dst: pathlib.Path) -> int:
    if dst.exists():
        dst.unlink()
    with h5py.File(src, "r") as fsrc, h5py.File(dst, "w") as fdst:
        for k, v in fsrc.attrs.items():
            fdst.attrs[k] = v
        fdst.create_group("data")
        src_data = fsrc["data"]
        for k, v in src_data.attrs.items():
            fdst["data"].attrs[k] = v

        good = 0
        bad = 0

        def _demo_order(name: str) -> int:
            return int(name.split("_")[1]) if name.startswith("demo_") else -1

        for key in sorted(src_data.keys(), key=_demo_order):
            g = src_data[key]
            if "actions" not in g or g["actions"].shape[0] < MIN_LEN:
                bad += 1
                continue
            new_name = f"demo_{good}"
            fsrc.copy(f"data/{key}", fdst["data"], name=new_name)
            for ak, av in src_data[key].attrs.items():
                fdst["data"][new_name].attrs[ak] = av
            good += 1

    print(f"kept {good}, dropped {bad} (min_len={MIN_LEN})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    return clean(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
