"""Inspect an Isaac Lab demo dataset (HDF5) and dump a summary report.

Reports for each dataset:
  - demo count, total frames, per-episode length stats
  - observation + action key list and shapes
  - per-demo success flag (if present)
  - action value ranges (useful to spot saturated IK outputs)
  - camera-frame mean brightness (useful to spot all-black frames)

Usage:
    /opt/IsaacSim/python.sh scripts/data/inspect_demos.py \\
        datasets/teleop/cube_scripted_clean.hdf5 \\
        datasets/mimic/cube_mimic_smoke.hdf5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np


def _log(msg: str) -> None:
    print(msg, flush=True)


def inspect(path: pathlib.Path) -> dict:
    summary: dict = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary

    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            summary["error"] = "no /data group"
            return summary

        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]) if x.startswith("demo_") else -1)
        summary["num_demos"] = len(demo_keys)
        summary["data_attrs"] = {
            k: (v if not isinstance(v, bytes) else v.decode("utf-8", errors="replace"))[:80]
            if isinstance(v, (str, bytes)) else str(v)
            for k, v in data.attrs.items()
        }

        lens = []
        successes = 0
        missing_actions = 0
        action_min = np.array([float("inf")] * 7)
        action_max = np.array([-float("inf")] * 7)
        sample_obs_keys = None
        wrist_bright_sum = 0.0
        third_bright_sum = 0.0
        bright_n = 0

        for key in demo_keys:
            demo = data[key]
            if "actions" not in demo:
                missing_actions += 1
                continue
            T = demo["actions"].shape[0]
            lens.append(T)
            if demo.attrs.get("success", False):
                successes += 1

            if sample_obs_keys is None and "obs" in demo:
                sample_obs_keys = {k: tuple(demo["obs"][k].shape) for k in demo["obs"].keys()
                                    if hasattr(demo["obs"][k], "shape")}

            acts = demo["actions"][...]
            action_min = np.minimum(action_min, acts.min(axis=0))
            action_max = np.maximum(action_max, acts.max(axis=0))

            # sample middle frame of first 3 demos for brightness
            if bright_n < 3 and "obs" in demo:
                mid = T // 2
                if "wrist_cam" in demo["obs"]:
                    wrist_bright_sum += float(demo["obs"]["wrist_cam"][mid].mean())
                    bright_n_wrist = bright_n + 1
                if "third_person_cam" in demo["obs"]:
                    third_bright_sum += float(demo["obs"]["third_person_cam"][mid].mean())
                bright_n += 1

        summary["num_missing_actions"] = missing_actions
        summary["num_successes"] = successes
        if lens:
            summary["ep_len_min"] = int(min(lens))
            summary["ep_len_max"] = int(max(lens))
            summary["ep_len_mean"] = float(np.mean(lens))
            summary["total_frames"] = int(sum(lens))
        summary["obs_keys_shape"] = sample_obs_keys
        summary["action_min"] = action_min.tolist()
        summary["action_max"] = action_max.tolist()
        if bright_n:
            summary["wrist_cam_mean_brightness"] = wrist_bright_sum / bright_n
            summary["third_person_cam_mean_brightness"] = third_bright_sum / bright_n

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", type=pathlib.Path, nargs="+")
    args = ap.parse_args()

    out: list[dict] = []
    for p in args.paths:
        out.append(inspect(p))

    _log(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
