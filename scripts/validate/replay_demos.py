"""Wrapper around Isaac Lab's replay_demos.py that registers our task first.

Usage:
    ./isaaclab.sh -p scripts/validate/replay_demos.py \\
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 \\
        --dataset_file datasets/teleop/cube_raw.hdf5
"""
from __future__ import annotations

import pathlib
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import envs  # noqa: F401  # register tasks

ISAACLAB_REPLAY = pathlib.Path.home() / "IsaacLab/scripts/tools/replay_demos.py"
if not ISAACLAB_REPLAY.exists():
    print(f"ERROR: Isaac Lab replay_demos.py not found at {ISAACLAB_REPLAY}", file=sys.stderr)
    sys.exit(2)

runpy.run_path(str(ISAACLAB_REPLAY), run_name="__main__")
