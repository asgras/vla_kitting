"""Wrapper around Isaac Lab's record_demos.py that registers our VLA pick-cube task first.

Usage:
    ./isaaclab.sh -p scripts/teleop/record_demos.py \\
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 \\
        --teleop_device keyboard \\
        --dataset_file datasets/teleop/cube_raw.hdf5 \\
        --num_demos 15
"""
from __future__ import annotations

import pathlib
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Register our envs. Must happen before record_demos.py's arg parsing / gym.make call.
import envs  # noqa: F401

# Re-exec the Isaac Lab record_demos.py as main
ISAACLAB_RECORD = pathlib.Path.home() / "IsaacLab/scripts/tools/record_demos.py"
if not ISAACLAB_RECORD.exists():
    print(f"ERROR: Isaac Lab record_demos.py not found at {ISAACLAB_RECORD}", file=sys.stderr)
    sys.exit(2)

runpy.run_path(str(ISAACLAB_RECORD), run_name="__main__")
