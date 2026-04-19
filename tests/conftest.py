"""Shared pytest fixtures. Isaac-dependent tests run Isaac scripts via subprocess
because Isaac Sim / Isaac Lab cannot be imported into an arbitrary Python interpreter."""
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ISAAC_PY = pathlib.Path("/opt/IsaacSim/python.sh")
ISAAC_LAB = pathlib.Path(os.environ.get("ISAAC_LAB", str(pathlib.Path.home() / "IsaacLab")))


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return REPO


@pytest.fixture(scope="session")
def isaac_python() -> pathlib.Path:
    if not ISAAC_PY.exists():
        pytest.skip(f"Isaac Sim python not found at {ISAAC_PY}")
    return ISAAC_PY


@pytest.fixture(scope="session")
def isaaclab_sh() -> pathlib.Path:
    sh = ISAAC_LAB / "isaaclab.sh"
    if not sh.exists():
        pytest.skip(f"Isaac Lab not installed at {ISAAC_LAB}")
    return sh


def run_isaac_script(script_path, timeout=300, extra_args=None):
    """Run a script under Isaac Sim's bundled Python, capturing output."""
    args = [str(ISAAC_PY), str(script_path), *(extra_args or [])]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def run_isaaclab_script(script_path, timeout=600, extra_args=None):
    """Run a script under the isaaclab.sh wrapper."""
    sh = ISAAC_LAB / "isaaclab.sh"
    args = [str(sh), "-p", str(script_path), *(extra_args or [])]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
