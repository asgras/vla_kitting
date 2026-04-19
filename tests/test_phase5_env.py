"""Phase 5 env validation tests."""
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ISAACLAB = pathlib.Path(os.environ.get("ISAAC_LAB", str(pathlib.Path.home() / "IsaacLab")))


def _isaaclab_env():
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":")
                           if not p.endswith(".venv/bin"))
    return env


@pytest.mark.skipif(
    not (ISAACLAB / "isaaclab.sh").exists(), reason="Isaac Lab not installed"
)
def test_env_smoke_passes():
    script = REPO / "scripts/validate/env_smoke.py"
    result = subprocess.run(
        ["timeout", "480", str(ISAACLAB / "isaaclab.sh"), "-p", str(script)],
        capture_output=True, text=True, timeout=540,
        env=_isaaclab_env(), cwd=str(ISAACLAB),
    )
    stdout = result.stdout or ""
    assert "[env_smoke] result: OK" in stdout, (
        f"env_smoke did not report OK (exit {result.returncode})\n"
        f"STDOUT tail:\n{stdout[-4000:]}\n"
        f"STDERR tail:\n{(result.stderr or '')[-2000:]}"
    )
