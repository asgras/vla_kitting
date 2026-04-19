"""Subprocess-based test for the Isaac smoke script."""
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ISAAC_LAB = pathlib.Path(os.environ.get("ISAAC_LAB", str(pathlib.Path.home() / "IsaacLab")))


@pytest.mark.skipif(
    not (ISAAC_LAB / "isaaclab.sh").exists(), reason="Isaac Lab not installed"
)
def test_isaac_smoke_runs():
    """Runs the smoke script and asserts progress markers appear.

    We time out the wrapper generously because Isaac Sim shutdown can be slow;
    a nonzero exit from the timeout wrapper is not a failure as long as the
    script printed 'result: OK' before then.
    """
    script = REPO / "scripts/validate/isaac_smoke.py"
    # Strip VIRTUAL_ENV so isaaclab.sh uses Isaac Sim's bundled Python, not the
    # pytest venv (which doesn't have isaaclab installed).
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":")
                           if not p.endswith(".venv/bin"))
    result = subprocess.run(
        ["timeout", "240", str(ISAAC_LAB / "isaaclab.sh"), "-p", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ISAAC_LAB),
        env=env,
    )
    stdout = result.stdout or ""
    assert "[isaac_smoke] result: OK" in stdout, (
        f"smoke did not complete (exit {result.returncode}):\n"
        f"STDOUT tail:\n{stdout[-3000:]}\n"
        f"STDERR tail:\n{(result.stderr or '')[-2000:]}"
    )
