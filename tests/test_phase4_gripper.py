"""Phase 4 gripper validation tests."""
import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
COMBINED_USD = REPO / "assets/hc10dt_with_gripper_v1.usd"
SCENE_WITH_GRIPPER = REPO / "assets/scene_with_gripper.usda"
ISAACLAB = pathlib.Path(os.environ.get("ISAAC_LAB", str(pathlib.Path.home() / "IsaacLab")))


def _isaaclab_env():
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":")
                           if not p.endswith(".venv/bin"))
    return env


@pytest.fixture(scope="module")
def combined_dump() -> dict:
    if not COMBINED_USD.exists():
        pytest.skip(f"combined USD not built: {COMBINED_USD}")
    sh = ISAACLAB / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("Isaac Lab not installed")

    script = REPO / "scripts/validate/scene_inspect_with_app.py"
    out_json = REPO / "logs/scene_dump_combined.json"
    result = subprocess.run(
        ["timeout", "300", str(sh), "-p", str(script),
         str(COMBINED_USD), "--json", str(out_json)],
        capture_output=True, text=True, timeout=360,
        env=_isaaclab_env(), cwd=str(ISAACLAB),
    )
    assert out_json.exists(), f"inspect failed:\n{(result.stderr or '')[-2000:]}"
    return json.loads(out_json.read_text())


def test_combined_usd_exists():
    assert COMBINED_USD.exists()
    assert COMBINED_USD.stat().st_size > 1024


def test_has_12_revolute_joints(combined_dump):
    revolute = [j for j in combined_dump["joints"] if j["type"] == "PhysicsRevoluteJoint"]
    assert len(revolute) == 12


def test_has_arm_and_gripper_joints(combined_dump):
    names = {j["name"] for j in combined_dump["joints"]}
    arm_joints = {"joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"}
    gripper_joints = {"robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"}
    assert arm_joints <= names
    assert gripper_joints <= names


def test_single_articulation(combined_dump):
    assert len(combined_dump["articulations"]) == 1


def test_scene_with_gripper_exists():
    assert SCENE_WITH_GRIPPER.exists()
    text = SCENE_WITH_GRIPPER.read_text()
    assert "hc10dt_with_gripper_v1.usd" in text
    assert 'def "Robot"' in text


def test_gripper_smoke_drives_joints():
    """Run the gripper smoke test and check it reports OK."""
    sh = ISAACLAB / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("Isaac Lab not installed")
    script = REPO / "scripts/validate/gripper_smoke.py"
    result = subprocess.run(
        ["timeout", "300", str(sh), "-p", str(script)],
        capture_output=True, text=True, timeout=360,
        env=_isaaclab_env(), cwd=str(ISAACLAB),
    )
    assert "[gripper_smoke] result: OK" in (result.stdout or ""), (
        f"gripper smoke did not report OK\n"
        f"STDOUT tail:\n{(result.stdout or '')[-3000:]}\n"
        f"STDERR tail:\n{(result.stderr or '')[-2000:]}"
    )
