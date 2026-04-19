"""Phase 3 scene validation tests.

Runs after `just build-scene` has produced assets/hc10dt_v1.usd and assets/scene_cube_v1.usda.
"""
import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ARM_USD = REPO / "assets/hc10dt_v1.usd"
SCENE_USDA = REPO / "assets/scene_cube_v1.usda"
ISAACLAB = pathlib.Path(os.environ.get("ISAAC_LAB", str(pathlib.Path.home() / "IsaacLab")))


@pytest.fixture(scope="module")
def scene_dump() -> dict:
    if not ARM_USD.exists():
        pytest.skip(f"arm USD not built: {ARM_USD}")
    sh = ISAACLAB / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("Isaac Lab not installed")

    script = REPO / "scripts/validate/scene_inspect_with_app.py"
    out_json = REPO / "logs/scene_dump.json"
    # Scrub venv env so isaaclab.sh uses Isaac Sim's Python
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":")
                           if not p.endswith(".venv/bin"))
    result = subprocess.run(
        ["timeout", "300", str(sh), "-p", str(script),
         str(ARM_USD), "--json", str(out_json)],
        capture_output=True, text=True, timeout=360, env=env,
        cwd=str(ISAACLAB),
    )
    assert out_json.exists(), (
        f"scene_inspect did not produce JSON (exit {result.returncode})\n"
        f"STDOUT: {(result.stdout or '')[-2000:]}\n"
        f"STDERR: {(result.stderr or '')[-2000:]}"
    )
    return json.loads(out_json.read_text())


def test_arm_usd_exists():
    assert ARM_USD.exists(), f"{ARM_USD} not built"
    assert ARM_USD.stat().st_size > 1024, f"{ARM_USD} too small"


def test_arm_has_six_revolute_joints(scene_dump):
    revolute = [j for j in scene_dump["joints"] if j["type"] == "PhysicsRevoluteJoint"]
    assert len(revolute) == 6, f"expected 6 revolute joints, got {len(revolute)}"


def test_arm_joint_names_match_hc10dt(scene_dump):
    expected = {"joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"}
    names = {j["name"] for j in scene_dump["joints"] if j["type"] == "PhysicsRevoluteJoint"}
    assert expected <= names, f"missing: {expected - names}"


def test_arm_has_articulation_root(scene_dump):
    assert len(scene_dump["articulations"]) >= 1, "no articulation root found"


def test_scene_usda_exists():
    assert SCENE_USDA.exists(), f"{SCENE_USDA} not built"
    text = SCENE_USDA.read_text()
    assert "hc10dt_v1.usd" in text
    assert 'def "Arm"' in text
    assert 'def Cube "Table"' in text
