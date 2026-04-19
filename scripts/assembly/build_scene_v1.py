"""Phase 3 scene builder for V1 (cube pick-place).

Pipeline:
  1. Generate a plain URDF from our V1 xacro (runs xacro in ROS env first)
  2. Use Isaac Lab's convert_urdf.py to produce assets/hc10dt_v1.usd
  3. Build assets/scene_cube_v1.usda that references the arm USD and adds
     ground plane, table box, and lighting. (The cube is spawned at env reset.)

Run under Isaac Sim Python via isaaclab.sh:
    ./isaaclab.sh -p scripts/assembly/build_scene_v1.py
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
XACRO = ASSETS / "hc10dt_v1.urdf.xacro"
URDF = pathlib.Path("/tmp/hc10dt_v1.urdf")
ARM_USD = ASSETS / "hc10dt_v1.usd"
SCENE_USDA = ASSETS / "scene_cube_v1.usda"

ISAACLAB = pathlib.Path.home() / "IsaacLab"
CONVERT_URDF = ISAACLAB / "scripts/tools/convert_urdf.py"


def step_generate_urdf() -> pathlib.Path:
    print(f"[build_scene] generating URDF from xacro → {URDF}")
    helper = REPO / "scripts/assembly/generate_urdf.py"
    # generate_urdf.py shells out to ROS — run with system python, not Isaac Sim python.
    # Scrub PYTHONPATH/PYTHONHOME/LD_LIBRARY_PATH so /usr/bin/python3 (3.12) doesn't
    # accidentally load Isaac Sim's 3.11 stdlib.
    clean_env = {
        k: v for k, v in subprocess.os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH",
                     "LD_PRELOAD", "CARB_APP_PATH", "EXP_PATH", "ISAAC_PATH")
    }
    result = subprocess.run(
        ["/usr/bin/python3", str(helper), "--xacro", str(XACRO), "--output", str(URDF)],
        capture_output=True, text=True, env=clean_env,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"generate_urdf failed (exit {result.returncode})")
    print(result.stdout.strip())
    if not URDF.exists():
        raise FileNotFoundError(URDF)
    return URDF


def step_convert_to_usd() -> pathlib.Path:
    print(f"[build_scene] converting URDF → USD: {ARM_USD}")
    ARM_USD.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale outputs
    if ARM_USD.exists():
        ARM_USD.unlink()
    cfg_dir = ASSETS / "configuration"
    if cfg_dir.exists():
        for f in cfg_dir.glob("hc10dt_v1_*.usd"):
            f.unlink()
    # Use our direct importer script (Isaac Lab's UrdfConverter has broken deps on Isaac Sim 5.0)
    urdf2usd = REPO / "scripts/assembly/urdf_to_usd.py"
    result = subprocess.run(
        ["/opt/IsaacSim/python.sh", str(urdf2usd),
         "--urdf", str(URDF), "--usd", str(ARM_USD), "--fix-base"],
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"urdf_to_usd failed (exit {result.returncode})")
    if not ARM_USD.exists():
        raise FileNotFoundError(f"converter did not produce {ARM_USD}")
    print(f"[build_scene] arm USD: {ARM_USD} ({ARM_USD.stat().st_size} bytes)")
    return ARM_USD


SCENE_TEMPLATE = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World"
{{
    def "Arm" (
        prepend references = @{arm_ref}@
    )
    {{
        double3 xformOp:translate = (0.0, 0.0, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def Cube "Table"
    {{
        token physics:approximation = "boundingCube"
        bool physics:collisionEnabled = 1
        double size = 1
        float3 xformOp:scale = (1.2, 0.8, 0.04)
        double3 xformOp:translate = (0.6, 0.0, -0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.35, 0.25, 0.18)]
    }}

    def DistantLight "KeyLight"
    {{
        float intensity = 3000
        double3 xformOp:rotateXYZ = (-45, 0, 25)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def DomeLight "Ambient"
    {{
        float intensity = 400
        color3f color = (0.9, 0.9, 0.95)
    }}
}}
"""


def step_write_scene(arm_usd: pathlib.Path) -> pathlib.Path:
    rel = pathlib.Path("./") / arm_usd.name  # relative path so scene is portable
    content = SCENE_TEMPLATE.format(arm_ref=str(rel))
    SCENE_USDA.parent.mkdir(parents=True, exist_ok=True)
    SCENE_USDA.write_text(content)
    print(f"[build_scene] scene: {SCENE_USDA} ({SCENE_USDA.stat().st_size} bytes)")
    return SCENE_USDA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-urdf", action="store_true", help="reuse existing /tmp/hc10dt_v1.urdf")
    ap.add_argument("--skip-convert", action="store_true", help="reuse existing arm USD")
    args = ap.parse_args()

    if not args.skip_urdf:
        step_generate_urdf()
    else:
        assert URDF.exists(), URDF

    if not args.skip_convert:
        step_convert_to_usd()
    else:
        assert ARM_USD.exists(), ARM_USD

    step_write_scene(ARM_USD)
    print("[build_scene] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
