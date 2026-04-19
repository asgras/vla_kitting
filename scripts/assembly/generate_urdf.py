"""Run xacro on the HC10DT xacro and resolve package:// URIs to absolute paths.

Usage (outside Isaac Sim — needs ROS environment):
    python3 scripts/assembly/generate_urdf.py \
        --xacro /home/ubuntu/kitting_ws/src/motoman_ROS2/motoman_description/robots/hc10dt.xacro \
        --output /tmp/hc10dt_v1.urdf
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


SOURCE_SCRIPT = """
    source /opt/ros/jazzy/setup.bash &&
    source ~/kitting_ws/install/setup.bash &&
"""


def _bash(cmd: str) -> str:
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"bash failed ({result.returncode}):\n{result.stderr}")
    return result.stdout


def resolve_package(pkg: str) -> pathlib.Path:
    out = _bash(f"{SOURCE_SCRIPT} ros2 pkg prefix {pkg}").strip()
    share = pathlib.Path(out) / "share" / pkg
    if not share.exists():
        raise FileNotFoundError(f"package share dir missing: {share}")
    return share


def generate(xacro_path: pathlib.Path, output_urdf: pathlib.Path) -> pathlib.Path:
    urdf_text = _bash(
        f"{SOURCE_SCRIPT} xacro {xacro_path} sim_gazebo:=false use_mock_hardware:=true"
    )
    # Find all package:// references and rewrite each to the absolute path.
    packages = set(re.findall(r"package://([A-Za-z0-9_]+)/", urdf_text))
    for pkg in packages:
        abs_path = resolve_package(pkg)
        urdf_text = urdf_text.replace(f"package://{pkg}", str(abs_path))

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(urdf_text)
    return output_urdf


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xacro", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args(argv)

    out = generate(args.xacro, args.output)
    print(f"[generate_urdf] wrote {out} ({out.stat().st_size} bytes)")

    # Basic sanity: the URDF must contain all 6 joints.
    text = out.read_text()
    for j in ("joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"):
        if f'name="{j}"' not in text:
            print(f"[generate_urdf] WARNING: joint {j} not found in URDF", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
