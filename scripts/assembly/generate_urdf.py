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


# Per-joint (lower, upper) in radians, matching each Robotiq mimic multiplier
# (see robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro).
_MIMIC_JOINT_LIMITS_RAD = {
    "robotiq_85_left_inner_knuckle_joint": (0.0, 0.85),    # mult=+1
    "robotiq_85_right_inner_knuckle_joint": (-0.85, 0.0),  # mult=-1
    "robotiq_85_left_finger_tip_joint": (-0.85, 0.0),      # mult=-1
    "robotiq_85_right_finger_tip_joint": (0.0, 0.85),      # mult=+1
}


def _add_mimic_joint_limits(urdf_text: str) -> str:
    """PhysX requires finite limits on joints used with the mimic feature. The stock
    Robotiq xacro declares the inner_knuckle and finger_tip joints as type="continuous"
    with no <limit> element. Convert to revolute and set per-joint limits matching
    each joint's mimic sign. Also strip <mimic> tags so each joint is independent;
    Isaac Sim's <mimic>-derived constraint fights our PD and drives joints to limits.
    """
    for jname, (lo, hi) in _MIMIC_JOINT_LIMITS_RAD.items():
        pattern = rf'(<joint\s+name="{jname}"\s+)type="continuous"(\s*>)'
        replacement = rf'\1type="revolute"\2'
        new_text, n = re.subn(pattern, replacement, urdf_text)
        if n == 0:
            continue
        urdf_text = new_text

        # Insert <limit> right before </joint> for this joint block.
        joint_block_pattern = rf'(<joint\s+name="{jname}"[^>]*>.*?)(</joint>)'
        limit_elem = f'<limit lower="{lo}" upper="{hi}" velocity="2.0" effort="50" />'
        urdf_text = re.sub(
            joint_block_pattern,
            rf"\1{limit_elem}\2",
            urdf_text,
            count=1,
            flags=re.DOTALL,
        )

    # Strip all <mimic .../> tags. Our action independently drives each joint with
    # the correct multiplier, which is more robust than Isaac Sim's mimic import.
    urdf_text = re.sub(r'\s*<mimic\s+joint="[^"]*"(?:\s+multiplier="[^"]*")?\s*/>', "", urdf_text)
    return urdf_text


def generate(xacro_path: pathlib.Path, output_urdf: pathlib.Path) -> pathlib.Path:
    urdf_text = _bash(
        f"{SOURCE_SCRIPT} xacro {xacro_path} sim_gazebo:=false use_mock_hardware:=true"
    )
    # Find all package:// references and rewrite each to the absolute path.
    packages = set(re.findall(r"package://([A-Za-z0-9_]+)/", urdf_text))
    for pkg in packages:
        abs_path = resolve_package(pkg)
        urdf_text = urdf_text.replace(f"package://{pkg}", str(abs_path))

    urdf_text = _add_mimic_joint_limits(urdf_text)

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
