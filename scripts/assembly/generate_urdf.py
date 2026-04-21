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


# Per-joint (lower, upper) in radians, matching each mimic multiplier.
# Simplified Robotiq (robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro):
_MIMIC_JOINT_LIMITS_RAD_SIMPLE = {
    "robotiq_85_left_inner_knuckle_joint": (0.0, 0.85),    # mult=+1
    "robotiq_85_right_inner_knuckle_joint": (-0.85, 0.0),  # mult=-1
    "robotiq_85_left_finger_tip_joint": (-0.85, 0.0),      # mult=-1
    "robotiq_85_right_finger_tip_joint": (0.0, 0.85),      # mult=+1
}
# Canonical ros-industrial-attic Robotiq (robotiq_arg2f_85_macro.urdf.xacro).
# The attic xacro already declares finite limits, but finger joints that are
# children of mimic-driven links sometimes lose their limits during composition
# — we re-declare them here to be safe. Mimic multipliers from the macro:
#   right_outer_knuckle_joint:   +1
#   left/right_inner_knuckle_joint:  +1
#   left/right_inner_finger_joint:   -1  (range flips sign)
_MIMIC_JOINT_LIMITS_RAD_RIA = {
    "right_outer_knuckle_joint":        (0.0, 0.81),
    "left_inner_knuckle_joint":         (0.0, 0.8757),
    "right_inner_knuckle_joint":        (0.0, 0.8757),
    "left_inner_finger_joint":          (-0.8757, 0.0),
    "right_inner_finger_joint":         (-0.8757, 0.0),
}


def _add_mimic_joint_limits(urdf_text: str, joint_limits: dict, strip_mimics: bool) -> str:
    """Ensure each joint in `joint_limits` is revolute and has finite limits.

    If strip_mimics is True, also remove every <mimic .../> tag — use this when
    the control loop will drive each joint explicitly. If False, keep mimic
    tags so scripts/assembly/urdf_to_usd.py can lift them into PhysX's
    MimicJointAPI at USD-import time.
    """
    for jname, (lo, hi) in joint_limits.items():
        pattern = rf'(<joint\s+name="{jname}"\s+)type="continuous"(\s*>)'
        replacement = rf'\1type="revolute"\2'
        new_text, n = re.subn(pattern, replacement, urdf_text)
        if n:
            urdf_text = new_text

        # Force-replace any existing <limit> inside this joint's block with
        # the desired range. Mimic joints whose multiplier is negative need
        # a matching negative position range — stock Robotiq xacros ship
        # positive-only limits, which locks PhysX when the mimic constraint
        # tries to drive the joint past 0.
        block_re = re.compile(rf'(<joint\s+name="{jname}"[^>]*>)(.*?)(</joint>)', re.DOTALL)
        m = block_re.search(urdf_text)
        if not m:
            continue
        body = m.group(2)
        limit_elem = f'<limit lower="{lo}" upper="{hi}" velocity="2.0" effort="50" />'
        stripped = re.sub(r"\s*<limit\b[^/]*/>", "", body)
        new_body = stripped + limit_elem
        urdf_text = urdf_text[:m.start(2)] + new_body + urdf_text[m.end(2):]

    if strip_mimics:
        urdf_text = re.sub(
            r'\s*<mimic\s+joint="[^"]*"(?:\s+multiplier="[^"]*")?(?:\s+offset="[^"]*")?\s*/>',
            "", urdf_text,
        )
    return urdf_text


def generate(
    xacro_path: pathlib.Path,
    output_urdf: pathlib.Path,
    *,
    keep_mimic: bool = False,
    gripper_flavor: str = "simple",
) -> pathlib.Path:
    urdf_text = _bash(
        f"{SOURCE_SCRIPT} xacro {xacro_path} sim_gazebo:=false use_mock_hardware:=true"
    )
    # Find all package:// references and rewrite each to the absolute path.
    packages = set(re.findall(r"package://([A-Za-z0-9_]+)/", urdf_text))
    for pkg in packages:
        abs_path = resolve_package(pkg)
        urdf_text = urdf_text.replace(f"package://{pkg}", str(abs_path))

    joint_limits = _MIMIC_JOINT_LIMITS_RAD_RIA if gripper_flavor == "ria" else _MIMIC_JOINT_LIMITS_RAD_SIMPLE
    urdf_text = _add_mimic_joint_limits(
        urdf_text,
        joint_limits=joint_limits,
        strip_mimics=not keep_mimic,
    )

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(urdf_text)
    return output_urdf


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xacro", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--keep-mimic", action="store_true", default=False,
                   help="Preserve <mimic> tags so the USD importer can lift "
                        "them into PhysxMimicJointAPI. Default: strip them.")
    p.add_argument("--gripper", choices=("simple", "ria"), default="simple",
                   help="Which Robotiq 2F-85 flavour to expect when patching "
                        "mimic-joint limits.")
    args = p.parse_args(argv)

    out = generate(args.xacro, args.output, keep_mimic=args.keep_mimic, gripper_flavor=args.gripper)
    print(f"[generate_urdf] wrote {out} ({out.stat().st_size} bytes)")

    # Basic sanity: the URDF must contain all 6 joints.
    text = out.read_text()
    for j in ("joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"):
        if f'name="{j}"' not in text:
            print(f"[generate_urdf] WARNING: joint {j} not found in URDF", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
