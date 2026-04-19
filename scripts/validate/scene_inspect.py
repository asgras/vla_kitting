"""Inspect a USD scene: enumerate articulations, rigid bodies, joints.

Run via Isaac Sim's Python:
    /opt/IsaacSim/python.sh scripts/validate/scene_inspect.py <path-to-usd>

Writes reports/scene_inspection.md and logs/scene_dump.json.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("usd", type=pathlib.Path)
    p.add_argument("--report", type=pathlib.Path, default=REPO / "reports/scene_inspection.md")
    p.add_argument("--json", type=pathlib.Path, default=REPO / "logs/scene_dump.json")
    args = p.parse_args(argv)

    if not args.usd.exists():
        print(f"USD not found: {args.usd}", file=sys.stderr)
        return 2

    # Use pxr directly — no need to launch the full Isaac Sim app for static inspection.
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        print(f"Failed to open stage {args.usd}", file=sys.stderr)
        return 2

    dump = {
        "file": str(args.usd),
        "prims": [],
        "articulations": [],
        "joints": [],
        "rigid_bodies": [],
        "cameras": [],
    }

    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        path = str(prim.GetPath())
        dump["prims"].append({"path": path, "type": type_name})

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            dump["articulations"].append({"path": path, "type": type_name})

        if type_name == "PhysicsRevoluteJoint" or type_name == "PhysicsPrismaticJoint":
            joint = UsdPhysics.Joint(prim)
            lower = upper = None
            if type_name == "PhysicsRevoluteJoint":
                rj = UsdPhysics.RevoluteJoint(prim)
                lower = rj.GetLowerLimitAttr().Get()
                upper = rj.GetUpperLimitAttr().Get()
            dump["joints"].append({
                "path": path,
                "type": type_name,
                "name": prim.GetName(),
                "lower": lower,
                "upper": upper,
            })

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            dump["rigid_bodies"].append({"path": path})

        if type_name == "Camera":
            dump["cameras"].append({"path": path})

    dof_count = len(dump["joints"])
    dump["dof_count"] = dof_count

    # Write JSON
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dump, indent=2, default=str))

    # Write markdown report
    lines = [
        f"# Scene inspection: `{args.usd.name}`",
        "",
        f"- File: `{args.usd}`",
        f"- Total prims: {len(dump['prims'])}",
        f"- Articulations: {len(dump['articulations'])}",
        f"- Joints (revolute + prismatic): {dof_count}",
        f"- Rigid bodies: {len(dump['rigid_bodies'])}",
        f"- Cameras: {len(dump['cameras'])}",
        "",
        "## Articulations",
    ]
    for a in dump["articulations"]:
        lines.append(f"- `{a['path']}` (`{a['type']}`)")
    lines.append("")
    lines.append("## Joints")
    for j in dump["joints"]:
        lo = j["lower"]
        hi = j["upper"]
        lo_d = f"{lo:.1f}°" if isinstance(lo, (int, float)) else "—"
        hi_d = f"{hi:.1f}°" if isinstance(hi, (int, float)) else "—"
        lines.append(f"- `{j['name']}` ({j['type']}): [{lo_d}, {hi_d}]  — `{j['path']}`")
    lines.append("")
    lines.append("## Cameras")
    for c in dump["cameras"]:
        lines.append(f"- `{c['path']}`")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(f"[scene_inspect] prims={len(dump['prims'])} articulations={len(dump['articulations'])} "
          f"joints={dof_count} report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
