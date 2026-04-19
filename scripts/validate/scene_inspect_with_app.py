"""Inspect a USD scene. Launches Isaac Sim to get access to pxr.Usd.

Usage:
    ./isaaclab.sh -p scripts/validate/scene_inspect_with_app.py <path-to-usd>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "logs/scene_dump.json")
    ap.add_argument("--report", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "reports/scene_inspection.md")
    args = ap.parse_args()

    if not args.usd.exists():
        print(f"USD not found: {args.usd}", file=sys.stderr)
        return 2

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        print(f"Failed to open stage", file=sys.stderr)
        app.close()
        return 2

    dump = {
        "file": str(args.usd),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prims": [],
        "articulations": [],
        "joints": [],
        "rigid_bodies": [],
    }

    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        path = str(prim.GetPath())
        dump["prims"].append({"path": path, "type": type_name})

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            dump["articulations"].append({"path": path, "type": type_name})

        if type_name in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            lower = upper = None
            if type_name == "PhysicsRevoluteJoint":
                rj = UsdPhysics.RevoluteJoint(prim)
                lower = rj.GetLowerLimitAttr().Get()
                upper = rj.GetUpperLimitAttr().Get()
            dump["joints"].append({
                "path": path, "type": type_name, "name": prim.GetName(),
                "lower": lower, "upper": upper,
            })

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            dump["rigid_bodies"].append({"path": path})

    revolute = [j for j in dump["joints"] if j["type"] == "PhysicsRevoluteJoint"]
    dump["revolute_count"] = len(revolute)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dump, indent=2, default=str))

    lines = [
        f"# Scene inspection: `{args.usd.name}`",
        "",
        f"- File: `{args.usd}`",
        f"- Default prim: `{dump['default_prim']}`",
        f"- Total prims: {len(dump['prims'])}",
        f"- Articulations: {len(dump['articulations'])}",
        f"- Revolute joints: {len(revolute)}",
        f"- Rigid bodies: {len(dump['rigid_bodies'])}",
        "",
        "## Articulations",
    ]
    for a in dump["articulations"]:
        lines.append(f"- `{a['path']}` (`{a['type']}`)")
    lines.append("")
    lines.append("## Revolute joints")
    for j in revolute:
        lo, hi = j["lower"], j["upper"]
        lines.append(f"- `{j['name']}`: [{lo}, {hi}] — `{j['path']}`")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(f"[scene_inspect] articulations={len(dump['articulations'])} "
          f"revolute={len(revolute)} prims={len(dump['prims'])}")
    print(f"[scene_inspect] report: {args.report}")
    print(f"[scene_inspect] json: {args.json}")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
