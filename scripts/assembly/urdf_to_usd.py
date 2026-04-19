"""Direct URDF → USD converter using Isaac Sim's URDFParseAndImportFile command.

This bypasses Isaac Lab's UrdfConverter wrapper, which has extension-dependency
problems with Isaac Sim 5.0's module layout. Uses the standalone_examples pattern.

Run with Isaac Sim's bundled Python:
    /opt/IsaacSim/python.sh scripts/assembly/urdf_to_usd.py \\
        --urdf /tmp/hc10dt_v1.urdf --usd assets/hc10dt_v1.usd --fix-base
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=pathlib.Path)
    ap.add_argument("--usd", required=True, type=pathlib.Path)
    ap.add_argument("--fix-base", action="store_true", default=False)
    ap.add_argument("--merge-fixed-joints", action="store_true", default=False)
    ap.add_argument("--convex-decomp", action="store_true", default=False)
    ap.add_argument("--distance-scale", type=float, default=1.0)
    args = ap.parse_args()

    if not args.urdf.exists():
        print(f"URDF not found: {args.urdf}", file=sys.stderr)
        return 2

    from isaacsim import SimulationApp

    print(f"[urdf2usd] launching SimulationApp (headless)", flush=True)
    kit = SimulationApp({"headless": True})

    import omni.kit.commands
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics

    print(f"[urdf2usd] creating import config", flush=True)
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        print("URDFCreateImportConfig failed", file=sys.stderr)
        kit.close()
        return 1

    import_config.merge_fixed_joints = args.merge_fixed_joints
    import_config.convex_decomp = args.convex_decomp
    import_config.import_inertia_tensor = True
    import_config.fix_base = args.fix_base
    import_config.distance_scale = args.distance_scale
    # Make the importer produce a self-contained USD (embedded meshes)
    import_config.make_default_prim = True
    import_config.self_collision = False

    print(f"[urdf2usd] parsing+importing {args.urdf}", flush=True)
    args.usd.parent.mkdir(parents=True, exist_ok=True)
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(args.urdf),
        import_config=import_config,
        get_articulation_root=True,
        dest_path=str(args.usd),  # supported by the importer to save a USD
    )
    if not status:
        print("URDFParseAndImportFile failed", file=sys.stderr)
        kit.close()
        return 1

    print(f"[urdf2usd] prim path: {prim_path}", flush=True)

    # If dest_path didn't trigger a save, save the current stage to disk
    if not args.usd.exists():
        print(f"[urdf2usd] dest_path didn't auto-save; saving stage explicitly", flush=True)
        stage = omni.usd.get_context().get_stage()
        stage.Export(str(args.usd))

    if args.usd.exists():
        print(f"[urdf2usd] saved USD: {args.usd} ({args.usd.stat().st_size} bytes)", flush=True)
        ok = True
    else:
        print(f"[urdf2usd] ERROR: USD not produced at {args.usd}", file=sys.stderr)
        ok = False

    kit.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
