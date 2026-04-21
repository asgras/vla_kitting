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
    ap.add_argument("--mimic-api", action="store_true", default=False,
                    help="Apply PhysxMimicJointAPI to Robotiq mimic joints.")
    ap.add_argument("--gripper", choices=("simple", "ria"), default="simple",
                    help="Mimic-joint name/limit/axis convention. 'simple' "
                         "matches our old URDF (robotiq_85_*_joint). 'ria' "
                         "matches the ros-industrial-attic 2F-85 xacro used by "
                         "assets/hc10dt_with_ria_gripper.urdf.xacro.")
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
        return 1

    # Post-import patch: (1) re-apply position limits that the URDF importer
    # strips from <mimic> joints; (2) apply PhysxMimicJointAPI so PhysX enforces
    # the mimic coupling in the solver. Without (2) the 6 Robotiq gripper
    # joints drift asymmetrically — driving left succeeds but the right side
    # settles however gravity pulls it, which is exactly the behaviour we saw
    # in the scripted pick (right fingertip z ≠ left z at low tool0_z).
    print(f"[urdf2usd] patching mimic-joint limits + mimic API in {args.usd}", flush=True)
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(args.usd))
    # Multipliers follow each xacro's <mimic> tags.
    if args.gripper == "ria":
        REF_JOINT = "finger_joint"
        # (lower_deg, upper_deg, gearing) for each mimic joint in the
        # ros-industrial-attic Robotiq 2F-85 macro.
        mimic_spec = {
            "right_outer_knuckle_joint":  (0.0, 46.4, +1.0),   # mult=+1
            "left_inner_knuckle_joint":   (0.0, 50.2, +1.0),   # mult=+1
            "right_inner_knuckle_joint":  (0.0, 50.2, +1.0),   # mult=+1
            "left_inner_finger_joint":    (-50.2, 0.0, -1.0),  # mult=-1
            "right_inner_finger_joint":   (-50.2, 0.0, -1.0),  # mult=-1
        }
    else:
        # Simplified Robotiq 2F-85 from robotiq_description — our original USD.
        REF_JOINT = "robotiq_85_left_knuckle_joint"
        mimic_spec = {
            "robotiq_85_left_inner_knuckle_joint":  (0.0, 48.7, +1.0),
            "robotiq_85_right_inner_knuckle_joint": (-48.7, 0.0, -1.0),
            "robotiq_85_left_finger_tip_joint":     (-48.7, 0.0, -1.0),
            "robotiq_85_right_finger_tip_joint":    (0.0, 48.7, +1.0),
            "robotiq_85_right_knuckle_joint":       (-48.7, 0.0, -1.0),
        }
    # Find reference joint prim path once.
    ref_prim_path = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() == REF_JOINT:
            ref_prim_path = prim.GetPath()
            break
    if ref_prim_path is None:
        print(f"[urdf2usd] ERROR: reference joint {REF_JOINT} not found", file=sys.stderr)
    else:
        print(f"[urdf2usd]   reference joint: {ref_prim_path}", flush=True)

    def _rev_axis_token(rj: UsdPhysics.RevoluteJoint) -> str:
        """Return 'rotX'/'rotY'/'rotZ' matching the joint's UsdPhysics axis."""
        axis = rj.GetAxisAttr().Get() or "X"
        return f"rot{str(axis).upper()[:1]}"

    # Look up the reference joint's axis (to use as `referenceJointAxis` on
    # every mimic — both sides should agree since the URDF uses the same
    # axis for every Robotiq revolute).
    ref_axis_token = "rotX"
    if ref_prim_path is not None:
        ref_prim = stage.GetPrimAtPath(ref_prim_path)
        ref_axis_token = _rev_axis_token(UsdPhysics.RevoluteJoint(ref_prim))
        print(f"[urdf2usd]   reference axis: {ref_axis_token}", flush=True)

    patched = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if name not in mimic_spec:
            continue
        lo, hi, gearing = mimic_spec[name]
        rj = UsdPhysics.RevoluteJoint(prim)
        rj.CreateLowerLimitAttr().Set(lo)
        rj.CreateUpperLimitAttr().Set(hi)
        joint_axis_token = _rev_axis_token(rj)
        if args.mimic_api:
            try:
                from pxr import PhysxSchema
                if ref_prim_path is not None and gearing != 0.0:
                    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, joint_axis_token)
                    mimic.CreateReferenceJointRel().SetTargets([ref_prim_path])
                    mimic.CreateReferenceJointAxisAttr().Set(ref_axis_token)
                    mimic.CreateGearingAttr().Set(float(gearing))
                    mimic.CreateOffsetAttr().Set(0.0)
                    print(f"[urdf2usd]   mimic on {prim.GetPath()}: axis={joint_axis_token} "
                          f"gearing={gearing:+.1f} limits=[{lo}, {hi}] deg", flush=True)
                else:
                    print(f"[urdf2usd]   limits on {prim.GetPath()}: [{lo}, {hi}] deg (no mimic)", flush=True)
            except Exception as exc:
                print(f"[urdf2usd]   WARNING: mimic API not applied to {prim.GetPath()}: {exc}", flush=True)
        else:
            print(f"[urdf2usd]   limits on {prim.GetPath()}: [{lo}, {hi}] deg (mimic-api disabled)", flush=True)
        patched += 1
    if patched == 0:
        print(f"[urdf2usd] WARNING: no mimic joints found to patch", flush=True)
    else:
        stage.GetRootLayer().Save()
        print(f"[urdf2usd] patched {patched} mimic joints and saved USD", flush=True)

    kit.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
