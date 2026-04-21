"""Compose HC10DT arm USD with NVIDIA's canonical Robotiq 2F-85 USD.

NVIDIA's Robotiq_2F_85_edit.usd ships with a fully-configured `Physx_Mimic`
physics variant — finger_joint is the single drive, the 7 mimic joints carry
PhysxMimicJointAPI with the correct axes (rotX for the finger chain, rotZ for
the outer-knuckle coupling), and the inner_finger_pad bodies that do the
actual pinching are present. That's exactly what our hand-rolled URDF was
missing.

This script:
  1. Opens the arm-only HC10DT USD (no gripper) as the base stage.
  2. References the NVIDIA Robotiq gripper under /<arm_root>/tool0/gripper.
  3. Adds a fixed joint between tool0 and the gripper's base so the two
     articulations join at the wrist flange.
  4. Disables the gripper's own articulation root (we want ONE articulation
     spanning arm + gripper).
  5. Writes the composed USD to assets/hc10dt_with_nvidia_gripper.usd.
"""
from __future__ import annotations
import argparse
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-usd", default="assets/hc10dt_arm_only_v1.usd", type=pathlib.Path)
    ap.add_argument("--gripper-usd", default="assets/nvidia_robotiq_2f85/Robotiq_2F_85_edit.usd", type=pathlib.Path)
    ap.add_argument("--out-usd", default="assets/hc10dt_with_nvidia_gripper.usd", type=pathlib.Path)
    # Location of the gripper's base on the arm's tool0 flange. The real
    # Robotiq 2F-85 has a ~8 mm adapter before its base_link — we bundle
    # that into a small Z offset here.
    ap.add_argument("--mount-z", type=float, default=0.0)
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    arm_usd = (repo / args.arm_usd).resolve()
    gripper_usd = (repo / args.gripper_usd).resolve()
    out_usd = (repo / args.out_usd).resolve()

    print(f"[compose] arm     = {arm_usd}", flush=True)
    print(f"[compose] gripper = {gripper_usd}", flush=True)
    print(f"[compose] out     = {out_usd}", flush=True)

    from isaacsim import SimulationApp
    kit = SimulationApp({"headless": True})

    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema

    # Easiest compositional approach: copy the arm USD to the output path
    # verbatim, then open the copy as a mutable stage and edit-in-place.
    import shutil
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(arm_usd, out_usd)

    stage = Usd.Stage.Open(str(out_usd))

    arm_tool0_path = None
    for p in stage.Traverse():
        if p.GetName() == "tool0":
            arm_tool0_path = p.GetPath()
            break
    if arm_tool0_path is None:
        print("[compose] ERROR: could not find tool0 prim in arm USD", file=sys.stderr)
        kit.close()
        return 1
    print(f"[compose] arm tool0 at {arm_tool0_path}", flush=True)

    # Place the gripper as a child of tool0.
    gripper_path = arm_tool0_path.AppendChild("gripper")
    gripper_prim = stage.DefinePrim(gripper_path, "Xform")
    gripper_prim.GetReferences().AddReference(str(gripper_usd))
    # Offset along the flange's local +Z if requested.
    xform = UsdGeom.XformCommonAPI(gripper_prim)
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, args.mount_z))
    print(f"[compose] added gripper prim at {gripper_path}", flush=True)

    # Look for the gripper's base body so we can weld it to tool0.
    gripper_base_path = None
    for p in stage.Traverse():
        if p.GetPath().HasPrefix(gripper_path) and p.GetName() == "base_link":
            gripper_base_path = p.GetPath()
            break
    # Fallback names used in some NVIDIA gripper variants.
    if gripper_base_path is None:
        for name in ("Robotiq_2F_85", "base", "robotiq_base_link"):
            for p in stage.Traverse():
                if p.GetPath().HasPrefix(gripper_path) and p.GetName() == name:
                    gripper_base_path = p.GetPath()
                    break
            if gripper_base_path is not None:
                break
    if gripper_base_path is None:
        print("[compose] WARNING: couldn't locate gripper base body — the fixed joint "
              "will not be added. Check the gripper USD structure.", file=sys.stderr)
    else:
        print(f"[compose] gripper base at {gripper_base_path}", flush=True)
        # Add a PhysicsFixedJoint from tool0 -> gripper_base.
        joint_path = arm_tool0_path.AppendChild("tool0_to_gripper_joint")
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([arm_tool0_path])
        joint.CreateBody1Rel().SetTargets([gripper_base_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, args.mount_z))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(f"[compose] added fixed joint {joint_path}", flush=True)

    # If the gripper asset has its own articulation root, deactivate it so we
    # end up with a single articulation covering arm + gripper. We keep the
    # arm's articulation root intact.
    removed_ar = 0
    for p in stage.Traverse():
        if not p.GetPath().HasPrefix(gripper_path):
            continue
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed_ar += 1
    print(f"[compose] removed {removed_ar} ArticulationRootAPI inside gripper", flush=True)

    stage.GetRootLayer().Save()
    print(f"[compose] saved {out_usd} ({out_usd.stat().st_size} bytes)", flush=True)

    kit.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
