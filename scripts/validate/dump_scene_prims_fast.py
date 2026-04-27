"""Dump USD prim hierarchy under /World/envs without calling env.reset().
Calls only InteractiveScene initialization through gym.make, then traverses
the stage. Designed to bypass the hang in dump_scene_prims.py at env.reset()
on the current env config.
"""
from __future__ import annotations
import sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def main() -> int:
    import gymnasium as gym
    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    TASK_ID = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    env_cfg = parse_env_cfg(TASK_ID, device="cuda", num_envs=1)
    print("[dump] gym.make starting (no reset)...", flush=True)
    env = gym.make(TASK_ID, cfg=env_cfg)
    print("[dump] env created", flush=True)

    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade
    stage = omni.usd.get_context().get_stage()

    print("\n=== /World children ===", flush=True)
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for c in world.GetAllChildren():
            print(f"  {c.GetName()}  type={c.GetTypeName()}  path={c.GetPath()}")

    print("\n=== /World/envs/env_0 children ===", flush=True)
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    if env0.IsValid():
        for c in env0.GetAllChildren():
            print(f"  {c.GetName()}  type={c.GetTypeName()}  path={c.GetPath()}")

    print("\n=== All prims with 'cube' / 'Cube' in path or name (case-insensitive) ===", flush=True)
    for prim in stage.Traverse():
        n = prim.GetName().lower()
        p = str(prim.GetPath()).lower()
        if "cube" in n or "cube" in p:
            translate = None
            try:
                if prim.IsA(UsdGeom.Xformable):
                    xf = UsdGeom.Xformable(prim)
                    ops = xf.GetOrderedXformOps()
                    translate = [op.Get() for op in ops
                                 if op.GetOpName().startswith("xformOp:translate")]
            except Exception:
                pass
            print(f"  {prim.GetPath()}  type={prim.GetTypeName()}  translate={translate}")

    print("\n=== All Cube/Mesh geometry prims (translate inspection) ===", flush=True)
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t in ("Cube", "Mesh"):
            translate = None
            if prim.IsA(UsdGeom.Xformable):
                try:
                    xf = UsdGeom.Xformable(prim)
                    ops = xf.GetOrderedXformOps()
                    translate = [op.Get() for op in ops
                                 if op.GetOpName().startswith("xformOp:translate")]
                except Exception:
                    pass
            print(f"  {prim.GetPath()}  type={t}  translate={translate}")

    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
