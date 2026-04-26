"""Dump the USD prim hierarchy under /World/envs/env_0 to identify all cube-like prims.
Run after env creation to see why multiple cubes appear in renders."""
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
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()

    print("=== /World/envs/env_0 children ===")
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    if env0.IsValid():
        for c in env0.GetAllChildren():
            print(f"  {c.GetName()}  type={c.GetTypeName()}  path={c.GetPath()}")

    print("\n=== Walking for any prim with 'cube' in path or name (case-insensitive) ===")
    for prim in stage.Traverse():
        n = prim.GetName().lower()
        p = str(prim.GetPath()).lower()
        if "cube" in n or "cube" in p:
            xform = None
            try:
                from pxr import UsdGeom
                if prim.IsA(UsdGeom.Xformable):
                    xf = UsdGeom.Xformable(prim)
                    ops = xf.GetOrderedXformOps()
                    if ops:
                        xform = [op.GetAttr().Get() for op in ops if op.GetOpName().startswith("xformOp:translate")]
            except Exception:
                pass
            print(f"  {prim.GetPath()}  type={prim.GetTypeName()}  translate={xform}")

    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
