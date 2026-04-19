"""Minimum possible Isaac Sim smoke: launch AppLauncher and exit.

If this hangs, the issue is in the app startup itself (MDL/Nucleus/Vulkan),
not in any of our scene building code.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    print("[minimal] importing AppLauncher", flush=True)
    from isaaclab.app import AppLauncher

    print("[minimal] launching headless", flush=True)
    t0 = time.perf_counter()
    app_launcher = AppLauncher(headless=True)
    t1 = time.perf_counter()
    print(f"[minimal] app launched in {t1 - t0:.1f}s", flush=True)

    sim_app = app_launcher.app
    print("[minimal] sim_app handle obtained", flush=True)

    # Pump the app a few times so any deferred init can happen
    for i in range(10):
        sim_app.update()
    t2 = time.perf_counter()
    print(f"[minimal] 10 app updates done in {t2 - t1:.1f}s", flush=True)

    sim_app.close()
    print(f"[minimal] total wall: {time.perf_counter() - t0:.1f}s — OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
