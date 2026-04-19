"""Phase 2 smoke test: launch Isaac Sim headless, create a minimal stage,
step physics, report timing. Run via ./isaaclab.sh -p scripts/validate/isaac_smoke.py
"""

import sys

from isaaclab.app import AppLauncher


def _log(msg):
    print(f"[isaac_smoke] {msg}", flush=True)


_log("AppLauncher(headless=True)")
app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app
_log("app ready")

# Imports must come after AppLauncher.
import time

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def main() -> int:
    _log("building SimulationCfg")
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    _log("creating SimulationContext")
    sim = SimulationContext(sim_cfg)
    _log("sim context ready")

    _log("spawning ground plane")
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    _log("spawning dome light")
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    _log("calling sim.reset()")
    sim.reset()
    _log("reset complete, stepping")

    num_steps = 100
    t0 = time.perf_counter()
    for i in range(num_steps):
        sim.step()
        if i == 0:
            _log(f"first step done in {(time.perf_counter() - t0):.2f}s")
    dt = time.perf_counter() - t0

    step_rate = num_steps / dt
    _log(f"stepped {num_steps} times in {dt:.3f}s ({step_rate:.1f} Hz)")

    ok = step_rate >= 30.0  # sim-app step rate includes rendering; 30 Hz is a generous floor
    _log(f"result: {'OK' if ok else 'SLOW'}")
    sim_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
