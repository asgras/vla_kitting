"""Phase 5 env smoke test: register and step the cube pick-place env with random actions."""
from __future__ import annotations

import pathlib
import sys

# Make our envs package importable
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg):
    print(f"[env_smoke] {msg}", flush=True)


def main() -> int:
    import numpy as np
    import torch
    import gymnasium as gym

    _log("importing envs (triggers registration)")
    import envs  # noqa: F401
    _log(f"registered tasks: {[k for k in gym.envs.registry.keys() if 'HC10DT' in k]}")

    _log("creating env Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0")
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg("Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0", cfg=env_cfg, render_mode=None)
    _log(f"env created: obs_space={env.observation_space}, action_space={env.action_space}")

    obs, _ = env.reset()
    _log(f"reset OK, obs type: {type(obs)}")
    if isinstance(obs, dict) and "policy" in obs:
        pol = obs["policy"]
        _log(f"policy obs keys: {list(pol.keys()) if isinstance(pol, dict) else type(pol)}")

    # Step with zero actions for 60 steps
    action_dim = env.action_space.shape[-1]
    _log(f"action_dim: {action_dim}")

    zero_action = torch.zeros((1, action_dim), device="cuda:0")
    successes = 0
    timeouts = 0
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if terminated[0].item():
            successes += 1
            env.reset()
        if truncated[0].item():
            timeouts += 1
            env.reset()

    _log(f"60 steps done (zero-action) — terminated: {successes}, timeouts: {timeouts}")

    # Small random actions for 30 steps
    for i in range(30):
        action = torch.randn((1, action_dim), device="cuda:0") * 0.01
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated[0].item() or truncated[0].item():
            env.reset()
    _log("30 random-action steps done")

    env.close()
    sim_app.close()
    _log("result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
