"""Run a single eval episode and log every action[6] (gripper) value the
policy emits. Used to distinguish "model never outputs close" from "model
outputs close at wrong timing" when SR remains 0/N.

Output: prints summary stats to stdout, writes a per-step CSV to the
specified --csv_out path.
"""
from __future__ import annotations
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--csv_out", type=str,
                    default=str(REPO / "reports" / "runs" /
                               "v4_gripper_weight_2026-04-26" /
                               "gripper_probe.csv"))
parser.add_argument("--task", type=str,
                    default="pick up the cube and place it on the magenta circle")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.utils import init_logging
    from scripts.train.run_vla_closed_loop import (  # type: ignore
        _obs_env_to_lerobot,
    )
    from lerobot.processor.pipeline import (
        prepare_observation_for_inference,
    )

    init_logging()
    device = torch.device("cuda")
    print(f"[probe] loading policy from {args_cli.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args_cli.checkpoint, device=device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args_cli.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    env_cfg = parse_env_cfg(TASK, device=str(device), num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)

    obs_env, _ = env.reset()
    policy.reset()
    cube0 = obs_env["policy"]["cube_pos"][0].tolist()
    print(f"[probe] cube starts at ({cube0[0]:.3f}, {cube0[1]:.3f}, {cube0[2]:.3f})")

    rows = []
    for step in range(args_cli.max_steps):
        raw = _obs_env_to_lerobot(obs_env, drop_cube_pos=True)
        frame = prepare_observation_for_inference(
            observation=raw, device=device, task=args_cli.task,
            robot_type="yaskawa_hc10dt_robotiq_2f85",
        )
        batch = preprocess(frame)
        action = policy.select_action(batch)
        action = postprocess(action)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        action = action.view(1, -1)
        gripper_raw = float(action[0, 6].item())
        # Apply env's binary threshold for sending.
        gripper_send = 1.0 if gripper_raw > 0 else -1.0
        action[0, 6] = gripper_send
        action = action.to(env_cfg.sim.device)
        obs_env, reward, terminated, truncated, info = env.step(action)
        cube = obs_env["policy"]["cube_pos"][0].tolist()
        rows.append((step, gripper_raw, gripper_send, cube[0], cube[1], cube[2]))
        if step % 100 == 0:
            print(f"[probe] step={step} grip_raw={gripper_raw:+.3f} "
                  f"grip_send={gripper_send:+.0f} cube=({cube[0]:.2f},{cube[1]:.2f},{cube[2]:.3f})")
        if bool(terminated[0]) or bool(truncated[0]):
            print(f"[probe] terminated at step {step} term={bool(terminated[0])} trunc={bool(truncated[0])}")
            break

    out = pathlib.Path(args_cli.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("step,gripper_raw,gripper_send,cube_x,cube_y,cube_z\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"[probe] wrote {out}")

    grip_values = [r[1] for r in rows]
    cube_zs = [r[5] for r in rows]
    closes = sum(1 for v in grip_values if v < 0)
    print(f"[probe] summary: n_steps={len(rows)} "
          f"grip_min={min(grip_values):.3f} grip_max={max(grip_values):.3f} "
          f"grip_mean={sum(grip_values)/len(grip_values):.3f} "
          f"close_steps={closes}/{len(grip_values)} "
          f"cube_z_max={max(cube_zs):.3f}")
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
