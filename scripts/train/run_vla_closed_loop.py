"""Run a trained SmolVLA checkpoint as a closed-loop policy on the Isaac Lab
cube-pick environment.

Reads observations (wrist + third-person cameras, joint state, ee pose, cube
pose) from the env each step, feeds them to SmolVLA, and sends the returned
7D IK-rel + gripper action back into the env. Reports success rate across N
episodes and optionally saves gifs.

Usage:
    ./isaaclab.sh -p scripts/train/run_vla_closed_loop.py \\
        --checkpoint checkpoints/smoke/checkpoints/last/pretrained_model \\
        --num_episodes 3 \\
        --max_steps 1800 \\
        --save_gif reports/vla_rollout.gif
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a SmolVLA pretrained_model directory (checkpoints/last/pretrained_model).")
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--task", type=str,
                    default="pick up the cube and place it on the green target")
parser.add_argument("--save_gif", type=str, default=None,
                    help="Optional: write a side-by-side gif of the first episode here.")
parser.add_argument("--jsonl_out", type=str, default=None,
                    help="Optional: append per-episode structured results here.")
parser.add_argument("--ckpt_tag", type=str, default=None,
                    help="Optional short tag stamped on every JSONL record "
                         "(e.g. 'epoch_0023').")
parser.add_argument("--device", type=str, default="cuda")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[vla] {msg}", flush=True)


def _obs_env_to_lerobot(env_obs: dict) -> dict:
    """Remap env observation keys into the LeRobot feature names the policy
    was trained on, and convert CUDA tensors to CPU numpy. Handles both the
    Mimic env (eef_pos + eef_quat split) and the plain env (combined
    ee_pose) — the LeRobot dataset uses the 7D combined form either way.
    """
    import numpy as np
    import torch
    p = env_obs["policy"]

    def _np(t):
        return t[0].detach().cpu().numpy() if torch.is_tensor(t) else t[0]

    if "ee_pose" in p:
        ee_pose = _np(p["ee_pose"]).astype(np.float32)       # (7,)
    else:
        ee_pos = _np(p["eef_pos"]).astype(np.float32)        # (3,)
        ee_quat = _np(p["eef_quat"]).astype(np.float32)      # (4,)
        ee_pose = np.concatenate([ee_pos, ee_quat], axis=-1)

    return {
        "observation.state": _np(p["joint_pos"]).astype(np.float32),
        "observation.ee_pose": ee_pose,
        "observation.cube_pos": _np(p["cube_pos"]).astype(np.float32),
        "observation.images.wrist": _np(p["wrist_cam"]).astype(np.uint8),
        "observation.images.third_person": _np(p["third_person_cam"]).astype(np.uint8),
    }


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    import datetime as dt
    import json
    import numpy as np
    import torch
    import gymnasium as gym

    # Ensure our task is gym-registered AFTER AppLauncher ran.
    import envs  # noqa: F401

    # Lerobot src is a source clone — make it importable here.
    lerobot_src = pathlib.Path("/home/ubuntu/code/lerobot/src")
    if lerobot_src.exists() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from isaaclab_tasks.utils import parse_env_cfg
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    TASK_ID = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"  # non-Mimic; fewer obs terms, faster
    device = torch.device(args_cli.device)

    _log(f"loading SmolVLA from {args_cli.checkpoint}")
    ckpt_path = pathlib.Path(args_cli.checkpoint)
    adapter_cfg_file = ckpt_path / "adapter_config.json"
    if adapter_cfg_file.exists():
        # PEFT/LoRA checkpoint: adapter_config.json points to a base model on
        # the HF hub. The LOCAL config.json has the task-specific input_features
        # (our 2-cam wrist+third_person setup) — not the HF default 3-cam setup
        # — so we must use it to instantiate the policy.
        from peft import PeftModel
        from peft import PeftConfig as HfPeftConfig
        from lerobot.configs.policies import PreTrainedConfig

        hf_peft_cfg = HfPeftConfig.from_pretrained(str(ckpt_path))
        base_src = hf_peft_cfg.base_model_name_or_path
        # PreTrainedConfig.from_pretrained dispatches to the right subclass
        # via the 'type' field in config.json (must exist; injected manually
        # if lerobot's PEFT save path dropped it).
        local_cfg = PreTrainedConfig.from_pretrained(str(ckpt_path))
        _log(f"  PEFT detected: base={base_src}, adapter={ckpt_path}")
        _log(f"  using local config with input_features={list(local_cfg.input_features.keys())}")
        base_policy = SmolVLAPolicy.from_pretrained(base_src, config=local_cfg)
        policy = PeftModel.from_pretrained(base_policy, str(ckpt_path))
    else:
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()

    # Build preprocessors; they read normalization stats saved alongside the
    # policy (policy_preprocessor.json + the referenced .safetensors).
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args_cli.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    _log(f"creating env {TASK_ID}")
    env_cfg = parse_env_cfg(TASK_ID, device=str(device), num_envs=1)
    env = gym.make(TASK_ID, cfg=env_cfg)

    gif_frames: list[np.ndarray] = []
    saved_gif = False
    successes = 0

    for ep in range(args_cli.num_episodes):
        obs_env, _ = env.reset()
        policy.reset()
        cube0 = obs_env["policy"]["cube_pos"][0].tolist()
        _log(f"=== episode {ep} | cube at ({cube0[0]:.3f}, {cube0[1]:.3f}, {cube0[2]:.3f})")

        success_this_ep = False
        truncated_this_ep = False
        for step in range(args_cli.max_steps):
            # Build an inference frame matching the dataset schema, then
            # preprocess (normalize images, tokenize task string).
            raw = _obs_env_to_lerobot(obs_env)
            frame = prepare_observation_for_inference(
                observation=raw, device=device, task=args_cli.task, robot_type="yaskawa_hc10dt_robotiq_2f85"
            )
            batch = preprocess(frame)

            # SmolVLA outputs a (1, action_dim) tensor; unnormalize via postprocess.
            action = policy.select_action(batch)
            action = postprocess(action)

            # The env expects a (1, 7) torch tensor on its own device.
            if not torch.is_tensor(action):
                action = torch.as_tensor(action)
            action = action.view(1, -1).to(env_cfg.sim.device)

            obs_env, reward, terminated, truncated, info = env.step(action)

            # Save first-episode frames for gif.
            if ep == 0 and args_cli.save_gif and step % 4 == 0:
                wrist = obs_env["policy"]["wrist_cam"][0].detach().cpu().numpy().astype(np.uint8)
                third = obs_env["policy"]["third_person_cam"][0].detach().cpu().numpy().astype(np.uint8)
                from PIL import Image
                wrist_up = np.array(Image.fromarray(wrist).resize((256, 256), Image.NEAREST))
                gif_frames.append(np.concatenate([wrist_up, third], axis=1))

            if bool(terminated[0]) and not bool(truncated[0]):
                success_this_ep = True
                break
            if bool(truncated[0]):
                truncated_this_ep = True
                break

        successes += int(success_this_ep)
        _log(f"    episode {ep} result: {'SUCCESS' if success_this_ep else 'FAIL'} after {step + 1} steps")

        # Emit per-episode structured record so the orchestrator can plot
        # success rate / cube drop patterns / episode length separately.
        if args_cli.jsonl_out:
            cube_end = obs_env["policy"]["cube_pos"][0].detach().cpu().tolist()
            result = "success" if success_this_ep else ("timeout" if truncated_this_ep else "fail")
            _append_jsonl(pathlib.Path(args_cli.jsonl_out), {
                "ts": dt.datetime.utcnow().isoformat() + "Z",
                "ckpt": args_cli.checkpoint,
                "ckpt_tag": args_cli.ckpt_tag,
                "ep": ep,
                "result": result,
                "steps": step + 1,
                "cube_start": [float(x) for x in cube0],
                "cube_end": [float(x) for x in cube_end],
            })

        if ep == 0 and args_cli.save_gif and gif_frames and not saved_gif:
            import imageio
            out = pathlib.Path(args_cli.save_gif)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(out), gif_frames, fps=15, loop=0)
            _log(f"    wrote gif {out} ({out.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")
            saved_gif = True

    _log(f"total: {successes}/{args_cli.num_episodes} success "
         f"({100.0 * successes / args_cli.num_episodes:.0f}%)")

    env.close()
    sim_app.close()
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
