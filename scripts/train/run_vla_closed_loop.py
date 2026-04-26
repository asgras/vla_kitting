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
        --max_steps 450 \\
        --save_gif reports/vla_rollout.gif

--max_steps matches the env's policy rate × episode_length_s. At 15 Hz the
default is 450 (30 s). At 60 Hz (prior config) it was 1800.
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
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--task", type=str,
                    default="pick up the cube and place it on the pink square")
parser.add_argument("--save_gif", type=str, default=None,
                    help="Optional: write a side-by-side gif here. "
                         "If the path contains '{ep}', one gif is saved per "
                         "episode with the episode index substituted; "
                         "otherwise only the first episode's gif is saved.")
parser.add_argument("--jsonl_out", type=str, default=None,
                    help="Optional: append per-episode structured results here.")
parser.add_argument("--ckpt_tag", type=str, default=None,
                    help="Optional short tag stamped on every JSONL record "
                         "(e.g. 'epoch_0023').")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--cube_xy", type=str, default=None,
                    help="Override cube (x,y) in world-frame meters after each "
                         "env.reset() — used to rerun the policy from the same "
                         "cube start pose as a known training demo. One or more "
                         "comma-separated pairs joined by ';', cycled across "
                         "episodes: e.g. '0.479,-0.057;0.506,-0.104'. z=0.025.")
parser.add_argument("--zero_wrist_cam", action="store_true", default=False,
                    help="Replace wrist_cam observation with zeros at every "
                         "step (ablation: is the policy using wrist vision?).")
parser.add_argument("--zero_third_cam", action="store_true", default=False,
                    help="Replace third_person_cam observation with zeros "
                         "(ablation).")
parser.add_argument("--zero_cube_pos", action="store_true", default=False,
                    help="Replace cube_pos observation with zeros at every "
                         "step (ablation: is the policy using the privileged "
                         "cube_pos channel as a shortcut?).")
parser.add_argument("--drop_cube_pos", action="store_true", default=False,
                    help="Omit observation.cube_pos entirely from the frame "
                         "passed to the policy. Use when the policy was "
                         "trained on a dataset that does NOT include cube_pos "
                         "as an input feature. Different from --zero_cube_pos, "
                         "which keeps the feature but zeros its values.")
parser.add_argument("--gripper_threshold", type=float, default=None,
                    help="If set, apply sign-threshold to action[6] at "
                         "inference: action[6] = +1 if > threshold else -1. "
                         "Matches the env's BinaryJointPositionActionCfg "
                         "semantics (close on <0, open otherwise). Typical "
                         "value 0.0. Leave unset to pass policy output raw.")
parser.add_argument("--use_rtc", action="store_true", default=False,
                    help="Use predict_action_chunk-based inference (required "
                         "for RTC). When True the script predicts a chunk of "
                         "n_action_steps actions, executes them one at a "
                         "time, then re-queries. RTC's internal prefix-"
                         "attention mechanism is invoked inside "
                         "predict_action_chunk when rtc_config.enabled=True "
                         "in the saved policy config.")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[vla] {msg}", flush=True)


def _obs_env_to_lerobot(env_obs: dict,
                        zero_wrist_cam: bool = False,
                        zero_third_cam: bool = False,
                        zero_cube_pos: bool = False,
                        drop_cube_pos: bool = False) -> dict:
    """Remap env observation keys into the LeRobot feature names the policy
    was trained on, and convert CUDA tensors to CPU numpy. Handles both the
    Mimic env (eef_pos + eef_quat split) and the plain env (combined
    ee_pose) — the LeRobot dataset uses the 7D combined form either way.

    Ablation flags force specific observation channels to zero before the
    policy sees them.
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

    cube_pos = _np(p["cube_pos"]).astype(np.float32)
    wrist = _np(p["wrist_cam"]).astype(np.uint8)
    third = _np(p["third_person_cam"]).astype(np.uint8)

    if zero_cube_pos:
        cube_pos = np.zeros_like(cube_pos)
    if zero_wrist_cam:
        wrist = np.zeros_like(wrist)
    if zero_third_cam:
        third = np.zeros_like(third)

    out = {
        "observation.state": _np(p["joint_pos"]).astype(np.float32),
        "observation.ee_pose": ee_pose,
        "observation.images.wrist": wrist,
        "observation.images.third_person": third,
    }
    if not drop_cube_pos:
        out["observation.cube_pos"] = cube_pos
    return out


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

    per_episode_gif = bool(args_cli.save_gif and "{ep}" in args_cli.save_gif)
    gif_frames: list[np.ndarray] = []
    saved_gif = False
    successes = 0

    cube_xy_overrides: list[tuple[float, float]] = []
    if args_cli.cube_xy:
        for pair in args_cli.cube_xy.split(";"):
            xs = [float(v) for v in pair.split(",")]
            assert len(xs) == 2, "--cube_xy expects 'x,y' pairs joined by ';'"
            cube_xy_overrides.append((xs[0], xs[1]))

    for ep in range(args_cli.num_episodes):
        obs_env, _ = env.reset()
        policy.reset()

        if cube_xy_overrides:
            xy = cube_xy_overrides[ep % len(cube_xy_overrides)]
            sim_dev = env.unwrapped.sim.device
            origin = env.unwrapped.scene.env_origins[0]
            pose = torch.tensor([[
                xy[0] + origin[0].item(),
                xy[1] + origin[1].item(),
                0.025 + origin[2].item(),
                1.0, 0.0, 0.0, 0.0,
            ]], device=sim_dev)
            cube_rb = env.unwrapped.scene["cube"]
            cube_rb.write_root_pose_to_sim(pose)
            cube_rb.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_dev))
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
            obs_env = {"policy": env.unwrapped.observation_manager.compute()["policy"]}

        cube0 = obs_env["policy"]["cube_pos"][0].tolist()
        _log(f"=== episode {ep} | cube at ({cube0[0]:.3f}, {cube0[1]:.3f}, {cube0[2]:.3f})")

        success_this_ep = False
        truncated_this_ep = False
        # Chunk-based inference state (only used when --use_rtc).
        chunk_actions = None  # (n_action_steps, action_dim) tensor of post-processed actions
        chunk_idx = 0
        n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
        for step in range(args_cli.max_steps):
            # Build an inference frame matching the dataset schema, then
            # preprocess (normalize images, tokenize task string).
            raw = _obs_env_to_lerobot(
                obs_env,
                zero_wrist_cam=args_cli.zero_wrist_cam,
                zero_third_cam=args_cli.zero_third_cam,
                zero_cube_pos=args_cli.zero_cube_pos,
                drop_cube_pos=args_cli.drop_cube_pos,
            )
            frame = prepare_observation_for_inference(
                observation=raw, device=device, task=args_cli.task, robot_type="yaskawa_hc10dt_robotiq_2f85"
            )
            batch = preprocess(frame)

            if args_cli.use_rtc:
                # Re-query the policy every n_action_steps frames. RTC's
                # internal prefix-attention is engaged inside
                # predict_action_chunk when rtc_config.enabled is set on the
                # saved policy config.
                if chunk_actions is None or chunk_idx >= n_action_steps:
                    chunk = policy.predict_action_chunk(batch)
                    # chunk shape: (1, chunk_size, action_dim). Take first n_action_steps.
                    chunk = postprocess(chunk[:, :n_action_steps, :])
                    if not torch.is_tensor(chunk):
                        chunk = torch.as_tensor(chunk)
                    chunk_actions = chunk.view(n_action_steps, -1)
                    chunk_idx = 0
                action = chunk_actions[chunk_idx].view(1, -1)
                chunk_idx += 1
            else:
                # SmolVLA outputs a (1, action_dim) tensor; unnormalize via postprocess.
                action = policy.select_action(batch)
                action = postprocess(action)

            # The env expects a (1, 7) torch tensor on its own device.
            if not torch.is_tensor(action):
                action = torch.as_tensor(action)
            action = action.view(1, -1).to(env_cfg.sim.device)

            # Gripper sign-threshold: optionally snap action[6] to ±1 so the
            # policy's continuous output matches the env's binary wrapper
            # (BinaryJointPositionActionCfg thresholds at 0 internally, so
            # this is a crispness/logging aid more than a behavior change).
            if args_cli.gripper_threshold is not None:
                action[:, 6] = torch.where(
                    action[:, 6] > args_cli.gripper_threshold,
                    torch.ones_like(action[:, 6]),
                    -torch.ones_like(action[:, 6]),
                )

            obs_env, reward, terminated, truncated, info = env.step(action)

            # Collect frames for gif (first episode only unless template uses {ep}).
            if args_cli.save_gif and (per_episode_gif or ep == 0) and step % 4 == 0:
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

        if args_cli.save_gif and gif_frames and (per_episode_gif or (ep == 0 and not saved_gif)):
            import imageio
            gif_path = args_cli.save_gif.format(ep=ep) if per_episode_gif else args_cli.save_gif
            out = pathlib.Path(gif_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(out), gif_frames, fps=15, loop=0)
            _log(f"    wrote gif {out} ({out.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")
            saved_gif = True
            if per_episode_gif:
                gif_frames = []

    _log(f"total: {successes}/{args_cli.num_episodes} success "
         f"({100.0 * successes / args_cli.num_episodes:.0f}%)")

    env.close()
    sim_app.close()
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
