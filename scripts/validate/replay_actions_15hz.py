"""Feed a recorded mimic demo's actions — aggregated from 60 Hz to 15 Hz —
into the live 15 Hz env (decimation=8), starting from the demo's recorded
cube pose. Purpose: test whether the 15 Hz action semantics at the current
env decimation actually reproduce the demo trajectory. If the replay doesn't
approximately reproduce the original pick, our conversion pipeline (or IK
rate interpretation) is broken independently of any policy.

Usage:
    ./isaaclab.sh -p scripts/validate/replay_actions_15hz.py \\
        --hdf5 datasets/mimic/cube_mimic_all.hdf5 \\
        --demo 0 \\
        --save_gif reports/replay_demo_000_15hz.gif
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", type=pathlib.Path, required=True)
parser.add_argument("--demo", type=int, default=0)
parser.add_argument("--stride", type=int, default=4,
                    help="Downsample ratio 60 Hz -> 15 Hz (=4).")
parser.add_argument("--action_mode", type=str, default="observed",
                    choices=["sum", "observed"],
                    help="'sum': aggregate recorded 60 Hz commands by summing "
                         "each window's 6 deltas (broken because the arm lags "
                         "the target and the sum double-counts that lag). "
                         "'observed': reconstruct action from the demo's "
                         "observed EE motion between frames t and t+stride — "
                         "self-consistent under use_relative_mode=True.")
parser.add_argument("--env_scale", type=float, default=0.1,
                    help="Env IK action scale. Matches envs/yaskawa_pick_cube_cfg.py.")
parser.add_argument("--action_boost", type=float, default=1.0,
                    help="Multiply the reconstructed action by this factor. The "
                         "IK only converges ~64%% of each commanded target delta "
                         "at decimation=8, so arm lags by ~36%%. Boosting actions "
                         "by ~1/0.64 overshoots per-call to cancel the lag.")
parser.add_argument("--save_gif", type=pathlib.Path, required=True)
parser.add_argument("--jsonl_out", type=pathlib.Path, default=None)
parser.add_argument("--device", type=str, default="cuda")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[replay15] {msg}", flush=True)


def _aggregate_actions(actions, stride):
    """Original sum-aggregation. Broken for use_relative_mode controllers — kept
    as the baseline the rest of the convert pipeline currently uses."""
    import numpy as np
    T = actions.shape[0]
    T_new = T // stride
    deltas = actions[: T_new * stride, :6].reshape(T_new, stride, 6).sum(axis=1)
    gripper = actions[: T_new * stride, 6:7].reshape(T_new, stride, 1)[:, -1, :]
    return np.concatenate([deltas, gripper], axis=-1).astype(np.float32)


def _quat_conj(q):
    import numpy as np
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(a, b):
    """Hamilton quaternion multiplication for (w, x, y, z) convention."""
    import numpy as np
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _quat_to_axis_angle(q):
    """(w,x,y,z) quaternion -> 3D axis-angle vector (angle * unit_axis).
    Picks the short-path (|angle| <= pi)."""
    import numpy as np
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Enforce w >= 0 for short path
    q = np.where(q[..., 0:1] < 0, -q, q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:4]
    angle = 2.0 * np.arccos(w)              # [0, pi] thanks to w >= 0
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = np.where(s[..., None] > 1e-8, xyz / np.maximum(s[..., None], 1e-8), np.zeros_like(xyz))
    return axis * angle[..., None]


def _reconstruct_actions_from_ee(ee_pos, ee_quat, gripper_col, stride, env_scale):
    """Reconstruct 15 Hz actions directly from the demo's observed EE trajectory.

    For each window [i*stride, (i+1)*stride), build an action that, when the env
    computes target = current_EE + action * env_scale, sets the target to
    ee_pose at the next 15 Hz frame. Under decimation=8 the IK then has enough
    physics ticks to converge.

    Args:
        ee_pos:   (T, 3) float positions.
        ee_quat:  (T, 4) float (w, x, y, z) orientations.
        gripper_col: (T, 1) or (T,) gripper command stream (from raw 60 Hz actions).
        stride:   downsampling factor (4 for 60->15 Hz).
        env_scale: env's IK action scale (default 0.1).
    """
    import numpy as np
    T = ee_pos.shape[0]
    # Need obs at idx + stride, so cap T_new accordingly.
    T_new = (T - 1) // stride
    if T_new <= 0:
        return np.zeros((0, 7), dtype=np.float32)

    idx = np.arange(T_new) * stride
    idx_next = idx + stride

    # Position delta (world frame), scaled so env.scale * action ≈ observed motion.
    dpos = (ee_pos[idx_next] - ee_pos[idx]) / env_scale

    # Orientation delta: q_rel = q_next * q_prev^{-1}; convert to axis-angle.
    q_rel = _quat_mul(ee_quat[idx_next], _quat_conj(ee_quat[idx]))
    drot = _quat_to_axis_angle(q_rel) / env_scale

    # Gripper: value at the END of the window (the intent we want held for 1/15 s).
    g = gripper_col.reshape(-1)[idx_next].reshape(-1, 1)

    return np.concatenate([dpos, drot, g], axis=-1).astype(np.float32)


def main() -> int:
    import datetime as dt
    import json
    import h5py
    import numpy as np
    import torch
    import gymnasium as gym
    from PIL import Image

    import envs  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg

    TASK_ID = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    device = torch.device(args_cli.device)

    # --- Load the demo, resample actions ---
    _log(f"reading demo_{args_cli.demo} from {args_cli.hdf5}")
    with h5py.File(args_cli.hdf5, "r") as f:
        d = f["data"][f"demo_{args_cli.demo}"]
        actions_60 = d["actions"][...].astype(np.float32)
        cube_start = d["obs"]["cube_pos"][0].astype(float).tolist()  # world frame
        ee_pos_60 = d["obs"]["eef_pos"][...].astype(np.float32)
        ee_quat_60 = d["obs"]["eef_quat"][...].astype(np.float32)

    if args_cli.action_mode == "sum":
        actions_15 = _aggregate_actions(actions_60, args_cli.stride)
    elif args_cli.action_mode == "observed":
        actions_15 = _reconstruct_actions_from_ee(
            ee_pos_60, ee_quat_60, actions_60[:, 6:7],
            args_cli.stride, args_cli.env_scale,
        )
    else:
        raise ValueError(args_cli.action_mode)

    if args_cli.action_boost != 1.0:
        # Boost pose dims only; keep gripper unchanged.
        actions_15[:, :6] *= args_cli.action_boost
        _log(f"  applied action_boost={args_cli.action_boost} to pose dims")
    ee_pos_15 = ee_pos_60[:: args_cli.stride][: actions_15.shape[0]]
    T = actions_15.shape[0]
    _log(f"action_mode={args_cli.action_mode}  {actions_60.shape[0]} x 60Hz -> {T} x 15Hz actions")
    _log(f"  action magnitudes: pos|{np.abs(actions_15[:,:3]).mean(axis=0)}  "
         f"rot|{np.abs(actions_15[:,3:6]).mean(axis=0)}  grip|{np.unique(actions_15[:,6])}")
    _log(f"cube start (world): {cube_start}")

    # --- Env setup ---
    _log(f"creating env {TASK_ID}")
    env_cfg = parse_env_cfg(TASK_ID, device=str(device), num_envs=1)
    env = gym.make(TASK_ID, cfg=env_cfg)

    obs_env, _ = env.reset()

    # Teleport cube to demo start pose
    sim_dev = env.unwrapped.sim.device
    origin = env.unwrapped.scene.env_origins[0]
    pose = torch.tensor([[
        cube_start[0] + origin[0].item(),
        cube_start[1] + origin[1].item(),
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

    # --- Step through the recorded actions ---
    gif_frames = []
    cube_track = []
    ee_track = []

    for t in range(T):
        action = torch.as_tensor(actions_15[t]).view(1, -1).to(sim_dev)
        obs_env, reward, terminated, truncated, info = env.step(action)

        # Collect frames at full 15 Hz (every step).
        wrist = obs_env["policy"]["wrist_cam"][0].detach().cpu().numpy().astype(np.uint8)
        third = obs_env["policy"]["third_person_cam"][0].detach().cpu().numpy().astype(np.uint8)
        wrist_up = np.array(Image.fromarray(wrist).resize((256, 256), Image.NEAREST))
        gif_frames.append(np.concatenate([wrist_up, third], axis=1))

        cube_p = obs_env["policy"]["cube_pos"][0].detach().cpu().numpy().tolist()
        # Non-Mimic env exposes 7D combined ee_pose; Mimic env exposes eef_pos.
        if "eef_pos" in obs_env["policy"]:
            ee_p = obs_env["policy"]["eef_pos"][0].detach().cpu().numpy().tolist()
        elif "ee_pose" in obs_env["policy"]:
            ee_p = obs_env["policy"]["ee_pose"][0, :3].detach().cpu().numpy().tolist()
        else:
            ee_p = None
        cube_track.append(cube_p)
        ee_track.append(ee_p)

        if bool(terminated[0]):
            _log(f"    terminated at step {t + 1}")
            break
        if bool(truncated[0]):
            _log(f"    truncated at step {t + 1}")
            break

    # --- Save outputs ---
    import imageio
    out_gif = args_cli.save_gif
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_gif), gif_frames, fps=15, loop=0)
    _log(f"wrote gif {out_gif} ({out_gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")

    # Compare trajectories
    cube_track = np.array(cube_track)
    ee_track = np.array([e if e is not None else [np.nan] * 3 for e in ee_track])
    # Peak z lift of the cube
    peak_cube_z = float(cube_track[:, 2].max())
    final_cube = cube_track[-1].tolist()
    start_cube = cube_track[0].tolist()
    # Demo reference — max z of ee_pos (best surrogate for lift height in demo)
    demo_peak_ee_z = float(ee_pos_15[:, 2].max())
    live_peak_ee_z = float(np.nanmax(ee_track[:, 2]))
    _log(f"cube z peak (live replay): {peak_cube_z:.4f}  (spawn = 0.025)")
    _log(f"cube  start {start_cube} -> final {final_cube}")
    _log(f"ee   peak z: demo-recorded = {demo_peak_ee_z:.4f}  live = {live_peak_ee_z:.4f}")
    # Net ee x/y: compare demo's net motion vs live's net motion
    if not np.isnan(ee_track).any():
        demo_net = (ee_pos_15[-1] - ee_pos_15[0]).tolist()
        live_net = (ee_track[-1] - ee_track[0]).tolist()
        _log(f"ee  net displacement: demo = {demo_net}  live = {live_net}")

    if args_cli.jsonl_out:
        with args_cli.jsonl_out.open("w") as f:
            for t_i, (c, e) in enumerate(zip(cube_track.tolist(), ee_track.tolist())):
                f.write(json.dumps({
                    "t": t_i, "cube_pos": c, "ee_pos": e,
                }) + "\n")
        _log(f"wrote {args_cli.jsonl_out}")

    env.close()
    sim_app.close()
    # Success if cube lifted meaningfully above spawn
    return 0 if peak_cube_z > 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
