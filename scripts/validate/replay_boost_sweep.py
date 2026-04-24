"""Boost-factor sweep of observed-EE action reconstruction in the 15 Hz env.

Runs demo_0 through the env multiple times with different action_boost values
and measures whether the cube lifts. Purpose: find a boost that cancels the
per-call IK lag.
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
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--env_scale", type=float, default=0.1)
parser.add_argument("--boosts", type=str, default="1.0,1.3,1.56,2.0,3.0")
parser.add_argument("--save_gif_prefix", type=pathlib.Path, required=True,
                    help="e.g. reports/replay_boost_ -> reports/replay_boost_<boost>.gif")
parser.add_argument("--jsonl_out", type=pathlib.Path, default=None)
parser.add_argument("--device", type=str, default="cuda")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def _log(msg: str) -> None:
    print(f"[boost_sweep] {msg}", flush=True)


def _reconstruct_actions_from_ee(ee_pos, ee_quat, gripper_col, stride, env_scale):
    import numpy as np

    def quat_conj(q):
        out = q.copy(); out[..., 1:] = -out[..., 1:]; return out

    def quat_mul(a, b):
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], axis=-1)

    def quat_to_axis_angle(q):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        q = np.where(q[..., 0:1] < 0, -q, q)
        w = np.clip(q[..., 0], -1.0, 1.0)
        xyz = q[..., 1:4]
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
        axis = np.where(s[..., None] > 1e-8, xyz / np.maximum(s[..., None], 1e-8), np.zeros_like(xyz))
        return axis * angle[..., None]

    T = ee_pos.shape[0]
    T_new = (T - 1) // stride
    idx = np.arange(T_new) * stride
    idx_next = idx + stride
    dpos = (ee_pos[idx_next] - ee_pos[idx]) / env_scale
    q_rel = quat_mul(ee_quat[idx_next], quat_conj(ee_quat[idx]))
    drot = quat_to_axis_angle(q_rel) / env_scale
    g = gripper_col.reshape(-1)[idx_next].reshape(-1, 1)
    return np.concatenate([dpos, drot, g], axis=-1).astype(np.float32)


def _run_one(env, cube_start_world, actions, boost, save_gif_path):
    import numpy as np
    import torch
    from PIL import Image
    import imageio

    sim_dev = env.unwrapped.sim.device
    obs_env, _ = env.reset()

    # Teleport cube to demo start pose
    origin = env.unwrapped.scene.env_origins[0]
    pose = torch.tensor([[
        cube_start_world[0] + origin[0].item(),
        cube_start_world[1] + origin[1].item(),
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

    # Boost pose dims (not gripper).
    boosted = actions.copy()
    boosted[:, :6] *= boost

    frames = []
    cube_track = []
    for t in range(boosted.shape[0]):
        a = torch.as_tensor(boosted[t]).view(1, -1).to(sim_dev)
        obs_env, _, terminated, truncated, _ = env.step(a)
        wrist = obs_env["policy"]["wrist_cam"][0].detach().cpu().numpy().astype(np.uint8)
        third = obs_env["policy"]["third_person_cam"][0].detach().cpu().numpy().astype(np.uint8)
        wrist_up = np.array(Image.fromarray(wrist).resize((256, 256), Image.NEAREST))
        frames.append(np.concatenate([wrist_up, third], axis=1))
        cube_track.append(obs_env["policy"]["cube_pos"][0].detach().cpu().numpy().tolist())
        if bool(terminated[0]) or bool(truncated[0]):
            break

    cube_arr = np.array(cube_track)
    peak_z = float(cube_arr[:, 2].max())
    final = cube_arr[-1].tolist()
    save_gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_gif_path), frames, fps=15, loop=0)
    _log(f"  boost={boost:.2f} peak_cube_z={peak_z:.4f}  final_cube={final}  gif={save_gif_path.name}")
    return peak_z, final


def main() -> int:
    import json
    import h5py
    import numpy as np
    import torch
    import gymnasium as gym

    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    TASK_ID = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    device = torch.device(args_cli.device)

    _log(f"reading demo_{args_cli.demo}")
    with h5py.File(args_cli.hdf5, "r") as f:
        d = f["data"][f"demo_{args_cli.demo}"]
        actions_60 = d["actions"][...].astype(np.float32)
        cube_start = d["obs"]["cube_pos"][0].astype(float).tolist()
        ee_pos_60 = d["obs"]["eef_pos"][...].astype(np.float32)
        ee_quat_60 = d["obs"]["eef_quat"][...].astype(np.float32)

    actions_15 = _reconstruct_actions_from_ee(
        ee_pos_60, ee_quat_60, actions_60[:, 6:7],
        args_cli.stride, args_cli.env_scale,
    )
    _log(f"reconstructed {actions_60.shape[0]} -> {actions_15.shape[0]} actions")

    _log(f"creating env {TASK_ID}")
    env_cfg = parse_env_cfg(TASK_ID, device=str(device), num_envs=1)
    env = gym.make(TASK_ID, cfg=env_cfg)

    boosts = [float(b) for b in args_cli.boosts.split(",")]
    _log(f"boost sweep: {boosts}")

    results = []
    for b in boosts:
        gif_path = pathlib.Path(f"{args_cli.save_gif_prefix}{b:g}.gif")
        peak_z, final = _run_one(env, cube_start, actions_15, b, gif_path)
        results.append({"boost": b, "peak_cube_z": peak_z, "final_cube": final})

    print()
    _log("=== summary ===")
    _log(f"cube spawn z = 0.025. peak_cube_z > 0.05 means cube lifted.")
    for r in results:
        ok = "PICKED" if r["peak_cube_z"] > 0.05 else "NO PICK"
        _log(f"  boost={r['boost']:.2f}: peak_z={r['peak_cube_z']:.4f}  {ok}")

    if args_cli.jsonl_out:
        with args_cli.jsonl_out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
