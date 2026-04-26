"""Compare train-time vs eval-time observations at the same cube spawn position.

For demo 0 (cube at 0.465, 0.186), pull frame 0 from the LeRobot dataset and
the equivalent freshly-reset env observation. Run the env observation through
the same eval pipeline (`_obs_env_to_lerobot` + `prepare_observation_for_inference`)
that `run_vla_closed_loop.py` uses. Compare:
  - state, ee_pose: max abs diff
  - images: pixel-level diff (mean, max, % pixels off-by-1)
  - dump side-by-side PNGs and a diff-heatmap

If they differ in any way (resize interpolation, color order, normalization),
that's the train-test pipeline mismatch we suspect is causing the closed-loop
failure at training positions.
"""
from __future__ import annotations
import sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--cube_x", type=float, default=0.465)
ap.add_argument("--cube_y", type=float, default=0.186)
ap.add_argument("--out_dir", type=str,
                default=str(REPO / "reports/runs/vision_grounded_30hz_2026-04-24/sweep/obs_compare"))
args_cli, _ = ap.parse_known_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app


def main() -> int:
    import os
    os.makedirs(args_cli.out_dir, exist_ok=True)

    import numpy as np
    import torch
    import gymnasium as gym

    sys.path.insert(0, "/home/ubuntu/code/lerobot/src")
    import envs  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # ===== TRAIN-TIME: pull frame 0 from LeRobot dataset =====
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"[compare] loading LeRobot dataset...")
    ds = LeRobotDataset(
        repo_id="vla_kitting/cube_pick_v3_scripted",
        root="/home/ubuntu/vla_kitting/datasets/lerobot/cube_pick_v1",
    )

    # Frame 0 of episode 0
    frame_idx = 0
    sample = ds[frame_idx]
    print(f"[compare] dataset sample 0 keys: {list(sample.keys())}")
    train_state = sample["observation.state"]
    train_ee_pose = sample["observation.ee_pose"]
    train_wrist = sample["observation.images.wrist"]
    train_third = sample["observation.images.third_person"]

    def to_np(t):
        if hasattr(t, "numpy"): return t.detach().cpu().numpy()
        return np.asarray(t)

    train_state = to_np(train_state)
    train_ee_pose = to_np(train_ee_pose)
    train_wrist = to_np(train_wrist)
    train_third = to_np(train_third)

    print(f"[compare] TRAIN frame 0:")
    print(f"  state: shape={train_state.shape}, dtype={train_state.dtype}, "
          f"min={train_state.min():.3f}, max={train_state.max():.3f}")
    print(f"  ee_pose: {train_ee_pose}")
    print(f"  wrist: shape={train_wrist.shape}, dtype={train_wrist.dtype}, "
          f"min={float(train_wrist.min()):.3f}, max={float(train_wrist.max()):.3f}")
    print(f"  third: shape={train_third.shape}, dtype={train_third.dtype}, "
          f"min={float(train_third.min()):.3f}, max={float(train_third.max()):.3f}")

    # ===== EVAL-TIME: fresh env reset + first observation =====
    print(f"\n[compare] creating env, placing cube at ({args_cli.cube_x}, {args_cli.cube_y})...")
    TASK_ID = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    env_cfg = parse_env_cfg(TASK_ID, device="cuda", num_envs=1)
    env = gym.make(TASK_ID, cfg=env_cfg)
    obs_env, _ = env.reset()

    # Override cube position to match the training cube
    sim_dev = env.unwrapped.sim.device
    origin = env.unwrapped.scene.env_origins[0]
    pose = torch.tensor([[
        args_cli.cube_x + origin[0].item(),
        args_cli.cube_y + origin[1].item(),
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

    p = obs_env["policy"]
    eval_joint_pos = p["joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    eval_ee_pose = p["ee_pose"][0].detach().cpu().numpy().astype(np.float32)
    eval_wrist = p["wrist_cam"][0].detach().cpu().numpy().astype(np.uint8)
    eval_third = p["third_person_cam"][0].detach().cpu().numpy().astype(np.uint8)

    print(f"\n[compare] EVAL fresh frame:")
    print(f"  joint_pos: shape={eval_joint_pos.shape}, dtype={eval_joint_pos.dtype}, "
          f"min={eval_joint_pos.min():.3f}, max={eval_joint_pos.max():.3f}")
    print(f"  ee_pose: {eval_ee_pose}")
    print(f"  wrist: shape={eval_wrist.shape}, dtype={eval_wrist.dtype}, "
          f"min={int(eval_wrist.min())}, max={int(eval_wrist.max())}")
    print(f"  third: shape={eval_third.shape}, dtype={eval_third.dtype}, "
          f"min={int(eval_third.min())}, max={int(eval_third.max())}")

    # ===== NUMERICAL COMPARISON =====
    print("\n[compare] === NUMERICAL DIFFS ===")

    # State
    state_diff = np.abs(eval_joint_pos - train_state)
    print(f"\n  state:")
    print(f"    train: {train_state}")
    print(f"    eval : {eval_joint_pos}")
    print(f"    abs diff: max={float(state_diff.max()):.6f}, mean={float(state_diff.mean()):.6f}")

    # ee_pose
    ee_diff = np.abs(eval_ee_pose - train_ee_pose)
    print(f"\n  ee_pose:")
    print(f"    train: {train_ee_pose}")
    print(f"    eval : {eval_ee_pose}")
    print(f"    abs diff: max={float(ee_diff.max()):.6f}, mean={float(ee_diff.mean()):.6f}")

    # Images — compare carefully. Train images may be float [0,1] CHW (post-transform)
    # or uint8 HWC depending on the dataset transform setting. Eval images are uint8 HWC.
    def normalize_image_to_uint8_hwc(img):
        """Coerce to uint8 (H, W, 3) for fair comparison."""
        a = np.asarray(img)
        if a.dtype == np.uint8:
            if a.ndim == 3 and a.shape[0] == 3:  # CHW
                a = np.transpose(a, (1, 2, 0))
            return a
        # float [0,1] → uint8
        if a.dtype in (np.float32, np.float64):
            if a.ndim == 3 and a.shape[0] == 3:  # CHW
                a = np.transpose(a, (1, 2, 0))
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            return a
        return a.astype(np.uint8)

    train_wrist_u8 = normalize_image_to_uint8_hwc(train_wrist)
    train_third_u8 = normalize_image_to_uint8_hwc(train_third)

    print(f"\n  wrist: train shape post-norm={train_wrist_u8.shape}, eval shape={eval_wrist.shape}")
    if train_wrist_u8.shape == eval_wrist.shape:
        wrist_diff = np.abs(train_wrist_u8.astype(int) - eval_wrist.astype(int))
        print(f"    abs pixel diff: max={int(wrist_diff.max())}, mean={float(wrist_diff.mean()):.2f}")
        n_off = int((wrist_diff > 5).sum())
        n_total = wrist_diff.size
        print(f"    pixels with |diff|>5: {n_off}/{n_total} ({100*n_off/n_total:.2f}%)")
    else:
        print(f"    SHAPE MISMATCH — cannot directly compare")

    print(f"\n  third: train shape post-norm={train_third_u8.shape}, eval shape={eval_third.shape}")
    if train_third_u8.shape == eval_third.shape:
        third_diff = np.abs(train_third_u8.astype(int) - eval_third.astype(int))
        print(f"    abs pixel diff: max={int(third_diff.max())}, mean={float(third_diff.mean()):.2f}")
        n_off = int((third_diff > 5).sum())
        n_total = third_diff.size
        print(f"    pixels with |diff|>5: {n_off}/{n_total} ({100*n_off/n_total:.2f}%)")
    else:
        print(f"    SHAPE MISMATCH — cannot directly compare")

    # ===== SAVE PNGs FOR VISUAL REVIEW =====
    from PIL import Image
    out = pathlib.Path(args_cli.out_dir)

    Image.fromarray(train_wrist_u8).save(out / "train_wrist.png")
    Image.fromarray(eval_wrist).save(out / "eval_wrist.png")
    Image.fromarray(train_third_u8).save(out / "train_third.png")
    Image.fromarray(eval_third).save(out / "eval_third.png")

    if train_wrist_u8.shape == eval_wrist.shape:
        wd = np.abs(train_wrist_u8.astype(int) - eval_wrist.astype(int)).astype(np.uint8)
        # Boost small diffs to be visible (multiply by 5, clip)
        wd_vis = np.clip(wd * 5, 0, 255).astype(np.uint8)
        Image.fromarray(wd_vis).save(out / "wrist_diff_x5.png")
    if train_third_u8.shape == eval_third.shape:
        td = np.abs(train_third_u8.astype(int) - eval_third.astype(int)).astype(np.uint8)
        td_vis = np.clip(td * 5, 0, 255).astype(np.uint8)
        Image.fromarray(td_vis).save(out / "third_diff_x5.png")

    print(f"\n[compare] saved visual comparison to {out}/")
    print("  train_wrist.png, eval_wrist.png, wrist_diff_x5.png")
    print("  train_third.png, eval_third.png, third_diff_x5.png")

    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
