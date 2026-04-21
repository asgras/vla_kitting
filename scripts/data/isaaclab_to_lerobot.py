"""Convert an Isaac Lab Mimic / scripted HDF5 dataset into LeRobot v3 format.

Expected input (HDF5):
    /data                                  (group)
        .attrs["env_args"]                 (json string)
        /demo_K                            (group)
            /obs/joint_pos                 shape (T, J)
            /obs/joint_vel                 shape (T, J)
            /obs/eef_pos                   shape (T, 3)    # Mimic-annotated demos
            /obs/eef_quat                  shape (T, 4)    # Mimic-annotated demos
            /obs/gripper_pos               shape (T, 1)    # Mimic-annotated demos
            /obs/gripper_closed            shape (T, 1)
            /obs/cube_pos                  shape (T, 3)
            /obs/wrist_cam                 shape (T, 128, 128, 3) uint8
            /obs/third_person_cam          shape (T, 256, 256, 3) uint8
            /actions                       shape (T, 7)

Output (LeRobotDataset v3 directory):
    observation.state          -> joint_pos (12D) — arm 6 + gripper 6
    observation.ee_pose        -> eef_pos + eef_quat (7D) if present, else ee_pose
    observation.gripper        -> gripper_pos (1D) if present, else gripper_closed
    observation.cube_pos       -> cube_pos (3D) — privileged, useful as a side channel
    observation.images.wrist   -> wrist_cam (video)
    observation.images.third_person -> third_person_cam (video)
    action                     -> 7D IK-rel + gripper

Usage:
    # From Isaac Lab bundled Python (h5py + PIL already installed)
    /opt/IsaacSim/python.sh scripts/data/isaaclab_to_lerobot.py \\
        --input datasets/mimic/cube_mimic.hdf5 \\
        --output datasets/lerobot/cube_pick_v1 \\
        --repo_id vla_kitting/cube_pick_v1 \\
        --task "pick up the cube and place it on the green target"

By default this runs WITHOUT video encoding (use_videos=False) — saves as PNG
frames, which is simpler + doesn't require ffmpeg + svt-av1. Add --use_videos
to switch to mp4 output, which is what SmolVLA expects for production training.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np


def _log(msg: str) -> None:
    print(f"[isaaclab2lerobot] {msg}", flush=True)


def _build_features(state_dim: int, use_videos: bool) -> dict:
    """LeRobot feature schema for the HC10DT cube pick-place task."""
    # Joint-name labels for the 12D state vector. The URDF exposes the arm
    # (joint_1_s..joint_6_t) plus the 6 Robotiq gripper joints.
    state_names = [
        "joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t",
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]
    if state_dim != len(state_names):
        # Be forgiving if the asset evolves; fall back to numeric names.
        state_names = [f"q{i}" for i in range(state_dim)]

    action_names = [
        "ee_dx", "ee_dy", "ee_dz", "ee_drx", "ee_dry", "ee_drz", "gripper",
    ]

    cam_dtype = "video" if use_videos else "image"

    # Note: no separate observation.gripper feature — LeRobot treats shape=(1,)
    # features as scalar Value() which then breaks type validation in add_frame.
    # The gripper finger_joint angle is already included in observation.state.
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "observation.cube_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.images.wrist": {
            "dtype": cam_dtype,
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.third_person": {
            "dtype": cam_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": action_names,
        },
    }


def _read_obs(demo: h5py.Group, key: str, fallback: str | None = None) -> np.ndarray | None:
    """Read /obs/<key>; if missing, try /obs/<fallback>."""
    obs = demo["obs"]
    if key in obs:
        return obs[key][...]
    if fallback and fallback in obs:
        return obs[fallback][...]
    return None


def _build_ee_pose(demo: h5py.Group) -> np.ndarray:
    """Produce a (T, 7) ee_pose [x,y,z,qw,qx,qy,qz] array.

    Mimic-annotated demos have split eef_pos + eef_quat. Older non-Mimic demos
    have the combined `ee_pose` (7D). Use whichever is present.
    """
    if "eef_pos" in demo["obs"] and "eef_quat" in demo["obs"]:
        pos = demo["obs"]["eef_pos"][...]  # (T, 3)
        quat = demo["obs"]["eef_quat"][...]  # (T, 4) wxyz
        return np.concatenate([pos, quat], axis=-1).astype(np.float32)
    if "ee_pose" in demo["obs"]:
        return demo["obs"]["ee_pose"][...].astype(np.float32)
    raise KeyError("demo has neither (eef_pos + eef_quat) nor ee_pose")


def _build_gripper(demo: h5py.Group) -> np.ndarray:
    """(T, 1) finger-joint position. Falls back to gripper_closed binary."""
    if "gripper_pos" in demo["obs"]:
        return demo["obs"]["gripper_pos"][...].astype(np.float32).reshape(-1, 1)
    if "gripper_closed" in demo["obs"]:
        return demo["obs"]["gripper_closed"][...].astype(np.float32).reshape(-1, 1)
    raise KeyError("demo has neither gripper_pos nor gripper_closed")


def convert(
    src_path: pathlib.Path,
    dst_root: pathlib.Path,
    repo_id: str,
    task: str,
    fps: int,
    use_videos: bool,
    max_episodes: int | None,
) -> int:
    # Import here so the CLI can still print help without lerobot installed.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        _log("ERROR: lerobot not importable. Install with `pip install lerobot` or "
             "add /home/ubuntu/code/lerobot/src to PYTHONPATH.")
        return 2

    if dst_root.exists():
        _log(f"refusing to overwrite existing {dst_root}; delete it first")
        return 1

    with h5py.File(src_path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]
        _log(f"converting {len(demo_keys)} episodes from {src_path}")

        # Peek at first demo to determine state dim.
        sample = data[demo_keys[0]]
        state_dim = int(sample["obs"]["joint_pos"].shape[1])
        _log(f"state dim = {state_dim}")

        features = _build_features(state_dim, use_videos)
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(dst_root),
            robot_type="yaskawa_hc10dt_robotiq_2f85",
            use_videos=use_videos,
        )

        for i, key in enumerate(demo_keys):
            demo = data[key]
            actions = demo["actions"][...].astype(np.float32)
            T = actions.shape[0]

            state = demo["obs"]["joint_pos"][...].astype(np.float32)
            ee_pose = _build_ee_pose(demo)
            gripper = _build_gripper(demo)
            cube_pos = demo["obs"]["cube_pos"][...].astype(np.float32)
            wrist = demo["obs"]["wrist_cam"][...]
            third = demo["obs"]["third_person_cam"][...]

            assert state.shape[0] == T == ee_pose.shape[0] == wrist.shape[0] == third.shape[0]

            for t in range(T):
                ds.add_frame({
                    "observation.state": state[t],
                    "observation.ee_pose": ee_pose[t],
                    "observation.cube_pos": cube_pos[t],
                    "observation.images.wrist": wrist[t],
                    "observation.images.third_person": third[t],
                    "action": actions[t],
                    "task": task,
                })
            ds.save_episode()
            _log(f"  [{i + 1}/{len(demo_keys)}] saved {key} ({T} frames)")

    _log(f"done — wrote {len(demo_keys)} episodes to {dst_root}")
    return 0


def main() -> int:
    REPO = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", type=str, required=True,
                    help="HF-style repo id, e.g. vla_kitting/cube_pick_v1")
    ap.add_argument("--task", type=str,
                    default="pick up the cube and place it on the green target")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--use_videos", action="store_true", default=False,
                    help="encode cameras as mp4 (needs ffmpeg + svt-av1). "
                         "Default off — stores PNGs, simpler but bigger.")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="convert only the first N episodes (debugging).")
    args = ap.parse_args()

    # Allow running from /opt/IsaacSim/python.sh even if lerobot is only
    # available as a source clone.
    lerobot_src = pathlib.Path("/home/ubuntu/code/lerobot/src")
    if lerobot_src.exists() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    return convert(
        src_path=args.input,
        dst_root=args.output,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        use_videos=args.use_videos,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
