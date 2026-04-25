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


def _build_features(state_dim: int, use_videos: bool, drop_cube_pos: bool = False) -> dict:
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
    features = {
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
    if not drop_cube_pos:
        features["observation.cube_pos"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        }
    return features


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


def _aggregate_actions(actions: np.ndarray, stride: int) -> np.ndarray:
    """Resample a (T, 7) action stream to a stride-decimated (T_new, 7) stream.

    Semantics: action[t] in the 60 Hz HDF5 is the 7D command applied FROM obs[t]
    to obs[t+1]. When we downsample observations by taking every stride-th frame,
    the resampled action[i] must be the command that moves the robot FROM
    obs_new[i]=obs[i*stride] TO obs_new[i+1]=obs[(i+1)*stride]. For this env
    (Isaac-PickCube-HC10DT-Robotiq-IK-Rel) dims 0-5 are small IK-relative
    pos/axis-angle deltas, so summing them across the stride window is the
    correct composition to first order (equivalent to applying them serially).
    Dim 6 is the gripper command ∈ {-1, 0, +1} — a latched intent, not a delta;
    we take the value at the last step of the window as the committed state.

    Partial trailing window (T % stride != 0) is dropped so every resampled
    action has a full stride-length backing in the original stream.
    """
    T = actions.shape[0]
    T_new = T // stride
    if T_new == 0:
        return actions[:0]
    deltas = actions[: T_new * stride, :6].reshape(T_new, stride, 6).sum(axis=1)
    gripper = actions[: T_new * stride, 6:7].reshape(T_new, stride, 1)[:, -1, :]
    return np.concatenate([deltas, gripper], axis=-1).astype(np.float32)


def _sanity_check_stride(
    demo_key: str,
    actions: np.ndarray,
    stride: int,
    aggregated: np.ndarray,
    tol: float = 1e-3,
) -> None:
    """Cheap offline check: sum of aggregated deltas must equal sum of original
    deltas over the retained prefix, up to float32 summation-order noise. The
    summation tree (stride-groups vs flat sum) differs, so exact equality is
    not expected; tolerance is set well above the float32 floor while still
    catching any real bug (e.g. reshape stride mixup).
    """
    T_new = aggregated.shape[0]
    orig = actions[: T_new * stride, :6].sum(axis=0)
    agg = aggregated[:, :6].sum(axis=0)
    err = float(np.max(np.abs(orig - agg)))
    assert err < tol, f"{demo_key}: action aggregation drift {err:.2e} > {tol}"


def convert(
    src_path: pathlib.Path,
    dst_root: pathlib.Path,
    repo_id: str,
    task: str,
    fps: int,
    use_videos: bool,
    max_episodes: int | None,
    stride: int,
    drop_cube_pos: bool = False,
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
        # Filter out placeholder entries (the scripted recorder writes a
        # post-reset stub for each attempt; only entries with both obs and
        # actions groups are real trajectories) AND degenerate short demos
        # (e.g. cube spawned on target → 1-frame "auto-success"). SmolVLA's
        # chunk_size is 50 so episodes shorter than that cannot meaningfully
        # be trained against.
        all_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        MIN_FRAMES = 50
        demo_keys = []
        skipped_placeholder = 0
        skipped_short = 0
        for k in all_keys:
            if "obs" not in data[k] or "actions" not in data[k]:
                skipped_placeholder += 1
                continue
            T = data[k]["actions"].shape[0]
            if T < MIN_FRAMES:
                skipped_short += 1
                continue
            demo_keys.append(k)
        if skipped_placeholder:
            _log(f"skipped {skipped_placeholder} placeholder entries (no obs/actions)")
        if skipped_short:
            _log(f"skipped {skipped_short} degenerate short demos (<{MIN_FRAMES} frames)")
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]
        _log(f"converting {len(demo_keys)} episodes from {src_path}")

        # Peek at first demo to determine state dim.
        sample = data[demo_keys[0]]
        state_dim = int(sample["obs"]["joint_pos"].shape[1])
        _log(f"state dim = {state_dim}")

        features = _build_features(state_dim, use_videos, drop_cube_pos=drop_cube_pos)
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
            actions_full = demo["actions"][...].astype(np.float32)
            T_full = actions_full.shape[0]

            state_full = demo["obs"]["joint_pos"][...].astype(np.float32)
            ee_pose_full = _build_ee_pose(demo)
            cube_pos_full = demo["obs"]["cube_pos"][...].astype(np.float32)
            wrist_full = demo["obs"]["wrist_cam"][...]
            third_full = demo["obs"]["third_person_cam"][...]

            assert (state_full.shape[0] == T_full == ee_pose_full.shape[0]
                    == wrist_full.shape[0] == third_full.shape[0])

            if stride > 1:
                actions = _aggregate_actions(actions_full, stride)
                _sanity_check_stride(key, actions_full, stride, actions)
                T = actions.shape[0]
                sel = np.arange(T) * stride
                state = state_full[sel]
                ee_pose = ee_pose_full[sel]
                cube_pos = cube_pos_full[sel]
                wrist = wrist_full[sel]
                third = third_full[sel]
            else:
                actions = actions_full
                T = T_full
                state, ee_pose, cube_pos = state_full, ee_pose_full, cube_pos_full
                wrist, third = wrist_full, third_full

            for t in range(T):
                frame = {
                    "observation.state": state[t],
                    "observation.ee_pose": ee_pose[t],
                    "observation.images.wrist": wrist[t],
                    "observation.images.third_person": third[t],
                    "action": actions[t],
                    "task": task,
                }
                if not drop_cube_pos:
                    frame["observation.cube_pos"] = cube_pos[t]
                ds.add_frame(frame)
            ds.save_episode()
            _log(f"  [{i + 1}/{len(demo_keys)}] saved {key} ({T} frames"
                 f"{f', stride={stride} from {T_full}' if stride > 1 else ''})")

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
                    default="pick up the cube and place it on the pink square")
    ap.add_argument("--drop_cube_pos", action="store_true", default=False,
                    help="Omit observation.cube_pos from the LeRobot feature "
                         "schema so the policy cannot regress from privileged "
                         "cube position and must learn visual grounding.")
    ap.add_argument("--fps", type=int, default=60,
                    help="FPS metadata to stamp on the LeRobot dataset. If "
                         "--stride>1, this should be the DOWNSAMPLED fps "
                         "(e.g. original 60 Hz / stride 4 → --fps 15).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Downsample the HDF5 time axis by this factor. For "
                         "each retained obs frame, the 6 IK-rel action deltas "
                         "are summed across the stride window and the gripper "
                         "value at the window's last step is used. Stride 4 "
                         "converts 60 Hz → 15 Hz natively.")
    ap.add_argument("--use_videos", action="store_true", default=False,
                    help="encode cameras as mp4 (needs ffmpeg + svt-av1). "
                         "Default off — stores PNGs, simpler but bigger.")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="convert only the first N episodes (debugging).")
    args = ap.parse_args()

    if args.stride < 1:
        ap.error("--stride must be >= 1")

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
        stride=args.stride,
        drop_cube_pos=args.drop_cube_pos,
    )


if __name__ == "__main__":
    raise SystemExit(main())
