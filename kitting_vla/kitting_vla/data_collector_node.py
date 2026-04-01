"""
data_collector_node.py — Records (image, joint_state, action) tuples at 10 Hz.

Runs alongside the existing task planner stack in Isaac Sim to generate
training data for the VLA model. Records synchronized observations and
computes delta-joint actions from consecutive states.

Data is saved as HDF5 files (one per episode), then converted to RLDS
format using convert_dataset.py.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from kitting_interfaces.msg import KitOrder
from sensor_msgs.msg import Image, JointState


@dataclass
class Timestep:
    """Single timestep of recorded data."""
    image: np.ndarray          # (256, 256, 3) uint8
    joint_state: np.ndarray    # (7,) float32 [j1..j6, gripper_width]
    action: np.ndarray         # (7,) float32 [dj1..dj6, gripper_cmd]
    language: str
    timestamp: float


class EpisodeRecorder:
    """Accumulates timesteps for a single episode and writes to HDF5."""

    def __init__(self, episode_id: int, output_dir: str, language: str):
        self.episode_id = episode_id
        self.output_dir = output_dir
        self.language = language
        self.timesteps: list[Timestep] = []
        self.success: bool = False

    def add(self, image: np.ndarray, joint_state: np.ndarray):
        """Add an observation. Action is computed later from consecutive states."""
        self.timesteps.append(Timestep(
            image=image,
            joint_state=joint_state,
            action=np.zeros(7, dtype=np.float32),  # filled in finalize
            language=self.language,
            timestamp=time.monotonic(),
        ))

    def finalize(self, success: bool):
        """Compute actions from consecutive states and save to HDF5."""
        self.success = success

        # Compute delta-joint actions from consecutive joint states
        for i in range(len(self.timesteps) - 1):
            curr = self.timesteps[i].joint_state
            nxt = self.timesteps[i + 1].joint_state
            # Delta joints for arm (first 6), gripper command from next state
            delta = nxt[:6] - curr[:6]
            # Gripper: 1.0 if open (width > 40mm), 0.0 if closed
            gripper_cmd = 1.0 if nxt[6] > 0.040 else 0.0
            self.timesteps[i].action = np.concatenate(
                [delta, [gripper_cmd]]).astype(np.float32)

        # Last timestep: zero action (episode end)
        if self.timesteps:
            self.timesteps[-1].action = np.zeros(7, dtype=np.float32)

        self._save_hdf5()

    def _save_hdf5(self):
        """Write episode to HDF5 file."""
        try:
            import h5py
        except ImportError:
            # Fallback: save as numpy archive
            self._save_npz()
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f'episode_{self.episode_id:06d}.hdf5')

        n = len(self.timesteps)
        if n == 0:
            return

        with h5py.File(path, 'w') as f:
            f.create_dataset('image',
                             data=np.stack([t.image for t in self.timesteps]),
                             dtype=np.uint8, chunks=True, compression='gzip')
            f.create_dataset('state',
                             data=np.stack([t.joint_state for t in self.timesteps]),
                             dtype=np.float32)
            f.create_dataset('action',
                             data=np.stack([t.action for t in self.timesteps]),
                             dtype=np.float32)
            f.attrs['language'] = self.language
            f.attrs['success'] = self.success
            f.attrs['episode_id'] = self.episode_id
            f.attrs['num_timesteps'] = n

    def _save_npz(self):
        """Fallback: save as compressed numpy archive."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f'episode_{self.episode_id:06d}.npz')

        n = len(self.timesteps)
        if n == 0:
            return

        np.savez_compressed(
            path,
            image=np.stack([t.image for t in self.timesteps]),
            state=np.stack([t.joint_state for t in self.timesteps]),
            action=np.stack([t.action for t in self.timesteps]),
            language=self.language,
            success=self.success,
        )


# Joint names in the order expected by the VLA model
JOINT_NAMES = [
    'joint_1_s', 'joint_2_l', 'joint_3_u',
    'joint_4_r', 'joint_5_b', 'joint_6_t',
]
GRIPPER_JOINT = 'finger_joint'

# Language prompt templates (cycled for variety)
PROMPTS = [
    "Pick the box and place it in slot {slot}",
    "Pick up the box from the bin",
    "Grab the box and put it in slot {slot}",
    "Pick the {sku} and place it in slot {slot}",
]


def _make_node():
    class DataCollectorNode(Node):
        def __init__(self):
            super().__init__('data_collector_node')

            self.declare_parameter('output_dir', './data/raw')
            self.declare_parameter('record_hz', 10.0)
            self.declare_parameter('hardware_type', 'mock')

            self._output_dir = self.get_parameter('output_dir').value
            record_hz = self.get_parameter('record_hz').value
            hw_type = self.get_parameter('hardware_type').value

            # State
            self._current_image: np.ndarray | None = None
            self._current_joints: np.ndarray | None = None  # (7,) with gripper
            self._recorder: EpisodeRecorder | None = None
            self._episode_count = self._count_existing_episodes()
            self._current_sku = 'box'
            self._current_slot = 0
            self._recording = False

            # Image subscription
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=1)
            image_topic = '/sim/image_raw' if hw_type == 'isaac' else '/camera/image_raw'
            self._image_sub = self.create_subscription(
                Image, image_topic, self._image_cb, sensor_qos)

            # Joint state subscription
            self._js_sub = self.create_subscription(
                JointState, '/joint_states', self._js_cb, 10)

            # Kit order subscription (triggers episode start)
            self._order_sub = self.create_subscription(
                KitOrder, '/kit_order', self._order_cb, 10)

            # Recording timer
            period = 1.0 / record_hz
            self._timer = self.create_timer(period, self._record_tick)

            self.get_logger().info(
                f'Data collector ready. Output: {self._output_dir}, '
                f'existing episodes: {self._episode_count}')

        def _count_existing_episodes(self) -> int:
            """Count existing episode files to resume numbering."""
            d = Path(self._output_dir)
            if not d.exists():
                return 0
            return len(list(d.glob('episode_*.hdf5')) +
                       list(d.glob('episode_*.npz')))

        def _image_cb(self, msg: Image):
            h, w = msg.height, msg.width
            if msg.encoding in ('rgb8', 'bgr8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                if msg.encoding == 'bgr8':
                    arr = arr[:, :, ::-1].copy()
            else:
                return

            # Resize to 256x256
            if arr.shape[:2] != (256, 256):
                from PIL import Image as PILImage
                pil = PILImage.fromarray(arr)
                pil = pil.resize((256, 256), PILImage.BILINEAR)
                arr = np.array(pil)
            self._current_image = arr

        def _js_cb(self, msg: JointState):
            joints = np.zeros(7, dtype=np.float32)
            names = list(msg.name)
            positions = list(msg.position)
            for i, jname in enumerate(JOINT_NAMES):
                if jname in names:
                    joints[i] = positions[names.index(jname)]
            if GRIPPER_JOINT in names:
                finger_pos = positions[names.index(GRIPPER_JOINT)]
                joints[6] = max(0.0, min(0.0415, finger_pos)) * 2.0
            self._current_joints = joints

        def _order_cb(self, msg: KitOrder):
            """New kit order → start recording a new episode."""
            if self._recorder is not None:
                # Finalize previous episode as failure
                self._recorder.finalize(success=False)
                self.get_logger().warn('Previous episode finalized as failure')

            self._current_sku = msg.sku_ids[0] if msg.sku_ids else 'box'
            self._current_slot = 0
            prompt_idx = self._episode_count % len(PROMPTS)
            prompt = PROMPTS[prompt_idx].format(
                slot=self._current_slot, sku=self._current_sku)

            self._recorder = EpisodeRecorder(
                episode_id=self._episode_count,
                output_dir=self._output_dir,
                language=prompt,
            )
            self._recording = True
            self.get_logger().info(
                f'Episode {self._episode_count} started: "{prompt}"')

        def _record_tick(self):
            """Record one timestep if actively recording."""
            if not self._recording or self._recorder is None:
                return
            if self._current_image is None or self._current_joints is None:
                return
            self._recorder.add(
                self._current_image.copy(),
                self._current_joints.copy())

        def finalize_episode(self, success: bool):
            """Called externally to mark episode complete."""
            if self._recorder is None:
                return
            self._recorder.finalize(success=success)
            self.get_logger().info(
                f'Episode {self._episode_count} saved '
                f'({len(self._recorder.timesteps)} steps, '
                f'success={success})')
            self._episode_count += 1
            self._recorder = None
            self._recording = False

    return DataCollectorNode()


def main(args=None):
    rclpy.init(args=args)
    node = _make_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Finalize any in-progress recording
        if hasattr(node, '_recorder') and node._recorder is not None:
            node.finalize_episode(success=False)
        node.destroy_node()
        rclpy.shutdown()
