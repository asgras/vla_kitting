"""
vla_inference_node.py — ROS2 node: RGB image + joints → joint actions.

Replaces the entire task_planner + perception + move_group stack with a
single VLA inference loop running at 10 Hz. Uses Octo-Small (27M params)
to predict action chunks (delta joint positions + gripper command).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from kitting_interfaces.msg import KitOrder
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from kitting_vla.episode_manager import EpisodeManager, EpisodeState
from kitting_vla.safety_wrapper import SafetyConfig, SafetyWrapper


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class VLAPolicy:
    """Wrapper around the Octo model for inference.

    Handles model loading, observation history, and action prediction.
    Separated from ROS for testability.
    """

    def __init__(self, checkpoint_path: str, pred_horizon: int = 4):
        self.pred_horizon = pred_horizon
        self._model = None
        self._task = None
        self._obs_history: list[dict] = []
        self._checkpoint_path = checkpoint_path
        self._rng_key = None

    def load(self):
        """Load the fine-tuned Octo model. Call once at startup."""
        try:
            import jax
            from octo.model.octo_model import OctoModel

            self._model = OctoModel.load_pretrained(self._checkpoint_path)
            self._rng_key = jax.random.PRNGKey(0)
        except ImportError:
            raise RuntimeError(
                "Octo not installed. Install with: "
                "pip install octo-model jax jaxlib")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def set_task(self, prompt: str):
        """Update the language task for the current episode."""
        if self._model is not None:
            self._task = self._model.create_tasks(texts=[prompt])

    def reset(self):
        """Clear observation history for a new episode."""
        self._obs_history = []

    def predict(
        self,
        image: np.ndarray,
        joint_state: np.ndarray,
    ) -> np.ndarray | None:
        """Predict action chunk from current observation.

        Args:
            image: (256, 256, 3) uint8 RGB image.
            joint_state: (7,) float32 [j1..j6, gripper_width].

        Returns:
            (7,) float32 first action [dj1..dj6, gripper_cmd], or None.
        """
        if self._model is None or self._task is None:
            return None

        import jax

        obs = {
            "image_primary": image[None],         # (1, 256, 256, 3)
            "proprio": joint_state[None],         # (1, 7)
        }
        self._obs_history.append(obs)
        if len(self._obs_history) > 2:
            self._obs_history.pop(0)

        # Stack observation history
        stacked = jax.tree.map(
            lambda *xs: np.concatenate(xs, axis=0),
            *self._obs_history)

        # Predict action chunk: (1, H, 7)
        self._rng_key, subkey = jax.random.split(self._rng_key)
        action_chunk = self._model.sample_actions(
            stacked, self._task, rng=subkey)

        # Return first action from chunk (receding horizon)
        return np.array(action_chunk[0, 0])  # (7,)


class VLAInferenceCore:
    """VLA inference logic, separated from ROS for testability.

    Wired up by VLAInferenceNode to ROS subscriptions and action clients.
    """

    def __init__(self, config: dict, checkpoint_path: str):
        self.cfg = config
        self.joint_names = config['joint_names']
        self.home_joints = np.array([
            config['home_joints'][j] for j in self.joint_names],
            dtype=np.float64)

        gripper_cfg = config['gripper']
        self.gripper_joint = gripper_cfg['joint_name']
        self.gripper_max_width = gripper_cfg['max_width']
        self.gripper_open_threshold = gripper_cfg['open_threshold']

        safety_cfg = SafetyConfig.from_config(config)
        self.safety = SafetyWrapper(safety_cfg)
        self.episode = EpisodeManager(
            episode_timeout=safety_cfg.episode_timeout)

        self.policy = VLAPolicy(
            checkpoint_path=checkpoint_path,
            pred_horizon=config.get('model', {}).get('pred_horizon', 4))

        # State
        self.current_image: np.ndarray | None = None
        self.current_joints: np.ndarray | None = None  # (6,) arm joints
        self.current_gripper_width: float = 0.0
        self._gripper_state: str = 'unknown'  # 'open', 'closed', 'unknown'

    def on_image(self, image: np.ndarray):
        """Process incoming RGB image (resized to 256x256)."""
        from PIL import Image as PILImage
        if image.shape[:2] != (256, 256):
            pil = PILImage.fromarray(image)
            pil = pil.resize((256, 256), PILImage.BILINEAR)
            image = np.array(pil)
        self.current_image = image

    def on_joint_state(self, names: list[str], positions: list[float]):
        """Process incoming joint state message."""
        joints = np.zeros(6, dtype=np.float64)
        for i, jname in enumerate(self.joint_names):
            if jname in names:
                idx = names.index(jname)
                joints[i] = positions[idx]
        self.current_joints = joints

        if self.gripper_joint in names:
            idx = names.index(self.gripper_joint)
            finger_pos = positions[idx]
            self.current_gripper_width = max(0.0, min(0.0415, finger_pos)) * 2.0

    def on_kit_order(self, sku_ids: list[str], quantities: list[int]):
        """Handle incoming kit order."""
        total = sum(quantities)
        sku = sku_ids[0] if sku_ids else 'box'
        prompt = self.episode.start_order(total, sku)
        self.policy.reset()
        self.policy.set_task(prompt)

    def step(self) -> tuple[np.ndarray | None, float | None]:
        """Run one control step. Returns (target_joints, gripper_width) or (None, None).

        Called at 10 Hz by the ROS timer.
        """
        if not self.episode.is_active:
            return None, None
        if self.current_image is None or self.current_joints is None:
            return None, None

        # Build state vector: [j1..j6, gripper_width]
        state = np.concatenate([
            self.current_joints,
            [self.current_gripper_width],
        ]).astype(np.float32)

        # Predict action
        action = self.policy.predict(self.current_image, state)
        if action is None:
            return None, None

        joint_deltas = action[:6]
        gripper_cmd = float(action[6])

        # Safety check and clip
        result = self.safety.check_and_clip(
            joint_deltas, self.current_joints, control_hz=10.0)

        target_joints = self.current_joints + result.clipped_deltas

        # Gripper command: binary open/close
        gripper_open = gripper_cmd > self.gripper_open_threshold
        if gripper_open:
            gripper_width = self.gripper_max_width
            self._gripper_state = 'open'
        else:
            gripper_width = 0.0
            self._gripper_state = 'closed'

        # Update episode state
        at_home = self.safety.is_at_home(self.current_joints, self.home_joints)
        ep_state = self.episode.tick(
            is_at_home=at_home,
            gripper_open=(self._gripper_state == 'open'),
            gripper_closed=(self._gripper_state == 'closed'),
        )

        # If episode finished, update prompt for next episode
        if ep_state == EpisodeState.PICKING:
            self.policy.set_task(self.episode.get_prompt())
        elif self.episode.is_active:
            self.policy.reset()
            self.policy.set_task(self.episode.get_prompt())

        return target_joints, gripper_width


def _make_node():
    """Create and return the VLA inference ROS2 node."""

    class VLAInferenceNode(Node):
        def __init__(self):
            super().__init__('vla_inference_node')

            # Parameters
            self.declare_parameter('cell_config', '')
            self.declare_parameter('checkpoint_path', './checkpoints/kitting_octo_small')
            self.declare_parameter('hardware_type', 'mock')

            config_path = self.get_parameter('cell_config').value
            checkpoint = self.get_parameter('checkpoint_path').value
            self._hw_type = self.get_parameter('hardware_type').value

            if not config_path:
                from ament_index_python.packages import get_package_share_directory
                config_path = str(Path(
                    get_package_share_directory('kitting_vla'),
                    'config', 'vla_cell_config.yaml'))

            config = _load_config(config_path)
            self._core = VLAInferenceCore(config, checkpoint)

            # Try to load the model (may fail if Octo not installed)
            try:
                self._core.policy.load()
                self.get_logger().info('Octo model loaded successfully')
            except RuntimeError as e:
                self.get_logger().warn(
                    f'Model not loaded: {e}. Node will run but not predict.')

            cb_group = ReentrantCallbackGroup()

            # Camera image subscription
            image_topic = (config['camera']['topic_sim']
                           if self._hw_type == 'isaac'
                           else config['camera']['topic_real'])
            self._image_sub = self.create_subscription(
                Image, image_topic, self._image_cb, 10)

            # Joint state subscription
            self._js_sub = self.create_subscription(
                JointState, '/joint_states', self._js_cb, 10)

            # Kit order subscription
            self._order_sub = self.create_subscription(
                KitOrder, '/kit_order', self._order_cb, 10)

            # Arm trajectory action client
            self._arm_client = ActionClient(
                self, FollowJointTrajectory,
                '/hc10dt_arm_controller/follow_joint_trajectory',
                callback_group=cb_group)

            # Gripper trajectory action client
            self._gripper_client = ActionClient(
                self, FollowJointTrajectory,
                '/gripper_controller/follow_joint_trajectory',
                callback_group=cb_group)

            self._joint_names = config['joint_names']
            self._last_gripper_width: float | None = None

            # Control loop at 10 Hz
            self._timer = self.create_timer(0.1, self._control_loop)

            self.get_logger().info('VLA inference node ready')

        def _image_cb(self, msg: Image):
            """Convert ROS Image to numpy array."""
            h, w = msg.height, msg.width
            if msg.encoding == 'rgb8':
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            elif msg.encoding == 'bgr8':
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                image = bgr[:, :, ::-1].copy()  # BGR → RGB
            else:
                # Try cv_bridge as fallback
                try:
                    from cv_bridge import CvBridge
                    bridge = CvBridge()
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    image = np.array(cv_img)
                except Exception:
                    self.get_logger().warn(
                        f'Unsupported image encoding: {msg.encoding}',
                        throttle_duration_sec=5.0)
                    return
            self._core.on_image(image)

        def _js_cb(self, msg: JointState):
            self._core.on_joint_state(list(msg.name), list(msg.position))

        def _order_cb(self, msg: KitOrder):
            self.get_logger().info(
                f'Kit order received: {msg.sku_ids} x {msg.quantities}')
            self._core.on_kit_order(list(msg.sku_ids), list(msg.quantities))

        def _control_loop(self):
            target_joints, gripper_width = self._core.step()
            if target_joints is None:
                return

            # Send arm command
            self._send_arm_command(target_joints)

            # Send gripper command if changed
            if (self._last_gripper_width is None or
                    abs(gripper_width - self._last_gripper_width) > 0.01):
                self._send_gripper_command(gripper_width)
                self._last_gripper_width = gripper_width

        def _send_arm_command(self, target_joints: np.ndarray):
            """Send joint trajectory goal to arm controller."""
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self._joint_names
            pt = JointTrajectoryPoint()
            pt.positions = target_joints.tolist()
            pt.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms
            traj.points.append(pt)
            goal.trajectory = traj

            if self._arm_client.server_is_ready():
                self._arm_client.send_goal_async(goal)

        def _send_gripper_command(self, width: float):
            """Send gripper width command."""
            finger_pos = max(0.0, min(0.0415, width / 2.0))
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = ['finger_joint', 'finger_joint_mimic']
            pt = JointTrajectoryPoint()
            pt.positions = [finger_pos, -finger_pos]
            pt.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 500ms
            traj.points.append(pt)
            goal.trajectory = traj

            if self._gripper_client.server_is_ready():
                self._gripper_client.send_goal_async(goal)

    return VLAInferenceNode()


def main(args=None):
    rclpy.init(args=args)
    node = _make_node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
