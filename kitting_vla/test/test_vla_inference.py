"""Tests for VLA inference core logic (no ROS, no Octo dependency)."""
import numpy as np
import pytest

from kitting_vla.safety_wrapper import SafetyConfig
from kitting_vla.vla_inference_node import VLAInferenceCore


@pytest.fixture
def config():
    return {
        'joint_names': [
            'joint_1_s', 'joint_2_l', 'joint_3_u',
            'joint_4_r', 'joint_5_b', 'joint_6_t',
        ],
        'home_joints': {
            'joint_1_s': 0.0, 'joint_2_l': 0.0, 'joint_3_u': 0.0,
            'joint_4_r': 0.0, 'joint_5_b': 1.5708, 'joint_6_t': 0.0,
        },
        'gripper': {
            'joint_name': 'finger_joint',
            'max_width': 0.083,
            'open_threshold': 0.5,
        },
        'safety': {
            'max_delta_joint': 0.05,
            'max_joint_velocity': 0.5,
            'workspace_bounds': {
                'x': [-0.2, 1.2],
                'y': [-1.0, 0.7],
                'z': [-0.02, 1.0],
            },
            'episode_timeout': 60.0,
        },
    }


@pytest.fixture
def core(config):
    return VLAInferenceCore(config, checkpoint_path='./fake_checkpoint')


class TestOnJointState:
    def test_parses_joint_state(self, core):
        names = ['joint_1_s', 'joint_2_l', 'joint_3_u',
                 'joint_4_r', 'joint_5_b', 'joint_6_t', 'finger_joint']
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02]
        core.on_joint_state(names, positions)

        np.testing.assert_allclose(
            core.current_joints, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert abs(core.current_gripper_width - 0.04) < 1e-6  # 0.02 * 2

    def test_missing_gripper(self, core):
        names = ['joint_1_s', 'joint_2_l', 'joint_3_u',
                 'joint_4_r', 'joint_5_b', 'joint_6_t']
        positions = [0.0] * 6
        core.on_joint_state(names, positions)
        assert core.current_gripper_width == 0.0


class TestOnImage:
    def test_stores_image(self, core):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        core.on_image(img)
        assert core.current_image is not None
        assert core.current_image.shape == (256, 256, 3)

    def test_resizes_large_image(self, core):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        core.on_image(img)
        assert core.current_image.shape == (256, 256, 3)


class TestOnKitOrder:
    def test_starts_episode(self, core):
        core.on_kit_order(['single_gang_box'], [3])
        assert core.episode.is_active
        assert core.episode.total_picks == 3

    def test_generates_prompt(self, core):
        core.on_kit_order(['round_box'], [1])
        prompt = core.episode.get_prompt()
        assert 'slot 0' in prompt


class TestStep:
    def test_returns_none_when_idle(self, core):
        target, gripper = core.step()
        assert target is None
        assert gripper is None

    def test_returns_none_without_observations(self, core):
        core.on_kit_order(['box'], [1])
        target, gripper = core.step()
        assert target is None
