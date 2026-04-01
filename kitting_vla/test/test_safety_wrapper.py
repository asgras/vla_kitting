"""Tests for safety_wrapper.py — joint limits, workspace bounds, clipping."""
import numpy as np
import pytest

from kitting_vla.safety_wrapper import SafetyConfig, SafetyResult, SafetyWrapper


@pytest.fixture
def wrapper():
    return SafetyWrapper(SafetyConfig())


@pytest.fixture
def home_joints():
    return np.array([0.0, 0.0, 0.0, 0.0, 1.5708, 0.0])


class TestDeltaClipping:
    def test_small_deltas_pass_through(self, wrapper):
        deltas = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03])
        current = np.zeros(6)
        result = wrapper.check_and_clip(deltas, current)
        assert result.safe
        np.testing.assert_allclose(result.clipped_deltas, deltas)

    def test_large_deltas_clipped(self, wrapper):
        deltas = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
        current = np.zeros(6)
        result = wrapper.check_and_clip(deltas, current)
        assert result.safe
        max_d = wrapper.cfg.max_delta_joint
        np.testing.assert_array_less(np.abs(result.clipped_deltas),
                                     max_d + 1e-9)
        assert 'delta_clipped' in result.reason

    def test_zero_deltas(self, wrapper):
        result = wrapper.check_and_clip(np.zeros(6), np.zeros(6))
        assert result.safe
        np.testing.assert_array_equal(result.clipped_deltas, np.zeros(6))
        assert result.reason == ''


class TestJointPositionLimits:
    def test_at_lower_limit(self, wrapper):
        # joint_2_l lower limit is -1.92
        current = np.array([0, -1.90, 0, 0, 0, 0])
        deltas = np.array([0, -0.05, 0, 0, 0, 0])
        result = wrapper.check_and_clip(deltas, current)
        # Target would be -1.95, clipped to -1.92
        target = current + result.clipped_deltas
        assert target[1] >= -1.92 - 1e-9

    def test_at_upper_limit(self, wrapper):
        # joint_2_l upper limit is 2.27
        current = np.array([0, 2.25, 0, 0, 0, 0])
        deltas = np.array([0, 0.05, 0, 0, 0, 0])
        result = wrapper.check_and_clip(deltas, current)
        target = current + result.clipped_deltas
        assert target[1] <= 2.27 + 1e-9

    def test_within_limits_unchanged(self, wrapper):
        current = np.array([0, 0, 0, 0, 1.0, 0])
        deltas = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = wrapper.check_and_clip(deltas, current)
        np.testing.assert_allclose(result.clipped_deltas, deltas)


class TestWorkspaceBounds:
    def test_tcp_in_bounds(self, wrapper):
        result = wrapper.check_and_clip(
            np.zeros(6), np.zeros(6),
            tcp_position=np.array([0.5, 0.0, 0.5]))
        assert result.safe

    def test_tcp_below_table(self, wrapper):
        result = wrapper.check_and_clip(
            np.array([0.01] * 6), np.zeros(6),
            tcp_position=np.array([0.5, 0.0, -0.05]))
        assert not result.safe
        assert 'workspace_z' in result.reason
        np.testing.assert_array_equal(result.clipped_deltas, np.zeros(6))

    def test_tcp_out_of_x_range(self, wrapper):
        result = wrapper.check_and_clip(
            np.array([0.01] * 6), np.zeros(6),
            tcp_position=np.array([2.0, 0.0, 0.5]))
        assert not result.safe
        assert 'workspace_x' in result.reason

    def test_no_tcp_skips_workspace_check(self, wrapper):
        result = wrapper.check_and_clip(
            np.array([0.01] * 6), np.zeros(6))
        assert result.safe


class TestVelocityLimits:
    def test_velocity_within_limits(self, wrapper):
        # At 10 Hz, delta 0.05 → velocity 0.5 rad/s (within 2.36 limit)
        deltas = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        result = wrapper.check_and_clip(deltas, np.zeros(6), control_hz=10.0)
        assert result.safe

    def test_velocity_clipped_at_high_hz(self, wrapper):
        # At 100 Hz, delta 0.05 → velocity 5.0 rad/s (exceeds 2.36 limit)
        deltas = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        result = wrapper.check_and_clip(deltas, np.zeros(6), control_hz=100.0)
        # Should be clipped
        velocities = np.abs(result.clipped_deltas) * 100.0
        for i, vel in enumerate(velocities[:5]):
            assert vel <= 2.36 + 1e-6


class TestIsAtHome:
    def test_at_home(self, wrapper, home_joints):
        assert wrapper.is_at_home(home_joints, home_joints)

    def test_away_from_home(self, wrapper, home_joints):
        current = home_joints + 0.5
        assert not wrapper.is_at_home(current, home_joints)

    def test_near_home(self, wrapper, home_joints):
        current = home_joints + 0.1
        assert wrapper.is_at_home(current, home_joints, tolerance=0.15)


class TestSafetyConfigFromYaml:
    def test_from_config(self):
        cfg = {
            'safety': {
                'max_delta_joint': 0.03,
                'max_joint_velocity': 0.3,
                'workspace_bounds': {
                    'x': [-0.1, 1.0],
                    'y': [-0.5, 0.5],
                    'z': [0.0, 0.8],
                },
                'episode_timeout': 45.0,
                'joint_limits': [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            }
        }
        sc = SafetyConfig.from_config(cfg)
        assert sc.max_delta_joint == 0.03
        assert sc.workspace_z == (0.0, 0.8)
        assert sc.episode_timeout == 45.0

    def test_defaults(self):
        sc = SafetyConfig.from_config({})
        assert sc.max_delta_joint == 0.05
        assert sc.episode_timeout == 60.0
