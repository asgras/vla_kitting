"""Tests for data collection — EpisodeRecorder and action computation."""
import os
import tempfile

import numpy as np
import pytest

from kitting_vla.data_collector_node import EpisodeRecorder


class TestEpisodeRecorder:
    def test_add_timesteps(self):
        rec = EpisodeRecorder(
            episode_id=0, output_dir='/tmp/test_data', language='test')
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        state = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.04], dtype=np.float32)
        rec.add(img, state)
        rec.add(img, state + 0.01)
        assert len(rec.timesteps) == 2

    def test_action_computation(self):
        """Actions should be delta between consecutive joint states."""
        rec = EpisodeRecorder(
            episode_id=0, output_dir='/tmp/test_data', language='test')
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        s0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)
        s1 = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04], dtype=np.float32)
        rec.add(img, s0)
        rec.add(img, s1)
        rec.finalize(success=True)

        # First action should be delta joints
        expected_deltas = s1[:6] - s0[:6]
        np.testing.assert_allclose(rec.timesteps[0].action[:6], expected_deltas)
        # Gripper cmd: s1 gripper_width=0.04 > 0.040 → 0.0 (edge case, exactly 0.04)
        assert rec.timesteps[0].action[6] == 0.0

        # Last action is zeros
        np.testing.assert_array_equal(rec.timesteps[-1].action, np.zeros(7))

    def test_gripper_open_action(self):
        """Gripper open (width > 40mm) → action[6] = 1.0."""
        rec = EpisodeRecorder(
            episode_id=0, output_dir='/tmp/test_data', language='test')
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        s0 = np.zeros(7, dtype=np.float32)
        s1 = np.zeros(7, dtype=np.float32)
        s1[6] = 0.083  # fully open
        rec.add(img, s0)
        rec.add(img, s1)
        rec.finalize(success=True)

        assert rec.timesteps[0].action[6] == 1.0

    def test_save_npz(self):
        """Test NPZ fallback saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = EpisodeRecorder(
                episode_id=42, output_dir=tmpdir, language='test prompt')
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            state = np.random.randn(7).astype(np.float32)
            rec.add(img, state)
            rec.add(img, state + 0.01)
            rec._save_npz()

            path = os.path.join(tmpdir, 'episode_000042.npz')
            assert os.path.exists(path)

            data = np.load(path, allow_pickle=True)
            assert data['image'].shape == (2, 256, 256, 3)
            assert data['state'].shape == (2, 7)

    def test_empty_episode(self):
        """Empty episode should not crash on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = EpisodeRecorder(
                episode_id=0, output_dir=tmpdir, language='test')
            rec.finalize(success=False)
            assert len(rec.timesteps) == 0
