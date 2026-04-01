"""Tests for episode_manager.py — episode orchestration and state tracking."""
import time
from unittest.mock import patch

import pytest

from kitting_vla.episode_manager import EpisodeManager, EpisodeState


@pytest.fixture
def manager():
    return EpisodeManager(total_picks=3, sku='single_gang_box', episode_timeout=60.0)


class TestStartOrder:
    def test_start_returns_prompt(self, manager):
        prompt = manager.start_order(3, 'single_gang_box')
        assert 'slot 0' in prompt
        assert manager.is_active
        assert manager.current_slot == 0

    def test_start_resets_state(self, manager):
        manager.current_slot = 5
        manager.start_order(2, 'round_box')
        assert manager.current_slot == 0
        assert manager.total_picks == 2


class TestGetPrompt:
    def test_prompt_includes_slot(self, manager):
        manager.start_order(3)
        assert 'slot 0' in manager.get_prompt()


class TestEpisodeTick:
    def test_idle_stays_idle(self, manager):
        state = manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)
        assert state == EpisodeState.IDLE

    def test_picking_stays_picking_initially(self, manager):
        manager.start_order(3)
        state = manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)
        assert state == EpisodeState.PICKING

    def test_full_episode_cycle(self, manager):
        """Simulate: leave home → close gripper → return home → done."""
        manager.start_order(3)

        # Still at home, hasn't left yet
        manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)
        assert manager.state == EpisodeState.PICKING

        # Leave home
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        assert manager._left_home

        # Close gripper (grasping)
        manager.tick(is_at_home=False, gripper_open=False, gripper_closed=True)
        assert manager._had_grasp

        # Move to place
        manager.tick(is_at_home=False, gripper_open=False, gripper_closed=True)

        # Open gripper (release)
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)

        # Return home with gripper open — but need elapsed > 3s
        with patch('kitting_vla.episode_manager.time') as mock_time:
            mock_time.monotonic.return_value = manager._episode_start + 5.0
            manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)

        # Should have advanced to next slot
        assert manager.current_slot == 1
        assert manager.state == EpisodeState.PICKING

    def test_timeout(self, manager):
        manager.start_order(3)
        manager._episode_start = time.monotonic() - 61.0  # expired
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        # Timeout finishes the episode and auto-starts next one
        # The timed-out result should be in results
        assert len(manager.results) == 1
        assert manager.results[0].state == EpisodeState.TIMED_OUT
        # Slot does NOT advance on timeout
        assert manager.results[0].slot_id == 0

    def test_no_completion_without_grasp(self, manager):
        """Returning home without having grasped doesn't complete episode."""
        manager.start_order(3)
        # Leave home
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        # Return home without grasping (gripper open, never closed)
        with patch('kitting_vla.episode_manager.time') as mock_time:
            mock_time.monotonic.return_value = manager._episode_start + 5.0
            manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)
        # Still picking — haven't grasped
        assert manager.state == EpisodeState.PICKING
        assert manager.current_slot == 0


class TestCompletion:
    def test_all_picks_done(self, manager):
        manager.start_order(1)

        # Simulate full cycle
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        manager.tick(is_at_home=False, gripper_open=False, gripper_closed=True)

        with patch('kitting_vla.episode_manager.time') as mock_time:
            mock_time.monotonic.return_value = manager._episode_start + 5.0
            manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)

        assert manager.current_slot == 1
        assert manager.is_complete
        assert not manager.is_active


class TestAbort:
    def test_abort_during_pick(self, manager):
        manager.start_order(3)
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        manager.abort()
        assert manager.state == EpisodeState.IDLE
        assert len(manager.results) == 1
        assert manager.results[0].state == EpisodeState.ABORTED

    def test_abort_when_idle(self, manager):
        manager.abort()
        assert manager.state == EpisodeState.IDLE


class TestResults:
    def test_results_tracked(self, manager):
        manager.start_order(1)
        manager.tick(is_at_home=False, gripper_open=True, gripper_closed=False)
        manager.tick(is_at_home=False, gripper_open=False, gripper_closed=True)

        with patch('kitting_vla.episode_manager.time') as mock_time:
            mock_time.monotonic.return_value = manager._episode_start + 5.0
            manager.tick(is_at_home=True, gripper_open=True, gripper_closed=False)

        results = manager.results
        assert len(results) == 1
        assert results[0].slot_id == 0
        assert results[0].state == EpisodeState.DONE
