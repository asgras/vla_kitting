"""
episode_manager.py — Orchestrates pick-place episodes for VLA inference.

Tracks which slot to fill next, detects episode completion,
and generates language prompts for the VLA model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class EpisodeState(Enum):
    IDLE = auto()
    PICKING = auto()
    DONE = auto()
    TIMED_OUT = auto()
    ABORTED = auto()


@dataclass
class EpisodeResult:
    slot_id: int
    state: EpisodeState
    duration: float = 0.0
    prompt: str = ''


class EpisodeManager:
    """Manages the sequence of pick-place episodes.

    Given a kit order (N items), generates language prompts and tracks
    which slot to fill. Detects episode completion via a heuristic:
    the arm returns near home with gripper open after having been away.
    """

    PROMPT_TEMPLATES = [
        "Pick the box and place it in slot {slot}",
        "Pick the {sku} and place it in slot {slot}",
        "Pick up the box from the bin",
        "Grab the box from the bin and put it in the tray",
    ]

    def __init__(
        self,
        total_picks: int = 9,
        sku: str = 'box',
        episode_timeout: float = 60.0,
    ):
        self.total_picks = total_picks
        self.sku = sku
        self.episode_timeout = episode_timeout

        self.current_slot: int = 0
        self.state: EpisodeState = EpisodeState.IDLE
        self._episode_start: float = 0.0
        self._left_home: bool = False
        self._had_grasp: bool = False
        self._results: list[EpisodeResult] = []

    @property
    def is_active(self) -> bool:
        return self.state == EpisodeState.PICKING

    @property
    def is_complete(self) -> bool:
        return self.current_slot >= self.total_picks

    @property
    def results(self) -> list[EpisodeResult]:
        return list(self._results)

    def get_prompt(self) -> str:
        """Get the language prompt for the current episode."""
        return f"Pick the box and place it in slot {self.current_slot}"

    def start_order(self, total_picks: int, sku: str = 'box') -> str:
        """Begin a new kit order. Returns the first episode prompt."""
        self.total_picks = total_picks
        self.sku = sku
        self.current_slot = 0
        self._results = []
        return self._start_episode()

    def _start_episode(self) -> str:
        """Start a single pick-place episode."""
        self.state = EpisodeState.PICKING
        self._episode_start = time.monotonic()
        self._left_home = False
        self._had_grasp = False
        return self.get_prompt()

    def tick(
        self,
        is_at_home: bool,
        gripper_open: bool,
        gripper_closed: bool,
    ) -> EpisodeState:
        """Called each control cycle to update episode state.

        Args:
            is_at_home: True if arm is near home position.
            gripper_open: True if gripper width > open threshold.
            gripper_closed: True if gripper width < closed threshold.

        Returns:
            Current episode state.
        """
        if self.state != EpisodeState.PICKING:
            return self.state

        elapsed = time.monotonic() - self._episode_start

        # Timeout check
        if elapsed > self.episode_timeout:
            self._finish_episode(EpisodeState.TIMED_OUT, elapsed)
            return self.state

        # Track if we've left home (arm moved away)
        if not is_at_home:
            self._left_home = True

        # Track if we've grasped something (gripper closed while away from home)
        if gripper_closed and self._left_home:
            self._had_grasp = True

        # Episode done: returned home with gripper open after having grasped
        if (is_at_home and gripper_open and
                self._left_home and self._had_grasp and elapsed > 3.0):
            self._finish_episode(EpisodeState.DONE, elapsed)

        return self.state

    def _finish_episode(self, state: EpisodeState, duration: float):
        """Complete the current episode and advance slot if successful."""
        result = EpisodeResult(
            slot_id=self.current_slot,
            state=state,
            duration=duration,
            prompt=self.get_prompt(),
        )
        self._results.append(result)

        if state == EpisodeState.DONE:
            self.current_slot += 1

        if self.current_slot >= self.total_picks:
            self.state = EpisodeState.IDLE
        else:
            # Auto-start next episode
            self._start_episode()

    def abort(self):
        """Abort the current episode."""
        if self.state == EpisodeState.PICKING:
            elapsed = time.monotonic() - self._episode_start
            self._finish_episode(EpisodeState.ABORTED, elapsed)
        self.state = EpisodeState.IDLE
