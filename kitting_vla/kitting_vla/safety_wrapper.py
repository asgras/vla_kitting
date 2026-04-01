"""
safety_wrapper.py — Joint limits, workspace bounds, and e-stop logic.

Thin safety layer that clips VLA actions and checks workspace bounds.
This is the only "traditional robotics" code in the VLA pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SafetyConfig:
    """Safety parameters loaded from vla_cell_config.yaml."""
    max_delta_joint: float = 0.05          # rad per step
    max_joint_velocity: float = 0.5        # rad/s
    workspace_x: tuple[float, float] = (-0.2, 1.2)
    workspace_y: tuple[float, float] = (-1.0, 0.7)
    workspace_z: tuple[float, float] = (-0.02, 1.0)
    episode_timeout: float = 60.0
    joint_vel_limits: list[float] = field(
        default_factory=lambda: [2.36, 2.36, 2.36, 2.36, 2.36, 3.14])

    # Joint position limits (rad) from HC10DT URDF
    joint_pos_limits: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (-3.14, 3.14),    # joint_1_s
            (-1.92, 2.27),    # joint_2_l
            (-1.22, 3.49),    # joint_3_u
            (-3.14, 3.14),    # joint_4_r
            (-2.18, 2.18),    # joint_5_b
            (-6.28, 6.28),    # joint_6_t
        ])

    @classmethod
    def from_config(cls, cfg: dict) -> SafetyConfig:
        safety = cfg.get('safety', {})
        bounds = safety.get('workspace_bounds', {})
        return cls(
            max_delta_joint=safety.get('max_delta_joint', 0.05),
            max_joint_velocity=safety.get('max_joint_velocity', 0.5),
            workspace_x=tuple(bounds.get('x', [-0.2, 1.2])),
            workspace_y=tuple(bounds.get('y', [-1.0, 0.7])),
            workspace_z=tuple(bounds.get('z', [-0.02, 1.0])),
            episode_timeout=safety.get('episode_timeout', 60.0),
            joint_vel_limits=safety.get('joint_limits',
                                        [2.36, 2.36, 2.36, 2.36, 2.36, 3.14]),
        )


@dataclass
class SafetyResult:
    """Result of a safety check."""
    safe: bool
    clipped_deltas: np.ndarray   # (6,) clipped joint deltas
    reason: str = ''             # non-empty if action was clipped or blocked


class SafetyWrapper:
    """Clips VLA joint-delta actions to safe bounds.

    Does NOT do forward kinematics (would require a URDF parser).
    Workspace bound checking requires an external FK callback.
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.cfg = config or SafetyConfig()

    def check_and_clip(
        self,
        joint_deltas: np.ndarray,
        current_joints: np.ndarray,
        control_hz: float = 10.0,
        tcp_position: np.ndarray | None = None,
    ) -> SafetyResult:
        """Clip joint deltas to safe range and check limits.

        Args:
            joint_deltas: (6,) predicted delta joint positions (rad).
            current_joints: (6,) current joint positions (rad).
            control_hz: Control loop frequency for velocity calculation.
            tcp_position: (3,) TCP xyz in base_link frame, if available.

        Returns:
            SafetyResult with clipped deltas and safety status.
        """
        deltas = np.array(joint_deltas, dtype=np.float64)
        current = np.array(current_joints, dtype=np.float64)
        reasons = []

        # 1. Clip per-step delta magnitude
        max_d = self.cfg.max_delta_joint
        pre_clip = deltas.copy()
        deltas = np.clip(deltas, -max_d, max_d)
        if not np.allclose(pre_clip, deltas):
            reasons.append('delta_clipped')

        # 2. Clip effective velocity (delta * hz)
        velocities = np.abs(deltas) * control_hz
        for i, (vel, limit) in enumerate(
                zip(velocities, self.cfg.joint_vel_limits)):
            if vel > limit:
                scale = limit / vel
                deltas[i] *= scale
                reasons.append(f'j{i+1}_vel_clipped')

        # 3. Clip target position to joint limits
        target = current + deltas
        for i, (lo, hi) in enumerate(self.cfg.joint_pos_limits):
            if target[i] < lo:
                deltas[i] = lo - current[i]
                reasons.append(f'j{i+1}_pos_lo')
            elif target[i] > hi:
                deltas[i] = hi - current[i]
                reasons.append(f'j{i+1}_pos_hi')

        # 4. Workspace bounds check (if TCP position available)
        if tcp_position is not None:
            x, y, z = tcp_position
            blocked = False
            if not (self.cfg.workspace_x[0] <= x <= self.cfg.workspace_x[1]):
                reasons.append('workspace_x')
                blocked = True
            if not (self.cfg.workspace_y[0] <= y <= self.cfg.workspace_y[1]):
                reasons.append('workspace_y')
                blocked = True
            if not (self.cfg.workspace_z[0] <= z <= self.cfg.workspace_z[1]):
                reasons.append('workspace_z')
                blocked = True
            if blocked:
                # Zero out the action — don't move if TCP is out of bounds
                deltas[:] = 0.0
                return SafetyResult(
                    safe=False,
                    clipped_deltas=deltas,
                    reason='workspace_violation: ' + ','.join(reasons),
                )

        reason = ','.join(reasons) if reasons else ''
        return SafetyResult(safe=True, clipped_deltas=deltas, reason=reason)

    def is_at_home(self, current_joints: np.ndarray,
                   home_joints: np.ndarray,
                   tolerance: float = 0.15) -> bool:
        """Check if current joints are within tolerance of home position."""
        return bool(np.all(np.abs(current_joints - home_joints) < tolerance))
