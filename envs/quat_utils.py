"""Shared quaternion utilities for scripted controllers.

Imported by scripts/validate/scripted_pick_demo.py (data generation) and
scripts/validate/gripper_descent_probe.py (geometry probe). Quaternions are
in (w, x, y, z) order to match Isaac Lab's convention.

This module is imported AFTER `AppLauncher` in the call sites, so a
module-level torch import is safe. Do not import it from data-conversion
scripts that run without the Isaac kit.
"""
from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions in (w, x, y, z) form."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a (w, x, y, z) quaternion (negates the vector part)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_err_axis_angle(q_cur: torch.Tensor, q_des: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) representing the shortest-path rotation from
    q_cur to q_des. Both inputs are (w, x, y, z) quaternions."""
    q_err = quat_mul(q_des, quat_conj(q_cur))
    if q_err[..., 0].item() < 0:
        q_err = -q_err
    w = q_err[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=1e-8))
    axis = q_err[..., 1:] / sin_half
    return axis * angle
