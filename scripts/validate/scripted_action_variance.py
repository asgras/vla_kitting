"""Diagnose P-controller saturation in scripted demos.

The recovery plan (recovery_plan_2026-04-24 §2) hypothesizes that the v3/v4
training data is bimodal because the scripted controller uses a saturated
P-controller — at gain 10 any pos_err > 10 cm clamps the action delta to
±1, so the FIRST few actions of every demo are nearly identical regardless
of cube position. The policy then learns "always go down-left from home"
and only varies at the per-cube level once the EE is close.

This script reads a scripted-demo HDF5 and tells you whether the action
distribution actually has cube-position-dependent variance, or whether the
first several steps collapse to a single saturated trajectory.

Diagnostic:
  - For each step t in [0, K), compute the per-dimension variance of
    action[t] ACROSS demos.
  - Compare against a "noise floor" estimated from late-trajectory steps
    (where the EE has converged; actions are tiny pos errors → near zero).
  - A healthy training set shows VAR(t) > NOISE_FLOOR for all 6 EE-delta
    dims at every t. Saturation looks like a flat plateau at the saturation
    cap variance for the first ~50 steps.

Usage:
    python scripts/validate/scripted_action_variance.py \\
        --hdf5 /tmp/yaw_30/cube_scripted_yaw30.hdf5 \\
        --first_k 50 \\
        --out_png reports/.../action_variance.png

Acceptance (per beads issue vla_kitting-e9y):
  variance plot + verdict line printed to stdout:
  - "DIVERSE" if max(var) over first_k steps >> noise_floor in all 6 EE
    dims, with a numeric ratio.
  - "SATURATED" if any of the first 6 dims has var ≲ 0.01² for all of the
    first_k steps (flat clamp).
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np


def _stack_first_k_actions(hdf5_path: pathlib.Path, first_k: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Returns (actions, cube_xy_per_demo) where actions has shape
    (n_demos, first_k, 7). Pads short demos by repeating the last action
    (rare; scripted demos are usually >=833 steps).
    """
    demos = []
    cube_xys: list[tuple[float, float]] = []
    with h5py.File(str(hdf5_path), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            d = f["data"][k]
            if int(d.attrs.get("num_samples", 0)) <= 0:
                continue
            actions = d["actions"][...]  # (T, 7)
            if actions.shape[0] < 1:
                continue
            T = actions.shape[0]
            if T >= first_k:
                a = actions[:first_k]
            else:
                a = np.concatenate(
                    [actions, np.tile(actions[-1:], (first_k - T, 1))], axis=0
                )
            demos.append(a)
            # First cube_pos is the cube's spawn (before any action).
            cp = d["obs"]["cube_pos"][0]
            cube_xys.append((float(cp[0]), float(cp[1])))
    if not demos:
        raise RuntimeError(f"no real demos found in {hdf5_path}")
    return np.stack(demos, axis=0).astype(np.float32), cube_xys


def _noise_floor(hdf5_path: pathlib.Path, sample_steps: int = 100) -> np.ndarray:
    """Estimate per-dim noise floor from late-trajectory near-converged
    actions. Take the last `sample_steps` steps of each demo and compute
    the per-dim variance — when the EE is at the place hover/release the
    actions are dominated by tiny pos errors and gripper toggles, so this
    sets a baseline below which "no signal" looks identical to noise.
    """
    floors = []
    with h5py.File(str(hdf5_path), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            d = f["data"][k]
            if int(d.attrs.get("num_samples", 0)) <= 0:
                continue
            a = d["actions"][...]
            if a.shape[0] < sample_steps + 1:
                continue
            tail = a[-sample_steps:]
            floors.append(tail.var(axis=0))
    if not floors:
        return np.zeros(7, dtype=np.float32)
    return np.mean(np.stack(floors, axis=0), axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=pathlib.Path, required=True)
    ap.add_argument("--first_k", type=int, default=50)
    ap.add_argument("--out_png", type=pathlib.Path, default=None)
    ap.add_argument(
        "--saturation_floor",
        type=float,
        default=1e-4,
        help="Variance below this for an entire window of first_k steps "
             "= the dimension is saturated/clamped.",
    )
    ap.add_argument(
        "--diversity_ratio",
        type=float,
        default=10.0,
        help="Min ratio max(var_first_k) / noise_floor to call dim DIVERSE.",
    )
    args = ap.parse_args()

    actions, cube_xys = _stack_first_k_actions(args.hdf5, args.first_k)
    n_demos = actions.shape[0]
    print(f"[scripted-var] read {n_demos} demos, first_k={args.first_k}")

    var_per_step = actions.var(axis=0)  # (first_k, 7)
    floor = _noise_floor(args.hdf5)  # (7,)
    print("[scripted-var] noise floor (per dim):", np.array2string(floor, precision=4))

    dim_names = ["ee_dx", "ee_dy", "ee_dz", "ee_drx", "ee_dry", "ee_drz", "gripper"]
    # Per-demo means tell us the per-dim AVERAGE action over the first_k
    # window. If those means cluster near a single saturation cap (±1) the
    # P-controller is clamping; if they cluster near ~0 the dim is just
    # (deterministically) zero in the early phase. The latter is the
    # expected behavior of dims that don't naturally vary with cube
    # position in the early-phase script (e.g. ee_dz during approach is
    # the same constant descent profile in every demo).
    per_demo_mean = actions.mean(axis=1)  # (n_demos, 7)
    per_dim_mean_abs = np.abs(per_demo_mean).mean(axis=0)  # (7,)
    print(
        f"{'dim':>8} {'min_var':>10} {'max_var':>10} {'mean_var':>10} "
        f"{'mean|a|':>10} {'noise':>10} {'verdict':>10}"
    )
    verdicts = {}
    for i, name in enumerate(dim_names):
        v = var_per_step[:, i]
        mn, mx, av = float(v.min()), float(v.max()), float(v.mean())
        nz = float(floor[i])
        ma = float(per_dim_mean_abs[i])
        if mx < args.saturation_floor:
            # Across-demo variance is essentially zero. Distinguish a
            # CLAMPED dim (action mean magnitude near 1) from a NULL
            # dim (action mean magnitude near 0). Only the CLAMPED case
            # is the saturated-P-controller pathology the recovery plan
            # is worried about.
            verdict = "SATURATED" if ma > 0.5 else "CONSTANT"
        elif nz > 0 and mx / nz < args.diversity_ratio:
            verdict = "WEAK"
        else:
            verdict = "DIVERSE"
        verdicts[name] = verdict
        print(
            f"{name:>8} {mn:>10.5f} {mx:>10.5f} {av:>10.5f} "
            f"{ma:>10.5f} {nz:>10.5f} {verdict:>10}"
        )

    saturated_pos_dims = [
        d for d in dim_names[:6] if verdicts.get(d) == "SATURATED"
    ]
    if saturated_pos_dims:
        print(f"[scripted-var] VERDICT: SATURATED on {saturated_pos_dims}. "
              "Lower P-gain (e.g. from 10 to 2) to break the clamp.")
    else:
        # CONSTANT dims (e.g. ee_dz during the deterministic descent
        # profile) are expected — they're the same script value in every
        # demo, not a clamp. Only DIVERSE / WEAK are real diagnostics.
        weak = [d for d in dim_names[:6] if verdicts.get(d) == "WEAK"]
        if weak:
            print(f"[scripted-var] VERDICT: WEAK signal in {weak} — variance "
                  "barely above noise floor. Consider widening cube box / "
                  "yaw randomization to force more action diversity.")
        else:
            print("[scripted-var] VERDICT: data has cube-position-dependent "
                  "variance in all 6 EE-delta dims that vary with cube "
                  "(CONSTANT dims are deterministic-script artifacts, "
                  "not saturation). P-controller is not clamping.")

    if args.out_png is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
        axes = axes.flatten()
        for i, name in enumerate(dim_names):
            ax = axes[i]
            ax.plot(var_per_step[:, i], label="var across demos", color="C0")
            ax.axhline(float(floor[i]), color="C3", ls="--", label="noise floor")
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_title(f"{name} ({verdicts.get(name, '?')})")
            ax.set_xlabel("step")
            if i == 0:
                ax.legend(fontsize=8)
        axes[-1].axis("off")
        fig.suptitle(
            f"Per-step action variance ({n_demos} scripted demos, first {args.first_k} steps)",
            fontsize=12,
        )
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.out_png), dpi=130)
        print(f"[scripted-var] wrote {args.out_png}")

    # Also print the cube-position spread so the reader can confirm the
    # demos cover the widened box.
    if cube_xys:
        xs, ys = zip(*cube_xys)
        print(
            f"[scripted-var] cube X span: [{min(xs):.3f}, {max(xs):.3f}]  "
            f"Y span: [{min(ys):.3f}, {max(ys):.3f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
