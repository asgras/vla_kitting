"""Generate the canonical fixed-seed 30-position eval cube_xy set.

Why: every prior closed-loop eval used a different N and seed list, so
cross-run SR comparisons were unreliable. This script produces a single
canonical JSON file (configs/eval_seed_30.json) with 30 (x, y) cube
positions sampled from the widened cube box, deterministic under
np.random.seed(42). The eval invocation reads this list and feeds it to
run_vla_closed_loop.py via --cube_xy.

Cube box: env's reset_root_state_uniform samples a delta from the cube's
default spawn (0.55, 0.0, 0.025). With the current widened ranges
(X delta in [-0.15, 0.15], Y delta in [-0.22, 0.22]) the absolute cube
positions span X in [0.40, 0.70], Y in [-0.22, 0.22]. We sample in
absolute coordinates because that's what --cube_xy expects.

Usage:
    python scripts/orchestrate/build_eval_seed_set.py
    # writes configs/eval_seed_30.json

    # then in eval:
    python -c 'import json; d=json.load(open("configs/eval_seed_30.json")); \\
        print(";".join(f"{x},{y}" for x,y in d["positions"]))'

Acceptance: re-running this script produces a byte-identical JSON file;
running an eval with --cube_xy "$(cat …)" twice produces byte-identical
per-episode cube0 logs.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=30)
    p.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=(0.40, 0.70),
        help="Absolute cube X range, matches widened pose_range x in env.",
    )
    p.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=(-0.22, 0.22),
        help="Absolute cube Y range, matches widened pose_range y in env.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(REPO / "configs" / "eval_seed_30.json"),
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    xs = rng.uniform(args.x_range[0], args.x_range[1], size=args.n)
    ys = rng.uniform(args.y_range[0], args.y_range[1], size=args.n)
    positions = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, ys)]

    payload = {
        "seed": args.seed,
        "n": args.n,
        "x_range": list(args.x_range),
        "y_range": list(args.y_range),
        "rng": "numpy.random.default_rng",
        "positions": positions,
        "cube_xy_string": ";".join(f"{x:.4f},{y:.4f}" for x, y in positions),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} ({args.n} positions, seed={args.seed})")
    print(f"X span: [{min(xs):.3f}, {max(xs):.3f}]  Y span: [{min(ys):.3f}, {max(ys):.3f}]")


if __name__ == "__main__":
    main()
