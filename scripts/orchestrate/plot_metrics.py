"""Build training curves from the orchestrator's JSONL logs.

Reads:
  - logs/continual/train_steps.jsonl    (per-step: step, loss, grad, lr)
  - logs/continual/epoch_summary.jsonl  (per-epoch: p50/p95 loss, eval_sr)
  - logs/continual/eval_episodes.jsonl  (per-episode: result, steps)

Emits PNGs under reports/curves/:
  - loss_vs_step.png       loss with p95 band + LR on twin axis + epoch ticks
  - grad_norm_vs_step.png  gradient norm over steps
  - eval_sr_vs_epoch.png   eval success rate with running mean
  - episode_steps.png      per-episode step count, colored by result

Usage:
    python plot_metrics.py --log-dir logs/continual --out reports/curves
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def plot_loss(steps_rows: list[dict], out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    if not steps_rows:
        print("  no step rows; skipping loss plot")
        return

    xs = [r["step"] for r in steps_rows]
    losses = [r["loss"] for r in steps_rows]
    lrs = [r["lr"] for r in steps_rows]

    # Rolling window p95 / p50, window = max(50, 5% of samples).
    w = max(50, len(losses) // 20)

    def _rolling(vs, q):
        out = []
        for i in range(len(vs)):
            lo = max(0, i - w // 2)
            hi = min(len(vs), i + w // 2)
            s = sorted(vs[lo:hi])
            idx = int(round((len(s) - 1) * q))
            out.append(s[idx])
        return out

    p50 = _rolling(losses, 0.5)
    p95 = _rolling(losses, 0.95)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(xs, losses, color="C0", alpha=0.15, lw=0.8, label="loss (raw)")
    ax1.plot(xs, p50, color="C0", lw=1.8, label=f"loss p50 (w={w})")
    ax1.plot(xs, p95, color="C0", lw=1.0, ls="--", alpha=0.7, label=f"loss p95 (w={w})")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    # Mark epoch boundaries (any point where `epoch` increments).
    last_epoch = None
    for r in steps_rows:
        e = r.get("epoch")
        if last_epoch is not None and e != last_epoch:
            ax1.axvline(r["step"], color="gray", lw=0.4, alpha=0.5)
        last_epoch = e

    ax2 = ax1.twinx()
    ax2.plot(xs, lrs, color="C3", lw=1.0, alpha=0.7, label="lr")
    ax2.set_ylabel("lr", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_grad(steps_rows: list[dict], out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    if not steps_rows:
        return
    xs = [r["step"] for r in steps_rows]
    gns = [r["grad_norm"] for r in steps_rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, gns, color="C2", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_eval_sr(epoch_rows: list[dict], out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    evals = [(r["epoch"], r["eval_sr"]) for r in epoch_rows if r.get("eval_sr") is not None]
    if not evals:
        print("  no eval rows; skipping eval_sr plot")
        return
    xs = [e[0] for e in evals]
    ys = [e[1] for e in evals]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, "o-", color="C1", label="eval_sr")
    if len(ys) >= 3:
        import statistics
        w = min(5, len(ys))
        roll = [statistics.mean(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]
        ax.plot(xs, roll, color="C1", alpha=0.4, lw=2, label=f"rolling mean (w={w})")
    ax.set_xlabel("epoch"); ax.set_ylabel("eval success rate")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_episodes(ep_rows: list[dict], out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    if not ep_rows:
        print("  no episode rows; skipping episodes plot")
        return
    # Use sample order as x-axis; color by result.
    colors = {"success": "C2", "timeout": "C3", "fail": "C7"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(ep_rows):
        ax.scatter([i], [r.get("steps", 0)], color=colors.get(r.get("result"), "C0"),
                   s=25, alpha=0.7)
    ax.set_xlabel("episode index (chronological)")
    ax.set_ylabel("steps to termination")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in colors.items()])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("logs/continual"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/curves"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"reading from {args.log_dir}; writing to {args.out}")

    steps = _load_jsonl(args.log_dir / "train_steps.jsonl")
    epochs = _load_jsonl(args.log_dir / "epoch_summary.jsonl")
    eps = _load_jsonl(args.log_dir / "eval_episodes.jsonl")

    print(f"  train_steps.jsonl    rows={len(steps)}")
    print(f"  epoch_summary.jsonl  rows={len(epochs)}")
    print(f"  eval_episodes.jsonl  rows={len(eps)}")

    plot_loss(steps, args.out / "loss_vs_step.png")
    plot_grad(steps, args.out / "grad_norm_vs_step.png")
    plot_eval_sr(epochs, args.out / "eval_sr_vs_epoch.png")
    plot_episodes(eps, args.out / "episode_steps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
