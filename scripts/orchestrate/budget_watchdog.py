#!/usr/bin/env python
# Budget watchdog for continual_train.sh.
#
# Touches $LOG_DIR/STOP when either:
#   (a) elapsed wallclock >= BUDGET_HOURS (default 8h), OR
#   (b) eval success-rate plateau: no new best eval_sr across the last
#       PLATEAU_EVALS (default 3) eval cycles AND loss has not improved by
#       at least PLATEAU_LOSS_DELTA (default 1%) across that window.
#
# Grace period: plateau is only considered once wallclock >= GRACE_MINUTES
# (default 60 min). Before that we let the pipeline bootstrap.
#
# Additional sanity: we need at least PLATEAU_EVALS completed eval rows in
# epoch_summary.jsonl. If eval hasn't run that many times yet, no plateau.
#
# The watchdog does NOT kill anything directly. It only writes STOP — the
# orchestrator's own _check_stop hook exits the loops gracefully.

import argparse
import datetime as dt
import json
import math
import pathlib
import sys
import time


def load_epochs(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def plateau_reached(
    epochs: list[dict],
    plateau_evals: int,
    loss_delta_frac: float,
) -> tuple[bool, str]:
    evals = [e for e in epochs if e.get("eval_sr") is not None]
    if len(evals) < plateau_evals + 1:
        return False, f"only {len(evals)} eval rows, need {plateau_evals + 1}"

    # Compare best eval_sr in the trailing window vs. the cumulative best
    # before the window. If the trailing window didn't produce a new best,
    # that's one half of plateau.
    window = evals[-plateau_evals:]
    pre = evals[:-plateau_evals]
    best_pre = max(e["eval_sr"] for e in pre)
    best_win = max(e["eval_sr"] for e in window)
    if best_win > best_pre:
        return False, f"window best eval_sr={best_win:.3f} > pre={best_pre:.3f}"

    # Loss half: compare mean loss of the first eval in the window vs. the
    # last. If loss hasn't dropped by at least loss_delta_frac, plateau.
    losses = [e.get("loss") for e in window if e.get("loss") is not None]
    if len(losses) < 2:
        return False, f"insufficient loss samples in window ({len(losses)})"
    first, last = losses[0], losses[-1]
    if first <= 0 or math.isnan(first) or math.isnan(last):
        return False, f"invalid loss values ({first}, {last})"
    drop = (first - last) / first
    if drop >= loss_delta_frac:
        return False, f"loss dropped {drop:.3%} >= {loss_delta_frac:.1%}"

    return True, (
        f"plateau: no new best eval_sr over {plateau_evals} cycles "
        f"(pre={best_pre:.3f}, win={best_win:.3f}); "
        f"loss drop {drop:.3%} < {loss_delta_frac:.1%}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--budget-hours", type=float, default=8.0)
    p.add_argument("--grace-minutes", type=float, default=60.0)
    p.add_argument("--plateau-evals", type=int, default=3)
    p.add_argument("--plateau-loss-delta", type=float, default=0.01)
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stop_file = log_dir / "STOP"
    watchdog_log = log_dir / "watchdog.log"
    epoch_jsonl = log_dir / "epoch_summary.jsonl"

    start = time.time()
    deadline = start + args.budget_hours * 3600.0

    def log(msg: str) -> None:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        line = f"[{ts}] {msg}\n"
        with watchdog_log.open("a") as f:
            f.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    log(
        f"watchdog starting: budget={args.budget_hours}h "
        f"grace={args.grace_minutes}m "
        f"plateau_evals={args.plateau_evals} "
        f"plateau_loss_delta={args.plateau_loss_delta:.1%}"
    )

    while True:
        if stop_file.exists():
            log("STOP already present; watchdog exiting")
            return 0

        now = time.time()
        elapsed_h = (now - start) / 3600.0

        if now >= deadline:
            log(f"BUDGET EXHAUSTED: elapsed={elapsed_h:.2f}h >= {args.budget_hours}h; writing STOP")
            stop_file.touch()
            return 0

        if (now - start) >= args.grace_minutes * 60.0:
            epochs = load_epochs(epoch_jsonl)
            plateau, reason = plateau_reached(
                epochs,
                plateau_evals=args.plateau_evals,
                loss_delta_frac=args.plateau_loss_delta,
            )
            if plateau:
                log(f"PLATEAU: {reason}; writing STOP")
                stop_file.touch()
                return 0
            # Heartbeat every ~10 min.
            if int(now - start) % 600 < args.poll_seconds:
                log(
                    f"heartbeat: elapsed={elapsed_h:.2f}h epochs={len(epochs)} "
                    f"check={reason}"
                )
        else:
            if int(now - start) % 600 < args.poll_seconds:
                log(f"heartbeat: elapsed={elapsed_h:.2f}h (grace period)")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
