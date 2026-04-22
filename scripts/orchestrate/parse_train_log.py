"""Parse lerobot_train.py stdout live and emit one JSONL line per logged step.

LeRobot logs a step summary every `--log_freq` steps in this format:

    INFO 2026-04-22 07:09:32 ot_train.py:423 \\
        step:7K smpl:28K ep:17 epch:0.70 loss:0.186 grdn:0.190 lr:7.8e-05 \\
        updt_s:0.304 data_s:0.005

We read the stream line-by-line, regex-match that pattern, and append a
normalized JSON record to an output file. Unknown lines are echoed to stdout
so the caller can still see them (via tee).

Usage:
    lerobot_train.py ... 2>&1 | python parse_train_log.py \\
        --out logs/continual/train_steps.jsonl --epoch 3

The `--epoch` value is stamped on every emitted record so downstream plots
can correlate with epoch_summary.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys


STEP_LINE_RE = re.compile(
    r"step:(?P<step>[\d.]+[KMB]?)"
    r".*?smpl:(?P<smpl>[\d.]+[KMB]?)"
    r".*?ep:(?P<ep>\d+)"
    r".*?epch:(?P<epch>[\d.]+)"
    r".*?loss:(?P<loss>[\d.]+(?:[eE][+-]?\d+)?)"
    r".*?grdn:(?P<grdn>[\d.]+(?:[eE][+-]?\d+)?)"
    r".*?lr:(?P<lr>[\d.]+(?:[eE][+-]?\d+)?)"
    r".*?updt_s:(?P<updt_s>[\d.]+)"
    r".*?data_s:(?P<data_s>[\d.]+)"
)


def _expand_suffix(s: str) -> int:
    """LeRobot prints step/sample counts as e.g. '7K' or '1.2M'. Expand."""
    s = s.strip()
    mult = 1
    if s.endswith("K"):
        mult = 1_000
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1_000_000
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1_000_000_000
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return -1


def parse_line(line: str) -> dict | None:
    m = STEP_LINE_RE.search(line)
    if not m:
        return None
    g = m.groupdict()
    return {
        "step": _expand_suffix(g["step"]),
        "samples": _expand_suffix(g["smpl"]),
        "ep": int(g["ep"]),
        "epch": float(g["epch"]),
        "loss": float(g["loss"]),
        "grad_norm": float(g["grdn"]),
        "lr": float(g["lr"]),
        "update_s": float(g["updt_s"]),
        "data_s": float(g["data_s"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="Append JSONL records here.")
    ap.add_argument("--epoch", type=int, default=-1,
                    help="Stamp this epoch number on every record "
                         "(tie-breaker for plotting).")
    ap.add_argument("--echo", action="store_true", default=True,
                    help="Also echo every input line to stdout (default on).")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.out.open("a") as out_f:
            for raw in sys.stdin:
                # Echo every line so an outer `tee` sees them.
                if args.echo:
                    sys.stdout.write(raw)
                    sys.stdout.flush()
                rec = parse_line(raw)
                if rec is None:
                    continue
                rec["ts"] = dt.datetime.utcnow().isoformat() + "Z"
                rec["epoch"] = args.epoch
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
    except BrokenPipeError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
