#!/usr/bin/env python
"""Background watcher that normalizes every saved LeRobot+PEFT checkpoint so
`run_vla_closed_loop.py` can load it without manual surgery.

For every `checkpoints/*/pretrained_model/` it finds, it ensures:

1. `adapter_config.json` has `base_model_name_or_path == "lerobot/smolvla_base"`.
   LeRobot+PEFT save this field as the current `cfg.policy.pretrained_path`.
   On the FIRST epoch that's correct; on every resume it becomes the LOCAL
   previous checkpoint (adapter-only, no model.safetensors) — so the next
   resume would try to load base weights from an adapter-only dir and crash.

2. `config.json` has a `type: smolvla` key. PreTrainedConfig.from_pretrained
   dispatches via this field; lerobot's save path sometimes drops it, which
   breaks our eval script.

Runs until the STOP sentinel appears in $LOG_DIR.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

TARGET_BASE = "lerobot/smolvla_base"
POLICY_TYPE = "smolvla"


def _rewrite_json(path: pathlib.Path, patch: dict) -> bool:
    """Load JSON at `path`, apply `patch` if any key differs, rewrite atomically.

    Returns True if the file was changed.
    """
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    changed = False
    for k, v in patch.items():
        if d.get(k) != v:
            d[k] = v
            changed = True
    if not changed:
        return False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(path)
    return True


def fix_pretrained_model_dir(pm_dir: pathlib.Path) -> list[str]:
    """Apply all normalizations to one pretrained_model dir.

    Returns a list of human-readable strings describing what changed (empty if
    already normalized).
    """
    changes = []

    adapter_cfg = pm_dir / "adapter_config.json"
    if adapter_cfg.exists():
        if _rewrite_json(adapter_cfg, {"base_model_name_or_path": TARGET_BASE}):
            changes.append(f"adapter_config.base_model→{TARGET_BASE}")

    policy_cfg = pm_dir / "config.json"
    if policy_cfg.exists():
        if _rewrite_json(policy_cfg, {"type": POLICY_TYPE}):
            changes.append(f"config.type→{POLICY_TYPE}")

    return changes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="e.g. checkpoints/continual")
    p.add_argument("--log-dir", required=True, help="STOP sentinel dir")
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--one-shot", action="store_true",
                   help="Process all existing checkpoints once and exit (no daemon).")
    args = p.parse_args()

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    stop_file = pathlib.Path(args.log_dir) / "STOP"
    log_file = pathlib.Path(args.log_dir) / "adapter_fixer.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with log_file.open("a") as f:
            f.write(f"[{ts}] {msg}\n")
        sys.stdout.write(f"[{ts}] {msg}\n")
        sys.stdout.flush()

    if args.one_shot:
        log(f"adapter_fixer one-shot scan of {ckpt_dir}")
    else:
        log(f"adapter_fixer starting (daemon); watching {ckpt_dir}/checkpoints/*/pretrained_model/")

    def scan_once() -> None:
        # Enumerate pretrained_model dirs rather than individual files, so we
        # can normalize adapter_config.json + config.json in the same pass.
        for pm_dir in (ckpt_dir / "checkpoints").glob("*/pretrained_model"):
            # Skip the 'last/' symlink — it points at a real numbered dir that
            # we'll also visit directly.
            if "last" in pm_dir.parts:
                continue
            changes = fix_pretrained_model_dir(pm_dir)
            if changes:
                rel = pm_dir.relative_to(ckpt_dir)
                log(f"fixed {rel}: {', '.join(changes)}")

    if args.one_shot:
        scan_once()
        return 0

    while not stop_file.exists():
        scan_once()
        time.sleep(args.poll_seconds)

    log("STOP detected; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
