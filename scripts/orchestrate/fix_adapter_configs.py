#!/usr/bin/env python
"""Background watcher that rewrites adapter_config.json's base_model_name_or_path
to 'lerobot/smolvla_base' for every saved checkpoint.

LeRobot + PEFT save the adapter config with base_model_name_or_path set to the
current pretrained_path, which after a resume is the LOCAL previous checkpoint
(an adapter-only dir with no base model weights). Next resume then fails trying
to load model.safetensors from there. We always want the original HF base, so
we overwrite the field as soon as a new checkpoint appears.

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


def fix_one(cfg_path: pathlib.Path) -> bool:
    try:
        d = json.loads(cfg_path.read_text())
    except Exception as exc:
        return False
    cur = d.get("base_model_name_or_path", "")
    if cur == TARGET_BASE:
        return False
    d["base_model_name_or_path"] = TARGET_BASE
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(cfg_path)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="e.g. checkpoints/continual")
    p.add_argument("--log-dir", required=True, help="STOP sentinel dir")
    p.add_argument("--poll-seconds", type=int, default=5)
    args = p.parse_args()

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    stop_file = pathlib.Path(args.log_dir) / "STOP"
    log_file = pathlib.Path(args.log_dir) / "adapter_fixer.log"

    def log(msg: str) -> None:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with log_file.open("a") as f:
            f.write(f"[{ts}] {msg}\n")
        sys.stdout.write(f"[{ts}] {msg}\n")
        sys.stdout.flush()

    log(f"adapter_fixer starting; watching {ckpt_dir}/checkpoints/*/pretrained_model/adapter_config.json")

    while not stop_file.exists():
        glob_path = ckpt_dir / "checkpoints" / "*" / "pretrained_model" / "adapter_config.json"
        # expand glob
        found = list((ckpt_dir / "checkpoints").glob("*/pretrained_model/adapter_config.json"))
        for cfg in found:
            # skip symlinks pointing into 'last'
            if "last" in cfg.parts:
                continue
            if fix_one(cfg):
                log(f"fixed {cfg.relative_to(ckpt_dir)}")
        time.sleep(args.poll_seconds)

    log("STOP detected; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
