"""Strip the `peft:` section from a saved LeRobot train_config.json.

Context: on resume, `lerobot_train.py` calls `make_policy()` which (via
`cfg.policy.use_peft=True`) correctly loads the saved LoRA adapter onto base
SmolVLA. It then sees `cfg.peft is not None` in the saved config and calls
`policy.wrap_with_peft(...)` AGAIN — layering a fresh zero-init LoRA on top of
the already-loaded adapter. The optimizer trains only the new zero-init
adapter; the previously-trained one is frozen and ignored. Loss regresses to
~0.75 on every resumed epoch.

Fix: blank the top-level `peft` section before each resume. `policy.use_peft`
stays True so `make_policy` still loads the adapter; but with `cfg.peft=None`
the train script's second `wrap_with_peft` call is skipped.

Usage:
    python prepare_resume_config.py \\
        --config checkpoints/continual/checkpoints/last/pretrained_model/train_config.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def strip_peft(cfg_path: pathlib.Path) -> bool:
    if not cfg_path.exists():
        print(f"[prepare_resume] skipping: {cfg_path} does not exist", flush=True)
        return False
    try:
        d = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[prepare_resume] ERROR reading {cfg_path}: {exc}", flush=True)
        return False

    had_peft = d.get("peft") is not None
    if not had_peft:
        print(f"[prepare_resume] {cfg_path.name}: peft already None, nothing to do", flush=True)
        return False

    d["peft"] = None
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(cfg_path)
    print(f"[prepare_resume] {cfg_path.name}: stripped peft section", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True,
                    help="Path to train_config.json inside a saved pretrained_model dir.")
    args = ap.parse_args()
    strip_peft(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
