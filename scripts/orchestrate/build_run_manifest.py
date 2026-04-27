"""Generate manifest.json for a training run.

A run manifest captures everything that determined what came out of a training
session — base model, LoRA config, LR, action loss weights, dataset pointer,
env-cfg sha at launch, plus the final results pulled from epoch_summary.jsonl.
This is what lets a future bisect answer "which dataset + env-cfg + adapter
config produced checkpoint X?" without re-deriving from commit messages.

Two modes:

  - From a live training session:

        python scripts/orchestrate/build_run_manifest.py \\
            --run-name v4_gripper_weight_2026-04-26 \\
            --logs-dir logs/continual \\
            --dataset datasets/lerobot/cube_pick_v3_scripted_20260425_011050 \\
            --base-model lerobot/smolvla_base \\
            --lora-r 64 --lora-alpha 64 --lora-dropout 0.05 \\
            --action-loss-weights 1,1,1,1,1,1,16 \\
            --note "FXAA fix + gripper weight x16; bottleneck = EE positioning"

    Writes `reports/runs/<run-name>/manifest.json`.

  - Render the schema (no IO):

        python scripts/orchestrate/build_run_manifest.py --schema
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_sha_for(rel: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "--", rel],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def summarize_results(logs_dir: Path) -> dict[str, Any]:
    epoch_jsonl = logs_dir / "epoch_summary.jsonl"
    eval_jsonl = logs_dir / "eval_episodes.jsonl"
    epochs = _read_jsonl(epoch_jsonl)
    evals = _read_jsonl(eval_jsonl)

    last = epochs[-1] if epochs else {}
    losses = [e["loss_mean"] for e in epochs if e.get("loss_mean") is not None]
    eval_srs = [(e.get("epoch"), e.get("eval_sr")) for e in epochs if e.get("eval_sr") is not None]
    best_sr_epoch, best_sr = (None, None)
    for ep, sr in eval_srs:
        if sr is None:
            continue
        if best_sr is None or sr > best_sr:
            best_sr_epoch, best_sr = ep, sr

    return {
        "epochs_run": last.get("epoch"),
        "global_step_end": last.get("global_step_end"),
        "loss_floor": min(losses) if losses else None,
        "loss_last": last.get("loss_mean"),
        "best_eval_sr": best_sr,
        "best_eval_sr_epoch": best_sr_epoch,
        "eval_count": len(evals),
        "epoch_summary_path": str(epoch_jsonl.relative_to(REPO_ROOT))
        if epoch_jsonl.exists()
        else None,
        "eval_episodes_path": str(eval_jsonl.relative_to(REPO_ROOT))
        if eval_jsonl.exists()
        else None,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).resolve() if args.dataset else None
    dataset_manifest = (
        dataset_path / "manifest.json" if dataset_path and (dataset_path / "manifest.json").exists() else None
    )

    started_at = (
        dt.datetime.fromtimestamp(args.start_unix, tz=dt.UTC).isoformat()
        if args.start_unix
        else None
    )

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "type": "run",
        "run_id": args.run_name,
        "started_at": started_at,
        "ended_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "git_head_at_build": _git_head(),
        "code_sha": {
            "envs/yaskawa_pick_cube_cfg.py": _git_sha_for("envs/yaskawa_pick_cube_cfg.py"),
            "scripts/orchestrate/train_only.sh": _git_sha_for("scripts/orchestrate/train_only.sh"),
            "scripts/train/run_vla_closed_loop.py": _git_sha_for(
                "scripts/train/run_vla_closed_loop.py"
            ),
            "scripts/validate/scripted_pick_demo.py": _git_sha_for(
                "scripts/validate/scripted_pick_demo.py"
            ),
        },
        "dataset": {
            "path": str(dataset_path.relative_to(REPO_ROOT))
            if dataset_path
            else None,
            "manifest": str(dataset_manifest.relative_to(REPO_ROOT))
            if dataset_manifest
            else None,
        },
        "policy": {
            "base_model": args.base_model,
            "load_vlm_weights": args.load_vlm_weights,
            "lora": {
                "enabled": args.lora_r is not None and args.lora_r > 0,
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules_regex": args.lora_targets_regex,
                "modules_to_save": args.modules_to_save,
            },
        },
        "training": {
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "batch_size": args.batch_size,
            "n_action_steps": args.n_action_steps,
            "action_loss_dim_weights": (
                [float(x) for x in args.action_loss_weights.split(",")]
                if args.action_loss_weights
                else None
            ),
            "save_freq_steps": args.save_freq,
            "eval_episodes": args.eval_episodes,
            "eval_every_n_epochs": args.eval_every_n,
        },
        "results": summarize_results(Path(args.logs_dir).resolve()) if args.logs_dir else {},
        "checkpoints_dir": args.checkpoints_dir,
        "command_line": args.cmdline,
        "notes": args.note,
    }
    return manifest


def schema_blob() -> dict[str, Any]:
    """Return a documented dict illustrating every field. Useful as `--schema`."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "type": "run",
        "run_id": "<short slug e.g. v4_gripper_weight_2026-04-26>",
        "started_at": "<ISO 8601 UTC>",
        "ended_at": "<ISO 8601 UTC, set when manifest is written>",
        "host": "<gethostname>",
        "user": "<USER>",
        "git_head_at_build": "<sha of HEAD at manifest write time>",
        "code_sha": "<{path: last-commit-sha-touching-path}>",
        "dataset": {
            "path": "datasets/lerobot/<dataset_dir>",
            "manifest": "datasets/lerobot/<dataset_dir>/manifest.json",
        },
        "policy": {
            "base_model": "<HF id e.g. lerobot/smolvla_base>",
            "load_vlm_weights": True,
            "lora": {
                "enabled": True,
                "r": 64,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules_regex": "<PEFT regex>",
                "modules_to_save": ["action_out_proj"],
            },
        },
        "training": {
            "lr": 1e-4,
            "lr_schedule": "constant|cosine|...",
            "batch_size": 4,
            "n_action_steps": 10,
            "action_loss_dim_weights": [1, 1, 1, 1, 1, 1, 16],
            "save_freq_steps": 1000,
            "eval_episodes": 10,
            "eval_every_n_epochs": 5,
        },
        "results": {
            "epochs_run": 48,
            "global_step_end": 48000,
            "loss_floor": 0.027,
            "loss_last": 0.0273,
            "best_eval_sr": 0.10,
            "best_eval_sr_epoch": 15,
            "eval_count": 8,
            "epoch_summary_path": "logs/continual/epoch_summary.jsonl",
            "eval_episodes_path": "logs/continual/eval_episodes.jsonl",
        },
        "checkpoints_dir": "checkpoints/continual/checkpoints/",
        "command_line": "<full bash invocation>",
        "notes": "<free-form one-paragraph lesson; what surprised you, what to check next>",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", help="slug, e.g. v4_gripper_weight_2026-04-26")
    ap.add_argument("--logs-dir", default="logs/continual", help="Where epoch_summary.jsonl lives")
    ap.add_argument("--dataset", help="Path to LeRobot dataset (resolved through symlinks)")
    ap.add_argument("--base-model", default="lerobot/smolvla_base")
    ap.add_argument("--load-vlm-weights", action="store_true")
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--lora-alpha", type=int)
    ap.add_argument("--lora-dropout", type=float)
    ap.add_argument("--lora-targets-regex")
    ap.add_argument("--modules-to-save", action="append", default=[])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-schedule", default="constant")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--n-action-steps", type=int)
    ap.add_argument(
        "--action-loss-weights",
        help="Comma-separated weights matching the action dim, e.g. 1,1,1,1,1,1,16",
    )
    ap.add_argument("--save-freq", type=int)
    ap.add_argument("--eval-episodes", type=int)
    ap.add_argument("--eval-every-n", type=int)
    ap.add_argument("--start-unix", type=float, help="Run start time as unix seconds")
    ap.add_argument("--checkpoints-dir", default="checkpoints/continual/checkpoints/")
    ap.add_argument("--cmdline", help="The exact orchestrator invocation")
    ap.add_argument("--note", help="Free-form lesson / context for this run")
    ap.add_argument("--schema", action="store_true", help="Print the schema and exit")
    ap.add_argument(
        "--out",
        type=Path,
        help="Where to write manifest.json (default reports/runs/<run-name>/manifest.json)",
    )
    args = ap.parse_args()

    if args.schema:
        print(json.dumps(schema_blob(), indent=2))
        return 0

    if not args.run_name:
        ap.error("--run-name is required (or pass --schema)")

    manifest = build_manifest(args)
    out = args.out or REPO_ROOT / "reports" / "runs" / args.run_name / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
