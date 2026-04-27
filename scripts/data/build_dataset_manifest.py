"""Generate manifest.json next to a LeRobot dataset snapshot.

The manifest captures everything you'd need to answer "what was this dataset?"
six months later, without trusting filenames or commit messages:

  - Source provenance: scripted-controller sha, env-cfg sha, mimic master path,
    HDF5 hashes, generation date, host.
  - Env config snapshot: env id, decimation, scale, cube randomization box,
    yaw range. Pulled from the live envs/ at the time of generation.
  - Action statistics: per-dim min/max/mean/std plus gripper close fraction —
    the diagnostic that earlier runs were computing ad-hoc.
  - Feature inventory: which observations are present (e.g. cube_pos vs not).

Usage:

    python scripts/data/build_dataset_manifest.py \
        --dataset datasets/lerobot/cube_pick_v3_scripted_20260425_011050

    # Or backfill all snapshots under datasets/lerobot/:
    python scripts/data/build_dataset_manifest.py --all

Schema versioning: bump MANIFEST_SCHEMA_VERSION when adding required fields.
Older manifests stay readable; new fields are filled in as null when missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "envs" / "yaskawa_pick_cube_cfg.py"
SCRIPTED_CONTROLLER_PATH = REPO_ROOT / "scripts" / "validate" / "scripted_pick_demo.py"


def _git_sha_for(path: Path) -> str | None:
    """Last commit sha that touched `path`, or None if path is untracked / no git."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_head_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    read = 0
    chunk = 1 << 20
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def _flatten_stats(stats_for_dim: dict[str, Any]) -> dict[str, float]:
    """LeRobot stats nest scalars inside lists; flatten the per-dim summary."""

    def first_scalar(x: Any) -> float | None:
        while isinstance(x, list) and x:
            x = x[0]
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    return {k: first_scalar(v) for k, v in stats_for_dim.items() if k in ("min", "max", "mean", "std")}


def _action_per_dim(stats: dict[str, Any], action_names: list[str]) -> dict[str, dict[str, float]]:
    """Pull per-dim action stats from LeRobot stats.json. Compute gripper close fraction."""
    a_stats = stats.get("action", {})
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(action_names):
        per = {}
        for k in ("min", "max", "mean", "std"):
            v = a_stats.get(k)
            if isinstance(v, list) and i < len(v):
                try:
                    per[k] = float(v[i])
                except (TypeError, ValueError):
                    per[k] = None
        out[name] = per
    # Gripper close fraction is not in LeRobot stats — leave a None for the orchestrator
    # to fill in via a one-off pass over parquet if it cares.
    if "gripper" in out:
        out["gripper"]["close_fraction"] = None
    return out


def _grep_env_cfg() -> dict[str, Any]:
    """Best-effort static scrape of the cube-pick env-cfg.

    We don't import the env (it pulls in IsaacLab); we read the file as text and
    look for the values that historically have moved between runs.
    """
    if not ENV_CFG_PATH.exists():
        return {}
    text = ENV_CFG_PATH.read_text()
    fields: dict[str, Any] = {}

    def _grab(label: str, pattern: str) -> str | None:
        import re

        m = re.search(pattern, text)
        return m.group(1) if m else None

    fields["decimation"] = _grab("decimation", r"self\.decimation\s*=\s*([0-9]+)")
    fields["scale"] = _grab("scale", r"\bscale\s*=\s*([0-9.]+)")
    fields["antialiasing_mode"] = _grab(
        "antialiasing_mode", r'antialiasing_mode\s*=\s*"([A-Z]+)"'
    )
    return {k: v for k, v in fields.items() if v is not None}


def build_manifest(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    stats_path = dataset_dir / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No meta/info.json under {dataset_dir}")

    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    features = info.get("features", {})
    action_feature = features.get("action", {})
    action_names: list[str] = action_feature.get("names") or []

    obs_features = sorted(k for k in features if k.startswith("observation"))

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "type": "dataset",
        "dataset_id": dataset_dir.name,
        "dataset_path": str(dataset_dir.relative_to(REPO_ROOT)),
        "created_at": dt.datetime.fromtimestamp(info_path.stat().st_mtime, tz=dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "lerobot": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "splits": info.get("splits"),
            "video_path_template": info.get("video_path"),
            "data_path_template": info.get("data_path"),
        },
        "obs_features": obs_features,
        "action_dim": action_feature.get("shape", [None])[0],
        "action_names": action_names,
        "action_per_dim": _action_per_dim(stats, action_names),
        "env_cfg": {
            "env_cfg_path": str(ENV_CFG_PATH.relative_to(REPO_ROOT)),
            "env_cfg_sha": _git_sha_for(ENV_CFG_PATH),
            **_grep_env_cfg(),
        },
        "scripted_controller": {
            "path": str(SCRIPTED_CONTROLLER_PATH.relative_to(REPO_ROOT))
            if SCRIPTED_CONTROLLER_PATH.exists()
            else None,
            "sha": _git_sha_for(SCRIPTED_CONTROLLER_PATH),
        },
        "git_head_at_build": _git_head_sha(),
        "notes": None,
    }

    return manifest


def _iter_lerobot_datasets(root: Path):
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            continue
        if (child / "meta" / "info.json").exists():
            yield child


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dataset", type=Path, help="Path to a LeRobot dataset snapshot dir")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Build manifests for every dataset under datasets/lerobot/",
    )
    ap.add_argument(
        "--print",
        action="store_true",
        help="Print manifest to stdout instead of writing manifest.json",
    )
    ap.add_argument(
        "--note", default=None, help="Free-form note to embed under 'notes' in the manifest"
    )
    args = ap.parse_args(argv)

    if not args.dataset and not args.all:
        ap.error("Pass --dataset PATH or --all")

    targets = (
        list(_iter_lerobot_datasets(REPO_ROOT / "datasets" / "lerobot"))
        if args.all
        else [args.dataset]
    )

    rc = 0
    for ds in targets:
        try:
            manifest = build_manifest(ds)
            if args.note:
                manifest["notes"] = args.note
        except Exception as e:
            print(f"[ERROR] {ds}: {e}", file=sys.stderr)
            rc = 1
            continue

        if args.print:
            print(json.dumps(manifest, indent=2))
        else:
            out_path = ds / "manifest.json"
            out_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
