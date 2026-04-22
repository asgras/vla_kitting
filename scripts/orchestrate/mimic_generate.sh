#!/bin/bash
# Mimic generation orchestrator — Phase A of the split pipeline.
#
# Does ONLY dataset generation; never touches training. Exits cleanly when
# either MIMIC_TARGET_DEMOS is reached OR MIMIC_BUDGET_HOURS elapses.
#
# Steps:
#   1. Seed:     scripted pick (if SCRIPTED missing) → clean → annotate
#   2. Mimic loop: batches of MIMIC_BATCH trials, merged into MASTER.
#   3. Convert:  write LeRobot v3 dataset at datasets/lerobot/cube_pick_v1.
#
# Usage:
#   bash scripts/orchestrate/mimic_generate.sh              # resume/extend
#   bash scripts/orchestrate/mimic_generate.sh --reset      # wipe datasets first
#
# Stop early with: touch logs/mimic/STOP

set -euo pipefail
export LC_ALL=C.UTF-8

REPO=/home/ubuntu/vla_kitting
ISAAC_LAB=/home/ubuntu/IsaacLab
ISAAC_PY=/home/ubuntu/IsaacLab/_isaac_sim/python.sh
LEROBOT_SRC=/home/ubuntu/code/lerobot/src
VENV_PY=$REPO/.venv/bin/python

# --- Knobs (override via env) ---
SEED_DEMOS=${SEED_DEMOS:-25}
MIMIC_BATCH=${MIMIC_BATCH:-25}
MIMIC_TARGET_DEMOS=${MIMIC_TARGET_DEMOS:-150}
MIMIC_BUDGET_HOURS=${MIMIC_BUDGET_HOURS:-4}

# --- Paths ---
TELEOP=$REPO/datasets/teleop
MIMIC=$REPO/datasets/mimic
POOL=$MIMIC/pool
LEROBOT_ROOT=$REPO/datasets/lerobot
LEROBOT_LIVE=$LEROBOT_ROOT/cube_pick_v1
LOG_DIR=$REPO/logs/mimic

SCRIPTED=$TELEOP/cube_scripted.hdf5
SCRIPTED_CLEAN=$TELEOP/cube_scripted_clean.hdf5
ANNOTATED=$TELEOP/cube_annotated.hdf5
MASTER=$MIMIC/cube_mimic_all.hdf5

MIMIC_LOG=$LOG_DIR/mimic_loop.log
BATCH_JSONL=$LOG_DIR/batch_summary.jsonl
SUMMARY_JSON=$LOG_DIR/mimic_summary.json
STATE_JSON=$LOG_DIR/state.json
STOP_FILE=$LOG_DIR/STOP

mkdir -p "$TELEOP" "$MIMIC" "$POOL" "$LEROBOT_ROOT" "$LOG_DIR"

# --- Reset handling ---
if [[ "${1:-}" == "--reset" ]]; then
  echo "[mimic] --reset: wiping mimic state (keeping scripted seed)"
  rm -f "$MASTER"
  rm -rf "$POOL" "$LEROBOT_LIVE" "$LEROBOT_ROOT"/cube_pick_v1_batch_*
  rm -f "$BATCH_JSONL" "$STATE_JSON" "$STOP_FILE" "$SUMMARY_JSON"
  # Seed (scripted + clean + annotate) is deliberately preserved — those
  # files take ~1h to regenerate and don't need to change between runs.
  # Use `rm datasets/teleop/cube_scripted.hdf5` manually if you want to
  # regenerate them.
  mkdir -p "$POOL"
fi

_log() { echo "[$(date +%H:%M:%S)] $*"; }

_count_master_demos() {
  if [[ ! -f "$MASTER" ]]; then echo 0; return; fi
  $ISAAC_PY - "$MASTER" <<'PY'
import h5py, sys
p = sys.argv[1]
try:
    with h5py.File(p, "r") as f:
        d = f["data"]
        n = sum(1 for k in d if bool(d[k].attrs.get("success", False)))
        print(n)
except Exception:
    print(0)
PY
}

_check_stop() {
  [[ -f "$STOP_FILE" ]] && { _log "STOP file detected — exiting gracefully"; return 0; }
  return 1
}

# ============================================================
# Phase 1 — Seed: scripted demos, clean, annotate (one-time)
# ============================================================
phase_seed() {
  _log "=== Phase 1: seed ==="

  if [[ ! -f "$SCRIPTED" ]]; then
    _log "generating $SEED_DEMOS scripted demos (~45 min)"
    cd "$ISAAC_LAB"
    ./isaaclab.sh -p "$REPO"/scripts/validate/scripted_pick_demo.py \
      --num_demos "$SEED_DEMOS" --max_steps_per_demo 2000 \
      --dataset_file "$SCRIPTED" 2>&1 | tee -a "$MIMIC_LOG" \
      | grep -E "scripted_pick.*(attempt|success|total)" || true
  else
    _log "scripted demos already exist → $SCRIPTED (skipping)"
  fi

  if [[ ! -f "$SCRIPTED_CLEAN" ]]; then
    _log "cleaning scripted demos"
    $ISAAC_PY "$REPO"/scripts/data/clean_demos.py \
      --input "$SCRIPTED" --output "$SCRIPTED_CLEAN" 2>&1 | tee -a "$MIMIC_LOG"
  else
    _log "cleaned seed exists → $SCRIPTED_CLEAN (skipping)"
  fi

  if [[ ! -f "$ANNOTATED" ]]; then
    _log "running Mimic annotation (~15-25 min)"
    cd "$ISAAC_LAB"
    ./isaaclab.sh -p "$REPO"/scripts/data/annotate_demos.py \
      --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
      --input_file "$SCRIPTED_CLEAN" --output_file "$ANNOTATED" \
      --auto --headless --enable_cameras 2>&1 | tee -a "$MIMIC_LOG" | tail -40
  else
    _log "annotated seed exists → $ANNOTATED (skipping)"
  fi

  _log "seed phase complete"
}

# ============================================================
# Phase 2 — Mimic batch loop (runs until target demos or budget)
# ============================================================
mimic_loop() {
  _log "=== Phase 2: Mimic batch loop starting ==="

  local start_ts=$(date +%s)
  local budget_s=$(echo "$MIMIC_BUDGET_HOURS * 3600" | bc -l | cut -d. -f1)
  local batch_num=$(ls "$POOL"/batch_*.hdf5 2>/dev/null | wc -l)

  while true; do
    _check_stop && break

    local now=$(date +%s)
    local elapsed=$((now - start_ts))
    if (( elapsed >= budget_s )); then
      _log "[mimic] budget $MIMIC_BUDGET_HOURS h reached (elapsed=${elapsed}s); exiting loop"
      break
    fi

    local current=$(_count_master_demos)
    if (( current >= MIMIC_TARGET_DEMOS )); then
      _log "[mimic] reached target $MIMIC_TARGET_DEMOS demos (have $current); exiting loop"
      break
    fi

    batch_num=$((batch_num + 1))
    local batch_file=$(printf "$POOL/batch_%03d.hdf5" "$batch_num")

    _log "[mimic] batch $batch_num starting (have $current/$MIMIC_TARGET_DEMOS) → $batch_file"
    cd "$ISAAC_LAB"
    set +e
    ./isaaclab.sh -p "$REPO"/scripts/data/generate_dataset.py \
      --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
      --input_file "$ANNOTATED" \
      --output_file "$batch_file" \
      --generation_num_trials "$MIMIC_BATCH" \
      --num_envs 1 --headless --enable_cameras 2>&1 | tail -200 \
      >> "$MIMIC_LOG"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      _log "[mimic] batch $batch_num failed (rc=$rc); continuing"
      continue
    fi

    if [[ -f "$batch_file" ]]; then
      _log "[mimic] merging pool → master"
      local stats=$($ISAAC_PY "$REPO"/scripts/orchestrate/merge_mimic_pool.py \
        --pool "$POOL" --output "$MASTER" 2>&1 | tail -1)
      local total=$(echo "$stats" | $VENV_PY -c \
        "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('total_demos',0))")

      # Append JSONL summary.
      $VENV_PY - "$BATCH_JSONL" "$batch_num" "$total" <<'PY'
import json, sys, datetime
p, batch, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
line = {
  "ts": datetime.datetime.utcnow().isoformat() + "Z",
  "batch": batch, "total_demos": total,
}
with open(p, "a") as f:
    f.write(json.dumps(line) + "\n")
PY
      _log "[mimic] batch $batch_num merged: total=$total demos"
    fi
  done
  _log "[mimic] loop exited"
}

# ============================================================
# Phase 3 — Convert merged master to LeRobot v3 dataset (once)
# ============================================================
phase_convert() {
  _log "=== Phase 3: LeRobot conversion ==="

  local total=$(_count_master_demos)
  if (( total == 0 )); then
    _log "[convert] master has 0 demos; skipping conversion"
    return
  fi

  # Versioned output dir + atomic symlink swap. Unlike the old interleaved
  # pipeline, we only do this ONCE at the end, not per-batch.
  local ts=$(date +%Y%m%d_%H%M%S)
  local dst=$LEROBOT_ROOT/cube_pick_v1_$ts
  _log "[convert] writing LeRobot dataset ($total demos) → $dst"

  PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$VENV_PY" \
    "$REPO"/scripts/data/isaaclab_to_lerobot.py \
    --input "$MASTER" --output "$dst" \
    --repo_id vla_kitting/cube_pick_v1 \
    --task "pick up the cube and place it on the green target" \
    2>&1 | tee -a "$MIMIC_LOG" | tail -10

  ln -sfn "$dst" "$LEROBOT_LIVE.next"
  mv -Tf "$LEROBOT_LIVE.next" "$LEROBOT_LIVE"
  _log "[convert] cube_pick_v1 symlink → $dst"

  # Emit summary JSON so downstream train_only.sh can inspect what we built.
  $VENV_PY - "$SUMMARY_JSON" "$total" "$dst" <<'PY'
import json, sys, datetime
p, total, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
summary = {
  "ts": datetime.datetime.utcnow().isoformat() + "Z",
  "total_demos": total,
  "lerobot_dataset": dst,
}
with open(p, "w") as f:
    f.write(json.dumps(summary, indent=2))
PY
  _log "[convert] summary → $SUMMARY_JSON"
}

# ============================================================
# Main
# ============================================================
_log "=========================================================="
_log "mimic_generate.sh starting (pid=$$)"
_log "  SEED_DEMOS=$SEED_DEMOS MIMIC_BATCH=$MIMIC_BATCH"
_log "  MIMIC_TARGET_DEMOS=$MIMIC_TARGET_DEMOS MIMIC_BUDGET_HOURS=$MIMIC_BUDGET_HOURS"
_log "=========================================================="

phase_seed
mimic_loop
phase_convert

_log "mimic_generate.sh done"
