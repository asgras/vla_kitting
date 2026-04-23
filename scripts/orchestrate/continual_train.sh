#!/bin/bash
# Continual train + Mimic orchestrator. See reports/continual_training_plan.md
# for the full spec and resumption contract.
#
# Usage:
#   bash scripts/orchestrate/continual_train.sh            # resume if possible
#   bash scripts/orchestrate/continual_train.sh --reset    # wipe datasets/checkpoints first
#
# Stop cleanly with:  touch logs/continual/STOP

set -euo pipefail
export LC_ALL=C.UTF-8

REPO=/home/ubuntu/vla_kitting
ISAAC_LAB=/home/ubuntu/IsaacLab
ISAAC_PY=/home/ubuntu/IsaacLab/_isaac_sim/python.sh
LEROBOT_SRC=/home/ubuntu/code/lerobot/src
VENV_PY=$REPO/.venv/bin/python

# --- Knobs (override via env) ---
SEED_DEMOS=${SEED_DEMOS:-25}
MIMIC_MIN_DEMOS=${MIMIC_MIN_DEMOS:-20}
MIMIC_BATCH=${MIMIC_BATCH:-25}
TRAIN_STEPS=${TRAIN_STEPS:-1000}
TRAIN_BATCH=${TRAIN_BATCH:-4}
EVAL_EPISODES=${EVAL_EPISODES:-2}
EVAL_EVERY_N=${EVAL_EVERY_N:-5}
USE_LORA=${USE_LORA:-1}
LORA_R=${LORA_R:-16}

# --- Paths ---
TELEOP=$REPO/datasets/teleop
MIMIC=$REPO/datasets/mimic
POOL=$MIMIC/pool
LEROBOT_ROOT=$REPO/datasets/lerobot
LEROBOT_LIVE=$LEROBOT_ROOT/cube_pick_v1
CKPT_DIR=$REPO/checkpoints/continual
LOG_DIR=$REPO/logs/continual

SCRIPTED=$TELEOP/cube_scripted.hdf5
SCRIPTED_CLEAN=$TELEOP/cube_scripted_clean.hdf5
ANNOTATED=$TELEOP/cube_annotated.hdf5
MASTER=$MIMIC/cube_mimic_all.hdf5

MIMIC_LOG=$LOG_DIR/mimic_loop.log
TRAIN_LOG=$LOG_DIR/train_loop.log
BATCH_JSONL=$LOG_DIR/batch_summary.jsonl
EPOCH_JSONL=$LOG_DIR/epoch_summary.jsonl
STATE_JSON=$LOG_DIR/state.json
STOP_FILE=$LOG_DIR/STOP

mkdir -p "$TELEOP" "$MIMIC" "$POOL" "$LEROBOT_ROOT" "$CKPT_DIR" "$LOG_DIR"

# --- Reset handling ---
if [[ "${1:-}" == "--reset" ]]; then
  echo "[orch] --reset: wiping continual state"
  rm -f "$SCRIPTED" "$SCRIPTED_CLEAN" "$ANNOTATED" "$MASTER"
  rm -rf "$POOL" "$LEROBOT_LIVE" "$LEROBOT_ROOT"/cube_pick_v1_batch_* "$CKPT_DIR"
  rm -f "$BATCH_JSONL" "$EPOCH_JSONL" "$STATE_JSON" "$STOP_FILE"
  mkdir -p "$POOL" "$CKPT_DIR"
fi

_log() { echo "[$(date +%H:%M:%S)] $*"; }

_state_write() {
  local field=$1 value=$2
  $VENV_PY - "$STATE_JSON" "$field" "$value" <<'PY'
import json, os, sys, pathlib
p = pathlib.Path(sys.argv[1])
field, value = sys.argv[2], sys.argv[3]
try: value = json.loads(value)
except Exception: pass
data = {}
if p.exists():
    try: data = json.loads(p.read_text())
    except Exception: data = {}
data[field] = value
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2))
tmp.replace(p)
PY
}

_count_master_demos() {
  # Prints the count of successful demos in $MASTER, or 0 if missing/empty.
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
      | grep -E "scripted_pick.*(attempt|success|total)"
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

  _state_write seed_complete true
  _log "seed phase complete"
}

# ============================================================
# Phase 2 — Mimic batch loop (runs forever in background)
# ============================================================
mimic_loop() {
  _log "=== Phase 2: Mimic batch loop starting ==="
  local batch_num=$(ls "$POOL"/batch_*.hdf5 2>/dev/null | wc -l)

  while true; do
    _check_stop && break
    batch_num=$((batch_num + 1))
    local batch_file=$(printf "$POOL/batch_%03d.hdf5" "$batch_num")

    _log "[mimic] batch $batch_num starting → $batch_file"
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
    fi

    if [[ -f "$batch_file" ]]; then
      _log "[mimic] merging pool → master"
      local stats=$($ISAAC_PY "$REPO"/scripts/orchestrate/merge_mimic_pool.py \
        --pool "$POOL" --output "$MASTER" 2>&1 | grep -E '^\{' | tail -1)
      local total=$(echo "$stats" | $VENV_PY -c \
        "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('total_demos',0))")

      # Rebuild LeRobot snapshot for this batch (training will pick this up
      # on its next epoch). Use a versioned dir + symlink swap so training
      # never reads a half-written dataset.
      if [[ "$total" -gt 0 ]]; then
        local snapshot=$(printf "$LEROBOT_ROOT/cube_pick_v1_batch_%03d" "$batch_num")
        _log "[mimic] rebuilding LeRobot snapshot → $snapshot"
        rm -rf "$snapshot"
        PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$VENV_PY" \
          "$REPO"/scripts/data/isaaclab_to_lerobot.py \
          --input "$MASTER" --output "$snapshot" \
          --repo_id vla_kitting/cube_pick_v1 \
          --task "pick up the cube and place it on the green target" \
          2>&1 | tail -5 >> "$MIMIC_LOG"

        # Atomic symlink swap.
        ln -sfn "$snapshot" "$LEROBOT_LIVE.next"
        mv -Tf "$LEROBOT_LIVE.next" "$LEROBOT_LIVE"

        # Disk hygiene: keep only the 3 most-recent snapshots. Older ones
        # aren't needed once the symlink has moved on.
        ls -dt "$LEROBOT_ROOT"/cube_pick_v1_batch_* 2>/dev/null \
          | tail -n +4 | xargs -r rm -rf
      fi

      # Append JSONL summary and update state.json.
      $VENV_PY - "$BATCH_JSONL" "$batch_num" "$total" <<'PY'
import json, sys, datetime, pathlib
p, batch, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
line = {
  "ts": datetime.datetime.utcnow().isoformat() + "Z",
  "batch": batch, "total_demos": total,
}
with open(p, "a") as f:
    f.write(json.dumps(line) + "\n")
PY
      _state_write mimic_batch "$batch_num"
      _state_write demos_total "$total"
      _log "[mimic] batch $batch_num merged: total=$total demos"
    fi
  done
  _log "[mimic] loop exited"
}

# ============================================================
# Phase 3 — Training loop (runs forever, blocks until >= min demos)
# ============================================================
train_loop() {
  _log "=== Phase 3: Training loop starting ==="
  _log "[train] waiting for at least $MIMIC_MIN_DEMOS Mimic demos..."
  while true; do
    _check_stop && return
    local n=$(_count_master_demos)
    if [[ "$n" -ge "$MIMIC_MIN_DEMOS" ]]; then
      _log "[train] have $n demos (≥ $MIMIC_MIN_DEMOS); starting training"
      break
    fi
    sleep 60
  done

  local epoch=$(jq -r '.epoch // 0' "$STATE_JSON" 2>/dev/null || echo 0)
  local last_ckpt=""

  while true; do
    _check_stop && break
    epoch=$((epoch + 1))
    local num_demos=$(_count_master_demos)

    _log "[train] === epoch $epoch (num_demos=$num_demos, steps=$TRAIN_STEPS) ==="

    # Ensure symlinked live dataset exists.
    if [[ ! -e "$LEROBOT_LIVE" ]]; then
      _log "[train] no LeRobot dataset yet; sleeping 60s"
      sleep 60
      continue
    fi

    # Training args.
    local train_args=(
      --dataset.repo_id=vla_kitting/cube_pick_v1
      --dataset.root="$LEROBOT_LIVE"
      --policy.device=cuda
      --policy.push_to_hub=false
      --policy.repo_id=vla_kitting/cube_pick_v1
      --output_dir="$CKPT_DIR"
      --batch_size="$TRAIN_BATCH"
      --steps="$((epoch * TRAIN_STEPS))"
      --log_freq=100 --save_freq="$TRAIN_STEPS"
      --wandb.enable=false
    )

    if [[ -e "$CKPT_DIR/checkpoints/last" ]]; then
      # Resume from last checkpoint. LeRobot computes policy_dir = Path(config_path).parent
      # (no symlink resolution) and passes that as pretrained_path to PEFT. So we must
      # point config_path at the train_config.json FILE inside pretrained_model/ — not the
      # dir itself — so that .parent lands on pretrained_model/ where adapter_config.json lives.
      local resume_cfg_path="$CKPT_DIR/checkpoints/last/pretrained_model/train_config.json"
      train_args+=(--config_path="$resume_cfg_path" --resume=true)
      _log "[train] resuming from $CKPT_DIR/checkpoints/last"
    else
      # First epoch — fresh SmolVLA-base. LeRobot refuses to write into an
      # existing non-empty dir with resume=false, so scrub any empty dir
      # created by our top-level mkdir -p.
      if [[ -d "$CKPT_DIR" && -z "$(ls -A "$CKPT_DIR")" ]]; then
        rmdir "$CKPT_DIR"
      fi
      train_args+=(--policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base)
      if [[ "$USE_LORA" == "1" ]]; then
        train_args+=(--peft.method_type=LORA --peft.r="$LORA_R")
        _log "[train] fresh start from lerobot/smolvla_base + LoRA r=$LORA_R"
      else
        _log "[train] fresh start from lerobot/smolvla_base (full fine-tune)"
      fi
    fi

    set +e
    PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$VENV_PY" \
      "$LEROBOT_SRC"/lerobot/scripts/lerobot_train.py \
      "${train_args[@]}" 2>&1 | tail -200 >> "$TRAIN_LOG"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      _log "[train] epoch $epoch failed (rc=$rc); sleeping 60s and retrying"
      sleep 60
      continue
    fi

    last_ckpt="$CKPT_DIR/checkpoints/last/pretrained_model"

    # Eval every Nth epoch.
    local sr=-1
    if (( epoch % EVAL_EVERY_N == 0 )); then
      _log "[train] eval at epoch $epoch ($EVAL_EPISODES episodes)"
      local eval_out="$CKPT_DIR/eval_epoch_$(printf '%04d' $epoch)"
      mkdir -p "$(dirname "$eval_out")"
      set +e
      PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$ISAAC_LAB"/isaaclab.sh \
        -p "$REPO"/scripts/train/run_vla_closed_loop.py \
        --checkpoint "$last_ckpt" \
        --num_episodes "$EVAL_EPISODES" --max_steps 1800 \
        --save_gif "$eval_out.gif" 2>&1 | tail -50 >> "$TRAIN_LOG"
      sr=$(grep -oP 'total: \K[0-9]+/[0-9]+' "$TRAIN_LOG" | tail -1 \
           | awk -F/ '{ if ($2 > 0) printf "%.2f", $1/$2; else print "0" }')
      set -e
    fi

    # Grab last loss reported in the train log.
    local loss=$(grep -oP "loss:\K[0-9.]+" "$TRAIN_LOG" | tail -1 || echo "nan")

    $VENV_PY - "$EPOCH_JSONL" "$epoch" "$num_demos" "$loss" "$sr" <<'PY'
import json, sys, datetime
p, epoch, nd, loss, sr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
line = {
  "ts": datetime.datetime.utcnow().isoformat() + "Z",
  "epoch": epoch, "num_demos": nd,
  "loss": float(loss) if loss not in ("", "nan") else None,
  "eval_sr": float(sr) if sr not in ("-1", "", "nan") else None,
}
with open(p, "a") as f:
    f.write(json.dumps(line) + "\n")
PY
    _state_write epoch "$epoch"
    _state_write last_ckpt "$last_ckpt"

    _log "[train] epoch $epoch done (loss=$loss, sr=$sr, demos=$num_demos)"
  done
  _log "[train] loop exited"
}

# ============================================================
# Main
# ============================================================
_log "=========================================================="
_log "continual_train.sh starting (pid=$$)"
_log "  SEED_DEMOS=$SEED_DEMOS MIMIC_MIN_DEMOS=$MIMIC_MIN_DEMOS"
_log "  MIMIC_BATCH=$MIMIC_BATCH TRAIN_STEPS=$TRAIN_STEPS TRAIN_BATCH=$TRAIN_BATCH"
_log "=========================================================="

phase_seed

# Start Mimic loop in background.
(mimic_loop) &
MIMIC_PID=$!
_log "[orch] mimic loop pid=$MIMIC_PID"

# Training loop in foreground.
train_loop

# If training exits (STOP signaled), bring down the Mimic loop too.
if kill -0 "$MIMIC_PID" 2>/dev/null; then
  _log "[orch] stopping mimic loop (pid=$MIMIC_PID)"
  kill "$MIMIC_PID" 2>/dev/null || true
  wait "$MIMIC_PID" 2>/dev/null || true
fi

_log "continual_train.sh done"
