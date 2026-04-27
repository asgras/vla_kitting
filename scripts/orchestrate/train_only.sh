#!/bin/bash
# Training orchestrator — Phase B of the split pipeline.
#
# Assumes datasets/lerobot/cube_pick_v1 exists and is frozen (produced by
# mimic_generate.sh). Does NOT run Mimic in parallel — that's the whole point
# of the split. Loops over lerobot_train.py with all the bugfixes baked in:
#
#   - strips cfg.peft before each resume (fixes double-PEFT-wrap)
#   - pins constant LR on fresh start (fixes LR-decay-to-floor)
#   - runs adapter_fixer daemon that normalizes saved checkpoints
#   - advances the epoch counter only on successful calls
#   - streams per-step metrics into logs/continual/train_steps.jsonl
#
# Usage:
#   bash scripts/orchestrate/train_only.sh           # resume if possible
#   bash scripts/orchestrate/train_only.sh --reset   # wipe checkpoints first
#
# Stop cleanly with:  touch logs/continual/STOP
#
# Environment knobs:
#   TRAIN_STEPS          steps per epoch call            (default 1000)
#   TRAIN_SAVE_FREQ      save_freq within each call      (default =TRAIN_STEPS)
#   TRAIN_BATCH          batch size                      (default 16)
#   TRAIN_LR             constant learning rate          (default 1e-3)
#   EVAL_EPISODES        rollouts per eval               (default 10)
#   EVAL_EVERY_N         eval every N epochs             (default 10)
#   USE_LORA             1/0 for LoRA                    (default 1)
#   LORA_R               LoRA rank                       (default 64)
#   LORA_ALPHA           LoRA alpha (scaling)            (default 64)
#   LORA_DROPOUT         LoRA dropout                    (default 0.05)
#   LORA_TARGETS_REGEX   regex of module names to adapt  (default: q,k,v,o + gate,up,down in lm_expert + action projections; vision tower NOT included)
#   N_ACTION_STEPS       action-chunk steps executed     (default 10)

set -euo pipefail
export LC_ALL=C.UTF-8

REPO=/home/ubuntu/vla_kitting
ISAAC_LAB=/home/ubuntu/IsaacLab
LEROBOT_SRC=/home/ubuntu/code/lerobot/src
VENV_PY=$REPO/.venv/bin/python

# --- Knobs ---
TRAIN_STEPS=${TRAIN_STEPS:-1000}
TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-$TRAIN_STEPS}
TRAIN_BATCH=${TRAIN_BATCH:-4}
TRAIN_LR=${TRAIN_LR:-1e-4}
EVAL_EPISODES=${EVAL_EPISODES:-10}
EVAL_EVERY_N=${EVAL_EVERY_N:-10}
USE_LORA=${USE_LORA:-1}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# v3.1 (2026-04-25): vision tower LoRA RE-ENABLED. v3 froze the vision tower
# entirely per SmolVLA paper canonical (real-robot finding from "Accessible
# Physical AI"). At epoch 30 the policy showed mode collapse — cube was
# pushed but never lifted, eval cubes received the same approach trajectory
# regardless of cube location. Diagnosis: frozen internet-pretrained ViT
# features lacked spatial discrimination for our synthetic Isaac Sim scene
# (small cube on brown table, fixed third-person camera). Re-adding LoRA on
# vision attention projections gives the encoder a small task-specific
# surface to specialize cube-position features. Module path verified by
# instantiating SmolVLM2 — uses `self_attn` (not self_attention) and
# `out_proj` (not o_proj).
LORA_TARGETS_REGEX=${LORA_TARGETS_REGEX:-"(model\.vlm_with_expert\.(lm_expert\..*\.(q|k|v|o|gate|up|down)_proj|vlm\.model\.vision_model\.encoder\.layers\..*\.self_attn\.(q|k|v|out)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"}
# v3.2 (2026-04-25): N_ACTION_STEPS reduced to 10 to test hypothesis 3.
# At chunk_size=50 (pretrain default), the policy commits to 50 steps of
# action (≈1.6s at 30Hz) before re-querying observations. Combined with our
# saturated P-controller demos producing stereotyped initial trajectories,
# this likely makes the model emit a memorized stereotyped chunk that
# ignores mid-execution visual feedback. With n_action_steps=10, the policy
# re-queries every 10 steps (~0.33s), giving 5× more visual feedback within
# an episode. chunk_size remains 50 internally (the prediction horizon),
# only the EXECUTION horizon shrinks. Trade-off: 5× more inference calls
# per eval episode → eval ~5× slower (~8.5 min for 10 episodes vs 7 min).
N_ACTION_STEPS=${N_ACTION_STEPS:-10}

# Budget watchdog. Set BUDGET_HOURS=0 to disable.
BUDGET_HOURS=${BUDGET_HOURS:-0}

# --- Paths ---
LEROBOT_ROOT=$REPO/datasets/lerobot
LEROBOT_LIVE=$LEROBOT_ROOT/cube_pick_v1
CKPT_DIR=$REPO/checkpoints/continual
LOG_DIR=$REPO/logs/continual

EPOCH_JSONL=$LOG_DIR/epoch_summary.jsonl
STEP_JSONL=$LOG_DIR/train_steps.jsonl
STATE_JSON=$LOG_DIR/state.json
STOP_FILE=$LOG_DIR/STOP
TRAIN_LOG=$LOG_DIR/train_loop.log

mkdir -p "$LOG_DIR"

# --- Reset handling ---
if [[ "${1:-}" == "--reset" ]]; then
  echo "[train] --reset: wiping checkpoints and training logs"
  rm -rf "$CKPT_DIR"
  rm -f "$EPOCH_JSONL" "$STEP_JSONL" "$STATE_JSON" "$STOP_FILE" "$TRAIN_LOG"
fi

_log() { echo "[$(date +%H:%M:%S)] $*"; }

_state_write() {
  local field=$1 value=$2
  $VENV_PY - "$STATE_JSON" "$field" "$value" <<'PY'
import json, sys, pathlib
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

_check_stop() {
  [[ -f "$STOP_FILE" ]] && { _log "STOP detected — exiting"; return 0; }
  return 1
}

# ============================================================
# Pre-flight: dataset must exist and load
# ============================================================
if [[ ! -e "$LEROBOT_LIVE" ]]; then
  _log "ERROR: $LEROBOT_LIVE does not exist. Run mimic_generate.sh first."
  exit 2
fi
_log "dataset: $LEROBOT_LIVE → $(readlink -f "$LEROBOT_LIVE" 2>/dev/null || echo "$LEROBOT_LIVE")"

# ============================================================
# Background: adapter_fixer daemon (normalizes every saved checkpoint)
# ============================================================
(
  "$VENV_PY" "$REPO"/scripts/orchestrate/fix_adapter_configs.py \
    --ckpt-dir "$CKPT_DIR" --log-dir "$LOG_DIR" \
    > "$LOG_DIR/adapter_fixer.out" 2>&1
) &
FIXER_PID=$!
_log "adapter_fixer pid=$FIXER_PID"

WATCHDOG_PID=""
if [[ "$BUDGET_HOURS" != "0" ]]; then
  (
    "$VENV_PY" "$REPO"/scripts/orchestrate/budget_watchdog.py \
      --log-dir "$LOG_DIR" --budget-hours "$BUDGET_HOURS" \
      --grace-minutes "$(echo "$BUDGET_HOURS * 60" | bc -l | cut -d. -f1)" \
      > "$LOG_DIR/watchdog.out" 2>&1
  ) &
  WATCHDOG_PID=$!
  _log "budget_watchdog pid=$WATCHDOG_PID budget=${BUDGET_HOURS}h"
fi

_cleanup() {
  _log "cleanup: stopping adapter_fixer pid=$FIXER_PID"
  kill "$FIXER_PID" 2>/dev/null || true
  wait "$FIXER_PID" 2>/dev/null || true
  if [[ -n "$WATCHDOG_PID" ]]; then
    _log "cleanup: stopping budget_watchdog pid=$WATCHDOG_PID"
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap _cleanup EXIT

# ============================================================
# Main training loop
# ============================================================
_log "=== training loop starting ==="
_log "  TRAIN_STEPS=$TRAIN_STEPS SAVE_FREQ=$TRAIN_SAVE_FREQ BATCH=$TRAIN_BATCH LR=$TRAIN_LR"
_log "  EVAL_EVERY_N=$EVAL_EVERY_N EVAL_EPISODES=$EVAL_EPISODES USE_LORA=$USE_LORA"
_log "  LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT N_ACTION_STEPS=$N_ACTION_STEPS"
_log "  LORA_TARGETS_REGEX=$LORA_TARGETS_REGEX"

epoch=$(jq -r '.epoch // 0' "$STATE_JSON" 2>/dev/null || echo 0)

while true; do
  _check_stop && break

  # NOTE: epoch is NOT incremented here — we increment only AFTER a successful
  # train call below. This prevents the counter from inflating during any
  # transient waits or failed calls (old bug: --steps=$((epoch * TRAIN_STEPS))
  # blew up when the counter ticked during dataset waits).

  next_epoch=$((epoch + 1))
  _log "=== epoch $next_epoch (steps target=$((next_epoch * TRAIN_STEPS))) ==="

  # Build training args.
  train_args=(
    --dataset.repo_id=vla_kitting/cube_pick_v1
    --dataset.root="$LEROBOT_LIVE"
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.repo_id=vla_kitting/cube_pick_v1
    --output_dir="$CKPT_DIR"
    --batch_size="$TRAIN_BATCH"
    --steps="$((next_epoch * TRAIN_STEPS))"
    --log_freq=50 --save_freq="$TRAIN_SAVE_FREQ"
    --wandb.enable=false
  )

  if [[ -e "$CKPT_DIR/checkpoints/last" ]]; then
    # Resume branch. Strip the peft: section so lerobot_train.py doesn't
    # double-wrap. Point config_path at the JSON FILE so .parent resolves
    # to pretrained_model/ where adapter_config.json actually lives.
    resume_cfg="$CKPT_DIR/checkpoints/last/pretrained_model/train_config.json"
    _log "resuming from $CKPT_DIR/checkpoints/last"
    "$VENV_PY" "$REPO"/scripts/orchestrate/prepare_resume_config.py \
      --config "$resume_cfg" 2>&1 | tee -a "$TRAIN_LOG" || true
    train_args+=(--config_path="$resume_cfg" --resume=true)
  else
    # Fresh-start branch. LeRobot refuses to write into an existing non-empty
    # dir with resume=false, so scrub any empty dir the env left behind.
    if [[ -d "$CKPT_DIR" && -z "$(ls -A "$CKPT_DIR")" ]]; then
      rmdir "$CKPT_DIR"
    fi
    train_args+=(
      --policy.type=smolvla
      --policy.pretrained_path=lerobot/smolvla_base
      # CRITICAL: load_vlm_weights=true is required to actually load the
      # pretrained SmolVLM2-500M backbone. Without this, the VLM is randomly
      # initialized and you train from scratch — the warning from
      # _validate_peft_config: "Training SmolVLA from scratch using PEFT.
      # This is unlikely to yield good results." Default in
      # configuration_smolvla.py is False (designed for from-scratch training
      # of the action expert with VLM frozen-pretrained); we want True.
      --policy.load_vlm_weights=true
      # Constant LR — previous default (warmup 1000 → decay to 2.5e-6 by
      # step 30000) was zeroing the LR before we had any useful training.
      --policy.optimizer_lr="$TRAIN_LR"
      --policy.scheduler_warmup_steps=0
      --policy.scheduler_decay_steps=1000000
      --policy.scheduler_decay_lr="$TRAIN_LR"
      # n_action_steps at train time = pretrain chunk default (50). Eval
      # call below uses n_action_steps=10 per lerobot maintainer guidance
      # (see reports/runs/vision_grounded_wide_15hz_2026-04-24/run_diary.md).
      --policy.n_action_steps="$N_ACTION_STEPS"
    )
    if [[ "$USE_LORA" == "1" ]]; then
      train_args+=(
        --peft.method_type=LORA
        --peft.r="$LORA_R"
        --peft.lora_alpha="$LORA_ALPHA"
        --peft.lora_dropout="$LORA_DROPOUT"
        --peft.target_modules="$LORA_TARGETS_REGEX"
        # modules_to_save (via lerobot's full_training_modules alias):
        # promote only the final action-head output projection (action_out_proj)
        # from LoRA-delta to fully-trained — it's the one module directly
        # emitting the 7D action. action_time_mlp_out is LoRA'd per the
        # broader community consensus that over-promoting modules_to_save
        # shrinks the LoRA regularization benefit.
        --peft.full_training_modules='[action_out_proj]'
        # v4 (2026-04-26): per-dim action loss weight on action[6] (gripper).
        # v3/v3.1/v3.2 ran "canonical uniform MSE" (recovery plan §3 Hole B
        # ablation-add-back) and produced policies that bulldoze the cube
        # without ever closing the gripper. The gripper is +1.0 for ~30% of
        # demo frames then -1.0, with the transition localized to ~1 step;
        # under flat L2 across 7 dims, regressing the gripper to ~+0.9
        # throughout is near-optimal-lazy. Weight 8.0 is the recovery plan's
        # "strong but not dominant" recommendation. The lerobot patch wiring
        # this through is documented in
        # project_lerobot_peft_resume_patch.md; field name confirmed at
        # lerobot/policies/smolvla/configuration_smolvla.py:113.
        # Length must match max_action_dim (32 in SmolVLA flow matching),
        # NOT the env's action_dim (7). First 6 entries cover pose deltas,
        # entry 7 is gripper (×16 — bumped from 8 mid-run on 2026-04-26
        # when v4-w8 hit 0/N real successes through epoch 20 with loss
        # plateaued at 0.031 and dz=0 across all eval episodes; gripper
        # was clearly not being learned), entries 8-32 are padding dims
        # masked out via action_padding_mask (weight value doesn't matter).
        --policy.action_loss_dim_weights='[1.0,1.0,1.0,1.0,1.0,1.0,16.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]'
      )
      _log "fresh start from lerobot/smolvla_base + LoRA r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT, constant LR=$TRAIN_LR"
      _log "  vision tower LoRA + lm_expert + action projections; modules_to_save=[action_out_proj]"
      _log "  action_loss_dim_weights=[1,1,1,1,1,1,16] (gripper × 16) — v4-w16 stronger Hole B push"
    else
      _log "fresh start from lerobot/smolvla_base (full fine-tune), constant LR=$TRAIN_LR"
    fi
  fi

  # Stream lerobot's stdout through parse_train_log.py so every logged step
  # becomes a JSONL record. Full stream also appended to $TRAIN_LOG.
  set +e
  PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$VENV_PY" \
    "$LEROBOT_SRC"/lerobot/scripts/lerobot_train.py "${train_args[@]}" 2>&1 \
    | tee -a "$TRAIN_LOG" \
    | "$VENV_PY" "$REPO"/scripts/orchestrate/parse_train_log.py \
        --out "$STEP_JSONL" --epoch "$next_epoch" \
        > /dev/null
  # Capture every stage's exit code. We must check parse_train_log too:
  # if it crashes mid-epoch the STEP_JSONL is left half-written and every
  # downstream consumer (plot_metrics, budget_watchdog, eval-summary jq)
  # silently sees corrupted data while the epoch counter advances.
  rcs=("${PIPESTATUS[@]}")
  rc_train=${rcs[0]}
  rc_parse=${rcs[2]}
  set -e
  if [[ $rc_train -ne 0 || $rc_parse -ne 0 ]]; then
    _log "epoch $next_epoch failed (train rc=$rc_train, parse rc=$rc_parse); sleeping 30s and retrying (epoch NOT advanced)"
    sleep 30
    continue
  fi

  # Successful call — advance the canonical counter.
  epoch=$next_epoch
  last_ckpt="$CKPT_DIR/checkpoints/last/pretrained_model"

  # Eval cadence.
  sr=""
  if (( epoch % EVAL_EVERY_N == 0 )); then
    _log "eval at epoch $epoch ($EVAL_EPISODES episodes)"
    eval_gif="$CKPT_DIR/eval_epoch_$(printf '%04d' "$epoch").gif"
    eval_log="$LOG_DIR/eval_epoch_$(printf '%04d' "$epoch").out"
    set +e
    # max_steps 900 = 30 s at 30 Hz policy rate (match the env's
    # episode_length_s=30). n_action_steps=10 at eval per lerobot
    # maintainer recommendation (train uses chunk_size=50).
    PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$ISAAC_LAB"/isaaclab.sh \
      -p "$REPO"/scripts/train/run_vla_closed_loop.py \
      --checkpoint "$last_ckpt" \
      --num_episodes "$EVAL_EPISODES" --max_steps 900 \
      --drop_cube_pos --gripper_threshold 0.0 \
      --save_gif "$eval_gif" \
      --jsonl_out "$LOG_DIR/eval_episodes.jsonl" \
      --ckpt_tag "epoch_$(printf '%04d' "$epoch")" \
      2>&1 | tee "$eval_log" | tail -5
    erc=$?
    set -e
    if [[ $erc -ne 0 ]]; then
      _log "eval failed (rc=$erc); continuing"
    fi
    # Pull SR from the final "total: X/Y" line of the eval log.
    sr=$(grep -oP 'total: \K[0-9]+/[0-9]+' "$eval_log" | tail -1 \
         | awk -F/ '{ if ($2 > 0) printf "%.3f", $1/$2; else print "0" }')
  fi

  # Compute summary stats for this epoch from the step JSONL (faster +
  # accurate vs. grepping the text log).
  $VENV_PY - "$EPOCH_JSONL" "$STEP_JSONL" "$epoch" "$LEROBOT_LIVE" "$sr" <<'PY'
import json, sys, datetime, pathlib
out_p, steps_p, epoch_s, ds_p, sr_s = sys.argv[1:]
epoch = int(epoch_s)
# Pull step records for this epoch.
losses, grads, lrs, updates = [], [], [], []
step_end = -1
if pathlib.Path(steps_p).exists():
    for line in pathlib.Path(steps_p).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("epoch") != epoch:
            continue
        losses.append(r["loss"])
        grads.append(r["grad_norm"])
        lrs.append(r["lr"])
        updates.append(r["update_s"])
        step_end = max(step_end, r.get("step", -1))

def pct(xs, q):
    if not xs: return None
    s = sorted(xs)
    i = int(round((len(s) - 1) * q))
    return s[i]

def mean(xs):
    return sum(xs) / len(xs) if xs else None

# Count demos directly from the (symlinked) dataset's info.json, which
# LeRobot writes at build time — faster than opening h5py.
num_demos = None
info = pathlib.Path(ds_p) / "meta" / "info.json"
if info.exists():
    try:
        num_demos = json.loads(info.read_text()).get("total_episodes")
    except Exception:
        pass

line = {
  "ts": datetime.datetime.utcnow().isoformat() + "Z",
  "epoch": epoch,
  "num_demos": num_demos,
  "global_step_end": step_end,
  "loss_mean": mean(losses),
  "loss_p50": pct(losses, 0.5),
  "loss_p95": pct(losses, 0.95),
  "grad_norm_p95": pct(grads, 0.95),
  "lr_end": lrs[-1] if lrs else None,
  "update_s_mean": mean(updates),
  "num_step_samples": len(losses),
  "eval_sr": float(sr_s) if sr_s not in ("", "nan") else None,
}
with open(out_p, "a") as f:
    f.write(json.dumps(line) + "\n")
print(json.dumps(line))
PY

  _state_write epoch "$epoch"
  _state_write last_ckpt "$last_ckpt"
  _log "epoch $epoch done (step_end=$((epoch * TRAIN_STEPS)))"
done

_log "training loop exited"
