#!/bin/bash
# Parallel scripted-demo generation orchestrator (vla_kitting-8rf).
#
# Why this exists
# ---------------
# scripts/validate/scripted_pick_demo.py runs num_envs=1 single-process. After
# vla_kitting-2hp lengthened phases A and C, a single demo takes ~50-70 s of
# wall-clock. To reach the 750-demo v5 target serially that's ~12-15 h.
#
# The L40S (46 GB VRAM) easily holds 4-5 concurrent Isaac Sim apps (~7 GB
# each). Fan-out: K shards × (total / K) demos each, distinct seeds → distinct
# cube positions / colors / yaws → no on-disk collisions and a representative
# union dataset. Wall-clock target: ~3-4 h for 4×188.
#
# Usage
# -----
#   bash scripts/orchestrate/parallel_scripted_demo_gen.sh \
#       --total 752 --shards 4 --run_id v5_2026_04_27
#
# Outputs
#   datasets/teleop/parallel_<run_id>/shard_<i>.hdf5     (one per shard)
#   logs/parallel_demo_gen_<run_id>/shard_<i>.log        (full Isaac Sim stdout)
#   logs/parallel_demo_gen_<run_id>/orchestrator.log     (this script's diary)
#
# After this finishes successfully, run:
#   /opt/IsaacSim/python.sh scripts/orchestrate/merge_scripted_shards.py \
#       --shard_dir datasets/teleop/parallel_<run_id> \
#       --output    datasets/teleop/parallel_<run_id>/merged.hdf5
#
# Then point isaaclab_to_lerobot.py at the merged HDF5 with the usual flags
# (--drop_cube_pos --stride 1) for the LeRobot conversion.
#
# Robustness notes
#   - Each shard is fully independent — if shard 2 crashes, just relaunch
#     scripted_pick_demo.py manually with the same --seed and --dataset_file
#     (delete the partial shard hdf5 first; --overwrite handles existing).
#   - We stagger shard launches by SHARD_STAGGER_S to avoid clobbering on
#     Isaac Sim's shared kit cache during first-time shader compilation.
#   - Exit non-zero if ANY shard reports failure, so the caller's CI / land-
#     the-plan check fires.

set -uo pipefail
export LC_ALL=C.UTF-8

REPO=/home/ubuntu/vla_kitting
ISAAC_LAB=/home/ubuntu/IsaacLab

# --- Defaults ---
TOTAL=752
SHARDS=4
RUN_ID="$(date +%Y%m%d_%H%M%S)"
BASE_SEED=42
SHARD_STAGGER_S=30          # seconds between shard launches; mitigates kit-cache lock contention.
SEED_STRIDE=10000           # large gap so reset-RNG sequences across shards never overlap.
MAX_STEPS_PER_DEMO=1300     # match scripted_pick_demo.py default after 2hp.

# --- Arg parse ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --total)         TOTAL="$2"; shift 2 ;;
    --shards)        SHARDS="$2"; shift 2 ;;
    --run_id)        RUN_ID="$2"; shift 2 ;;
    --base_seed)     BASE_SEED="$2"; shift 2 ;;
    --stagger_s)     SHARD_STAGGER_S="$2"; shift 2 ;;
    --max_steps)     MAX_STEPS_PER_DEMO="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--total N] [--shards K] [--run_id slug] [--base_seed S] [--stagger_s sec] [--max_steps N]

  --total       Total demo target across all shards (default 752).
  --shards      Number of concurrent Isaac Sim processes (default 4).
  --run_id      Slug for output directory (default: timestamp).
  --base_seed   Seed of shard 0; shard i gets BASE_SEED + i*SEED_STRIDE.
  --stagger_s   Sleep between shard launches; lets shard 0 finish kit-cache
                warm-up before shard 1 starts (default 30s).
  --max_steps   Per-demo step cap passed through to scripted_pick_demo.

EOF
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if (( SHARDS < 1 )); then
  echo "ERROR: --shards must be >= 1" >&2
  exit 2
fi
if (( TOTAL < SHARDS )); then
  echo "ERROR: --total ($TOTAL) must be >= --shards ($SHARDS)" >&2
  exit 2
fi

# --- Per-shard demo budget split. Front-load remainder onto first shards. ---
PER_SHARD_BASE=$(( TOTAL / SHARDS ))
REMAINDER=$(( TOTAL % SHARDS ))

# --- Output paths ---
SHARD_DIR=$REPO/datasets/teleop/parallel_$RUN_ID
LOG_DIR=$REPO/logs/parallel_demo_gen_$RUN_ID
mkdir -p "$SHARD_DIR" "$LOG_DIR"

ORCH_LOG=$LOG_DIR/orchestrator.log

_log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH_LOG"; }

_log "=================================================================="
_log "parallel_scripted_demo_gen.sh — vla_kitting-8rf"
_log "  TOTAL=$TOTAL  SHARDS=$SHARDS  RUN_ID=$RUN_ID"
_log "  BASE_SEED=$BASE_SEED  SEED_STRIDE=$SEED_STRIDE  STAGGER=${SHARD_STAGGER_S}s"
_log "  PER_SHARD_BASE=$PER_SHARD_BASE  REMAINDER=$REMAINDER"
_log "  shard dir   → $SHARD_DIR"
_log "  log dir     → $LOG_DIR"
_log "=================================================================="

# Track child PIDs and the shard index they correspond to.
declare -a PIDS
declare -a SHARD_OF_PID
declare -a DEMOS_OF_SHARD
declare -a SEED_OF_SHARD
declare -a HDF5_OF_SHARD
declare -a LOG_OF_SHARD

START_TS=$(date +%s)

for (( i=0; i<SHARDS; i++ )); do
  # Front-load remainder demos: first REMAINDER shards each get +1.
  shard_demos=$PER_SHARD_BASE
  if (( i < REMAINDER )); then
    shard_demos=$(( shard_demos + 1 ))
  fi
  shard_seed=$(( BASE_SEED + i * SEED_STRIDE ))
  shard_hdf5=$SHARD_DIR/shard_${i}.hdf5
  shard_log=$LOG_DIR/shard_${i}.log

  DEMOS_OF_SHARD[$i]=$shard_demos
  SEED_OF_SHARD[$i]=$shard_seed
  HDF5_OF_SHARD[$i]=$shard_hdf5
  LOG_OF_SHARD[$i]=$shard_log

  _log "launching shard $i: demos=$shard_demos seed=$shard_seed → $shard_hdf5"

  # cd into IsaacLab so `./isaaclab.sh` finds its bundled python entrypoint.
  # Each shard gets --overwrite so a relaunch (after fixing a crash) is
  # idempotent without needing the operator to rm the partial file.
  (
    cd "$ISAAC_LAB"
    ./isaaclab.sh -p "$REPO"/scripts/validate/scripted_pick_demo.py \
      --num_demos "$shard_demos" \
      --max_steps_per_demo "$MAX_STEPS_PER_DEMO" \
      --dataset_file "$shard_hdf5" \
      --seed "$shard_seed" \
      --overwrite \
      > "$shard_log" 2>&1
  ) &
  pid=$!
  PIDS+=("$pid")
  SHARD_OF_PID[$pid]=$i
  _log "  shard $i pid=$pid"

  # Stagger to avoid hitting the kit shader cache simultaneously on first warm-
  # up. After shard 0 has compiled, subsequent shards reuse the cached blobs
  # and start ~10x faster. Skipped after the last shard.
  if (( i < SHARDS - 1 )) && (( SHARD_STAGGER_S > 0 )); then
    _log "  staggering ${SHARD_STAGGER_S}s before next shard launch"
    sleep "$SHARD_STAGGER_S"
  fi
done

_log "all $SHARDS shards launched; waiting for completion..."

# Wait on each PID individually so we can report per-shard exit codes.
declare -a EXIT_CODES
overall_ok=1
for pid in "${PIDS[@]}"; do
  set +e
  wait "$pid"
  rc=$?
  set -e
  i=${SHARD_OF_PID[$pid]}
  EXIT_CODES[$i]=$rc
  if (( rc == 0 )); then
    _log "shard $i (pid=$pid) FINISHED OK (rc=$rc)"
  else
    _log "shard $i (pid=$pid) FAILED (rc=$rc) — see ${LOG_OF_SHARD[$i]}"
    overall_ok=0
  fi
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

_log "=================================================================="
_log "parallel run wall-clock: ${ELAPSED}s ($(printf '%dh:%02dm:%02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))))"
_log "per-shard summary:"
for (( i=0; i<SHARDS; i++ )); do
  rc=${EXIT_CODES[$i]:-?}
  _log "  shard $i  rc=$rc  demos_target=${DEMOS_OF_SHARD[$i]}  seed=${SEED_OF_SHARD[$i]}  hdf5=${HDF5_OF_SHARD[$i]}"
done
_log "=================================================================="

if (( overall_ok == 1 )); then
  _log "OK — all shards succeeded. Next: scripts/orchestrate/merge_scripted_shards.py"
  _log "  --shard_dir $SHARD_DIR --output $SHARD_DIR/merged.hdf5"
  exit 0
else
  _log "FAIL — at least one shard failed. Inspect logs in $LOG_DIR before merging."
  exit 1
fi
