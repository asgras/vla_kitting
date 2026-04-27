#!/bin/bash
# Periodically prune old per-step checkpoints to keep disk free during a long
# training run. Keeps:
#   - the symlinked "last" target
#   - every Nth milestone (default N=5, i.e. every 5 epochs)
#   - the last 3 checkpoints regardless
#
# Invoked as a background daemon by the user during long runs. Exits when
# logs/continual/STOP exists.
#
# Usage:
#   nohup bash scripts/orchestrate/prune_old_checkpoints.sh > logs/continual/prune_daemon.out 2>&1 &

set -euo pipefail
REPO=/home/ubuntu/vla_kitting
CKPT_DIR=$REPO/checkpoints/continual/checkpoints
LOG_DIR=$REPO/logs/continual
KEEP_EVERY=${KEEP_EVERY:-5}
KEEP_LAST=${KEEP_LAST:-3}
SLEEP_S=${SLEEP_S:-300}

mkdir -p "$LOG_DIR"
echo "[prune] starting (KEEP_EVERY=$KEEP_EVERY KEEP_LAST=$KEEP_LAST sleep=${SLEEP_S}s)"

while true; do
  if [[ -f "$LOG_DIR/STOP" ]]; then
    echo "[prune] STOP detected — exiting"
    break
  fi
  if [[ ! -d "$CKPT_DIR" ]]; then
    sleep "$SLEEP_S"
    continue
  fi
  # Resolve the "last" symlink to know which one is the live checkpoint.
  last_target=""
  if [[ -L "$CKPT_DIR/last" ]]; then
    last_target=$(readlink "$CKPT_DIR/last")
  fi

  # Collect numeric checkpoint dirs, sorted ascending.
  mapfile -t ckpts < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+$' -printf '%f\n' | sort -n)
  total=${#ckpts[@]}
  if (( total <= KEEP_LAST )); then
    sleep "$SLEEP_S"; continue
  fi

  # Compute step indices to KEEP.
  declare -A keep
  # Last KEEP_LAST.
  for ((i=total-KEEP_LAST; i<total; i++)); do
    keep[${ckpts[i]}]=1
  done
  # Every KEEP_EVERY epoch milestone (assumes step = epoch * 1000).
  for c in "${ckpts[@]}"; do
    epoch=$(( 10#$c / 1000 ))
    if (( epoch % KEEP_EVERY == 0 )); then
      keep[$c]=1
    fi
  done
  # The "last" target.
  [[ -n "$last_target" ]] && keep[$last_target]=1

  removed=0
  for c in "${ckpts[@]}"; do
    if [[ -z "${keep[$c]:-}" ]]; then
      rm -rf "$CKPT_DIR/$c" && removed=$((removed+1)) || true
    fi
  done
  if (( removed > 0 )); then
    echo "[prune] $(date +%H:%M:%S) pruned $removed checkpoints; kept ${#keep[@]} of $total"
  fi
  sleep "$SLEEP_S"
done
