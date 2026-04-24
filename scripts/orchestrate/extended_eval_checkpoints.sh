#!/usr/bin/env bash
# Run a single eval of the supplied checkpoint against a fixed set of cube
# positions (so different checkpoints can be compared apples-to-apples).
#
# Usage: extended_eval_checkpoints.sh <step_tag> <cube_xy_string>
#   step_tag     e.g. 010000 (matches checkpoints/continual/checkpoints/<step_tag>)
#   cube_xy_str  'x1,y1;x2,y2;...' — N positions; runs one episode per position.
set -euo pipefail

STEP_TAG="$1"
CUBE_XY="$2"

REPO=/home/ubuntu/vla_kitting
ISAAC_LAB=/home/ubuntu/IsaacLab
LEROBOT_SRC=/home/ubuntu/code/lerobot/src

CKPT="$REPO/checkpoints/continual/checkpoints/$STEP_TAG/pretrained_model"
if [[ ! -d "$CKPT" ]]; then
  echo "ERR: checkpoint $CKPT not found" >&2
  exit 1
fi

# Count episodes = number of positions.
N_EP=$(awk -F';' '{print NF}' <<< "$CUBE_XY")

OUT_LOG="$REPO/logs/continual/extended_eval_${STEP_TAG}.out"
JSONL="$REPO/logs/continual/extended_eval_episodes.jsonl"

echo "[ext-eval] step_tag=$STEP_TAG episodes=$N_EP" >&2
PYTHONPATH="$LEROBOT_SRC:${PYTHONPATH:-}" "$ISAAC_LAB"/isaaclab.sh \
  -p "$REPO"/scripts/train/run_vla_closed_loop.py \
  --checkpoint "$CKPT" \
  --num_episodes "$N_EP" --max_steps 1800 \
  --cube_xy "$CUBE_XY" \
  --save_gif "$REPO/checkpoints/continual/extended_eval_${STEP_TAG}.gif" \
  --jsonl_out "$JSONL" \
  --ckpt_tag "ext_${STEP_TAG}" \
  2>&1 | tee "$OUT_LOG"

SR_LINE=$(grep -oP 'total: \K[0-9]+/[0-9]+' "$OUT_LOG" | tail -1)
echo "[ext-eval] $STEP_TAG => $SR_LINE"
