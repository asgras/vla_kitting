# Overnight run 2026-04-23 — 128-demo SmolVLA + LoRA r=16

Launched 03:31 UTC, 8h budget (BUDGET_HOURS=8), auto-stop via budget_watchdog.
Fresh start on the newly-built 128-demo LeRobot dataset.

## What ran

- Dataset: `datasets/lerobot/cube_pick_v1 → cube_pick_v1_20260423_021729`
  128 episodes, 205,440 frames, 60 fps, 11 GB (PNG frames)
- Policy: SmolVLA base + LoRA r=16
- LR: constant 5e-5 (warmup=0, decay=1e6 steps so effectively flat)
- Batch size: 4, save_freq=1000, eval every 10 epochs × 10 rollouts

## Eval trajectory

| Epoch | SR (/10) | loss_mean | Notes |
|------:|---------:|----------:|-------|
| 10 | **1/10** | 0.190 | Best result. Prior 25-demo run never hit this. |
| 20 | 0/10 | 0.183 | |
| 30 | 0/10 | 0.179 | |
| 40 | 0/10 | 0.179 | |
| 50 | 0/10 | 0.178 | |
| 60 | 0/10 | 0.176 | Loss tick down but no SR progress |
| 70 | 0/10 | 0.177 | Loss oscillating; no SR progress |
| 80 | 0/10 | 0.175 | Same plateau pattern. Budget exhaust ~11:31 UTC. |

**Only epoch 10 produced non-zero SR** across the full 8h run. The epoch 10
checkpoint is the artifact to use.

## Diagnosis

Loss floor sits at ~0.178, essentially identical to the prior 25-demo r=16 run's
0.18 floor despite 4× more data. This suggests the bottleneck is **not data
quantity** — it's either LoRA adapter capacity at r=16 or the LR being too
low for LoRA to fit the action distribution properly (5e-5 is conservative
for LoRA fine-tuning; typical recipes use 1e-4 to 5e-4).

The epoch-10 1/10 SR looks like either:
  (a) A genuine early-peak before the adapter overfits / plateaus, or
  (b) Single-trial variance on a 10-episode eval.

Subsequent flat 0/10 across evals 20-50 is consistent with either hypothesis.

## Backups

Preserved outside the `--reset` path:
- `reports/saved_checkpoints/r16_epoch10_sr0.10/` — best-SR adapter so far
- `reports/archive_25demo_run/` — prior run's epoch/step metrics for comparison

## Suggested next experiments (tomorrow)

Ordered by expected impact:

1. **Bump LR to 1e-4 or 2e-4** (single-variable change).
   Likely the cheapest win. LoRA often wants 2-10× higher LR than full FT.
   ```
   TRAIN_LR=2e-4 BUDGET_HOURS=4 bash scripts/orchestrate/train_only.sh --reset
   ```

2. **Double LoRA rank to r=32** (more adapter capacity).
   ```
   LORA_R=32 BUDGET_HOURS=4 bash scripts/orchestrate/train_only.sh --reset
   ```

3. **Combine** (2) and (3): `LORA_R=32 TRAIN_LR=1e-4`.

4. **Full fine-tune** (disable LoRA entirely). Most expensive — check VRAM
   headroom first (we have 46 GB L40S, SmolVLA at fp16 with batch 4 should
   fit but close).
   ```
   USE_LORA=0 TRAIN_LR=5e-5 BUDGET_HOURS=6 bash scripts/orchestrate/train_only.sh --reset
   ```

5. **Larger batch** if memory allows (TRAIN_BATCH=8 or 16). Lower per-step
   noise could help the plateau.

6. Look at the eval gifs (`checkpoints/continual/eval_epoch_*.gif`) to see
   *how* the policy is failing — does it approach the cube and miss? Freeze?
   Wander? That diagnosis should guide whether the issue is data
   distribution, action scale, or something else.

## Changes landed this session (keep)

- `scripts/orchestrate/mimic_generate.sh`, `continual_train.sh`: merge-output
  JSON capture hardened (`| grep -E '^\{' | tail -1`) so stray stdout
  doesn't break the json.loads pipe.
- `scripts/orchestrate/train_only.sh`: budget_watchdog now runs with
  `--grace-minutes` = budget, so the plateau-detector can't short-circuit
  a long run that's structurally flat (our prior run would have been
  killed at epoch ~40).
