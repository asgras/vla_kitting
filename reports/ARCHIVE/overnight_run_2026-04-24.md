# Overnight run 2026-04-24 — 60 Hz broader-LoRA retrain

Launched 02:41 UTC per plan in `reports/next_steps_2026-04-24.md`. Stopped
cleanly at 07:18 UTC after epoch 41 because SR was monotonically decaying
(overfit pattern; see "Eval trajectory" below). Total training time: 4h 37m.

## What ran

- Dataset: `datasets/lerobot/cube_pick_v1 → cube_pick_v1_20260423_021729`
  128 episodes, 205,440 frames, 60 fps
- Policy: SmolVLA base + LoRA r=32 α=32 dropout=0.05
- LoRA targets: `lm_expert` q/k/v/o + gate/up/down, plus state_proj, action_in/out_proj, action_time_mlp_in/out
- LR: constant 1e-4
- Batch 4, save every 1k steps, eval every 10 epochs × 10 rollouts

## Eval trajectory (10-episode in-loop evals — noisy, fresh cube positions each eval)

| Epoch | SR (/10) | loss_mean | Notes |
|------:|---------:|----------:|-------|
| 10 | **2/10** | 0.1765 | Peak eval SR for the run |
| 20 | 1/10 | 0.1735 | |
| 30 | 0/10 | 0.1710 | |
| 40 | 0/10 | 0.1699 | Loss still ticking down; SR dead |

**SR decay while loss keeps descending = classic overfit.** I wrote STOP at
07:13 so the run exited after epoch 41 to free GPU for proper eval.

## Extended eval (fixed 20 cube positions, seed=42)

Same 20 XY positions across every checkpoint, so SR is comparable.
Cube range: x ∈ [0.45, 0.65], y ∈ [-0.13, 0.13], z=0.025.

| Epoch (step tag) | SR (/20) | Notes |
|------:|---------:|-------|
| 008000 | 1/20 (5%) | |
| 009000 | 0/20 (0%) | |
| 010000 | **2/20 (10%)** | ep7 (0.623, 0.004) + ep8 (0.570, 0.024) — both ~front-center |
| 011000 | 0/20 (0%) | |
| 012000 | — | eval killed on user request |
| 015000 | 0/20 (0%) | |
| 020000 | 1/20 (5%) | Partial recovery from epoch 15 trough |

**Peak is narrow:** epochs 9 and 11 flank epoch 10 with 0/20 each. The two
successful positions at epoch 10 are close to the front-center of the
workspace (small |y|, moderate x). All other positions fail at every
checkpoint tested.

_(Table updates as evals complete. Each fixed-seed eval takes ~37 min.)_

## Diagnosis

Second retrain in a row where the only non-zero SR is at or near epoch 10, and
where more training strictly degrades SR while loss keeps improving. Pattern
across the two runs:

| Run | r | α | LR | Peak SR (epoch) |
|-----|--:|--:|---:|:----------------|
| 2026-04-22 (25 demos) | 16 | 16 | 5e-5 | 0/10 all epochs |
| 2026-04-23 (128 demos) | 16 | 16 | 5e-5 | **1/10** at epoch 10 |
| 2026-04-24 (this run) | 32 | 32 | 1e-4 | **2/10** at epoch 10 |

Each axis (more data, bigger LoRA + higher LR) gains roughly one successful
episode and nothing else. Eval scales with extended-seed eval to ~10% on
20-position set → the model has memorized a narrow slice of positions and
can't generalize. Loss floor ≈0.170 is only ~4% below the prior run's 0.178
floor — limited additional headroom.

## Best checkpoint to use

**`checkpoints/continual/checkpoints/010000/pretrained_model`** — SR ~10% on
fixed 20-position set. (If later extended evals beat this I'll update.)

Note: this checkpoint already scored 2/10 in the in-loop eval and 2/20 in the
extended eval. Consistent ~10% SR.

## Recommended next experiments (ordered by expected impact)

1. **Widen data distribution** (plan Step 2) — 4–6h compute.
   The two-run pattern strongly suggests data narrowness is the primary
   bottleneck, not LoRA capacity or LR. Widen cube randomization
   (`envs/yaskawa_pick_cube_cfg.py:randomize_cube_pose` to x ∈ [0.40, 0.70],
   y ∈ [-0.20, +0.20]), regenerate 300+ Mimic demos, re-convert, retrain.

2. **Unfreeze lm_expert FFN bias terms** (plan Step 3.1) — small code change,
   tiny param count, highest-locality fine-tune. Cheap to try before the
   expensive data-regen path.

3. **Remove `observation.cube_pos` from the dataset features** (plan Step 3.2)
   — force visual grounding. Prior ablations showed cube_pos may not be
   load-bearing but this is worth confirming by training without it.

4. **Constrain action** to 3D pos + gripper (drop `joint_6_t` from action) —
   would eliminate the orientation drift mode we observed during descent.

## Artifacts

- `checkpoints/continual/checkpoints/010000/` — 2/20 SR checkpoint (best)
- `checkpoints/continual/extended_eval_010000.gif` — last rollout from 20-pos eval
- `logs/continual/extended_eval_010000.log` — full per-episode results
- `logs/continual/extended_eval_episodes.jsonl` — structured per-episode records
- `logs/continual/epoch_summary.jsonl` — all in-loop epoch summaries (41 epochs)
- `logs/continual/fixed_cube_positions.txt` — the 20 XY positions used (seed=42)
- `logs/continual/orchestrator_60hz_broader_lora_launch.out` — orchestrator log
- `reports/next_steps_2026-04-24.md` — prior plan this run followed

## State after the run

- STOP file removed (ready for next launch).
- No running training processes; GPU free.
- Disk: ~13 GB free at 90% used (stable throughout run; cleaned pip + uv cache
  before launch, freed 15 GB).
- Isaac Sim kit version-check zombie process killed pre-launch.
- No changes to repo source files. Checkpoints, logs, reports additive only.
