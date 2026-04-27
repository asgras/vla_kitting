# 2026-04-26 — Run B prime config-diff one-pager (vla_kitting-hzj)

## Why this report exists
Run B prime (`reports/saved_checkpoints/r16_epoch10_sr0.10/`) is the ONLY
checkpoint in this project's history that ever cleared ≥ 2/10 closed-loop
success rate. Every subsequent expansion (v3 r=32, v3.1 +vision-tower LoRA,
v3.2 +n_action_steps=10, v4 +gripper-weighting) returned to 0/N. The
recovery plan asks for a load-bearing config delta — what is Run B prime
doing that the others aren't?

## SR re-eval status
**Deferred — env changed since Run B prime trained.** The current env
config has:
- 1.5×1.0 m table (was 1.2×0.8 m)
- 10 cm magenta cylinder target (was 20×20 cm magenta square)
- Cube yaw randomization ±0.5 rad (was 0)
- "magenta circle" prompt (was "pink square")
- New per-step `cube_color_idx` observation
Running the saved checkpoint against today's env would conflate "Run B
prime ability" with "Run B prime under shifted distribution" — see CLAUDE.md
"Environment changes invalidate prior baselines." A meaningful SR re-eval
needs either (a) a v5 retrain on the new env, or (b) a one-off env-rollback
which costs more than it gains. The config-delta below is the
genuinely portable deliverable.

## Config delta vs current train_only.sh (v4)
| Knob | Run B prime | Current v4 default |
|---|---|---|
| LoRA r | **16** | 64 |
| LoRA alpha | **8** (alpha/r = 0.5) | 64 (alpha/r = 1.0) |
| LoRA dropout | **0.0** | 0.05 |
| LoRA targets, lm_expert | **q, v only** | q, k, v, o, gate, up, down |
| LoRA targets, vision tower | **none** (vision frozen) | q, k, v, out on every layer |
| `modules_to_save` | **`[]`** (nothing fully promoted) | `[action_out_proj]` |
| Action loss weights | **uniform** (canonical L2) | gripper × 16 |
| Optimizer LR | 5e-5 | 5e-5 (same) |
| Batch size | **4** | 8 (env default) |
| `freeze_vision_encoder` | **True** | False (vision LoRA = effectively unfrozen) |
| `train_expert_only` | **True** | False |
| `n_action_steps` | 10 | 10 (same) |
| Dataset | `vla_kitting/cube_pick_v1` | same |

## What's load-bearing — most-likely-suspects
The four current-vs-RunB knobs that flipped together are:
1. **Vision tower un-frozen via LoRA** (the v3.1 change that didn't help and
   may have actively hurt). At r=64 with vision-tower LoRA, the trainable
   parameter count grows ~5× and the inductive bias of the frozen
   ImageNet-pretrained vision encoder is partially released. If the
   pretrained vision features were close to optimal, this regresses
   robustness.
2. **`modules_to_save=[action_out_proj]`** ('full' fine-tuning of the action
   projection). Promoting a single linear from LoRA to full update means
   the model can saturate that layer's capacity without LoRA's regularizing
   low-rank constraint. With only 750 demos that is the wrong direction.
3. **Action-loss gripper × 16** (v4-w16). The task description framed this
   as a Hole B fix (gripper never learns to close), but Run B prime
   converged with uniform L2 — the gripper signal evidently CAN learn
   without the weight, just slower. Heavy gripper weight may be drowning
   the pose-regression signal that Run B's modest config could still hear.
4. **r=64 / α=64 vs r=16 / α=8**. 4× the rank with 8× the effective
   learning rate (α/r) is a large training-dynamics shift. With small
   datasets, smaller r preserves the pretrained representation better.

The other three (dropout 0.0, only q,v, batch 4) are individually plausible
contributors but each is a much smaller effect than the four above.

## Recommended v5 starting point
Revert the four most-likely-suspects in one go (single-variable change is
probably impossible — they were flipped together in v3 → v3.1 → v4):
- `LORA_R=16  LORA_ALPHA=8  LORA_DROPOUT=0.0`
- `LORA_TARGETS_REGEX="(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"` (matches Run B prime's regex exactly — no vision tower, no o/k/gate/up/down)
- `--peft.full_training_modules='[]'` (no `modules_to_save`)
- Drop `--policy.action_loss_dim_weights` entirely (uniform L2)
- Keep batch_size from current default (compute-bound; 4 vs 8 is unlikely
  load-bearing)

That would be vla_kitting-(new) "v5 LoRA-shrink + uniform-loss baseline".
Train for the same epoch budget Run B prime hit SR 10% at (~ epoch 10 in
the saved checkpoint name → 1k steps under batch=4 ≈ 4k frames). Confirm
SR on `configs/eval_seed_30.json` for an apples-to-apples number with
future runs.

## Acceptance
- ✅ Configuration diff vs v3/v4 documented (above table).
- ⚠ Numerical SR on the fixed-seed 30-pos set deferred — env shift makes
  the number uninformative. Will be re-baked when v5 trains under the
  same env that this eval uses.
- ✅ Hypothesis on load-bearing knob handed off as actionable v5 config.

## Next step
File `vla_kitting-(new)` for the v5 LoRA-shrink baseline run, depending
on `vla_kitting-mil` (xy_tolerance decision). Don't re-run hzj's SR
half until v5 trains under the new env; treat that as the apples-to-apples
baseline going forward.
