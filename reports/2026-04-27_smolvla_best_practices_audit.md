# 2026-04-27 — SmolVLA best-practices audit (bd vla_kitting-18n)

**Type:** pre-flight config review for the next training run. NOT itself a run.

**Sources of authority used (cited inline as `[P]` paper, `[MC]` model card, `[Doc-SV]` lerobot SmolVLA tutorial, `[Doc-PEFT]` lerobot PEFT tutorial, `[Code]` lerobot main):**

- [P] Shukor et al., *SmolVLA*, arXiv:2506.01844 (Jun 2025)
- [MC] https://huggingface.co/lerobot/smolvla_base
- [Doc-SV] https://huggingface.co/docs/lerobot/main/en/smolvla
- [Doc-PEFT] https://huggingface.co/docs/lerobot/main/en/peft_training
- [Code] `~/code/lerobot/src/lerobot/policies/smolvla/{configuration,modeling}_smolvla.py`

**Our reference invocation:** `scripts/orchestrate/train_only.sh` (Knobs-block + train_args block) on dataset `datasets/lerobot/cube_pick_v1` (370 episodes, 307,866 frames, features: `observation.{state, ee_pose, images.wrist, images.third_person}`, action 7D). Camera streams now wrist 256×256 + third 512×512 RGB after the 2026-04-27 reframing.

## Audit table

| Item | Authoritative recommendation | Our setting | Verdict | Notes |
|---|---|---|---|---|
| `load_vlm_weights` when FT'ing from `smolvla_base` | `True` (required) [Code, Doc-SV, MC] | `--policy.load_vlm_weights=true` | ✅ ALIGNED | Memory `project_smolvla_load_vlm_weights_gotcha` already covers this |
| Resize target | `(512, 512)` square pad-on-left+top [Code] | wrist 256→bilinear-up 512, third 512 native | ✅ ALIGNED | Verified in [vision-input verification report](2026-04-27_smolvla_vision_input_verification.md) |
| Visual normalization mode | `IDENTITY` [Code] | default; not overridden | ✅ ALIGNED | |
| Optimizer | AdamW, β=(0.9, 0.95), wd=1e-10, eps=1e-8 [P], [Code] | code defaults | ✅ ALIGNED | We don't override |
| Grad clip | `10` [Code] | code default | ✅ ALIGNED | |
| `chunk_size` | `50` (paper sweet spot 10–50) [Code, P Table 12] | `50` (default) | ✅ ALIGNED | |
| `n_action_steps` | `50` default; `10–50` sweet spot [P] | `10` | ✅ ALIGNED | Justified by lerobot maintainer guidance + our v3.2 finding (longer chunks → stereotyped exec, less visual feedback) |
| `n_obs_steps` | `1` [Code] | default | ✅ ALIGNED | |
| `tokenizer_max_length` | `48` [Code] | default | ✅ ALIGNED | Our prompt ≈14 tokens |
| Min demos | "≥50, paper used 50; 25 was not enough" [P, Doc-SV] | 370 episodes | ✅ ALIGNED | 7.4× recommended floor |
| Camera count | 2–3 views in paper FT [P] | 2 (wrist + third) | ✅ ALIGNED | Paper's 2-camera setup matches |
| Image-key namespace | `observation.images.*` (any subkey, FeatureType.VISUAL) [Code] | `observation.images.{wrist,third_person}` | ✅ ALIGNED | Auto-inferred via `image_features` |
| Task prompt style | Short, action-led, single sentence per episode [P §3.2, Doc-SV] | "pick up the {color} cube and place it on the magenta circle" | ✅ ALIGNED | |
| Control / data rate | Paper assumed Δt=33 ms (~30 fps) [P §4] | env publishes at 60 Hz; we down-sample to 30 Hz at training time via lerobot stride | ✅ ALIGNED | Memory: prior 15 Hz attempt was shelved per `project_active_experiment` |
| LoRA `r` | `64` example [Doc-PEFT] | `64` | ✅ ALIGNED | |
| LoRA `alpha`, `dropout` | not specified [—] | `64`, `0.05` | ✅ ACCEPTABLE | No authoritative guidance — α=r is the conventional "neutral" scaling |
| LoRA target_modules | default = `q_proj`, `v_proj` of `lm_expert` + projections [Code, Doc-PEFT] | `lm_expert.{q,k,v,o,gate,up,down}_proj` + vision-tower `self_attn.{q,k,v,out}_proj` + `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_{in,out}` | ⚠️ DIVERGENCE | We're more aggressive (full QKVO + MLP on lm_expert; vision LoRA on). Documented rationale in `train_only.sh` (mode collapse with frozen ViT in v3). See **CONCERN #2** below. |
| `modules_to_save` (`full_training_modules`) | `[]` default [Code] | `[action_out_proj]` | ⚠️ DIVERGENCE | One module promoted to full FT. Documented rationale: it's the dim-7 emitter of the action vector. Defensible. |
| `freeze_vision_encoder`, `train_expert_only` | `True`, `True` (paper canonical) [P, Code] | effectively False — vision-tower LoRA on | ⚠️ CONCERN | See **CONCERN #2** |
| Action representation | Paper used absolute joint position; no general delta-vs-absolute prescription [P] | IK-rel deltas (6D pose delta + binary gripper) | ⚠️ ACCEPTABLE | Paper-silent for our embodiment. Our choice is ROS-controller-native; documented in env cfg comments |
| LR (LoRA FT) | `1e-3` peak, `1e-4` decay (10× full-FT) [Doc-PEFT] | `1e-4` constant | ⚠️ CONCERN | See **CONCERN #1** |
| Warmup | `100` (paper pretrain) / `1000` (code default) [P, Code] | `0` | ⚠️ ACCEPTABLE | Defensible because we run constant LR, but only because of that. Any switch to cosine MUST restore warmup. |
| Total steps | `20k` tutorial starter, paper 100k–200k [Doc-SV, P] | open-ended (1k/epoch loop until STOP) | ⚠️ ACCEPTABLE | Continual-train loop semantics, not a single fixed budget |
| Batch size | `64` [Doc-SV, P] | `4` | 🔴 CONCERN | See **CONCERN #3** |
| Gradient accumulation | not mentioned [Doc-SV] | not configured | 🔴 CONCERN | Combined with batch=4 means effective batch is 16× smaller than recommended. See **CONCERN #3** |
| `action_loss_dim_weights` | not in paper / docs [—] | gripper × 16 (others × 1) | ⚠️ ACCEPTABLE | Our own derivation, documented in `train_only.sh` and reports/runs/v4_gripper_weight_2026-04-26 |

## CONCERNS — fix before next run

### CONCERN #3 (most likely root cause of slow convergence) — batch size

We run `--batch_size=4` with no gradient accumulation. The tutorial [Doc-SV] and the paper [P] both state batch size 64. Our effective batch is **16× smaller** than recommended. With LoRA on top, gradient noise per update is high; this plausibly explains the slow loss decrease and sensitivity to learning rate that we documented in v3 / v4.

**Recommendation:**

- Try `--batch_size=16` with gradient accumulation 4 (effective batch 64 matching the paper). Confirm fits in GPU memory at the new image resolutions (256² wrist + 512² third = 4× more pixels per sample than the prior config).
- If 16 + accum 4 OOMs on the new resolution, fall back to 8 + accum 8 or 4 + accum 16. Effective batch is what matters for the optimizer; per-step memory is what matters for the GPU.
- Document the effective-batch chosen as a config knob in `train_only.sh` so it's not lost.

### CONCERN #2 — Vision-tower LoRA contradicts paper canonical

[P] explicitly trains the action expert only with the VLM (vision encoder + LLM) frozen. Our v3.1 turned vision-tower LoRA back on after a mode-collapse symptom in v3. The follow-on diagnostic in [reports/2026-04-27_v4_vision_grounding](runs/v4_trajectory_overlay_2026-04-27/) shows v4's third-person vision **is not being used as a cube-localization signal** (median residual-argmax error 147 px on a 256² image; corr_X ≈ 0.15 on cube-X to ee-dx). So vision LoRA is on AND not paying for itself.

**Recommendation:**

- Revert vision-tower LoRA (drop the vision-tower clause from `LORA_TARGETS_REGEX`) for the next run.
- If mode collapse re-appears, prefer the alternatives the v5 plan already calls out (auxiliary cube-localization head, aug pipeline, wider distribution) over re-enabling vision LoRA. Cite [Doc-SV §"freeze the vision encoder; only the action expert is trained"] for the revert rationale in the run diary.

### CONCERN #1 — LoRA LR may be too low

[Doc-PEFT] says LoRA fine-tuning typically uses 10× the full-FT LR (peak 1e-3, decay 1e-4). We use a constant 1e-4. Per the docs this is the *full-FT* number, applied to a LoRA-FT scenario. With our small effective batch (CONCERN #3), this compounds — small batch + low LR = very slow learning.

**Recommendation:**

- Once the batch issue (CONCERN #3) is addressed, sweep LR ∈ {1e-4, 3e-4, 1e-3} as a single-variable change. Keep the constant-LR schedule for now (justified in `train_only.sh` comments).

## ALIGNED watchouts (no action, just track)

- `load_vlm_weights=true` is set in fresh-start branch; verified in `train_only.sh`. Resume branch trusts the saved config — confirm the resumed config still has it via `prepare_resume_config.py` invariants (memory: `project_lerobot_peft_resume_patch`).
- Visual normalization is `IDENTITY`. Our converter does NOT need to emit per-image mean/std stats. Verify next time we regenerate the dataset that `dataset_stats.json` has no rogue image normalization entries.
- We render square (1:1) on both cameras → the `resize_with_pad` left+top pad branch is never exercised. Keeping cameras square is now a load-bearing assumption (per [vision-input verification report](2026-04-27_smolvla_vision_input_verification.md)).

## Lesson

The biggest config divergence we have is the one that's hardest to see: `batch_size=4` is 16× under the published default and was inherited from an earlier GPU-memory-constrained config that no longer reflects our hardware. **Reading the code defaults isn't enough — we have to read the *tutorial* (which gives the recommended task-tuned values, not the library safe defaults).**

## Next step

Treat the three CONCERN items as the single config delta for the next training run. Keep camera resolutions and framing fixed (recently changed). Hypothesis to test: at effective batch 64, with vision-tower LoRA off, our v4 vision-grounding deficit plateau lifts — measured by both train loss curve shape and (more importantly) `attention_difference.py` argmax-residual on the new run.
