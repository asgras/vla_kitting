# Attention diagnosis and v5 plan

> **PARTIALLY SUPERSEDED — 2026-04-26 evening.**
> The argmax-on-image alignment check showed the "peaks" land on
> patch-positional biases (register tokens) rather than the cube or
> the magenta target. The peak/uniform attention numbers in this
> report ARE NOT a clean signal of cube grounding. Specifically the
> "Critical update" section's claims that base SmolVLA has working
> cube grounding and that fine-tuning degraded it 4 of 5 samples are
> unsafe. See `reports/2026-04-26_attention_diagnostic_invalidated.md`
> for the diagnosis and a proposed difference-based diagnostic.
> The v5 fix-list (augmentations, wider distribution, aux cube
> localization loss, longer training) is still valid — those are
> data/regularization arguments that don't depend on the attention
> interpretation.


**Date:** 2026-04-26 (afternoon, post-overnight session)
**Status:** diagnostic deep-dive after v4 ended at 0/N real SR. Captures
what we learned from cross-attention + text-attention overlays, ranks
the candidate fixes, and writes the v5 plan we're about to act on.
**Read first:** `reports/2026-04-26_overnight_session_summary.md` for
the v4 trail and the action-log diagnostic that motivated this work.

## Critical update — base SmolVLA has BETTER vision grounding than our fine-tuned model

After running the same diagnostic on `lerobot/smolvla_base` (the
pretrained SmolVLA without any of our fine-tuning), the comparison
flips the story. The earlier numbers in this report used a buggy
`embed_image` probe that returned 64 tokens (the fallback) instead of
the actual 4 (wrist) + 16 (third-person). With correct slicing, the
"cube" → vision peak attention is:

| Sample | Base SmolVLA | v4 fine-tuned | Winner |
|---|---|---|---|
| 00 | 0.023 (4.0× uniform) | 0.031 (5.4×) | finetuned |
| 01 | 0.034 (5.9×) | 0.010 (1.7×) | **base** |
| 02 | 0.023 (4.0×) | 0.010 (1.7×) | **base** |
| 03 | 0.024 (4.2×) | 0.020 (3.5×) | base |
| 04 | 0.021 (3.6×) | 0.010 (1.7×) | **base** |

In **4 of 5** samples, our fine-tuning made cube-localization WORSE
than the pretrained base. The base SmolVLA already has working
vision-language grounding for cubes (peak/uniform ~4–6× consistently);
our fine-tuning has reduced it (1.7–5.4× variably).

This is the textbook attention-layer signature of overfitting to a
narrow action-prediction shortcut. The pretrained model knows what
cubes look like; our LoRA training has shifted attention toward
whatever maximizes action-prediction loss on the narrow training
distribution, which is "memorize the average trajectory." That
average doesn't need precise vision, so the LoRA adapters learn to
weaken the vision-attention links the base used for grounding.

Implications for v5:

- **Don't worry about the model's ability to see cubes.** Base
  SmolVLA's pretraining gives it cube-localization out of the box.
- **The job is to fine-tune without breaking it.** Image
  augmentations, wider distribution, and the auxiliary
  cube-localization loss aren't just nice-to-haves for sharper
  attention — they're necessary to prevent fine-tuning from
  destroying the base capability.
- **The aux-loss head reading vision-tower features should regularize
  toward "preserve cube-localization information"**, which is a much
  cleaner objective than the indirect loss-shaping we tried with
  image augmentations alone in earlier sessions.

Compare visually:
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/base_vs_finetuned_text_attention.png`
shows the full set, sample-by-sample side-by-side.

## TL;DR

Built a cross-attention overlay that hooks SmolVLA's
`eager_attention_forward`, captures softmax probs, and renders
heatmaps showing where the model "looks" in the input image when
generating its action chunk. Ran two variants:

1. **Action → vision**: from action-query positions in the suffix-only
   forward to vision tokens.
2. **Text → vision**: from language-token positions in the prefix-only
   forward to vision tokens, sliced at specific words ("cube", "pink",
   "square").

**Headline finding:** vision tracks the cube — barely. Across 5 cube
positions, the action-attention heatmap peak shifts with the cube
position, but at peak / uniform ratios of only ~2.5–3× (a well-trained
vision system would be 10–50×). The text-attention is even weaker —
"cube" tokens point at the cube region at ~1.5× uniform. Both
heatmaps regress to the center of the image.

This is the textbook symptom of **shortcut learning + insufficient
visual precision**: the model has learned that the "average cube" sits
in the middle, and modulates only weakly from there. It's why the EE
gets within ~10 cm of the cube but never closer.

The fix that defeats the experiment (re-add `cube_pos` as a
privileged input) is off the table per user direction. The remaining
options ranked below.

## What the overlays look like

Output PNGs at
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/`:

- `epoch48_sample{00..04}_overlay.png` — action → vision per sample
- `epoch48_sample{00..04}_text_overlay.png` — text → vision per sample,
  one row per word (cube, pink, square)
- `epoch48_all_samples_compare.png` — vertical stack of all 5 action
  overlays
- `epoch48_sample00_data.npz` — raw heatmap data per sample

### Action → vision quantitative summary

| Sample | third-person peak | uniform | peak/uniform |
|---|---|---|---|
| 00 | 0.017 | 0.0057 | 3.0× |
| 01 | 0.017 | 0.0057 | 3.0× |
| 02 | 0.017 | 0.0057 | 3.0× |
| 03 | 0.014 | 0.0057 | 2.4× |
| 04 | 0.013 | 0.0057 | 2.3× |

Peak attention region tracks the cube across positions but is broad
(3–5 patches in the 8×8 grid). Each patch covers ~10 cm at table
distance, so a 3–5 patch spread = ~15–25 cm of spatial uncertainty.
This precisely matches the EE-to-cube error from the action-log
diagnostic (mean 9.6 cm, max 23 cm).

### Text → vision quantitative summary

| Word | third peak | third sum | peak/uniform |
|---|---|---|---|
| cube | 0.008–0.010 | 0.17–0.21 | 1.5× |
| pink | 0.006–0.009 | 0.11–0.13 | 1.3× |
| square | 0.006–0.007 | 0.10 | 1.1× |

The language→vision grounding is weaker than action→vision. "cube"
weakly attends to the cube; "pink"/"square" barely above noise. So
the language head treats these words as nearly generic and isn't
visually grounding them on the magenta target either.

This rules out one variant of the diagnosis (vision is fine but action
head is bad — both are blurry in the same way).

## Why I think these heatmaps explain the EE-positioning offset

Three pieces of evidence converging:

1. **Action log on epoch 48** (final_debug/action_log.csv): EE→cube mean
   distance 9.6 cm, max 23 cm; gripper closes correctly ~50% of steps but
   in empty space.
2. **Vision attention is broad, not focused** (this report): peak/uniform
   ~2.5×, cube spread across 3–5 patches = ~20 cm uncertainty.
3. **Pose-delta means biased toward generic motion** (action log):
   a[0] mean +0.021 (forward), a[1] mean -0.016 (right), a[2] mean
   -0.025 (down) regardless of cube position.

The 20 cm vision uncertainty bound matches the 9.6 cm mean / 23 cm
max EE error. The model knows "cube is somewhere over there" with
~ patch-grid resolution and reaches accordingly.

## What can be done about it (no `cube_pos`)

Ranked by impact-vs-cost given what we now see:

### (a) Image augmentations — cheapest, highest expected impact

**Why it matters here.** The regression-to-the-mean blur in the
heatmaps is the signature of a model that memorized "average cube
pixel position" rather than learning to find the cube wherever it
ends up. Random translate / color jitter / small affine breaks
that memorization shortcut: the cube ends up at different pixel
positions across training samples, and the model has to learn
features that find it regardless.

**Side effects.** Three real risks, all manageable with mild settings:

1. Train/eval distribution mismatch — minor for translate/jitter,
   bad for big rotations or crops. Skip rotations (camera is fixed).
2. Random crops can crop out the cube — don't use crops, use small
   translations only.
3. Throughput cost — a few percent slower per step. Negligible.

Cube color randomization (`mdp.randomize_cube_color`) already runs
at the env level; image-space augmentations are orthogonal.

**Concrete recipe to try:** ColorJitter brightness/contrast 0.2,
RandomAffine translate 0.05 (no rotation, no crop). Set
`dataset.image_transforms.enable=True` in the lerobot training config.

### (b) Wider spawn distribution + more demos — moderate cost, foundational

Already on the v5 todo list. Combined with (a), this is the cheapest
path to "model has to actually use vision."

### (c) Auxiliary cube-localization loss — research-grade fix

**The idea.** Train a small linear head on top of a vision-tower
hidden state to predict cube_xy. Add the head's MSE loss to the
total training loss with a small weight (e.g., 0.1). The head is
**discarded at inference** — the policy never sees cube_pos directly.
You get the "vision must learn precise localization" pressure
without the privileged-input crutch.

This is genuinely the right fix for the regression-to-the-mean
problem, because it forces the vision encoder to encode cube
position explicitly, regardless of what shortcut the policy might
otherwise prefer.

**Variants:**

1. **Concurrent auxiliary loss** (during VLA training). One lerobot
   patch (~30 lines): add an extra MLP head reading from
   `vision_tower.last_hidden_state` (or a configurable layer),
   predicting (cube_x, cube_y), MSE-weighted at 0.1, summed into
   the total loss. Discard at inference.
2. **Sequential pre-train** (separate stage). First train *only* the
   vision tower + cube-xy head on a cheap (image, cube_xy) dataset
   from random env resets. Then start the full VLA training with
   the warmed vision tower. More controllable; you can see vision
   precision before risking it on the harder VLA task.

I'd start with (1) because it's a single patch and a single
training run. Move to (2) only if (1) underdelivers.

### (d) Higher-rank vision-tower LoRA — cheap, unproven

Currently `r=64` on q/k/v/out attention projections. Could double to
128, or add LoRA to MLP up/down/gate of the vision tower. Roughly
doubles vision-tower trainable params. Free to try; no clear
evidence either way.

### (e) Higher input resolution / SmolVLM tiling — invasive

SmolVLM2 supports dynamic image tiling: a 768×768 input becomes 4
tiles of 384×384, each encoded by the vision tower independently,
producing 4× as many vision tokens. This gives the vision encoder
finer spatial resolution.

**Practical implications:**

- Our third-person frame is 256×256, smaller than the native tile
  size, so already encoded as one tile (~64 tokens). To get more
  tokens, render at 512×512 or 768×768 and update dataset schema.
- 4–9× more vision tokens means ~4–9× more attention compute on the
  vision side. Combined with language and action experts, expect
  ~2× slower training overall.
- lerobot's wrapper around SmolVLAProcessor's tiling code path
  needs verification — not all fine-tuning setups handle mismatched
  tile counts cleanly.
- Dataset regeneration cost: a multi-hour rebuild.

Save this for later if cheaper interventions plateau.

### (f) Train longer — automatic, cheap, but not solving root cause

SmolVLA's published recipe runs 100k+ steps; we've been running
~30–50k. Vision encoders take longer to fine-tune than action
heads. Combined with (a)/(b)/(c), more training time is more
likely to convert the new pressure into sharper attention.

## What we are *not* doing and why

- **Re-adding `cube_pos` as a privileged observation:** off the
  table per user direction. Would defeat the purpose of forcing
  vision-grounded policy learning, and we already have evidence
  (Run B′) that this regime works trivially — the question is
  whether the model can learn pure visual grounding.
- **Pre-training the cross-attention layer specifically:**
  cross-attention learns from end-to-end gradient signal, not from a
  cheap auxiliary task. Some research uses CLIP-style pretraining,
  but it needs meaningful negative pairs we don't have for a
  single-task setup. Not worth it at this scale.

## Overfitting concerns and what to do

The v4 loss/SR pattern (loss 0.16→0.027 over 50 epochs while real SR
stays 0/N) is overfitting, specifically: memorizing **action
sequences** (training metric) without learning
**observation→action grounding** (eval metric). Eval SR was already
flat 0/N at epoch 5 — the shortcut existed essentially from day one
of training, suggesting data shape, not training duration.

Standard fixes stack:

- (a) augmentations + (b) wider distribution + (c) aux loss are
  data-side fixes — most promising for this failure mode.
- Earlier-checkpoint selection (early-stop on eval SR rather than
  saving "last") would protect against further overfit beyond peak;
  worth wiring into orchestration.
- Weight decay tuning is a regularization-side fix; usually less
  impactful than data-side fixes for this kind of failure.

## v5 plan (committed direction)

Ordered, with checkpoints for "is this worth continuing."

**Phase 1 — env + data prep (no training).**

1. Generate the spawn/target/exclusion region visualization
   (todo #8). Verify region buffer values with user.
2. Apply env config changes (todo #10): wider cube spawn box,
   smaller magenta target, exclusion zone around target so cube
   cannot spawn within auto-success radius.
3. Slow down early scripted-pipeline phases (todo #9): more frames
   in hover-above-cube and approach phases give the model more
   "planning" frames.
4. Regenerate dataset (5–6 h scripted gen + ~1 h convert) with new
   env, dropped `cube_pos`, **larger demo count** (target 600+ given
   wider distribution).

**Gate G5.1**: open-loop scripted SR ≥ 95 % across new spawn box on
30 fresh seeds. If not, narrow the box to fix it before training.

**Phase 2 — code changes.**

5. Patch lerobot to support concurrent auxiliary cube-localization
   loss (item c.1). Linear head reading vision-tower hidden state
   → predicts (cube_x, cube_y), MSE-weighted at 0.1, discarded at
   inference. Document the patch in
   `project_lerobot_peft_resume_patch.md`.
6. Enable `dataset.image_transforms.enable=True` in train config
   with mild settings: ColorJitter brightness/contrast 0.2,
   RandomAffine translate 0.05.

**Phase 3 — training.**

7. Launch v5: same r=64 LoRA setup as v4 (vision tower attention
   projections + lm_expert + action projs), gripper × 8 (revert
   from 16 — gripper learning was already fine at 8, the bump was
   about plateau-chasing not learning), longer schedule (100 k
   steps), augmentations on, auxiliary loss on, eval every 10
   epochs.
8. **Re-run attention overlays at epochs 10, 30, 50, 100** to track
   whether the heatmap sharpens. This is the diagnostic we did NOT
   have during v4 — having it during v5 means we'll see vision
   precision improve (or fail to) in real time.

**Gate G5.2:** at epoch 25, peak/uniform action attention ≥ 5× on a
fixed test sample. If not, the auxiliary loss isn't doing its job
and we re-evaluate.

**Gate G5.3:** at epoch 50, real SR ≥ 3/30 on a fixed-seed eval.

## Files of interest

- `scripts/validate/attention_overlay.py` — the diagnostic script.
- `reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/` — all
  generated heatmaps (action and text variants).
- `reports/2026-04-26_overnight_session_summary.md` — the v4 run trail
  and the action-log diagnostic that motivated all of this.
