# Attention diagnostic — invalidated; what we actually know

**Date:** 2026-04-26 (afternoon, after attention overlay work)
**Status:** the heatmap diagnostic in
`scripts/validate/attention_overlay.py` is **not reliable** for the
question "does the model attend to the cube." Conclusions in
`reports/2026-04-26_attention_diagnosis_and_v5_plan.md` that depended
on it (specifically the "base SmolVLA has working cube grounding" and
the "fine-tuning degrades cube attention 4 of 5 samples" claims) are
**not safe to act on** until the diagnostic is fixed. The v5 plan
itself (augmentations, wider distribution, auxiliary cube-localization
loss, longer training) still stands on its own merits — they're
data/regularization fixes that don't depend on the attention
interpretation. Just don't quote the peak/uniform numbers from the
prior report.

## What we did

Built `scripts/validate/attention_overlay.py` to:

1. Hook `SmolVLMWithExpertModel.eager_attention_forward` and capture
   per-layer attention probabilities.
2. Run a single forward pass on a saved third-person + wrist camera
   frame pair (no Isaac Sim needed for the inference call).
3. Average attention probs across heads + the last 4 layers, slice
   the rows for either action queries (suffix-only forward) or text
   queries (specific words like "cube" / "pink" / "square" in the
   prefix-only forward).
4. Reshape the per-key attention vector for the third-person camera
   token block (16 tokens, 4×4 grid) and the wrist camera token block
   (4 tokens, 2×2 grid), upsample with bilinear interpolation, and
   render as a red-yellow heatmap overlaid on the frame.

Generated overlays for 5 cube positions, both base SmolVLA and v4
fine-tuned, for the words "cube", "pink", "square". Files at:

```
reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/
    base_sample{00..04}_overlay.png
    base_sample{00..04}_text_overlay.png
    finetuned_sample{00..04}_overlay.png
    finetuned_sample{00..04}_text_overlay.png
    base_vs_finetuned_text_attention.png      # full side-by-side
    compare_cube_attention_base_vs_finetuned.png
```

## What seemed to be true (and isn't)

After fixing an embed_image probe bug that had been falling back to
64 tokens-per-image instead of the actual 4 (wrist) + 16 (third), the
"cube" → vision peak attention values came out as:

| Sample | Base SmolVLA | v4 fine-tuned |
|---|---|---|
| 00 | 0.023 (4.0× uniform) | 0.031 (5.4×) |
| 01 | 0.034 (5.9×) | 0.010 (1.7×) |
| 02 | 0.023 (4.0×) | 0.010 (1.7×) |
| 03 | 0.024 (4.2×) | 0.020 (3.5×) |
| 04 | 0.021 (3.6×) | 0.010 (1.7×) |

I read this as: "base model has working cube-localization (4-6×
uniform), fine-tuning degrades it (1.7-5.4×)."

That conclusion is **not supported by the data** once you check where
the peaks actually land in pixel space.

## What broke the diagnostic

I drew the argmax patch (the brightest 4×4 grid cell) as a red
rectangle on the input image to verify the heatmap was localizing
correctly. Output:
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/argmax_patch_alignment_check.png`

For the BASE model's "cube" → vision attention:

| Sample | Argmax grid cell | Cube actual location | Box on cube? |
|---|---|---|---|
| 00 | (3, 2) — bottom-mid | upper-left | NO — bottom edge of table, no cube |
| 01 | (2, 2) — mid-mid-right | mid-left | NO — on the magenta target |
| 02 | (2, 2) — mid-mid-right | upper-center | NO — on the magenta target |
| 03 | (1, 2) — upper-mid-right | upper-left | NO — on table edge |
| 04 | (2, 2) — mid-mid-right | on target | partially — cube spawned on target |

In 4 of 5 samples the "cube" attention peak is NOT on the cube.

## The smoking gun: invariant positional bias

I ran the same diagnostic for the word "pink", expecting it to land
on the magenta target square (which is bright pink and impossible to
miss). Result across all 5 samples:

```
'pink' raw img2 attention values per sample:
row 0: [0.0043 0.0037 0.0037  0.0125]  ← cell (0,3) ALWAYS ~0.011-0.013
row 1: [0.0037 0.0038 0.0039  0.0081]
row 2: [0.0044 0.0062 0.0041  0.0086]
row 3: [0.0038 0.0031 0.0031  0.0048]

argmax: token 3 = grid(0,3)   ← every sample, every time
```

Cell (0, 3) — the **top-right corner of the image** — has the highest
"pink" attention in all 5 samples. The magenta target is in the
**lower-right**. Top-right is empty gray background above the table
edge. So "pink" attention isn't grounding the magenta target at all
either.

The pattern is the same shape every sample: cell (0,3) at ~0.012,
everything else 0.003-0.008. **This is a positional bias, not
semantic attention.** Vision transformers commonly have certain patch
positions that act like "register tokens" — sinks that absorb extra
attention regardless of image content. SmolVLM2's visual processing
appears to put a register-like sink at (0, 3).

## What this means for the metric I was reporting

The "peak / uniform attention ratio" I quoted is dominated by these
register positions, not by content-driven attention to the cube. A
"4× uniform peak at base, 1.7× at fine-tuned" reads, in practice, as
"the corner-register bias is mostly preserved at base and slightly
disturbed by fine-tuning" — which is a much weaker and less
interesting claim. It does NOT support the "base model can localize
cubes" conclusion.

What we **can** still say with this data:

- The model's attention is content-modulated *somewhat* — values do
  shift with cube position (sample-to-sample variance in non-corner
  cells is real).
- We do not have a clean signal showing it grounds on the cube
  specifically. The dominant signal is positional, and our
  off-positional cells move only weakly.

What we **cannot** say:

- "Base SmolVLA has working cube grounding."
- "Fine-tuning degrades cube grounding 4 of 5 samples."
- Any quantitative peak-magnitude claim from the prior report.

## What a correct diagnostic looks like

Two ways to remove positional bias and recover a meaningful signal:

### (1) Attention-difference across cube positions — preferred

Render N (≥ 5) third-person frames identical except for cube
position, on a grid like x ∈ {0.45, 0.55, 0.65}, y ∈ {-0.15, 0, +0.15}.
Run the model on each, capture the same attention rows. The
positional biases (registers, learned constants) are present in
**every** sample identically, so they cancel on subtraction. Two
useful renderings:

- For each sample i, render `(attn_i − attn_mean)` with attn_mean
  averaged across all samples. The residual heatmap should peak where
  cube i actually is, IF the model is content-modulating attention.
- For pairs (i, j), render `attn_i − attn_j`. The peak should move
  from cube_j's position to cube_i's position.

If the residuals are flat noise, that's evidence the model isn't
localizing the cube in attention. If they track the cube, we have
real grounding signal even if the absolute values were noise-floor.

### (2) Sanity test with exaggerated cube — fast gut check

Render one frame with a 50 cm bright-red cube at a known position
(scale up the cube 10×). If the model attends meaningfully on
*anything*, this should move the heatmap. If it doesn't, the
diagnostic itself is broken before we even get to model questions.

I'd run (1). It produces a quantitative answer instead of a single
anecdote, and the comparison set is what we'd want for an
attention-precision metric anyway.

## What this changes about v5

The v5 plan in `reports/2026-04-26_attention_diagnosis_and_v5_plan.md`
proposed:

- (a) Image augmentations
- (b) Wider spawn distribution + more demos
- (c) Auxiliary cube-localization loss (concurrent or sequential
  pre-train)
- (d) Higher-rank vision LoRA
- (e) Higher input resolution / SmolVLM tiling
- (f) Train longer

These are **mostly fine to keep**. They were motivated by:

- The action-log diagnostic (genuine signal: EE→cube mean 9.6 cm,
  classic "approaches the average position" failure mode).
- General overfitting indicators (loss descends 0.16 → 0.027 with
  flat 0/N eval — independent of attention).
- The wider/cleaner dataset arguments, which don't depend on
  attention interpretation.

But two specific arguments need to be **walked back**:

- "The base model already has working cube grounding, we just need
  to preserve it." That was the strongest argument for the auxiliary
  loss. It's possible but not established. The aux loss is still a
  reasonable bet, but it's no longer well-evidenced as preserving an
  existing capability — it's now hypothesized as creating one.
- "Fine-tuning has degraded grounding." Unproven. The model could
  have always been weak; we don't know.

The recommended order is unchanged but with weaker confidence:

1. Build the attention-difference diagnostic. Report whether base
   and fine-tuned models have cube-position-modulated attention at
   all.
2. Apply env spawn/target/exclusion changes (todo #10).
3. Slow down early scripted phases (todo #9).
4. Generate region visualization (todo #8).
5. Regenerate dataset, longer demo set, augmentations on, and
   train v5 with the auxiliary loss. The aux loss is now a
   "creator-of-grounding" hypothesis to test, not a "preserver."

## Files

**This report:** `reports/2026-04-26_attention_diagnostic_invalidated.md`

**Diagnostic script:** `scripts/validate/attention_overlay.py` (works
mechanically; the visualization is just dominated by positional bias
when used naively. Subtraction-based variants would fix it.)

**Existing overlays (kept for reference):** `reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/`

**Argmax-on-image alignment check (the smoking gun):**
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/argmax_patch_alignment_check.png`

**Prior (now-superseded) interpretations:**
`reports/2026-04-26_attention_diagnosis_and_v5_plan.md` — the
quantitative peak/uniform table and the "base vs fine-tuned"
ranking are unsafe; the rest of the v5 plan is fine.
