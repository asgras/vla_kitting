# 2026-04-27 — Base SmolVLA vision-feature linear probe (vla_kitting-k98)

## Hypothesis
If base SmolVLA's vision encoder (the SmolVLM2 wrapper around SigLIP, with
truncated 16-layer vision stack) encodes cube position at all, a linear
probe from its patch features to `cube_xy` should score reasonably well
under leave-one-out CV. Verdict thresholds (proposed at issue filing):
- R² ≥ 0.7 → ENCODES_CUBE → frozen vision is viable for v5.
- 0.4 ≤ R² < 0.7 → PARTIAL → aux loss may help.
- R² < 0.4 → MISSING → v5 needs explicit vision supervision.

## Config
- Probe: `scripts/validate/probe_vision_features_for_cube.py`
- Model: base `lerobot/smolvla_base`, NO LoRA, no v4 adapter — pure pretrained
  weights as the v5 candidate "frozen vision" baseline would see them.
- Vision tower path: `policy.model.vlm_with_expert.vlm.model.vision_model`
  (SmolVLMVisionTransformer, expected `image_size=512`).
- Frames: 24 third-person `obs/third_person_cam[0]` from
  `/tmp/yaw_30/cube_scripted_yaw30.hdf5` (the 30-demo G1 dataset minus 6
  short auto-success stubs). Original 256×256, bilinearly upsampled to
  512×512 to match the encoder's expected input.
- Normalization: `(x/255 - 0.5) / 0.5` (SigLIP convention).
- Dtype: bfloat16 to match the encoder weights.
- Output: per-frame patch tokens of shape (1024, 768) — a 32×32 patch grid.
- Probe: leave-one-out CV ridge with inner-loop alpha selection over
  {0.01, 0.1, 1, 10, 100}.
- Three variants:
  - **A — mean-pool patch features → PCA(20) → cube_xy**
  - **B — highest-norm patch's (row, col) → cube_xy**
  - **C — per-patch L2 norms (1024-D) → PCA(20) → cube_xy**

## Result
| Variant | R² X | R² Y | Verdict |
|---|---|---|---|
| Mean-pool PCA | -0.84 | -1.05 | MISSING |
| Argmax patch location | **-0.09** | **-0.09** | MISSING (essentially "predict the mean") |
| Per-patch norms PCA | -55.0 | -64.0 | MISSING (catastrophic; norms dominated by non-cube components) |
| Best-of-3 | -0.09 | -0.09 | **MISSING** |

Sanity check on the mean-pool features themselves:
- Pairwise cosine similarity across 24 frames: **mean 0.973**, std 0.026, range [0.922, 0.999].
- Correlation between feature L2 distance and cube_xy distance: **+0.096**.

So even though the 24 frames have visibly different cube colors, lights,
yaws and positions, their mean-pooled SigLIP features cluster in a tight
cone (cos-sim ~0.97), and the small variation that exists doesn't track
cube position.

## Interpretation (with caveats)
1. **The mean-pool readout washes out cube info.** The cube occupies maybe
   5 of 1024 patches; averaging dilutes the signal by ~200×. Mean-pool
   probes are a known weak baseline for spatial recovery from ViT features.
2. **The argmax-norm patch lands on register-token positional bias, not
   the cube.** This is the same failure mode the earlier
   `attention_diagnostic_invalidated.md` flagged: SmolVLM register tokens
   produce stable corner-magnitude patches that dominate `argmax|x|`
   regardless of image content.
3. **Per-patch norms PCA is dominated by register variance.** R² = -55
   means PCA found the register-bias components and the regression
   wildly extrapolated on unseen variance directions. Not informative.
4. **N=24 is small.** Linear probes on >100D features with 23 train
   samples are data-starved even with ridge + PCA. A re-run at N≥100
   from regenerated scripted demos (under the new 512×512 camera config
   the user just landed) would be more conclusive.
5. **The frames were rendered at the OLD camera config** (256×256
   resolution, narrower FOV centered at (0.60, 0.10, 0.02) from
   (1.15, 0.10, 0.50)). The user has since moved to 512×512 + wider FOV
   from a different vantage — that change alone could materially shift
   probe results.

## Bottom-line verdict
**Suggestive — not conclusive — that v5's "frozen vision + better action
head" strategy alone is risky.** The probe failed to recover cube_xy from
any clean readout of the encoder's output, and the mean-similarity 0.97
indicates the encoder's global summary is not differentiating frames by
cube position. There remains a real possibility that *local patch-level
features* DO encode the cube (we didn't directly test that — would need a
spatial-aware probe with a known cube-pixel mask), but the readouts
that the action head can plausibly use (mean-pool / norm-weighted aggregate)
do not appear to carry usable cube-position signal.

## Implication for v5 (`vla_kitting-8ux`)
Update the v5 plan to **add an auxiliary cube-localization loss** alongside
the LoRA-shrink + uniform-loss recipe. Concretely:
- Tap the vision tower's pre-projection patch features.
- Add a small MLP head: patch_features → cube_xy regression.
- Supervise with the env-provided `cube_pos[:2]` (already in obs).
- Add to the total loss with weight ~0.1 of the action loss.

This is a small, additive change: if the encoder already encodes the cube
locally, aux loss is harmless redundant supervision; if it doesn't, aux
loss FORCES it to. Either way it removes the upstream uncertainty.

## Optional follow-up (defer)
- Render N≥100 frames at the NEW camera config (512×512, repositioned),
  rerun the probe with a spatial-aware variant (project the cube center
  into image space, average the 9 patches around the cube vs the 9
  patches at the image center, compare magnitudes / cosine to the
  detection-style "region tokens" approach).
- That would give a definitive answer to "does the encoder encode the
  cube somewhere," at a cost of ~20 minutes wall.

## Artifacts
- `scripts/validate/probe_vision_features_for_cube.py`
- `reports/runs/vision_probe_2026-04-27/probe_results.npz`
- `reports/runs/vision_probe_2026-04-27/probe_scatter.png`

## Lesson
Linear probes on small N + high-D ViT outputs are the wrong tool unless
you (a) have N at least 5× feature dim or (b) inject spatial structure
into the readout. Future probes should use a spatial-aware aggregator
(masked patches over the cube ROI) rather than mean-pool, and target
N ≥ 100.
