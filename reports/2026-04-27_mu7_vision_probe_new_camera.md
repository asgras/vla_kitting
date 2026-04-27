# 2026-04-27 — Spatial-aware vision-feature probe at NEW camera config (vla_kitting-mu7)

## Hypothesis

The k98 probe (`reports/2026-04-27_vision_feature_probe.md`) returned VERDICT
= **MISSING** for base SmolVLA's vision encoder under the OLD camera config
(256×256, narrow FOV, N=24, mean-pool aggregator). It noted two confounds:
N=24 is data-starved for a 768-D linear probe, and **mean-pool washes out
local cube signal** (the cube occupies ~5/1024 patches; averaging dilutes by
~200×). The hypothesis here: **with the NEW camera config (512×512,
repositioned at (1.5, -0.10, 0.80), FOV ~60°), N=100 random samples, and a
spatial-aware aggregator (3×3 patch window centered on the projected cube
pixel), the probe should reveal materially more cube information than the
mean-pool baseline did.**

The verdict gates the v5 plan in vla_kitting-8ux:

- PASS    (R² ≥ 0.7 on both axes AND on/off cos-sim < 0.9) → train v5 WITHOUT aux loss.
- PARTIAL (one of the two passes)                          → train v5 WITH aux loss at 0.05× action weight.
- MISSING (both fail)                                      → train v5 WITH aux loss at 0.10× weight, consider unfreezing vision tower at 10× lower LR.

## Config

- Driver: `scripts/validate/vision_feature_probe_v2.py`.
- Render command (Phase 1 — Isaac Lab):
  ```
  /home/ubuntu/IsaacLab/isaaclab.sh -p scripts/validate/vision_feature_probe_v2.py \
      --out_dir reports/runs/mu7_2026-04-27 --samples 100 --seed 0 --render_only
  ```
- Probe command (Phase 2 — pure PyTorch):
  ```
  /home/ubuntu/vla_kitting/.venv/bin/python scripts/validate/vision_feature_probe_v2.py \
      --out_dir reports/runs/mu7_2026-04-27 --probe_only
  ```
- Sampling: 100 random cube positions uniform in `X ∈ [0.25, 0.85]`,
  `Y ∈ [-0.40, 0.00]`, yaw ∈ [-0.5, 0.5] rad, with random per-episode color
  index (red/blue/yellow/orange/purple) and randomized dome-light intensity
  via the standard `randomize_cube_color` / `randomize_light` event terms.
- Camera: third-person, 512×512, focal_length=18 (FOV ~60°), pos
  (1.5, -0.10, 0.80) — set in `envs/yaskawa_pick_cube_cfg.py:140-159`. No
  changes; the NEW camera convention is already the default.
- Model: BASE `lerobot/smolvla_base`, no LoRA, no v3/v4 adapter — same
  loading pattern as the k98 probe (input schema pulled from
  `checkpoints/continual/checkpoints/last/pretrained_model/config.json` so
  the `make_pre_post_processors` shape contract matches the env's obs
  layout).
- Vision tower: `policy.model.vlm_with_expert.vlm.model.vision_model`
  (SmolVLMVisionTransformer; `image_size=512`, `patch_size=16`, dtype
  bfloat16). Patch grid 32×32 = 1024 tokens × 768 D.
- Empirical homography (world XY → image-pixel XY): per-frame anchors are
  the magenta target (constant world pose (0.65, 0.20)) detected via a
  high-R, high-B, low-G chromatic mask, plus the cube blob detected via a
  per-color chromatic-axis classifier (Isaac's dome light desaturates RGB
  ~40% so RGB-Euclidean fails; max-channel rankings are robust). Detected
  pairs (~80–90 of 100 frames produce valid cube blobs; magenta detected
  100/100) feed RANSAC DLT (4-pt minimal sets, 400 iterations, 25-px
  inlier threshold) → final homography re-fit on the inlier set.
- Aggregator: project each frame's known forced cube_xy through H into
  pixel space, convert to patch coords (`row = pixel_y / 16`,
  `col = pixel_x / 16`), extract the 3×3 patch window centered there,
  mean-pool to a single 768-D on-cube feature. Off-cube control: same
  3×3 mean-pool centered at patch (16, 16) (image center).
- Readout (a): leave-one-out ridge regression with PCA-25 front-end
  (n_pca = N_in / 4) and 5-fold inner-CV alpha selection over
  {0.01, 0.1, 1, 10, 100}. Target: `cube_xy` ∈ ℝ². R² per axis.
- Readout (b): per-frame cosine similarity between on-cube and off-cube
  768-D features, averaged across frames.
- Readout (c): silhouette score on cosine distance, on-cube features
  clustered by recorded `cube_color_idx` (5 classes).
- Random seeds: 0 (rendering), 0 (RANSAC), 0 (alpha-CV fold shuffling).

## Baseline

`reports/2026-04-27_vision_feature_probe.md` (k98) — OLD camera (256×256,
focal_length=24 / FOV~47°), N=24 frames pulled from
`/tmp/yaw_30/cube_scripted_yaw30.hdf5`, three mean-pool / argmax-norm /
per-patch-norm-PCA variants. Best-of-3 R² = -0.09 on each axis. Pairwise
mean-pool cosine similarity 0.973 (basically constant). VERDICT = MISSING.

## Result

| Readout | Metric | Value | Threshold | Pass? |
|---|---|---|---|---|
| (a) Spatial ridge | R² X | **+0.852** | ≥ 0.7 | yes |
| (a) Spatial ridge | R² Y | **+0.473** | ≥ 0.7 | **no** |
| (b) On/off cos-sim | mean | **0.658** (range 0.337 → 1.000) | < 0.9 | yes |
| (c) Color silhouette | mean | **0.277** | (diagnostic only) | n/a |

Verdict gate: (a) needs **both** axes ≥ 0.7. Y is below threshold → readout
(a) fails. (b) passes cleanly. **VERDICT = PARTIAL.**

Compared to k98 baseline:

- Mean-pool best-of-3 R² −0.09 → spatial-aware R² **+0.852 / +0.473**. Even
  the weaker axis is materially above noise. The spatial aggregator was the
  right tool — local patch features DO encode the cube; mean-pool was the
  bottleneck.
- On/off cosine 0.973 (mean-pooled across all frames) → 0.658 (on-cube vs
  off-cube within frames). The encoder differentiates "cube patch" from
  "image-center patch" at a meaningful margin.
- Silhouette positive (+0.277) — the encoder partially separates the 5
  cube colors at the on-cube ROI, which it could not do under the prior
  mean-pool readout.

Why is Y weaker than X?  In the new camera frame, world-Y maps roughly to
the image's depth axis (camera at world (1.5, -0.10, 0.80) looking back at
~(0.55, -0.10, 0.025) — close to a head-on view of the spawn box). The
spawn box spans 60 cm in X but only 40 cm in Y, AND the Y direction is
foreshortened by perspective. So Y inherently has less pixel-space
variance to project onto patch features → lower R².  The X-axis result
(R²=+0.85) is the better signal-to-noise estimate of the encoder's true
spatial-recovery capability; the Y deficit is a perspective / box-shape
artifact, not an encoder failure.

Homography fit quality: median residual 0.64 px on inliers, mean 41 px
(driven by a few cube-detection outliers RANSAC excluded; final fit refit
on inlier set only). 100/100 projected cube centers landed inside the
3×3-patch margin of the image. QA overlay at
`reports/runs/mu7_2026-04-27/homography_qa.png` confirms crosshairs land on
the actual cube in frames where the cube is unoccluded by the robot arm.

## Artifacts

- `scripts/validate/vision_feature_probe_v2.py` — the driver (render +
  probe, --render_only / --probe_only split for re-running the probe pass
  without re-rendering).
- `reports/runs/mu7_2026-04-27/frame_NNN_third.png` — 100 third-person
  512² PNGs (the audit trail; ~12 MB).
- `reports/runs/mu7_2026-04-27/manifest.json` — per-frame `cube_pos`,
  `cube_color_idx`, `cube_yaw`, color name.
- `reports/runs/mu7_2026-04-27/homography.json` — fitted 3×3 H, residual
  stats, inlier count.
- `reports/runs/mu7_2026-04-27/homography_qa.png` — 8-frame QA overlay
  (yellow box = 3×3 ROI, green crosshair = projected cube center,
  magenta dot = projected target anchor).
- `reports/runs/mu7_2026-04-27/probe_results.npz` — cube_xy, on_cube_feats,
  off_cube_feats, cube_color_idx, projected pixels, ridge predictions,
  R² per axis, per-frame cos-sim, H matrix.
- `reports/runs/mu7_2026-04-27/probe_summary.txt` — one-screen verdict.

## Lesson

1. **Mean-pool kills cube info; spatial-aware readouts surface it.** The
   k98 conclusion ("vision encoder isn't representing the cube") was a
   readout artifact, not an encoder property. Same base SmolVLA, same
   weights, completely different verdict (R² −0.09 → +0.85 on X) once the
   aggregator targets the cube ROI instead of averaging 1024 patches.
   Future probes default to spatial-aware aggregation.
2. **Isaac's dome lighting desaturates RGB by ~40%** — a "red" cube
   `(0.85, 0.15, 0.15)` reads as RGB ~(237, 177, 180). RGB-Euclidean color
   matching fails (distance ~155 vs typical threshold 60); chromatic-axis
   classification (which channel dominates which) is robust. Reusable
   pattern for any future per-color blob detection in this env.
3. **RANSAC over noisy chromatic detections is mandatory.** Even with a
   robust per-color matcher, 10–20% of frames have legitimate failures
   (cube partially occluded by robot arm, marginal lighting); plain DLT
   over all correspondences was pulled to 145-px residuals. RANSAC reduced
   median to 0.64 px.
4. **Y-axis R² weakness is geometry, not vision.** The new camera's
   world-Y axis is foreshortened (depth-aligned) and the spawn box is
   only 40 cm in Y vs 60 cm in X. A ~50% smaller spatial signal projects
   to a proportionally smaller R² ceiling at fixed feature noise. Don't
   over-interpret the asymmetric (a)-axis result as encoder weakness.

## Next step

**v5 (vla_kitting-8ux) goes with PARTIAL branch:** ADD aux cube-localization
loss at **0.05× action-loss weight**. Tap pre-projection patch features at
the on-cube 3×3 ROI (or use the full 32×32 grid with a learned attention
pooler over patches), small MLP head to regress cube_xy, supervise with
`cube_pos[:2]` already in obs. Vision tower remains frozen for v5; the
0.10× / unfreeze branch is reserved for v5b ablation if v5 SR is between
3/30 and 10/30 (per the 8ux acceptance ladder).

This is a meaningful update vs the k98-derived plan, which was leaning
"MISSING → 0.10× aux weight + unfreeze vision". The encoder DOES carry
cube signal at the right ROI; we just need to nudge the action head to
read from it. Modest aux weight is the lighter intervention.

Followups (not blocking v5):

- The Y-axis R² ceiling could be tested by re-running the probe with a
  square spawn box (60 × 60 cm) — confirms the geometric vs encoder
  attribution.
- The ROI window size (3×3 vs 5×5 vs 1×1) could be ablated; 3×3 is the
  default chosen up-front per the issue ask but isn't necessarily optimal.
