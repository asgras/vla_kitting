---
title: "Disk cleanup archive — 2026-04-27"
type: archive
---

# Disk cleanup before v5 training run (vla_kitting-8ux)

## Why this happened

Before launching the v5 training run (vla_kitting-8ux), disk was at 84% (21 GB free of 123 GB). v5 needs:

- ~30–40 GB for new dataset gen at the longer scripted phases (vla_kitting-2hp shipped 415 pre-grasp frames per demo vs 185 prior, ~2.2× more frames per demo × 750 demos)
- ~20 GB for 4–5 intermediate checkpoints during the 100k-step run
- LeRobot conversion adds another ~roughly equal share

The combined footprint exceeded available disk. Per CLAUDE.md "do not delete or overwrite demo datasets, checkpoints, or GIFs without archiving" — this dir is the audit trail; the heavy bytes were deleted from their original locations after metadata + GIFs were preserved here.

## What was deleted (and why)

### `datasets/lerobot/cube_pick_v3_scripted_20260425_011050/` (16 GB)
- Created 2026-04-25 by the v3 training run (370 episodes, 307,866 frames at 30 Hz). Manifest and `meta/info.json` preserved here as `v3_dataset_manifest.json` and `v3_dataset_info.json`.
- Superseded by the v5 dataset (vla_kitting-8ux), which uses the new camera config (512², repositioned, FOV 60°), the stretched scripted phases (Phase A 70→200, C 100→200), and the tightened `xy_tolerance=0.075` (vla_kitting-mil).
- Re-derivable: `scripts/validate/scripted_pick_demo.py` is the source of truth; recreate by running `scripts/orchestrate/parallel_scripted_demo_gen.sh` (see vla_kitting-8rf).

### `~/.cache/huggingface/datasets/` (16 GB)
- The HF datasets cache that the LeRobot loader populated when the v3 dataset was loaded for training. Orphaned once the v3 dataset was deleted.
- Re-derivable on demand by re-loading the v5 dataset.

### `checkpoints/v3_2_archive_20260426/checkpoints/` (4.7 GB) — adapter weights only
- v3_2 training run was archived 2026-04-26 with 0/N eval SR (per `reports/2026-04-26_run_b_prime_config_diff.md` and the broader research-log synthesis). Adapter weights are not load-bearing for any future experiment — Run B prime is the only checkpoint that ever cleared >=2/10, and that is preserved separately.
- Adapter configs (`adapter_config.json`, `config.json`, `train_config.json`, `README.md`, the two preprocessor JSONs) are preserved here under `v3_2_configs/` so the adapter shape and training recipe can be reconstructed.
- Eval GIFs (`eval_epoch_0010.gif`, `eval_epoch_0020.gif`) preserved under `v3_2_gifs/`.
- Re-derivable: re-train from the recipe in `train_config.json` if v3_2 ever needs to be reproduced for bisection.

## What is still on disk

- `checkpoints/continual/` (3.0 GB) — current/active continual-learning checkpoints + GIFs. Not touched.
- `checkpoints/sweep/` (215 MB) — small. Not touched.
- `~/.cache/huggingface/hub/` (2.8 GB) — model weights (base SmolVLA). Required for any training/eval. Not touched.
- All `reports/` — audit trail. Untouched.

## Disk before / after

- **Before**: 21 GB free / 123 GB total (84% used)
- **After (target)**: ~58 GB free (16 + 16 + 4.7 = 36.7 GB recovered)

## Reference

- `bd show vla_kitting-3my` — the disk-management bd issue
- `bd show vla_kitting-8ux` — the v5 training run that motivated this cleanup
- `reports/2026-04-26_run_b_prime_config_diff.md` — context for why v3_2 was a dead end
