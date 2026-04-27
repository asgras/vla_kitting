---
title: "vla_kitting-8rf — parallel scripted demo orchestrator"
date: 2026-04-27
type: feature
status: shipped
---

# Parallel scripted demo generation orchestrator

## Hypothesis
`scripts/validate/scripted_pick_demo.py` runs single-process with `num_envs=1`. After vla_kitting-2hp lengthened phases A and C, a single demo takes ~1.7 min wall-clock at 30 Hz. Generating ~750 demos serially → ~21 h. The L40S (46 GB VRAM) easily holds 4–5 concurrent Isaac Sim apps (~7–9 GB each), so fan-out via independent processes with distinct seeds should cut wall-clock proportionally.

## Config
- New: `scripts/orchestrate/parallel_scripted_demo_gen.sh` — bash orchestrator that launches K shards in parallel, waits, reports per-shard exit codes. Args: `--total N --shards K --run_id <slug> --base_seed S --stagger_s <sec> --max_steps N`.
- New: `scripts/orchestrate/merge_scripted_shards.py` — h5py merger that walks `shard_<i>.hdf5` files in numerical order and concatenates demo groups under sequential `demo_<K>` keys into a single `merged.hdf5`. Sanity-checks per-shard real-demo counts vs the merged file. Optional `--delete_after_copy` halves peak disk usage by unlinking each shard right after its contents land in the merged file.
- Edit: `scripts/validate/scripted_pick_demo.py` — added `--seed` plumbed through to `env_cfg.seed`. Distinct seeds per shard guarantee distinct cube position / yaw / color sequences across shards.

Smoke command:
```
bash scripts/orchestrate/parallel_scripted_demo_gen.sh \
    --total 20 --shards 2 --run_id smoke_8rf_2026_04_27 --stagger_s 30
```

## Baseline
Single-process serial demo generation at the post-2hp phase lengths: ~1.7 min/demo extrapolated from the 2hp smoke run (3 demos in 4 min, single-shard).

Estimated serial wall-clock for 750 demos: ~21 h (1.7 min × 750).

## Result
**Smoke 2 × 10 demos, wall-clock 17m12s** (orchestrator log timestamps 02:19:43 → 02:36:55). Both shards finished `rc=0` with 10/10 successful pick-and-place each. Per-shard HDF5 size 1.44 GB (gzip-compressed; uncompressed nominal ~5.7 GB per shard for 1063 frames × 2 cameras × 512²×3 + 256²×3 + states/actions).

Merge: `merge_scripted_shards.py` ran cleanly — 40 groups (20 real demos + 20 placeholder) concatenated into merged.hdf5 (2.9 GB). Sanity check passed.

LeRobot conversion (partial measurement; killed mid-run after sufficient sizing data): ~75 MB / demo on disk for the new 512² + longer-phase demos. Extrapolation:
- 200 demos → ~15 GB on LeRobot
- 380 demos → ~28 GB on LeRobot
- 752 demos → ~56 GB on LeRobot

Wall-clock extrapolation for v5 production at 4 shards parallel:
- 4 × 50 demos = 200 demos: ~17m × 5 = ~85 min wall-clock
- 4 × 95 demos = 380 demos: ~17m × 9.5 = ~2h45m
- 4 × 188 demos = 752 demos: ~17m × 19 = ~5h20m

## Disk math (load-bearing for v5 production sizing)

Pre-v5: 57 GB free. Per-demo disk (raw HDF5): 144 MB. Per-demo on LeRobot: ~75 MB.

| demos | shards × demos | peak HDF5 | merged | LeRobot | merged + LeRobot peak |
|------:|:--------------:|----------:|-------:|--------:|----------------------:|
| 200   | 4 × 50         | 28.8 GB   | 28.8 GB| 15 GB   | 43.8 GB ≤ 57 ✅      |
| 240   | 4 × 60         | 34.5 GB   | 34.5 GB| 18 GB   | 52.5 GB ≤ 57 ✅ tight |
| 380   | 4 × 95         | 54.8 GB   | 54.8 GB| 28 GB   | 82.8 GB > 57 ❌      |
| 752   | 4 × 188        | 108 GB    | 108 GB | 56 GB   | 164 GB ≫ 57 ❌       |

**Conclusion**: at the new long phases + 512² resolution, the v5 production target needs to be ≤ 240 demos, OR the merge → conversion pipeline needs to be streamed (per-shard convert + delete shard, then concatenate LeRobot datasets). For tonight, going with **200 demos** (safe headroom) for vla_kitting-8ux; this is fewer than Run B prime's 370 but at 4× the visual resolution and 2× the per-demo frame count, so total visual-token volume in the dataset is comparable.

## Artifacts
- `scripts/orchestrate/parallel_scripted_demo_gen.sh`
- `scripts/orchestrate/merge_scripted_shards.py` (with `--delete_after_copy`)
- `scripts/validate/scripted_pick_demo.py` (now honors `--seed`)
- Smoke logs (deleted after sizing collected): formerly `logs/parallel_demo_gen_smoke_8rf_2026_04_27/`
- Smoke datasets (deleted): `datasets/teleop/parallel_smoke_8rf_2026_04_27/`, `datasets/lerobot/cube_pick_smoke_v5_size_test/`

## Lesson
Parallel speedup on the L40S is roughly the shard count up to 4–5 procs, but per-demo HDF5 size at 512² is ~5× the v3 (256²) baseline. The 2hp longer-phase change adds another 2× per-demo frame count. Together that is ~10× more disk per demo than v3 — so the natural "match Run B's 370 demos" target was disk-infeasible without streamed conversion. We hit the sizing constraint before hitting the wall-clock or compute constraint.

## Next step
Run vla_kitting-8ux with `parallel_scripted_demo_gen.sh --total 200 --shards 4 --run_id v5_2026_04_27 --stagger_s 30`. If v5 lands ≥ 3/30 SR, file a follow-up beads issue to implement streamed per-shard LeRobot conversion so we can reach 380+ demos for v5b without the disk bottleneck.
