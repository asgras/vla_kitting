# Next steps — Mimic recovery + training launch

Written 2026-04-23 02:06 UTC after the first real 4h Mimic run crashed during
batch 6's merge step (rc=1). Use this as the pick-up point in a fresh Claude
Code session.

## Current state

- **Master dataset:** `datasets/mimic/cube_mimic_all.hdf5` = 103 demos (all
  successful). Last clean merge was at batch 5.
- **Unmerged:** `datasets/mimic/pool/batch_006.hdf5` (~25 more successes
  sitting there). The orchestrator crashed because
  `merge_mimic_pool.py | tail -1 | json.loads` got an empty line on this
  batch's merge — stdout capture edge case, not a data problem.
- **Failed-batch HDF5s deleted** to free disk (they were ~8 GB, no training
  value). Pool now holds only the 6 success batches.
- **Live training dataset:** `datasets/lerobot/cube_pick_v1` symlink still
  points at `cube_pick_v1_batch_25demo` (25 demos, the OLD seed). Was not
  updated — `phase_convert` never ran.
- **Disk:** 6.3 GB free of 123 GB (~95% used). `isaaclab_to_lerobot.py` with
  default PNG-per-frame output will need roughly 30–50 GB for ~128 demos. Plan
  on either trimming more (old checkpoints, /tmp, logs) OR using
  `--use_videos` to write MP4s instead of PNGs.
- **Fixes landed earlier this session (keep):**
  - `scripts/orchestrate/mimic_generate.sh`: `MIMIC_NUM_ENVS` env var, skip
    redundant phase_convert when master unchanged, full log output (no
    `tail -200`).
  - `scripts/orchestrate/train_only.sh`: `EVAL_EPISODES=10`, `EVAL_EVERY_N=10`.
  - `envs/mdp/observations.py` `cube_above_target_xy`: subtracts
    `env.scene.env_origins` for per-env target coords.
  - `envs/mdp/terminations.py` `cube_lifted_over_target` +
    `cube_placed_at_target`: same env-origins subtraction.
- **Known: 4-env Mimic still 0% success rate** even with the env-origins
  fix. Single-env works (50% per-trial, ~25 demos per 46-min batch). Stick
  with `num_envs=1` until multi-env is diagnosed separately — not a blocker.

## Step 0 — Free more disk before converting

`isaaclab_to_lerobot.py` default path (PNG-per-frame) needs tens of GB for
~128 demos. Reclaim space first:

```bash
cd /home/ubuntu/vla_kitting

# Biggest wins, usually safe:
du -sh checkpoints/continual/checkpoints/*/ | sort -h
# Keep 022000 (reference in reports) + last/ symlink target. Delete the rest
# of checkpoints/continual/checkpoints/NNNNNN that you don't need.

du -sh /tmp 2>/dev/null
# /tmp/isaaclab/logs/ in particular — multi-GB of old Isaac Sim run logs.
rm -rf /tmp/isaaclab/logs

# Old parquet output of previous convert attempts, if any:
ls -d datasets/lerobot/cube_pick_v1_20260422_* 2>/dev/null
# Safe to delete — the final convert below will make a fresh one.

df -h /home/ubuntu
```

Target ≥ 40 GB free before kicking off the convert.

## Step 1 — Recover batch 6 into master

```bash
cd /home/ubuntu/vla_kitting

/home/ubuntu/IsaacLab/_isaac_sim/python.sh scripts/orchestrate/merge_mimic_pool.py \
  --pool datasets/mimic/pool \
  --output datasets/mimic/cube_mimic_all.hdf5

# Confirm count
/home/ubuntu/IsaacLab/_isaac_sim/python.sh -c \
"import h5py; f=h5py.File('datasets/mimic/cube_mimic_all.hdf5','r'); d=f['data']; \
print('master:', sum(1 for k in d if bool(d[k].attrs.get('success',False))), \
'successes /', len(d), 'total')"
```

Expect master to go 103 → ~128.

## Step 2 — Patch the merge-output capture (small, prevents re-crash)

`scripts/orchestrate/mimic_generate.sh` has two spots that do
`$( ... merge_mimic_pool.py ... | tail -1)` then pipe to
`json.loads`. If the merger emits ANY trailing non-JSON output the capture
breaks. Make it robust by picking only JSON-looking lines:

- Find each `| tail -1` that reads merge output and change to
  `| grep -E '^\{' | tail -1`.
- Same change applies to `scripts/orchestrate/continual_train.sh` if it's
  still in use.

This is a 2-line edit and low risk.

## Step 3 — Build fresh LeRobot dataset

```bash
cd /home/ubuntu/vla_kitting
TS=$(date +%Y%m%d_%H%M%S)

PYTHONPATH=/home/ubuntu/code/lerobot/src \
  /home/ubuntu/vla_kitting/.venv/bin/python \
  scripts/data/isaaclab_to_lerobot.py \
  --input datasets/mimic/cube_mimic_all.hdf5 \
  --output datasets/lerobot/cube_pick_v1_$TS \
  --repo_id vla_kitting/cube_pick_v1 \
  --task "pick up the cube and place it on the green target"
```

- Runtime: ~1h at ~128 demos if PNG-per-frame (default); much less with
  `--use_videos`.
- If disk pressure is still tight, add `--use_videos` — shrinks the output
  ~10× at the cost of requiring ffmpeg + svt-av1 in the env.
- Expect `total_episodes` in `meta/info.json` to match the master count.

## Step 4 — Atomic symlink swap

```bash
cd /home/ubuntu/vla_kitting/datasets/lerobot
ln -sfn cube_pick_v1_$TS cube_pick_v1.next
mv -Tf cube_pick_v1.next cube_pick_v1
readlink cube_pick_v1   # should point at the new dir
```

## Step 5 — Launch long training run

Only after steps 1–4 complete cleanly.

```bash
cd /home/ubuntu/vla_kitting

BUDGET_HOURS=8 TRAIN_STEPS=1000 EVAL_EPISODES=10 EVAL_EVERY_N=10 \
  bash scripts/orchestrate/train_only.sh --reset
```

Notes:
- `--reset` wipes `checkpoints/continual/` and per-epoch logs so we start
  fresh on the new dataset. Leave off only if you want to resume — NOT
  recommended here since the dataset's normalization stats change with
  the 4× larger corpus.
- `BUDGET_HOURS=8` engages `budget_watchdog.py` for auto-stop.
- Watch `logs/continual/epoch_summary.jsonl` for loss + `eval_sr`. With
  ~128 demos the loss floor should sit below the prior 0.18; eval SR is
  the metric that matters — first non-zero hits are the real signal.

## Step 6 — Multi-env Mimic (defer until after training completes)

Single-env gives 25 successes per 46 min. 4× speedup would cut a ~3h
re-run to ~45 min. Applied `env.scene.env_origins` fix to:

- `envs/mdp/observations.py::cube_above_target_xy`
- `envs/mdp/terminations.py::cube_lifted_over_target`
- `envs/mdp/terminations.py::cube_placed_at_target`

But 4-env success rate is still 0%. Unchecked hypotheses:

1. Source demos in `cube_annotated.hdf5` store actions/states recorded at
   env_0's world origin. Mimic's `target_eef_pose_to_action` may compose
   absolute world targets on replay — in envs 1-3 those would drive the EE
   outside each env's local workspace.
2. Any other mdp function reading `root_pos_w` / `body_pos_w` without
   subtracting origins. Quick grep to audit:
   `grep -n 'root_pos_w\|body_pos_w' envs/mdp/*.py`.
3. `generation_relative=True` in our datagen cfg should transform actions to
   be EE-relative on replay, but this only helps if the source reference
   trajectory itself was stored relative. Verify by reading the Mimic env
   class's override of `target_eef_pose_to_action`.

Not a blocker for the first long training run.

## Quick sanity commands when resuming

```bash
cd /home/ubuntu/vla_kitting

# Master size
/home/ubuntu/IsaacLab/_isaac_sim/python.sh -c \
"import h5py; f=h5py.File('datasets/mimic/cube_mimic_all.hdf5','r'); d=f['data']; \
print(sum(1 for k in d if bool(d[k].attrs.get('success',False))), '/', len(d))"

# Live dataset
readlink datasets/lerobot/cube_pick_v1

# Running procs
pgrep -a -f 'mimic_generate|train_only|isaac_sim|lerobot_train'

# Disk
df -h /home/ubuntu
```
