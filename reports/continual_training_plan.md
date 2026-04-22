# Continual training plan — run for many hours unattended

This document tells a **fresh Claude Code session** exactly what state the
pipeline is in, how to start/stop it, what to monitor, and what success looks
like. It is deliberately self-contained — you should not need the prior
session's context to resume from here.

## Goal

Regenerate all data from scratch, then:

1. Seed with ~25 scripted pick-place demos, clean, and annotate once.
2. Run Isaac Lab Mimic generation **forever** in the background, accumulating
   successful demos into a growing pool.
3. Start SmolVLA training as soon as ≥40 successful Mimic demos exist.
4. Each training epoch, rebuild the LeRobot dataset from the *latest* Mimic
   pool and resume training from the last checkpoint. New demos get folded
   into the training set continuously.
5. Keep both loops running overnight / over days until explicitly stopped.

## Top-level command

```bash
cd /home/ubuntu/vla_kitting
bash scripts/orchestrate/continual_train.sh
```

That script does everything below. It is safe to re-run — each phase skips
work that's already been done. To fully restart from scratch:

```bash
bash scripts/orchestrate/continual_train.sh --reset
```

All state lives under `logs/continual/` and `datasets/{teleop,mimic,lerobot}`.
Checkpoints go to `checkpoints/continual/`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Seed (one-time, ~45 min)                                        │
│    scripts/validate/scripted_pick_demo.py  → cube_scripted.hdf5  │
│    scripts/data/clean_demos.py             → cube_scripted_clean │
│    scripts/data/annotate_demos.py          → cube_annotated.hdf5 │
└──────────────────────────────────────────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       │                                               │
       ▼                                               ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ Mimic batch loop (bg)    │              │ Training loop (fg)       │
│ forever:                 │              │ wait until >= 40 demos   │
│   mimic-generate 25 →    │              │ forever:                 │
│     pool/batch_NNN.hdf5  │              │   rebuild LeRobot from   │
│   merge pool → master    │              │     current master       │
│   rebuild LeRobot symlink│              │   train K steps, resume  │
│   log stats              │              │   save ckpt + eval       │
└──────────────────────────┘              └──────────────────────────┘
```

**File layout** (all under `/home/ubuntu/vla_kitting`):

```
datasets/
  teleop/
    cube_scripted.hdf5           seed, one-time
    cube_scripted_clean.hdf5     seed, one-time
    cube_annotated.hdf5          seed, one-time
  mimic/
    pool/batch_NNN.hdf5          one file per Mimic invocation
    cube_mimic_all.hdf5          merged master; rebuilt after each batch
  lerobot/
    cube_pick_v1_batch_NNN/      one LeRobot dataset snapshot per batch
    cube_pick_v1 -> .../batch_NNN   symlink; swapped atomically

checkpoints/
  continual/                     LeRobot train output dir; has checkpoints/{NNNNNN, last}

logs/
  continual/
    mimic_loop.log               rolling log of the background Mimic loop
    train_loop.log               rolling log of the training loop
    epoch_summary.jsonl          one line per training epoch (step, loss, eval_sr, num_demos)
    batch_summary.jsonl          one line per Mimic batch (batch_num, new_demos, total)
    state.json                   { epoch: N, mimic_batch: M, demos_total: K, last_ckpt: "..." }
```

## Resumption contract for future sessions

Each loop keeps a small JSON-lines log so any session can reconstruct state:

- `batch_summary.jsonl`: `{"ts": iso, "batch": N, "new_demos": K, "total_demos": T}`
- `epoch_summary.jsonl`: `{"ts": iso, "epoch": E, "step": S, "loss": L, "eval_sr": R, "num_demos": T}`

The top-level `state.json` is updated atomically after each epoch and each
batch. If the orchestrator dies, re-running the same command picks up from
the last completed batch / epoch.

## Knobs worth knowing

Defaults in `scripts/orchestrate/continual_train.sh`:

| Knob              | Default | Why                                            |
|-------------------|---------|------------------------------------------------|
| `SEED_DEMOS`      | 25      | Scripted demos to generate for the Mimic seed  |
| `MIMIC_MIN_DEMOS` | 40      | Wait this many successful demos before training|
| `MIMIC_BATCH`     | 25      | Trials per Mimic invocation (~12 successes)    |
| `TRAIN_STEPS`     | 1000    | Per epoch. ~4 min on L40S at batch_size=4      |
| `TRAIN_BATCH`     | 4       | Small because memory headroom matters with     |
|                   |         | Isaac Sim running in parallel                  |
| `EVAL_EPISODES`   | 2       | Closed-loop rollouts at epoch end (~3 min)     |
| `EVAL_EVERY_N`    | 5       | Run eval every N epochs                        |

Override by editing the script or exporting env vars before launching:
```bash
MIMIC_BATCH=50 TRAIN_STEPS=2000 bash scripts/orchestrate/continual_train.sh
```

## How to monitor a running job

```bash
# rolling tails
tail -f /home/ubuntu/vla_kitting/logs/continual/mimic_loop.log
tail -f /home/ubuntu/vla_kitting/logs/continual/train_loop.log

# summary lines
jq . /home/ubuntu/vla_kitting/logs/continual/state.json
tail -20 /home/ubuntu/vla_kitting/logs/continual/epoch_summary.jsonl | jq .
tail -20 /home/ubuntu/vla_kitting/logs/continual/batch_summary.jsonl  | jq .

# which processes are alive
pgrep -af 'mimic_loop|train_loop|generate_dataset|lerobot_train'

# GPU use
nvidia-smi
```

To stop cleanly:

```bash
touch /home/ubuntu/vla_kitting/logs/continual/STOP
# both loops check for this file after each iteration and exit gracefully
```

## What "done" looks like

There is no "done" by default — the plan is designed to run until stopped.
Reasonable stopping criteria:

- Eval success rate hits >= 70% on held-out cube positions for 5 consecutive
  epochs (written in `epoch_summary.jsonl`).
- Dataset grows past ~750 demos (Phase 9 target from the original plan).
- Out of disk (each episode is ~150 MB; monitor `df -h datasets/`).

## Known caveats (2026-04-22)

- Wrist cam was recently re-aimed. The current config in
  `envs/yaskawa_pick_cube_cfg.py` mounts at `pos=(0.08, 0, 0.05)` with
  `rot=(0.66095, -0.25129, -0.25129, 0.66095)`, focal 12mm. Verified on
  `reports/wrist_iter/03_wider.png` — cube visibly centered in frame.
- Scripted pick-place success is 100% on the current env config.
- Mimic success rate was ~33-50% in the last smoke; should be higher now
  that `action_noise` dropped from 0.03 to 0.02 in
  `envs/yaskawa_pick_cube_mimic_env_cfg.py`.
- `max_num_failures` raised 50 → 500 so one unlucky streak doesn't stop a
  batch early.
- Training uses full-parameter fine-tune (100M trainable) — no LoRA yet.
  With only tens of demos this WILL overfit. Acceptable for pipeline
  validation; swap in LoRA before a "real" training pass.
- The old `cube_scripted_smoke.hdf5` etc. are not auto-deleted; `--reset`
  wipes everything under `datasets/{mimic,lerobot}` and the seed files, but
  preserves unrelated data.
