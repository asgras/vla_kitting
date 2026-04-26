# Next steps — post 15 Hz investigation

Written 2026-04-24. 15 Hz retrain is shelved (see `15hz_investigation_2026-04-24.md`).
Immediate plan: revert to the known-good 60 Hz pipeline, train with the
broader LoRA config + higher LR on the 60 Hz dataset.

## Resuming from a fresh Claude session — start here

**Context files to read first (in order):**
1. `reports/15hz_investigation_2026-04-24.md` — what was tried at 15 Hz and why it was shelved.
2. `reports/next_steps_2026-04-24.md` — this file.
3. `reports/overnight_run_2026-04-23.md` — the prior 60 Hz run that plateaued at 0/10 SR.

**State at the time this was written — what's already done on disk (2026-04-24 02:33 UTC):**
- `envs/yaskawa_pick_cube_cfg.py`: `decimation=2`, `scale=0.1` (reverted to 60 Hz). Verify with `grep -n "self.decimation\|scale=" envs/yaskawa_pick_cube_cfg.py`.
- `scripts/orchestrate/train_only.sh`: `N_ACTION_STEPS=50` (reverted), eval `--max_steps 1800` (reverted). Broader LoRA config IS kept: `LORA_R=32`, `LORA_ALPHA=32`, `LORA_DROPOUT=0.05`, broadened `LORA_TARGETS_REGEX`, `TRAIN_LR=1e-4`.
- `scripts/train/run_vla_closed_loop.py`: default `--max_steps 1800` (reverted).
- `datasets/lerobot/cube_pick_v1` → symlinked to `cube_pick_v1_20260423_021729` (60 Hz, 128 episodes, 205 440 frames, fps=60). Verify with `readlink datasets/lerobot/cube_pick_v1`.
- `/home/ubuntu/code/lerobot/src/lerobot/configs/default.py`: patched with `lora_alpha`, `lora_dropout` fields on `PeftConfig`. Also the `is_trainable=True` PEFT resume patch is applied. Both tracked in memory note `project_lerobot_peft_resume_patch.md`.

**What's pending — do this first in the new session:**

```bash
cd /home/ubuntu/vla_kitting

# 1. Sanity check all reverts are in place.
grep -n "self.decimation" envs/yaskawa_pick_cube_cfg.py                 # expect "self.decimation = 2"
grep -n "^N_ACTION_STEPS" scripts/orchestrate/train_only.sh             # expect "N_ACTION_STEPS=${N_ACTION_STEPS:-50}"
grep -nE "^LORA_R=|^LORA_ALPHA=|^TRAIN_LR=" scripts/orchestrate/train_only.sh  # r=32, alpha=32, LR=1e-4
readlink datasets/lerobot/cube_pick_v1                                  # expect cube_pick_v1_20260423_021729
python3 -c "import json; i=json.load(open('datasets/lerobot/cube_pick_v1/meta/info.json')); print('fps',i['fps'],'N',i['total_episodes'])"  # fps 60  N 128

# 2. Confirm no stale training processes and GPU is free.
ps aux | grep -E "lerobot_train|train_only|adapter_fixer|budget_watchdog" | grep -v grep  # expect empty
nvidia-smi --query-compute-apps=pid,process_name --format=csv  # expect only dcvagent

# 3. Launch the 60 Hz broader-LoRA training run. `--reset` wipes checkpoints/continual and starts fresh.
BUDGET_HOURS=8 bash scripts/orchestrate/train_only.sh --reset > logs/continual/orchestrator_60hz_broader_lora_launch.out 2>&1 &

# 4. Monitor.
tail -F logs/continual/train_loop.log logs/continual/orchestrator_60hz_broader_lora_launch.out
```

**What to watch for in the first hour:**
- Epoch 1 loss should be in the ~0.19 range (matches the prior 60 Hz fresh-start).
- Loss should descend faster than the prior run (target: <0.17 by epoch 10 vs the prior run's 0.174 plateau).
- First eval at epoch 10 (~50 min wall). **Success criterion: eval SR > 1/10.** If SR is 2/10 or better and the loss trend is down, let it run the full 8h. If SR stays 0/10 through epoch 30, the hypothesis failed — go to Step 2 below.

## Immediate moves (doing now)

1. **Revert env rate** — `envs/yaskawa_pick_cube_cfg.py`: `self.decimation = 8` → `self.decimation = 2`. `scale=0.1` is already correct for 60 Hz.
2. **Re-point dataset symlink** — `datasets/lerobot/cube_pick_v1` → `cube_pick_v1_20260423_021729` (the 60 Hz dataset with 128 demos, PNG frames, fps=60).
3. **Revert rate-dependent training knobs** in `scripts/orchestrate/train_only.sh`:
   - `N_ACTION_STEPS`: 12 → 50 (back to default for 60 Hz)
   - Eval `--max_steps`: 450 → 1800 (30 s at 60 Hz)
4. **Keep all other LoRA/LR changes** from the 15 Hz attempt:
   - `LORA_R=32`
   - `LORA_ALPHA=32`
   - `LORA_DROPOUT=0.05`
   - `LORA_TARGETS_REGEX` targeting q,k,v,o + gate,up,down in `lm_expert` + common projections
   - `TRAIN_LR=1e-4`
5. **Clean stale checkpoints** from the 15 Hz run, then `BUDGET_HOURS=8 bash scripts/orchestrate/train_only.sh --reset`.

Expected behavior:
- ~5 min/epoch (same as before — per-step compute is same; dataset is 4× bigger but batch is same).
- First eval at epoch 10 (~50 min wall).
- If broader LoRA is the fix, we should see SR > 1/10 at epoch 10 and rising thereafter.
- If SR stays at 0/10 by epoch 30, capacity/LR is not the blocker and the problem is data narrowness — move to Step 2 below.

## Step 2 — if capacity/LR doesn't move SR

Data diversity hypothesis: 128 demos from a single procedural controller with ±10 cm × ±13 cm cube randomization is too narrow.

1. Widen cube randomization in `envs/yaskawa_pick_cube_cfg.py:randomize_cube_pose` to x ∈ [0.40, 0.70] (±15 cm), y ∈ [−0.20, +0.20] (±20 cm).
2. Enable yaw randomization ±0.5 rad (requires teaching the scripted controller to read `cube_rot` and align `joint_6_t`, or accept that some demos will fail).
3. Regenerate Mimic to 300+ successful demos at 60 Hz with the widened distribution.
4. Re-convert and retrain.

## Step 3 — if even wider data doesn't help

Architecture-level options, in order:
1. Unfreeze the lm_expert FFN bias terms (tiny param count, highest-locality fine-tune).
2. Remove `observation.cube_pos` from the dataset features. Force visual grounding — the current model likely uses cube_pos as a shortcut (confirmed partially by the 1c ablation: both wrist-cam-zero and cube-pos-zero ablations produced MORE cube motion than baseline, suggesting neither signal is load-bearing on its own).
3. Add orientation regularization or constrain the action to 3D pos + gripper (eliminates `joint_6_t` drift observed during descent).
4. Larger LoRA rank (r=64).

## Shelved — 15 Hz regeneration (from earlier investigation)

To properly run at 15 Hz, would need:
1. Re-tune scripted controller phase step counts (divide by 4) in `scripts/validate/scripted_pick_demo.py`.
2. Regenerate scripted seed HDF5 at decimation=8.
3. Re-annotate with Mimic at decimation=8.
4. Re-run batch generation at decimation=8.
5. Re-convert (direct copy; no aggregation needed since demos are natively at 15 Hz).

Estimated: 4–6 hours of compute. Only worth doing if 60 Hz retraining hits a wall AND we've ruled out data/capacity hypotheses.

## Tooling to keep around

- `scripts/validate/replay_actions_15hz.py` — demo-action replay tool, useful for any action-alignment questions.
- `scripts/validate/replay_boost_sweep.py` — boost-sweep runner.
- `scripts/validate/render_60vs15_comparison.py` — side-by-side visualization.
- `scripts/data/isaaclab_to_lerobot.py` still has the `--stride` flag — not used at 60 Hz but keep the code.

## Environment state after revert

- `envs/yaskawa_pick_cube_cfg.py`: `decimation=2`, `scale=0.1`. Ready for Mimic generation or training.
- `datasets/lerobot/cube_pick_v1` → 60 Hz dataset.
- `scripts/orchestrate/train_only.sh`: broader LoRA defaults, 60 Hz-appropriate eval `max_steps=1800`, `N_ACTION_STEPS=50`.
- Existing 15 Hz dataset `cube_pick_v1_15hz_20260423_140430` and old checkpoints under `checkpoints/continual/` preserved for reference but not used.
