# 2026-04-27 — Scripted controller phase stretch (vla_kitting-2hp)

## Hypothesis

v4's vision attention-difference diagnostic (reports/runs/attn_diff_2026-04-27/) showed the third-person vision encoder is NOT localizing the cube — median residual-argmax error 147 px on a 256² image, i.e. invariant to cube position. One contributing factor in the training data: the scripted controller's approach window is short — Phase A (70 steps) + B (15) + C (100) = 185 steps before grasp. With smoothstep velocity profile most of those frames are decelerating into place; the dataset under-teaches "cube position drives EE motion" because there are too few frames where the cube is centrally visible AND the EE-to-cube relative pose is varying.

Hypothesis: lengthening Phase A 70→200 and Phase C 100→200 (B unchanged at 15) produces 415 pre-grasp frames per demo (+124%), with smoothstep auto-rescaling so per-step deltas shrink and the cube stays centrally visible to the third-person camera longer. This should (a) increase the volume of frames that teach cube-localization-to-EE-motion mapping in the v5 dataset, and (b) leave success rate of the scripted controller unchanged.

## Config

- File edited: `scripts/validate/scripted_pick_demo.py`
  - Phase A `steps`: 70 → 200 (line 177)
  - Phase C `steps`: 100 → 200 (line 182)
  - Phase B unchanged at 15
  - All other phases (D/E/F/G1/G2/G3/H/I/J/K) unchanged
  - Docstring updated (lines 138–141)
  - `--max_steps_per_demo` default 1000 → 1300 (line 30) — new total ~1140 would have truncated under the prior cap
  - `env_cfg.episode_length_s = 45.0` override added at script level (lines 247–248) — env default 30.0 s × 30 Hz = 900 steps would truncate the new ~38 s trajectory. Override lives in the script, NOT in `envs/yaskawa_pick_cube_cfg.py`, so prior baselines are not invalidated.

Smoke command:

```bash
/home/ubuntu/IsaacLab/isaaclab.sh -p scripts/validate/scripted_pick_demo.py \
    --num_demos 3 \
    --dataset_file /home/ubuntu/vla_kitting/datasets/teleop/2hp_smoke_2026-04-27/cube_scripted.hdf5 \
    --overwrite
```

## Baseline

Prior phase counts (v4 training data + recovery_plan_2026-04-24.md):
- A=70, B=15, C=100 → 185 pre-grasp frames per demo
- Total demo length ~910 steps @ 30 Hz ≈ 30 s

Reference for the motivation: `reports/runs/attn_diff_2026-04-27/` (residual-argmax 147 px), `reports/2026-04-26_scene_data_integrity_pack.md`.

## Result

**Smoke: 3/3 successful demos in 3 attempts** (each at 1063 total steps; success fires at phase 11 step 2).

First run (with the default `episode_length_s=30.0`) gave 0/3 — env truncated each attempt at ~900 steps mid-transport (phase 8 would always end with the env auto-resetting back to home before phase 9). Diagnosed and fixed in-script by overriding `env_cfg.episode_length_s = 45.0` per the autonomous-bug-fix authority in CLAUDE.md (resolution unambiguous: lengthening phases without lengthening the episode silently invalidates the run).

**Pre-grasp frame count, measured from the recorded HDF5 (gripper-command transition):**

| Demo    | total steps | first close-step | Phase A+B+C frames |
|---------|-------------|------------------|--------------------|
| demo_0  | 1063        | 445              | 415                |
| demo_2  | 1063        | 445              | 415                |
| demo_4  | 1063        | 445              | 415                |

**Central-visibility check on demo_0's third_person_cam (512² uint8):**
- Purple-cube pixel mask within 150 px Euclidean radius of image center, threshold ≥30 cube pixels: 415 / 415 pre-grasp frames pass.
- Same mask in a 300×300 central crop: 415 / 415 pre-grasp frames pass.
- Bd-issue acceptance bar was ">100 frames"; result is 4× the bar, ~14× the prior baseline of "~30 frames" stated in the issue.

EE–cube relative XY range over the 415-frame approach window: 0.378 m × 0.419 m (rich relative-motion signal — exactly the variation the policy needs to learn cube-localization-to-EE).

## Artifacts

- Edited file: `scripts/validate/scripted_pick_demo.py` (uncommitted, working tree only).
- Smoke dataset deleted post-verification: `datasets/teleop/2hp_smoke_2026-04-27/` (was 416 MB; cleaned to free disk).
- Run logs: `/tmp/2hp_smoke_run.log` (first failed run, 0/3), `/tmp/2hp_smoke_run2.log` (passing run, 3/3) — ephemeral, kept for the session.

## Lesson

1. **Phase-stretch + episode-length must be edited together.** Bumping `steps` in `script_trajectory_waypoints` without also extending `episode_length_s` silently truncates every demo just before the place phase, costing 0/3 in the first smoke attempt. Future scripted-controller edits that change total step count should grep `episode_length_s` first.
2. The smoothstep velocity profile auto-rescales cleanly — no hand-tuning of P-gains or rot-gains was needed at the new step counts; same per-step-delta envelope, just stretched in time.
3. Pre-grasp frame count went from 185 → 415 (+124%); central-visibility frames went from ~30 → 415 (~14×). The increase in *centrally-visible* frames is much larger than the raw frame count because the prior 70-step Phase A spent most of its frames in transit (cube outside the central radius); the new 200-step Phase A reaches the over-cube hover early enough that the cube stays in the central radius for almost all of A+B+C.

## Next step

Unblock vla_kitting-8ux (v5 LoRA-shrink + uniform-loss training run). Regenerate the v5 dataset with the stretched scripted controller, then proceed to training. Open question for v5 dataset gen: dataset disk size scales ~1.16× per demo (1063 / 910 steps), so a 100-demo dataset goes from ~14.5 GB to ~16.8 GB — fits in the current 21 GB free, but watch the 84% disk pressure.
