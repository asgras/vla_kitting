# 15 Hz investigation — learnings

Written 2026-04-24 after attempting to move the VLA training pipeline from 60 Hz
to 15 Hz. The 15 Hz training run produced 0/10 SR at epochs 10, 20, and 30
despite loss dropping from 0.196 → 0.123. Root cause traced to a dataset
self-inconsistency at 15 Hz, which I was unable to repair cleanly in a single
session. These are the findings.

## Setup

- Robot / task: Yaskawa HC10DT + Robotiq 2F-85, Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0.
- Env pre-change: `decimation=2`, `sim.dt=1/120`, `DifferentialInverseKinematicsActionCfg(scale=0.1, use_relative_mode=True, command_type="pose", ik_method="dls")`.
- Dataset: 128 Mimic-generated successful demos in `datasets/mimic/cube_mimic_all.hdf5`, each 1605 frames × 60 Hz ≈ 27 s.
- Training goal of 15 Hz move: 4× smaller dataset, 4× faster per epoch, less redundant signal.

## What was changed for 15 Hz

1. `envs/yaskawa_pick_cube_cfg.py:305` — `self.decimation = 2` → `self.decimation = 8` (120 Hz physics, 15 Hz policy).
2. `scripts/data/isaaclab_to_lerobot.py` — added `--stride` with a sum-aggregation of the 6 IK-rel action deltas across each stride-4 window and last-value for gripper.
3. Re-converted dataset → `cube_pick_v1_15hz_20260423_140430` (128 × 401 = 51,328 frames @ 15 fps).
4. `scripts/orchestrate/train_only.sh` — bumped LoRA (r=32, alpha=32, broadened targets q/k/v/o + gate/up/down), LR=1e-4, `--policy.n_action_steps=12`, eval `--max_steps 450`.

## What happened

- Training ran cleanly, loss descended faster than the 60 Hz baseline: 0.196 → 0.123 in 33 epochs vs the 60 Hz run's 0.174 plateau after 80.
- Eval SR at epochs 10, 20, 30: **0/10 each time.**
- Arm visibly interacts with the cube in rollouts (pushes it 2–17 cm laterally) but never lifts it.

## Root-cause diagnosis

The replay test `scripts/validate/replay_actions_15hz.py` feeds the LeRobot-dataset actions for demo_0 back into the 15 Hz env with the cube teleported to demo_0's start pose. The result reveals the problem:

| Test | Env | Actions | Cube peak z | Notes |
|---|---|---|---|---|
| Baseline (known-good) | decimation=2 (60 Hz) | demo_0's recorded 60 Hz actions, unaggregated | **0.263** | Perfect reproduction, EE matches demo to <1 mm |
| Sum-aggregation at 15 Hz, default scale | decimation=8, scale=0.1 | stride=4 sum of 6 deltas per window | 0.025 | No pick. Arm moves but wrong trajectory. |
| Sum-aggregation at 15 Hz, rate-compensated | decimation=8, scale=0.025 | same | 0.025 | No pick. Arm under-moves (70 % of demo). |
| Observed-EE reconstruction | decimation=8, scale=0.1 | action = (ee_pose[t+stride] − ee_pose[t]) / scale | 0.025 | No pick. EE tracking error ~19 cm mean. |
| Observed-EE + boost=1.56 | decimation=8 | ×1.56 pose dims | **0.270 (demo_0), 0.275 (demo_40), 0.027 (demo_80)** | Works for 2/3 demos. |

### Why sum-aggregation is wrong

Mimic records `action[t] = target_eef_pose[t] − current_eef_pose[t]`. The arm always lags the target (per-call IK convergence is partial). Summing 4 such deltas:

```
sum = (tgt[0] − cur[0]) + (tgt[1] − cur[1]) + (tgt[2] − cur[2]) + (tgt[3] − cur[3])
```

captures both "where we wanted to go" and "how far behind we were." When fed through `target = current + sum * scale`, the target is set way beyond where the arm actually needs to go at 15 Hz. At scale=0.1 the commanded cumulative target is **4.3× the demo's net EE motion** — causes divergent trajectory.

### Why observed-EE alone is wrong

Setting `action = (ee_pose[t+stride] − ee_pose[t]) / scale` makes the cumulative target exactly equal to the demo's net EE motion (verified numerically — telescoping). But the env's DLS IK at `decimation=8` only closes **~64 %** of each commanded per-call delta (measured empirically — at decimation=2 the per-call closure was ~22 %, and the per-physics-tick factor is ~0.88, giving 1 − 0.88⁸ ≈ 64 % at decimation=8). The lag compounds: after N steps the arm is at `(1 − 0.36) = 64 %` of the way to the cumulative target. Arm under-traverses by 36 % in every dimension.

### Why boost is fragile

Pre-multiplying pose dims of the observed-EE action by 1/0.64 ≈ 1.56 overshoots per call, canceling the lag. Empirically works on demo_0 and demo_40 but fails on demo_80 — likely because the **per-call convergence factor varies with arm configuration** (Jacobian at a different pose has different conditioning). No single scalar boost covers every demo's workspace.

### Why native 15 Hz Mimic generation also fails

Ran `scripts/validate/scripted_pick_demo.py` at decimation=8: **0/3 successes in 9 attempts** — but not because the controller can't pick. The scripted controller's **phase step counts are hard-coded for 60 Hz**:

```
A=140, B=30, C=200, D=40, E=180, F=180, G=600, H=60, I=150, J=80, K=80
total = 1,740 steps
```

At 60 Hz, 1,740 ticks ≈ 29 s (fits in `episode_length_s=30`). At 15 Hz, 1,740 ticks = 116 s (far beyond the 30 s budget). Arm reaches the cube and closes the gripper (phases 0–3 complete), then the episode truncates before phase 4 (lift-and-place) can run.

## What we have that's reusable

- `scripts/validate/replay_actions_15hz.py` — demo-action replay in a live env, supports both `sum` and `observed` reconstruction modes, with `--action_boost` for the overshoot experiment.
- `scripts/validate/replay_boost_sweep.py` — boost-factor sweep within a single Isaac Sim session.
- `scripts/validate/render_60vs15_comparison.py` — renders a mimic demo at 60 Hz, 15 Hz, and side-by-side.
- Conversion logic in `scripts/data/isaaclab_to_lerobot.py` supports `--stride N` with the sum-aggregation mode. Broken for this env; keep the code but don't use it.
- Broader LoRA config in `train_only.sh` (r=32, alpha=32, q/k/v/o/gate/up/down targets, lora_dropout, etc.) — rate-independent and worth keeping for 60 Hz retraining.
- `lerobot/configs/default.py` patch exposing `lora_alpha`, `lora_dropout` — required to set those from CLI; tracked in memory note `project_lerobot_peft_resume_patch.md`.
- GIF evidence of the failure: `reports/demo_000_mimic_vs_replay_15hz.gif`, `reports/replay_demo_000_15hz_observed.gif`, `reports/replay_boost_*.gif`.

## Conclusion

Moving to 15 Hz natively would require:

1. Re-tune the scripted pick controller's phase step counts (divide by 4).
2. Regenerate the scripted seed HDF5 (`datasets/teleop/cube_scripted.hdf5`) at decimation=8.
3. Re-annotate with Mimic at decimation=8.
4. Re-run Mimic batch generation at decimation=8.
5. Re-convert with the stride/aggregation logic (using observed-EE mode) — but since demos would already be native 15 Hz, no aggregation needed, just direct copy.

This is a multi-hour, multi-phase effort. The hypothesized training benefits (bigger LoRA, higher LR, broader targets) are all rate-independent and should be tested at 60 Hz first. If they move SR meaningfully, the 15 Hz work is premature optimization. If they don't move SR, the problem is probably capacity/diversity, not rate, and 15 Hz still doesn't help.

The 15 Hz branch is shelved, not abandoned. The infrastructure to revisit it is in place.
