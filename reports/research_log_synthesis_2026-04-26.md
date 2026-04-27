# VLA Cube-Pick — Comprehensive Research Log (synthesis)

**Author:** Claude (synthesized from `reports/`, `reports/runs/`, `reports/archive_*/`, JSONL eval data, and the saved chat logs at `/home/ubuntu/.claude/projects/-home-ubuntu-vla-kitting/`).
**Compiled:** 2026-04-26.
**Purpose:** A single index of every experiment, ablation, and diagnostic run on this repo so far — what was tried, what came out, and what each attempt taught us. Intended as a future-reference document; **not** a new plan. Where this conflicts with a more recent dated report under `reports/`, trust the dated report.

---

## 0. Status snapshot (2026-04-26)

- **Best ever closed-loop SR:** 2/10 in-loop, 2/20 fixed-seed (~10 %), achieved by `checkpoints/continual/checkpoints/010000/pretrained_model` (60 Hz, 128 demos, LoRA r=32 α=32, run B′ on 2026-04-24). Every other run plateaued at 0–1/10.
- **Best loss floor:** 0.017 (v3.1 epoch 30, with `load_vlm_weights=true`) — but **0/N closed-loop SR**, confirming loss is decoupled from SR.
- **No run has ever lifted the cube under closed-loop policy control.** Cube ends at z=0.025 (table) in every eval episode across every run except the spawn-on-target trivial cases (≤ 2 frames).
- **Most recent activity (Apr 25→26):** v3 → v3.1 → v3.2 swept frozen-vision → vision-LoRA → short execution chunk; all 0/N. Post-v3.2 chunk2/chunk5/RTC sweep on epoch-22 checkpoint still 0/N and surfaced an `obs_compare` finding (stale wrist frame-0 + ghost cubes in eval render) that has not yet been root-caused.
- **Current open root-cause hypothesis (highest weight):** training data flaw — saturated P-controller produces near-identical first action across all 400 scripted demos, so the policy memorizes the *first action* and never learns to use later visual feedback. (See §6.4.) Closely tied: ghost-cube / stale-wrist rendering (§6.5).

---

## 1. Project background and scope

Fine-tune **SmolVLA-450M** (LoRA) on scripted demos to pick a 50 mm cube and place it on a target.
Robot: Yaskawa HC10DT + Robotiq 2F-85. Env id: `Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0`.

> **2026-04-27 amendment.** The original scope statement (and §4.1, §4.4, §6.3 below) framed this as a Mimic-driven pipeline. As of 2026-04-27 we use SCRIPTED demos exclusively going forward; Mimic is no longer part of the data-generation plan. See `reports/2026-04-27_scripted_only_data_pipeline.md`. Historical sections describing past Mimic runs are preserved as the experimental record.
Action space: 6D EE-delta (axis-angle rotation) + 1D gripper (binary, sign-thresholded by `BinaryJointPositionActionCfg`). LIBERO-equivalent format.

| Phase | What it covered | Outcome |
|---|---|---|
| 1–5 | Asset import (HC10DT URDF→USD, Robotiq composition), env scaffolding | Stable as of mid-Apr |
| 6 | Keyboard teleop in DCV | **Blocked** by Isaac Sim 5.1 viewport-freeze (see §6.1). Teleop abandoned in V1; replaced by privileged-state scripted controller. |
| 7 | Scripted pick controller + gripper physics tuning | **Working** after gear_assembly actuator-gain port (see §6.2). |
| 8–9 | Mimic generation, LeRobot conversion, training loop | Pipeline stable, but policy never breaks 10 % SR. |

The remainder of this document is an evidence-based log of every training run, ablation, and diagnostic since the pipeline became operational.

---

## 2. Training runs (chronological)

| # | Date | Run name | Rate | Demos | LoRA r/α/dropout | LR | Targets | Special | Best SR | Loss floor | Result file |
|---|------|----------|-----:|------:|-----------------:|---:|---------|---------|--------:|-----------:|-------------|
| A0 | 2026-04-22 | First continual run | 60 Hz | 25 | (full FT, no LoRA) | 5e-5 | n/a | Resume bug — silent regress to 0.75 every resume | 0/1 @ ckpt 022000 (eval) | 0.18 (single epoch only) | `reports/next_training_plan.md` |
| A1 | 2026-04-22 | r=16 25-demo | 60 Hz | 25 | 16/16/0 | 5e-5 | q,v lm_expert | First valid LoRA run | **0/10 every epoch (ep5–ep35)** | 0.179 @ ep35 | `reports/archive_25demo_run/epoch_summary.jsonl` |
| B  | 2026-04-23 | r=16 128-demo (8h) | 60 Hz | 128 | 16/16/0 | 5e-5 | q,v lm_expert | First Mimic-amplified | **1/10 @ ep10**, 0/10 elsewhere through ep80 | 0.175 @ ep80 | `reports/overnight_run_2026-04-23.md` |
| C  | 2026-04-23 | 15 Hz r=32 broader | 15 Hz (post-hoc decim=8) | 128 | 32/32/0.05 | 1e-4 | q,k,v,o,gate,up,down lm_expert + state_proj + action_in/out_proj + action_time_mlp_in/out | Post-hoc 60→15 sum-aggregation | **0/10** at ep10/20/30 | **0.123 @ ep32** (lowest loss of any full run) | `reports/15hz_investigation_2026-04-24.md` |
| B′ | 2026-04-24 | 60 Hz r=32 broader | 60 Hz | 128 | 32/32/0.05 | 1e-4 | (same as C) | C config, reverted to 60 Hz | **2/10 @ ep10 (peak), 2/20 fixed-seed** — best ever | 0.170 @ ep41 | `reports/overnight_run_2026-04-24.md` |
| v2 | 2026-04-24 | 15 Hz wide-box Mimic (aborted) | 15 Hz native | (~40 scripted, Mimic stalled at ~25 % SR) | 64/64/0.05 | 1e-4 | + vision_tower attn + modules_to_save=[action_out_proj, action_time_mlp_out] | Time-based phases, gripper×8 loss weight, drop_cube_pos, sign-threshold | n/a — killed for research before training launched | n/a | `reports/runs/vision_grounded_30hz_2026-04-24/run_diary.md` (historical section) |
| v3.0 | 2026-04-25 | 30 Hz vision-grounded (frozen vision) | 30 Hz native | 370 scripted (no Mimic) | 64/64/0.05 | 1e-4 | lm_expert q/k/v/o/gate/up/down + action projections, **vision tower frozen**, modules_to_save=[action_out_proj] | `load_vlm_weights=true`; uniform action loss; n_action_steps=50; drop_cube_pos | **0/8 real, 0/10, 0/10** at ep10/20/30 (2 trivial spawn-on-target only) | 0.018 @ ep29 | `reports/runs/vision_grounded_30hz_2026-04-24/run_diary.md` |
| v3.1 | 2026-04-25 | + vision LoRA | 30 Hz native | 370 | 64/64/0.05 | 1e-4 | + `vlm.model.vision_model.encoder.layers.*.self_attn.(q\|k\|v\|out)_proj` (4.7 M extra params; total 18.7 M trainable) | One-variable change vs v3.0 | **0/10, 0/8, 0/7** at ep10/20/30 | 0.019 @ ep29 | (same diary) |
| v3.2 | 2026-04-25 | + n_action_steps=10 | 30 Hz native | 370 | 64/64/0.05 (vision LoRA on) | 1e-4 | (same as v3.1) | One-variable change vs v3.1: shorter execution chunk (visual feedback ≈ 5× more frequent at eval) | **1/8 @ ep10** (single bulldoze success — see §3.4), **0/9 @ ep20** | 0.022 @ ep20 | (same diary; killed at ~ep22) |

Notes:
- "Real SR" excludes trivial 2-step auto-success cases where the cube spawned within the placement tolerance and `cube_placed_at_target` fired immediately. The success terminator is too generous (it does not require a lift event). Tracked in v3.0 epoch-10 entry but never tightened.
- B′ (the 2/20 fixed-seed peak) is the only checkpoint anyone has shipped. Kept at `reports/saved_checkpoints/r16_epoch10_sr0.10/` (note: saved-checkpoint dir is misnamed — that subdir actually contains the run B r=16 checkpoint, not B′).
- v3.x runs all share the run-diary at `reports/runs/vision_grounded_30hz_2026-04-24/run_diary.md`.

### 2.1 Loss-vs-SR pattern across runs

| Run | Loss floor | Peak SR (best epoch) | Peak SR (later epoch) |
|---|---:|:---|:---|
| A1 (r=16, 25 demos) | 0.179 | 0/10 ever | 0/10 |
| B (r=16, 128 demos) | 0.175 | 1/10 @ 10 | 0/10 |
| B′ (r=32, 128 demos, broader) | 0.170 | 2/10 @ 10 | 0/10 by ep30, decay |
| C (r=32, 15 Hz post-hoc, 128 demos) | 0.123 | 0/10 ever | 0/10 |
| v3.0–v3.2 (r=64, 370 demos, 30 Hz, drop_cube_pos, load_vlm_weights) | 0.018 | 0/N — except 1/8 v3.2@ep10 (bulldoze) | 0/N |

**Lesson the table teaches:** loss is monotonically decoupling from SR. Going from 0.175 → 0.018 (10× lower) bought zero closed-loop improvement. Eval SR has to be the signal; training loss is not.

---

## 3. Eval / diagnostic experiments

### 3.1 Replay-from-training-start "is it overfitting?" test (Run B, 2026-04-23)

Cube teleported to demo_0/40/80's exact spawn position and the policy rolled out. Result: **0/3 SR on known cube positions**, identical to fresh-cube SR.
*Lesson:* the policy has not even memorized its training data. Loss being "low" reflected memorization of approach behavior, not the grasp. Underfit, not overfit. (`reports/vla_replay_trainstart_ep{0,1,2}.gif`.)

### 3.2 Zero-cube_pos vs zero-wrist ablations (Run C, ep33, 2026-04-23)

Two episodes each, fixed cube positions:

| Ablation | Episode 0 cube_end displacement | Episode 1 cube_end displacement | Lift? |
|---|---|---|---|
| `zero_cube_pos` | 0.16 m | 0.13 m (laterally) | No |
| `zero_wrist` | 0.04 m | 0.05 m | No |

The zero-cube_pos ablation moved the cube **more** than the zero-wrist ablation, and both moved the cube more than baseline (which moves it 2–17 cm). *Lesson:* neither single signal (privileged cube_pos, nor wrist cam) is load-bearing on its own; the policy is plausibly using both, weakly. Justified dropping `cube_pos` to force visual grounding in subsequent runs.

Files: `reports/ablation_zero_cubepos_episodes.jsonl`, `reports/ablation_zero_wrist_episodes.jsonl`, `reports/ablation_zero_*_ep0/1.gif`.

### 3.3 Prompt A/B "pink_square" config probe (2026-04-24)

Single eval on `checkpoints/continual/checkpoints/010000/pretrained_model` (Run B, ep10) with task string changed to "pick up the cube and place it on the pink square" and cube positioned at (0.623, 0.004) — a known-success position from extended eval.

Result: **0/1 timeout** (1800 steps). Cube_end = (0.467, 0.019), z=0.025.
*Lesson:* prompt change alone doesn't break or fix the policy at this scale (one task string, single task). Prompt is decorative for this single-task setup. Worth re-running once we have any working policy. Files: `reports/prompt_ab/`.

### 3.4 v3.2 epoch-22 chunk-size and RTC sweeps (2026-04-25 → 26)

After v3.2 was killed at ep22, three follow-up evals on the same checkpoint (`checkpoints/sweep/v3_2_ep22_*`):

| Sweep | n_action_steps | Episodes | Real SR | Notes |
|---|---:|---:|---|---|
| v3_2_ep22_chunk5 | 5 | 5 | 0/5 | All timeouts |
| v3_2_ep22_chunk2 | 2 | 10 | 0/7 (3 trivial 1–2 step "successes") | "Successes" are spawn-on-target artifacts, not picks |
| v3_2_ep22_rtc (Real-Time Chunking) | RTC | 1 | 0/1 | Tested on a **training-seen** position (0.465, 0.186) — still timed out |

*Lesson:* (a) the chunk-size knob is not the bottleneck — neither shorter (2/5/10) nor longer (50) execution-horizon produces a real lift; (b) the policy fails even on cube positions it has been trained on with RTC's prefix-attention smoothing.

Files: `reports/runs/vision_grounded_30hz_2026-04-24/sweep/{eval_chunk2,eval_chunk5,eval_rtc}.log`, `sweep_episodes.jsonl`.

### 3.5 Train-vs-eval observation comparison (2026-04-26)

`obs_compare` diagnostic compared a single training-frame against a freshly-rendered eval frame at the same cube position.

Surface findings (must be re-verified before acting):

- **Stale wrist frame-0:** the dataset's frame-0 wrist image shows gripper interior with no workspace; frame-1 onward shows the workspace. Inconsistent with a 1 mm EE motion between frame 0 and 1 — suggests rendering or camera-init lag baked into every demo's first frame. The policy's first-action prediction is conditioned on a near-blank wrist image. Plausibly load-bearing given §6.4.
- **Ghost cubes in eval render:** the third-person camera at eval time shows multiple cube-shaped artifacts (faded pink + pink + light blue in some samples). Not present in HDF5-recorded training frames. Hypothesis: with `use_fabric=True`, the cube's default-spawn copy in USD is not being cleared when the randomized pose writes to Fabric, so eval renders a stale USD copy + the live Fabric copy.
- **Prim-tree dump attempted but Isaac Sim hung at `env.reset()` for 10+ min**; killed. So the ghost hypothesis is unconfirmed.

Files: `reports/runs/vision_grounded_30hz_2026-04-24/sweep/obs_compare.log`, `obs_compare/{train,eval}_{wrist,third}.png`, `*_diff_x5.png`.

*Lesson:* before the next training run we need to know whether the eval observation distribution matches the training observation distribution. Any answer that's not "yes" voids every run since v3.0. This is currently the **highest-priority unresolved diagnostic**.

---

## 4. Data-pipeline experiments

### 4.1 Native 60 Hz Mimic generation (2026-04-23)

Standard pipeline: 25 scripted seeds → `clean_demos.py` → `annotate_demos.py` → Mimic batch loop → merge → LeRobot.

- `MIMIC_NUM_ENVS=1`: ~25 successes / 46 min batch.
- `MIMIC_NUM_ENVS=4`: **0 % success rate**. Diagnosed but not fixed: env-origins not subtracted in some `mdp/*.py` functions; one fix landed (`cube_above_target_xy` and the cube-lifted-/placed-at-target terminations) but a second still suspected (Mimic's `target_eef_pose_to_action` writing absolute world targets that fall outside non-env-0 workspaces).
- `MIMIC_BATCH=25` produced ~12 successes per batch on the wider-box config.
- 128-demo master @ 60 Hz: `cube_mimic_all.hdf5`, used for runs B, B′, C.

Files: `reports/next_steps.md` §6, `reports/next_training_plan.md` Bug #7.

### 4.2 Post-hoc 60 → 15 Hz down-conversion (2026-04-23, broken)

Attempted in `scripts/data/isaaclab_to_lerobot.py --stride 4`:
- **Sum-aggregation of 6 IK-rel deltas** + last-frame gripper. Closed-loop replay: cube peak-z = 0.025 (no pick).
- Reason: at scale=0.1 the cumulative target is 4.3× the demo's net EE motion (per §3 of `reports/15hz_investigation_2026-04-24.md`). Fed-back error compounds.
- **Observed-EE reconstruction** (`action = (ee_pose[t+stride] − ee_pose[t]) / scale`): mathematically telescopes to the demo's net EE motion, but DLS IK at decim=8 only closes 64 % of each per-call delta → arm under-traverses 36 %.
- **Boost ×1/0.64 ≈ 1.56** sweep (`replay_boost_results.jsonl`):
  | boost | peak cube z |
  |---|---|
  | 1.0 | 0.025 |
  | 1.56 | **0.270** ✓ |
  | 2.0 | 0.025 |
  | 3.0 | 0.025 |
  | 4.0 | 0.025 |
  Works on demo_0 and demo_40 but **fails on demo_80** (per-call IK convergence varies with arm configuration). No single scalar covers every demo's workspace.

*Lesson:* never down-convert a 60 Hz dataset to 15 Hz post-hoc. Generate natively at the target rate. The `--stride` code is preserved but disabled. Files: `reports/15hz_investigation_2026-04-24.md`, `reports/replay_boost_*.gif`, `replay_demo_000_15hz_*.gif`.

### 4.3 Native 15 Hz scripted gen (2026-04-23, blocked)

Direct generation at decimation=8 with the existing scripted controller: **0/3 successes in 9 attempts**. Root cause: `scripted_pick_demo.py` phase counts hard-coded for 60 Hz; at 15 Hz, total budget overshoots `episode_length_s=30`. *Lesson:* phase counts must be parameterized in seconds, not frames. Fixed in v3 (Apr 25).

### 4.4 v2 native 15 Hz wide-box Mimic (2026-04-24, aborted)

Phase 1 scripted seed: **39/40 = 97.5 % open-loop SR** on the widened box (x ∈ [0.35, 0.75], y ∈ ±0.28). Passed Gate G1 cleanly.
Phase 2 Mimic: stalled at **~25 % SR per batch** (6/24). At that pace, 400 demos = ~45 h vs the 10 h budget. Killed for research. *Lesson:* Mimic's splice-based amplification breaks down on wide-distribution boxes — splice endpoints land in unreachable / collision poses. Practitioners typically narrow the box, increase scripted-seed diversity, or skip Mimic entirely on this kind of task.

### 4.5 v3 native 30 Hz direct-scripted gen (2026-04-25, succeeded)

Skipped Mimic entirely. Generated **400 scripted demos at 30 Hz** with a re-parameterized scripted controller (phase counts in seconds), widened cube box `x ∈ [0.40, 0.70], y ∈ ±0.22`, no yaw, "pink square" task string, `--drop_cube_pos`.

- **400/400 successful** in 4h 38m. ~85 demos/hour at 30 Hz.
- After filtering placeholder-attempts and demos shorter than 50 frames (cube spawned on target → instant auto-success), **370 usable demos / 307 866 frames**.
- HDF5: 16 GB; LeRobot conversion: 16 GB; ~85 min. Final dataset: `cube_pick_v3_scripted_20260425_011050`.

This is the dataset all v3.x runs trained on. *Lesson:* scripted-only is a viable substitute for Mimic when amplification SR is < 30 %. Open question: scripted-only's monotonous first-action signature may be the cause of v3.x's mode collapse (see §6.4).

---

## 5. Infrastructure / pipeline experiments

### 5.1 Resume-config fixes (2026-04-22)

**Three coupled bugs caused silent loss-regression-on-resume in run A0** (loss 0.18 at end of epoch 22, ~0.75 on every resumed epoch):

1. `mkdir -p $CKPT_DIR` collided with LeRobot's "directory must not exist" check on fresh-start.
2. `--config_path` pointed at the dir, not the file; `Path(...).parent` resolved wrong.
3. PEFT saved `base_model_name_or_path` as the local checkpoint dir, so resume tried to load adapter on top of itself.
4. **DOUBLE PEFT WRAP** — the silent killer. On resume, `make_policy()` correctly loaded the saved adapter onto base SmolVLA; then `lerobot_train.py` saw `cfg.peft is not None` from saved `train_config.json` and called `wrap_with_peft` *again* with a zero-init LoRA on top. Optimizer trained only the new adapter; trained one was frozen. Source warning: `"You are trying to modify a model with PEFT for a second time."`
5. **LR schedule decayed to floor** by step 22000 (`scheduler_decay_steps=30000, decay_lr=2.5e-6`).

Resolutions:
- `scripts/orchestrate/fix_adapter_configs.py` daemon rewrites `base_model_name_or_path → lerobot/smolvla_base` after every save.
- Orchestrator strips `cfg.peft` from saved `train_config.json` before resume.
- One-line patch in `lerobot/src/lerobot/policies/factory.py` forces `is_trainable=True` on reloaded adapter (tracked in `project_lerobot_peft_resume_patch.md`).
- Constant LR 5e-5 (later 1e-4) via `scheduler_decay_steps=10⁶, decay_lr=5e-5`.

After these, loss descended properly across resume boundaries and the rest of the training story (runs B, B′, C, v3) became possible. Files: `reports/next_training_plan.md` Bugs 1–5.

### 5.2 Eval-script modernization

`run_vla_closed_loop.py` originally called `SmolVLAPolicy.from_pretrained(ckpt)` — failed on PEFT checkpoints (no `model.safetensors`). Three coupled patches:
- Detect `adapter_config.json` and use `PeftModel.from_pretrained` on top of base SmolVLA from HF.
- Inject `type: smolvla` into saved `config.json` (initially manual, then automated via `fix_adapter_configs.py`).
- Use the local config when instantiating base policy (HF base config expects `observation.images.camera{1,2,3}`, our task uses `wrist + third_person`).

Later additions:
- `--cube_xy "x,y;x,y"` to force cube starts (replay-from-training tests).
- `{ep}` template for per-episode GIFs.
- `--gripper_threshold` (sign threshold at inference, default 0).
- `--drop_cube_pos` (omit from inference frames; matches trained schema).
- `--zero_cube_pos` (keep key, zero values; older ablation flag).

### 5.3 LeRobot patches landed

Kept locally (re-apply if `lerobot` is re-pulled — see `project_lerobot_peft_resume_patch.md`):

1. `factory.py`: `is_trainable=True` on PEFT resume.
2. `configs/default.py`: `PeftConfig.lora_alpha`, `lora_dropout` fields.
3. `policies/smolvla/{configuration_smolvla,modeling_smolvla}.py`: per-dim `action_loss_dim_weights` buffer + broadcast in flow-matching loss. CLI: `--policy.action_loss_dim_weights='[1,1,1,1,1,1,8]'`. Coded but not used in v3 (canonical-first).
4. `peft.full_training_modules` → PEFT's `modules_to_save` (already wired in `policies/pretrained.py:364-365`; no source patch, just a CLI flag in `train_only.sh`).

### 5.4 IsaacLab / Isaac Sim patches landed

- `IsaacLab/apps/isaaclab.python.kit:41` — `"isaacsim.asset.importer.urdf" = {}` (relaxed pin from 2.4.31 to whatever ships, since 5.1 ships 2.4.19). Tracked in `project_isaaclab_kit_patch.md`.

### 5.5 SmolVLA `load_vlm_weights` gotcha (2026-04-25)

First v3 launch emitted `"Training SmolVLA from scratch using PEFT"` warning. `SmolVLAConfig.load_vlm_weights` defaults to **False**; with False, `SmolVLMWithExpertModel` constructs a fresh random VLM. `--policy.pretrained_path=lerobot/smolvla_base` only loads the action expert (~28 MB), NOT the VLM (500 MB). Killed and relaunched with `--policy.load_vlm_weights=true`. Loss dropped 5× immediately (run B′ floor 0.17 → v3 floor 0.018), confirming runs A1, B, B′, and C trained on a **random-init VLM backbone**. Tracked in `project_smolvla_load_vlm_weights_gotcha.md`.

*Lesson:* This is the single largest "we were wrong about what we were running" finding. Every pre-v3 run had a useless 500 MB random-init vision-language backbone, which makes the 60 Hz / 128-demo / r=32 plateau a less negative result than it looked: those runs were essentially LoRA-on-noise. The fact that v3 now learns much faster (loss-wise) but still 0/N closed-loop is what suggests the bottleneck is upstream of capacity.

---

## 6. Failed-and-instructive sub-investigations

### 6.1 Phase 6 — keyboard teleop in DCV (2026-04-19 → 04-20, abandoned)

Goal: 15 human-teleop demos via keyboard in DCV. Result: physics + IK + input-pipeline all confirmed working (logged EE motions of 10–16 cm under teleop), but **the Isaac Sim viewport never updates** — RTX renderer pulls transforms via the Fabric Scene Delegate, which is a no-op when `use_fabric=False`. Setting `use_fabric=True` triggers `WarpCodegenError: Could not find function wp.transform_compose` because `omni.warp.core 1.7.1` (bundled with Isaac Sim 5.1) lacks the function (added in warp ≥ 1.8). Multiple attempted workarounds (kit-file `useFabricSceneDelegate=false`, manual `quat_to_matrix` patch of `fabric.py`, OpenCV side-channel render) all hung or recompiled for 10 min. **Decision:** abandoned teleop in V1; replaced by privileged-state scripted controller. Documented at `reports/phase_6_viewport_handoff.md` for future revisit.

### 6.2 Phase 7 — gripper physics (2026-04-19 → 04-21)

The single longest debugging marathon. Eight categorized fixes over many iterations before the cube reliably lifted:

1. Cube randomization range was delta-style, not absolute → cube spawned at x=1.0 outside reach.
2. IK action scale 0.05 × 0.03 controller clamp = 1.5 mm/step → bumped to 0.1 + ±1 raw.
3. Gripper action semantics inverted (`<0 → CLOSE` per `BinaryJointPositionAction`).
4. Yaw randomization rotated cube corners outside straight-finger expectation; pinned to 0 until controller can read cube_rot.
5. Open-pose drift under inertial loading: stiffness 500 → 5000, damping 10 → 100.
6. Effort cap 200 N·m kicked the 50 g cube before friction stabilized; dropped to 50.
7. Close target 0.79 rad was past contact — pads kept pushing the cube out of the gripper. 0.79 → 0.65 → 0.5.
8. Cube tracking added per-step in phases 0–4 (was only initial pose).

**Breakthrough — gear_assembly actuator config port (2026-04-21):** copied UR10e+Robotiq actuator values from IsaacLab's working `UR10e2F85GearAssemblyEnvCfg`:
- `gripper_finger`: stiffness 0.2 → **10.0**, damping 0.001 → 0.05, effort 1 → 10, vel 1 → 10.
- `gripper_drive`: 40 / 1 / 10 (vs our prior 5000/100/50).
- **`disable_gravity=True` on robot spawn** — critical; without it fingers sag shut under gravity during fast arm motion.
- `contact_offset=0.005, rest_offset=0.0` on collision props.
- **`drive_type=force` on gripper joints** in the USD (was acceleration; URDF importer default). In acceleration mode stiffness=40 only produces ~6 rad/s natural frequency. Forced in `scripts/assembly/urdf_to_usd.py` post-conversion.
- Mimic joint constraint: gearing only, no naturalFreq / dampingRatio (high values destabilize the articulation).

After these, scripted pick lifts the cube to z=0.153 across 3/3 attempts.

**Ruled out — NVIDIA Robotiq 2F-85 USD:** loaded `assets/hc10dt_with_nvidia_gripper.usd` (composed via `compose_arm_with_nvidia_gripper.py`). Two fatal PhysX errors: nested rigid bodies (`tool0` already has RigidBodyAPI; gripper's `base_link` adds another) and unmapped joint body0/body1 absolute paths. IsaacLab's canonical UR10e composition uses a USD *variant* baked into the arm USD (`spawn.variants = {"Gripper": "Robotiq_2f_85"}`) — a different pattern than runtime references. Out of scope. Reverted to RIA URDF-derived gripper.

Documented in `reports/phase_7_findings.md`.

### 6.3 Multi-env Mimic (2026-04-23, deferred)

`MIMIC_NUM_ENVS=4` → 0 % SR vs 100 % single-env. One env-origin fix landed (`cube_above_target_xy` + cube terminations). Suspected remaining cause: Mimic's `target_eef_pose_to_action` writes absolute world targets that fall outside non-env-0 workspaces. Not blocking V1; deferred.

### 6.4 Mode-collapse hypothesis tree (v3.x, 2026-04-25 → 26)

After v3.0 GIFs showed the arm consistently grasping the same wrong location across episodes (independent of cube pose), four hypotheses were prioritized:

1. **Frozen vision lacks task-specific features.** Test: v3.1 added vision-tower attention LoRA (4.7 M extra params, total 18.7 M trainable). Result: 0/N at ep10/20/30. **Falsified.**
2. **Action chunking ignores late-frame inputs.** Test: v3.2 dropped `n_action_steps` from 50 to 10 (5× more visual-feedback queries per episode at eval). Result: 1/8 at ep10 (single bulldoze success), 0/9 at ep20. *Falsified* — the 1/8 was a lucky bulldoze, not learned behavior; cube_end x-stdev *tightened* with more training (more mode collapse, not less).
3. **Multi-cube rendering / ghost cube** (§3.5, §6.5). **Unresolved.**
4. **Training data flaw — saturated P-controller.** All 400 scripted demos use the same approach pattern; the P-gain is saturated for the first ~50 frames (home → cube), so the *first action* of every demo is roughly identical regardless of cube position. The policy may have memorized the first action and never learned to refine via subsequent visual feedback. **Currently the highest-weight unresolved hypothesis.** Cheapest test: drop controller gain from 10 to 2, regen demos, retrain.

5. **Privileged-info shortcut leak via initial EE pose.** Even with `cube_pos` removed, the IK trajectory's first delta could leak target info via `observation.ee_pose`. Untested; cheap to ablate.

### 6.5 Train-vs-eval observation mismatch (2026-04-26, in flight)

`obs_compare` flagged two artifacts in eval but not in training (§3.5). Combined with §3.4's RTC-on-training-pos failure, this is currently the most plausible "everything before this is invalid" finding. The `dump_scene_prims.py` diagnostic to confirm hung Isaac Sim at `env.reset()` for 10+ min and was killed. A non-blocking variant (`dump_scene_prims_fast.py`) was discussed but not run.

---

## 7. Things that worked

For balance, the things that have stayed solid across all this churn:

- **Scripted pick controller** at 60 Hz: 25/25 success on baseline box; 39/40 on widened box; 400/400 at 30 Hz. The controller is reliable.
- **HC10DT + Robotiq 2F-85 physics** post-§6.2 fixes: cube lifts to z=0.153 on every scripted attempt.
- **Atomic dataset symlink swap** in the continual loop: training reader sees a static path, never races the writer.
- **JSONL-everywhere observability**: `epoch_summary.jsonl`, `eval_episodes.jsonl`, `train_steps.jsonl`, `batch_summary.jsonl`. Every report in this document was reconstructed from these.
- **Resume-safe orchestration** (post-§5.1): training survives stop / restart / kit reboots without silent loss regression.
- **PEFT adapter normalization daemon** (`fix_adapter_configs.py`): keeps every saved `adapter_config.json` and `config.json` evaluable by the standalone eval script without manual intervention.

---

## 8. What the evidence currently supports

(Consolidated and ranked by weight of evidence, as of 2026-04-26.)

1. **Loss is decoupled from SR.** Going from 0.175 → 0.018 produced zero closed-loop improvement. Weight: very high. Implication: stop tuning training-side levers (LR, batch, LoRA rank, action loss weights) until the closed-loop diagnostic in §3.5 is resolved.
2. **Pre-v3 runs trained on a random-init VLM** (§5.5). Weight: confirmed. Implication: every conclusion drawn from runs A1/B/B′/C about LoRA capacity, LR, data quantity, etc. is **suspect** and should be re-tested under v3's `load_vlm_weights=true` regime before being trusted.
3. **Post-hoc rate conversion is fundamentally broken** (§4.2). Weight: confirmed via boost sweep + IK convergence math. Implication: never down-convert; native generation only.
4. **The placeholder task string is decorative**: every trained policy sees a single fixed prompt. Prompt A/B (§3.3) was a single sample but supports this. Weight: medium-high. Implication: SmolVLA's L is unused at this scale; treat the L weights as image-conditioning fine-tuning, not language conditioning.
5. **Mimic amplification has a wide-distribution ceiling near 25 %** (§4.4). Weight: medium. Implication: scripted-only can be the right call for a 1-task setup, but it introduces the saturated-first-action concern (§6.4).
6. **Frozen vs vision-LoRA is not the v3 bottleneck** (§6.4). Weight: confirmed by ablation (3 epoch points). Implication: do not spend more time on vision-tower LoRA tuning until something else moves SR off zero.
7. **Action-chunk execution horizon is not the v3 bottleneck** (§6.4). Same provenance.
8. **Train/eval observation mismatch is plausibly load-bearing** (§3.5, §6.5). Weight: speculative but well-formed. Implication: this is the cheapest unresolved diagnostic and should be run before the next training launch.

---

## 9. Hypotheses still untested

These are open and ordered by expected-information-per-hour:

- **A.** Drop scripted-controller P-gain from 10 → 2 (de-saturate first-action), regen, retrain v3.x config. Tests §6.4 hypothesis 4. Cost: ~6 h of regen + 1 night training.
- **B.** Resolve the ghost-cube and stale-wrist-frame-0 questions (§3.5). Cost: ~1 day of diagnostic time, no training compute.
- **C.** Add `cube_pos` back as a Bernoulli-dropout input (forces visual usage most of the time but rescues the head). Listed as fallback 4b in `reports/recovery_plan_2026-04-24.md`. Cost: 1 night training.
- **D.** Strip rotation dims from action (3D pos + gripper, 4D total; let env IK fix orientation). Tests whether the joint_6_t tilt observed during descent is causal. Cost: 1 night training; small code change.
- **E.** Per-dim action loss weight `gripper × 8` — coded but not tested in v3 (canonical-first). The Hole B fix from `recovery_plan_2026-04-24.md` §3. Cost: 1 night training.
- **F.** Larger LoRA rank (r=128) under the v3 `load_vlm_weights=true` regime. Re-tests the §2.1 capacity question that was confounded by random-init VLM. Cost: 1 night training.
- **G.** Different base model (π0, OpenVLA): listed in `ultraplan_context_2026-04-23.md` §P5. High-cost control experiment.

The recovery plan (`reports/recovery_plan_2026-04-24.md`) prescribes E + cube-box widening + yaw + gripper-control fixes as a bundle (Phase 1–3). The v3 work landed only some of those (broader box, drop_cube_pos, sign-threshold, vision LoRA, modules_to_save) but explicitly *deferred* the per-dim action loss weight to test "canonical first." Re-bundling E into the next experiment is consistent with that plan.

---

## 10. Reference index

Active reports:
- `reports/recovery_plan_2026-04-24.md` — currently authoritative plan (Phase 0 → 5 with gates G0-G5).
- `reports/runs/vision_grounded_30hz_2026-04-24/run_diary.md` — running diary for v2/v3/v3.1/v3.2 with all timestamps.
- `reports/known_issues.md` — wrist-cam + other gotchas.
- `reports/system_overview.md` — architecture and design intuition (verify before citing — has aged in places).

Older reports (still useful for evidence, but check for "superseded by" in `reports/README.md`):
- `reports/15hz_investigation_2026-04-24.md` — §4.2 evidence trail.
- `reports/overnight_run_2026-04-24.md` — run B′ details.
- `reports/overnight_run_2026-04-23.md` — run B details.
- `reports/phase_7_findings.md` — gripper physics breakthrough.
- `reports/phase_6_viewport_handoff.md` — teleop block.
- `reports/ultraplan_context_2026-04-23.md` — earlier context dump.
- `reports/next_training_plan.md` — resume-bug write-up (now historical).

Raw evaluation data:
- `reports/archive_25demo_run/{epoch_summary,train_steps}.jsonl` — run A1.
- `reports/archive_60hz_run/{epoch_summary,eval_episodes,batch_summary,state}.json{l,}` — run B.
- `reports/replay_boost_results.jsonl` — §4.2 boost sweep numbers.
- `reports/ablation_zero_*_episodes.jsonl` — §3.2 ablation numbers.
- `reports/prompt_ab/prompt_ab_episodes.jsonl` — §3.3 single-sample.
- `reports/runs/vision_grounded_30hz_2026-04-24/sweep/sweep_episodes.jsonl` — §3.4 chunk/RTC sweep.
- `logs/continual/epoch_summary.jsonl` — v3.x rolling epoch summaries (most recent: ep22 @ 2026-04-25 23:14, eval_sr=0.1 at ep20, no eval after).

Saved chat sessions (raw provenance, not authoritative):
- `/home/ubuntu/.claude/projects/-home-ubuntu-vla-kitting/*.jsonl` — 30+ sessions; the post-v3.2 sweep/diagnostic work is captured in `8fb49195-…jsonl` and `e890c76a-…jsonl` from 2026-04-26.

Memory entries referenced:
- `project_active_experiment.md`, `project_isaaclab_kit_patch.md`, `project_lerobot_peft_resume_patch.md`, `project_smolvla_load_vlm_weights_gotcha.md`, `feedback_autonomous_bug_fixes.md` (all under `~/.claude/projects/-home-ubuntu-vla-kitting/memory/`).
