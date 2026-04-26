# VLA cube-pick — context dump for `/ultraplan`

**Paste everything below into the ultraplan session.**
Written 2026-04-23 after the 15 Hz retrain failed to produce a policy that succeeds at the task. Everything from the live transcript, consolidated.

---

## 1. The task

Fine-tune SmolVLA-450M (LoRA) on Isaac Lab Mimic demos to pick a 50 mm cube and place it on a green target. Arm: Yaskawa HC10DT. Gripper: Robotiq 2F-85. Action space: `Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0` — 6D IK-relative pose delta + 1D gripper command.

Env rates (final):
- sim.dt = 1/120, decimation = 8 → **policy rate = 15 Hz**
- Camera render_interval = decimation → one image per policy step
- episode_length_s = 30 → eval max_steps = 450

## 2. Training runs so far

### Run A — 60 Hz, 25 demos, r=16 LoRA (q,v only)
Loss floor ~0.18, best SR = 10% at epoch 10, 0% at all later epochs.

### Run B — 60 Hz, 128 demos, r=16 LoRA (q,v only), 84 epochs
- Loss plateaued ~0.174 from epoch 40 onward
- Eval SR: 1/10 at epoch 10, **0/10 everywhere else through epoch 80**
- Best checkpoint saved at `reports/saved_checkpoints/r16_epoch10_sr0.10/`
- Critical test: replayed the policy from **exact training-demo cube starts** (demos 0, 40, 80). **0/3 SR** on those known positions. This disproved the overfitting hypothesis and pointed at severe underfitting. Loss was "low" because the policy memorized approach behavior but never learned the grasp.

### Run C — 15 Hz, 128 demos (resampled from 60 Hz), r=32 LoRA (q,k,v,o + gate,up,down), 33 epochs
- Trainable params: 7.03 M (9.5× the old 0.74 M)
- LR 1e-4 (up from 5e-5), lora_alpha=32 (up from PEFT default 8), lora_dropout=0.05
- Loss trajectory: 0.196 → 0.154 → 0.143 → 0.140 → ... → **0.123 at epoch 32**
  - Already below Run B's final plateau (0.174) by epoch 2
- **Eval SR: 0/10 at epochs 10, 20, 30.** Three consecutive zero-SR evals despite monotonically decreasing loss.
- Cube never lifts (z stays 0.025 = spawn height); arm nudges it 2–17 cm laterally.

**Training stopped at epoch 33.** Latest LoRA adapter: `checkpoints/continual/checkpoints/033000/pretrained_model/`.

## 3. Sample GIFs produced (evidence archive)

Training demos (ground truth):
- `reports/mimic_demo_000.gif`, `mimic_demo_040.gif`, `mimic_demo_080.gif`, `mimic_demo_127.gif`

Run B (60 Hz, epoch 10 best checkpoint):
- `reports/vla_rollout_epoch10_ep{0,1,2}.gif` — random cube starts, all timeouts
- `reports/vla_replay_trainstart_ep{0,1,2}.gif` — **exact training-demo cube starts**, all timeouts. This is the decisive "not overfitting" evidence.

Run C (15 Hz, epoch 33 checkpoint):
- `reports/vla_15hz_epoch33_ep{0,1}.gif` — latest attempts

## 4. Key infrastructure changes landed on branch `vla-pipeline-v1`

- `envs/yaskawa_pick_cube_cfg.py` — decimation 2 → 8 (60 Hz → 15 Hz policy rate)
- `scripts/data/isaaclab_to_lerobot.py` — added `--stride` with action aggregation: sum of 6 IK-rel deltas per stride window + last gripper value. Unit-tested on synthetic data + verified against 3 real demos (commanded-vs-observed displacement ratio ~0.023 consistent across stride=1 and stride=4)
- `scripts/orchestrate/train_only.sh` — new knobs: `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGETS_REGEX`, `N_ACTION_STEPS`; defaults bumped
- `scripts/train/run_vla_closed_loop.py` — added `--cube_xy "x,y;x,y"` override (force cube start pose for replay-from-training-demo tests), added `{ep}` template for per-episode GIFs, default max_steps 1800 → 450 for 15 Hz
- `/home/ubuntu/code/lerobot/src/lerobot/configs/default.py` — `PeftConfig` extended with `lora_alpha` + `lora_dropout` fields (this is a patch to an external clone; tracked in memory note `project_lerobot_peft_resume_patch.md`)

## 5. Dataset inventory

- `datasets/mimic/cube_mimic_all.hdf5` — 128 successful demos, 60 Hz, 1605 frames each. Cube randomization: x ∈ [0.45, 0.65], y ∈ [−0.13, +0.13], yaw ∈ [0, 0] (disabled). Gripper value is tri-state (−1 open, 0 neutral, +1 close); mostly +1 in every demo.
- `datasets/lerobot/cube_pick_v1` → `cube_pick_v1_15hz_20260423_140430` (current) — 128 episodes × 401 frames = 51,328 frames @ 15 fps, 2.7 GB PNG frames
- `cube_pick_v1_20260423_021729` — old 60 Hz, 11 GB (can delete)

Disk: 12 GB free on /home/ubuntu (91% used).

## 6. Specific empirical observations from the GIFs

- Arm does approach the cube from above — visual grounding of cube location is at least partially learned.
- Arm **does not descend low enough** to enclose the cube with the gripper fingers. It tends to hover and eventually bump/push.
- **joint_6_t (flange/wrist roll) is noticeably tilted during descent.** The action space is Cartesian pose-delta + gripper; joint_6_t is whatever the IK solver picks to satisfy the commanded EE orientation. Because the policy is emitting (possibly small but nonzero) rotational deltas (rx, ry, rz) continuously, orientation drifts cumulatively, producing a tilted gripper at grasp time.
- Cube_end z is always exactly 0.025 (spawn height). In demos, the cube ends at z ≈ 0.025–0.029 (slight float from the scripted release motion) but had a peak z of +0.23–0.25 mid-episode (actual lift). Our policy never causes any peak z.

## 7. Brainstorm of root causes (wide net)

### Data-side
1. Narrow cube randomization (±10 / ±13 cm) — 128 demos sparsely tile this box.
2. Zero yaw randomization — policy has never seen a rotated cube.
3. Single procedural trajectory archetype — one motion primitive, scaled/translated per demo.
4. Sparse gripper-transition supervision (gripper is +1 for ~95% of frames; open→close is 1–2 frames per demo, effectively lost in L2 chunk loss).
5. No failure/recovery data — all demos are clean successes. Classic covariate shift.
6. **`observation.cube_pos` shortcut.** Privileged cube position is in the obs. Policy likely regresses actions from cube_pos alone and never learns visual grasp alignment.
7. Wrist cam's 128×128 resolution is the only signal with the precision needed at grasp distance. If the policy ignores images (see #6), fine grasp is impossible.
8. 12-dim state includes 6 gripper joints, 5 of which are redundant (all driven by one finger_joint).

### joint_6_t tilt specifically
9. Orientation channels (rx, ry, rz) in the action produce cumulative drift. At training, they're tiny but nonzero; at inference, noise compounds.
10. Fix direction: either zero rot channels at inference, or switch to abs-orientation action convention, or add an orientation regularization loss.

### Training
11. Batch size 4 → noisy gradients.
12. chunk_size=50 at 15 Hz covers 3.3 s; policy only executes first 12. Long-horizon loss variance may dominate.
13. No warmup on LR=1e-4 (constant from step 0).
14. No weight decay.
15. L2 loss over a 7D action vector where gripper is ±1 and pose deltas are ~0.03 — gripper gradient is numerically dominated by pose loss. Gripper misprediction is nearly free to the loss function.
16. lora_alpha=32 with r=32 (scale 1); often alpha=2r=64 helps.
17. VLM backbone fully frozen. Could unfreeze last 1–2 layers with tiny LR, but risks catastrophic forgetting.

### 15 Hz / deployment
18. **IK controller rate interpretation.** Unverified in closed loop: does the IK action term treat the action as per-physics-step delta or per-policy-step target delta? If per-physics-step, our 15 Hz summed deltas are 4× too large; if per-policy-step, they're correct. Offline static check passed; closed-loop playback not yet done.
19. Action-chunk queue semantics — policy predicts 50 actions, enqueues first 12, re-predicts when queue empties. Possible off-by-one.
20. Image channel order (RGB vs BGR) HDF5 vs Isaac Sim runtime — unverified.
21. Preprocessor normalization stats recomputed per new dataset; should be consistent but worth confirming.

### Architecture / base model
22. SmolVLA pretraining corpus (likely Libero/Bridge) may not generalize to 6D IK-rel + Yaskawa geometry.
23. chunk_size=50 baked into pretrained action-expert weights; shrinking just slices the output.

### Env / task
24. Gripper/cube geometry: 50 mm cube, 85 mm opening. Fingers must descend well below the top of the cube before closing. Small z errors → failure.
25. `cube_placed_at_target` termination — worth reading the code to confirm it fires correctly at 15 Hz (not implicitly rate-dependent via velocity thresholds).

## 8. Prioritized plan (proposed)

### P1 — diagnose before regenerating data (cheap, high-info — ~1 h total)

**P1a. Frame-by-frame GIF analysis.** Specifically measure from rendered frames:
- Minimum gripper-to-cube z distance across the episode (does the gripper ever get below the cube's top?)
- joint_6_t angle during approach and descent (is it tilted > 15° off vertical-neutral?)
- Time between "arm near cube" and gripper-close command (or is gripper always closed?)

**P1b. Playback-in-env test.** Feed `demo_0`'s recorded 15 Hz actions one-by-one into the live env with the cube at demo_0's start pose. Render a GIF. If the recorded action stream doesn't reproduce a successful pick, hypothesis 18 (IK rate semantics at decimation=8) is confirmed and data regen is pointless until fixed.

**P1c. Image-ignorance ablation.** Run the epoch-33 policy with the wrist camera (or both cameras) replaced by zeros. If SR and behavior are basically unchanged, the policy is ignoring visual input and relying on cube_pos — confirms hypothesis 6, motivates removing cube_pos from obs in the next retrain.

**P1d. Observations-only ablation (inverse of P1c).** Run with cube_pos set to zero. If behavior goes wild, cube_pos is the main signal and we need to remove it.

### P2 — data regeneration (expensive; only fire if P1 passes cleanly)

**P2a. Widen cube randomization** x ∈ [0.40, 0.70] (±15 cm), y ∈ [−0.20, +0.20] (±20 cm), within reachable workspace.

**P2b. Add yaw randomization** ±0.5 rad, now that we want the policy to learn orientation handling.

**P2c. Regenerate ~300 Mimic demos** at 15 Hz natively (decimation=8 is already set, so any new Mimic run is natively 15 Hz). Test Mimic scripted controller still succeeds at 15 Hz before committing to a full run.

**P2d. Remove `observation.cube_pos`** from the convert script's feature set. Force visual grounding.

### P3 — training hyperparameter sweep (can run in parallel with P2)

**P3a. Batch 4 → 16** (LR may need to drop to 5e-5).
**P3b. chunk_size 50 → 24** (match short-horizon execution).
**P3c. Weight decay 0.01** on LoRA.
**P3d. Gripper-specific loss weighting** (3–5× the gripper dim's contribution to action MSE) so the open→close transition isn't drowned.

### P4 — action-space change (addresses joint_6_t tilt specifically)

**P4a. Strip rotation dims from action.** Make action 3D pose delta + gripper (4D total). The env's IK term can fix orientation to "gripper pointing down." Breaks compatibility with smolvla_base's action head but we can zero-pad.

OR

**P4b. Switch to absolute EE orientation.** Record and predict quaternion target (not delta) for the rotational components.

### P5 — architecture, last resort

**P5a. Unfreeze lm_expert FFN bias terms** (tiny param count, highest-locality fine-tune).
**P5b. Try Pi0 or Wall-X base model.** Different action-head inductive biases.

## 9. Open questions for the plan session

1. Given 0/10 SR at 3 consecutive 10/20/30 evals with loss still descending, is it likely more training alone would fix this? My read: no — we'd end up with a lower-loss variant of the same failure mode.
2. P1a/P1b/P1c ordering — which diagnostic is most likely to yield actionable info first?
3. If P2 (widen + regen) is the right call, worth waiting for the human-teleop route instead of adding more Mimic variety?
4. Is the joint_6_t tilt a cause or a symptom? Cause: unconstrained rotation deltas drift → gripper arrives tilted → no grasp. Symptom: the grasp was going to fail anyway (e.g., height issue), and the drift is irrelevant. P1a frame analysis would resolve.
5. Should we bring in a second base model (Pi0) as a control before further investing in SmolVLA?

## 10. Current branch state

- Branch: `vla-pipeline-v1`, not pushed
- Modified files pending commit:
  - `envs/yaskawa_pick_cube_cfg.py` (decimation)
  - `scripts/data/isaaclab_to_lerobot.py` (stride)
  - `scripts/orchestrate/train_only.sh` (new knobs)
  - `scripts/train/run_vla_closed_loop.py` (cube_xy override, per-ep gifs, 450 default)
  - `/home/ubuntu/code/lerobot/src/lerobot/configs/default.py` (patch to external clone)
- Logs archived from Run B at `reports/archive_60hz_run/`
- Checkpoints from Run C at `checkpoints/continual/checkpoints/` (001000 … 033000 + last)

End of context dump.
