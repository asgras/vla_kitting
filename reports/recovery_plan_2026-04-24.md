# VLA Cube-Pick Recovery Plan — 2026-04-24

**Status:** supersedes `reports/next_steps_2026-04-24.md` as the active plan.
**Author:** Claude + Andrew, synthesized from the 2026-04-22 → 2026-04-24 report trail.
**Scope:** full restart from scripted-demo generation through training and closed-loop eval.

> **2026-04-27 amendment — Mimic is OUT.** Going forward we generate the
> training dataset by SCRIPTED demos exclusively. Phase 2 below was
> originally a Mimic-generation phase; it has been rewritten as
> "scripted-demo scale-up". Any reader who arrived at the Phase 2 section
> via a stale link should treat the original wording as historical only.
> See `reports/2026-04-27_scripted_only_data_pipeline.md` for the rationale.

---

## 1. Executive summary

Two runs at 60 Hz (r=16 and r=32) and one run at 15 Hz have all produced 0–2 / 10 closed-loop success rates. The trail of ablations, replay sweeps, and loss curves says the bottleneck is **not** LoRA capacity or training schedule — it is the upstream data pipeline plus a gripper control-path mismatch. More compute on the current setup will not break through.

**The three things that have to change together:**

1. **Data distribution** — generate a much broader, natively-15 Hz scripted dataset (hundreds of demos, cube XY expanded to nearly the full visible table, yaw variation added) and **drop `cube_pos` from observations** so the policy has to learn the visual grasp. (Mimic was originally part of this plan; dropped 2026-04-27 — see Phase 2 below and the dated decision report.)
2. **Gripper control path** — threshold the policy's gripper output at inference to match the env's binary wrapper, add a loss weight on the gripper dim during training, and lengthen the visible close phase in demos so the transition isn't drowned in arm-joint signal.
3. **LoRA surface** — raise rank to 64, add `modules_to_save` for the action head, include the vision encoder's attention projections in targets, and keep adapter dropout at 0.05.

**Verdict on restarting at 15 Hz:** Yes — with one caveat. The 15 Hz attempt was shelved because of fixable scripted-controller bugs (phase step counts hardcoded for 60 Hz) and an action-aggregation artifact in the 60→15 converter, not because 15 Hz itself is wrong. Given everything we now know, a clean restart at 15 Hz is the *better* path: 4× less frames to iterate through, no aggregation math to get wrong, and it matches SmolVLA's expected control rate. The caveat is that we haven't yet confirmed a SmolVLA + Yaskawa policy can learn this task at 15 Hz even with perfect data, so we keep a 60 Hz fallback branch at gate G3.

---

## 2. What we actually learned (the evidence)

Pulled from `reports/15hz_investigation_2026-04-24.md`, `overnight_run_2026-04-24.md`, `phase_7_findings.md`, `ultraplan_context_2026-04-23.md`, and the ablation JSONLs.

| Finding | Evidence | Implication |
|---|---|---|
| Policy regresses from `cube_pos`, not from vision | Zero-`cube_pos` ablation: policy wanders far off target. Zero-wrist-cam ablation: baseline-identical timeouts. | Vision is a passenger. Remove the shortcut. |
| Wrist camera is mostly black | GIFs of `wrist_cam` and `save_camera_samples.py` output | Mount offset wastes model capacity on a useless input. |
| Gripper close is 1–2 frames in the aggregated dataset | `isaaclab_to_lerobot.py:154-160` takes gripper at stride-end only | 1 frame of `-1.0` inside a 1600-frame demo is statistically invisible under L2 over 7 dims. |
| Env uses binary threshold; policy emits continuous | `BinaryJointPositionActionCfg` in `envs/yaskawa_pick_cube_cfg.py:193-202`; no threshold in `run_vla_closed_loop.py:243-250` | A policy output of `0.3` opens the gripper; `-0.01` closes it. Train on continuous, deploy on sign. |
| 60→15 Hz sum-aggregation breaks replay | Boost sweep: 1.56× works for 2/3 demos, fails for the third; observed-EE path is configuration-dependent | Do not try to down-convert; generate natively at 15 Hz. |
| Scripted-controller phase counts are frame-based, not time-based | `scripted_pick_demo.py:144-205` ("Phases (total ~1660 @ 60 Hz ≈ 28 s)") | Hardcoded 60 Hz. Must be reparameterized in seconds. |
| Cube randomization is ±10 cm × ±13 cm, no yaw | `yaskawa_pick_cube_cfg.py:244-261` | Too narrow. Policy memorizes a small patch. |
| Loss plateaus around 0.17 at epoch 10 on 60 Hz; SR peaks there | `reports/overnight_run_2026-04-24.md` eval table | Overfitting to the narrow distribution; extra epochs don't help. |
| 15 Hz loss descended further (0.123) with SR still 0/10 | `reports/15hz_investigation_2026-04-24.md` | Low loss alone means nothing. Eval has to be the signal, not the training curve. |

---

## 3. Gripper control — how it actually flows today, and the three holes

Traced end to end (scripted demo → HDF5 → LeRobot → training → policy → env):

1. **Scripted demo** (`scripts/validate/scripted_pick_demo.py:166-200`) writes `gripper ∈ {+1.0 OPEN, -1.0 CLOSE}` as a held binary latch per phase. Phase E (close) is 180 steps at 60 Hz ≈ 3 s.
2. **Dataset conversion** (`scripts/data/isaaclab_to_lerobot.py:70-160`) stores the action vector as 7D: `[ee_dx, ee_dy, ee_dz, ee_drx, ee_dry, ee_drz, gripper]`. When downsampling, pose dims are *summed* across the stride window (delta accumulation), but gripper is taken as the **last-frame value** of the window. No normalization specific to gripper.
3. **Training** (`scripts/orchestrate/train_only.sh`, `lerobot` stack) uses a flat MSE over the 7D action vector. There is **no per-dim weighting**; the gripper dim contributes 1/7 of the loss magnitude despite being the semantically load-bearing bit for grasp success.
4. **Closed-loop inference** (`scripts/train/run_vla_closed_loop.py:243-250`) calls `postprocess(action)` to denormalize, then hands the raw 7D tensor to `env.step`. The gripper dim is not thresholded.
5. **Env** (`envs/yaskawa_pick_cube_cfg.py:193-202`) applies `BinaryJointPositionActionCfg`: `action[6] < 0 → close (0.5 rad)`, `action[6] ≥ 0 → open (0.0 rad)`. Binary, at the zero boundary.

### The three holes

**Hole A — zero-boundary sensitivity at inference.** The policy outputs continuous values. When it's uncertain in a pose similar to "I should be closing soon," a predicted `+0.05` keeps the gripper wide open, and a predicted `-0.05` closes it on empty air. We are letting rounding noise decide the grasp.
*Fix:* simple sign-threshold on `action[6]` at inference (`sign()` at 0). This matches the env's own threshold and is the approach taken by Octo, ACT, Diffusion Policy, π0, and most continuous-output VLAs. RT-1 / RT-2 / OpenVLA avoid the problem entirely by discretizing the gripper into action tokens — a cleaner architectural solution but out of scope for a SmolVLA fine-tune.
*Fallback only if chatter is observed in rollouts:* add a deadband or hysteretic threshold (e.g. require `> +0.3` to reopen once closed). This is a debouncing patch on top of the sign-threshold, not a first-line design choice. Don't add it prophylactically; add it only if we see frame-to-frame toggling.

**Hole B — the close transition is statistically invisible.** In a 1600-frame 60 Hz demo the gripper value is +1.0 for ~420 frames then -1.0 for the remaining ~1200, with the transition localized at one step. Under flat L2, correctly predicting the transition is worth roughly the same as correctly predicting one pose-delta step. Regressing the gripper to +0.9 everywhere is a near-optimal lazy solution.
*Fix (root-cause, preferred):* apply a per-dim loss weight with `gripper_weight = 5.0–10.0` on action[6]. Alternatively add a BCE-on-sign auxiliary loss against the gripper dim. We will start with weight = 8.0 (strong but not dominant). This is the load-bearing fix — once the model's gripper output is crisp (values clustered near ±1 instead of smeared around 0), the inference-time threshold becomes unambiguous and Hole A mostly goes away on its own.

**Hole C — the close phase is brief and undifferentiated.** Phase E is 180 frames at 60 Hz but the EE pose is nearly identical for the ~40 frames surrounding it (D + E + first part of F share `ee_pos`). The policy has to infer "close now" from a near-identical visual/state context across dozens of steps, which is a hard credit-assignment problem.
*Fix:* lengthen phase D → 60 steps and phase E → 240 steps at 60 Hz equivalent, AND inject a 5–10 cm pre-grasp "commit" movement (slow forward-push) that gives the model a distinguishable pose trajectory as the close command fires. See section 5.2.

---

## 4. LoRA expansion

Current config (`scripts/orchestrate/train_only.sh:49-56, 218-225`):

```
r=32, alpha=32 (scale=1.0), dropout=0.05
targets = (lm_expert.(q|k|v|o|gate|up|down)_proj
         | state_proj | action_in_proj | action_out_proj
         | action_time_mlp_in | action_time_mlp_out)
modules_to_save = <none>
vision encoder = frozen
LM head = frozen
```

Proposed config for the v2 run:

```
r=64, alpha=64 (scale=1.0), dropout=0.05
targets = (lm_expert.(q|k|v|o|gate|up|down)_proj
         | state_proj | action_in_proj | action_out_proj
         | action_time_mlp_in | action_time_mlp_out
         | vision_tower.*\.(q|k|v|o)_proj)
modules_to_save = ["action_out_proj", "action_time_mlp_out"]
LM head = still frozen (no evidence it helps; one lever at a time)
```

Rationale:

- **r=64:** moderate bump. r=32 with ~1M trainable params was not hitting capacity on the current narrow distribution, but the upcoming data has ~2–3× as many demos with 3× broader XY coverage and yaw variation; we want slack.
- **`modules_to_save` for the action head output layers:** removes the LoRA bottleneck from the final two projections. These are the layers closest to the 7D action, where a capacity limit most directly shows up as jittery outputs. Makes them fully trainable instead of low-rank deltas over a frozen base.
- **Add vision-tower attention projections:** the ablations say the policy currently ignores vision. Giving the visual stream a small LoRA surface lets the model actually specialize visual features to the cube/gripper/target-mat cues. If this harms convergence we drop it on the next iteration — but we have evidence that the current "frozen vision" approach is *not* learning visual grasp alignment, so we have to change that knob.
- **Dropout stays 0.05:** we've seen no overfitting-from-adapter-capacity symptoms; overfitting is from data, not adapter params.
- **LM head frozen:** contrarian choice to the second research report — the task string is a single fixed prompt, so task-specific LM-head tuning has nothing to learn from. Adding it would be a pure capacity sink. Revisit only if we multiply tasks.

---

## 5. Execution plan (ordered, with gates)

Each phase writes its own dated report under `reports/` per the new CLAUDE.md discipline. A gate (G#) is a measurable check — fail the gate and you do not proceed, you diagnose.

### Phase 0 — Instrumentation & guardrails (half day)

Before generating a single new demo, make sure we can tell truth from noise.

- **0.1** Add fixed-seed eval harness: a held-out set of 30 cube XY positions (spread across the target distribution we're about to train on) with a frozen random seed, used for every checkpoint eval. No more comparing runs on different seeds.
- **0.2** Add a gripper-aware metric to `run_vla_closed_loop.py`: per-episode log "first close step", "lift height at first close", "cube z at 2 s after first close". Lets us distinguish "never closed" from "closed at wrong time" from "closed but couldn't hold."
- **0.3** Add a simple sign-threshold on the gripper dim at inference (Hole A fix) behind a `--gripper_threshold` flag (default 0.0, i.e. `sign()`). Keep a `--gripper_deadband` flag available but default OFF — only enable if phase-3 rollouts show frame-to-frame chatter.
- **0.4** Add per-dim action loss weights in the lerobot training path behind `--action_loss_weights` (keep defaults to 1.0 for backward compat; we'll set gripper=8.0 for the real run). If lerobot upstream doesn't expose this, patch it and log the patch.
- **0.5** Verify wrist camera mount. Look at a fresh `save_camera_samples.py` output; if still black, either fix the mount offset or drop the wrist cam from the obs space for v2. We will not retrain with an input known to be useless.

**Gate G0:** a single epoch-0 (untrained adapter) closed-loop eval against the 30-position set runs clean end-to-end with the new instrumentation, writes per-episode JSONL with gripper metrics, and produces a GIF. Numbers will be garbage; format must be right.

### Phase 1 — Scripted demo rework for 15 Hz + diversity (1–2 days)

**1.1 Reparameterize the scripted controller in seconds, not frames.**
`scripts/validate/scripted_pick_demo.py` — every `"steps": N` entry becomes `"duration_s": N / 60.0`, and the runner multiplies by `sim_hz` (15 in this case) at execution time. This fixes the bug that broke native 15 Hz generation. See `reports/15hz_investigation_2026-04-24.md` for the observed symptom.

**1.2 Lengthen the grasp-commit window.** Gripper-control Hole C fix:
- Phase D (settle at grasp): 40 → 60 steps at 60 Hz equivalent.
- Phase E (close): 180 → 240 steps.
- Insert phase "D2" between D and E: a 40-step micro-descent of ~1 cm with gripper still open, so the close command fires during a visibly distinct pose trajectory.

**1.3 Widen cube randomization.**
Edit `envs/yaskawa_pick_cube_cfg.py:248-252`:
```python
"pose_range": {
    "x": (-0.15, 0.15),   # sampled cube X ∈ [0.40, 0.70]   (was ±0.10)
    "y": (-0.22, 0.22),   # sampled cube Y ∈ [-0.22, 0.22] (was ±0.13)
    "yaw": (-0.5, 0.5),   # sampled yaw ∈ ±28.6°            (was disabled)
},
```

Two prerequisites before enabling this:
- **(a) Scripted controller must read cube yaw.** Current `script_trajectory_waypoints` takes only `cube_pos_w`, not `cube_rot_w`. Extend it to take the cube yaw and use `_quat_from_downward_xy_yaw(yaw)` when building the EE target quat (it already has the machinery at line 43-66, just isn't being fed).
- **(b) Validate every corner of the new box is reachable.** Run the scripted controller on 30 positions sampled from the new box and confirm all succeed in open-loop. Any that fail narrow the allowed box before dataset generation starts.

**1.4 Verify camera FOV coverage.** Use `scripts/validate/save_camera_samples.py` on cube positions at the four corners of the new box. Any corner where the cube is clipped out of the third-person camera shrinks the box. Document which corners survive in the phase report.

**1.5 Generate the new scripted-seed HDF5.** Target ~120 diverse scripted successes at native 15 Hz (sim runs at 60 Hz physically, but the recorder emits every 4th frame). Log the per-position success distribution.

**Gate G1:** 30-position open-loop scripted success on the widened box ≥ 28/30. Any failures must be positions narrower than the final box, not spread randomly.

### Phase 2 — Scripted-demo scale-up at 15 Hz (1 day) — REVISED 2026-04-27

**Original Phase 2 was Mimic generation; that path has been retired (see
`reports/2026-04-27_scripted_only_data_pipeline.md`). Going forward,
the training dataset comes from running `scripted_pick_demo.py` at
scale.**

**2.1 Generate the new scripted dataset directly.**
Run `scripts/validate/scripted_pick_demo.py` to produce ~400-750
successful demos (the 30-demo G1 run took ~30 min; ~6 hr × num_demos / 30
gives wall-clock). The widened cube box, yaw randomization, and
per-episode color injection (vla_kitting-y5b) all already fire on every
demo, so the variety Mimic was producing comes "for free" from
randomized resets — what we lose is the cross-demo splicing, gained back
by raw demo count.
- Output: `datasets/teleop/cube_scripted_<date>.hdf5`
- Each demo has `attrs[task]` with the per-episode color word and
  `obs/cube_color_idx` per frame.

**2.2 Convert to LeRobot.**
`scripts/data/isaaclab_to_lerobot.py --stride 1 --drop_cube_pos`. The
converter already prefers `attrs[task]` for per-episode prompts, so the
LeRobot dataset gets per-episode color-aware tasks automatically.
- **Drop `cube_pos` from the exported observation set.** Forces visual grounding.
- Keep the third-person cam. Drop `wrist_cam` only if phase 0.5 concluded the mount is unfixable.

**2.3 Dataset sanity:** visualize 10 random demos with `render_demo.py`,
confirm cube positions span the full widened box, confirm gripper
trajectories show clean open→close→open edges, confirm per-episode
prompt distribution is roughly uniform across the 5 cube colors.

**Gate G2:** dataset has ≥ 350 demos, cube-start positions cover ≥ 85% of the widened box area when plotted, gripper transitions are clean, per-color prompt count is each within ±30% of the uniform expectation.

### Phase 3 — Training v2 (1 day compute, overnight)

**3.1 Configuration:**
```
dataset: datasets/lerobot/cube_pick_v2_15hz_<date>
decimation: 1   (already 15 Hz in the dataset)
batch: 8        (2× previous; more demos support it)
lr: 1e-4        (same as last run)
epochs: 40      (early stop on extended-eval plateau)
peft:
  r: 64
  alpha: 64
  dropout: 0.05
  target_modules: see section 4
  modules_to_save: ["action_out_proj", "action_time_mlp_out"]
action_loss_weights: [1,1,1,1,1,1,8]   (gripper = 8)
gripper_threshold (eval only): 0.0     (sign-threshold; deadband off by default)
```

**3.2 Eval every 3 epochs** on the frozen 30-position set from phase 0.1. Write per-episode JSONL.

**3.3 Kill switch:** if SR at epoch 15 is worse than 5/30, stop. That's the go/no-go. Do not grind to epoch 40 hoping something clicks — it didn't before, it won't now.

**Gate G3:** peak eval SR ≥ 15/30 (50%) on the held-out set by epoch 30. This is the "we have a working policy" bar. Hit this and we harden; miss this and we go to phase 4.

### Phase 4 — Fallback branches (only if G3 missed)

In priority order, each is a single variable change:

- **4a. Drop yaw variation.** Maybe ±0.5 rad was too aggressive with scripted-only demos. Keep wider XY, set yaw = 0 again, retrain.
- **4b. Add the `cube_pos` observation back but only as Bernoulli-dropout input.** Forces vision usage most of the time while still giving the head a rescue signal.
- **4c. 60 Hz branch.** Same data generation approach (wider box, diversity, yaw) but at 60 Hz and with the broader-LoRA config. Expensive (4× data) but is the documented fallback if 15 Hz at SmolVLA just doesn't converge.
- **4d. Unfreeze the LM-expert FFN bias terms** (cheap, ~0.1% params). Only after 4a-c because evidence for it is weakest.

Each fallback gets its own dated report with its own G3-equivalent gate.

### Phase 5 — Hardening (only after G3 passes)

- **5.1** Expand eval to 100 positions; publish SR heatmap over the box.
- **5.2** Add distractors (second cube, clutter) to the scene and re-eval for generalization.
- **5.3** Schedule the first human teleop collection — scripted data is a floor, teleop diversity is the ceiling.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Widened box has corners the diff-IK can't reach without singularity swing | Medium | Phase 1.3(b) validates each corner open-loop before dataset generation. |
| Dropping `cube_pos` catastrophically regresses learning | Medium | Gate G3 is the check; if missed, fallback 4b reintroduces it under dropout. |
| Gripper loss weight = 8 destabilizes other dims | Low | Start at 8, sweep {4, 8, 16} as a small phase-3.5 ablation if G3 is marginal. |
| Gripper output remains smeared near 0 even with loss weight (sign-threshold still ambiguous) | Low | Add deadband or hysteretic threshold at inference as a debouncing patch. Only if observed — don't pre-solve. |
| Native 15 Hz scripted controller has behaviors the 60 Hz version doesn't | Medium | Phase 1 re-validates open-loop success before any training. |
| `modules_to_save` clashes with resume-config fixups | Low | `scripts/orchestrate/fix_adapter_configs.py` already normalizes adapter JSON; extend it if needed. Log the patch. |
| Vision-tower LoRA blows up VRAM | Low | r=64 on a small ViT is ~2–3M extra params; negligible. Monitor peak memory during phase 3. |

---

## 7. Estimated cost

| Phase | Wall-clock | GPU-hours |
|---|---|---|
| 0 — instrumentation | ~4 hours | ~0.5 (eval runs only) |
| 1 — scripted demos at 15 Hz, widened box | ~8 hours | ~2 (sim only, no training) |
| 2 — scripted-demo scale-up + convert | ~10-15 hours wall (~400-750 demos × 30 s/demo serial) | ~6 (scripted-demo gen is sim-bound, parallelizable across GPUs if available) |
| 3 — training v2 | overnight | ~12 |
| 4 — fallbacks (if needed) | 1-3 days | 8-30 |
| **Total to G3 decision** | **~2.5 days** | **~18 GPU-hours** |

---

## 8. What NOT to do

- Do not retrain on the existing `cube_pick_v1_20260423_021729` dataset expecting a different result. We have three runs saying it won't work.
- Do not tune LoRA rank and data diversity in the same run. One variable at a time past G3; bundle only during the initial restart.
- Do not ship code changes to `envs/` or `scripts/train/` mid-run without noting in the active report. Silent environment edits have bitten us before (per `reports/known_issues.md`).
- Do not convert 60 Hz → 15 Hz post-hoc ever again. The boost sweep closed that door (`reports/15hz_investigation_2026-04-24.md`).
- Do not compare runs on floating eval seeds. The frozen 30-position set from phase 0.1 is the only valid eval.

---

## 9. Open questions to resolve in phase 0

1. Is the wrist camera actually fixable, or do we drop it from v2's obs space?
2. Does lerobot's training loop expose per-dim action loss weights natively, or do we need a local patch (add to `project_lerobot_peft_resume_patch.md` if so)?
3. What's the exact reachable footprint of the diff-IK at the new widened box corners? Measured in phase 1.3(b).

---

## 10. Success definition

**Minimum ship:** closed-loop success ≥ 50% (15/30) on the frozen held-out positions, at any epoch of the v2 run. This is what "we have a policy" means for this repo, for the first time.

**Stretch:** ≥ 80% and a visible heatmap with no dead zones in the widened box.

Everything else — distractor generalization, yaw robustness, teleop data, multi-task — lives past that bar.
