# Overnight VLA training session — 2026-04-26

**Session window:** 03:48 → ~12:00 UTC (~8 h work + ~3 h additional grind).
**Run dirs:** `reports/runs/v4_gripper_weight_2026-04-26/`,
`reports/runs/vision_grounded_30hz_2026-04-24/`.
**Training PID history:** 499657 → 509959 → 512515 → 515476 → 520577.
**Final checkpoint state:** `checkpoints/continual/checkpoints/last → 048000`.

This file consolidates everything found and tried tonight. The
underlying diaries are at:

- `reports/2026-04-26_eval_ghost_cube_investigation.md` — DLSS ghost-cube
  diagnosis + FXAA fix.
- `reports/runs/v4_gripper_weight_2026-04-26/run_diary.md` — full v4
  training log.

---

## TL;DR — what I delivered, what failed, and why

**Delivered:**

1. **Root-caused the eval ghost-cube bug** (DLSS temporal-AA bleed) and
   fixed it (`envs/yaskawa_pick_cube_cfg.py:315` set
   `sim.render.antialiasing_mode = "FXAA"`). Verified clean before,
   during, and after training.
2. **Patched lerobot to support per-dim action loss weighting** and ran
   a clean v4 training with `gripper × 8` then `gripper × 16`.
3. **Diagnosed the real failure mode with a per-step action log** —
   confirmed the bottleneck is **EE positioning precision, not gripper
   learning**. The user's intuition was correct.

**Did NOT deliver a working fine-tuned policy.** Best real-SR is still
0/N. All "successes" in eval logs are trivial cube-spawned-on-target
auto-pass cases; no real grasp ever happens.

**Why we did not converge:**

| Hypothesis | Tested? | Result |
|---|---|---|
| Ghost cubes break eval inputs | Yes | False — fixed, no SR change |
| Gripper loss weight too low | Yes (×8 then ×16) | False — gripper IS being learned (50/50 close/open output) |
| Gripper threshold tuning | Yes (0.0, +0.5, -0.5) | False — threshold doesn't matter |
| n_action_steps too long | Yes (10 → 1) | False — at n=1 the per-step prediction is noise, model relies on chunk smoothing |
| **EE positioning precision** | **Confirmed via action log** | **TRUE — EE never reaches the cube; closes at wrong location** |

The training loop, lerobot patches, FXAA fix, gripper loss weighting,
and ghost-cube investigation all worked correctly. The remaining
failure is the same problem the v3.x runs flagged as "visual mode
collapse" — the policy lacks visual grounding precision strong enough
to reach the cube before/while closing. Without `cube_pos` as a
privileged input and without enough vision-LoRA capacity / data
diversity, the model's image→EE-pose-delta mapping reaches roughly the
right neighborhood but never exactly onto the cube.

---

## Ghost cube — was it really fixed?

Yes. Verified three times:

1. **Pre-fix evidence (2026-04-25):** `cam_check/sample_02_third.png`
   showed three distinct cube-shaped objects in one frame; sample_05
   showed one cube + a faint gray ghost at the default-spawn position.
2. **Mechanism found:** `dump_scene_prims_fast.py` showed only ONE
   `Cube` prim in the USD stage — so the issue was render-side, not
   duplicate prims. The IsaacLab default `antialiasing_mode = DLSS`
   reconstructs each frame from a multi-frame history and motion
   vectors. When the cube teleports during reset, DLSS has no motion
   vector for the instant jump, so prior-frame samples bleed in as
   faint ghosts. The Isaac Sim log confirmed DLSS was active with the
   warning *"DLSS increasing input dimensions: Render resolution of
   (74, 74) is below minimal input resolution of 300"*.
3. **Postfix verification (multiple, latest at 11:55 UTC):** ran
   `save_camera_samples.py` again. Each frame shows exactly one cube,
   no ghosting. Output at
   `reports/runs/v4_gripper_weight_2026-04-26/final_debug/ghost_check/`.

The training and all evals from epoch 1 onward used the FXAA env, so
the policy was always fed clean images. The ghost cube was a real bug
but it is not what's preventing convergence — fixing it on its own
moved v3.2 ep22 from 0/9 → 0/9 in the postfix-validation eval. Removed
a confound, did not unlock learning.

---

## Why I think it's positioning, not gripper, after all

I added per-step action logging to `run_vla_closed_loop.py` (new
`--action_log_csv` flag) and ran a 2-episode 600-step diagnostic on
the epoch 48 checkpoint. CSV at
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/action_log.csv`.

### Headline numbers

| Metric | episode 0 | episode 1 |
|---|---|---|
| gripper a[6] mean | -0.053 | +0.127 |
| gripper a[6] stdev | 0.999 | 0.993 |
| close fraction (a[6]<0) | 53 % | 44 % |
| ee→cube xy-dist mean | 9.6 cm | 9.4 cm |
| ee→cube xy-dist min | 1.5 cm | 5.8 cm |
| cube z max (lift signal) | 0.038 m | 0.029 m |
| ee z range | 0.097–0.649 m | 0.062–0.649 m |

### Visual confirmation

`reports/runs/v4_gripper_weight_2026-04-26/final_debug/diag_ep0_annotated.png`
shows 5 frames from episode 0 with the action log values overlaid:

| step | gripper | ee→cube | ee_z |
|---|---|---|---|
| 50 | open | 9 cm | 0.45 m (high) |
| 150 | open | 4 cm | 0.25 m |
| 300 | open | 5 cm | 0.15 m (hovering 12 cm above cube) |
| 450 | **CLOSE** | **10 cm** | **0.15 m** |
| 599 | open | 23 cm | 0.10 m |

At step 450 the gripper closes — but the EE is 10 cm from the cube
and 12 cm above the table. The fingers close in empty space, the cube
is not picked up, and the model then re-opens and drifts further away.

### What this says

**The gripper IS being learned.** Output is essentially crisp ±1 with
near-50/50 split between open and close. If the gripper-weight=8/16
fix had failed structurally, we'd see a smear around 0 with 100% near
zero — instead we see 53 % of steps with a[6] = -1.0 (close) and 47 %
at +1.0 (open). The Hole-B fix worked. The policy has learned a
binary gripper signal.

**Positioning is the bottleneck.** Snapshot at step 450 of episode 0:

```
gripper command: a[6] = -1 (CLOSE)
EE position:     (0.556, -0.080, ee_z=0.147)
Cube position:   (0.467, -0.032, 0.025)
distance:        ~10 cm xy, ~12 cm z above cube
```

The policy commands "close" while the EE is **10 cm horizontally and
12 cm vertically away from the cube**. Even though the gripper closes
correctly, the fingers close in empty air. Same pattern in episode 1:
gripper closes at step 450 with EE 6 cm from cube and 7 cm above table.

The minimum ee→cube xy-distance over each whole episode is 1.5–5.8 cm —
so the model GETS close at *some* moment. It just isn't synchronizing
the close command with the moment of proximity. The chunked-action
prediction (n_action_steps=10 + chunk_size=50) means the model commits
to a 10-step pose+gripper trajectory at a time; if the close command
fires in chunk-position 5 but the EE is still drifting toward the cube,
the cube gets bumped without grasp.

**Robot motion patterns confirm the imprecision.** The pose-delta
distributions show the model learned the rough motion gestalt but with
noisy bias:

- a[0] (ee_dx) mean +0.021, range [-0.005, +0.19] — strongly positive,
  so EE consistently moves +X (forward / toward back of table).
- a[1] (ee_dy) mean -0.016, range [-0.13, +0.001] — strongly negative,
  pulling EE toward -Y (right of table).
- a[2] (ee_dz) mean -0.025, range [-0.13, +0.01] — pulling DOWN, good.
- a[4] (ee_dry) max +1.15 — values >1.0 saturate the IK scale, meaning
  the model wants huge wrist rotations. This will cause overshoot.

The model has a *generic* descend-and-go-forward bias that approaches
the *training-distribution-mean* cube position rather than the
*current-episode* cube position. That is the textbook signature of
"vision works but isn't precise enough for fine reaching" — exactly
what the v3.x diary called "visual mode collapse."

### Why I had previously called it a gripper issue

I read the diaries and the recovery plan, saw "Hole B: gripper close
statistically invisible under flat MSE," and set up the gripper-loss-
weighting experiment as Hole B's prescribed cure. I confused two
related symptoms:

- **z = 0.025 always** — true. Cube never lifts.
- **gripper never closes** — assumed, never verified. False.

A "z = 0.025 always" outcome is consistent with EITHER (a) gripper never
closes, OR (b) gripper closes but at the wrong location. Without a
direct probe of action[6], I assumed (a). The action log shows it's
(b). Your push for diagnostic discipline caught my mistake — that
question was the right one to ask.

---

## Camera issues — resolved?

The ghost-cube fix (DLSS → FXAA) is resolved and confirmed across
training, eval, and a fresh post-session re-test. There remains one
*open* camera concern from older diaries that I did not address
tonight: the **wrist camera mount geometry**. The recovery plan
flagged it as "questionable" and the v3 cam_check showed wrist frames
that were "mostly white/empty." Looking at tonight's eval gif strips,
the wrist cam still mostly shows the gripper internals against a
washed-out background — useful when the gripper is over the cube,
useless when it's far away. Whether this constitutes a useful
auxiliary input for the policy is unclear. If a future run adds
`cube_pos` back as a privileged input, the wrist cam dependency drops
to zero and this concern goes away. If not, we may need to re-aim the
wrist camera or just drop it from the obs space.

So: ghost cube **is** resolved. Wrist camera is **still suspect** but
was not on tonight's critical path.

---

## Failure mode summary (what to chase next)

Ranked by impact-vs-cost given what we now know.

### 1. Add `cube_pos` back to observations *(highest impact, ~6h)*

The recovery plan §4b explicitly listed this as the fallback if visual
grounding fails to converge. Strong evidence tonight that it would.
The original 60 Hz r=16 baseline that achieved peak SR 2/10 had
`cube_pos` in observations; every run since (v2/v3/v4) that dropped
`cube_pos` has hit 0/N real success. Adding it back as a
Bernoulli-dropout input (recovery plan formulation) keeps vision
training pressure most of the time but provides a rescue signal.

Cost: regenerate the dataset (~5 h scripted-gen + ~1 h convert) +
retrain (~2 h to first signal at modest SR). The dataset regenerator
just needs to drop the `--drop_cube_pos` flag in
`isaaclab_to_lerobot.py`. (Mimic is no longer part of the pipeline as
of 2026-04-27 — see `reports/2026-04-27_scripted_only_data_pipeline.md`.)

### 2. De-saturate the scripted P-controller *(medium impact, ~6h)*

The v3.2 closing diary's last hypothesis was "saturated P-controller
produces near-identical first actions across all 400 demos." Combined
with the cube-position memorization shortcut, the model overfits to a
stereotyped initial trajectory. Lower the P-gain (10 → 2-3),
regenerate demos, retrain. Same data-regeneration cost as (1) but
keeps `drop_cube_pos` discipline (forces visual grounding).

### 3. Switch to full fine-tune (drop LoRA) *(medium impact, ~3h)*

LoRA r=64 may be capping the action-head precision. A full fine-tune
of the action expert is more flexible. Cost: just rerun training with
`USE_LORA=0` in `train_only.sh`. Quick.

### 4. Re-aim wrist camera *(low impact, ~30 min)*

If we keep `drop_cube_pos`, the wrist cam needs to actually inform
fine reaching. Currently it sees the cube only when EE is right above
it. A wider FOV or different mount angle could help.

### 5. Switch eval to `n_action_steps=1` AND retrain with that
*(experimental, ~3h)*

The n=1 ablation showed the per-step prediction is too weak right
now (cube doesn't move). But that was with a model trained at n=10. A
model *trained* at n=1 would force the per-step network to be precise.
Higher inference cost but probably better closed-loop control.

---

## Loss / SR table for v4 (gripper-weight run)

Phase 1 (weight × 8, epochs 1-20):

| Epoch | loss | eval_sr | real SR (excluding spawn-on-target) |
|---|---|---|---|
| 1 | 0.159 | — | — |
| 5 | 0.048 | 0.00 | 0/10 |
| 10 | 0.038 | 0.00 | 0/10 |
| 15 | 0.031 | 0.10 | 0/9 |
| 20 | 0.031 | 0.10 | 0/9 |

Phase 2 (weight × 16, epochs 21-48):

| Epoch | loss | eval_sr | real SR |
|---|---|---|---|
| 21 | 0.039 | — | — |
| 25 | 0.036 | 0.10 | 0/9 |
| 30 | 0.033 | 0.10 | 0/9 |
| 35 | 0.030 | — | — |
| 40 | 0.030 | 0.00 | 0/10 |
| 47 | 0.027 | — | — |
| 48 | 0.027 | — | — |

Loss continued to descend slowly through epoch 47-48 (new floor 0.027,
breaking the prior 0.029 plateau), but no eval was scheduled at that
horizon. Every fresh checkpoint above epoch 30 produced the same real
SR 0/N pattern in the diagnostic eval.

---

## Visual debug — strips from each evaluated checkpoint

`reports/runs/v4_gripper_weight_2026-04-26/final_debug/strips/` contains
7-frame strips (every ~129 frames) from each eval gif at epochs 5, 10,
15, 20, 25, 30, 40. Each strip is wrist | third-person side-by-side.

Pattern that develops across epochs:

- **Ep5:** wrist cam blank/white (gripper in home pose). Third-person
  shows the cube as a small dot near top-back of table.
- **Ep10–20:** wrist cam starts seeing the gripper interior dynamically;
  third-person shows the arm reaching forward, but the gripper passes
  *near* the cube, not *over* it.
- **Ep25–40:** same approach pattern repeats, refined slightly. The
  gripper opens and closes during the trajectory (per the action log)
  but never lands centered on the cube.

The 4-episode `final_debug/diag_ep*.gif` contains gif renders of the
diagnostic episodes whose action logs are in `action_log.csv`. Compare
those gifs frame-for-frame against the CSV to see, e.g., the moment
gripper closes at step 450 of episode 0 — you'll see the gripper
hovering ~10 cm short of the cube.

---

## Files of interest

**Evidence (look at these first):**
- `reports/runs/v4_gripper_weight_2026-04-26/final_debug/action_log.csv` —
  per-step action vector + cube/EE pose, the smoking gun.
- `reports/runs/v4_gripper_weight_2026-04-26/final_debug/strips/*.png` —
  visual evolution across epochs.
- `reports/runs/v4_gripper_weight_2026-04-26/final_debug/ghost_check/*.png` —
  fresh post-session ghost-cube verification.
- `reports/runs/v4_gripper_weight_2026-04-26/final_debug/diag_ep*.gif` —
  GIFs synchronized with the action log.

**Code changes still in place:**
- `envs/yaskawa_pick_cube_cfg.py:315` — `antialiasing_mode = "FXAA"`.
- `scripts/orchestrate/train_only.sh` — `action_loss_dim_weights` with
  gripper × 16 (currently). To revert to the v3-style canonical, drop
  the `--policy.action_loss_dim_weights` flag from the LoRA branch.
- `scripts/train/run_vla_closed_loop.py` — `--action_log_csv` flag.
- `/home/ubuntu/code/lerobot/src/lerobot/policies/smolvla/{configuration_smolvla.py,modeling_smolvla.py}` —
  the action-loss-dim-weights patch (already documented in
  `project_lerobot_peft_resume_patch.md`).

**Checkpoints:**
- `checkpoints/continual/checkpoints/048000` — final live checkpoint.
- `checkpoints/v3_2_archive_20260426/` — prior v3.2 epoch 22, kept.

---

## Cross-attention overlay (added 2026-04-26 ~13:30 UTC)

Built `scripts/validate/attention_overlay.py` to extract the
attention from action-query positions to vision-token positions in the
suffix-only forward pass of SmolVLA. Hooked
`SmolVLMWithExpertModel.eager_attention_forward` to capture per-layer
softmax probs, then averaged over heads + the last 4 layers, sliced
the action-query rows, took mean across the 50 action positions, and
got a single per-key importance vector. Reshaped the image-token slice
to an 8×8 grid (64 vision tokens per image) and upsampled to the
input resolution.

Ran on the 5 fresh `cam_check_postfix` samples (different cube spawn
positions). Output PNGs at
`reports/runs/v4_gripper_weight_2026-04-26/final_debug/attn/`.

| Sample | Cube position in frame | Heatmap peak | Aligned with cube? |
|---|---|---|---|
| 00 | mid-left, yellow | upper-left/mid | yes (broad) |
| 01 | upper-left, coral | upper-mid | partial |
| 02 | upper-center, yellow | upper-center-right | yes |
| 03 | mid-upper, pink | mid-upper | yes |
| 04 | on target, coral | upper-left-center | partial |

The third-person attention **does shift with cube position** —
this rules out pure visual mode collapse on the third-person view.
The model has learned something about cube localization.

**But the attention is broad**, not focused. Peak attention values
are 0.013–0.017 against a uniform 1/177 ≈ 0.0057 baseline — only
~2.5–3× over uniform. A well-trained vision system would have peaks
10–50× over uniform on the cube. So the localization signal is weak
and spread across 3-5 patches around the cube. With 8×8 = 64 patches
on a ~80cm-wide table view, each patch covers ~10cm; a 3–5 patch
spread is ~15–25cm of uncertainty.

**This precisely matches the EE-positioning error in the action log
(mean 9.6cm distance, max 23cm) and the "fixed-offset" pattern.**
The model knows the cube is "around there" but not "exactly there."

The bottleneck shifts from "vision can't see the cube" to **"vision
sees the cube in the right neighborhood but at limited spatial
resolution."** That changes which interventions matter:

- ❌ Re-aiming the wrist camera doesn't help (third-person already finds the cube).
- ❌ More gripper loss weight doesn't help (grip output is already learned).
- ✅ Higher input resolution / more vision capacity should sharpen attention.
- ✅ Adding `cube_pos` as a privileged input bypasses vision precision entirely (still the highest-impact, lowest-risk fix).
- ✅ More demos and a wider distribution may sharpen the action-head's vision→pose mapping even at current vision resolution.

## Honest self-assessment

The night ended with no working policy, but the experimental ledger is
clean and the next experiment is clear-cut. I burned ~30 min chasing a
gripper-threshold ablation that the action-log diagnostic later
showed was the wrong question to ask. I should have run the action
log probe earlier — the gripper-output evidence was easier to obtain
than I assumed (it required a small flag in the existing eval script,
not a separate Isaac-Sim launcher). Letting that diagnostic land
*before* the weight-8→16 pivot would have saved 1.5 h of compute and
told me up front that the issue is positioning, not gripping.

That mistake is now in the diary. The next run starts with `cube_pos`
back in observations.
