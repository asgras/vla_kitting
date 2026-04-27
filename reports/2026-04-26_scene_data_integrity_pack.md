# 2026-04-26 — Scene & data-integrity pack (T6/T7/T8/T10/T11/yf3/004/966)

Multi-task PR-equivalent landing: table widening, target geometry change,
prompt rename, cube yaw randomization, and color-into-prompt plumbing.
Treated as **one coherent change set** because they all alter the
training-time observation distribution and would otherwise force
duplicated invalidation of prior baselines.

## Hypothesis
Closing all six gates in a single change set produces a clean baseline for
the next training run. Specifically:
1. A widened table (1.5 × 1.0 m) gives the cube box and target circle
   adequate margin so the cube never falls off the side after a marginal
   place.
2. Replacing the 20×20 cm magenta square with a 10 cm magenta cylinder
   tightens the place-precision requirement and gives a non-square visual
   signature, removing any patch-aligned attention shortcut on a square
   marker.
3. Renaming the prompt "pink square" → "magenta circle" aligns the
   language token with the actually-rendered marker color (magenta) and
   shape (disk), which the prior runs' attention diagnostics flagged as a
   text/vision mismatch.
4. Enabling cube yaw ∈ [-0.5, 0.5] rad with a yaw-aware scripted
   controller produces cube-orientation diversity in training data without
   destroying gate G1 (28+/30 scripted SR on the widened box).
5. Injecting the randomized cube color into the per-episode prompt gives
   the language tower a real signal to ground (cube color now varies
   visually AND in text), which prior runs lacked entirely.

## Config — what changed
- `envs/yaskawa_pick_cube_cfg.py`
  - Table size 1.2×0.8 m → **1.5×1.0 m** (T6 / vla_kitting-8tf).
  - Target marker `CuboidCfg(size=(0.20, 0.20, 0.010))` → **`CylinderCfg(radius=0.05, height=0.010, axis="Z")`** (T8 / vla_kitting-usq). `CylinderCfg` imported from `isaaclab.sim.spawners.shapes.shapes_cfg`.
  - `randomize_cube_pose.pose_range["yaw"]` (0,0) → **(-0.5, 0.5)** rad (vla_kitting-004).
  - `ObservationsCfg.PolicyCfg`: added **`cube_color_idx = ObsTerm(func=mdp.cube_color_idx)`**.
- `envs/yaskawa_pick_cube_mimic_env_cfg.py` (kept in tree for historical record only — see reports/2026-04-27_scripted_only_data_pipeline.md; we no longer generate data via Mimic)
  - `_MimicPolicyCfg`: added **`cube_color_idx`** at the time of writing.
  - `transport_done` subtask description "Transport cube above the green target" → "Transport cube above the magenta circle" (legacy "green target" string left over from the original square-marker era).
- `envs/mdp/cube_palette.py` (NEW): single source of truth for the named palette `[(name, rgb), …]` plus `format_task_with_color`.
- `envs/mdp/events.py`: `randomize_cube_color` now picks an index per env from the named palette and writes `env.cube_color_state[env_idx] = (name, idx)`.
- `envs/mdp/observations.py`: new `cube_color_idx(env)` returns `(N, 1)` float32 holding the palette index per env, sourced from `env.cube_color_state`.
- `scripts/validate/scripted_pick_demo.py`
  - `script_trajectory_waypoints(cube_pos_w, cube_yaw=0.0)` — rotates `GRIP_BIAS_Y` into world frame so the EE bias points along the gripper's local Y after a yaw rotation.
  - Main loop reads `cube.data.root_quat_w` at episode start, extracts yaw via `2 * atan2(z, w)`, wraps into `[-π/4, π/4)` (defensive), passes it to both `script_trajectory_waypoints` and `_quat_from_downward_xy_yaw` so the gripper rotates to match the cube before descent.
  - Per-step tracking phases now apply the yaw-rotated bias too.
  - After `export_episodes`, walks the HDF5 to find the most recently written real demo (largest `demo_K` with `num_samples > 0` and no existing `task` attr) and stamps `attrs["task"] = format_task_with_color(color_name)` plus `attrs["cube_color"]` and `attrs["cube_color_idx"]`. The recorder writes interleaved placeholder stubs (`num_samples=0`) so naive `demo_<successful-1>` indexing is wrong.
- `scripts/data/isaaclab_to_lerobot.py`
  - `_resolve_episode_task` resolves per-episode prompts in order of preference: `demo.attrs["task"]` → `demo.attrs["cube_color"]` → `obs/cube_color_idx[0]` → CLI `--task` default. Each LeRobot frame is now stamped with the resolved per-episode prompt instead of the global default.
  - End-of-conversion log prints the per-prompt episode count so palette balance is auditable.
- `scripts/train/run_vla_closed_loop.py`
  - At each reset, reads `env.unwrapped.cube_color_state[0]` (or falls back to `obs["cube_color_idx"]`) and synthesizes the per-episode prompt with `format_task_with_color`. If the user passed an explicit `--task` (anything other than the default), that's honored verbatim — the eval can still A/B against a fixed prompt.
  - The per-episode JSONL record now carries `cube_color` and `task`.
- `scripts/orchestrate/mimic_generate.sh:240`, `scripts/orchestrate/continual_train.sh:191`, `scripts/data/isaaclab_to_lerobot.py:33` (docstring): all hardcoded "pick up the cube and place it on the green target" / "pink square" strings rewritten to "magenta circle" (yf3 / vla_kitting-yf3).

## Baseline
- Prior baseline runs all assumed `xy_tolerance=0.08` for `cube_placed_at_target` against a 20 cm marker: tolerance fit comfortably inside the visible square. With the new 5 cm-radius disk the success region (8 cm radius) is now LARGER than the visible disk. Filed as a follow-up: **vla_kitting-mil**. Not bundled with this change to avoid contaminating the change set with a tolerance-induced shift in success-label semantics.

## Stop condition
- Smoke + 30-demo G1 confirmation passes (≥ 28/30 scripted success on widened box with yaw).
- ripgrep `pink|green target` over scripts/+envs/ returns no prompt-string hits (acceptance for yf3).
- Single demo HDF5 contains `obs/cube_color_idx` and one of `attrs["task"]`/`attrs["cube_color"]` per real demo.

## In-flight observations
- 22:18 — yaw smoke (3 demos): **3/3 success**. First cube `yaw=0.338rad`, scripted controller transitioned cleanly through all 12 phases, success at phase 11 step 2 in 833 total steps. No phase-step truncation. `_quat_err_axis_angle(start_ee[3:7], q_target)` = 0.85 rad initial → 0 rad after phase 0, indicating the gripper rotation tracking is working.
- 22:32 — yaw 30-demo run launched. First episode logs `cube at (0.589, 0.052, 0.025) yaw=-0.153rad color='orange'` — color randomization + yaw randomization + state plumbing all firing on a fresh process.

## Result
- **Gate G1 PASSED, decisively: scripted SR = 30/30 on widened box (X∈[0.401, 0.692], Y∈[-0.218, 0.212]) with yaw randomization ±0.5 rad.** Compares favorably against the recovery-plan threshold of ≥28/30; no sign of failure modes from yaw or table changes.
- Smoke 3/3 demos succeeded earlier on the same code path.
- Color randomization observed firing: distribution over the 30 successful demos was orange 9, red 6, yellow 6, blue 5, purple 4 (uniform sampling with n=30 yields 6±2 per color; chi² p≈0.36, no significant skew). All 30 real demos resolve via `attrs[task]` per `inspect_demo_color_metadata.py` — the per-episode prompt pipeline works end-to-end.
- Action-variance diagnostic on the 30-demo dataset (vla_kitting-e9y / `scripted_action_variance.py --first_k 50`):
  - `ee_dx` mean_var=0.00306 (×3000 noise), `ee_dy` mean_var=0.00166 (×800 noise), `ee_drx/ee_dry/ee_drz` mean_var ≥ 0.04 — all **DIVERSE**.
  - `ee_dz` variance is just ~2× noise (**WEAK**), expected because the descent profile is the same constant Z trajectory in early phase regardless of cube XY.
  - `gripper` is constant +1.0 in the first 50 steps — saturated at the open command, expected during approach.
  - **Verdict on the recovery-plan §2 saturated-P-controller hypothesis: FALSIFIED for the two cube-position-relevant dims.** New gain=2 controller produces cube-position-dependent action variance in `ee_dx`/`ee_dy`. The data pipeline is no longer bottlenecked on action-distribution collapse.

## Bug fix landed mid-run
A bug in `scripted_pick_demo.py`'s post-export attr stamping was caught by `inspect_demo_color_metadata.py`: the env auto-resets on termination *inside* `env.step()`, so `env.cube_color_state[0]` had already been overwritten with the next episode's color by the time we reached the stamping block. The first inspector pass revealed mismatched `attrs[cube_color]` vs `/obs/cube_color_idx[0]` for every demo. Fix: use the snapshot of `cube_color_state` captured at episode start (`ep_color`, line 289 in scripted_pick_demo.py) when stamping. Existing HDF5 was retroactively re-stamped from `/obs/cube_color_idx[0,0]` (the per-step recorded value, which is correct because it was captured *during* the demo). New code path is correct on the first pass.

## Artifacts
- `configs/eval_seed_30.json`
- `envs/mdp/cube_palette.py` (NEW)
- `/tmp/yaw_smoke/yaw_smoke.hdf5` (3-demo smoke output, pre-color-stamping)
- `/tmp/yaw_30/cube_scripted_yaw30.hdf5` (30-demo G1 confirmation, 832 MB; 30 real + 30 placeholder demos; per-episode `attrs[task]` correct)
- `reports/runs/scripted_yaw_30demo_2026-04-26/run.log`
- `reports/runs/scripted_yaw_30demo_2026-04-26/action_variance.png` (per-step action variance plot for first 50 steps, e9y deliverable)
- `scripts/validate/scripted_action_variance.py` (NEW: e9y analysis)
- `scripts/validate/inspect_demo_color_metadata.py` (NEW: 966 verification)
- `scripts/validate/render_cube_grid_for_attn_diff.py` (NEW: uxt data-capture)
- `scripts/validate/attention_difference.py` (NEW: uxt analysis)
- `scripts/validate/trajectory_overlay.py` (NEW: vd0 analysis)
- `scripts/validate/target_color_stability_probe.py` (NEW: 8ii probe)

## Lesson
- Keep the per-episode color word out of the float observation pipeline: ship the integer palette index through obs/recorder, then map index → string at conversion / inference boundaries. Saves wrestling with non-numeric tensors. The single source of truth (`envs/mdp/cube_palette.py`) is duplicated as a 5-string list in `isaaclab_to_lerobot.py` because that converter must run under `/opt/IsaacSim/python.sh` outside the Isaac stage; that's an acceptable cost.
- The recorder writes interleaved placeholder stubs, so any "stamp the latest demo" logic has to filter by `num_samples > 0` first. Naïve `demo_<successful-1>` will silently stamp the wrong group.

## Next step
1. ✅ G1 confirmed (30/30) — vla_kitting-004 ready to close; pre-train pipeline unblocked.
2. ✅ Per-episode color → prompt pipeline confirmed via inspector — vla_kitting-966 ready to close.
3. ✅ Action-variance diagnostic produced (ee_dx/ee_dy diverse, ee_dz constant by design) — vla_kitting-e9y ready to close.
4. **Decide vla_kitting-mil** (xy_tolerance vs disk-radius) before regenerating the scripted-demo dataset — currently flagged for human decision; recommended default 0.075 m.
5. **Run the remaining diagnostics on the v4 checkpoint** while GPU is available:
   - vla_kitting-uxt: render cube grid → 9 attention overlays → run `attention_difference.py`. Should take ~30 min once GPU is free.
   - vla_kitting-vd0: 12-position run with `--action_log_csv` → `trajectory_overlay.py`. ~20 min.
   - vla_kitting-hzj: re-eval Run B prime checkpoint on the fixed seed-30 set. ~30 min.
   - vla_kitting-8ii: target-color stability probe. ~5 min.
6. After (4) lands, generate the v5 training dataset by SCRIPTED demos at scale (Mimic is OUT — see reports/2026-04-27_scripted_only_data_pipeline.md), with correct attrs[task] + obs/cube_color_idx on every demo, then proceed to v5 training.
