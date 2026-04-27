# Ghost cube in eval renders — investigation snapshot

**Date:** 2026-04-26
**Status:** ROOT CAUSE FOUND AND FIXED. DLSS temporal-AA was synthesizing
ghost cubes from prior-reset render history. FXAA fix applied at
`envs/yaskawa_pick_cube_cfg.py:315`. Cam samples post-fix at
`reports/runs/vision_grounded_30hz_2026-04-24/cam_check_postfix/` show a
single cube per frame across 6 resets. Eval of v3.2 ep22 with the fix
remained 0/10 — ghost was a real confound but the trained policy still
fails to lift the cube (gripper-never-closes, Hole B). Next experiment:
gripper loss weighting on existing data.
**Context:** user noticed a second cube visible in the eval third-person camera. Asked whether this is an artifact or a bug. Tied to broader train/eval observation mismatch we were already chasing (stale start frame, wrist cam difference).

## TL;DR

The "second cube" is real — visible in multiple eval renders and in the
`save_camera_samples.py` outputs from 2026-04-25. It is **not a single-frame
artifact**, and it is **not present in the recorded scripted-demo HDF5
frames**. So at minimum the eval scene is rendering differently than the
scene the policy was trained on. Cause is not yet pinned down — strongest
remaining hypothesis is a USD/Fabric sync issue around `write_root_pose_to_sim`
when the cube is teleported during reset, leaving a render of the cube at
its default spawn pos `(0.55, 0.0, 0.025)` alongside the randomized
position. The previous attempt at `dump_scene_prims.py` hung at
`env.reset()` for 10+ min on 2026-04-25 16:11 and was killed without a
diagnosis (`reports/runs/vision_grounded_30hz_2026-04-24/run_diary.md:223`).

## Evidence the bug is real

Frames inspected:

- `reports/runs/vision_grounded_30hz_2026-04-24/cam_check/sample_00..05_third.png`
  — `save_camera_samples.py` outputs from 2026-04-25 16:11. **Multiple
  cube-shaped objects visible across resets.** Examples:
  - sample_02: 3 distinct cubes (yellow, pink, blue) at three different table
    positions.
  - sample_05: vivid yellow cube + faint grayish "ghost" cube at a fixed
    center-back table position.
- `reports/runs/vision_grounded_30hz_2026-04-24/vgw_30hz_v3_2_eval_epoch_0020.gif`
  frame 0 (extracted to `/tmp/eval_frame_000.png`) — vivid pink cube AND a
  faint pink cube next to it, plus the magenta target marker. Wrist cam
  also shows the faint cube.
- Compare: `vgw_30hz_scripted_demo_0.gif` and `vgw_30hz_scripted_demo_200.gif`
  frame 0 (`render_demo.py` rendering from HDF5 `obs.third_person_cam`) —
  **only one cube on the table.** Clean.

Pattern: the actual cube has the randomized palette color from
`mdp.events.randomize_cube_color`. The "ghost" cube appears at roughly the
**default spawn pos `(0.55, 0.0, 0.025)`** in samples 04/05 (the gray ghost
is centered on the table at the cube's pre-randomization location).

## What rules in / rules out

- **Not a Mimic vs non-Mimic env difference.** Both `scripted_pick_demo.py`
  and `run_vla_closed_loop.py` use `Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0`
  (the non-Mimic env). Only `mimic_generate.sh` uses the `-Mimic-v0` env.
- **Not the `--cube_xy` override path.** Regular eval through
  `train_only.sh:298-306` does NOT pass `--cube_xy`, so the
  `write_root_pose_to_sim` block at
  `scripts/train/run_vla_closed_loop.py:228-244` is not executed. The cube
  uses the env's natural reset+randomize.
- **Not the data pipeline.** `render_demo.py` replays HDF5 frames, and those
  show 1 cube. So the camera-captured frames during scripted data
  collection are clean.
- **Not the magenta target marker.** That's a thin (1 cm) 20×20 cm cuboid
  at `(0.65, 0.20, 0.005)` and is clearly visible as the magenta square on
  the right of every frame. The "ghost cube" is a separate, cube-shaped,
  smaller object, sitting up on the table surface.

So the divergence is between **runtime camera frames during eval** and
**runtime camera frames during scripted data collection**. Both run on
the same env config, num_envs=1, and call `env.reset()` once per episode.

## Probable cause (highest-likelihood hypothesis)

The env config sets `self.sim.use_fabric = True`
(`envs/yaskawa_pick_cube_cfg.py:315`). With Fabric, physics-driven
transforms live in Fabric, and the RTX renderer reads from Fabric for
moving objects. The cube reset path is two-stage:

1. `mdp.reset_scene_to_default` — writes default pose `(0.55, 0.0, 0.025)`
   via `rigid_object.write_root_pose_to_sim`
   (`/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:1369-1375`).
2. `mdp.reset_root_state_uniform` — samples a random delta and writes the
   randomized pose through the same API.

If, on the eval path, the renderer ends up displaying both the USD-resident
cube prim (still sitting at the default pose, never rewritten to USD) and
the Fabric-resident cube (now at the randomized pose), we'd see two cubes
— one at the default spawn, one at the randomized spawn. That matches the
observation that the ghost in samples 04/05 sits at center-back of the
table where the default `(0.55, 0.0, 0.025)` projects.

The discrepancy with scripted-demo HDF5 frames is consistent with this
hypothesis: those frames are also rendered via the cameras at run time —
but possibly the recorder's internal `env.step` cadence or settle behavior
masks the issue. Need to confirm.

## Other hypotheses (not yet ruled out)

- **Decimation 2→4 change** between when scripted data was collected
  (decimation=2) and when cam_check / eval ran (decimation=4). The
  `sim.render_interval = self.decimation` line means render cadence
  changed. Plausible but no clear mechanism for spawning a ghost.
- **Material sharing / replication artifact.** A standing comment in the
  env config notes that "green/cyan/emissive" target-marker variants
  rendered in the cube's color due to scene-replication material sharing
  (`envs/yaskawa_pick_cube_cfg.py:67-70`). Magenta was chosen specifically
  to avoid this. Possible that the same sharing mechanism is creating a
  visible cube-shaped ghost somewhere.
- **`randomize_cube_color` walking the wrong subtree.** The function walks
  the prim path subtree and applies `diffuseColor` recursively
  (`envs/mdp/events.py:32-55`). If the subtree contains stale duplicate
  prims, they'd all get colored — but they wouldn't appear/disappear, just
  re-color.

## What I tried / what's left

- **Tried:** read the env cfg, the closed-loop script, the scripted demo
  script, the events code, IsaacLab's `reset_scene_to_default`. Compared
  scripted-HDF5-rendered frames vs eval-live-rendered frames — confirmed
  the bug is on the live-render path.
- **Did not finish:** running a USD prim-tree dump to confirm whether
  there's actually >1 cube prim or just 1 prim being rendered twice. I
  wrote `scripts/validate/dump_scene_prims_fast.py` (does NOT call
  `env.reset()` — only `gym.make` then traverses the stage), which avoids
  the hang the original `dump_scene_prims.py` hit. **User stopped me
  before the run completed**; the script is staged.

## Resolution (2026-04-26 ~03:30 UTC)

1. Ran the staged dump (`scripts/validate/dump_scene_prims_fast.py`).
   Output: `/tmp/prim_dump.log`. Result: **only one cube prim exists**
   at `/World/envs/env_0/Cube`. No siblings, no leftover prototypes.
   Render-side issue confirmed.
2. The dump output included this revealing warning:
   ```
   [Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions:
       Render resolution of (74, 74) is below minimal input resolution of 300.
   ```
   DLSS (the IsaacLab default `antialiasing_mode`) is active. DLSS
   reconstructs each frame from a lower-resolution input plus motion
   vectors and a multi-frame history. When the cube teleports during
   reset (`write_root_pose_to_sim`), DLSS has no motion vector for the
   instantaneous jump, so prior-frame samples of the cube at its
   previous spawn pose bleed into the new frame as faint cube ghosts.
3. **Fix:** set `self.sim.render.antialiasing_mode = "FXAA"` in
   `envs/yaskawa_pick_cube_cfg.py:315`. FXAA is non-temporal so no
   history bleeds across resets.
4. **Verified:** re-ran `save_camera_samples.py --samples 6` and saved
   to `cam_check_postfix/`. Every frame now shows exactly one cube.
   Compare `cam_check/sample_05_third.png` (yellow cube + gray ghost)
   vs `cam_check_postfix/sample_05_third.png` (single blue cube).

## Postfix eval — ghost was a real confound but didn't unlock learning

Ran 10 eval episodes on `checkpoints/continual/checkpoints/last`
(v3.2 epoch 22) with the FXAA fix. Result: **still 0/10 success**.

| Metric | v3.2 ep22 (DLSS on) | v3.2 ep22 (FXAA on, postfix) |
|---|---|---|
| Real SR | 0/9 | 0/10 |
| Lifts (z > 0.05) | 0 | 0 |
| cube_end x stdev | 0.043 | ~0.06 |
| cube_end y stdev | 0.146 | ~0.13 |

Cube positions at episode end (z always 0.025 — never lifted):

| ep | cube_start | cube_end | Δxy |
|---|---|---|---|
| 0 | (0.51, -0.10) | (0.53, -0.12) | 3 cm |
| 1 | (0.69, -0.04) | (0.47, -0.10) | 22 cm |
| 2 | (0.49,  0.04) | (0.52,  0.20) | 16 cm (toward target!) |
| 6 | (0.53,  0.05) | (0.65,  0.14) | 14 cm (toward target!) |
| 7 | (0.65, -0.14) | (0.48,  0.16) | 35 cm |
| 9 | (0.49,  0.00) | (0.50,  0.19) | 19 cm (toward target!) |

The policy DOES move cubes — sometimes meaningfully toward the target —
but never closes the gripper. Gripper stays open throughout every
episode (verified visually in `postfix_validation/eval_ep2.gif`).

**Diagnosis: Hole B from `reports/recovery_plan_2026-04-24.md` §3.** The
close transition is statistically invisible under flat L2 over a 7D
action vector — gripper is +1.0 for ~30% of demo frames, then -1.0 for
the rest, with the transition localized to one step. Regressing the
gripper to ~+0.9 throughout is a near-optimal lazy solution.

The recovery plan explicitly disabled action_loss_dim_weights for v3
("canonical-first") with note "ablation-add-back if gripper behavior is
bad." It is bad. Time to add it back.

## Files touched this session

- `envs/yaskawa_pick_cube_cfg.py` — added
  `self.sim.render.antialiasing_mode = "FXAA"` with explanatory comment.
- `reports/runs/vision_grounded_30hz_2026-04-24/cam_check_postfix/` —
  6 wrist + 6 third-person samples confirming the fix.
- `reports/runs/vision_grounded_30hz_2026-04-24/postfix_validation/` —
  10 eval gifs + episodes.jsonl confirming 0/10 with the fix.

## Resume from here (historical, kept for posterity)

1. Run the staged dump:
   ```
   timeout 240 /home/ubuntu/IsaacLab/isaaclab.sh \
     -p scripts/validate/dump_scene_prims_fast.py 2>&1 | tee /tmp/prim_dump.log
   ```
   Look for: more than one prim with "cube" in its path; whether
   `/World/envs/env_0/Cube` has siblings; whether there's a leftover
   prototype prim somewhere under `/World` or `/World/envs`.

2. If only one cube prim exists, the issue is render-side
   (Fabric/USD sync). Quick test: set `use_fabric = False` temporarily
   and re-run `save_camera_samples.py`. If the ghost disappears, that
   confirms it.

3. If multiple cube prims exist, dump ancestry to find what's creating
   them — likely something interacting badly with the
   `replicate_physics` / scene replication path or the
   `RigidObjectCfg.spawn` proto-prim.

4. Either way: re-render a current eval rollout AND a current
   scripted-demo open-loop rollout under the SAME (current) env config,
   to confirm the ghost is now in BOTH paths. The earlier
   `vgw_30hz_scripted_demo_0.gif` was made on the prior `decimation=2`
   env, which is a confound.

## Files touched this session

- (new) `scripts/validate/dump_scene_prims_fast.py` — fast prim dump that
  skips the `env.reset()` hang.
- (no other code changes)

## Index

To add to `reports/README.md` (when continuing):
- 2026-04-26 — eval_ghost_cube_investigation: confirmed second cube in
  eval renders, NOT in HDF5 dataset frames; prim-tree dump pending.
