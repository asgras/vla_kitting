# Phase 6 Teleop — Viewport Freeze Handoff

**Session paused:** 2026-04-20 ~02:25 UTC
**Branch:** `vla-pipeline-v1`
**Status:** Blocked on Isaac Sim 5.1 viewport not updating during simulation. Physics works; IK works; teleop input reaches the env; the Isaac Sim viewport stays frozen on initial USD poses.

## The goal

Get keyboard teleop working inside the DCV session so the user can record ~15 successful cube pick-and-place demos to `datasets/teleop/cube_raw.hdf5`. This is the only human gate in `CLAUDE_CODE_PLAN.md`.

## What definitely works

Confirmed from earlier teleop runs that reached ~step #750 before this debug session:

- `just teleop 1 1` launches Isaac Sim on DCV display `:1` (after the AppLauncher and kit-file fixes below).
- The `Se3Keyboard` device receives W/S/A/D/Q/E/I/J/K/L/U/O via `carb.input` — verified by the instrumented key-event logger in `scripts/teleop/record_demos.py`.
- Our key-state-driven replacement for `_on_keyboard_event` resolves an upstream bug where `CHARACTER` events crash `event.input.name` and leave the PRESS/RELEASE accumulator asymmetric. State now tracks held keys correctly; a tap produces `[0] → [+0.3] → [0]` not `[0] → [0] → [-0.3]`.
- `env.step()` runs, actions reach the IK controller, physics updates (user confirmed EE moved 10 cm in X and 16 cm in Z between step #1 and step #150 while logs showed the arm "not moving" in the viewport).
- `wrist_cam` and `third_person_cam` sensors attach via Replicator (`RGB annotator attached` in kit log).

## What doesn't work

**The Isaac Sim viewport does not update during simulation.** The arm is rendered once at its initial `init_state` pose and stays frozen there forever, even though physics is stepping through vastly different joint configurations.

## Root cause

1. Isaac Sim 5.1's RTX renderer pulls transforms via the **Fabric Scene Delegate** (`usdrt.hydra.fabric_scene_delegate.plugin`). The delegate reads from *Fabric*, not USD.
2. Setting `self.sim.use_fabric = True` in the env cfg *would* make PhysX write to Fabric. But that path imports `isaaclab/utils/warp/fabric.py:164`, which calls `wp.transform_compose`. In the **`omni.warp.core 1.7.1`** extension bundled with Isaac Sim 5.1, `transform_compose` does not exist (only the base Python `warp 1.12.1` at `/home/ubuntu/.local/lib/python3.11/site-packages/warp` has it). Result: `WarpCodegenError: Could not find function wp.transform_compose as a built-in or user-defined function.`
3. So Phase 7 set `use_fabric = False`. Physics then writes only to USD. BUT Isaac Lab's `simulation_context.forward()` (source at `/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:533-539`) reads:

   ```python
   def forward(self) -> None:
       """Updates articulation kinematics and fabric for rendering."""
       if self._fabric_iface is not None:
           if self.physics_sim_view is not None and self.is_playing():
               self.physics_sim_view.update_articulations_kinematic()
           self._update_fabric(0.0, 0.0)
   ```

   With `use_fabric=False`, `_fabric_iface is None`, and `forward()` is a no-op. Physics-to-Fabric sync never happens. Fabric stays at its initial snapshot. The delegate renders stale state. **The viewport never changes.**

### Kitting_ws comparison
Explore agent searched `~/kitting_ws` for a workaround. It uses `await app.next_update_async()` after every stage mutation and manual OmniGraph impulse-event ticking, but kitting_ws leaves `use_fabric=True` (default) — it never hit the Phase 5.1 warp crash because it was built before that particular patch. Not directly applicable.

## Attempts tried (all fail or are too slow)

| Attempt | Result |
|---|---|
| Re-enable `use_fabric=True`, leave `fabric.py` unpatched | WarpCodegenError at env init (confirmed) |
| Re-enable `use_fabric=True` + patch `fabric.py` to compose from `wp.quat_to_matrix` | Past warp compile, then hangs deep in `libomni.fabric.plugin` for >5 minutes, never reaches step #1 |
| Set `app.useFabricSceneDelegate = false` in `isaaclab.python.kit` | Triggers ~10 min RtPso recompile, eventually reaches env init, then hangs in Vulkan/glcore poll |
| OpenCV `third_person_cam` live-camera window (cv2.imshow in step hook) | Sim never reaches the step hook — same underlying hang |
| Kill a leaked 24-hour-old Phase 5 `env_smoke.py` (PID 29566, held 4 GB GPU) | GPU cleaned up but subsequent launches still hang 10+ min without step #1 |

## Files modified this session (mix of fixes kept + patches for debugging)

### Kept — legitimate fixes
- **`/home/ubuntu/IsaacLab/apps/isaaclab.python.kit` line 41** (outside the repo): `"isaacsim.asset.importer.urdf" = {}` (was `{version="2.4.31", exact=true}`). Isaac Sim 5.1 only ships 2.4.19. Without this the AppLauncher dep solver fails.
- **`justfile` `teleop` recipe**: added `--enable_cameras` flag (our env defines wrist + third-person cameras, so env creation errors out without it).
- **`envs/yaskawa_pick_cube_cfg.py`**:
  - `pos_sensitivity` 0.05 → 0.3, `rot_sensitivity` 0.05 → 0.5 (`Se3KeyboardCfg` was producing 5 mm/step commands that couldn't overcome gravity sag)
  - Comment updated on `use_fabric = False` (still False)
- **`envs/yaskawa_robot_cfg.py`** actuator config: arm `stiffness` 1200 → 8000, `damping` 80 → 500, `effort_limit_sim` 400 → 1000. Needed so the IK-commanded joint targets actually overcome gravity. User-verified: at step #150 the arm moved 10 cm/16 cm under teleop.

### Kept — debugging scaffolding in `scripts/teleop/record_demos.py`
This wrapper does three things now:
1. Text-rewrites Isaac Lab's `/home/ubuntu/IsaacLab/scripts/tools/record_demos.py` to inject `import envs` right after `AppLauncher` (we can't import `envs` top-level because that triggers `isaaclab.assets` → `omni.physics` which doesn't exist pre-SimulationApp).
2. Monkey-patches `Se3Keyboard._on_keyboard_event` with a key-state-tracking replacement (fixes the accumulator drift from the CHARACTER-event crash).
3. Monkey-patches `ManagerBasedRLEnv.step` to log `act / joint_pos / joint_delta / ee_tool0` every ~0.5 s.

The cv2 live-camera hook inside `_patched_step` was REMOVED (it never triggered because step #1 was never reached on recent runs, and it was my primary suspect for the hangs). Current patch only logs.

### Reverted
- `/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/utils/warp/fabric.py` — back to upstream (`wp.transform_compose`). Re-apply the quat_to_matrix patch only when you revisit fabric=True.
- `/home/ubuntu/IsaacLab/apps/isaaclab.python.kit` line 245 — back to `app.useFabricSceneDelegate = true` (the `false` path triggered a 10 min PSO recompile).

## Memory entries relevant to this
- `project_isaaclab_kit_patch.md` — tracks the urdf pin fix (will need reapplying if IsaacLab is re-pulled)
- `project_vla_pipeline.md` — overall V1 plan pointer

## Before re-starting the session

**Kill any leftover processes first** — this session spawned lots of Isaac Sim processes and at least one leaked for 24 hours:
```bash
pgrep -af "record_demos\|isaaclab.sh\|python3.*kit/python\|python3.*Isaac"   # audit
# then kill each PID that looks like it's been running for an unreasonable time
kill -9 <pid>...
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv  # confirm GPU clean
```

## Current git state to check
```bash
git -C /home/ubuntu/vla_kitting status
git -C /home/ubuntu/vla_kitting diff envs/ justfile scripts/teleop/
```
No commits made this session — all changes are on the working tree of `vla-pipeline-v1`.

## Next step: research

**Hypothesis:** Using Isaac Sim standalone + Isaac Lab + omni.warp.core 1.7.1 with live teleop must be a hit path that others have solved. We're probably missing a known workaround, a carb setting, or a specific ext upgrade.

Concrete research queries to run (in order of promise):

1. **Direct community signal.** Search GitHub issues + NVIDIA forums + Isaac Sim Discord logs for:
   - `"transform_compose" "omni.warp.core" 1.7.1`
   - `"warp.codegen.WarpCodegenError" "transform_compose"`
   - `isaaclab use_fabric viewport frozen`
   - `isaaclab 2.3.2 "Isaac Sim 5.1" warp fabric`
   - The specific kit line: `isaaclab/utils/warp/fabric.py` + "transform_compose"
   Likely fix: upgrade `omni.warp.core` extension, OR symlink the bundled warp dir to the newer pip warp install (`/opt/IsaacSim/extscache/omni.warp.core-1.7.1+lx64/warp` → `/home/ubuntu/.local/lib/python3.11/site-packages/warp`).
2. **Isaac Sim 5.1 → newer patch.** Check if Isaac Sim has a 5.1.x patch that ships `omni.warp.core ≥ 1.8` (where `transform_compose` was added). NVIDIA might have published a patch pack.
3. **Isaac Lab version pinning.** Our Isaac Lab is v2.3.2 on Isaac Sim 5.1. Check if a newer IsaacLab (v2.4+) drops the `wp.transform_compose` call for a version-compatible implementation. `git log -- source/isaaclab/isaaclab/utils/warp/fabric.py` in the IsaacLab repo.
4. **LeIsaac reference project** — referenced in `CLAUDE_CODE_PLAN.md` Phase 10 for dataset conversion, but they also do Isaac Lab teleop. Look at their env configs and launch scripts for known workarounds.
5. **Alternative teleop tools** — SpaceMouse + `Se3SpaceMouse`, the `record_demos` script with a different `--teleop_device`, or a stand-alone OpenCV viewer process that reads the USD stage and renders separately.
6. **Kit experience file survey** — the IsaacLab `apps/*.kit` files expose per-variant rendering settings. Grep for what the non-frozen-viewport variants have differently, especially `isaaclab.python.rendering.kit` vs `isaaclab.python.kit`.
7. **`/usdrt` carb settings** — search Isaac Sim docs for settings like `/omnihydra/parallelHydraSprim`, `/usdrt/scenegraph/enablePrimCaching`, `/app/renderer/forceSceneUpdate`. There may be a setting that makes the Fabric Scene Delegate pull from USD on change.
8. **Physx-USD sync.** `omni.physx.acquire_physx_interface().force_load_physics_from_usd()` already runs in `reset_async`. Check if there's a reverse-direction `force_flush_physics_to_usd_to_fabric` or equivalent.

For each, cite source link + exact code change. Do NOT re-run Isaac Sim tests until at least one concrete candidate is identified — each blind launch eats 5-15 minutes.

## Fallback if research doesn't yield a fix

Switch to **Option B** from the session log: skip interactive teleop for V1, finish fixing the Phase 7 scripted-pick grasp (`reports/phase_7_findings.md` lists three hypotheses: finger-table collision, gripper drive gains, mimic joint limits), generate 15 scripted demos programmatically, and proceed with Mimic→LeRobot→SmolVLA. V1's stated goal is *pipeline validation*, achievable without a human in the loop.

## Key files to look at first next session

- This file.
- `CLAUDE_CODE_PLAN.md` — the original plan.
- `logs/PROGRESS.json` — phase tracker.
- `reports/phase_7_findings.md` — the scripted-pick grasp issues (the Option B path).
- `scripts/teleop/record_demos.py` — the wrapper with debug instrumentation.
- `envs/yaskawa_pick_cube_cfg.py` — env cfg (note `use_fabric=False`).
- `envs/yaskawa_robot_cfg.py` — actuator config (arm stiffness 8000).
- `/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:533` — the `forward()` no-op that is the root cause.
- `/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/utils/warp/fabric.py:164` — the `wp.transform_compose` that blocks `use_fabric=True`.
- `/home/ubuntu/IsaacLab/apps/isaaclab.python.kit` — kit experience file (note our urdf pin fix).
