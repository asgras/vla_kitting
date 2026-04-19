# VLA Picking Pipeline — Claude Code Execution Plan (V1)

**Target:** End-to-end working VLA policy for a cube pick-place task in Isaac Sim, using a **Yaskawa Motoman HC10DT cobot (6-DoF, 10kg payload, 1.2m reach, YRC1000micro controller)** + Robotiq 2F-85 gripper + SmolVLA-450M. This validates the entire pipeline before we apply it to real SKUs.

**Robot asset source of truth:** The HC10DT URDF is maintained in `ros-industrial/motoman` under `motoman_hc10_support` (also `motoman_hc10dt_support` in newer forks). The tool flange is ISO 9409-1-50-4-M6, which is the Robotiq 2F-85's native mount — no adapter geometry needed.

**Success criterion:** Fine-tuned SmolVLA achieves ≥70% success rate on 20 held-out cube-pick rollouts in sim, with <10s avg episode length, using a keyboard-teleop-collected dataset multiplied via Isaac Lab Mimic.

**Human involvement:** Exactly one gate — Phase 8 requires the user to teleop ~15 cube demos (~30–60 min). Everything else runs autonomously. Claude Code pauses, provides clear instructions + a single command, and resumes after user confirms demos are recorded.

---

## How to use this plan

You are Claude Code executing this plan end-to-end.

- Execute phases top-to-bottom. Do NOT skip or reorder.
- Each phase has an **entry gate** (must be true to start), **steps**, **validation tests** (must pass to proceed), and **exit gate**.
- If a validation test fails, STOP and either self-repair (max 3 attempts) or ask the user. Never silently move on.
- Log everything to `logs/phase_<N>_<slug>.log` with timestamps. Append, don't overwrite.
- Maintain `logs/PROGRESS.json` with `{"current_phase": N, "completed": [...], "blocked_on": null|str, "last_update": ISO}`. Read it on startup — if non-empty, resume from last incomplete phase.
- **HUMAN GATE** markers mean STOP. Print a clear instruction block and exit. The user will re-invoke you to continue.
- Test after every meaningful change. Assume nothing works until you've verified it.

## Assumptions (Phase 0 verifies)

- OS: Ubuntu 22.04 on EC2 g5.2xlarge (or equivalent with ≥24GB VRAM, A10G/L4/A100-class GPU, RTX-compatible)
- **NICE DCV is already installed and running on the EC2 instance.** User connects from their laptop via the DCV client — no browser/WebRTC setup needed. Isaac Sim will run with a visible window on the DCV virtual display.
- Python 3.10+ available
- Network access available (PyPI, HF Hub, NVIDIA registries)

### What this plan expects in the repo

At kickoff, the repo only needs to contain:

1. **A USD scene file** — anywhere under the repo root, findable by `fd -e usd -e usda`. Must contain (at minimum) a 6-DoF Yaskawa HC10DT articulation and a table/worksurface. Phase 0 auto-discovers it; Phase 3 validates the articulation matches HC10DT spec.
2. **Write access to the repo root** so Claude Code can create new directories (`envs/`, `scripts/`, `datasets/`, `logs/`, `checkpoints/`, `reports/`, `tests/`) and top-level files (`justfile`, append to `.gitignore`).

### What the plan will NOT touch

- Existing ROS launch files, nodes, URDFs, or Python modules
- Any Isaac Sim scene files it didn't create (it always writes a new `assets/scene_with_gripper.usd` — never overwrites the original)
- Existing Python virtualenvs (it uses `uv` to create an isolated one)
- The main/master branch (all work happens on a new `vla-pipeline-v1` branch)

This V1 is a **parallel Isaac Lab workflow** running alongside your existing ROS-based picking code, not a refactor or integration of it. Sim-to-real and ROS integration are out of scope for V1.

### HuggingFace token

**Not needed at kickoff.** The SmolVLA base model is gated-but-public, downloadable anonymously. A token is only required for Phase 10's optional dataset push to HF Hub. Phase 10 will prompt for it then, and the user can still say "skip" at that point.

---

## Phase 0 — Environment Audit & Clarification

**Entry gate:** You have received this plan.

**Objective:** Understand the actual environment. Ask the user upfront for anything you can't auto-detect, so the rest of the plan runs without interruption.

**Steps:**

1. Run environment audit and save to `logs/phase_0_audit.log`:
   - `uname -a`, `lsb_release -a`, `nvidia-smi`, `free -h`, `df -h`, `python3 --version`
   - Check CUDA version: `nvcc --version || true`
   - Check if Isaac Sim is installed: look in `~/.local/share/ov/pkg/`, `/isaac-sim`, `/opt/isaac-sim`. Log version if found.
   - Check if Isaac Lab is cloned: look for `IsaacLab/` or `isaac_lab/` directories under home and repo root.
   - Check existing GPU memory availability with a 30s idle `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5`.
2. Scan the repo for the existing Isaac scene:
   - Find USD files: `fd -e usd -e usda` (install `fd` in Phase 1 if not present; fall back to `find . -name "*.usd*"`)
   - For each USD found, inspect with a small Python script using `pxr.Usd` to list prims, find the robot articulation, and confirm it matches HC10DT (6 revolute joints named similarly to `joint_s`, `joint_l`, `joint_u`, `joint_r`, `joint_b`, `joint_t` — Yaskawa's standard joint nomenclature).
   - Save scene manifest to `logs/phase_0_scene_manifest.json` with keys: `{file, robot_prim_path, joint_names: [...], dof_count, world_prim_path, existing_objects: [...], camera_prims: [...]}`
   - If the articulation does not have 6 DoF or joints don't match HC10DT nomenclature, flag in clarification prompt — scene may be a different Yaskawa model.
3. Verify DCV is the display method:
   - `systemctl is-active dcvserver || pgrep -f dcvserver` — confirm DCV is running
   - `echo $DISPLAY` and `xdpyinfo 2>&1 | head -5` — confirm a display is attached
   - Log which DISPLAY value the DCV session exposes (typically `:0`). Save to `logs/phase_0_display.log`.
   - If DCV is NOT running, STOP and ask user to start it before continuing.
4. **Clarification prompt:** Write a single-screen block asking the user to confirm only what can't be auto-detected. Use this exact format:

```
=== PHASE 0 CLARIFICATION ===
Detected:
  - Isaac Sim:     <version or NOT FOUND>
  - Isaac Lab:     <version or NOT FOUND>
  - DCV:           <running / NOT RUNNING>
  - DISPLAY:       <e.g. :0>
  - Scene file:    <path>
  - Articulation:  <N DoF, joints: [s,l,u,r,b,t]>  ← should be 6 DoF for HC10DT

Please confirm or correct:
  1. Scene articulation matches Yaskawa HC10DT (6 DoF, joint names s/l/u/r/b/t)? [Y / N + notes]
  2. Scene has a table/worksurface? [Y/N]
  3. Proceed? [Y/N]

Reply with one message answering all. I will not interrupt again until teleop.
(HuggingFace token is NOT needed now — I'll ask in Phase 10 if you want to push
the dataset to the Hub, and you can skip it then too.)
=============================
```

5. Wait for user reply. Parse into `logs/phase_0_clarifications.json`. If user says "proceed" with only "Y" answers, use defaults: HC10DT confirmed + DCV display.

**Validation tests:**
- [ ] `phase_0_audit.log` exists and is non-empty
- [ ] `phase_0_scene_manifest.json` exists with a valid `robot_prim_path` and `dof_count == 6`
- [ ] `phase_0_clarifications.json` exists
- [ ] `phase_0_display.log` shows DCV active and a valid DISPLAY set
- [ ] GPU has ≥20GB free VRAM (if less, warn user and suggest `g5.2xlarge` or larger)

**Exit gate:** All clarifications captured. Scene file path known. Robot model confirmed.

---

## Phase 1 — Tooling & Repo Setup

**Entry gate:** Phase 0 complete.

**Objective:** Install tooling that makes the rest of this plan fast and debuggable. Set up repo structure and task runner.

**Steps:**

1. Install fast CLI tooling (system-level or userspace as permissions allow):
   ```bash
   # uv — 10–100x faster Python package management
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Fast file search / find
   sudo apt-get update && sudo apt-get install -y ripgrep fd-find jq htop tmux

   # nvtop — GPU monitoring during training
   sudo apt-get install -y nvtop || echo "nvtop not available, using nvidia-smi"

   # just — task runner (makes every command in this plan one-liner re-runnable)
   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
   ```

2. Python tooling for our workspace:
   ```bash
   uv pip install ruff pytest pytest-xdist rich typer
   ```

3. Create the project directory structure (DO NOT overwrite existing files):
   ```
   ./
   ├── envs/              # Isaac Lab task env definitions
   ├── scripts/
   │   ├── validate/      # sanity-check scripts (one per phase)
   │   ├── teleop/        # keyboard teleop
   │   ├── data/          # mimic, conversion, inspection
   │   └── train/         # fine-tune + eval
   ├── datasets/          # HDF5 + LeRobot output (gitignored)
   ├── checkpoints/       # model checkpoints (gitignored)
   ├── logs/              # all phase logs + progress marker
   ├── reports/           # end-of-phase summaries, final report
   ├── tests/             # pytest suite
   ├── justfile           # task runner entry points
   └── CLAUDE_CODE_PLAN.md  # this file
   ```

4. Create `.gitignore` entries (append if exists):
   ```
   datasets/
   checkpoints/
   logs/*.log
   logs/*.mp4
   *.hdf5
   wandb/
   ```

5. Create `justfile` with these initial recipes (add more in each phase):
   ```make
   default:
       @just --list

   audit:
       python scripts/validate/phase_0_audit.py

   scene-inspect:
       python scripts/validate/scene_inspect.py

   test-all:
       pytest tests/ -v

   logs-tail phase:
       tail -f logs/phase_{{phase}}_*.log

   gpu:
       nvtop
   ```

6. Initialize `logs/PROGRESS.json`:
   ```json
   {"current_phase": 1, "completed": [0], "blocked_on": null, "last_update": "<ISO>"}
   ```

7. Create new git branch: `git checkout -b vla-pipeline-v1` (don't commit yet).

**Validation tests:**
- [ ] `uv --version`, `rg --version`, `jq --version`, `just --version` all succeed
- [ ] `tree -L 2` shows the expected structure
- [ ] `just --list` shows recipes
- [ ] On current branch `vla-pipeline-v1`

**Exit gate:** All tooling works. Repo structure in place. Progress marker updated.

---

## Phase 2 — Isaac Sim / Isaac Lab Verification

**Entry gate:** Phase 1 complete.

**Objective:** Prove Isaac Sim and Isaac Lab work headless on this GPU before building on top of them.

**Steps:**

1. If Isaac Sim not installed (from Phase 0), install via pip:
   ```bash
   uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
   export ACCEPT_EULA=Y
   export PRIVACY_CONSENT=Y
   ```

2. If Isaac Lab not cloned:
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
   cd ~/IsaacLab && git checkout v2.3.0
   ./isaaclab.sh -i
   ```

3. Run Isaac Lab's built-in verification task **headless** (proves renderer + physics + GPU work together):
   ```bash
   cd ~/IsaacLab
   ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 16 --headless
   ```
   Expected: runs without CUDA errors, prints environment step rates ≥200 Hz aggregate.

4. Run a short camera-enabled task to verify RTX rendering works headless:
   ```bash
   ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_envs 2 --headless --enable_cameras
   ```

5. Write `scripts/validate/isaac_smoke.py` that:
   - Imports `isaaclab`, `isaacsim`
   - Launches a SimulationApp with headless=True
   - Creates a minimal stage with one cube
   - Steps physics 100 times
   - Reports step rate
   - Exits cleanly

6. Write a pytest: `tests/test_isaac_smoke.py` invoking the above via subprocess (isaaclab scripts can't be imported directly in pytest — must use subprocess).

**Validation tests:**
- [ ] Cartpole runs without CUDA OOM or driver errors
- [ ] Camera-enabled task completes
- [ ] Smoke test: `pytest tests/test_isaac_smoke.py -v` passes
- [ ] GPU memory used by Isaac during smoke test logged to `logs/phase_2_gpu_profile.log`

**Failure modes:**
- CUDA/driver mismatch → log `nvidia-smi` + `nvcc --version` and ask user to resolve
- GL/EGL errors in headless → try `export DISPLAY=:0` and `vulkaninfo`, log and ask user

**Exit gate:** Isaac Sim + Lab confirmed working headless, performance baseline captured.

---

## Phase 3 — Scene & Robot Validation

**Entry gate:** Phase 2 complete. Existing scene file path known.

**Objective:** Load the existing Isaac scene. Confirm the Yaskawa HC10DT is a controllable articulation with correct kinematics. Validate workspace reachability.

**HC10DT reference spec (use for sanity-checking the USD):**
- 6 revolute joints: S, L, U, R, B, T
- Joint limits (deg): S ±180, L -65 to +150, U -86 to +255, R ±180, B ±180, T ±180
- Max reach: ~1200 mm from S-axis centerline
- Payload: 10 kg
- Tool flange: ISO 9409-1-50-4-M6 (Robotiq 2F-85 native mount)
- Nominal repeatability: ±0.1 mm (for reference; sim won't match this anyway)

**Steps:**

1. Write `scripts/validate/scene_inspect.py`:
   - Opens the existing USD headless
   - Enumerates all articulations, rigid bodies, joints
   - Prints joint limits, default positions, drive types
   - Saves to `reports/scene_inspection.md` as a human-readable report
   - Saves to `logs/scene_dump.json` for machine parsing

2. Write `scripts/validate/robot_fk_ik.py`:
   - Loads the Yaskawa in a minimal Isaac Lab env
   - Runs forward kinematics on 20 random joint configs within limits, logs EE poses
   - Runs IK (via Isaac Lab's DifferentialIKController) to 20 target poses in the workspace in front of the robot
   - Measures IK success rate and solve time
   - Saves workspace reachability visualization as PNG (matplotlib scatter of reachable vs. unreachable targets)

3. Write `scripts/validate/robot_motion_test.py`:
   - Loads robot in Isaac Lab
   - Commands a smooth trajectory: home → 4 corners of a 30cm × 30cm workspace square at 15cm above table → home
   - Records joint positions, EE poses, timing
   - Saves trajectory video (MP4) to `logs/phase_3_motion.mp4`
   - Asserts no joint limit violations, no self-collision flags from PhysX

4. Pytest: `tests/test_robot.py` with fixtures that run the above as subprocess and assert success.

**Validation tests:**
- [ ] Scene loads, all expected prims found
- [ ] Articulation has exactly 6 DoF
- [ ] IK success rate ≥90% on in-workspace targets
- [ ] Motion test produces clean video with no warnings
- [ ] Joint limits from USD match HC10DT spec above (within 5° tolerance for each joint). If off, log warning but don't block — URDF conversions sometimes round limits.
- [ ] Max reach in XY plane from base is approximately 1.2m (±50mm)

**Exit gate:** Robot is a controllable articulation with verified IK and motion.

---

## Phase 4 — Robotiq 2F-85 Gripper Integration

**Entry gate:** Phase 3 complete.

**Objective:** Attach a working Robotiq 2F-85 parallel jaw to the Yaskawa flange. Validate open/close commands and verify a test grasp succeeds on a static cube.

**Steps:**

1. Obtain a Robotiq 2F-85 USD asset. Sources in priority order:
   - Check NVIDIA's Isaac Assets library at `omniverse://localhost/NVIDIA/Assets/Robots/Robotiq/` (may be available via Nucleus mount)
   - Fallback: download the URDF from `ros-industrial/robotiq` GitHub repo and convert to USD using Isaac Lab's `urdf_to_usd` tool (`~/IsaacLab/scripts/tools/convert_urdf.py`)
   - Save to `assets/robotiq_2f85.usd`

2. Write `scripts/assembly/attach_gripper.py`:
   - Opens the existing scene USD
   - Adds the Robotiq as a referenced prim
   - Creates a FixedJoint between the HC10DT tool flange and the Robotiq base.
     - In the ros-industrial `motoman_hc10_support` URDF, the flange link is typically named `tool0` (standard ROS convention) or `flange`. Phase 3's scene_inspect output tells you which.
     - The ISO 9409-1-50-4-M6 flange means zero offset / zero adapter — fixed joint is at identity transform from flange to Robotiq base.
   - Saves a new scene USD at `assets/scene_with_gripper.usd` (NEVER overwrite the original)
   - Returns the joint paths for the gripper fingers

3. Configure gripper in Isaac Lab env config (`envs/yaskawa_robotiq_cfg.py`):
   - ArticulationCfg with combined arm + gripper
   - Gripper action: binary (open=0.0, close=0.04 rad per finger) OR continuous
   - Keep both arm IK controller and gripper command in the action space

4. Write `scripts/validate/gripper_test.py`:
   - Loads combined robot in a test scene with a 5cm rigid cube on the table at EE-reachable position
   - Scripted sequence: home → approach cube (2cm above) → descend (touch) → close gripper → lift (10cm) → hold 2s → open → home
   - Records cube trajectory and gripper joint positions
   - Success criterion: cube lifted >5cm and held for >1s without slipping
   - Saves video to `logs/phase_4_grasp.mp4`

5. Tune contact physics if grasp fails:
   - V-HACD convex decomposition on cube and gripper pads (Isaac Lab has a utility; else use `pip install trimesh` + `coacd`)
   - Set gripper pad material friction coefficient to 1.0, restitution to 0.0
   - Set cube material friction to 0.8
   - Increase physics substeps to 4 (effective 240 Hz from 60 Hz base)
   - Increase solver iterations: positionIterationCount=16, velocityIterationCount=4
   - Max 3 tuning attempts; if still failing, log details and ask user

**Validation tests:**
- [ ] `scene_with_gripper.usd` loads in isolation
- [ ] Scripted grasp succeeds (cube lifted + held)
- [ ] Gripper open/close responds to action commands within 300ms
- [ ] No PhysX warnings about contact explosion or interpenetration in logs

**Exit gate:** Deterministic scripted grasp works reliably (5/5 attempts).

---

## Phase 5 — Cube Pick-Place Task Environment

**Entry gate:** Phase 4 complete.

**Objective:** Define the RL-style task environment that teleop and training will use.

**Steps:**

1. Create `envs/yaskawa_pick_cube_env.py` as an Isaac Lab `ManagerBasedRLMimicEnv` (required for later Mimic support) with:
   - **Scene:** existing scene + Robotiq gripper + one 5cm cube with random color per episode + a target zone marker (visual only)
   - **Observations:**
     - `robot_joint_pos` (6 DoF arm + 2 DoF gripper fingers = 8)
     - `robot_joint_vel` (8)
     - `ee_pose` (7D: position + quaternion)
     - `gripper_state` (2D: open fraction, is_closed)
     - `wrist_cam_rgb` (128×128, mounted on EE)
     - `third_person_cam_rgb` (256×256, angled overhead, fixed)
   - **Actions:** `ee_pose_delta` (6D) + `gripper_command` (1D binary). IK converts delta to joint targets internally.
   - **Reset:** cube position randomized in 20cm × 20cm region in front of robot; robot to home pose
   - **Success:** cube center above target zone (5cm × 5cm) and >10cm above table for 1s
   - **Mimic subtasks (decorated on the env):**
     1. `approach_cube` — EE above cube, gripper open
     2. `grasp_cube` — gripper closed on cube, lifted 2cm
     3. `transport_to_target` — cube above target zone, >10cm above table
     4. `release` — gripper open, cube resting on target
   - Each subtask has a termination signal Claude Code defines (position thresholds on the cube+EE).

2. Create `envs/yaskawa_pick_cube_cfg.py` with both relative-IK-visuomotor and absolute variants.

3. Register env with Isaac Lab's gymnasium registry: `Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0` and `-Mimic-v0` variants.

4. Write `scripts/validate/env_smoke.py` that runs the env with random actions for 50 episodes, asserts env resets cleanly, no NaN observations, no physics explosions.

5. Pytest: `tests/test_env.py`.

**Validation tests:**
- [ ] `just env-smoke` (new justfile recipe) runs 50 random-action episodes without errors
- [ ] Cameras return valid images (check shape + not all-zero)
- [ ] Success detector fires correctly on a scripted successful trajectory

**Exit gate:** Env is stable, observation space well-defined, success criteria verified.

---

## Phase 6 — Keyboard Teleop System (DCV)

**Entry gate:** Phase 5 complete.

**Objective:** Ergonomic keyboard teleop via the DCV session user is already connected to. No browser, no WebRTC, no extra ports. Isaac Sim renders into the DCV virtual display; keyboard events flow through DCV directly.

**Steps:**

1. Confirm Isaac Sim can launch with a visible window on the DCV display:
   - Use the DISPLAY value captured in Phase 0 (typically `:0`).
   - Sanity-check: `DISPLAY=:0 glxinfo | grep "OpenGL renderer"` should report the NVIDIA GPU (not `llvmpipe`). If it reports llvmpipe, the DCV session isn't using the GPU — STOP and ask user to check DCV's dcv-gl configuration.
   - Launch the Isaac Lab teleop script with `DISPLAY=:0` and **without** `--headless`. The Isaac Sim window will appear on the user's DCV client.

2. Write `scripts/teleop/keyboard_teleop.py` extending Isaac Lab's `Se3Keyboard` with enhanced keymap. The `Se3Keyboard` class uses Isaac Sim's `carb.input` system, which receives keystrokes from the active window — DCV forwards the user's laptop keystrokes directly to that window. Keymap:

```
=== MOVEMENT (left hand on WASD + QE) ===
  W / S          : +Y / -Y   (forward/back in robot frame)
  A / D          : -X / +X   (left/right)
  Q / E          : +Z / -Z   (up/down)
  Hold SHIFT     : 5x speed (coarse travel)
  Hold ALT       : 0.2x speed (fine positioning, use near grasp)

=== ROTATION (right hand) ===
  I / K          : pitch + / -
  J / L          : yaw + / -
  U / O          : roll + / -

=== GRIPPER ===
  SPACE          : toggle gripper open/close (single press, not hold)

=== EPISODE CONTROL ===
  ENTER          : mark current episode SUCCESS + save + reset
  BACKSPACE      : discard current episode + reset
  R              : reset without saving (use if you flub mid-episode)
  P              : pause / unpause simulation
  ESC            : quit teleop session (saves any completed demos)

=== HUD (top-left overlay drawn with omni.ui) ===
  Episode #      : current count / target
  State          : RECORDING / RESETTING / PAUSED
  EE pose        : live xyz (mm) + rpy (deg)
  Gripper        : OPEN / CLOSED
  Subtask hint   : detected phase (approach / grasp / transport / release)
  Elapsed        : episode time (s)
```

3. Implement the HUD as an on-screen overlay using Isaac Sim's built-in UI (`omni.ui`). A draggable, always-on-top overlay panel is fine. If `omni.ui` turns out to be flaky in the specific Isaac Sim version, fall back to a separate terminal window running `scripts/teleop/hud_terminal.py` that reads live state from a JSON file the teleop script writes each step, rendered with `rich.live.Live`.

4. Write `scripts/teleop/record_demos.py` — wraps Isaac Lab's `record_demos.py` with:
   - Target N demos (configurable, default 15)
   - Per-episode validation: joint limits respected, no NaN, episode length 5–30s
   - Auto-save every episode to `datasets/teleop/cube_raw.hdf5`
   - Backup after every 5 episodes to `datasets/teleop/cube_raw.hdf5.backup.<timestamp>`
   - Print success rate and running count after each episode to the terminal (visible in the user's DCV session)

5. Write `scripts/validate/teleop_dry_run.py` — invokes the teleop system in a headless mode, programmatically injects 3 fake episodes via a scripted IK trajectory, confirms HDF5 is well-formed, observations recorded correctly, camera frames non-zero. This runs without DCV and proves the recording pipeline end-to-end.

6. Add a replay helper: `scripts/validate/replay_demos.py <path>` that plays back a recorded HDF5 dataset in sim and reports any replay failures.

7. Add two justfile recipes:
   ```make
   teleop:
       DISPLAY=:0 ./isaaclab.sh -p scripts/teleop/record_demos.py \
           --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 \
           --teleop_device keyboard \
           --dataset_file datasets/teleop/cube_raw.hdf5 \
           --num_demos 15

   validate-demos:
       python scripts/validate/replay_demos.py datasets/teleop/cube_raw.hdf5
   ```

**Validation tests:**
- [ ] `DISPLAY=:0 glxinfo | grep "OpenGL renderer"` shows NVIDIA GPU (not llvmpipe)
- [ ] Dry-run generates 3 synthetic episodes, all replay successfully
- [ ] HDF5 schema matches LeRobot-compatible expectations

**Exit gate:** Teleop system ready. Dry-run passes. Instructions block prepared for the user.

---

## Phase 7 — Scripted Grasp Sanity Test (pre-human)

**Entry gate:** Phase 6 complete.

**Objective:** Before asking the human to teleop, prove that a successful demo is achievable by the system. This catches 90% of bugs that would otherwise waste the user's time.

**Steps:**

1. Write `scripts/validate/scripted_pick_demo.py`:
   - Uses privileged state (cube pose) to script a perfect pick-and-place trajectory
   - Saves result in the SAME HDF5 format as real teleop demos
   - Generates 5 scripted demos with randomized cube positions
   - Saves to `datasets/teleop/cube_scripted.hdf5`

2. Run the replay validator on these scripted demos.

3. Visualize by generating a video montage: `scripts/validate/dataset_video.py` produces `reports/phase_7_scripted_demos.mp4`.

**Validation tests:**
- [ ] 5/5 scripted demos succeed per the env's success criteria
- [ ] All 5 replay correctly
- [ ] Video renders with no visual artifacts

**Exit gate:** System-level correctness proven. Any failure here is a blocker — the user should not burn time teleoping a broken system.

---

## 🛑 HUMAN GATE — Collect 10–15 Teleop Demos

**Instructions to print to user (exact block):**

```
================================================================
READY FOR TELEOP. Your input needed (~30–60 minutes).
================================================================

1. Make sure you're connected to the EC2 instance via the DCV client.
   You should see the Ubuntu desktop on the EC2 virtual display.

2. Open a terminal inside your DCV session and run:
   cd <repo-path>
   just teleop

3. An Isaac Sim window will open showing the HC10DT + Robotiq gripper
   + a cube on a table. Click on the Isaac Sim window to give it
   keyboard focus (important — without focus, keys won't reach the sim).

4. Record 15 successful cube pick demos.
   - Vary cube starting position across the 20×20cm zone
   - Vary approach angle slightly
   - Keep motions smooth — the policy will clone your jerkiness
   - Press ENTER after each success, BACKSPACE to discard failures
   - Use ALT (fine) when near the cube; SHIFT (coarse) for travel

5. Key controls reminder:
   WASD = XY, QE = up/down, IJKL/UO = rotation
   SPACE = toggle gripper, ENTER = save+next, ESC = quit

6. When done, verify dataset in the same terminal:
   just validate-demos

7. To resume this plan, re-invoke Claude Code and say:
   "continue from Phase 8"

Demos saved to: datasets/teleop/cube_raw.hdf5
================================================================
```

**Exit gate (after user returns):**
- [ ] `datasets/teleop/cube_raw.hdf5` exists
- [ ] Contains ≥10 episodes (target 15)
- [ ] All episodes replay successfully
- [ ] Success rate visible in `reports/teleop_summary.md` auto-generated by `just validate-demos`

If <10 demos or replay failures: report to user, ask them to collect more or fix issues before proceeding.

---

## Phase 8 — Dataset Validation & Mimic Annotation

**Entry gate:** Human gate cleared. Teleop dataset exists.

**Objective:** Validate the human-collected demos and annotate them with subtask boundaries so Mimic can multiply them.

**Steps:**

1. Write `scripts/data/inspect_demos.py` — produces `reports/dataset_inspection.md` with:
   - Episode count, mean/median/min/max length
   - Action distribution histograms (per dim)
   - Observation sanity: camera RGB mean/std per episode (detect black frames)
   - Trajectory plots (EE path overlay, one PNG per episode)
   - Flag any episodes with outlier length (>2σ) for review

2. Run Isaac Lab's automatic subtask annotation:
   ```bash
   ./isaaclab.sh -p scripts/tools/annotate_demos.py \
       --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
       --input_file datasets/teleop/cube_raw.hdf5 \
       --output_file datasets/teleop/cube_annotated.hdf5
   ```

3. Validate annotations: `scripts/validate/check_annotations.py` — for each episode, confirm all 4 subtask boundaries are present and in correct order. If algorithmic annotation fails on any episode, log which one and exclude it (don't block Mimic on one bad demo).

**Validation tests:**
- [ ] Inspection report generated
- [ ] ≥90% of demos successfully annotated
- [ ] Subtask order is monotonic in every annotated demo

**Exit gate:** Clean annotated dataset ready for Mimic.

---

## Phase 9 — Mimic Data Generation

**Entry gate:** Phase 8 complete.

**Objective:** Generate 500–1000 synthetic demos from the ~12 annotated human demos.

**Steps:**

1. Configure Mimic data gen in `configs/mimic_cube.yaml`:
   - Target: 750 demos
   - Randomize cube position over 25×25cm (slightly wider than teleop range for generalization)
   - Randomize cube color (red/green/blue/yellow)
   - Keep 4 subtasks with interpolation between
   - Enable image observations (critical — visuomotor policy)

2. Run generation:
   ```bash
   ./isaaclab.sh -p scripts/tools/generate_dataset.py \
       --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
       --input_file datasets/teleop/cube_annotated.hdf5 \
       --output_file datasets/mimic/cube_mimic_750.hdf5 \
       --num_envs 16 \
       --generation_num_trials 750 \
       --enable_cameras
   ```
   This is GPU-parallelized across num_envs. Expected runtime on g5.2xlarge: 2–4 hours.

3. Monitor generation in a tmux session: `just mimic-monitor` (tails log + watches nvtop).

4. After generation, run the replay validator on a random sample of 50 generated demos to confirm quality.

5. Produce `reports/mimic_summary.md`: generated count, success rate from Mimic internal metric, replay success rate, average episode length.

**Validation tests:**
- [ ] ≥500 successful generated demos (target 750, allow up to 33% internal Mimic failure)
- [ ] 50-demo replay validation: ≥45 succeed
- [ ] Episode length distribution reasonable (not all too short or too long)

**Exit gate:** Large, validated synthetic dataset ready.

---

## Phase 10 — LeRobot Format Conversion

**Entry gate:** Phase 9 complete.

**Objective:** Convert Isaac Lab HDF5 to LeRobot v3.0 dataset format for SmolVLA.

**Steps:**

1. Install LeRobot:
   ```bash
   uv pip install "lerobot[smolvla]" huggingface_hub
   ```

2. Write `scripts/data/isaaclab_to_lerobot.py` adapting existing open-source converters (the LeIsaac project is a reference). Must handle:
   - Joint state → `observation.state`
   - EE pose → `observation.ee_pose`
   - Two camera streams → `observation.images.wrist` and `observation.images.third_person`
   - Action (EE delta + gripper) → `action`
   - Task description → `"pick up the cube and place it on the target"`
   - Per-episode metadata + frame-level timestamps
   - Compute normalization stats (IMPORTANT: use stats from OUR dataset, not SmolVLA's pretraining stats)

3. Convert:
   ```bash
   python scripts/data/isaaclab_to_lerobot.py \
       --input datasets/mimic/cube_mimic_750.hdf5 \
       --output datasets/lerobot/cube_pick_750 \
       --fps 30 \
       --task "pick up the cube and place it on the target"
   ```

4. Validate with LeRobot's built-in tools:
   ```bash
   python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
              ds = LeRobotDataset('datasets/lerobot/cube_pick_750'); \
              print(ds); print(ds[0].keys()); print(ds[0]['observation.state'].shape)"
   ```

5. **Optional HF Hub push.** Ask the user inline (this is NOT the full-stop human gate — it's a one-line yes/no with a short timeout and a safe default):

   ```
   === OPTIONAL: Push dataset to HuggingFace Hub? ===
   Dataset is ready locally at datasets/lerobot/cube_pick_750/

   Pushing to HF gives you:
     - Easy sharing with teammates
     - A backup outside EC2
     - LeRobot ecosystem compatibility

   To push, reply with either:
     - an HF token (starts with 'hf_')
     - a path to a file containing the token
     - the username to upload under, if already logged in (`huggingface-cli whoami`)

   To skip, reply 'skip'. (You can always push later — not a blocker.)
   Auto-skipping in 60s if no reply.
   ```

   If user provides credentials, run:
   ```bash
   huggingface-cli upload <username>/yaskawa-cube-pick-v1 datasets/lerobot/cube_pick_750
   ```
   If user says skip or times out, log the decision and move on.

**Validation tests:**
- [ ] LeRobotDataset loads without errors
- [ ] First sample has all expected keys and correct shapes
- [ ] Stats file exists and has finite values in all dims
- [ ] Sample a random episode and visualize: save `reports/phase_10_sample_episode.mp4`

**Exit gate:** LeRobot dataset is valid.

---

## Phase 11 — SmolVLA LoRA Fine-tune

**Entry gate:** Phase 10 complete. GPU has ≥16GB VRAM.

**Objective:** Fine-tune SmolVLA-450M on the cube-pick dataset using LoRA.

**Steps:**

1. Download SmolVLA base:
   ```bash
   huggingface-cli download lerobot/smolvla_base --local-dir checkpoints/smolvla_base
   ```

2. Configure training in `configs/train_smolvla.yaml`:
   - Base: `lerobot/smolvla_base`
   - Dataset: `datasets/lerobot/cube_pick_750`
   - PEFT: LoRA, rank 32, alpha 64, target_modules=all-linear
   - Batch size: 16 (g5.2xlarge / A10G). Drop to 8 if OOM.
   - LR: 1e-4 cosine schedule, warmup 500 steps
   - Steps: 20,000 (~4 hrs on A10G, ~2 hrs on A100)
   - AMP: bf16
   - Eval every 1000 steps: run 10 closed-loop rollouts in sim, compute success rate
   - Save every 2000 steps, keep best-by-eval-success

3. Launch in tmux so it survives disconnect:
   ```bash
   tmux new -d -s train "just train 2>&1 | tee logs/phase_11_train.log"
   ```

4. Monitoring:
   - `just train-monitor` tails log + shows nvtop
   - If W&B configured, log metrics there; otherwise plot loss curve from log file every 1000 steps

5. After training, identify best checkpoint by eval success rate.

**Validation tests:**
- [ ] Training loss trending down, plateaus in second half
- [ ] No NaN losses
- [ ] Final eval success rate at some checkpoint ≥50% (preliminary — full eval in Phase 12)
- [ ] GPU memory usage logged, no OOM

**Failure modes:**
- Immediate OOM → drop batch size to 8 then 4, reduce `num_images_in_input` if possible
- Loss explodes → LR too high, rerun with 5e-5
- Success rate stuck at 0 → check dataset stats are from OUR data (not SmolVLA defaults); check action unnormalization key matches

**Exit gate:** Best checkpoint saved with eval success ≥50%.

---

## Phase 12 — Full Sim Evaluation

**Entry gate:** Phase 11 complete.

**Objective:** Rigorously evaluate the best policy. This is the number that proves the pipeline works.

**Steps:**

1. Write `scripts/train/evaluate.py`:
   - Loads best SmolVLA-LoRA checkpoint
   - Runs 50 closed-loop rollouts in the cube-pick env with randomized cube positions from BOTH the train distribution AND a 1.5× expanded distribution (held-out)
   - Records success, episode length, failure mode (no-grasp / dropped / wrong-place)
   - Saves rollout videos for 10 random episodes (5 success + 5 failure) to `reports/rollouts/`

2. Generate final report `reports/FINAL_REPORT.md`:
   - Pipeline stats (time per phase, data volumes)
   - Dataset summary (teleop count, mimic count)
   - Training curves (loss, eval success over steps)
   - Eval results: in-dist success rate, held-out success rate, failure mode breakdown
   - Key learnings and recommendations for next iteration (apply to real SKUs)

**Validation tests:**
- [ ] 50 rollouts completed
- [ ] In-distribution success rate ≥70% (target)
- [ ] Held-out success rate ≥50%
- [ ] Videos render correctly

**Exit gate:** V1 success criterion met OR gap to target clearly characterized with actionable recommendations.

---

## Phase 13 — Handoff

**Steps:**

1. Commit everything to `vla-pipeline-v1` branch. Push if remote configured.
2. Print final summary to terminal with key metrics.
3. Surface `reports/FINAL_REPORT.md` as the definitive artifact.
4. Suggest next steps: (a) scale to real SKUs using same pipeline, (b) if success rate low, recommend specific improvements (more demos, Cosmos augmentation, π0.5 upgrade).

**Done.**

---

## Efficiency tooling summary (Phase 1 installs these)

| Tool | Why |
|------|-----|
| `uv` | 10-100x faster pip, essential for iteration |
| `ripgrep` | Fast repo search for debugging |
| `fd` | Fast file discovery |
| `jq` | Parse JSON manifests / logs |
| `just` | Task runner, makes every step a one-liner |
| `tmux` | Long-running jobs survive disconnect |
| `nvtop` | GPU monitoring during training / mimic |
| `ruff` | Lint before committing |
| `pytest` | Testing framework |
| `rich` | Readable terminal output for HUD and reports |

## Testing strategy

- **Per-phase validation script** in `scripts/validate/phase_<N>_*.py` — runs within that phase.
- **Pytest suite** in `tests/` — run via `just test-all` before any major gate.
- **Integration smoke test** `tests/test_e2e_smoke.py` — runs a minimal version of Phases 5-11 on 1 teleop demo (scripted) + 10 Mimic demos + 500 training steps. Used to validate plan changes without full runs.
- **Replay validation** after every data-producing phase — catches data format bugs early.
- **Video artifacts** from every meaningful phase — visual inspection is faster than reading logs for physics issues.

## Ask-the-user triggers (hard rules)

You ask the user for input ONLY in these cases:
1. Phase 0 clarifications (once, upfront)
2. Human gate (teleop demos) — full stop, user must re-invoke
3. Phase 10 optional HF push — inline prompt with 60s timeout and "skip" default (NOT a full stop)
4. Any validation test fails 3 times in a row with no clear fix
5. Destructive operation (deleting files, rewriting main branch) — ALWAYS ask
6. Cost-implicating operation (e.g., spinning up a bigger EC2 instance, pushing 10GB to HF) — ALWAYS ask

Otherwise, you self-repair and proceed. Log every self-repair attempt.
