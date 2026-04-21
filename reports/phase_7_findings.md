# Phase 7 scripted pick demo — findings and open issues

Status: **still partial**. Many iteration rounds have eliminated the earlier
failure modes — the cube now stays perfectly aligned between the fingers
through approach, descent, and close — but the gripper never holds the cube
firmly enough to LIFT it off the table. The close command reaches contact
angle and the fingers stall there, yet when the arm rises the cube remains
at z = 0.025 (on the table). Believed to be a Robotiq-pad contact-geometry
or mimic-chain-coupling issue in the PhysX simulation.

## Bugs found and fixed across all iterations

(Listed in order they were identified; all are in the committed code.)

1. **Cube randomization range was delta-style, not absolute**, so `x: (0.45, 0.65)`
   on a default 0.55 landed the cube at x ≈ 1.0 — outside the arm envelope.
   Fixed to deltas: `x: (-0.08, 0.08)` etc.
2. **IK action scale too small** — `scale=0.05` × controller 0.03 clamp gave
   ~1.5 mm per step. Raised to `scale=0.1` + raw controller clamp ±1.
3. **Gripper action semantics inverted.** Isaac Lab's `BinaryJointPositionAction`
   treats `actions < 0` as CLOSE. The scripted demo had them reversed.
4. **Cube yaw randomization ±0.5 rad (±28°) rotated the cube corners outside
   the gripper's straight-face expectation**, causing finger knuckles to
   clip the cube during descent and bump it 5–20 mm sideways. Yaw range
   pinned to 0 for Phase 7 scripted pick. (Re-enable once the scripted
   controller reads `cube_rot` and aligns gripper yaw.)
5. **Gripper drive stiffness 500 / damping 10 was too weak to hold the
   commanded OPEN target during rapid arm motion.** Finger_joint drifted
   to ~0.48 rad under inertial loading even with gripper commanded OPEN,
   which pre-closed the pads and scooped the cube during descent. Raised
   to stiffness 5000 / damping 100 — now finger_q stays at ~0.07 rad when
   commanded open. Cube completely undisturbed through approach + descent.
6. **Gripper drive effort cap 200 N·m was too aggressive on close** — the
   huge impulsive closing torque kicked the 50 g cube out of the pads
   before contact friction could stabilize. Dropped to 50 N·m (realistic
   continuous 2F-85 grasp force at the knuckle: ~14 N·m).
7. **Close command target 0.79 rad was too far past contact.** For a 50 mm
   cube, contact occurs near finger_q ≈ 0.50 rad. A target of 0.79 means
   that even after the pads contact the cube, the PD drive applies max
   effort trying to close further — continuously pushing the cube out of
   one pad due to the mimic-chain asymmetry. Dropped to 0.65 (a touch past
   contact, so grip is firm but not overshooting aggressively).
8. **Scripted demo used only the initial cube pose.** Tiny nudges during
   descent moved the cube out of the EE's target, so close happened at a
   stale cube XY. Added per-step cube-XY tracking for phases 0–4
   (approach, hover, descent, close, lift). With tracking active, the
   cube stays within ~2 mm of EE XY through the entire pick.

## Remaining open issue

**The cube will not lift, even when it is visibly centered between closed
pads at the end of phase 3.** Across every configuration tried:

- Phase 3 midway: cube at its init position ±1–2 mm, EE centered on it,
  `grip_closed=1.0` (finger_q > 0.4 threshold).
- Phase 4 lift: EE rises to z = 0.318 but cube stays at z = 0.025 (on
  table). finger_q continues closing from ~0.47 → ~0.67 rad (i.e. fingers
  close past the cube instead of grabbing it).

What this tells us: by phase 4 start, the fingers HAVE reached contact
angle (finger_q=0.47 is just past the 0.50 rad contact point for a 50 mm
cube), but the contact is evidently not solid enough to lift the cube.
Then during lift the fingers continue closing (finger_q→0.67), which is
physically impossible with a 50 mm cube between them — so the cube must
have escaped the pads before the actual lift motion begins.

## Hypotheses still untried

1. **Pad collision geometry may differ from pad visual geometry.** The
   `left_inner_finger_pad` body shows body_pos_w at a specific location,
   but the actual collision box PhysX uses could be smaller / offset. The
   USD would need inspection with a stage viewer.
2. **Switch to the NVIDIA Robotiq 2F-85 USD** (`assets/hc10dt_with_nvidia_gripper.usd`,
   2.3 KB shell; the payloads are in `assets/nvidia_robotiq_2f85/`). This
   is the canonical NVIDIA-packaged Robotiq with tuned contact materials.
   The current setup uses a custom ros-industrial-attic URDF converted
   via our own `urdf_to_usd.py`.
3. **Increase PhysX articulation solver iterations** — current uses Isaac
   Lab defaults. The Robotiq mimic chain has 5 dependent joints that all
   need to converge per step, plus pad-cube contact constraints. Try
   bumping `solver_position_iteration_count=32` and `velocity=4` on the
   articulation cfg (currently only set on the cube).
4. **Contact-offset / rest-offset on the cube collision.** Current cfg
   uses defaults; experiments with small offsets sometimes fix
   "grip slides off" issues.
5. **`enable_external_forces_every_iteration=True` on PhysxCfg** — sim
   logged a warning about this being off.

## P1 (NVIDIA Robotiq USD) — tried and ruled out

Loaded `assets/hc10dt_with_nvidia_gripper.usd` (a runtime-reference
composition of the arm USD + NVIDIA's Robotiq 2F-85 USD built by
`scripts/assembly/compose_arm_with_nvidia_gripper.py`). Immediately hit two
fatal PhysX errors on env creation:

- `Rigid Body of (/.../tool0/gripper/Robotiq_2F_85/base_link) missing
  xformstack reset when child of another enabled rigid body
  (/.../tool0)`: the gripper's base_link RigidBodyAPI is physically
  parented under tool0, which also has RigidBodyAPI. Nested rigid
  bodies are undefined in PhysX.
- `PhysicsUSD: CreateJoint - no bodies defined at body0 and body1`:
  every gripper joint's body0/body1 reference uses absolute paths that
  weren't remapped into the composed scope. `finger_joint`,
  `right_outer_knuckle_joint`, the four mimic rotX joints — all broken.

Result: articulation init failed; the gripper's six joints weren't part
of any articulation, so Isaac Lab's `ImplicitActuatorCfg` regexes didn't
match and training aborted before a single physics step.

Investigation showed Isaac Lab's canonical `UR10e_ROBOTIQ_2F_85_CFG` in
`IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py`
combines arm + gripper by selecting a **USD variant** baked into the arm
USD (`spawn.variants = {"Gripper": "Robotiq_2f_85"}`), NOT by a runtime
USD reference. That's a fundamentally different composition pattern than
our URDF → USD → reference pipeline and would require authoring a new
HC10DT USD with a gripper variant — out of scope for a quick tactical
fix. Reverted to the RIA URDF-derived gripper USD.

Silver lining: while reading the UR10e config, found canonical drive
gains very different from ours (stiffness 11.25, damping 0.1, effort 10
vs our 5000/100/50). Tried the canonical values with our asset: fingers
immediately drifted to finger_q=0.78 rad under OPEN command during the
arm's rapid phase-0 swing. Tried intermediate 400/8/20: fingers still
drifted to 0.53. So Isaac Lab's canonical gains don't transfer to our
asset (different mass properties in the URDF-derived USD). Kept
5000/100/50 as the only combination that holds OPEN cleanly.

## Recommendation

- **(P2 — now the primary path) Human teleop.** V1's plan was always
  teleop → Mimic → train. Teleop bypasses the scripted alignment
  problem entirely because a human can nudge the EE so the pads make
  solid contact. Blocker is the DCV viewport freeze from
  `reports/phase_6_viewport_handoff.md` — has concrete research
  candidates (warp symlink, kit-file edits) that are worth pursuing.
- **(P3) Author a HC10DT USD with a Gripper variant** (the Isaac Lab
  way). Bigger asset-authoring job but would let us reuse NVIDIA's
  canonical Robotiq. Would fix the composition, but doesn't guarantee
  the lift works — still might need P5 on top.
- **(P4) Rewrite the compose script to flatten the stage** and fix the
  xformstack issue. Less work than P3 but the xformstack-reset problem
  with nested rigid bodies is nontrivial — PhysX really doesn't like
  the tool0 → gripper/base_link parenting pattern even with flattening.
- **(P5) Inspect the RIA gripper USD's pad collision geometry** by
  opening it in the Isaac Sim viewer (once the viewport is unblocked).
  If pad-cube contact regions don't look right, we can hand-edit the
  collision meshes or use a convex hull approximation.
