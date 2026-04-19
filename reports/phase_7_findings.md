# Phase 7 scripted pick demo — findings and open issues

Status: **partial success** — the scripted demo runs end-to-end (env stepping,
recorder manager, HDF5 output, IK control, cube randomization), but the scripted
pick-and-place physics doesn't yet produce a success. The script exposed several
real bugs that would have broken human teleop too — fixing these before teleop
was the explicit purpose of Phase 7.

## Bugs found and fixed

1. **Cube randomization was out of workspace.** The `reset_root_state_uniform`
   event adds the range to the default pose. My original range `x: (0.45, 0.65)`
   on a default of 0.55 put the cube at x = 1.0-1.2 m — outside the HC10DT's
   practical reach envelope.
   - Fixed to delta-style range: `x: (-0.08, 0.08)`, cube stays around (0.55, 0).

2. **IK action scale was too small for practical motion.** With `scale=0.05`
   and my controller's 0.03 action clamp, effective EE motion was 0.0015 m per
   control step — too slow for the 30cm transport in the allotted step budget.
   - Fixed to `scale=0.1` + controller feeds raw `pos_err * 10` (clamped to ±1),
     giving ~1 cm per step effective motion.

3. **Gripper action semantics were inverted.** Isaac Lab's
   `BinaryJointPositionAction.process_actions` treats `actions < 0` as CLOSE
   and `actions >= 0` as OPEN. I had written the scripted demo assuming the
   opposite, so the gripper was opening during the grasp phase and closing
   during approach.
   - Fixed in `scripts/validate/scripted_pick_demo.py`.

4. **Grasp height needs tuning.** The tool0 frame (wrist) is ~11-14cm above
   the Robotiq fingertips in our URDF, but the exact offset depends on finger
   pose. With `grasp_h=0.20` (tool0 at z=0.20), fingertips are roughly at
   z=0.06, above the cube top (z=0.05) — closing around air.
   - Current best estimate: `grasp_h ~ 0.14` (fingertips near cube center 0.025).

## Open issue blocking scripted success

Even with the gripper semantics fixed, the cube is not being picked up reliably:
tested `grasp_h = 0.12`, `0.14`, `0.16`, `0.20`. At z=0.12 the gripper doesn't
seem to close (knuckle stays open) — likely the fingers are jamming against the
table surface at that height. At higher grasp_h, the fingers close but around air.

Hypotheses to investigate:
- **Finger-table collision interference.** Fingertip position when gripper is
  open is likely below z=0.05 — fingers hit the table before reaching the cube.
  Try lifting the table slightly or building the table with a proper collision
  mesh.
- **Robotiq drive gains too weak to overcome contact.** Gripper stiffness=500
  may not be enough to close around a 0.9-friction cube against the table.
  Could be increased to 2000+.
- **Mimic joint limits are infinite.** Phase 4's smoke test warns
  `robotiq_85_left_inner_knuckle_joint needs a finite limit set to be used by
  the mimic joint feature`. PhysX may not apply mimic constraints reliably.
  Patch URDF to give the continuous joints finite limits.

## Recommendation

Before the teleop human gate, the user should either:
- **(a) Manually teleop once in DCV** (`just teleop display=1 num=1`) and visually
  diagnose the gripper/grasp geometry, then adjust as needed. Human vision makes
  this much faster than scripted debugging.
- **(b) Invest 1-2 hours tuning the scripted demo** by trying the hypotheses
  above one by one.

Given the purpose of V1 is to validate the pipeline (teleop → Mimic → train),
and human teleop is already part of that pipeline, (a) is probably the right
call. If teleop produces successful demos, Mimic then amplifies those.
