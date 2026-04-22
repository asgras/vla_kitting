# Known issues (running list)

## Wrist camera still mostly black in demo frames (as of 2026-04-22)

**Observed:** in `reports/mimic_gifs/mimic_{0,4,9}.gif`, the left panel (wrist
cam) is mostly black throughout the demo. Only the bright finger-pad strips
appear at the edges; there's no visible cube / target / table between them.

**Current config** (envs/yaskawa_pick_cube_cfg.py):
```
wrist_cam offset: pos=(0.0, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0) ROS
focal_length=14.0, clipping_range=(0.02, 3.0)
```

Camera is parented under `/Robot/root_joint/tool0/wrist_cam`. The previous
rot `(0, 0.707, 0.707, 0)` was worse — it was looking along tool0's -Z
(upward at the ceiling during top-down grasp). Current identity rotation
at least shows the gripper pads, but still misses the table.

**Likely causes to investigate later:**
1. tool0 frame's +Z might not actually align with "away from the flange"
   for our URDF-converted HC10DT. Depending on the URDF orientation, tool0's
   +Z could point toward the flange body, meaning identity rotation has the
   camera looking INTO the robot arm.
2. The camera might be mounted *between* the pads and the workspace is
   outside clipping (far plane 3m is fine; near plane 0.02m = 2cm might be
   clipping the pad interior during close grasp).
3. Mount position (0, 0, 0.04) might be inside the robotiq_arg2f_base_link
   collision mesh; rendering might be culling because the camera is inside
   an occluding body.

**Workaround for now:** ignore and proceed with training. The third-person
camera is sufficient for the VLA to see the scene; the wrist camera is a
nice-to-have that can be fixed post-hoc without affecting the overall
pipeline.

**To revisit:** after the end-to-end training pipeline works, tune the
wrist offset by iteratively saving a frame during phase 3 (descent) of the
scripted demo, not at home pose. The home pose shows empty ceiling because
the arm hasn't moved to the workspace yet.
