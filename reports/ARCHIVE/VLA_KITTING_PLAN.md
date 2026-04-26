# VLA-Based Kitting Workcell: Architecture & Implementation Plan

## 1. Executive Summary

Replace the traditional robotics stack (MoveIt2 + perception pipeline + state machine) with a
Vision-Language-Action (VLA) model that directly maps camera images + language instructions to
robot joint actions. The goal is a small, fine-tunable model that learns the entire pick-and-place
behavior end-to-end, eliminating the need for:

- Point cloud processing and clustering
- MoveIt2 motion planning
- Collision scene management
- Hand-coded state machines
- Grasp width verification logic
- Per-SKU shape priors and parameters

**Scope**: Large boxes only (single_gang_box: 100x55x55mm, round_box: 70x70x55mm). No small
fittings (pex_elbow excluded). One input bin, one output tray, one box type at a time initially.

---

## 2. What We Keep From the Current Repo

### Hardware & Scene (reuse as-is)
| Component | Spec | Notes |
|-----------|------|-------|
| Robot | Motoman HC10DT (6-DOF) | Same URDF, same ros2_control |
| Gripper | Schunk EGK-40 (parallel, 0-83mm) | Same controller interface |
| Camera | Zivid 2+ (fixed overhead) or Isaac Sim camera | RGB image only (no point cloud) |
| Table | 1.4x1.8m at z=-0.01 | Same physical setup |
| Input bin | 350x350x100mm at [0.75, 0.0, 0.05] | Single bin, center position |
| Output tray | 350x350mm at [0.0, -0.75, 0.02] | Same 3x3 slot grid |
| Boxes | single_gang_box (100x55x55mm) or round_box (70x70x55mm) | One type at a time |

### ROS2 Infrastructure (reuse)
- `ros2_control` with mock or Isaac hardware plugins
- `robot_state_publisher` for TF
- Controller spawner (joint_trajectory_controller + gripper_controller)
- Launch file structure (`demo.launch.py` for hardware bringup)

### Isaac Sim (reuse for data collection)
- Scene setup with arm + gripper + bins + parts
- Topic-based ros2_control bridge
- Camera publishing RGB to `/sim/image_raw`
- Existing `hardware_type:=isaac` launch arg

### What We Throw Away
- MoveIt2 entirely (move_group, planning scene, collision objects, OMPL)
- `kitting_perception` package (point cloud pipeline, DBSCAN, shape priors)
- `kitting_task_planner` state machine, retry logic, collision recovery
- Per-SKU grasp parameters (grasp_width, grasp_depth, approach_offset)
- `planning_scene_setup.py` (no collision objects needed)
- All cartesian path planning logic

---

## 3. VLA Architecture Selection

### Recommended: Octo-Small (27M params) — Full Fine-Tune

| Property | Value |
|----------|-------|
| Base model | Octo-Small (27M params) |
| Vision encoder | Small ViT (pretrained on OXE dataset) |
| Language conditioning | FiLM-conditioned task embedding (lightweight, no LLM) |
| Action head | Diffusion head — predicts action chunks (sequence of future actions) |
| Fine-tuning | Full fine-tune (all 27M params), no LoRA needed at this size |
| Hardware req | 1x RTX 3090 24GB or 1x RTX 4070 Ti 12GB |
| Training time | ~4-6 hours for 10K episodes on RTX 3090 |
| Inference | ~15-25 Hz on RTX 3090 (fast enough for real-time 10 Hz control) |
| Deployed size | ~120MB (fp32), ~60MB (fp16) |

**Why Octo-Small:**
- **27M params** fits comfortably on consumer GPUs with full fine-tuning (no quantization tricks)
- **Diffusion action head** with action chunking produces smoother trajectories than single-step prediction
- **Pretrained on Open X-Embodiment** (800K+ real robot episodes) — already knows basic manipulation primitives
- **Jax/Flax** codebase with clean fine-tuning API (`octo.utils.train_utils`)
- **~$0 training cost** on own hardware, ~$5-10 on cloud GPU
- Fast iteration: 4-hour training cycle means you can run multiple experiments per day

**Why NOT the larger alternatives:**
- **OpenVLA (7B)**: Overkill for a fixed scene with 1-2 box types. The language understanding is wasted when prompts are simple templates. Needs quantization tricks to fit on consumer GPUs.
- **pi0 (3B)**: Strong model but heavier tooling requirements, less community fine-tuning examples
- **RT-2-X**: Not publicly available for fine-tuning

**Scaling path**: If Octo-Small plateaus at <80% success, upgrade to Octo-Base (93M) on the
same hardware before considering a 7B model. The training pipeline is identical.

### Octo Architecture Overview

```
                    256x256 RGB Image
                          |
                    [Small ViT Encoder]
                          |
                   visual tokens (patch embeddings)
                          |
     "Pick box, place slot 3" ---> [Language Tokenizer + FiLM]
                          |
                   [Transformer Backbone]
                   (cross-attention over visual + language tokens)
                          |
                   [Diffusion Action Head]
                   (iterative denoising, K=4 steps)
                          |
              Action Chunk: (H, 7) float32
              H = prediction horizon (e.g., 4 steps ahead)
              7 = [dj1..dj6, gripper_cmd]
```

**Action chunking** is a key Octo feature: instead of predicting one action per timestep,
the model predicts the next H actions at once. This produces smoother trajectories and is
more robust to observation noise. At inference, we execute the first action, then re-predict
(receding horizon). Typical H=4 at 10 Hz = 0.4s lookahead.

### Model I/O Specification

**Input (per timestep):**
```
Image:       256x256 RGB (resized from camera native resolution)
Task:        Language string — "Pick the box and place it in slot 3"
             (encoded via Octo's built-in language tokenizer, no external LLM)
State:       [j1, j2, j3, j4, j5, j6, gripper_width]  (7 floats, current joint state)
History:     Last 2 observation frames (image + state) for temporal context
```

**Output (per inference call):**
```
Action Chunk: (H, 7) float32 where H = prediction horizon (default 4)
  Per step:   [dj1, dj2, dj3, dj4, dj5, dj6, gripper_cmd]
              - dj1..dj6: delta joint positions (rad), clipped to [-0.05, 0.05]
              - gripper_cmd: 0.0 = close, 1.0 = open (binary after threshold)
Execute:     First action from chunk, then re-predict at next timestep
```

**Action space**: Delta joint positions (not velocities) per the Octo convention. At 10 Hz
with max delta 0.05 rad/step, this gives ~0.5 rad/s effective velocity — safe and smooth.
The gripper is binary open/close since we're grasping large boxes where partial opening
isn't needed.

---

## 4. Simplified Task Definition

### The Task (Plain English)
> Pick one box from the input bin and place it in the next empty slot of the output tray.
> Repeat until the bin is empty or the tray is full.

### Language Prompts (Training Vocabulary)
```
"Pick up the box from the bin"
"Pick the box and place it in slot 0"
"Pick the box and place it in slot 4"
"Grab the box from the bin and put it in the tray"
"Pick the single gang box"
"Pick the round box"
```

Prompt variation during training improves robustness. At inference time, a simple
orchestrator node cycles through slots.

### Episode Structure
One episode = one pick-and-place cycle:

```
1. Start at HOME position (all joints zero except j5=90deg)
2. [VLA controls] Move to bin area, descend, grasp box
3. [VLA controls] Lift box, move to tray area, descend, release
4. [VLA controls] Return to HOME-ish position
5. Episode terminates on: box in slot OR timeout (60s) OR safety limit
```

**Success criteria**: Box centroid within 50mm of target slot XY, box resting on tray floor.

### Simplified Scene (vs. current repo)

| Current Repo | VLA Version |
|-------------|-------------|
| 3 input bins, 3 SKUs | 1 input bin, 1-2 SKUs |
| 3 parts per bin | 1-3 parts per bin |
| Complex collision management | No collision objects (VLA learns avoidance) |
| Shape-aware perception | Raw RGB image |
| 13-state state machine | Continuous VLA policy |
| Cartesian path planning | Direct joint control |
| Grasp width verification | VLA learns grasp feedback from training |

---

## 5. Data Collection Pipeline

### Phase 1: Scripted Demonstrations in Isaac Sim

Use the existing task planner as an expert policy to generate demonstrations automatically.
This is the key advantage of having the traditional stack: it becomes your data factory.

**Data collection node** (`vla_data_collector.py`):
```python
# Runs alongside the existing task planner stack in Isaac Sim
# Records synchronized (image, joint_state, action) tuples at 10 Hz

@dataclass
class Timestep:
    image: np.ndarray          # 256x256x3 uint8 (RGB from /sim/image_raw)
    joint_state: np.ndarray    # [j1..j6, gripper_width] float32
    action: np.ndarray         # [dj1..dj6, gripper_cmd] float32 (computed from consecutive states)
    language: str              # Task instruction
    episode_id: int
    timestep: int
    success: bool              # Episode outcome (labeled at end)
```

**Recording flow:**
1. Launch Isaac Sim + full ROS2 stack (existing `sim_full.launch.py`)
2. Subscribe to `/sim/image_raw` (RGB), `/joint_states`, `/kit_order`
3. Send kit orders programmatically (vary slot targets)
4. Record all timesteps at 10 Hz
5. Label episode success/failure from ExecuteKit result
6. Save raw to HDF5, then convert to RLDS (Octo's native format)

**Domain randomization (in Isaac Sim):**
- Box position: random XY within bin (uniform, +-100mm from center)
- Box orientation: random yaw (0-360deg)
- Lighting: vary intensity +-30%, direction +-20deg
- Camera noise: Gaussian RGB noise sigma=5
- Box color/texture: 3-4 variants per box type
- Table color: 2-3 variants

**Target: 10,000 successful episodes** (~50K timesteps at 10 Hz, ~5 seconds per pick-place)
- At ~45s per episode (current stack speed), this is ~125 hours of sim time
- With 4x parallel Isaac Sim instances: ~31 hours wall time
- Storage: ~50GB in HDF5 format

### Phase 2: Human Teleoperation (Optional, for Sim-to-Real)

For real hardware deployment, supplement with 200-500 real-world demonstrations via teleoperation:

**Setup:**
- SpaceMouse or gamepad for joint velocity control
- Same data recording node, but subscribed to real camera + real joint states
- Operator performs pick-and-place while data is recorded

**This closes the sim-to-real gap** that pure sim data can't bridge (lighting, physics
fidelity, gripper contact dynamics).

### Data Format: RLDS (Octo Native)

Octo consumes data in TensorFlow RLDS format. Collect raw data as HDF5, then convert.

```
# Raw collection format (HDF5, one file per episode)
raw_data/
  episode_000000.hdf5
    image: (T, 256, 256, 3) uint8
    state: (T, 7) float32       # [j1..j6, gripper]
    action: (T, 7) float32      # [dj1..dj6, gripper_cmd]
    language: str
    success: bool

# Converted RLDS format (what Octo reads)
kitting_demos/
  1.0.0/
    kitting_demos-train.tfrecord-00000-of-00008
    ...
    dataset_info.json
    features.json
```

**RLDS episode spec:**
```python
episode = {
    "steps": [{
        "observation": {
            "image_primary": tf.Tensor(256, 256, 3),   # uint8
            "proprio": tf.Tensor(7,),                    # float32
        },
        "action": tf.Tensor(7,),                         # float32
        "language_instruction": "Pick the box and place it in slot 3",
        "is_terminal": False,
    }, ...],
}
```

---

## 6. Training Pipeline

### Hardware Requirements

| Option | GPU | VRAM | Training Time (10K episodes) | Cost |
|--------|-----|------|------------------------------|------|
| **Octo-Small (recommended)** | 1x RTX 3090 | ~18GB | ~4 hours | Own hardware |
| Octo-Small | 1x RTX 4070 Ti | ~10GB | ~6 hours | Own hardware |
| Octo-Small | 1x A100 40GB (cloud) | ~18GB | ~2 hours | ~$3-5 |
| Octo-Base (upgrade path) | 1x RTX 3090 | ~22GB | ~8 hours | Own hardware |

### Training Configuration (Octo-Small)

```yaml
# train_config.yaml
model:
  base: "hf://rail-berkeley/octo-small-1.5"
  pretrained: true                # Start from OXE pretrained weights
  action_head: "diffusion"        # Diffusion action head (default)
  pred_horizon: 4                 # Predict 4 future actions per step
  action_dim: 7                   # 6 joints + gripper

data:
  dataset_path: "./data/kitting_demos"
  dataset_format: "rlds"          # TensorFlow RLDS format (Octo native)
  image_size: [256, 256]
  action_dim: 7
  state_dim: 7
  action_normalization: "normal"  # Normalize to zero-mean, unit-variance
  window_size: 2                  # 2-frame observation history

training:
  batch_size: 64                  # Fits in 24GB with 27M params
  learning_rate: 3e-4
  warmup_steps: 200
  max_steps: 30000                # ~4 hours on RTX 3090
  weight_decay: 0.01
  save_every: 5000
  eval_every: 2500
  frozen_keys: []                 # Fine-tune everything (small model)

augmentation:
  random_crop: true
  color_jitter: 0.1
  random_erasing: 0.05
```

### Training Script (Conceptual)

```python
# Octo uses Jax/Flax, not PyTorch
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import TrainState
import tensorflow_datasets as tfds
import jax

# Load pretrained Octo-Small
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# Configure fine-tuning for our action space
# Octo rebuilds the action head for new action dims automatically
config = model.config.copy()
config["model"]["heads"]["action"]["kwargs"]["action_dim"] = 7
config["model"]["heads"]["action"]["kwargs"]["pred_horizon"] = 4

# Build fine-tuning dataset (RLDS format)
dataset = tfds.load("kitting_demos", split="train")
dataset = dataset.map(preprocess_fn)  # Resize images, normalize actions

# Fine-tune all parameters (27M — small enough for full fine-tune)
train_state = TrainState.create(
    model=model,
    optimizer=optax.adamw(learning_rate=3e-4, weight_decay=0.01),
)

# Training loop:
#   action_chunk = model(images, language, state)  # (batch, H, 7)
#   loss = diffusion_loss(action_chunk, ground_truth_chunk)
#   grads = jax.grad(loss)(train_state.params)
#   train_state = train_state.apply_gradients(grads=grads)
```

### Evaluation Metrics (During Training)

- **Action MSE**: Mean squared error on held-out episodes (target < 0.01)
- **Gripper accuracy**: Binary classification accuracy for open/close (target > 95%)
- **Rollout success rate**: Deploy in Isaac Sim every 5K steps, run 100 episodes (target > 80%)

---

## 7. Inference & Deployment

### ROS2 VLA Inference Node

Replace the entire `kitting_task_planner` + `kitting_perception` stack with a single node:

```python
# vla_inference_node.py
import jax
import numpy as np
from octo.model.octo_model import OctoModel

class VLAInferenceNode(Node):
    """Single node that replaces task_planner + perception + move_group."""

    def __init__(self):
        super().__init__('vla_inference_node')

        # Load fine-tuned Octo-Small model (~120MB)
        self.model = OctoModel.load_pretrained("./checkpoints/kitting_octo_small")
        self.task = self.model.create_tasks(texts=[""])  # Placeholder, updated per pick

        # Camera subscription (RGB only, no point cloud)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Joint state subscription
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.js_callback, 10)

        # Joint trajectory publisher (bypass move_group entirely)
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # Gripper client (same as current repo)
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')

        # Kit order trigger
        self.order_sub = self.create_subscription(
            KitOrder, '/kit_order', self.order_callback, 10)

        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        # Observation history buffer (Octo uses 2-frame window)
        self.obs_history = []
        self.current_image = None
        self.current_joints = None
        self.active_task = None
        self.current_slot = 0

    def control_loop(self):
        if self.active_task is None:
            return
        if self.current_image is None or self.current_joints is None:
            return

        # Build observation dict (Octo format)
        obs = {
            "image_primary": self.current_image[None],  # (1, 256, 256, 3)
            "proprio": np.array(self.current_joints)[None],  # (1, 7)
        }
        self.obs_history.append(obs)
        if len(self.obs_history) > 2:
            self.obs_history.pop(0)

        # Stack 2-frame history
        stacked_obs = jax.tree.map(
            lambda *xs: np.concatenate(xs, axis=0), *self.obs_history)

        # Update task language
        prompt = f"Pick the box and place it in slot {self.current_slot}"
        task = self.model.create_tasks(texts=[prompt])

        # Predict action chunk (H=4 steps ahead, ~15-25 Hz inference)
        action_chunk = self.model.sample_actions(
            stacked_obs, task, rng=jax.random.PRNGKey(0))
        # action_chunk shape: (1, 4, 7) — 4 future steps, 7 dims each

        # Execute first action from chunk (receding horizon)
        action = action_chunk[0, 0]  # (7,) = [dj1..dj6, gripper_cmd]
        joint_deltas = action[:6]
        gripper_cmd = action[6]

        # Apply delta to current joints
        target_joints = self.current_joints[:6] + joint_deltas
        self.send_joint_command(target_joints)

        # Handle gripper (binary: open if > 0.5, close otherwise)
        if gripper_cmd > 0.5 and self.gripper_state == 'closed':
            self.open_gripper()
        elif gripper_cmd <= 0.5 and self.gripper_state == 'open':
            self.close_gripper()

        # Check termination (simple heuristic: at home + gripper open + time > 3s)
        if self.is_episode_done():
            self.current_slot += 1
            if self.current_slot >= self.total_picks:
                self.active_task = None
```

### Launch File (VLA Mode)

```python
# vla.launch.py - replaces sim_full.launch.py
# Starts: RSP + ros2_control + controllers + vla_inference_node
# Does NOT start: move_group, planning_scene_setup, bin_perception_node
```

### Safety Layer

Even with a VLA, add a thin safety wrapper (not a full planner):

```python
# Joint velocity limits (same as URDF)
JOINT_VEL_LIMITS = [2.36, 2.36, 2.36, 2.36, 2.36, 3.14]  # rad/s

# Workspace bounds (keep arm in safe zone)
POSITION_BOUNDS = {
    'x': [-0.2, 1.2],
    'y': [-1.0, 0.7],
    'z': [-0.02, 1.0],   # Never below table
}

# Emergency stop if any joint exceeds limits or TCP exits bounds
def safety_check(target_joints, current_tcp_pose):
    # Clip joint velocities
    # Check forward kinematics against workspace bounds
    # Return clipped action or trigger e-stop
```

This is the only "traditional robotics" code needed. No MoveIt, no collision scene, no
planning. The VLA learns collision avoidance implicitly from training data.

---

## 8. Implementation Phases

### Phase 0: Infrastructure Setup (1-2 weeks)

- [ ] Add RGB camera publishing to Isaac Sim scene (if not already present)
- [ ] Create `kitting_vla` ROS2 package with data collection node
- [ ] Set up LeRobot/HDF5 data pipeline
- [ ] Verify Isaac Sim can run headless for batch data collection
- [ ] Set up training environment (install Octo, Jax, tensorflow_datasets)

### Phase 1: Data Collection (1-2 weeks)

- [ ] Run existing task planner in Isaac Sim to collect 10K episodes
- [ ] Implement domain randomization (box pose, lighting, texture)
- [ ] Validate data quality: check image-action alignment, correct labels
- [ ] Split: 9K train, 500 val, 500 test
- [ ] Compute dataset statistics (action normalization params)

### Phase 2: Training (1 week)

- [ ] Fine-tune Octo-Small on collected data (~4 hrs on RTX 3090)
- [ ] Monitor training curves (action MSE, gripper accuracy)
- [ ] Run sim evaluation rollouts every 5K steps
- [ ] Select best checkpoint by sim success rate
- [ ] If success rate < 70%, collect more data or upgrade to Octo-Base (93M)

### Phase 3: Sim Evaluation (1 week)

- [ ] Deploy best checkpoint in Isaac Sim closed-loop
- [ ] Measure: success rate, cycle time, collision rate
- [ ] Test generalization: unseen box positions, slight lighting changes
- [ ] Identify failure modes (missed grasps, collisions, wrong slot)
- [ ] If needed: collect targeted demonstrations for failure cases

### Phase 4: Real Hardware (2-3 weeks, if applicable)

- [ ] Collect 200-500 teleoperation demonstrations on real robot
- [ ] Fine-tune sim-trained model on mixed sim+real data
- [ ] Deploy with safety wrapper on real HC10DT
- [ ] Iterate on failure cases with additional real demonstrations

---

## 9. Simplified Cell Config for VLA

```yaml
# vla_cell_config.yaml
cell:
  robot: HC10DT
  gripper: Schunk EGK-40

# Single input bin (center position from current A2)
input_bin:
  position: [0.75, 0.0, 0.05]
  size: [0.350, 0.350, 0.100]
  parts_per_episode: 1          # Start with 1, scale to 3

# Output tray (unchanged)
output_tray:
  position: [0.0, -0.75, 0.02]
  slots: 9                      # 3x3 grid, 100mm spacing

# Camera
camera:
  type: RGB                     # No depth/point cloud needed
  position: [1.25, 0.0, 1.50]
  resolution: [640, 480]        # Downsampled to 256x256 for model
  topic: /camera/image_raw

# Parts (large boxes only)
parts:
  single_gang_box:
    dims: [100, 55, 55]        # mm
    grasp: parallel             # Gripper closes on 55mm Y face
  round_box:
    dims: [70, 70, 55]         # mm
    grasp: parallel             # Gripper closes on 70mm diameter

# Safety limits
safety:
  max_joint_velocity: 0.5       # rad/s (conservative for VLA)
  workspace_z_min: -0.01        # Never below table
  episode_timeout: 60           # seconds
```

---

## 10. Key Learnings From Current Repo (Applicable to VLA)

These are non-obvious lessons from the traditional stack that should inform VLA training:

1. **Gripper orientation matters**: All successful grasps use 180deg Y rotation (gripper
   pointing straight down). The VLA should converge on this naturally, but seeding training
   data with consistent top-down grasps helps.

2. **Bin walls are real obstacles**: The gripper enters from the open top. Current stack
   uses 150mm approach offset to clear 100mm walls. The VLA must learn this clearance
   implicitly - include episodes where the arm descends vertically into the bin.

3. **TCP offset**: Finger pads are 50mm above TCP link. In the current stack this is
   compensated explicitly. For VLA with joint-space actions, this is handled implicitly
   by the training data (the model sees images and learns the right joint positions).

4. **Home position matters**: Starting/ending at home (joints zeroed, j5=90deg) gives the
   camera a clear view and avoids self-collision. Include go-home in every episode.

5. **Place height**: The current stack approaches slots from 100mm above, then descends.
   This avoids catching tray walls. VLA training data should include this approach pattern.

6. **Gripper is binary for large boxes**: Open (83mm) or close to grasp width. No
   proportional control needed. Simplify the action space to binary gripper.

7. **Round box needs no yaw alignment**: Symmetric shape means any gripper yaw works.
   Single gang box benefits from alignment along the long axis, but the grasp succeeds
   at any angle since the 55mm face fits in the 83mm gripper.

8. **Cycle time baseline**: Current stack achieves ~45s per pick-place. VLA at 10 Hz with
   direct joint control should achieve ~15-25s (no planning overhead).

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VLA doesn't generalize to unseen box positions | Medium | More domain randomization, 2x training data |
| Sim-to-real gap on real hardware | High | Collect 200+ real demonstrations, mixed training |
| Collisions with bin walls during descent | Medium | Safety wrapper + include near-miss recoveries in training |
| Slow inference | Very Low | Octo-Small runs 15-25 Hz natively, well above 10 Hz target |
| Model confuses box types | Low | Distinct language prompts per SKU, visual difference is large |
| Gripper doesn't close fully on box | Medium | Include failed grasp + retry sequences in training data |

---

## 12. Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Pick success rate (sim) | >85% | 500 eval episodes in Isaac Sim |
| Place accuracy (sim) | >80% within 30mm of slot | Measure box-to-slot distance |
| Cycle time | <30s per pick-place | Wall clock in sim |
| Training cost | <$10 cloud or $0 own GPU | Cloud GPU hours |
| Model size (deployed) | ~120MB fp32, <2GB VRAM | Measure at inference |
| Collision rate | <5% of episodes | Count arm-bin and arm-tray contacts |

---

## 13. Repository Structure (New Package)

```
kitting_ws/src/
  kitting_vla/                      # NEW package
    kitting_vla/
      vla_inference_node.py         # ROS2 node: image+joints -> actions (replaces task_planner + perception)
      data_collector_node.py        # Records (image, state, action) from existing stack
      safety_wrapper.py             # Joint limits, workspace bounds, e-stop
      episode_manager.py            # Simple orchestrator: triggers picks, tracks slots
    config/
      vla_cell_config.yaml          # Simplified cell config (above)
      train_config.yaml             # Training hyperparameters
    launch/
      collect_data.launch.py        # Isaac Sim + full stack + data collector
      vla_inference.launch.py       # ros2_control + VLA node (no MoveIt)
    scripts/
      train_octo.py                 # Fine-tuning script (Jax, runs outside ROS)
      eval_sim.py                   # Sim rollout evaluation
      convert_dataset.py            # Raw HDF5 -> RLDS format (Octo native)
    test/
      test_safety_wrapper.py
      test_episode_manager.py
    package.xml
    setup.py

  # Existing packages (unchanged, used for data collection)
  kitting_description/
  kitting_bringup/
  kitting_task_planner/             # Used as expert policy for data gen
  kitting_perception/               # Used during data collection only
  kitting_moveit_config/            # Used during data collection only
```

---

## 14. Quick Start Checklist

1. **Verify Isaac Sim publishes RGB**: Check topic `/sim/image_raw` exists at 10+ Hz
2. **Build data collector**: Subscribe to image + joint_states, compute actions from consecutive states
3. **Run 100 test episodes**: Validate data pipeline end-to-end
4. **Install Octo**: `pip install octo-model jax jaxlib tensorflow tensorflow_datasets`
5. **Train on 100 episodes**: Smoke test — loss should decrease, actions should be non-random
6. **Scale to 10K episodes**: Full training run (~4 hrs on RTX 3090)
7. **Deploy in sim**: Close the loop, measure success rate
8. **Iterate**: More data for failure cases, adjust hyperparameters
