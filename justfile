set shell := ["bash", "-cu"]

# Canonical paths
ISAAC_SIM := "/opt/IsaacSim"
ISAAC_LAB := env_var_or_default("ISAAC_LAB", "/home/ubuntu/IsaacLab")
ISAAC_PY := "/opt/IsaacSim/python.sh"
REPO := justfile_directory()

default:
    @just --list

# ---------- Phase 0 ----------
audit:
    @cat {{REPO}}/logs/phase_0_audit.log 2>/dev/null || echo "Run Phase 0 first"

# ---------- Phase 2 ----------
isaac-smoke:
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/isaac_smoke.py

isaac-cartpole:
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 16 --headless

# ---------- Phase 3 ----------
scene-inspect usd="assets/hc10dt_v1.usd":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/scene_inspect_with_app.py {{REPO}}/{{usd}}

build-scene:
    {{ISAAC_PY}} {{REPO}}/scripts/assembly/build_scene_v1.py

build-combined:
    python3 {{REPO}}/scripts/assembly/generate_urdf.py --xacro {{REPO}}/assets/hc10dt_with_gripper_v1.urdf.xacro --output /tmp/hc10dt_with_gripper_v1.urdf
    {{ISAAC_PY}} {{REPO}}/scripts/assembly/urdf_to_usd.py --urdf /tmp/hc10dt_with_gripper_v1.urdf --usd {{REPO}}/assets/hc10dt_with_gripper_v1.usd --fix-base

# ---------- Phase 5 ----------
env-smoke:
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/env_smoke.py

# ---------- Phase 7 ----------
# Scripted pick demo in the Isaac Sim viewport (DCV). Use num=1 for visual debugging.
scripted-pick-gui display="1" num="1":
    DISPLAY=:{{display}} bash -c "cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/scripted_pick_demo.py \
        --num_demos {{num}} \
        --gui"

# Headless scripted pick demo (for producing the actual dataset).
scripted-pick num="15":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/scripted_pick_demo.py --num_demos {{num}}

camera-samples n="5":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/save_camera_samples.py --samples {{n}}

# ---------- Phase 8-9 (Mimic) ----------
# Annotate the scripted seed demos with subtask boundaries. Writes
# cube_annotated.hdf5 that Mimic's generator reads from.
mimic-annotate src="datasets/teleop/cube_scripted.hdf5" out="datasets/teleop/cube_annotated.hdf5":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/data/annotate_demos.py \
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
        --input_file {{REPO}}/{{src}} \
        --output_file {{REPO}}/{{out}} \
        --auto \
        --headless \
        --enable_cameras

# Generate synthetic demos via Mimic. Default 100 (bump with num= once verified).
mimic-generate src="datasets/teleop/cube_annotated.hdf5" out="datasets/mimic/cube_mimic.hdf5" num="100" envs="1":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/data/generate_dataset.py \
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-Mimic-v0 \
        --input_file {{REPO}}/{{src}} \
        --output_file {{REPO}}/{{out}} \
        --generation_num_trials {{num}} \
        --num_envs {{envs}} \
        --headless \
        --enable_cameras

# Inspect an Isaac Lab HDF5 dataset (demo count, lengths, action ranges, obs keys).
inspect-demos path="datasets/teleop/cube_scripted.hdf5":
    {{ISAAC_PY}} {{REPO}}/scripts/data/inspect_demos.py {{REPO}}/{{path}}

# ---------- Phase 10 (LeRobot conversion) ----------
# Convert Isaac Lab HDF5 -> LeRobot v3 dataset for SmolVLA training.
to-lerobot src="datasets/mimic/cube_mimic.hdf5" out="datasets/lerobot/cube_pick_v1" repo="vla_kitting/cube_pick_v1":
    PYTHONPATH=/home/ubuntu/code/lerobot/src:$PYTHONPATH \
      {{REPO}}/.venv/bin/python {{REPO}}/scripts/data/isaaclab_to_lerobot.py \
        --input {{REPO}}/{{src}} \
        --output {{REPO}}/{{out}} \
        --repo_id {{repo}} \
        --task "pick up the cube and place it on the green target"

# ---------- Phase 6 ----------
teleop display="1" num="15":
    DISPLAY=:{{display}} bash -c "cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/teleop/record_demos.py \
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 \
        --teleop_device keyboard \
        --dataset_file {{REPO}}/datasets/teleop/cube_raw.hdf5 \
        --num_demos {{num}} \
        --enable_cameras"

teleop-dryrun:
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/teleop_dry_run.py

validate-demos path="datasets/teleop/cube_raw.hdf5":
    cd {{ISAAC_LAB}} && ./isaaclab.sh -p {{REPO}}/scripts/validate/replay_demos.py --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 --dataset_file {{REPO}}/{{path}}

# ---------- Testing ----------
test-all:
    source {{REPO}}/.venv/bin/activate && pytest {{REPO}}/tests/ -v

lint:
    source {{REPO}}/.venv/bin/activate && ruff check {{REPO}}/scripts {{REPO}}/envs {{REPO}}/tests

# ---------- Monitoring ----------
logs-tail phase:
    tail -f {{REPO}}/logs/phase_{{phase}}_*.log

gpu:
    nvtop

progress:
    @jq . {{REPO}}/logs/PROGRESS.json
