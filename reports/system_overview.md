# VLA Kitting — What This Repo Is, and How to Think About It

> **2026-04-27 amendment.** The data-pyramid section ("human demos → Mimic
> → LeRobot") and the architecture diagrams that show `mimic_generate.sh`
> describe the pipeline as it was when this overview was written. As of
> 2026-04-27 we use SCRIPTED demos exclusively going forward; Mimic
> generation has been removed from the active pipeline. The Mimic env
> cfg, orchestration scripts, and annotate_demos wrapper remain in the
> tree as historical artifacts but are not invoked by any current plan.
> See `reports/2026-04-27_scripted_only_data_pipeline.md`. Read this
> document for the conceptual background and historical pipeline shape;
> consult dated reports for what's actually live now.

## Part 1 — Background concepts (robotics/ML intuition, no jargon dump)

### What a VLA actually is

"VLA" stands for **Vision-Language-Action** model. Think of it as three transformer stacks glued together:

1. **Vision encoder** — a pretrained image model (here, a small SigLIP-style ViT) that turns each camera frame into a grid of patch tokens, roughly "what do I see at each spot of the image."
2. **Language encoder / LLM backbone** — a small language model (SmolLM, ~360M params) that takes a plain-English instruction like `"pick up the cube and place it on the target"` and produces a token stream that conditions behavior on what the user asked for.
3. **Action head / decoder** — an extra transformer (in SmolVLA's case, a *flow-matching* / diffusion-style head) that takes the fused vision + language + robot state tokens and outputs a short **chunk of future actions** — e.g., the next 50 joint-delta commands plus a gripper bit — rather than one step at a time. Chunking is a big deal: it lets the model plan a brief maneuver (approach, press down, close) as one coherent thing, instead of policy-gradient-style twitching.

Why VLAs at all? Before this line of work, imitation-learning policies were task-specific and didn't understand language. A VLA is a general-purpose robot policy: swap the prompt, and in theory you can get different behavior from the same weights. In practice, fine-tuning on your task is still necessary, but you start from a base that has already learned "hands, tables, objects, pick-like motions" from millions of trajectories across many robots.

SmolVLA-450M specifically is LeRobot's tiny public VLA (Hugging Face: `lerobot/smolvla_base`). Its virtue is that it fits on an RTX laptop and trains in hours, not days. Its vice is that it's not very smart — you'll see this in the disadvantages below.

### What training + fine-tuning mean here

This project doesn't train SmolVLA from scratch — that costs hundreds of GPU-days and a dataset you don't have. Instead it does **LoRA fine-tuning**:

- Freeze the base model's ~450M weights.
- Insert small low-rank adapter matrices (here, rank 16) into every linear layer.
- Only the adapters are trainable — maybe 1–3% of the parameter count — which means tiny VRAM footprint, no risk of catastrophically forgetting what SmolVLA knew, and checkpoints that are 10s of MB instead of GBs.

The **loss** being minimized is just behavioral cloning: given (image, state, prompt), predict the expert's action chunk. With flow-matching it's slightly fancier — you're predicting a vector field that transports noise toward the expert action — but morally it's "mimic the demos."

### The data pyramid: human demos → Mimic → LeRobot

VLA fine-tuning is hungry for data. Collecting 15 human teleop demos takes an afternoon; collecting 500 does not. **Isaac Lab Mimic** (NVIDIA's data-multiplier for MimicGen) is the trick that bridges that gap:

1. You annotate ~15 seed demos into subtask segments (approach / grasp / transport / release).
2. Mimic splices segments across demos, transforms them into new cube starting positions via coordinate math, and replays the stitched trajectory in the simulator. The ones where the cube still ends up placed successfully become new "demos."
3. You keep only the successful replays. You've gone from 15 → 500+ synthetic demos without any human.

After that, the HDF5s get converted to **LeRobot's v3 dataset format** — Parquet files for state/actions and MP4/PNG for images — which is what the `lerobot_train.py` loader expects.

---

## Part 2 — What's actually in this repo

> **Stale banner (2026-04-27):** the orchestration architecture below describes the
> deprecated continual-train + Mimic loop. `scripts/orchestrate/continual_train.sh`
> and the Mimic-pool validators have been removed; the live pipeline is
> `scripts/validate/scripted_pick_demo.py` → `scripts/data/isaaclab_to_lerobot.py`
> → `scripts/orchestrate/train_only.sh`. See
> `reports/2026-04-27_scripted_only_data_pipeline.md` for the rationale and the
> rewritten Phase 2 of `recovery_plan_2026-04-24.md`.

### The layered architecture

```
┌────────────────────────── Orchestration ──────────────────────────┐
│  scripts/orchestrate/continual_train.sh  (the big red button)     │
│    ├─ seed: scripted_pick → clean → annotate                      │
│    ├─ mimic_generate.sh  (loop, generates batches of 25)          │
│    ├─ train_only.sh      (loop, trains on latest snapshot)        │
│    ├─ fix_adapter_configs.py  (daemon, normalizes LoRA ckpts)     │
│    └─ budget_watchdog.py      (daemon, stops on budget/plateau)   │
└───────────────────────────────────────────────────────────────────┘
             │                    │                       │
             ▼                    ▼                       ▼
     ┌──────────────┐    ┌──────────────┐       ┌──────────────┐
     │ Isaac Lab    │    │ HDF5 → Parquet│      │ LeRobot +    │
     │ env (envs/)  │    │ scripts/data/ │      │ SmolVLA LoRA │
     │ physics,     │    │ Mimic merge & │      │ (external    │
     │ cameras, IK  │    │ converter     │      │ clone)       │
     └──────────────┘    └──────────────┘       └──────────────┘
             │
             ▼
     ┌───────────────────┐
     │ kitting_vla/      │  ROS2 package to run the trained
     │ vla_inference_node│  policy on the real HC10DT (not
     │ + safety_wrapper  │  exercised yet — sim-only so far)
     └───────────────────┘
```

### Directory tour

- **`assets/`** — URDFs, xacros, and USDs for the Yaskawa HC10DT, the Robotiq 2F-85 gripper, and the cube-pick scene. Three arm variants are kept around because gripper composition was fiddly (see disadvantages). The one actually used is `hc10dt_with_ria_gripper.usd`. There's also a vendored `nvidia_robotiq_2f85/` pack that was an earlier, abandoned attempt.
- **`envs/`** — Isaac Lab `ManagerBasedRLEnv` definitions for the pick-cube task. Two variants: a plain one used for rollouts/eval, and a Mimic variant that exposes extra *subtask signals* (`ee_above_cube`, `cube_gripped`, `cube_above_target_xy`) so Mimic's annotator knows where phase boundaries are. The scene has a wrist camera (128×128) and a fixed third-person camera (256×256), a 5 cm cube with randomized position and color, a magenta target-zone marker (visual-only), and dome-light intensity randomization.
- **`configs/`** — just two Isaac Kit files (bare-bones headless app configurations). The *training* config is passed inline via the justfile / bash scripts, not stored here.
- **`datasets/`** — where everything lives at every stage: `teleop/` (scripted HDF5 + cleaned + annotated), `mimic/pool/` (per-batch HDF5s) + a merged master, and `lerobot/cube_pick_v1_batch_NNN/` snapshots. `cube_pick_v1` is a symlink that is atomically swapped to point at the newest batch after each Mimic round — so a training run always has a coherent dataset even while Mimic is writing new ones.
- **`scripts/`** — the workhorses:
  - `orchestrate/` — bash + Python glue for the continual loop.
  - `validate/` — per-phase sanity scripts (scene inspection, env smoke test, scripted-pick demo generator, gripper probes, demo replayer).
  - `assembly/` — URDF→USD conversion and scene building.
  - `data/` — HDF5 cleaning, Mimic annotation driver, HDF5→LeRobot converter, demo inspector, demo-video renderer.
  - `teleop/` — keyboard-driven demo recording (the human-gate step).
  - `train/` — `run_vla_closed_loop.py`, which loads a checkpoint, rolls it out in sim, saves a side-by-side GIF and a JSONL of per-episode results.
- **`kitting_vla/`** — a full ROS2 ament package with an inference node, a data-collector node, an episode manager, and a safety wrapper. This is the future on-hardware interface but nothing here has been exercised on the real robot yet — it's a stub for the next phase.
- **`checkpoints/continual/NNNNNN/pretrained_model/`** — where LoRA adapter weights land.
- **`logs/continual/`** — the control surface. Structured JSONL: `train_steps.jsonl` (step-level loss, grad_norm, lr), `epoch_summary.jsonl` (p50/p95 loss + eval success rate), `eval_episodes.jsonl` (per-rollout outcome). Plus a `state.json` the orchestrator reads to know where to resume.
- **`reports/`** — PNGs from the cameras, the rollout GIF (`vla_rollout_035000.gif` is a sample at 35K steps), some prose markdowns about phase findings and known issues. `scripts/orchestrate/plot_metrics.py` builds the loss / eval-SR / episode-step curves from the JSONL.
- **`tests/`** — pytest smoke tests for the repo layout, Isaac Sim startup, scene/USD validity, gripper contact, and env stepping. Quick CI-style checks, not deep.
- **`CLAUDE_CODE_PLAN.md` / `VLA_KITTING_PLAN.md`** — the design docs. The plan is phase-based (0–13), with one single human gate at Phase 6 (teleop) and everything else automated.
- **`justfile`** — the user-facing task runner. `just scripted-pick`, `just mimic-generate`, `just to-lerobot`, `just train`, `just run-vla`, `just continual-train`, etc. Everything a human types is one `just` recipe.

### How a run actually flows end-to-end

1. `just continual-train` launches `continual_train.sh`.
2. **Seed step** (one-time, skipped on resume): privileged scripted policy (`scripted_pick_demo.py`) uses the ground-truth cube pose to drive a perfect pick, producing ~25 HDF5 demos. These are filtered (`clean_demos.py`, reject anything <100 steps), then annotated with subtask boundaries.
3. **Mimic loop (background)**: calls Isaac Lab's `generate_dataset.py` with the Mimic env, producing `pool/batch_NNN.hdf5`. After each batch it merges into `cube_mimic_all.hdf5`, runs `isaaclab_to_lerobot.py` to build a LeRobot snapshot, and atomically swaps the `cube_pick_v1` symlink.
4. **Training loop (foreground)**: each epoch it runs a few thousand LeRobot training steps (`lerobot_train.py --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base` with LoRA on), then every N epochs it calls `run_vla_closed_loop.py` for K rollout episodes and appends success rate to `epoch_summary.jsonl`.
5. **Two daemons** keep everything alive:
   - `fix_adapter_configs.py` watches the checkpoint tree and rewrites `adapter_config.json` on each new save so LeRobot's resume path doesn't get tangled (see disadvantages).
   - `budget_watchdog.py` kills the run if wallclock budget expires or the eval-SR plateaus with no loss improvement.

### The clever bits worth calling out

- **Atomic dataset snapshots by symlink swap.** The training reader sees a static path; the Mimic writer never races it because it writes into a new directory and then flips the symlink. Simple and robust.
- **JSONL everywhere.** Logs are append-only JSON objects, not text. `plot_metrics.py` just pandas-reads them — you don't need W&B.
- **Scripted seed instead of teleop.** The plan originally had a teleop human gate, but the repo also ships a scripted pick using privileged state. That removes the last human in the loop at the cost of less motion diversity.
- **Strip-and-reload PEFT on resume.** LeRobot normally double-wraps LoRA adapters when you resume a fine-tune, which silently regresses loss to the initial plateau (~0.75). The orchestrator strips `cfg.peft` from the resumed config, and there's a matching one-line patch in `lerobot/src/lerobot/policies/factory.py` that forces `is_trainable=True` on the reloaded adapter. Loss descends properly across resume boundaries again.

---

## Part 3 — Advantages

1. **Minimal human effort.** The whole pipeline can run overnight from a single command with no teleop step at all; the only human gate documented in the plan has been engineered around by the scripted-demo path.
2. **Tight, cheap fine-tuning loop.** LoRA on a 450M model fits in ~8 GB VRAM, trains in batches of 4–8, and saves small adapters. You can iterate in hours, not days.
3. **Continual-training design.** Mimic generation and training run concurrently, so every epoch sees strictly more data than the last without humans in the loop. Plateau detection catches diminishing returns without wasting compute.
4. **Observability is real.** JSONL per-step logs + `plot_metrics.py` means you can always answer "where did the run go off the rails" without re-running. `run_vla_closed_loop.py` optionally dumps GIFs per eval, so you can eyeball regressions.
5. **Defensive orchestration.** Symlink-atomic snapshots, adapter-config normalizer daemon, budget watchdog, resume-safe state file — a lot of work has gone into the pipeline not blowing up during a long unattended run.
6. **Clean separation of concerns.** Env code (envs/) is pure Isaac Lab. Data code (scripts/data/) is pure HDF5/Parquet. Training code is LeRobot with a thin wrapper. The pieces are independently debuggable and independently replaceable — e.g., swap SmolVLA for π0 without touching the data layer.
7. **Plan-as-doc.** `CLAUDE_CODE_PLAN.md` is genuinely a commissioning playbook: phase gates, validation criteria, known failure modes. If a teammate shows up tomorrow, they can follow it.

---

## Part 4 — Disadvantages

1. **Scripted demos bake in their own biases.** The seed is a privileged-state pick, so Mimic is multiplying a narrow distribution. The cube yaw randomization is disabled because the scripted demo doesn't read cube orientation — so the policy will never learn to handle a rotated cube until that's fixed. Data diversity is capped by the seed, not the Mimic output count.
2. **Wrist camera is known-broken.** `reports/known_issues.md` notes the wrist camera is mostly black, likely a mount-frame or collision-mesh issue. The policy trains on it anyway, effectively learning to ignore a useless input — which also means the *real* wrist-camera signal is missing from the state. Training on garbage features costs capacity.
3. **Sim-only. No sim-to-real.** No domain randomization beyond cube color and dome intensity; no camera noise, camera-pose jitter, motor-dynamics randomization, or calibration noise. The `kitting_vla/` ROS package is a stub — the policy has not run on the real HC10DT, and it almost certainly won't transfer without more work.
4. **Brittle integration with upstream LeRobot.** The pipeline depends on two local monkey-patches (the `factory.py` `is_trainable=True` edit and the Isaac Lab kit-file pin relax) plus a runtime daemon to keep `adapter_config.json` sane. If LeRobot or IsaacLab gets pulled fresh, those patches need reapplying by hand. Documented, but fragile.
5. **Hardcoded paths.** Scripts assume `/home/ubuntu/IsaacLab`, `/home/ubuntu/code/lerobot/src`, `/home/ubuntu/vla_kitting`, Isaac Sim at `/opt/IsaacSim/`, DCV display `:1`. Moving to another machine is not a `git clone; make` exercise.
6. **SmolVLA ceiling is low.** 450M parameters and a tiny vision encoder means it can memorize this cube-pick well enough, but the language input is largely decorative — there's only one task prompt, so the model learns one trajectory regardless of the instruction. A stronger VLA (π0, Octo-Base) would be harder to train but would actually make the L in VLA earn its keep.
7. **Eval is thin.** Closed-loop evals run 10 episodes on the training distribution; held-out generalization is measured only in the plan (expanded pose ranges), not really enforced in the continual loop. Success rate numbers are optimistic vs. anything off-distribution.
8. **Gripper tuning is fragile.** Gripper close-target (0.5 rad), material friction (pad μ=1.0, cube μ=0.8), solver-iteration counts are all hand-tuned to one cube geometry. Any change to cube size/mass/material or gripper geometry re-opens a painful contact-physics tuning loop — documented in the phase findings.
9. **Continual loop is sequential, not interleaved.** Mimic and training share one GPU, and each epoch waits for a fresh dataset rebuild. A more efficient design would pipeline Mimic generation and training, but that'd need more plumbing and a second GPU.
10. **No versioning of datasets.** Batches are snapshotted by number, but there's no manifest of which seed demos + which Mimic config produced which batch. If a policy regresses, reproducing "what data was this trained on" means going through logs, not a metadata file.
