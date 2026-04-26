# Next training pass — options + recommended plan

This doc is for a **fresh Claude Code session** to pick up from. It captures
what the previous 8h continual-training run actually taught us, the bugs found
(some fixed on the fly, some still latent in the pipeline), and the decisions
you need to make before kicking off the next run. Written 2026-04-22 right
after the 8h run exhausted its budget.

The existing `reports/continual_training_plan.md` is still the reference for
the overall orchestrator design and file layout. This doc is specifically about
what to change for attempt #2.

## Previous run — one-paragraph summary

Kicked off `continual_train.sh --reset` at 04:29:54 UTC 2026-04-22. Phase 1
(25 scripted demos) + annotate completed in ~55 min at 100% success rate.
First Mimic batch produced 25 more demos at 100% rate, merged into a LeRobot
dataset. Training started at epoch 22 (first call = `--steps=22000` because the
loop counter had inflated during the wait for the dataset) and ran cleanly for
the first 22k steps, ending with training loss tightly clustered around 0.18.
Then every subsequent resumed epoch **silently regressed to loss ~0.75**. The
watchdog touched STOP at exactly 12:29:54 UTC and the orchestrator exited
gracefully at 13:01:28 after the in-flight eval completed. Only checkpoint
`022000` actually contains a trained model; everything after is a re-initialized
LoRA adapter on top of base SmolVLA (see bugs below).

## Bugs discovered during the run

All of these were diagnosed but ONLY the first three have orchestrator-side
fixes committed. The rest still need to be addressed before the next run.

### Fixed during the run (in-tree)
1. **CKPT_DIR existed-dir collision.** `mkdir -p "$CKPT_DIR"` at orchestrator
   startup made LeRobot refuse to write into it on the fresh-start epoch
   (`FileExistsError` unless `resume=true`). Fix: fresh-start branch now
   `rmdir`s the empty CKPT_DIR before invoking lerobot. See
   `scripts/orchestrate/continual_train.sh` around the `USE_LORA` branch.
2. **`--config_path` for resume.** LeRobot computes `policy_dir = Path(config_path).parent`
   (no symlink resolution). Pointing at the pretrained_model *directory* gave
   the wrong parent. Fix: point at the `train_config.json` *file* inside
   `pretrained_model/` so `.parent` lands on `pretrained_model/` where the
   adapter files live.
3. **PEFT `adapter_config.json` base_model path.** On save, PEFT captures
   `cfg.policy.pretrained_path` as `base_model_name_or_path`. After resume
   that path is the local previous checkpoint (adapter-only) rather than
   `lerobot/smolvla_base`. Fix: background watcher at
   `scripts/orchestrate/fix_adapter_configs.py` rewrites `base_model_name_or_path`
   to `lerobot/smolvla_base` whenever a new checkpoint is saved.

### Diagnosed but NOT yet fixed — these caused the flat loss post-resume
4. **DOUBLE PEFT WRAP (the real silent killer).** On resume, `make_policy()`
   correctly loads the saved adapter onto base SmolVLA via
   `PeftModel.from_pretrained(...)`. Then `lerobot_train.py` line 240 sees
   `cfg.peft is not None` (loaded from saved `train_config.json`) and calls
   `policy.wrap_with_peft(...)` AGAIN — layering a fresh, **zero-initialized**
   LoRA on top of the already-loaded adapter. The optimizer trains only the
   new zero-init adapter; the previously-trained one is frozen and ignored.
   Log evidence: `UserWarning: You are trying to modify a model with PEFT for a
   second time.` This is why loss went from 0.18 at end of epoch 22 to ~0.75
   on every single resumed epoch.

5. **LR schedule decays to floor by step ~22000.** Default smolvla config has
   `scheduler_decay_steps=30000` and `scheduler_decay_lr=2.5e-06`. At step
   22000, `_last_lr` was already 2.5e-6 (effectively zero) per
   `checkpoints/022000/training_state/scheduler_state.json`. Even without the
   double-wrap bug, training past step 22000 would have been near-no-op.

### Eval-script issues (patched this session, not yet in trunk-worthy shape)
6. `run_vla_closed_loop.py` originally did `SmolVLAPolicy.from_pretrained(ckpt)`
   directly, which fails on PEFT checkpoints (no `model.safetensors`). Patched
   to detect `adapter_config.json` and use `PeftModel.from_pretrained` on top
   of base SmolVLA loaded from HF. After patching, three more issues surfaced:
   - Saved `config.json` is missing the `type: smolvla` key that
     `PreTrainedConfig.from_pretrained` (the dispatcher) needs.
   - Direct `SmolVLAConfig.from_pretrained` doesn't accept `type` either —
     must call the base-class `PreTrainedConfig.from_pretrained`.
   - HF's base SmolVLA config expects `observation.images.camera{1,2,3}`;
     our task uses `wrist` + `third_person`. Must pass our local config
     when instantiating the base policy.

   All three issues are currently patched in `scripts/train/run_vla_closed_loop.py`.
   For 022000 I manually injected `type: smolvla` into `config.json`. The
   adapter_fixer should be extended to do this automatically for every new
   checkpoint.

### Mimic batch stall (not yet diagnosed)
7. **Mimic runs fine alone, stalls when training is active.** Batch 1
   generated 25 demos at 100% in 46 min while nothing else ran. Batches 2 and
   5 (both started while training was alive) produced output files that grew
   for a few minutes then stopped being written to, and the processes sat
   idle but alive for 1+ hours. Likely GPU contention or Isaac Sim shared-state
   interference. Root cause unknown — no time was spent debugging during the
   run.

## Single-episode eval on 022000 (run after the 8h budget)

- Command: `/home/ubuntu/IsaacLab/isaaclab.sh -p scripts/train/run_vla_closed_loop.py --checkpoint checkpoints/continual/checkpoints/022000/pretrained_model --num_episodes 1 --max_steps 1200 --save_gif reports/eval_022000.gif`
- Ran in ~4 min (corrects our earlier mis-impression that eval takes 45-90 min
  — the previously "stuck" eval was in some hung state, not genuinely slow).
- Result: **0/1 success after 1200 steps**. GIF at `reports/eval_022000.gif`
  (15 MB, 300 frames). User should inspect to see what the policy actually does.

## Decisions for the next run

### 1. Scope
- **(A) Minimal fix + same strategy.** Fix bugs 4, 5, 6(auto-inject), 7-or-workaround.
  Keep LoRA r=16 + 25 demos. Another 8h run.
- **(B) Redesign training strategy.** Full fine-tune, or larger LoRA rank
  (32/64), or different scheduler, or more demos.
- **Recommendation: (A).** The previous run literally never trained past step
  22000 — we don't yet know what LoRA r=16 on 25 demos can do. Don't redesign
  until we've seen the first working training.

### 2. Fix double-PEFT-wrap (bug #4)
- **(a) Orchestrator-side.** Before resume, read saved `train_config.json`,
  blank the `peft` section, write it back. Policy config still has
  `use_peft=True` so `make_policy` still loads the adapter; but with
  `cfg.peft=None` the train script won't double-wrap.
- **(b) LeRobot-side.** Patch `lerobot_train.py` to skip `wrap_with_peft` when
  `cfg.policy.use_peft` is already True.
- **Recommendation: (a).** Less invasive; orchestrator already does similar
  config munging (adapter_fixer).

### 3. Fix LR schedule (bug #5)
- **(a) Much larger `scheduler_decay_steps`** (e.g. 200000).
- **(b) Constant LR** at e.g. 5e-5 — simplest, standard for continual fine-tune.
- **(c) Warm-restart on each resume** — peak LR for first N steps of each
  resumed epoch.
- **Recommendation: (b).** Simplest, predictable, avoids the "decays out"
  failure mode. Pass via CLI on fresh-start, preserved on resume via
  `train_config.json`.

### 4. Eval cadence
Given eval is actually ~4 min (not 45), this is cheap now.
- Keep serial eval, `EVAL_EVERY_N=5`, `num_episodes=1`, `max_steps=1200`.
- Budget watchdog's plateau detector will have enough signal.
- **Recommendation:** as above. No change needed.

### 5. Mimic batch stall (bug #7)
- **(a) Run Mimic first, then training.** Clean separation; guarantees a
  grown dataset. ~2-3h of Mimic up front to reach ~100-200 demos, then
  training has a real dataset.
- **(b) Investigate GPU contention.** Unknown time; not guaranteed to fix.
- **(c) Accept 25 demos, skip Mimic-2+.** Fastest but caps model quality.
- **Recommendation: (a).** Stage the pipeline: full Mimic generation first,
  THEN train against the frozen-but-larger dataset.

### 6. Config-save hygiene (bug #6)
- Extend `scripts/orchestrate/fix_adapter_configs.py` to *also* inject
  `type: smolvla` into every saved `config.json` (not just rewrite
  adapter_config's base_model). Makes every checkpoint directly
  evaluable by `run_vla_closed_loop.py` without manual intervention.

### 7. Time budget
- **(a) Full 8h run,** as before.
- **(b) Short smoke test first** (~2h): apply fixes; train a few resumes
  with the 25-demo dataset; visually confirm loss decreases through a
  resume boundary. Then if OK, commit to 8h.
- **Recommendation: (b).** Cheap verification that fixes #2 and #3 actually
  make training progress survive a resume. Doing that in a 2h run before
  burning another 8h saves one full cycle of "oh no, still broken".

## Recommended combined plan

If nothing further changes your mind:

**Step 1 — Smoke test (~2h).**
- Apply fixes (2a) + (3b constant-LR) + (6 auto-inject type).
- Run existing dataset (25 demos). No Mimic in background.
- Force a resume boundary every ~500 steps (low `--save_freq`, re-invoke from
  `last/`) to stress-test the resume chain.
- Success criterion: loss continues to decrease across resume boundaries,
  not back up to ~0.75.

**Step 2 — Full run (~8h).** Only if smoke passes.
- Phase 1: run Mimic alone to grow the dataset to ~100-200 demos (2-3h).
  Keep orchestrator from launching training until Mimic is done.
- Phase 2: training loop with resume, LoRA r=16, constant LR 5e-5, eval
  every 5 epochs, watchdog for plateau + 8h budget.
- At end: inspect `reports/epoch_summary.jsonl` for loss + eval_sr trends,
  evaluate `checkpoints/continual/checkpoints/last/pretrained_model`.

## Concrete files the next session should touch

- `scripts/orchestrate/continual_train.sh` — add pre-resume train_config.json
  munging; pass `--policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=0
  --policy.scheduler_decay_steps=1000000 --policy.scheduler_decay_lr=5e-5`
  (or equivalent — goal is constant LR); add a Mimic-only mode gated by env
  var like `MIMIC_ONLY=1`.
- `scripts/orchestrate/fix_adapter_configs.py` — also ensure `type: smolvla`
  key is present in every `pretrained_model/config.json`.
- `scripts/train/run_vla_closed_loop.py` — already patched this session, keep.
  Possibly make it the single source of truth for "load a lerobot PEFT
  checkpoint" and reuse in orchestrator's eval path.
- NEW: `scripts/orchestrate/prepare_resume_config.py` (or inline in
  continual_train.sh) — strips `peft:` section from saved
  `train_config.json` before each resume.

## Things the next session should NOT change (stable)

- Environment cfg (`envs/yaskawa_pick_cube_cfg.py`) — wrist cam re-aim + focal
  was the right move, current config works. Leave alone.
- Scripted pick script — 100% success, don't touch.
- Mimic env cfg (`envs/yaskawa_pick_cube_mimic_env_cfg.py`) — 100% success on
  batch 1 at action_noise=0.02, don't touch.
- Watchdog, adapter_fixer general design — both worked fine in the last run.

## Quick-start for the next session

```bash
cd /home/ubuntu/vla_kitting
# 1. Read this file + reports/continual_training_plan.md first
# 2. Apply fixes per Decision 2, 3, 6 above
# 3. Optionally run smoke test before 8h run
bash scripts/orchestrate/continual_train.sh --reset
# watchdog auto-launches via: .venv/bin/python scripts/orchestrate/budget_watchdog.py ...
# monitor: tail -f logs/continual/{orchestrator.out.N,train_loop.log,watchdog.log}
# stop cleanly: touch logs/continual/STOP
```

## Known state snapshot (as of end of this session)

- `checkpoints/continual/checkpoints/022000/pretrained_model/` — only
  genuinely trained checkpoint. Eval result: 0/1 on 1 episode, 1200 steps.
- `checkpoints/continual/checkpoints/023000..035000/` — ignore, effectively
  zero-init LoRA on base SmolVLA.
- `datasets/teleop/{cube_scripted,cube_scripted_clean,cube_annotated}.hdf5` —
  seed files, 25 demos each. Keep.
- `datasets/mimic/cube_mimic_all.hdf5` — 25 merged demos from batch_001.
- `datasets/mimic/pool/batch_002*.hdf5`, `batch_005*.hdf5` — partial/stuck
  files from Mimic stalls; delete or ignore.
- `datasets/lerobot/cube_pick_v1` → `cube_pick_v1_batch_001` — 25-demo LeRobot
  dataset. Works fine.
- Orchestrator + watchdog + adapter_fixer — all exited cleanly, nothing
  running now.
