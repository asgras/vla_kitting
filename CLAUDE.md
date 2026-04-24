# CLAUDE.md — vla_kitting

This file is the first thing Claude Code reads when it enters the repo. Follow it exactly.

## Prime directive: treat this repo as a running training experiment

Every change here — a config tweak, a new script, a dataset regeneration, a fresh training run — is an experiment. It has a hypothesis, a configuration, a cost, a result, and a lesson. **None of that survives in your head or in chat; it has to land in `reports/` or it did not happen.**

The policy does not improve because we wrote more code. It improves because we tell the next run what the last run proved or disproved. Treat the report trail as the product.

## Logging discipline (non-negotiable)

For every training run, dataset regeneration, ablation, or validation sweep:

1. **Before you start**, write a short entry under `reports/` named `<YYYY-MM-DD>_<short_slug>.md` capturing:
   - **Hypothesis** — one sentence: what you expect to change and why.
   - **Config** — exact command line, dataset path, checkpoint path, LoRA r/alpha/targets, LR, batch, decimation, seed. Do not paraphrase — paste the invocation.
   - **Baseline** — the run this is compared against (file path + key metric).
   - **Stop condition** — when you will call it done (epoch budget, wall-clock, success rate threshold, or a specific diagnostic).
2. **While it runs**, append observations inline (loss curve inflection, OOM, adapter-config fix firing, etc.). Timestamps help.
3. **After it finishes** (success OR failure OR killed), append:
   - **Result** — concrete numbers: train loss floor, eval SR (x/N and seeds), failure mode description, checkpoint path.
   - **Artifacts** — GIF filenames, JSONL episode files, log paths.
   - **Lesson** — one or two sentences the *next* experiment needs to know. Failures are as valuable as successes; log them with the same rigor.
   - **Next step** — the single next experiment this result implies, or "dead end — pursue X instead."

**Failures get logged the same as successes.** A run that crashed at epoch 3 with OOM is a data point about batch size. A run that trained fine but got 0/10 eval is a data point about data diversity. Silent failures (you just move on to the next idea) are how we keep repeating the same mistakes.

Existing exemplars to match in tone and density: `reports/overnight_run_2026-04-24.md`, `reports/15hz_investigation_2026-04-24.md`, `reports/phase_7_findings.md`.

## Report index

Maintain an up-to-date pointer file at `reports/README.md` (create if missing). When you add a new dated report, add one line: `- <date> — <slug>: <one-line takeaway>`. If an older report's conclusion has since been invalidated, annotate the index entry with "(superseded by <later report>)" rather than deleting.

## The things you must not do

- Do not silently re-run a training config that has already been tried. Check `reports/` first. If it has been tried, the new run needs a specific changed variable and a written hypothesis.
- Do not delete or overwrite demo datasets, checkpoints, or GIFs without archiving. Move stale artifacts under `reports/archive_*/` with a README explaining why they were retired.
- Do not tune hyperparameters mid-run by editing files in-place without recording the before/after and the reason. Long training runs with silent edits are unreproducible.
- Do not ship code changes to scripts under `scripts/train/` or `envs/` without noting them in the active experiment's report entry. Environment changes invalidate prior baselines.

## Where to look

- `CLAUDE_CODE_PLAN.md`, `VLA_KITTING_PLAN.md` — historical macro plans. Read once for context; do not take as current truth.
- `reports/recovery_plan_2026-04-24.md` — the current execution plan (supersedes prior next-steps docs if present).
- `reports/system_overview.md` — architecture snapshot; verify before citing.
- `reports/known_issues.md` — running list of gotchas; append when you hit a new one.
- `~/.claude/projects/-home-ubuntu-vla-kitting/memory/` — auto-memory for cross-session facts about the user and project. Not a substitute for `reports/`; memory holds *meta* (preferences, references), reports hold *experimental record*.

## Patches to external deps

Several local patches live against upstream repos (lerobot PEFT, IsaacLab kit). See memory entries `project_isaaclab_kit_patch.md` and `project_lerobot_peft_resume_patch.md`. If an upstream is re-pulled, these must be reapplied — and that reapplication counts as an experiment config change and gets logged.

## Commit hygiene

Commit messages should name the experiment the change serves, not just the diff. "Widen cube randomization to ±20cm for diversity_v2 run" beats "update env config." A future bisect needs to map commits to experimental state.
