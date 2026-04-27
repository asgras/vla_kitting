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

## Authority to fix critical bugs autonomously

When you find a critical, clearly-diagnosable bug mid-run that has an obvious resolution, **fix it and continue without waiting for sign-off**. Examples that qualify:

- A misconfigured flag that silently invalidates the run (e.g. `load_vlm_weights=False` causing VLM to be random-init when we wanted to fine-tune the pretrained backbone — the fix is to add `--policy.load_vlm_weights=true` and relaunch).
- A typo / wrong-default value in code we wrote, where the correct value is unambiguous from context.
- A pipeline-blocking error with a single sensible recovery path (e.g. a stats.json missing → run the lerobot stats-compute step → continue).

The bar is: "would a competent collaborator who understood the goal need to ask?" If no, just fix it. Always document the bug, the fix, and why the resolution was unambiguous in the run diary so the audit trail survives.

What still needs sign-off:
- Architectural changes (action space, control rate, policy class).
- Plan changes that re-direct the run's hypothesis.
- Anything destructive on user data outside the current run's scope.
- When the diagnosis has more than one plausible resolution and the choice matters (then propose options).

## The things you must not do

- Do not silently re-run a training config that has already been tried. Check `reports/` first. If it has been tried, the new run needs a specific changed variable and a written hypothesis.
- Do not delete or overwrite demo datasets, checkpoints, or GIFs without archiving. Move stale artifacts under `reports/archive_*/` with a README explaining why they were retired.
- Do not tune hyperparameters mid-run by editing files in-place without recording the before/after and the reason. Long training runs with silent edits are unreproducible.
- Do not ship code changes to scripts under `scripts/train/` or `envs/` without noting them in the active experiment's report entry. Environment changes invalidate prior baselines.

## Where to look

- `reports/README.md` — index of live reports. Always start here.
- `reports/recovery_plan_2026-04-24.md` — the currently authoritative execution plan (supersedes prior next-steps docs if present). Phases past G3 have been advanced — verify status against the most recent dated report before acting.
- `reports/research_log_synthesis_2026-04-26.md` — comprehensive index of every experiment / ablation / diagnostic to date. Start here for "what has been tried."
- `reports/2026-04-26_overnight_session_summary.md` — most recent consolidated session summary (FXAA fix, v4 gripper-weight run, EE-positioning diagnosis).
- `reports/system_overview.md` — architecture snapshot; verify before citing (it has aged).
- `reports/known_issues.md` — running list of gotchas; append when you hit a new one.
- `~/.claude/projects/-home-ubuntu-vla-kitting/memory/` — auto-memory for cross-session facts about the user and project. Not a substitute for `reports/`; memory holds *meta* (preferences, references), reports hold *experimental record*.

## reports/ARCHIVE — do NOT reference

`reports/ARCHIVE/` holds historical instruction docs (old phase plans, "next steps" docs, superseded run reports, abandoned-branch investigations) that were correct at the time they were written but have since been superseded, executed, or abandoned. They are kept only as a paper trail for post-mortems and bisection.

**Rules for future sessions:**

- Do **not** read files in `reports/ARCHIVE/` to figure out what to do next. Their commands, file paths, env-cfg values, and CLI flags are stale and will lead you astray.
- Do **not** cite an archived document as a justification for an action.
- Do **not** restore files from `ARCHIVE/` back into `reports/`. If a piece of guidance there is still useful, copy the relevant text into a *new dated report* and re-validate it.
- Do **not** edit or extend files in `ARCHIVE/`. They are frozen.

If the only doc that seems to answer your question lives in `ARCHIVE/`, that means the question hasn't been re-answered in a current report yet. Treat it as a gap and write a fresh report rather than reviving the archived one.

`reports/ARCHIVE/README.md` lists every archived doc with a one-liner on why it was archived.

## Patches to external deps

Several local patches live against upstream repos (lerobot PEFT, IsaacLab kit). See memory entries `project_isaaclab_kit_patch.md` and `project_lerobot_peft_resume_patch.md`. If an upstream is re-pulled, these must be reapplied — and that reapplication counts as an experiment config change and gets logged.

## Commit hygiene

Commit messages should name the experiment the change serves, not just the diff. "Widen cube randomization to ±20cm for diversity_v2 run" beats "update env config." A future bisect needs to map commits to experimental state.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Land the plan (per-bd-issue closure protocol)

**Every "run" in this repo == one bd issue.** After you finish the work for an issue — before you say "done", before you move to the next issue, before you ask the user anything else — you MUST execute the Land the plan checklist below in order. No exceptions. The work is not complete until step 5 emits a handoff prompt and `git push` has succeeded.

### The five steps (do them in order, every time)

1. **Run quality gates.** If code changed, run the relevant tests and linters for the area you touched. Note results (pass/fail, which suite) in the bd issue's `--notes` or in the run's report entry. If a gate fails, fix it before proceeding — do not paper over with skips or `--no-verify`.
2. **File any remaining discovered work as bd issues.** Anything you noticed mid-run that's out of scope — a flaky test, a TODO you couldn't address, a follow-up experiment, a stale doc — gets a `bd create` *now*, with priority and a one-line description. Add `bd dep add` if it's blocked by something. Lost context = lost work.
3. **Close finished bd issues.** `bd close <id>` for the issue you just finished, plus any other issues this work resolved. Use `bd close <id1> <id2> ...` for multiples. If closing leaves something in a partial state, update the issue with `--notes` first, then close with `--reason`.
4. **Pull, sync, and push to remote.** This sequence is mandatory and must succeed. The current working branch (`vla-pipeline-v1`) tracks `origin/main` as its upstream, but the branch names differ — so plain `git push` is rejected by git's safety check. Use the explicit refspec form:
   ```bash
   git pull --rebase            # pulls from upstream (origin/main)
   bd dolt pull                  # OK if "no remote" — beads sync is local-only here
   bd dolt push                  # OK if "no remote"
   git push origin HEAD:main     # push current HEAD to origin/main (NOT plain `git push`)
   git status                    # MUST show "up to date with 'origin/main'"
   ```
   If push fails (rejected, conflicts, hook errors), resolve the root cause and retry. Do not stop with work stranded locally. Do not bypass hooks.
5. **Generate a handoff prompt for the next session.** After the issue is closed and pushed, emit a short handoff block as the *last* thing in your final reply for that issue. Format:
   ```
   ### Handoff — next session
   - Just closed: <bd-id> — <one-line summary of what shipped>
   - Repo state: branch <name>, HEAD <short-sha>, clean / dirty
   - Next ready issues (from `bd ready`): <id1>, <id2>, ...
   - Recommended next: <id> — <why this one is the right next step>
   - Open questions / risks: <anything the next session should know>
   - Resume command: `bd update <id> --claim && bd show <id>`
   ```
   The handoff is not optional — it is how the next session picks up cold without re-deriving context. Keep it tight: a future agent should be able to act on it in under 30 seconds.

### When to land the plan

- After **every bd issue you close**, even small ones. The protocol is per-issue, not per-session.
- If a single piece of work spans multiple bd issues, land the plan after the *last* one closes. Intermediate issues still get closed, but you can batch the push and handoff.
- If the user interrupts mid-issue and tells you to stop, still run steps 2–4 (file remainder, update status, push) before signing off, and emit a partial handoff noting the in-progress issue.

### Hard rules

- Work is NOT complete until `git push` succeeds AND the handoff prompt has been emitted.
- NEVER stop before pushing — that leaves work stranded locally.
- NEVER say "ready to push when you are" — YOU must push.
- NEVER skip the handoff because "the next session will figure it out." The next session is also you, with no memory.
- If you discover the issue you're closing was actually wrong-headed, say so in the close `--reason` and file a corrective issue in step 2 rather than silently abandoning.
<!-- END BEADS INTEGRATION -->
