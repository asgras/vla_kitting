# reports/ARCHIVE — historical, do not consult

**Read this first if you opened a file in this directory.**

Everything in `reports/ARCHIVE/` is **out of date**. The documents here describe
plans, hypotheses, configurations, file paths, and "next steps" that were
correct at the time they were written but have since been **superseded,
abandoned, or executed**. They are kept only as a paper trail for
post-mortems and bisection. They are NOT current truth.

## Rules

1. **Do not cite an archived document as a source for what to do next.**
   The next-step instructions inside have been replaced by newer reports
   under `reports/` (see `reports/README.md`).
2. **Do not re-run a command from an archived doc** without first
   verifying against the current code that the file paths, env-cfg
   values, and CLI flags still match. Many do not.
3. **Do not move files out of this directory** to "revive" them. If a
   piece of guidance here is still useful, copy the relevant text into a
   new dated report under `reports/` and re-validate it as part of the
   new entry.
4. **Do not extend or update files in this directory.** They are frozen.
   Make a new dated report instead.

## Why these were archived (one-liners)

| File | Why archived |
|------|--------------|
| `CLAUDE_CODE_PLAN.md` | Phase 0–13 kickoff plan; phases 0–9 are done, the rest are obsoleted by the SmolVLA path actually taken. |
| `VLA_KITTING_PLAN.md` | Pre-SmolVLA architecture proposal (Octo-Small / Octo-Base). The repo went a different direction. |
| `scene_inspection.md` | Phase-0 USD inventory snapshot of `hc10dt_with_gripper_v1.usd`. Reproducible via `scripts/validate/scene_inspect.py` if needed. |
| `phase_6_viewport_handoff.md` | Isaac Sim 5.1 viewport-freeze blocker; teleop was abandoned in V1, replaced by the scripted-pick + Mimic path. |
| `phase_7_findings.md` | Gripper physics breakthrough (gear_assembly actuator gains + `disable_gravity` + `drive_type=force`). Findings now live in code; the narrative is in `reports/research_log_synthesis_2026-04-26.md`. |
| `continual_training_plan.md` | Apr-22 design of the Mimic + train continual loop. Loop was rebuilt, this doc no longer matches the orchestrator. |
| `next_training_plan.md` | Apr-22 next-steps after the first 8 h continual run. All listed bugs were fixed and the recommendations have been executed. |
| `next_steps.md` | Apr-23 Mimic-recovery + 60 Hz training launch. Run completed; superseded by overnight_run_2026-04-23 and later. |
| `next_steps_2026-04-24.md` | Apr-24 post-15Hz revert plan. Explicitly superseded by `reports/recovery_plan_2026-04-24.md`. |
| `overnight_run_2026-04-23.md` | Run B (60 Hz, r=16, 128 demos). Superseded by run B′ on 2026-04-24. |
| `overnight_run_2026-04-24.md` | Run B′ (60 Hz, r=32 broader LoRA). Numbers are still cited in synthesis; full doc is archived because its "next steps" are stale. |
| `15hz_investigation_2026-04-24.md` | 15 Hz branch shelved — root cause of the post-hoc 60→15 conversion bug is now in synthesis, evidence GIFs remain in `reports/`. |
| `ultraplan_context_2026-04-23.md` | One-time context dump for an ultraplan session. Frozen. |

## Where to look instead

For the **current** state of the project, see `reports/README.md` (the
index of live reports) and `CLAUDE.md` (project discipline). The
authoritative-at-the-time-of-writing execution plan is
`reports/recovery_plan_2026-04-24.md`; partial supersessions are noted
in the most recent dated reports under `reports/`.
