# Reports index

One-line pointers to dated experimental reports and standing reference docs. Add a new line here whenever you write a new dated report under `reports/`.

> **About `reports/ARCHIVE/`:** historical, superseded, do-not-consult. See
> `ARCHIVE/README.md` and the rules in `CLAUDE.md`. Archived docs are not
> listed below.

## Standing reference docs (no date — read first)

- [recovery_plan_2026-04-24.md](recovery_plan_2026-04-24.md) — currently authoritative execution plan (Phase 0 → 5, gates G0-G5). Phases past G3 have been advanced under v3/v4; verify status against the most recent dated report before acting.
- [system_overview.md](system_overview.md) — architecture & design intuition (verify before citing — has aged)
- [known_issues.md](known_issues.md) — running list of gotchas (wrist cam, etc.)

## Synthesis / cross-cutting

- 2026-04-26 — research_log_synthesis: comprehensive index of every experiment / ablation / diagnostic to date, with results and lessons. **Start here for "what has been tried."**

## Dated experiment reports (live)

- 2026-04-26 — eval_ghost_cube_investigation: DLSS render-history was synthesizing ghost cubes from prior-reset positions in eval frames; FXAA fix at envs/yaskawa_pick_cube_cfg.py:315 — postfix samples clean. v3.2 ep22 eval with fix STILL 0/10 (gripper never closes — Hole B confirmed)
- 2026-04-26 — runs/v4_gripper_weight_2026-04-26/run_diary: v4 fresh restart with FXAA fix + action_loss_dim_weights gripper × 8 → × 16. Trained to epoch 48, 0/N real SR throughout. Diagnostic action log proved bottleneck is **EE positioning precision, NOT gripper learning** (gripper does emit ±1 cleanly).
- 2026-04-26 — attention_diagnosis_and_v5_plan: cross-attention overlay built, ranked v5 fixes (augmentations, wider distribution, aux cube-localization loss). **Quantitative peak/uniform claims SUPERSEDED by 2026-04-26_attention_diagnostic_invalidated below.** The v5 fix list itself is still valid.
- 2026-04-26 — attention_diagnostic_invalidated: argmax-on-image alignment check showed the heatmap peaks land on register-token positional biases (e.g., "pink" argmax always at top-right corner), NOT on the cube or magenta target. Prior peak/uniform numbers are dominated by positional bias, not semantic attention. Need attention-difference diagnostic (subtract across cube positions) before quoting any "model attends to cube" claim.
- 2026-04-26 — overnight_session_summary: **single-file consolidated report for the 2026-04-26 night.** Read first. Contains ghost-cube root cause + fix, full v4 loss/SR trail, action-log diagnostic, recommended next experiments.
- 2026-04-26 — eval_harness_fixed_seed_30: T1 / vla_kitting-7ky landed. Canonical 30-position cube_xy set in `configs/eval_seed_30.json` (seed 42, widened box X∈[0.40,0.70], Y∈[-0.22,0.22]). Use for all future SR comparisons.
- 2026-04-26 — scene_data_integrity_pack: vla_kitting-y5b epic landed end-to-end. Table 1.5×1.0 m, target = 10 cm magenta CylinderCfg, prompt = "magenta circle", cube yaw ±0.5 rad with yaw-aware scripted controller, per-episode color injection via shared palette + cube_color_idx obs. Gate G1 = 30/30 scripted SR on widened box. Action variance shows ee_dx/ee_dy DIVERSE — saturated-P-controller hypothesis FALSIFIED. Target color stable across 5 cube colors (max pairwise L2=0.043, 4% of inheritance baseline).
- 2026-04-26 — run_b_prime_config_diff: vla_kitting-hzj. SR re-eval deferred (env changed since RunB trained). Config delta surfaces 4 prime suspects: r=16→64 + α/r=0.5→1.0, vision-tower LoRA on, modules_to_save=[action_out_proj], gripper×16 loss weight. Recommends v5 = revert all four to RunB regime as single change.
- 2026-04-27 — v4 vision-grounding diagnostics: vd0 trajectory overlay = FAN-but-weak (mean_var 0.0146 m², corr_X=0.15 negligible, corr_Y=0.61 moderate). uxt attention-difference = INVARIANT (median residual-argmax err 147 px on 256² image; vision content not localizing cube). Combined finding: v4's third-person vision is NOT being used as a cube-localization signal — depth-precision deficit and corr_X≈0 are both explained by this. v5+ recommendations: aux cube-localization loss OR vision-tower full fine-tune. Artifacts: reports/runs/v4_trajectory_overlay_2026-04-27/, reports/runs/attn_diff_2026-04-27/.
- 2026-04-27 — scripted_only_data_pipeline: **Decision record.** Mimic generation is OUT of the data pipeline. v5 and onward train on scripted demos exclusively. Recovery plan Phase 2 amended in place, bd issues 8ux/mil cleaned of Mimic forward-looking language, Mimic env+orchestration files preserved in tree as historical record only.
- 2026-04-27 — camera_resolution_and_framing: env-cfg change (NOT a run). wrist 128→256, third-person 256→512, third-person re-aimed from `(1.15,0.10,0.50)` f=24 to `(1.5,-0.10,0.80)` f=18 (FOV ~47°→60°) so the full spawn box + target marker stay in-frame. Verified across 8 corner-spanning random poses. **Invalidates all prior datasets/checkpoints that consumed wrist_cam/third_person_cam.**
- 2026-04-27 — smolvla_vision_input_verification: bd 6l3. Confirmed our wrist (256²) + third-person (512²) RGB streams meet every SmolVLA preprocessing contract — float32 [0,1], channel-first, RGB, square aspect avoiding the `resize_with_pad` left+top pad branch. Greenlit, with 4 watchouts forwarded to the audit (load_vlm_weights, freeze_vision_encoder, IDENTITY visual norm, asymmetric pad).
- 2026-04-27 — smolvla_best_practices_audit: bd 18n. Compared our train_only.sh against paper + tutorial + model card. **3 concerns to fix before next run:** (1) batch=4 vs recommended 64 (16× under), (2) vision-tower LoRA contradicts paper canonical AND v4 diagnostics show vision isn't being used, (3) LoRA LR may be 10× too low. ~12 items aligned, ~6 acceptable divergences with documented rationale.
- 2026-04-27 — mu7_vision_probe_new_camera: bd vla_kitting-mu7. Spatial-aware (3×3 patch ROI on projected cube center) re-run of base SmolVLA vision probe at NEW camera config (512², N=100). **VERDICT = PARTIAL** (R²_x=+0.852 ✓, R²_y=+0.473 ✗ vs threshold ≥0.7; on/off cos-sim=0.658 ✓; color silhouette=+0.277). Supersedes k98's MISSING — k98 was a mean-pool readout artifact, not an encoder limitation. Y-axis weakness is camera-geometry / spawn-box aspect-ratio, not vision-encoder failure. **v5 (vla_kitting-8ux) → aux cube-loc loss at 0.05× action weight; vision tower stays frozen.**
- 2026-04-27 — 2hp_scripted_phase_stretch: bd vla_kitting-2hp. Phase A 70→200, C 100→200 (B=15 unchanged) in `scripts/validate/scripted_pick_demo.py`. Pre-grasp frames per demo 185 → 415 (+124%); centrally-visible frames ~30 → 415 (~14×). Smoke 3/3, demo length 910 → 1063 steps. Required script-level `episode_length_s=45` override (default 30 s would truncate). Unblocks v5 dataset regen for vla_kitting-8ux.
- 2026-04-24 → 2026-04-26 — runs/vision_grounded_30hz_2026-04-24/run_diary: v2 (15 Hz wide-box Mimic, aborted at 25 % Mimic SR), v3.0 (30 Hz, drop_cube_pos, frozen vision; 0/N), v3.1 (+ vision LoRA; 0/N), v3.2 (+ n_action_steps=10; 0/N) and post-v3.2 chunk/RTC sweep + obs_compare diagnostic

## Subdirectories

- `ARCHIVE/` — historical instruction docs and superseded reports. Do not consult; see `ARCHIVE/README.md`.
- `archive_25demo_run/`, `archive_60hz_run/`, `archive_*` — raw JSONL eval data for older runs preserved for comparison
- `prompt_ab/` — single-sample prompt-AB probe ("pink_square") on Run-B ep10 checkpoint (0/1)
- `runs/` — structured per-run subdirectories (v3 has its own diary, eval GIFs, sweep JSONLs)
- `saved_checkpoints/` — preserved best-SR adapters (currently r16_epoch10_sr0.10/)
- `camera_samples/` — wrist+third-person sample frames for camera FOV verification
