# 2026-04-26 — Fixed-seed 30-position eval harness (T1 / vla_kitting-7ky)

## Hypothesis
Cross-run SR comparison is unreliable today because every prior eval used a
different N and seed list. Need a single canonical 30-position cube-xy set
with deterministic generation so any future SR number is comparable head-to-head.

## Config
- Generator: `scripts/orchestrate/build_eval_seed_set.py`
- Output: `configs/eval_seed_30.json`
- Seed: `np.random.default_rng(42)`
- N: 30
- Sampling box (absolute world coords, matches widened env pose_range):
  - X ∈ [0.40, 0.70]
  - Y ∈ [-0.22, 0.22]
- Positions are stored both as a `[[x, y], …]` list and as a
  pre-formatted `"x,y;…"` string under key `cube_xy_string` so callers
  can `jq -r .cube_xy_string` and pipe directly to `--cube_xy`.

## Eval invocation (canonical)
```bash
CKPT=<path/to/pretrained_model>
CUBE_XY=$(python -c 'import json; print(json.load(open("configs/eval_seed_30.json"))["cube_xy_string"])')
./isaaclab.sh -p scripts/train/run_vla_closed_loop.py \
    --checkpoint "$CKPT" \
    --num_episodes 30 \
    --max_steps 900 \
    --cube_xy "$CUBE_XY" \
    --jsonl_out reports/eval_seed30/<run_tag>.jsonl \
    --ckpt_tag <epoch_or_run_tag> \
    --save_gif reports/eval_seed30/<run_tag>_ep{ep}.gif
```

The 30 episodes cycle through the cube_xy list in order
(`run_vla_closed_loop.py:233`), so episode index ↔ position index is fixed.

## Result
- Determinism check: `md5sum configs/eval_seed_30.json` matched across two
  fresh generations.
- X span observed: [0.413, 0.693], Y span: [-0.217, 0.206] — covers >90%
  of the widened box on both axes.

## Artifacts
- `configs/eval_seed_30.json`
- `scripts/orchestrate/build_eval_seed_set.py`

## Lesson
Tiny pre-generated JSON is enough; no need for a Python module import for
something this static. The `cube_xy_string` convenience field saves every
caller from re-implementing the join.

## Next step
Use this set for vla_kitting-hzj (re-eval Run B prime checkpoint) and
vla_kitting-vd0 (trajectory-overlay test). Once T6 (table widening) and
T8 (target circle) land, re-bake the seed set if the cube box widens
further — record that as a separate experiment with a new dated report.
