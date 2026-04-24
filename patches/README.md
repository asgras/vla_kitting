# External repo patches

Local edits to three external clones that the pipeline depends on. Re-cloning
any of them reverts the patch and breaks the pipeline silently — reapply with
`git apply` from the target repo's root.

## `isaaclab-urdf-pin-relax.patch`

Target: `~/IsaacLab` @ `37ddf626` (tag `v2.3.2`)
Touches: `apps/isaaclab.python.kit`

Relaxes the `isaacsim.asset.importer.urdf` version pin from `2.4.31` exact to
unpinned so the GUI variant's dep solver accepts the version shipped in our
local Isaac Sim 5.0 at `/opt/IsaacSim/`.

Symptom if missing: GUI launches (teleop, `scene-inspect`) fail at startup
with `dependency: 'isaacsim.asset.importer.urdf' = { version='=2.4.31' } can't
be satisfied`. The headless `.kit` (Phase 5+) doesn't pin this extension and
keeps working.

Apply:
```bash
cd ~/IsaacLab && git apply /home/ubuntu/vla_kitting/patches/isaaclab-urdf-pin-relax.patch
```

## `lerobot-peft-resume-and-groot.patch`

Target: `~/code/lerobot` @ `97e7e0f9`
Touches:
- `src/lerobot/configs/default.py` — exposes `lora_alpha` / `lora_dropout` on `PeftConfig` (upstream only has `r`). Required to override via `--peft.lora_alpha=` / `--peft.lora_dropout=` in `train_only.sh`.
- `src/lerobot/policies/factory.py` — passes `is_trainable=True` to `PeftModel.from_pretrained` so resumed LoRA runs continue training the loaded adapter. Our orchestrator strips `cfg.peft` on resume (see `scripts/orchestrate/prepare_resume_config.py`) to avoid the double-wrap regression, which means the second `wrap_with_peft` call that normally re-enables grads never runs. Without this patch, `backward()` fails with `RuntimeError: element 0 of tensors does not require grad`.
- `src/lerobot/policies/groot/groot_n1.py` — adds `default=None` to four `field(init=False)` declarations on `GR00TN15Config`. Needed because `dataclasses.field(init=False)` without a default raises at import time on the Python / dataclasses version in our venv, breaking any `import lerobot.policies` even when we're not using groot.

Apply:
```bash
cd ~/code/lerobot && git apply /home/ubuntu/vla_kitting/patches/lerobot-peft-resume-and-groot.patch
```
