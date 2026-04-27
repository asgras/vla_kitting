# 2026-04-27 — SmolVLA vision-input verification (bd vla_kitting-6l3)

**Type:** sanity check; not a run. Prereq for the broader best-practices audit (bd vla_kitting-18n).

## Hypothesis

After bumping camera resolutions (wrist 128→256, third 256→512) and re-aiming the third-person camera in the [2026-04-27_camera_resolution_and_framing](2026-04-27_camera_resolution_and_framing.md) change, our `observation.images.{wrist,third_person}` streams should still meet every contract that SmolVLA's preprocessing assumes. Specifically, the [0,1]-float / RGB / channel-first / square-aspect contract that flows from LeRobot dataset load → `SmolVLAPolicy.prepare_images` → SigLIP must hold without surprise.

## What SmolVLA expects (verified by source)

References: `/home/ubuntu/code/lerobot/src/lerobot/policies/smolvla/{configuration,modeling,processor}_smolvla.py`, `/home/ubuntu/code/lerobot/src/lerobot/datasets/utils.py`.

| Contract | Expectation | Source |
|---|---|---|
| Tensor layout | `(B, C, H, W)`, `C=3`, channel-first | `modeling_smolvla.py:138` (asserts ndim==4) |
| Dtype | `float32` | dataset `to_tensor` path |
| Pixel range on input to model | `[0.0, 1.0]` | `modeling_smolvla.py:404` docstring + `:423` (`img * 2.0 - 1.0` to map → `[-1, 1]` for SigLIP) |
| Channel order | RGB | `datasets/utils.py:411` (`PILImage.open(...).convert("RGB")`) |
| Resize / pad target | `resize_imgs_with_padding=(512, 512)` (default) | `configuration_smolvla.py:48` |
| Resize algorithm | bilinear, preserve aspect, pad **left+top** with `value=0` | `modeling_smolvla.py:135-153` |
| Visual normalization mode | `IDENTITY` (no per-image mean/std; the `[0,1]→[-1,1]` map is the entire normalization) | `configuration_smolvla.py:36-43` |
| Backbone resolution | 512×512 fed to SigLIP-derived vision tower in SmolVLM2-500M | `configuration_smolvla.py:91` `vlm_model_name` |
| Token budget for task prompt | `tokenizer_max_length=48` | `configuration_smolvla.py:69` |

## Our pipeline

| Stage | Output |
|---|---|
| Render (Isaac Lab `CameraCfg`) | `wrist=(256,256,3)` uint8 RGB, `third=(512,512,3)` uint8 RGB |
| Save (LeRobotDataset, `--use_videos=False`) | PNG (lossless RGB uint8) |
| Load (`datasets/utils.py:411-439`) | `PIL → ToTensor() → (3, H, W) float32, [0, 1]` |
| `prepare_images` resize | `(3, 512, 512)`; details below |
| `prepare_images` normalize | `* 2 - 1 → [-1, 1]` |

### Resize math under our shapes

`resize_with_pad((B, 3, H, W), 512, 512, pad_value=0)`:

- **third (512×512):** `ratio = max(512/512, 512/512) = 1.0` → no-op resize, no pad.
- **wrist (256×256):** `ratio = max(256/512, 256/512) = 0.5` → resized to `(512, 512)`, no pad. (2× bilinear upsample.)

Neither stream takes any **left-or-top zero padding** — both are square at the target. This matters because the padding is asymmetric (left+top only), which can shift the visual content within SigLIP's positional embedding grid; we avoid it entirely.

## Result — green-light, with one watchout

| Item | Status | Evidence |
|---|---|---|
| `(B,3,H,W)` channel-first contract | ✅ | LeRobot `to_tensor` produces `(C,H,W)` |
| `float32 [0,1]` contract | ✅ | `to_tensor` divides by 255 |
| RGB channel order | ✅ | `PIL.Image.convert("RGB")` |
| Square aspect, no asymmetric pad | ✅ | both 1:1, both ≤ 512 → no pad branch |
| 512×512 native on third-person | ✅ | NEW, native pixels with no resize |
| 256×256 native on wrist | ⚠️ Acceptable | bilinear 2× upsample inside `resize_with_pad`; harmless but doesn't add information |
| Task prompt under token budget | ✅ | `"pick up the <color> cube and place it on the magenta circle"` ≈ 14 tokens, budget 48 |
| `chunk_size=50, n_action_steps=50, n_obs_steps=1` defaults vs our use | ✅ | unchanged from default in our train cmds |

**No discrepancies block the next run.** The wrist 2× upsample is the only sub-optimal bit — if we want to eliminate it we'd render wrist at 512 native, but a 4× pixel volume increase per frame for a close-range camera is rarely a good trade. Kept at 256.

## Watchouts the audit (vla_kitting-18n) should pick up next

These are out-of-scope for this verification but I noted them while reading the source — flagging now so the audit doesn't have to re-discover:

1. **`load_vlm_weights` default is `False`.** Already in memory (`project_smolvla_load_vlm_weights_gotcha`). Ensure every `train_only.sh` / orchestrate cmd passes `--policy.load_vlm_weights=true` for fine-tunes.
2. **`freeze_vision_encoder=True` and `train_expert_only=True`** are the SmolVLA fine-tune defaults. Our v4 ran with vision-tower LoRA, which deviates. The audit should weigh "does our vision-grounding deficit (see 2026-04-27 v4 vd0/uxt finding) justify breaking this default" against the upstream recommendation.
3. **Visual normalization is `IDENTITY`.** Our `dataset_stats.json` does not need (and should not provide) image mean/std. Verify the converter's stats step does not emit per-image normalization stats that would ever silently flip mode.
4. **Asymmetric left+top zero-pad** is benign for us (we're square at target) but anyone widening the camera aspect ratio should know about it.

## Lesson

The biggest gotcha here was the *padding asymmetry* in `resize_with_pad` (lines 135-153). For non-square cameras, content shifts toward the bottom-right corner of the SigLIP grid, which interacts with positional bias in the way that the [2026-04-26 attention_diagnostic_invalidated](2026-04-26_attention_diagnosis_and_v5_plan.md) finding warned us about. We dodge it by rendering 1:1, but it's worth documenting that decision: keeping both cameras square is now a load-bearing assumption, not a stylistic choice.

## Next step

Proceed to bd vla_kitting-18n (best-practices audit).
