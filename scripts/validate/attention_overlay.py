"""Cross-attention overlay for a SmolVLA checkpoint — pure PyTorch (no Isaac).

Loads the policy, hooks the eager attention forward to capture per-layer
attention probs, then runs ONE forward pass on a saved third-person +
wrist camera frame pair (and a synthetic state vector). Extracts attention
from action-query positions to vision-token positions, averages over
heads + the last K layers, reshapes to the vision-encoder patch grid,
upsamples to the input resolution, and saves a side-by-side PNG of
(raw frame | attention heatmap | overlay).

Usage:
    /home/ubuntu/vla_kitting/.venv/bin/python scripts/validate/attention_overlay.py \
        --checkpoint checkpoints/continual/checkpoints/last/pretrained_model \
        --third_png reports/runs/v4_gripper_weight_2026-04-26/final_debug/ghost_check/sample_00_third.png \
        --wrist_png reports/runs/v4_gripper_weight_2026-04-26/final_debug/ghost_check/sample_00_wrist.png \
        --tag epoch48_sample00
"""
from __future__ import annotations
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/ubuntu/code/lerobot/src")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--third_png", type=str, required=True,
                    help="path to a third-person camera PNG (256x256x3).")
parser.add_argument("--wrist_png", type=str, required=True,
                    help="path to a wrist camera PNG (128x128x3).")
parser.add_argument("--task", type=str,
                    default="pick up the cube and place it on the magenta circle")
parser.add_argument("--tag", type=str, default="overlay")
parser.add_argument("--out_dir", type=str,
                    default=str(REPO / "reports" / "runs" /
                               "v4_gripper_weight_2026-04-26" / "final_debug" / "attn"))
parser.add_argument("--last_k_layers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--state_csv", type=str, default=None,
                    help="optional: path to a CSV row from action_log.csv to "
                         "pull ee_pose for synthetic state. If unset, uses "
                         "all-zeros state (model still runs but state-grounded "
                         "attention may be off).")
parser.add_argument("--text_query", type=str, default=None,
                    help="comma-separated words from the task string to "
                         "render text→vision attention for. Each word's "
                         "token-position attention to vision tokens is "
                         "extracted from the prefix-only forward pass. "
                         "Example: --text_query 'cube,magenta,circle'")
parser.add_argument("--base_only", action="store_true",
                    help="If --checkpoint points at a LoRA adapter dir, "
                         "load the base SmolVLA from HF Hub WITHOUT applying "
                         "the adapter. Uses the local config.json's "
                         "input_features so the schema matches the fine-"
                         "tuned model. Lets you compare base vs fine-tuned "
                         "attention on identical inputs.")
args_cli = parser.parse_args()


def _log(msg: str) -> None:
    print(f"[attn] {msg}", flush=True)


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import smolvlm_with_expert as swe

    out_dir = pathlib.Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args_cli.device)

    _log(f"loading policy from {args_cli.checkpoint} (base_only={args_cli.base_only})")
    ckpt_path = pathlib.Path(args_cli.checkpoint)
    if (ckpt_path / "adapter_config.json").exists():
        from peft import PeftModel
        from peft import PeftConfig as HfPeftConfig
        from lerobot.configs.policies import PreTrainedConfig
        hf_peft_cfg = HfPeftConfig.from_pretrained(str(ckpt_path))
        base_src = hf_peft_cfg.base_model_name_or_path
        local_cfg = PreTrainedConfig.from_pretrained(str(ckpt_path))
        _log(f"  PEFT base={base_src}")
        base_policy = SmolVLAPolicy.from_pretrained(base_src, config=local_cfg)
        if args_cli.base_only:
            policy = base_policy
            _log("  --base_only: skipping PEFT adapter; using BASE weights with local input schema")
        else:
            policy = PeftModel.from_pretrained(base_policy, str(ckpt_path))
    else:
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()
    _log("policy loaded")

    preprocess, postprocess = make_pre_post_processors(
        policy.config, args_cli.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ---------- monkey-patch eager attention to capture probs --------------
    captured: list[torch.Tensor] = []
    orig_eager = swe.SmolVLMWithExpertModel.eager_attention_forward

    def hooked_eager(self, attention_mask, batch_size, head_dim, q, k, v):
        num_att_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads
        groups = num_att_heads // num_kv_heads
        seq_len = k.shape[1]
        kk = k[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, groups, head_dim)
        kk = kk.reshape(batch_size, seq_len, num_kv_heads * groups, head_dim)
        vv = v[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, groups, head_dim)
        vv = vv.reshape(batch_size, seq_len, num_kv_heads * groups, head_dim)
        q_ = q.to(dtype=torch.float32)
        k_ = kk.to(dtype=torch.float32)
        q_ = q_.transpose(1, 2)
        k_ = k_.transpose(1, 2)
        att = torch.matmul(q_, k_.transpose(2, 3)) * (head_dim ** -0.5)
        att = att.to(dtype=torch.float32)
        big_neg = torch.finfo(att.dtype).min
        masked = torch.where(attention_mask[:, None, :, :], att, big_neg)
        probs = torch.nn.functional.softmax(masked, dim=-1)
        captured.append(probs.detach().to("cpu", dtype=torch.float32))
        probs = probs.to(dtype=vv.dtype)
        att_out = torch.matmul(probs, vv.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        att_out = att_out.reshape(batch_size, -1, num_kv_heads * groups * head_dim)
        return att_out

    swe.SmolVLMWithExpertModel.eager_attention_forward = hooked_eager

    # ---------- load images and build synthetic batch ---------------------
    third_np = np.asarray(Image.open(args_cli.third_png).convert("RGB"))
    wrist_np = np.asarray(Image.open(args_cli.wrist_png).convert("RGB"))
    _log(f"third {third_np.shape}, wrist {wrist_np.shape}")

    # Synthetic state matching the dataset schema (12 dims) + ee_pose (7 dims).
    # The model's behavior depends on state too — passing zeros means the
    # heatmap reflects "policy attention given vision + zeroed state at home
    # pose-ish" which is fine for a first-look diagnostic.
    state = np.zeros((12,), dtype=np.float32)
    ee_pose = np.zeros((7,), dtype=np.float32)
    ee_pose[3] = 1.0  # quaternion identity w
    if args_cli.state_csv:
        # Optional: pull state from a row of action_log.csv (ee_x, ee_y, ee_z)
        import csv
        with open(args_cli.state_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[0]
            ee_pose[0] = float(r["ee_x"])
            ee_pose[1] = float(r["ee_y"])
            ee_pose[2] = float(r["ee_z"])
            _log(f"using ee_pose from CSV row 0: {ee_pose[:3]}")

    raw = {
        "observation.state": state,
        "observation.ee_pose": ee_pose,
        "observation.images.wrist": wrist_np.astype(np.uint8),
        "observation.images.third_person": third_np.astype(np.uint8),
    }

    from lerobot.policies.utils import prepare_observation_for_inference
    frame = prepare_observation_for_inference(
        observation=raw, device=device, task=args_cli.task,
        robot_type="yaskawa_hc10dt_robotiq_2f85",
    )
    batch = preprocess(frame)
    _log("batch built; running select_action ...")

    captured.clear()
    with torch.no_grad():
        action = policy.select_action(batch)
    if torch.is_tensor(action):
        action_np = postprocess(action).detach().cpu().numpy() if hasattr(postprocess(action), "detach") else postprocess(action)
    else:
        action_np = action
    _log(f"action: {np.asarray(action_np).flatten()[:7]}")
    _log(f"captured {len(captured)} attention tensors")
    if not captured:
        _log("ERROR: no attention captured")
        return 1

    # captured[i] shape: (B, H, Q, K). For prefix-only fill: Q==K==prefix_len.
    # For suffix-only inference: Q=suffix_len, K=prefix_len+suffix_len. The
    # SUFFIX-ONLY forward is the one whose queries are action tokens — that's
    # what we want.
    shapes = [(t.shape[-2], t.shape[-1]) for t in captured]  # (Q, K)
    qk_set = sorted(set(shapes))
    _log(f"(Q,K) distribution: {qk_set[:6]}")
    chunk_size = int(policy.config.chunk_size)
    suffix_len = chunk_size + 1
    # Find the (Q, K) where Q == suffix_len.
    target_shape = next((qk for qk in qk_set if qk[0] == suffix_len), None)
    if target_shape is None:
        # Fall back to the unique (Q, K) with smaller Q.
        target_shape = min(qk_set, key=lambda x: x[0])
    _log(f"target (Q, K) = {target_shape}")
    full_layers = [t for t in captured if (t.shape[-2], t.shape[-1]) == target_shape]
    _log(f"matching layers: {len(full_layers)}")
    full_len = target_shape[1]
    prefix_len = full_len - suffix_len
    _log(f"chunk_size={chunk_size} suffix_len={suffix_len} prefix_len={prefix_len}")

    # Probe number of vision tokens per image by calling embed_image directly.
    # PEFT wraps the policy so policy.model.vlm_with_expert breaks; reach
    # through .base_model.model.* to find the actual SmolVLA module.
    def _find_vlm_with_expert(p):
        # Walk a few common attribute paths to find the underlying VLA model.
        candidates = [
            lambda x: x.model.vlm_with_expert,
            lambda x: x.base_model.model.model.vlm_with_expert,
            lambda x: x.base_model.model.vlm_with_expert,
        ]
        for fn in candidates:
            try:
                return fn(p)
            except Exception:
                continue
        return None
    try:
        img_keys = [k for k in batch if "images" in k]
        if img_keys:
            sample = batch[img_keys[0]]
            if isinstance(sample, torch.Tensor):
                _log(f"image tensor shape from batch: {sample.shape}")
                vlm = _find_vlm_with_expert(policy)
                if vlm is None:
                    raise RuntimeError("no vlm_with_expert path found")
                with torch.no_grad():
                    n_tokens = vlm.embed_image(sample.to(device)).shape[1]
            else:
                n_tokens = 64
        else:
            n_tokens = 64
    except Exception as e:
        _log(f"embed_image probe failed: {e}; falling back to 64")
        n_tokens = 64
    _log(f"tokens_per_image = {n_tokens}")
    # Also probe the second image's count (different size = different count).
    n_tokens_img2 = n_tokens
    try:
        if len(img_keys) >= 2:
            sample2 = batch[img_keys[1]]
            if isinstance(sample2, torch.Tensor):
                _log(f"image2 tensor shape from batch: {sample2.shape}")
                vlm = _find_vlm_with_expert(policy)
                if vlm is not None:
                    with torch.no_grad():
                        n_tokens_img2 = vlm.embed_image(sample2.to(device)).shape[1]
    except Exception as e:
        _log(f"image2 probe failed: {e}; using same as img1")
    _log(f"tokens_per_image2 = {n_tokens_img2}")

    add_special = bool(getattr(policy.config, "add_image_special_tokens", False))
    img1_s = (1 if add_special else 0)
    img1_e = img1_s + n_tokens
    img2_s = img1_e + (2 if add_special else 0) + (1 if add_special else 0)
    # Simpler: img2 starts right after img1 (with potential special tokens between)
    img2_s = img1_e + (2 if add_special else 0)
    img2_e = img2_s + n_tokens_img2
    _log(f"img1 vision: [{img1_s}:{img1_e})  img2 vision: [{img2_s}:{img2_e})")
    # In the suffix-only forward, queries are [0:chunk_size] = action chunk,
    # then [chunk_size] = time token. Use only the action queries.
    action_q_s = 0
    action_q_e = chunk_size
    _log(f"action queries (suffix-frame): [{action_q_s}:{action_q_e})")

    # Aggregate. Stack last K layers; replace NaN with 0 for any fully-masked
    # queries (shouldn't happen for action positions but safe).
    last_k = full_layers[-args_cli.last_k_layers:]
    stacked = torch.stack(last_k, dim=0)  # (L,B,H,Q,K)
    stacked = torch.nan_to_num(stacked, nan=0.0)
    mean_attn = stacked.mean(dim=(0, 2))[0]  # (Q,K)
    action_attn = mean_attn[action_q_s:action_q_e]  # (chunk, K)
    row_sums = action_attn.sum(dim=-1)
    _log(f"action query row_sums: min={row_sums.min().item():.4f} max={row_sums.max().item():.4f} mean={row_sums.mean().item():.4f}")
    valid_rows = row_sums > 1e-6
    _log(f"valid action rows: {int(valid_rows.sum())} / {action_attn.shape[0]}")
    if valid_rows.any():
        action_attn = action_attn[valid_rows]
    per_key = action_attn.mean(dim=0)  # (K,)
    img1_vec = per_key[img1_s:img1_e].numpy()
    img2_vec = per_key[img2_s:img2_e].numpy()
    _log(f"per_key sum (should be ~1 per row): {per_key.sum().item():.4f}")
    _log(f"per_key min/mean/max: {per_key.min().item():.6f}/{per_key.mean().item():.6f}/{per_key.max().item():.6f}")
    _log(f"img1 region (wrist) sum: {img1_vec.sum():.4f}, mean: {img1_vec.mean():.6f}, max: {img1_vec.max():.6f}")
    _log(f"img2 region (third) sum: {img2_vec.sum():.4f}, mean: {img2_vec.mean():.6f}, max: {img2_vec.max():.6f}")
    text_region = per_key[img2_e:prefix_len].numpy()
    _log(f"text+state region sum: {text_region.sum():.4f}, len: {text_region.shape[0]}")
    suffix_region = per_key[prefix_len:].numpy()
    _log(f"suffix-self region sum: {suffix_region.sum():.4f}, len: {suffix_region.shape[0]}")

    def _heatmap(vec, target_hw):
        n = vec.shape[0]
        side = int(round(n ** 0.5))
        if side * side == n:
            grid = vec.reshape(side, side)
        else:
            best = (1, n)
            for h in range(1, int(n ** 0.5) + 1):
                if n % h == 0:
                    w = n // h
                    if abs(h - w) < abs(best[0] - best[1]):
                        best = (h, w)
            grid = vec.reshape(best[0], best[1])
        gmin, gmax = grid.min(), grid.max()
        if gmax > gmin:
            grid = (grid - gmin) / (gmax - gmin)
        from PIL import Image as PI
        im = PI.fromarray((grid * 255).astype(np.uint8))
        im = im.resize((target_hw[1], target_hw[0]), PI.BILINEAR)
        return np.asarray(im)

    third_h, third_w = third_np.shape[:2]
    wrist_h, wrist_w = wrist_np.shape[:2]
    third_heat = _heatmap(img2_vec, (third_h, third_w))
    wrist_heat = _heatmap(img1_vec, (wrist_h, wrist_w))

    def _overlay(rgb, heat, alpha=0.55):
        h = heat.astype(np.float32) / 255.0
        col = np.zeros((*heat.shape, 3), dtype=np.float32)
        col[..., 0] = 1.0
        col[..., 1] = h
        col[..., 2] = 0.0
        col_u = (col * 255).astype(np.uint8)
        out = (rgb.astype(np.float32) * (1 - alpha * h[..., None]) +
               col_u.astype(np.float32) * (alpha * h[..., None]))
        return np.clip(out, 0, 255).astype(np.uint8)

    third_ov = _overlay(third_np, third_heat)
    wrist_ov = _overlay(wrist_np, wrist_heat)

    # Composite
    third_strip = np.concatenate([third_np, third_heat[..., None].repeat(3, axis=-1), third_ov], axis=1)
    wrist_strip = np.concatenate([wrist_np, wrist_heat[..., None].repeat(3, axis=-1), wrist_ov], axis=1)
    if wrist_strip.shape[1] < third_strip.shape[1]:
        pad_w = third_strip.shape[1] - wrist_strip.shape[1]
        wrist_strip = np.concatenate(
            [wrist_strip, np.full((wrist_strip.shape[0], pad_w, 3), 255, dtype=np.uint8)], axis=1)
    elif wrist_strip.shape[1] > third_strip.shape[1]:
        pad_w = wrist_strip.shape[1] - third_strip.shape[1]
        third_strip = np.concatenate(
            [third_strip, np.full((third_strip.shape[0], pad_w, 3), 255, dtype=np.uint8)], axis=1)
    spacer = np.full((8, third_strip.shape[1], 3), 255, dtype=np.uint8)
    full = np.concatenate([third_strip, spacer, wrist_strip], axis=0)
    out_path = out_dir / f"{args_cli.tag}_overlay.png"
    Image.fromarray(full).save(out_path)
    _log(f"wrote {out_path}")

    np.savez_compressed(
        out_dir / f"{args_cli.tag}_data.npz",
        third_heat=third_heat, wrist_heat=wrist_heat,
        third_np=third_np, wrist_np=wrist_np, action=np.asarray(action_np),
    )

    # ---------- text → vision attention -----------------------------------
    if args_cli.text_query:
        # Find prefix-only forward pass attention layers: shape (Q, K) where
        # Q == K == prefix length used in KV cache fill.
        prefix_target = next((qk for qk in qk_set if qk[0] == qk[1]), None)
        if prefix_target is None:
            _log("WARN: no prefix-only (Q==K) attention layers captured; cannot do text→vision")
        else:
            _log(f"prefix-only attention shape: {prefix_target}")
            prefix_layers = [t for t in captured if (t.shape[-2], t.shape[-1]) == prefix_target]
            _log(f"prefix-only matching layers: {len(prefix_layers)}")
            prefix_full_len = prefix_target[0]

            # Get the language tokens that were ACTUALLY fed to the model.
            # The batch contains either OBS_LANGUAGE_TOKENS or task text.
            lang_token_ids = None
            for k, v in batch.items():
                if "language" in k.lower() and "token" in k.lower() and "mask" not in k.lower():
                    if isinstance(v, torch.Tensor):
                        lang_token_ids = v[0].detach().cpu().tolist()
                        _log(f"language token ids ({len(lang_token_ids)}) from batch[{k}]")
                        break
            if lang_token_ids is None:
                _log("ERROR: could not find language tokens in batch")
            else:
                # Tokenize the query words individually to find their token IDs.
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
                # Decode the actual lang_token_ids so we can show them
                _log("actual lang tokens fed to model:")
                for i, tid in enumerate(lang_token_ids):
                    txt = tok.decode([tid]) if tid >= 0 else "<pad>"
                    _log(f"  [{i}] id={tid} text={txt!r}")
                queries = [q.strip() for q in args_cli.text_query.split(",")]
                # For each query word, find its position in the lang_tokens.
                query_positions = {}
                for q in queries:
                    # Try matching with leading-space variant first (mid-sentence)
                    for variant in (" " + q, q):
                        ids_q = tok.encode(variant, add_special_tokens=False)
                        if not ids_q:
                            continue
                        # Find first occurrence of ids_q[0] in lang_token_ids
                        for i, tid in enumerate(lang_token_ids):
                            if tid == ids_q[0]:
                                query_positions[q] = i
                                break
                        if q in query_positions:
                            break
                _log(f"query positions in lang_tokens: {query_positions}")

                # Translate lang-token positions to absolute prefix positions.
                # Prefix layout: [img1, img2, lang_tokens, state, padding].
                # img token start computed earlier as img1_s/img2_e for vision.
                lang_offset_in_prefix = img2_e
                _log(f"lang_offset_in_prefix = {lang_offset_in_prefix}")

                # Average attention across last K layers + heads, replace NaN with 0.
                stacked_p = torch.stack(prefix_layers[-args_cli.last_k_layers:], dim=0)
                stacked_p = torch.nan_to_num(stacked_p, nan=0.0)
                mean_p = stacked_p.mean(dim=(0, 2))[0]  # (Q, K)
                _log(f"prefix mean attn shape: {mean_p.shape}")

                # For each query word, slice its row, take attention to vision keys.
                text_overlays = {}
                for q, lpos in query_positions.items():
                    abs_pos = lang_offset_in_prefix + lpos
                    if abs_pos >= prefix_full_len:
                        _log(f"  WARN: {q} abs_pos {abs_pos} >= prefix_len; skipping")
                        continue
                    row = mean_p[abs_pos]  # (K,)
                    img1_v = row[img1_s:img1_e].numpy()
                    img2_v = row[img2_s:img2_e].numpy()
                    # DEBUG: dump raw per-token attention values for the
                    # third-person camera so we can verify spatial layout.
                    _log(f"  '{q}' raw img2 attention values ({img2_v.shape[0]} tokens):")
                    side = int(round(img2_v.shape[0] ** 0.5))
                    if side * side == img2_v.shape[0]:
                        grid = img2_v.reshape(side, side)
                        for r in range(side):
                            row_str = " ".join(f"{v:.4f}" for v in grid[r])
                            _log(f"    row {r}: [{row_str}]")
                        argmax_flat = int(img2_v.argmax())
                        argmax_rc = (argmax_flat // side, argmax_flat % side)
                        _log(f"    argmax: token {argmax_flat} = grid({argmax_rc[0]},{argmax_rc[1]})")
                    third_h_q = _heatmap(img2_v, (third_h, third_w))
                    wrist_h_q = _heatmap(img1_v, (wrist_h, wrist_w))
                    third_o_q = _overlay(third_np, third_h_q)
                    wrist_o_q = _overlay(wrist_np, wrist_h_q)
                    text_overlays[q] = (third_h_q, third_o_q, wrist_h_q, wrist_o_q,
                                        img1_v.sum(), img1_v.max(),
                                        img2_v.sum(), img2_v.max())

                # Composite: one row per query word: (raw third | heat third | overlay third) over wrist row.
                rows = []
                for q, (th, to, wh, wo, w1s, w1m, w2s, w2m) in text_overlays.items():
                    third_strip_q = np.concatenate([third_np, th[..., None].repeat(3, axis=-1), to], axis=1)
                    wrist_strip_q = np.concatenate([wrist_np, wh[..., None].repeat(3, axis=-1), wo], axis=1)
                    if wrist_strip_q.shape[1] < third_strip_q.shape[1]:
                        pad_w = third_strip_q.shape[1] - wrist_strip_q.shape[1]
                        wrist_strip_q = np.concatenate(
                            [wrist_strip_q, np.full((wrist_strip_q.shape[0], pad_w, 3), 255, dtype=np.uint8)], axis=1)
                    elif wrist_strip_q.shape[1] > third_strip_q.shape[1]:
                        pad_w = wrist_strip_q.shape[1] - third_strip_q.shape[1]
                        third_strip_q = np.concatenate(
                            [third_strip_q, np.full((third_strip_q.shape[0], pad_w, 3), 255, dtype=np.uint8)], axis=1)
                    spacer = np.full((4, third_strip_q.shape[1], 3), 255, dtype=np.uint8)
                    block = np.concatenate([third_strip_q, spacer, wrist_strip_q], axis=0)
                    # Add a label bar above
                    from PIL import ImageDraw, ImageFont
                    bar_h = 24
                    bar = np.full((bar_h, block.shape[1], 3), 230, dtype=np.uint8)
                    bar_im = Image.fromarray(bar)
                    draw = ImageDraw.Draw(bar_im)
                    label = f"text='{q}'  third: peak={w2m:.4f} (sum={w2s:.3f})  wrist: peak={w1m:.4f}"
                    draw.text((5, 4), label, fill=(0, 0, 0))
                    bar = np.array(bar_im)
                    rows.append(np.concatenate([bar, block], axis=0))

                if rows:
                    sep = np.full((10, rows[0].shape[1], 3), 255, dtype=np.uint8)
                    full_text = rows[0]
                    for r in rows[1:]:
                        full_text = np.concatenate([full_text, sep, r], axis=0)
                    out_path_text = out_dir / f"{args_cli.tag}_text_overlay.png"
                    Image.fromarray(full_text).save(out_path_text)
                    _log(f"wrote text overlay {out_path_text}")
                    _log("text→vision attention summary:")
                    for q, (_, _, _, _, w1s, w1m, w2s, w2m) in text_overlays.items():
                        _log(f"  '{q}': third sum={w2s:.4f} max={w2m:.4f} | wrist sum={w1s:.4f} max={w1m:.4f}")

    swe.SmolVLMWithExpertModel.eager_attention_forward = orig_eager
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
