"""Linear-probe base SmolVLA's vision encoder for cube_xy (vla_kitting-k98).

Reuses the 30-demo HDF5 at /tmp/yaw_30/cube_scripted_yaw30.hdf5: pulls
(third_person_cam[0], cube_pos[0]) per real demo, runs the BASE
SmolVLA vision tower (no LoRA), fits leave-one-out ridge regression
patch_features → cube_xy, reports R² per axis.

R² > 0.7 → vision encoder encodes cube position; bottleneck is the
action-head's queries (frozen-vision v5 strategy is correct).
R² < 0.5 → vision encoder isn't representing the cube; v5 needs
explicit vision supervision (aux cube-localization loss).

Quick & dirty by design.
"""
from __future__ import annotations

import pathlib
import sys

import h5py
import numpy as np
import torch

REPO = pathlib.Path("/home/ubuntu/vla_kitting")
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/ubuntu/code/lerobot/src")

H5_PATH = pathlib.Path("/tmp/yaw_30/cube_scripted_yaw30.hdf5")
LOCAL_CFG_DIR = REPO / "checkpoints/continual/checkpoints/last/pretrained_model"
OUT_DIR = REPO / "reports/runs/vision_probe_2026-04-27"


def _log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"loading frames + cube positions from {H5_PATH}")
    frames, cubes = [], []
    with h5py.File(str(H5_PATH), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            d = f["data"][k]
            if int(d.attrs.get("num_samples", 0)) <= 50:
                continue
            frames.append(d["obs"]["third_person_cam"][0])  # (H, W, 3) uint8
            cubes.append(d["obs"]["cube_pos"][0])  # (3,)
    X_imgs = np.stack(frames, axis=0)  # (N, H, W, 3)
    y = np.stack(cubes, axis=0)[:, :2].astype(np.float32)  # (N, 2)
    N = X_imgs.shape[0]
    _log(f"  N = {N}, image shape = {X_imgs.shape[1:]}, cube_xy span "
         f"X[{y[:, 0].min():.3f}, {y[:, 0].max():.3f}] "
         f"Y[{y[:, 1].min():.3f}, {y[:, 1].max():.3f}]")

    _log("loading base SmolVLA (lerobot/smolvla_base, no LoRA)")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    local_cfg = PreTrainedConfig.from_pretrained(str(LOCAL_CFG_DIR))
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=local_cfg)
    policy.to("cuda").eval()

    # Walk the model to find the vision tower. SmolVLM2 wraps SigLIP.
    # We just need a module that takes (B, 3, H, W) and returns (B, T, D)
    # patch tokens. Try common attribute paths.
    vision = None
    for path in [
        "model.vlm_with_expert.vlm.model.vision_tower",
        "model.vlm_with_expert.vlm.vision_tower",
        "model.vlm_with_expert.vlm.model.vision_model",
        "model.vlm_with_expert.vlm.vision_model",
    ]:
        try:
            mod = policy
            for p in path.split("."):
                mod = getattr(mod, p)
            vision = mod
            _log(f"found vision tower at policy.{path} ({type(vision).__name__})")
            break
        except AttributeError:
            continue
    if vision is None:
        # Fallback: print the module tree top-level so the operator can fix.
        _log("WARN: vision tower not found; dumping module tree for diagnosis:")
        for name, _ in policy.named_modules():
            if name.count(".") <= 4 and ("vision" in name or "vlm" in name):
                _log(f"  {name}")
        return 1

    # Preprocess images: SigLIP expects float32, mean=0.5, std=0.5, in [0,1] first.
    # Image dims may need to match the encoder's expected input. Pull from cfg if available.
    expected_size = None
    if hasattr(vision, "config"):
        expected_size = getattr(vision.config, "image_size", None)
    _log(f"vision encoder expected image_size = {expected_size}")

    imgs_t = torch.from_numpy(X_imgs).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)
    if expected_size and (imgs_t.shape[-1] != expected_size or imgs_t.shape[-2] != expected_size):
        _log(f"resizing {imgs_t.shape[-2:]} → {expected_size}x{expected_size}")
        imgs_t = torch.nn.functional.interpolate(
            imgs_t, size=(expected_size, expected_size), mode="bilinear", align_corners=False
        )
    imgs_t = (imgs_t - 0.5) / 0.5
    # Match vision-tower dtype (SmolVLM2 typically loads in bfloat16).
    vision_dtype = next(vision.parameters()).dtype
    imgs_t = imgs_t.to("cuda").to(vision_dtype)

    _log("running vision tower forward passes")
    feats_list = []
    with torch.no_grad():
        for i in range(N):
            out = vision(imgs_t[i:i + 1])
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            feats_list.append(tokens.float().cpu().numpy())
    feats = np.concatenate(feats_list, axis=0)  # (N, T, D)
    _log(f"  features shape (N, T, D) = {feats.shape}")
    T_tok = feats.shape[1]
    grid = int(np.sqrt(T_tok))
    _log(f"  inferred patch grid: {grid}x{grid}={grid * grid} (vs T={T_tok})")

    # Variant A: mean-pool over patches → (N, D)
    mean_feat = feats.mean(axis=1)

    # Variant B: per-patch L2-norm argmax → (N, 2) spatial coords
    # If the encoder concentrates "object" magnitude where the cube is, the
    # argmax patch's row/col should correlate with cube_xy.
    if grid * grid == T_tok:
        # No CLS token; pure patch grid.
        patch_feat = feats
    else:
        # Drop the first token (assume CLS).
        patch_feat = feats[:, 1:1 + grid * grid, :]
        T_tok = grid * grid
    norms = np.linalg.norm(patch_feat, axis=-1)  # (N, T)
    argmax_flat = norms.argmax(axis=1)
    argmax_row = (argmax_flat // grid).astype(np.float32) / max(grid - 1, 1)
    argmax_col = (argmax_flat % grid).astype(np.float32) / max(grid - 1, 1)
    argmax_xy = np.stack([argmax_row, argmax_col], axis=1)  # (N, 2)

    # Linear probe with leave-one-out CV (numpy-only ridge to avoid sklearn dep).
    def _ridge_fit(Xc: np.ndarray, yc: np.ndarray, lam: float) -> np.ndarray:
        # (Xc^T Xc + lam I)^-1 Xc^T yc; assumes column-mean already subtracted.
        d = Xc.shape[1]
        return np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)

    def _r2(true: np.ndarray, pred: np.ndarray) -> float:
        ss_res = float(((true - pred) ** 2).sum())
        ss_tot = float(((true - true.mean()) ** 2).sum()) + 1e-12
        return 1.0 - ss_res / ss_tot

    def loo_r2(X: np.ndarray, y: np.ndarray, alphas) -> tuple[float, float, np.ndarray]:
        N = X.shape[0]
        preds = np.zeros_like(y)
        for i in range(N):
            mask = np.arange(N) != i
            Xtr, ytr = X[mask], y[mask]
            Xte = X[i:i + 1]
            xmean, ymean = Xtr.mean(axis=0), ytr.mean(axis=0)
            Xtr_c = Xtr - xmean
            ytr_c = ytr - ymean
            # Pick lambda by inner LOO on Xtr (cheap for small N).
            best_lam, best_score = alphas[0], -np.inf
            for lam in alphas:
                inner_preds = np.zeros_like(ytr)
                for j in range(Xtr_c.shape[0]):
                    m2 = np.arange(Xtr_c.shape[0]) != j
                    Xtr2, ytr2 = Xtr_c[m2], ytr_c[m2]
                    W = _ridge_fit(Xtr2, ytr2, lam)
                    inner_preds[j] = (Xtr_c[j:j + 1] @ W).ravel() + ymean
                score = _r2(ytr[:, 0], inner_preds[:, 0]) + _r2(ytr[:, 1], inner_preds[:, 1])
                if score > best_score:
                    best_score, best_lam = score, lam
            W = _ridge_fit(Xtr_c, ytr_c, best_lam)
            preds[i] = ((Xte - xmean) @ W).ravel() + ymean
        return _r2(y[:, 0], preds[:, 0]), _r2(y[:, 1], preds[:, 1]), preds

    # PCA-reduce mean-pooled features to N-2 components so ridge isn't
    # data-starved (768 raw dims with N=24 train is hopeless even with ridge).
    def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD path: economical for tall-skinny X.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:n_components].T

    n_pca = max(2, mean_feat.shape[0] - 4)  # leave a few DOF for ridge
    mean_pca = _pca(mean_feat, n_pca)
    _log(f"variant A: mean-pooled patch features → PCA({n_pca}) → cube_xy")
    r2_ax, r2_ay, preds_a = loo_r2(mean_pca, y, [0.01, 0.1, 1.0, 10.0, 100.0])
    _log(f"  LOO R²: x={r2_ax:+.3f}  y={r2_ay:+.3f}")

    _log("variant B: argmax-patch spatial location (row,col) → cube_xy")
    r2_bx, r2_by, preds_b = loo_r2(argmax_xy, y, [0.001, 0.01, 0.1, 1.0])
    _log(f"  LOO R²: x={r2_bx:+.3f}  y={r2_by:+.3f}")

    # Variant C: per-patch L2 norms as 1024-D features → PCA → cube_xy.
    # If the cube's patch has higher (or distinctive) magnitude than other
    # patches, this should pick that up.
    norms_pca = _pca(norms, n_pca)
    _log(f"variant C: per-patch L2-norms ({grid}x{grid}=1024D) → PCA({n_pca}) → cube_xy")
    r2_cx, r2_cy, preds_c = loo_r2(norms_pca, y, [0.01, 0.1, 1.0, 10.0, 100.0])
    _log(f"  LOO R²: x={r2_cx:+.3f}  y={r2_cy:+.3f}")

    # Verdict
    def verdict(rx: float, ry: float) -> str:
        if min(rx, ry) >= 0.7:
            return "ENCODES_CUBE (proceed with frozen-vision v5)"
        if min(rx, ry) >= 0.4:
            return "PARTIAL (vision has some cube info; aux loss may help)"
        return "MISSING (vision encoder isn't picking up cube; v5 needs explicit supervision)"

    _log(f"VERDICT (mean-pool PCA):  {verdict(r2_ax, r2_ay)}")
    _log(f"VERDICT (argmax-loc):     {verdict(r2_bx, r2_by)}")
    _log(f"VERDICT (norms PCA):      {verdict(r2_cx, r2_cy)}")

    # Best-of: take the strongest variant per axis.
    best_x = max(r2_ax, r2_bx, r2_cx)
    best_y = max(r2_ay, r2_by, r2_cy)
    _log(f"BEST-OF-3:                R² x={best_x:+.3f}  y={best_y:+.3f}  → {verdict(best_x, best_y)}")

    # Persist results
    np.savez_compressed(
        str(OUT_DIR / "probe_results.npz"),
        cube_xy=y, mean_feat=mean_feat, argmax_xy=argmax_xy,
        preds_meanpool=preds_a, preds_argmax=preds_b, preds_norms=preds_c,
        r2_meanpool=np.array([r2_ax, r2_ay]),
        r2_argmax=np.array([r2_bx, r2_by]),
        r2_norms=np.array([r2_cx, r2_cy]),
    )
    _log(f"wrote {OUT_DIR}/probe_results.npz")

    # Quick scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for col, (name, preds, r2x, r2y) in enumerate([
            ("mean-pool PCA", preds_a, r2_ax, r2_ay),
            ("norms PCA", preds_c, r2_cx, r2_cy),
        ]):
            axes[0, col].scatter(y[:, 0], preds[:, 0], s=20)
            axes[0, col].plot([y[:, 0].min(), y[:, 0].max()], [y[:, 0].min(), y[:, 0].max()], 'k--', alpha=0.3)
            axes[0, col].set_xlabel("true cube X"); axes[0, col].set_ylabel("predicted cube X")
            axes[0, col].set_title(f"{name}: R² X = {r2x:+.3f}")
            axes[1, col].scatter(y[:, 1], preds[:, 1], s=20)
            axes[1, col].plot([y[:, 1].min(), y[:, 1].max()], [y[:, 1].min(), y[:, 1].max()], 'k--', alpha=0.3)
            axes[1, col].set_xlabel("true cube Y"); axes[1, col].set_ylabel("predicted cube Y")
            axes[1, col].set_title(f"{name}: R² Y = {r2y:+.3f}")
        fig.suptitle(f"Base SmolVLA vision-feature → cube_xy linear probe (LOO, N={N})")
        fig.tight_layout()
        out_png = OUT_DIR / "probe_scatter.png"
        fig.savefig(str(out_png), dpi=110)
        _log(f"wrote {out_png}")
    except Exception as e:
        _log(f"plotting failed (non-fatal): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
