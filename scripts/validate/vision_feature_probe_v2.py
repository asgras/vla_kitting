"""Spatial-aware vision-feature probe at the NEW camera config (vla_kitting-mu7).

Re-runs the base SmolVLA vision-feature probe under three improvements over
the prior k98 attempt (`reports/2026-04-27_vision_feature_probe.md`):

  1. **N >= 100** randomly-sampled cube positions (vs N=24 from the old HDF5).
  2. **NEW camera config** (third-person 512x512, repositioned at
     (1.5, -0.10, 0.80), focal_length=18 / FOV ~60 deg). Rendering uses the
     env in envs/yaskawa_pick_cube_cfg.py, no extra plumbing needed.
  3. **Spatial-aware aggregator**: instead of mean-pooling 1024 patch tokens
     into a single 768-D vector, project the world-frame cube center into
     the 32x32 patch grid via an empirical homography fit (from the two known
     anchors per frame: cube at known forced world-XY + magenta target at
     world (0.65, 0.20)) and read off the 3x3 patch window centered on the
     projected cube. Compare those on-cube features to a fixed off-cube
     control window (image center).

Three readouts (verdict thresholds frozen by issue mu7):
  (a) Leave-one-out ridge: on-cube 3x3 mean-pooled patch features -> cube_xy
      regression. R^2 per axis. PASS half if R^2 >= 0.7 on BOTH axes.
  (b) Cosine similarity (on-cube mean) vs (off-cube mean), averaged across
      frames. PASS half if mean cos-sim < 0.9.
  (c) Color silhouette: cluster on-cube features by recorded cube_color_idx,
      compute silhouette score (higher = features better separate the
      5 cube colors). Diagnostic only; not a verdict gate.

Verdict:
  - PASS    iff (a) AND (b)     -> v5 trains WITHOUT aux cube-loc loss.
  - PARTIAL iff exactly one     -> v5 adds aux loss at 0.05x action weight.
  - MISSING iff neither         -> v5 adds aux loss at 0.10x weight, consider
                                   unfreezing vision tower.

Usage:
    cd /home/ubuntu/vla_kitting
    /home/ubuntu/IsaacLab/isaaclab.sh -p \
        scripts/validate/vision_feature_probe_v2.py \
        --out_dir reports/runs/mu7_2026-04-27 --samples 100 --seed 0

Outputs (under --out_dir):
  - frame_NNN_third.png           per-frame third-person 512x512 PNG
  - manifest.json                  list[{frame, cube_pos, cube_color_idx, color_name}]
  - probe_results.npz              cube_xy, on/off feats, R^2, cos-sim, silhouette
  - homography.json                empirical world-XY -> image-pixel fit summary
  - probe_summary.txt              one-screen human-readable verdict
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/ubuntu/code/lerobot/src")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=pathlib.Path,
                    default=REPO / "reports" / "runs" / "mu7_2026-04-27")
parser.add_argument("--samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--patch_size", type=int, default=16,
                    help="SmolVLM2 vision-tower patch size at 512x512 "
                         "(=> 32x32 patch grid).")
parser.add_argument("--window", type=int, default=3,
                    help="Side length of the on-cube/off-cube patch window. "
                         "3 => 3x3 patches around the projected cube center.")
parser.add_argument("--render_only", action="store_true",
                    help="Just dump frames + manifest, skip the SmolVLA probe. "
                         "Useful for re-running the probe pass without "
                         "re-rendering 100 frames.")
parser.add_argument("--probe_only", action="store_true",
                    help="Skip rendering; load frames + manifest from --out_dir "
                         "and run the SmolVLA probe pass only.")
args_cli, _ = parser.parse_known_args()

# Spawn-box bounds (must match envs/yaskawa_pick_cube_cfg.py randomize_cube_pose):
# absolute X in [0.25, 0.85], Y in [-0.40, 0.00].
SPAWN_X_RANGE = (0.25, 0.85)
SPAWN_Y_RANGE = (-0.40, 0.00)
CUBE_Z = 0.025
TARGET_WORLD_XY = (0.65, 0.20)  # the magenta circle, fixed per env_cfg

# Cube color palette (red, blue, yellow, orange, purple), index 0..4. Pulled
# in lazily so the parser can run without importing envs.* before the app
# launcher.


def _log(msg: str) -> None:
    print(f"[mu7] {msg}", flush=True)


# ---------------------------------------------------------------- rendering ---


def _render_frames(out_dir: pathlib.Path, n_samples: int, seed: int,
                   settle_steps: int) -> list[dict]:
    """Render N frames at random cube_xy positions across the spawn box.
    Returns the manifest list (also written to disk as manifest.json).

    Note: launches Isaac Lab. Must be called BEFORE any heavy torch / lerobot
    import touches CUDA — the app launcher needs to come first.
    """
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import envs  # noqa: F401  (registers the task)
    from envs.mdp.cube_palette import color_name_for_idx
    from isaaclab_tasks.utils import parse_env_cfg

    TASK = "Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    sim_dev = env.unwrapped.sim.device
    origin = env.unwrapped.scene.env_origins[0]

    rng = np.random.default_rng(seed)
    xs = rng.uniform(SPAWN_X_RANGE[0], SPAWN_X_RANGE[1], size=n_samples)
    ys = rng.uniform(SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1], size=n_samples)
    yaws = rng.uniform(-0.5, 0.5, size=n_samples)

    manifest: list[dict] = []
    zero_action = torch.zeros((1, 7), device="cuda:0")
    cube_rb = env.unwrapped.scene["cube"]

    _log(f"rendering N={n_samples} frames -> {out_dir} (seed={seed})")
    for i in range(n_samples):
        # Reset the env (this also fires randomize_cube_color via EventCfg, so
        # we get a fresh per-episode color palette index in obs).
        obs, _ = env.reset()

        # Force the cube to the desired XY (overriding the random reset XY).
        # Yaw uses (cos(yaw/2), 0, 0, sin(yaw/2)) as Z-rotation quat.
        yaw = float(yaws[i])
        qw = float(np.cos(yaw / 2.0))
        qz = float(np.sin(yaw / 2.0))
        pose = torch.tensor([[
            float(xs[i]) + origin[0].item(),
            float(ys[i]) + origin[1].item(),
            CUBE_Z + origin[2].item(),
            qw, 0.0, 0.0, qz,
        ]], device=sim_dev)
        cube_rb.write_root_pose_to_sim(pose)
        cube_rb.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_dev))
        env.unwrapped.scene.write_data_to_sim()
        for _ in range(settle_steps):
            obs, *_ = env.step(zero_action)

        # Read back the actual cube_pos (post-settle) so manifest is ground truth.
        cube_pos = obs["policy"]["cube_pos"][0].cpu().numpy().astype(float).tolist()
        color_idx = int(obs["policy"]["cube_color_idx"][0].item())
        third = obs["policy"]["third_person_cam"][0].cpu().numpy().astype(np.uint8)
        third_path = out_dir / f"frame_{i:03d}_third.png"
        Image.fromarray(third).save(third_path)
        manifest.append({
            "frame_id": i,
            "third_png": third_path.name,
            "cube_pos": cube_pos,
            "cube_yaw": yaw,
            "cube_color_idx": color_idx,
            "cube_color_name": color_name_for_idx(color_idx),
        })
        if (i + 1) % 10 == 0:
            _log(f"  rendered {i + 1}/{n_samples}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "n_samples": n_samples,
        "seed": seed,
        "image_size": 512,
        "spawn_x_range": list(SPAWN_X_RANGE),
        "spawn_y_range": list(SPAWN_Y_RANGE),
        "target_world_xy": list(TARGET_WORLD_XY),
        "frames": manifest,
    }, indent=2))
    _log(f"wrote manifest {manifest_path}")
    env.close()
    return manifest


# ------------------------------------------------------ pixel-detection helpers


def _detect_magenta_pixel(rgb: "np.ndarray") -> tuple[float, float] | None:
    """Centroid (col, row) of the magenta target disk, or None if not found.
    Magenta diffuse_color = (1.0, 0.0, 1.0): R + B high, G low.
    """
    import numpy as np
    R = rgb[..., 0].astype(np.int32)
    G = rgb[..., 1].astype(np.int32)
    B = rgb[..., 2].astype(np.int32)
    # Magenta criterion: high R, high B, low G, and R~B (no warm/cool bias).
    mask = (R > 150) & (B > 150) & (G < 80) & (np.abs(R - B) < 50)
    if mask.sum() < 4:
        return None
    rs, cs = np.where(mask)
    return float(cs.mean()), float(rs.mean())


def _detect_cube_pixel(rgb: "np.ndarray", color_rgb01: tuple[float, float, float],
                        target_pixel: tuple[float, float] | None
                        ) -> tuple[float, float] | None:
    """Centroid (col, row) of the largest blob matching the cube's hue.

    Isaac's rendered cube colors are heavily desaturated by the dome light
    (e.g. an `(0.85, 0.15, 0.15)` "red" cube reads as RGB ~(237, 177, 180):
    chromaticity is preserved but absolute distance to pure red is ~155).
    So we score by *chromatic axis* (which channel dominates which) rather
    than RGB Euclidean. Each palette color has a distinctive R/G/B ranking:

      red    (0.85, 0.15, 0.15) -> R >> G ~ B          (R - max(G,B) > thr)
      blue   (0.15, 0.35, 0.90) -> B >> R              (B - max(R,G) > thr)
      yellow (0.95, 0.80, 0.10) -> R, G >> B           (min(R,G) - B > thr)
      orange (0.90, 0.45, 0.10) -> R >> G > B          (R - B > thr, R > 1.5*G)
      purple (0.70, 0.15, 0.80) -> R, B >> G           (min(R,B) - G > thr)

    Magenta target is (255, 0, 255): R >> G AND B >> G — overlaps purple.
    Suppress with a generous disc (radius 60 px) around the detected target.
    """
    import numpy as np
    R = rgb[..., 0].astype(np.int32)
    G = rgb[..., 1].astype(np.int32)
    B = rgb[..., 2].astype(np.int32)
    # Identify which palette color this is by max-channel pattern.
    r01, g01, b01 = color_rgb01
    if r01 > 0.7 and g01 < 0.3 and b01 < 0.3:        # red
        mask = (R - np.maximum(G, B) > 40)
    elif b01 > 0.7 and r01 < 0.3:                    # blue
        mask = (B - np.maximum(R, G) > 40)
    elif r01 > 0.7 and g01 > 0.6 and b01 < 0.3:      # yellow
        mask = (np.minimum(R, G) - B > 60)
    elif r01 > 0.7 and 0.3 < g01 < 0.6 and b01 < 0.3:  # orange
        mask = (R - B > 60) & (R > 1.4 * G)
    elif r01 > 0.5 and g01 < 0.3 and b01 > 0.6:      # purple
        mask = (np.minimum(R, B) - G > 40)
    else:
        return None

    if target_pixel is not None:
        # Magenta target shares purple's R-high + B-high pattern. Suppress a
        # disc around it; 60 px > the disk's projected radius (~20 px) plus
        # safety margin against bleed from anti-aliased edges.
        H_img, W_img = rgb.shape[:2]
        yy, xx = np.mgrid[0:H_img, 0:W_img]
        d2 = (xx - target_pixel[0]) ** 2 + (yy - target_pixel[1]) ** 2
        mask &= d2 > 60 ** 2

    if mask.sum() < 6:
        return None
    rs, cs = np.where(mask)
    # Use the *largest connected component* via a quick BFS-by-distance hack:
    # take points within a robust median-distance band of the mask centroid.
    # Cheaper: trim the 10% of pixels furthest from the median (suppresses
    # spurious bleed pixels), then return the trimmed centroid.
    pts = np.stack([cs, rs], axis=1).astype(np.float64)
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d < np.percentile(d, 90)
    pts = pts[keep] if keep.sum() >= 4 else pts
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


# --------------------------------------------------------- homography fitting


def _fit_homography_dlt(world_xy: "np.ndarray", pixel_xy: "np.ndarray") -> "np.ndarray":
    """Plain DLT homography fit (least-squares over ALL points). Sensitive to
    outliers — use _fit_homography (RANSAC) for noisy detections.
    """
    import numpy as np
    N = world_xy.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        X, Y = world_xy[i]
        u, v = pixel_xy[i]
        A[2 * i] = [X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u]
        A[2 * i + 1] = [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _fit_homography(world_xy: "np.ndarray", pixel_xy: "np.ndarray",
                     n_iter: int = 400, inlier_thr_px: float = 25.0,
                     seed: int = 0) -> "np.ndarray":
    """RANSAC homography from world-XY to pixel-XY. Cube-blob detection is
    noisy (some frames legitimately fail the chromatic threshold and hit on
    background pixels), so a plain DLT over all correspondences gets pulled
    by a few bad pairs. Sample 4-pt minimal sets, count inliers, refit on
    inliers of best model. With N~80 correspondences and ~20% outliers the
    expected inlier count after 400 iterations is reliable.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    N = world_xy.shape[0]
    if N < 4:
        return _fit_homography_dlt(world_xy, pixel_xy)
    best_inliers = None
    best_count = -1
    for _ in range(n_iter):
        idx = rng.choice(N, size=4, replace=False)
        try:
            H_try = _fit_homography_dlt(world_xy[idx], pixel_xy[idx])
        except np.linalg.LinAlgError:
            continue
        proj = _project(H_try, world_xy)
        err = np.linalg.norm(proj - pixel_xy, axis=1)
        inliers = err < inlier_thr_px
        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 4:
        return _fit_homography_dlt(world_xy, pixel_xy)
    return _fit_homography_dlt(world_xy[best_inliers], pixel_xy[best_inliers])


def _project(H: "np.ndarray", world_xy: "np.ndarray") -> "np.ndarray":
    """Project Nx2 world-XY through homography H -> Nx2 pixel-XY."""
    import numpy as np
    ones = np.ones((world_xy.shape[0], 1))
    homo = np.concatenate([world_xy, ones], axis=1) @ H.T
    return homo[:, :2] / homo[:, 2:3]


# -------------------------------------------------------- vision-tower probe


def _load_vision_tower():
    """Load the BASE SmolVLA vision tower exactly as the prior k98 probe did.
    Returns (vision_module, expected_size, dtype).
    """
    import torch  # noqa: F401  (sanity)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig

    LOCAL_CFG_DIR = REPO / "checkpoints/continual/checkpoints/last/pretrained_model"
    _log(f"loading base SmolVLA (lerobot/smolvla_base, no LoRA) "
         f"with input schema from {LOCAL_CFG_DIR}")
    local_cfg = PreTrainedConfig.from_pretrained(str(LOCAL_CFG_DIR))
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=local_cfg)
    policy.to("cuda").eval()

    vision = None
    for path in [
        "model.vlm_with_expert.vlm.model.vision_model",
        "model.vlm_with_expert.vlm.vision_model",
        "model.vlm_with_expert.vlm.model.vision_tower",
        "model.vlm_with_expert.vlm.vision_tower",
    ]:
        try:
            mod = policy
            for p in path.split("."):
                mod = getattr(mod, p)
            vision = mod
            _log(f"  found vision tower at policy.{path} ({type(vision).__name__})")
            break
        except AttributeError:
            continue
    if vision is None:
        raise RuntimeError("could not find vision tower on policy")

    expected_size = getattr(getattr(vision, "config", None), "image_size", 512)
    import torch
    dtype = next(vision.parameters()).dtype
    _log(f"  expected image_size={expected_size}, dtype={dtype}")
    return vision, int(expected_size), dtype


def _vision_forward(vision, expected_size: int, dtype, frames_uint8: "np.ndarray"
                    ) -> "np.ndarray":
    """Run the vision tower on N frames. Returns (N, T, D) float32 patch tokens.

    Uses the same SigLIP-style preprocessing as the k98 probe:
      pixel_values = (img/255 - 0.5) / 0.5
    """
    import numpy as np
    import torch

    imgs_t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
    if imgs_t.shape[-1] != expected_size or imgs_t.shape[-2] != expected_size:
        _log(f"  resizing {tuple(imgs_t.shape[-2:])} -> "
             f"{expected_size}x{expected_size} (bilinear)")
        imgs_t = torch.nn.functional.interpolate(
            imgs_t, size=(expected_size, expected_size),
            mode="bilinear", align_corners=False,
        )
    imgs_t = (imgs_t - 0.5) / 0.5
    imgs_t = imgs_t.to("cuda").to(dtype)

    feats_list = []
    with torch.no_grad():
        for i in range(imgs_t.shape[0]):
            out = vision(imgs_t[i:i + 1])
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            feats_list.append(tokens.float().cpu().numpy())
    feats = np.concatenate(feats_list, axis=0)
    _log(f"  features (N, T, D) = {feats.shape}")
    return feats


# --------------------------------------------------------- ridge / silhouette


def _ridge_fit(Xc: "np.ndarray", yc: "np.ndarray", lam: float) -> "np.ndarray":
    import numpy as np
    d = Xc.shape[1]
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)


def _r2(true: "np.ndarray", pred: "np.ndarray") -> float:
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def _loo_ridge_r2(X: "np.ndarray", y: "np.ndarray", alphas) -> tuple[float, float, "np.ndarray"]:
    """Leave-one-out ridge with simple held-out alpha selection on a 5-fold
    inner split (the inner LOO of k98 was O(N^3); 100 frames * 5 alphas would
    be 100*5*100=50k inner ridge solves which is fine but the held-out fold
    is faster and equally well-conditioned here). Returns (R^2_x, R^2_y, preds).
    """
    import numpy as np
    N = X.shape[0]
    preds = np.zeros_like(y, dtype=np.float64)
    rng = np.random.default_rng(0)
    for i in range(N):
        mask = np.arange(N) != i
        Xtr, ytr = X[mask], y[mask]
        Xte = X[i:i + 1]
        xmean, ymean = Xtr.mean(axis=0), ytr.mean(axis=0)
        Xtr_c = Xtr - xmean
        ytr_c = ytr - ymean

        # Pick lambda by 5-fold CV on Xtr.
        best_lam, best_score = alphas[0], -np.inf
        idx = np.arange(Xtr.shape[0])
        rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        for lam in alphas:
            preds_inner = np.zeros_like(ytr)
            for fold in folds:
                tr_mask = np.ones(Xtr.shape[0], dtype=bool)
                tr_mask[fold] = False
                W = _ridge_fit(Xtr_c[tr_mask], ytr_c[tr_mask], lam)
                preds_inner[fold] = (Xtr_c[fold] @ W) + ymean
            score = _r2(ytr[:, 0], preds_inner[:, 0]) + _r2(ytr[:, 1], preds_inner[:, 1])
            if score > best_score:
                best_score, best_lam = score, lam

        W = _ridge_fit(Xtr_c, ytr_c, best_lam)
        preds[i] = ((Xte - xmean) @ W).ravel() + ymean
    return _r2(y[:, 0], preds[:, 0]), _r2(y[:, 1], preds[:, 1]), preds


def _silhouette(X: "np.ndarray", labels: "np.ndarray") -> float:
    """Mean silhouette score (cosine distance) without sklearn. Returns NaN
    if there are fewer than 2 distinct labels with >=2 samples each."""
    import numpy as np
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = Xn @ Xn.T
    cos = np.clip(cos, -1.0, 1.0)
    D = 1.0 - cos
    np.fill_diagonal(D, 0.0)
    uniq, counts = np.unique(labels, return_counts=True)
    if (counts >= 2).sum() < 2:
        return float("nan")
    s = []
    for i in range(X.shape[0]):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b = np.inf
        for u in uniq:
            if u == labels[i]:
                continue
            other = labels == u
            if other.sum() == 0:
                continue
            b = min(b, D[i, other].mean())
        if max(a, b) < 1e-12:
            continue
        s.append((b - a) / max(a, b))
    return float(np.mean(s)) if s else float("nan")


# ---------------------------------------------------------------- main


def main() -> int:
    out_dir = pathlib.Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Phase 1: rendering (if needed) ------------------------------------
    manifest_path = out_dir / "manifest.json"
    if not args_cli.probe_only:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(headless=True, enable_cameras=True)
        sim_app = app_launcher.app
        try:
            manifest = _render_frames(
                out_dir, args_cli.samples, args_cli.seed, args_cli.settle_steps
            )
        finally:
            sim_app.close()
        if args_cli.render_only:
            _log("--render_only: stopping after render pass")
            return 0
    else:
        _log(f"--probe_only: loading manifest from {manifest_path}")
        manifest = json.loads(manifest_path.read_text())["frames"]

    # ----- Phase 2: load images + detect anchors -----------------------------
    import numpy as np
    from PIL import Image

    # Pull palette directly from the palette module to avoid importing the
    # full envs package (which transitively pulls isaaclab/omni.physics, only
    # available under isaaclab.sh's full launcher). cube_palette is dep-free.
    import importlib.util
    palette_path = REPO / "envs" / "mdp" / "cube_palette.py"
    spec = importlib.util.spec_from_file_location("cube_palette", palette_path)
    cube_palette = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube_palette)
    CUBE_COLOR_PALETTE = cube_palette.CUBE_COLOR_PALETTE

    _log(f"loading {len(manifest)} frames from disk")
    frames_uint8 = []
    cube_world_xy = []
    cube_pixels = []
    target_pixels = []
    color_idxs = []
    for entry in manifest:
        rgb = np.asarray(Image.open(out_dir / entry["third_png"]).convert("RGB"))
        frames_uint8.append(rgb)
        cube_world_xy.append(entry["cube_pos"][:2])
        color_idxs.append(int(entry["cube_color_idx"]))
        # Detect magenta target first; use it to suppress the red/purple cube
        # mask near the disk so the cube centroid doesn't get pulled towards it.
        tpx = _detect_magenta_pixel(rgb)
        target_pixels.append(tpx)
        cube_rgb01 = CUBE_COLOR_PALETTE[color_idxs[-1]][1]
        cpx = _detect_cube_pixel(rgb, cube_rgb01, tpx)
        cube_pixels.append(cpx)
    frames_uint8 = np.stack(frames_uint8, axis=0)
    cube_world_xy = np.array(cube_world_xy, dtype=np.float64)
    color_idxs = np.array(color_idxs, dtype=np.int64)
    n_target_ok = sum(1 for t in target_pixels if t is not None)
    n_cube_ok = sum(1 for c in cube_pixels if c is not None)
    _log(f"detected {n_target_ok}/{len(manifest)} magenta targets, "
         f"{n_cube_ok}/{len(manifest)} cube blobs")

    # ----- Phase 3: fit world->pixel homography ------------------------------
    # Use cube correspondences (varying world_xy <-> varying pixel_xy) plus the
    # fixed target anchor as 1 extra correspondence. Camera is rigidly fixed,
    # table is approximately planar, so a single 3x3 homography suffices.
    pairs_world = []
    pairs_pixel = []
    for cwxy, cpx in zip(cube_world_xy, cube_pixels):
        if cpx is None:
            continue
        pairs_world.append(cwxy)
        pairs_pixel.append(cpx)
    # Each frame has the same magenta anchor; include it once.
    if any(t is not None for t in target_pixels):
        valid_target_pixels = np.array([t for t in target_pixels if t is not None])
        target_pixel_avg = valid_target_pixels.mean(axis=0)
        pairs_world.append(np.array(TARGET_WORLD_XY))
        pairs_pixel.append(target_pixel_avg)
        _log(f"target anchor avg pixel = {target_pixel_avg}")
    pairs_world = np.array(pairs_world)
    pairs_pixel = np.array(pairs_pixel)
    if pairs_world.shape[0] < 6:
        _log(f"ERROR: only {pairs_world.shape[0]} valid correspondences, "
             f"cannot fit homography")
        return 2
    H = _fit_homography(pairs_world, pairs_pixel)
    proj = _project(H, pairs_world)
    fit_residuals = np.linalg.norm(proj - pairs_pixel, axis=1)
    _log(f"homography fit residuals (px): mean={fit_residuals.mean():.2f} "
         f"median={np.median(fit_residuals):.2f} max={fit_residuals.max():.2f}")
    (out_dir / "homography.json").write_text(json.dumps({
        "H": H.tolist(),
        "n_correspondences": int(pairs_world.shape[0]),
        "fit_residual_px_mean": float(fit_residuals.mean()),
        "fit_residual_px_median": float(np.median(fit_residuals)),
        "fit_residual_px_max": float(fit_residuals.max()),
    }, indent=2))

    # Project each cube_world_xy through H -> pixel center.
    projected_cube_pixels = _project(H, cube_world_xy)
    _log(f"projected cube pixel range: "
         f"col [{projected_cube_pixels[:, 0].min():.1f}, {projected_cube_pixels[:, 0].max():.1f}] "
         f"row [{projected_cube_pixels[:, 1].min():.1f}, {projected_cube_pixels[:, 1].max():.1f}]")

    # ----- Phase 4: vision tower forward + spatial aggregation --------------
    vision, expected_size, dtype = _load_vision_tower()
    feats = _vision_forward(vision, expected_size, dtype, frames_uint8)
    N, T, D = feats.shape
    grid = int(np.sqrt(T))
    assert grid * grid == T, f"expected square patch grid; got T={T}"
    _log(f"patch grid {grid}x{grid} (patch_size={expected_size // grid}px)")

    feats_grid = feats.reshape(N, grid, grid, D)

    # Convert projected cube pixel -> patch (row, col) on the encoder's input
    # grid. Encoder input is expected_size x expected_size; image was resized
    # from 512 to expected_size (no-op when expected_size==512). Patch row/col
    # = floor(pixel / patch_size).
    H_img = frames_uint8.shape[1]
    W_img = frames_uint8.shape[2]
    patch_px_w = W_img / grid
    patch_px_h = H_img / grid
    on_cube_feats = np.zeros((N, D), dtype=np.float64)
    off_cube_feats = np.zeros((N, D), dtype=np.float64)
    on_cube_in_bounds = np.zeros(N, dtype=bool)
    win = max(1, int(args_cli.window))
    half = win // 2
    center_row = grid // 2
    center_col = grid // 2
    for i in range(N):
        cu, cv = projected_cube_pixels[i]
        if not (0 <= cu < W_img and 0 <= cv < H_img):
            continue
        pr = int(round(cv / patch_px_h))
        pc = int(round(cu / patch_px_w))
        if not (half <= pr < grid - half and half <= pc < grid - half):
            continue
        on_cube_in_bounds[i] = True
        on_cube_feats[i] = feats_grid[i,
                                       pr - half:pr + half + 1,
                                       pc - half:pc + half + 1, :].mean(axis=(0, 1))
        off_cube_feats[i] = feats_grid[i,
                                        center_row - half:center_row + half + 1,
                                        center_col - half:center_col + half + 1, :].mean(axis=(0, 1))
    n_in = int(on_cube_in_bounds.sum())
    _log(f"on-cube patches in bounds (with {win}x{win} window): {n_in}/{N}")
    if n_in < 30:
        _log("ERROR: too few in-bounds frames; homography may be off")
        return 3

    cube_xy_in = cube_world_xy[on_cube_in_bounds]
    on_in = on_cube_feats[on_cube_in_bounds]
    off_in = off_cube_feats[on_cube_in_bounds]
    color_idx_in = color_idxs[on_cube_in_bounds]

    # Readout (a): LOO ridge from on-cube features -> cube_xy.
    # 768D features with N~100 is rank-deficient for plain ridge; use a small
    # PCA front-end so ridge has degrees of freedom, but keep enough components
    # that legitimate spatial signal isn't thrown out. n_pca ~= N//4.
    def _pca(X: "np.ndarray", n_components: int) -> "np.ndarray":
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        n_components = min(n_components, Vt.shape[0])
        return Xc @ Vt[:n_components].T

    n_pca = max(8, n_in // 4)
    on_pca = _pca(on_in, n_pca)
    _log(f"readout (a): on-cube features -> PCA({n_pca}) -> cube_xy ridge")
    r2_x, r2_y, preds_a = _loo_ridge_r2(
        on_pca, cube_xy_in.astype(np.float64),
        [0.01, 0.1, 1.0, 10.0, 100.0],
    )
    _log(f"  R^2: x={r2_x:+.3f}  y={r2_y:+.3f}")

    # Readout (b): per-frame on/off cosine similarity, then average.
    on_n = on_in / (np.linalg.norm(on_in, axis=1, keepdims=True) + 1e-12)
    off_n = off_in / (np.linalg.norm(off_in, axis=1, keepdims=True) + 1e-12)
    cos_per_frame = (on_n * off_n).sum(axis=1)
    mean_cos = float(cos_per_frame.mean())
    _log(f"readout (b): mean on/off cosine sim = {mean_cos:.4f} "
         f"(min={cos_per_frame.min():.4f}, max={cos_per_frame.max():.4f})")

    # Readout (c): silhouette of on-cube features clustered by recorded color.
    sil = _silhouette(on_in, color_idx_in)
    _log(f"readout (c): on-cube color silhouette = {sil:.4f}")

    # ----- Verdict -----------------------------------------------------------
    a_pass = (r2_x >= 0.7) and (r2_y >= 0.7)
    b_pass = mean_cos < 0.9
    if a_pass and b_pass:
        verdict = "PASS"
        next_step = "v5 trains WITHOUT aux cube-loc loss (frozen vision viable)."
    elif a_pass or b_pass:
        verdict = "PARTIAL"
        next_step = "v5 ADDS aux cube-loc loss at 0.05x action weight."
    else:
        verdict = "MISSING"
        next_step = ("v5 ADDS aux cube-loc loss at 0.10x weight, AND consider "
                     "unfreezing vision tower with 10x lower LR.")
    _log(f"VERDICT = {verdict}  ({'a:PASS' if a_pass else 'a:fail'}, "
         f"{'b:PASS' if b_pass else 'b:fail'})")
    _log(f"NEXT: {next_step}")

    # ----- Persist artifacts -------------------------------------------------
    np.savez_compressed(
        out_dir / "probe_results.npz",
        cube_xy=cube_xy_in,
        on_cube_feats=on_in.astype(np.float32),
        off_cube_feats=off_in.astype(np.float32),
        cube_color_idx=color_idx_in,
        projected_cube_pixels=projected_cube_pixels[on_cube_in_bounds],
        preds_readout_a=preds_a,
        r2=np.array([r2_x, r2_y]),
        cos_per_frame=cos_per_frame,
        homography=H,
    )
    summary = (
        f"vla_kitting-mu7 spatial-aware vision probe (NEW camera, N={n_in})\n"
        f"  Readout (a): R^2 x={r2_x:+.3f}  y={r2_y:+.3f}  "
        f"=> {'PASS' if a_pass else 'fail'} (>=0.7 on both)\n"
        f"  Readout (b): mean on/off cos-sim = {mean_cos:.4f}  "
        f"=> {'PASS' if b_pass else 'fail'} (<0.9)\n"
        f"  Readout (c): on-cube color silhouette = {sil:.4f}  (diagnostic)\n"
        f"VERDICT: {verdict}\n"
        f"NEXT:    {next_step}\n"
    )
    (out_dir / "probe_summary.txt").write_text(summary)
    _log("wrote probe_summary.txt and probe_results.npz")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
