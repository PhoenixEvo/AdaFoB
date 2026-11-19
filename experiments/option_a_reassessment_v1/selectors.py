"""Deterministic, query-label-free negative-prompt selectors for Option A."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from analysis.controls_v1.core import boundary_distances, proximity

ARMS = (
    "full10",
    "positive0",
    "boundary_near2",
    "resp_magnitude_k2",
    "resp_signed_global_k2",
    "resp_signed_localring_k2",
    "resp_signed_adaptive",
    "local_contrast_near2",
    "pba_fixed_k2",
)

EPS = float(np.finfo(np.float32).eps)


def mask_on_feature_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map native binary mask onto SAM (H_f, W_f) feature grid via pixel-center sampling."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m.squeeze(axis=0)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D/3D support mask, got shape {mask.shape}")
    h, w = m.shape
    fh, fw = shape
    iy = np.clip(np.floor((np.arange(fh) + 0.5) * h / fh).astype(int), 0, h - 1)
    ix = np.clip(np.floor((np.arange(fw) + 0.5) * w / fw).astype(int), 0, w - 1)
    return m[np.ix_(iy, ix)]


def image_on_feature_grid(image_hw: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sample 2D z-score CT image onto (H_f, W_f) feature grid using pixel centers."""
    img = np.asarray(image_hw, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    h, w = img.shape
    fh, fw = shape
    yy = (np.arange(fh) + 0.5) * h / fh - 0.5
    xx = (np.arange(fw) + 0.5) * w / fw - 0.5
    y, x = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(img, [y, x], order=1, mode="nearest", prefilter=False)


def compute_support_evidence(
    support_features: np.ndarray,
    support_mask_fg: np.ndarray,
    query_features: np.ndarray,
    native_hw: tuple[int, int],
    mode: str = "global",
    support_image_hw: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Compute signed support foreground-vs-background cosine contrast on native grid."""
    if mode not in ("global", "localring"):
        raise ValueError(f"Unsupported evidence mode: {mode}")
    s = np.asarray(support_features, dtype=np.float64)
    q = np.asarray(query_features, dtype=np.float64)
    m = np.asarray(support_mask_fg, dtype=bool)
    if s.ndim != 3 or q.ndim != 3 or s.shape[0] != q.shape[0] or m.shape != s.shape[1:]:
        raise ValueError("Feature/mask shape mismatch")
    if not np.isfinite(s).all() or not np.isfinite(q).all():
        return None, "invalid_evidence"

    sn = np.linalg.norm(s, axis=0)
    qn = np.linalg.norm(q, axis=0)
    if (qn <= EPS).any():
        return None, "invalid_evidence"
    su = s / np.maximum(sn, EPS)
    qu = q / np.maximum(qn, EPS)

    notes: list[str] = []
    struct = np.ones((3, 3), dtype=bool)
    core = ndimage.binary_erosion(m, structure=struct, border_value=0) & (sn > EPS)
    if not core.any():
        core = m & (sn > EPS)
        if not core.any():
            return None, "invalid_evidence"
        notes.append("core_uneroded")

    if mode == "global":
        bg = ~ndimage.binary_dilation(m, structure=struct, border_value=0) & (sn > EPS)
    else:
        inner = ndimage.binary_dilation(m, structure=struct, iterations=1, border_value=0)
        outer = ndimage.binary_dilation(m, structure=struct, iterations=5, border_value=0)
        ring = outer & ~inner & (sn > EPS)
        if support_image_hw is not None and ring.any():
            s_grid = image_on_feature_grid(support_image_hw, m.shape)
            air = s_grid <= (float(s_grid.min()) + 0.05)
            ring_tissue = ring & ~air
        else:
            ring_tissue = ring
        if ring_tissue.any():
            bg = ring_tissue
        elif ring.any():
            bg = ring
            notes.append("ring_with_air")
        else:
            bg = ~inner & (sn > EPS)
            notes.append("ring_fallback_global")

    if not bg.any():
        return None, "invalid_evidence"

    fg = su[:, core].mean(axis=1)
    back = su[:, bg].mean(axis=1)
    fn = float(np.linalg.norm(fg))
    bn = float(np.linalg.norm(back))
    if min(fn, bn) <= EPS:
        return None, "invalid_evidence"
    fg /= fn
    back /= bn
    if float(np.linalg.norm(fg - back)) <= EPS:
        return None, "invalid_evidence"

    coarse = np.clip(np.einsum("c,chw->hw", fg - back, qu) / 2.0, -1.0, 1.0)
    h, w = native_hw
    yy = (np.arange(h) + 0.5) * q.shape[1] / h - 0.5
    xx = (np.arange(w) + 0.5) * q.shape[2] / w - 0.5
    y, x = np.meshgrid(yy, xx, indexing="ij")
    out = ndimage.map_coordinates(coarse, [y, x], order=1, mode="nearest", prefilter=False)
    return out, (",".join(notes) if notes else None)


def _top_k_sorted(scores: np.ndarray, k: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(i)))
    return sorted(int(i) for i in order[:k])


def select_response_arms(
    full_raw: np.ndarray,
    deleted_masks: np.ndarray,
    e_global: np.ndarray | None,
    e_local: np.ndarray | None,
    fallback_ids: list[int] | tuple[int, ...],
    k_max_adaptive: int = 6,
) -> dict:
    """Select counterfactual leave-one-out response arms without Route B af<=lf veto."""
    full = np.asarray(full_raw, dtype=bool)
    deletions = np.asarray(deleted_masks, dtype=bool)
    if deletions.shape != (10,) + full.shape:
        raise ValueError(f"Expected (10, {full.shape}) deleted_masks, got {deletions.shape}")

    added = deletions & ~full
    lost = full & ~deletions
    n = float(full.size)
    fb = [int(i) for i in fallback_ids]
    notes: dict[str, str | None] = {}

    mag = (added | lost).mean(axis=(1, 2))
    resp_magnitude_k2 = _top_k_sorted(mag, 2)
    notes["resp_magnitude_k2"] = None

    if e_global is None or e_global.shape != full.shape or not np.isfinite(e_global).all():
        resp_signed_global_k2 = list(fb)
        score_g = None
        notes["resp_signed_global_k2"] = "fallback_invalid_evidence"
    else:
        score_g = ((lost.astype(np.float64) - added.astype(np.float64)) * e_global).sum(axis=(1, 2)) / n
        resp_signed_global_k2 = _top_k_sorted(score_g, 2)
        notes["resp_signed_global_k2"] = None

    if e_local is None or e_local.shape != full.shape or not np.isfinite(e_local).all():
        resp_signed_localring_k2 = list(fb)
        resp_signed_adaptive = list(fb)
        score_l = None
        notes["resp_signed_localring_k2"] = "fallback_invalid_evidence"
        notes["resp_signed_adaptive"] = "fallback_invalid_evidence"
    else:
        score_l = ((lost.astype(np.float64) - added.astype(np.float64)) * e_local).sum(axis=(1, 2)) / n
        resp_signed_localring_k2 = _top_k_sorted(score_l, 2)
        notes["resp_signed_localring_k2"] = None
        pos_ids = [i for i in range(10) if float(score_l[i]) > 0.0]
        pos_ranked = sorted(pos_ids, key=lambda i: (-float(score_l[i]), int(i)))
        resp_signed_adaptive = sorted(int(i) for i in pos_ranked[:k_max_adaptive])
        notes["resp_signed_adaptive"] = "zero_negatives" if len(resp_signed_adaptive) == 0 else None

    return {
        "resp_magnitude_k2": resp_magnitude_k2,
        "resp_signed_global_k2": resp_signed_global_k2,
        "resp_signed_localring_k2": resp_signed_localring_k2,
        "resp_signed_adaptive": resp_signed_adaptive,
        "notes": notes,
        "mag_scores": mag.tolist(),
        "global_scores": score_g.tolist() if score_g is not None else None,
        "local_scores": score_l.tolist() if score_l is not None else None,
    }


def select_local_contrast_near2(
    nr: np.ndarray,
    full_raw: np.ndarray,
    query_z_hw: np.ndarray,
) -> tuple[list[int], dict]:
    """Select k=2 FoB candidates weighted by z-score intensity similarity to coarse foreground."""
    pts = np.asarray(nr, dtype=np.float64)
    mask = np.asarray(full_raw, dtype=bool)
    img = np.asarray(query_z_hw, dtype=np.float64)
    if mask.sum() < 5:
        ids = proximity(pts, np.arange(len(pts)), 2, mask, far=False)
        return [int(i) for i in ids], {"fallback": "small_mask"}

    ds, _ = boundary_distances(pts, mask)
    fg_vals = img[mask]
    mu_fg = float(np.median(fg_vals))
    sigma_fg = max(float(np.std(fg_vals)), 0.25)

    h, w = img.shape
    in_bounds = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= w - 1.0)
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] <= h - 1.0)
    )
    cand_vals = np.zeros(len(pts), dtype=np.float64)
    if in_bounds.any():
        cand_vals[in_bounds] = ndimage.map_coordinates(
            img,
            [pts[in_bounds, 1], pts[in_bounds, 0]],
            order=1,
            mode="nearest",
            prefilter=False,
        )
    l_s = np.zeros(len(pts), dtype=np.float64)
    l_s[in_bounds] = np.exp(-((cand_vals[in_bounds] - mu_fg) ** 2) / (2.0 * (sigma_fg**2)))
    d_eff = ds / (0.1 + l_s)
    order = sorted(range(len(pts)), key=lambda i: (float(d_eff[i]), int(i)))
    selected = [int(i) for i in order[:2]]
    return selected, {
        "fallback": None,
        "mu_fg": mu_fg,
        "sigma_fg": sigma_fg,
        "l_s": l_s.tolist(),
        "d_eff": d_eff.tolist(),
    }


def place_pba_fixed_k2(
    full_raw: np.ndarray,
    query_z_hw: np.ndarray,
    n_samples: int = 2,
    offset_px: float = 5.0,
) -> tuple[np.ndarray | None, dict]:
    """Bug-fixed PBA placement: full contour weighted CDF with z-score sigma_local and SDF normals."""
    mask = np.asarray(full_raw, dtype=bool)
    img = np.asarray(query_z_hw, dtype=np.float64)
    if mask.sum() < 10:
        return None, {"fallback": "small_mask"}

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, {"fallback": "no_contours"}
    main_cnt = max(contours, key=lambda c: len(c)).reshape(-1, 2).astype(np.float64)
    if len(main_cnt) < 6:
        return None, {"fallback": "short_contour"}

    # Signed distance is positive outside mask and negative inside, so its gradient points outward.
    sdf = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    gy, gx = np.gradient(sdf)
    cx = np.clip(main_cnt[:, 0], 0.0, mask.shape[1] - 1.0)
    cy = np.clip(main_cnt[:, 1], 0.0, mask.shape[0] - 1.0)
    nx = ndimage.map_coordinates(gx, [cy, cx], order=1, mode="nearest", prefilter=False)
    ny = ndimage.map_coordinates(gy, [cy, cx], order=1, mode="nearest", prefilter=False)
    normals = np.stack([nx, ny], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    ys_fg, xs_fg = np.nonzero(mask)
    centroid = np.array([float(xs_fg.mean()), float(ys_fg.mean())], dtype=np.float64)
    radial = main_cnt - centroid
    r_norms = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    normals = np.where(norms >= 1e-6, normals / np.maximum(norms, 1e-6), radial / r_norms)

    h, w = img.shape
    pts_in = main_cnt - 3.0 * normals
    pts_out = main_cnt + 3.0 * normals
    i_in = ndimage.map_coordinates(
        img,
        [np.clip(pts_in[:, 1], 0.0, h - 1.0), np.clip(pts_in[:, 0], 0.0, w - 1.0)],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    i_out = ndimage.map_coordinates(
        img,
        [np.clip(pts_out[:, 1], 0.0, h - 1.0), np.clip(pts_out[:, 0], 0.0, w - 1.0)],
        order=1,
        mode="nearest",
        prefilter=False,
    )

    sigma_local = max(float(np.std(img[mask])), 0.25)
    l_s = np.exp(-((i_in - i_out) ** 2) / (2.0 * (sigma_local**2)))
    w_s = l_s + 0.1

    rolled = np.roll(main_cnt, -1, axis=0)
    ds_arc = np.maximum(np.linalg.norm(rolled - main_cnt, axis=1), 1e-6)
    cum = np.cumsum(w_s * ds_arc)
    total = float(cum[-1])
    if total <= 0.0 or not np.isfinite(total):
        return None, {"fallback": "degenerate_cdf"}
    cdf = cum / total

    quantiles = np.linspace(0.5 / n_samples, 1.0 - 0.5 / n_samples, n_samples)
    chosen = np.searchsorted(cdf, quantiles, side="left")
    chosen = np.clip(chosen, 0, len(main_cnt) - 1)

    placed = []
    for idx in chosen:
        base_pt = main_cnt[idx]
        nvec = normals[idx]
        pt = base_pt + offset_px * nvec
        for _ in range(3):
            px = int(np.clip(np.rint(pt[0]), 0, w - 1))
            py = int(np.clip(np.rint(pt[1]), 0, h - 1))
            if not mask[py, px]:
                break
            pt = pt + 3.0 * nvec
        pt_clipped = np.array([np.clip(pt[0], 0.0, w - 1.0), np.clip(pt[1], 0.0, h - 1.0)], dtype=np.float64)
        placed.append(pt_clipped)

    return np.stack(placed, axis=0), {
        "fallback": None,
        "sigma_local": sigma_local,
        "quantiles": quantiles.tolist(),
        "chosen_indices": [int(i) for i in chosen],
        "contour_len": int(len(main_cnt)),
    }
