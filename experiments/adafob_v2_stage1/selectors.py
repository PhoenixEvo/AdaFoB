"""Semantic-gated contour negative-prompt selectors for AdaFoB v2 Stage 1."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from experiments.option_a_reassessment_v1.selectors import (
    compute_support_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
)

ARMS = (
    "full10",
    "positive0",
    "boundary_near2",
    "pba_f10_int_k2",
    "pba_pos0_int_k2",
    "semgate_f10_k2_r5",
    "semgate_pos0_k2_r5",
    "semgate_pos0_only_k2_r5",
    "semgate_pos0_raymarch_k2",
    "semgate_pos0_safe_k2",
    "semgate_f10_safe_k2",
)


def place_semgate_pba(
    coarse_mask: np.ndarray,
    query_z_hw: np.ndarray,
    e_local: np.ndarray | None,
    n_samples: int = 2,
    offsets: tuple[float, ...] = (5.0,),
    use_intensity: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Place negative prompts along coarse contour gated by local-ring semantic evidence."""
    mask = np.asarray(coarse_mask, dtype=bool)
    img = np.asarray(query_z_hw, dtype=np.float64)
    if mask.sum() < 10:
        return np.zeros((0, 2), dtype=np.float64), {
            "fallback": "small_mask",
            "valid_arc_frac": 0.0,
            "mean_offset_px": 0.0,
            "k_placed": 0,
        }

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64), {
            "fallback": "no_contours",
            "valid_arc_frac": 0.0,
            "mean_offset_px": 0.0,
            "k_placed": 0,
        }
    main_cnt = max(contours, key=lambda c: len(c)).reshape(-1, 2).astype(np.float64)
    if len(main_cnt) < 6:
        return np.zeros((0, 2), dtype=np.float64), {
            "fallback": "short_contour",
            "valid_arc_frac": 0.0,
            "mean_offset_px": 0.0,
            "k_placed": 0,
        }

    if e_local is None:
        return None, {"fallback": "invalid_evidence", "valid_arc_frac": 0.0, "mean_offset_px": 0.0, "k_placed": 0}
    ev = np.asarray(e_local, dtype=np.float64)
    if ev.shape != mask.shape or not np.isfinite(ev).all():
        return None, {"fallback": "invalid_evidence", "valid_arc_frac": 0.0, "mean_offset_px": 0.0, "k_placed": 0}

    h, w = mask.shape
    # Outward signed distance field: negative inside mask, positive outside.
    sdf = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    gy, gx = np.gradient(sdf)
    cx = np.clip(main_cnt[:, 0], 0.0, w - 1.0)
    cy = np.clip(main_cnt[:, 1], 0.0, h - 1.0)
    nx = ndimage.map_coordinates(gx, [cy, cx], order=1, mode="nearest", prefilter=False)
    ny = ndimage.map_coordinates(gy, [cy, cx], order=1, mode="nearest", prefilter=False)
    normals = np.stack([nx, ny], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    ys_fg, xs_fg = np.nonzero(mask)
    centroid = np.array([float(xs_fg.mean()), float(ys_fg.mean())], dtype=np.float64)
    radial = main_cnt - centroid
    r_norms = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    normals = np.where(norms >= 1e-6, normals / np.maximum(norms, 1e-6), radial / r_norms)

    m = len(main_cnt)
    found = np.zeros(m, dtype=bool)
    chosen_pt = np.zeros((m, 2), dtype=np.float64)
    chosen_r = np.zeros(m, dtype=np.float64)
    sem_bg = np.zeros(m, dtype=np.float64)

    for r in offsets:
        rf = float(r)
        cand = main_cnt + rf * normals
        cand_x = np.clip(cand[:, 0], 0.0, w - 1.0)
        cand_y = np.clip(cand[:, 1], 0.0, h - 1.0)
        px = np.clip(np.rint(cand_x).astype(int), 0, w - 1)
        py = np.clip(np.rint(cand_y).astype(int), 0, h - 1)
        outside = ~mask[py, px]
        e_vals = ndimage.map_coordinates(ev, [cand_y, cand_x], order=1, mode="nearest", prefilter=False)
        ok = (~found) & outside & (e_vals < 0.0)
        if ok.any():
            found[ok] = True
            chosen_pt[ok, 0] = cand_x[ok]
            chosen_pt[ok, 1] = cand_y[ok]
            chosen_r[ok] = rf
            sem_bg[ok] = -e_vals[ok]

    if use_intensity:
        pts_in = main_cnt - 3.0 * normals
        default_out_x = np.clip(main_cnt[:, 0] + float(offsets[0]) * normals[:, 0], 0.0, w - 1.0)
        default_out_y = np.clip(main_cnt[:, 1] + float(offsets[0]) * normals[:, 1], 0.0, h - 1.0)
        out_x = np.where(found, chosen_pt[:, 0], default_out_x)
        out_y = np.where(found, chosen_pt[:, 1], default_out_y)
        i_in = ndimage.map_coordinates(
            img,
            [np.clip(pts_in[:, 1], 0.0, h - 1.0), np.clip(pts_in[:, 0], 0.0, w - 1.0)],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        i_out = ndimage.map_coordinates(
            img,
            [out_y, out_x],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        sigma_local = max(float(np.std(img[mask])), 0.25)
        l_s = np.exp(-((i_in - i_out) ** 2) / (2.0 * (sigma_local**2)))
        w_s = l_s * sem_bg
    else:
        w_s = sem_bg

    rolled = np.roll(main_cnt, -1, axis=0)
    ds_arc = np.maximum(np.linalg.norm(rolled - main_cnt, axis=1), 1e-6)
    total_arc = max(float(ds_arc.sum()), 1e-12)
    valid_arc_frac = float((ds_arc * (sem_bg > 0.0)).sum() / total_arc)

    cum = np.cumsum(w_s * ds_arc)
    total_w = float(cum[-1])
    if total_w <= 1e-12 or not np.isfinite(total_w):
        return np.zeros((0, 2), dtype=np.float64), {
            "fallback": "no_bg_arc",
            "valid_arc_frac": valid_arc_frac,
            "mean_offset_px": 0.0,
            "k_placed": 0,
        }

    cdf = cum / total_w
    quantiles = np.linspace(0.5 / n_samples, 1.0 - 0.5 / n_samples, n_samples)
    chosen = np.clip(np.searchsorted(cdf, quantiles, side="left"), 0, m - 1)

    dedup_pts: list[np.ndarray] = []
    dedup_rs: list[float] = []
    for idx in chosen:
        if not found[idx]:
            continue
        pt = chosen_pt[idx]
        if any(np.linalg.norm(pt - prev) < 1.0 for prev in dedup_pts):
            continue
        dedup_pts.append(pt.copy())
        dedup_rs.append(float(chosen_r[idx]))

    if not dedup_pts:
        return np.zeros((0, 2), dtype=np.float64), {
            "fallback": "no_bg_arc",
            "valid_arc_frac": valid_arc_frac,
            "mean_offset_px": 0.0,
            "k_placed": 0,
        }

    return np.stack(dedup_pts, axis=0), {
        "fallback": None,
        "valid_arc_frac": valid_arc_frac,
        "mean_offset_px": float(np.mean(dedup_rs)),
        "k_placed": int(len(dedup_pts)),
    }
