"""Deterministic selectors and counterfactual boundary verification for AdaFoB v2 Stage 2."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from experiments.option_a_reassessment_v1.selectors import (
    EPS,
    image_on_feature_grid,
    mask_on_feature_grid,
    place_pba_fixed_k2,
)

ARMS = (
    "full10",
    "positive0",
    "boundary_near2",
    "pba_f10_int_k2",
    "pba_f10_scale_k2",
    "cbv_f10_fixed_supp",
    "cbv_f10_scale_supp",
    "cbv_f10_scale_cal",
    "cbv_f10_scale_dual",
    "cbv_hybrid_scale_dual",
    "cbv_fast3_scale_dual",
)


def compute_equivalent_radius(mask: np.ndarray) -> float:
    """Return 2D area-equivalent disk radius sqrt(Area / pi) in pixels."""
    area = float(np.asarray(mask, dtype=bool).sum())
    if area <= 0.0:
        return 0.0
    return float(np.sqrt(area / np.pi))


def scale_adaptive_offset_px(
    mask: np.ndarray,
    alpha: float = 0.25,
    r_min: float = 5.0,
    r_max: float = 16.0,
) -> float:
    """Compute bounded scale-adaptive outward placement distance in pixels."""
    r_eq = compute_equivalent_radius(mask)
    if r_eq <= 0.0:
        return float(r_min)
    return float(np.clip(alpha * r_eq, r_min, r_max))


def place_pba_scale_adaptive(
    coarse_mask: np.ndarray,
    query_z_hw: np.ndarray,
    n_samples: int = 2,
    alpha: float = 0.25,
    r_min: float = 5.0,
    r_max: float = 16.0,
) -> tuple[np.ndarray | None, dict]:
    """Place negative prompts along full 360-deg coarse contour with scale-adaptive offset."""
    mask = np.asarray(coarse_mask, dtype=bool)
    r_eq = compute_equivalent_radius(mask)
    r_offset = scale_adaptive_offset_px(mask, alpha=alpha, r_min=r_min, r_max=r_max)
    pts, meta = place_pba_fixed_k2(mask, query_z_hw, n_samples=n_samples, offset_px=r_offset)
    out_meta = dict(meta)
    out_meta["r_eq_px"] = float(r_eq)
    out_meta["r_offset_px"] = float(r_offset)
    return pts, out_meta


def compute_query_anchored_evidence(
    e_local: np.ndarray | None,
    qry_feat: np.ndarray,
    pos0_cc: np.ndarray,
    pos0_logits_hw: np.ndarray,
    qry_img_hw: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """Compute query self-anchored evidence and midpoint-calibrated support evidence."""
    q = np.asarray(qry_feat, dtype=np.float64)
    pos0_mask = np.asarray(pos0_cc, dtype=bool)
    logits = np.asarray(pos0_logits_hw, dtype=np.float64)
    img = np.asarray(qry_img_hw, dtype=np.float64)
    if q.ndim != 3 or pos0_mask.ndim != 2 or logits.shape != pos0_mask.shape or img.shape != pos0_mask.shape:
        raise ValueError(
            f"Shape mismatch: qry_feat={q.shape}, pos0_cc={pos0_mask.shape}, "
            f"logits={logits.shape}, img={img.shape}"
        )

    h, w = pos0_mask.shape
    fh, fw = q.shape[1], q.shape[2]
    notes: list[str] = []

    e_loc_valid = (
        e_local is not None
        and np.asarray(e_local).shape == (h, w)
        and bool(np.isfinite(e_local).all())
    )
    e_loc_arr = np.asarray(e_local, dtype=np.float64) if e_loc_valid else None

    if not np.isfinite(q).all() or not np.isfinite(logits).all() or not np.isfinite(img).all():
        return (
            e_loc_arr,
            None,
            e_loc_arr,
            {
                "tau_cal": 0.0,
                "m_fg": None,
                "m_bg": None,
                "sep_margin": None,
                "cal_note": "invalid_query_inputs",
                "qry_note": "invalid_query_inputs",
            },
        )

    qn = np.linalg.norm(q, axis=0)
    if (qn <= EPS).any():
        return (
            e_loc_arr,
            None,
            e_loc_arr,
            {
                "tau_cal": 0.0,
                "m_fg": None,
                "m_bg": None,
                "sep_margin": None,
                "cal_note": "zero_query_feat_norm",
                "qry_note": "zero_query_feat_norm",
            },
        )
    qu = q / np.maximum(qn, EPS)

    if not pos0_mask.any():
        return (
            e_loc_arr,
            None,
            e_loc_arr,
            {
                "tau_cal": 0.0,
                "m_fg": None,
                "m_bg": None,
                "sep_margin": None,
                "cal_note": "empty_pos0",
                "qry_note": "empty_pos0",
            },
        )

    hi_thr = float(np.median(logits[pos0_mask]))
    q_core_native = pos0_mask & (logits >= hi_thr)
    if not q_core_native.any():
        q_core_native = pos0_mask

    q_mask_f = mask_on_feature_grid(pos0_mask, (fh, fw)) & (qn > EPS)
    q_core_f = mask_on_feature_grid(q_core_native, (fh, fw)) & (qn > EPS)
    if not q_core_f.any() and q_mask_f.any():
        q_core_f = q_mask_f
        notes.append("q_core_fallback_mask")
    if not q_core_f.any():
        ys, xs = np.nonzero(pos0_mask)
        iy = int(np.clip(np.floor((float(ys.mean()) + 0.5) * fh / h), 0, fh - 1))
        ix = int(np.clip(np.floor((float(xs.mean()) + 0.5) * fw / w), 0, fw - 1))
        if qn[iy, ix] > EPS:
            q_mask_f = q_mask_f.copy()
            q_mask_f[iy, ix] = True
            q_core_f = q_mask_f
            notes.append("q_core_centroid_cell")

    struct = np.ones((3, 3), dtype=bool)
    inner = ndimage.binary_dilation(q_mask_f, structure=struct, iterations=1, border_value=0)
    outer = ndimage.binary_dilation(q_mask_f, structure=struct, iterations=5, border_value=0)
    ring = outer & ~inner & (qn > EPS)
    q_grid = image_on_feature_grid(img, (fh, fw))
    air = q_grid <= (float(q_grid.min()) + 0.05)

    if ring.any():
        ring_tissue = ring & ~air
        if ring_tissue.any():
            q_bg_f = ring_tissue
        else:
            q_bg_f = ring
            notes.append("qry_ring_with_air")
    else:
        bg_global = (~inner) & (qn > EPS) & ~air
        if bg_global.any():
            q_bg_f = bg_global
            notes.append("qry_ring_fallback_global")
        else:
            q_bg_f = (~inner) & (qn > EPS)
            notes.append("qry_ring_fallback_allbg")

    e_qry: np.ndarray | None = None
    if not q_core_f.any() or not q_bg_f.any():
        notes.append("empty_q_regions")
    else:
        f_qry = qu[:, q_core_f].mean(axis=1)
        b_qry = qu[:, q_bg_f].mean(axis=1)
        fn = float(np.linalg.norm(f_qry))
        bn = float(np.linalg.norm(b_qry))
        if min(fn, bn) <= EPS:
            notes.append("degenerate_qry_proto")
        else:
            f_qry /= fn
            b_qry /= bn
            if float(np.linalg.norm(f_qry - b_qry)) <= EPS:
                notes.append("equal_qry_proto")
            else:
                coarse_q = np.clip(np.einsum("c,chw->hw", f_qry - b_qry, qu) / 2.0, -1.0, 1.0)
                yy = (np.arange(h) + 0.5) * fh / h - 0.5
                xx = (np.arange(w) + 0.5) * fw / w - 0.5
                gy, gx = np.meshgrid(yy, xx, indexing="ij")
                e_qry = ndimage.map_coordinates(coarse_q, [gy, gx], order=1, mode="nearest", prefilter=False)

    tau_cal = 0.0
    m_fg: float | None = None
    m_bg: float | None = None
    sep_margin: float | None = None
    cal_note: str | None = None
    e_cal_supp: np.ndarray | None = None

    if e_loc_arr is None:
        cal_note = "invalid_e_local"
    elif q_core_f.any() and q_bg_f.any():
        e_loc_f = image_on_feature_grid(e_loc_arr, (fh, fw))
        m_fg = float(np.median(e_loc_f[q_core_f]))
        m_bg = float(np.median(e_loc_f[q_bg_f]))
        sep_margin = float(m_fg - m_bg)
        if sep_margin > 0.0:
            tau_cal = float(0.5 * (m_fg + m_bg))
        else:
            tau_cal = 0.0
            cal_note = "no_sep"
        e_cal_supp = e_loc_arr - tau_cal
    else:
        cal_note = "missing_q_regions"
        e_cal_supp = e_loc_arr

    if e_cal_supp is not None and e_qry is not None:
        e_dual = 0.5 * e_cal_supp + 0.5 * e_qry
    elif e_qry is not None:
        e_dual = e_qry
    else:
        e_dual = e_cal_supp

    meta = {
        "tau_cal": float(tau_cal),
        "m_fg": m_fg,
        "m_bg": m_bg,
        "sep_margin": sep_margin,
        "cal_note": cal_note,
        "qry_note": ",".join(notes) if notes else None,
    }
    return e_cal_supp, e_qry, e_dual, meta


def score_mask_energy(mask_cc: np.ndarray, evidence_hw: np.ndarray) -> float:
    """Return normalized integrated region energy, or -inf for an empty mask."""
    m = np.asarray(mask_cc, dtype=bool)
    ev = np.asarray(evidence_hw, dtype=np.float64)
    if m.shape != ev.shape:
        raise ValueError(f"Mask shape {m.shape} does not match evidence shape {ev.shape}")
    if not m.any():
        return float("-inf")
    return float(ev[m].sum() / float(ev.size))


def select_best_candidate(
    candidates_cc: dict[str, np.ndarray],
    evidence_hw: np.ndarray | None,
    fallback_key: str,
) -> tuple[str, dict]:
    """Choose candidate mask maximizing normalized evidence energy with non-empty guard."""
    if evidence_hw is None:
        return fallback_key, {"fallback": "invalid_evidence", "chosen_key": fallback_key, "scores": {}}
    ev = np.asarray(evidence_hw, dtype=np.float64)
    if not np.isfinite(ev).all():
        return fallback_key, {"fallback": "invalid_evidence", "chosen_key": fallback_key, "scores": {}}

    scores: dict[str, float] = {}
    best_key: str | None = None
    best_score = float("-inf")
    for key, mask in candidates_cc.items():
        sc = score_mask_energy(mask, ev)
        scores[key] = sc
        if sc > best_score:
            best_score = sc
            best_key = key

    if best_key is None or not np.isfinite(best_score):
        return fallback_key, {"fallback": "all_empty", "chosen_key": fallback_key, "scores": scores}

    return best_key, {"fallback": None, "chosen_key": best_key, "scores": scores}
