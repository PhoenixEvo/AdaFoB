"""Backbone-invariant dual-background, FoB-prior, and logit-weighted counterfactual selectors."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from experiments.option_a_reassessment_v1.selectors import (
    EPS,
    image_on_feature_grid,
    mask_on_feature_grid,
    place_pba_fixed_k2,
)

RANDOM_SEEDS = (42, 43, 44, 45, 46)

CBV_EVIDENCE_FAMILIES = (
    "sam_localring",
    "sam_dualbg",
    "fob_prob",
    "fob_dualbg",
    "joint_prob_sam",
    "logit_fob_prob",
)

CBV_ARMS = tuple(
    arm
    for fam in CBV_EVIDENCE_FAMILIES
    for arm in (f"cbv_{fam}_all", f"cbv_{fam}_no0")
)

RAW_ARMS = (
    "fob_coarse_prior",
    "full10",
    "positive0",
    "random2_s42",
    "random2_s43",
    "random2_s44",
    "random2_s45",
    "random2_s46",
    "boundary_near2",
    "pba_f10_int_k2",
) + CBV_ARMS

SUMMARY_ARMS = (
    "fob_coarse_prior",
    "full10",
    "positive0",
    "random2_mean5",
    "boundary_near2",
    "pba_f10_int_k2",
) + CBV_ARMS

METHOD_ARMS = ("pba_f10_int_k2",) + CBV_ARMS

ALL_REPORTED_ARMS = SUMMARY_ARMS + tuple(f"random2_s{s}" for s in RANDOM_SEEDS)


def _upsample_to_native(coarse: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    fh, fw = coarse.shape
    h, w = int(native_hw[0]), int(native_hw[1])
    yy = (np.arange(h) + 0.5) * fh / h - 0.5
    xx = (np.arange(w) + 0.5) * fw / w - 0.5
    y, x = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(coarse, [y, x], order=1, mode="nearest", prefilter=False)


def compute_dualbg_evidence(
    support_features: np.ndarray,
    support_mask_fg: np.ndarray,
    query_features: np.ndarray,
    native_hw: tuple[int, int],
    support_image_hw: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Contrast query features against min(ring, global) support background separation."""
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

    inner = ndimage.binary_dilation(m, structure=struct, iterations=1, border_value=0)
    outer = ndimage.binary_dilation(m, structure=struct, iterations=5, border_value=0)
    ring = outer & ~inner & (sn > EPS)
    outside = (~inner) & (sn > EPS)

    if support_image_hw is not None:
        s_grid = image_on_feature_grid(support_image_hw, m.shape)
        air = s_grid <= (float(s_grid.min()) + 0.05)
        ring_tissue = ring & ~air
        outside_tissue = outside & ~air
    else:
        ring_tissue = ring
        outside_tissue = outside

    if ring_tissue.any():
        bg_ring = ring_tissue
    elif ring.any():
        bg_ring = ring
        notes.append("ring_with_air")
    elif outside_tissue.any():
        bg_ring = outside_tissue
        notes.append("ring_fallback_global")
    else:
        bg_ring = outside

    if outside_tissue.any():
        bg_global = outside_tissue
    else:
        bg_global = outside
        if bg_global.any():
            notes.append("global_with_air")

    if not bg_ring.any() or not bg_global.any():
        return None, "invalid_evidence"

    fg = su[:, core].mean(axis=1)
    br = su[:, bg_ring].mean(axis=1)
    bg = su[:, bg_global].mean(axis=1)
    fn = float(np.linalg.norm(fg))
    brn = float(np.linalg.norm(br))
    bgn = float(np.linalg.norm(bg))
    if min(fn, brn, bgn) <= EPS:
        return None, "invalid_evidence"
    fg /= fn
    br /= brn
    bg /= bgn

    cos_fg = np.einsum("c,chw->hw", fg, qu)
    cos_ring = np.einsum("c,chw->hw", br, qu)
    cos_global = np.einsum("c,chw->hw", bg, qu)
    coarse = np.clip(0.5 * (cos_fg - np.maximum(cos_ring, cos_global)), -1.0, 1.0)
    out = _upsample_to_native(coarse, native_hw)
    return out, (",".join(notes) if notes else None)


def compute_fob_prob_evidence(p_fob_hw: np.ndarray | None) -> np.ndarray | None:
    """Map FoB [0, 1] medical foreground probability map to signed [-1, 1] evidence."""
    if p_fob_hw is None:
        return None
    p = np.asarray(p_fob_hw, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all():
        return None
    return 2.0 * np.clip(p, 0.0, 1.0) - 1.0


def compute_joint_evidence(
    e_fob_prob: np.ndarray | None,
    e_sam_dualbg: np.ndarray | None,
) -> np.ndarray | None:
    """Average FoB medical prior evidence and SAM dual-background evidence safely."""
    ok_f = e_fob_prob is not None and np.asarray(e_fob_prob).ndim == 2 and bool(np.isfinite(e_fob_prob).all())
    ok_s = e_sam_dualbg is not None and np.asarray(e_sam_dualbg).ndim == 2 and bool(np.isfinite(e_sam_dualbg).all())
    if ok_f and ok_s:
        ef = np.asarray(e_fob_prob, dtype=np.float64)
        es = np.asarray(e_sam_dualbg, dtype=np.float64)
        if ef.shape != es.shape:
            raise ValueError(f"Shape mismatch in joint evidence: {ef.shape} vs {es.shape}")
        return 0.5 * ef + 0.5 * es
    if ok_f:
        return np.asarray(e_fob_prob, dtype=np.float64)
    if ok_s:
        return np.asarray(e_sam_dualbg, dtype=np.float64)
    return None


def select_best_candidate_logit_weighted(
    candidates_cc: dict[str, np.ndarray],
    candidates_logits: dict[str, np.ndarray],
    evidence_hw: np.ndarray | None,
    fallback_key: str,
) -> tuple[str, dict]:
    """Select candidate mask maximizing SAM-confidence weighted evidence energy."""
    if evidence_hw is None:
        return fallback_key, {"fallback": "invalid_evidence", "chosen_key": fallback_key, "scores": {}}
    ev = np.asarray(evidence_hw, dtype=np.float64)
    if not np.isfinite(ev).all():
        return fallback_key, {"fallback": "invalid_evidence", "chosen_key": fallback_key, "scores": {}}

    scores: dict[str, float] = {}
    best_key: str | None = None
    best_score = float("-inf")
    denom = float(ev.size)

    for key, mask in candidates_cc.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != ev.shape:
            raise ValueError(f"Mask shape {m.shape} does not match evidence shape {ev.shape}")
        if not m.any() or key not in candidates_logits:
            scores[key] = float("-inf")
            continue
        logits = np.asarray(candidates_logits[key], dtype=np.float64)
        if logits.shape != ev.shape or not np.isfinite(logits).all():
            scores[key] = float("-inf")
            continue
        w = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        sc = float((w[m] * ev[m]).sum() / denom)
        scores[key] = sc
        if sc > best_score:
            best_score = sc
            best_key = key

    if best_key is None or not np.isfinite(best_score):
        return fallback_key, {"fallback": "all_empty", "chosen_key": fallback_key, "scores": scores}

    return best_key, {"fallback": None, "chosen_key": best_key, "scores": scores}


__all__ = [
    "ALL_REPORTED_ARMS",
    "CBV_ARMS",
    "CBV_EVIDENCE_FAMILIES",
    "METHOD_ARMS",
    "RANDOM_SEEDS",
    "RAW_ARMS",
    "SUMMARY_ARMS",
    "compute_dualbg_evidence",
    "compute_fob_prob_evidence",
    "compute_joint_evidence",
    "mask_on_feature_grid",
    "place_pba_fixed_k2",
    "select_best_candidate_logit_weighted",
]
