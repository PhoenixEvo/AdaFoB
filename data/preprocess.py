"""Canonical preprocessing, alignment detection and prompt hygiene.

ONE implementation, imported by train.py / eval.py / diagnose.py /
check_alignment.py.  The original Dice collapse came from two scripts disagreeing
about what an input image is; keep it that way.

Contents
--------
* canonical intensity domain      -- to_canonical, resize_volume
* model-facing normalisation      -- norm_zscore, norm_unit_legacy
* SAM input                       -- sam_uint8_from_canonical
* oracle prompts                  -- oracle_prompts
* alignment metrics + detection   -- grad_map, contour_grad, edge_z,
                                     TRANSFORMS, detect_alignment,
                                     apply_label_transform
* prompt hygiene                  -- sanitize_prompts
"""

import numpy as np
import cv2

HU_WINDOW = (-125.0, 275.0)


# ---------------------------------------------------------------------------
# Canonical intensity domain
# ---------------------------------------------------------------------------

def to_canonical(vol, hu_window=HU_WINDOW):
    """Map an arbitrary CT volume into the canonical [0, 255] domain.

    Mimics Ouyang et al.'s `sabs_CT_normalized` output, which is what the FoB
    checkpoints were trained on.  Returns (volume_float32, detected_domain).
    """
    vol = np.asarray(vol, dtype=np.float64)
    vmin, vmax = float(vol.min()), float(vol.max())

    if vmin < -20.0:                       # raw Hounsfield units
        lo, hi = float(hu_window[0]), float(hu_window[1])
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo) * 255.0
        domain = "raw_hu"
    elif vmax <= 1.5:                      # already [0, 1]
        vol = vol * 255.0
        domain = "unit"
    elif vmax <= 260.0:                    # already [0, 255]
        domain = "byte"
    else:                                  # unknown positive range
        lo, hi = np.percentile(vol, [0.5, 99.5])
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo + 1e-8) * 255.0
        domain = "percentile"

    return vol.astype(np.float32), domain


def resize_volume(vol, size=(256, 256), is_label=False):
    """Resize each slice. `size` is (width, height), matching cv2 convention."""
    w, h = size
    if vol.shape[1:] == (h, w):
        return vol
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    src = vol
    # cv2 is picky about integer depths across versions; route labels through
    # int16 when the range allows and restore the original dtype afterwards.
    cast_back = None
    if is_label and vol.dtype not in (np.uint8, np.int16, np.float32, np.float64):
        if int(vol.max()) < 32767 and int(vol.min()) > -32768:
            cast_back = vol.dtype
            src = vol.astype(np.int16)
    out = np.zeros((vol.shape[0], h, w), dtype=src.dtype)
    for z in range(vol.shape[0]):
        out[z] = cv2.resize(src[z], (w, h), interpolation=interp)
    return out.astype(cast_back) if cast_back is not None else out


# ---------------------------------------------------------------------------
# Model-facing normalisation
# ---------------------------------------------------------------------------

def norm_zscore(vol_canon, mean, std):
    """FoB-style input: z-score of the canonical volume -> (Z, 3, H, W)."""
    img = (np.asarray(vol_canon, dtype=np.float64) - mean) / (std + 1e-8)
    return np.stack(3 * [img], axis=1).astype(np.float32)


def norm_unit_legacy(vol_canon):
    """LEGACY [0,1] transform from the original train.py `_load_slice`.

    Kept only so old checkpoints stay reproducible; incompatible with
    FoB-pretrained weights.
    """
    vol_canon = np.asarray(vol_canon)
    out = np.empty((vol_canon.shape[0], 3) + vol_canon.shape[1:], dtype=np.float32)
    for z in range(vol_canon.shape[0]):
        s = vol_canon[z].astype(np.float64)
        s = (s - s.mean()) / (s.std() + 1e-8)
        s = np.clip(s * 50.0 + 128.0, 0, 255).astype(np.uint8)
        out[z] = np.stack(3 * [s.astype(np.float32) / 255.0], axis=0)
    return out


def sam_uint8_from_canonical(slice_canon):
    """(H, W) canonical float -> (H, W, 3) uint8 for SAM.

    Same min-max as FoB's SAM.pre_process but on a *windowed* slice, so
    soft-tissue contrast survives.  min-max is affine-invariant, so the z-score
    never changes this -- only the window does.
    """
    s = np.asarray(slice_canon, dtype=np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-8:
        u8 = np.zeros_like(s, dtype=np.uint8)
    else:
        u8 = ((s - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack(3 * [u8], axis=-1)


# ---------------------------------------------------------------------------
# Oracle prompts
# ---------------------------------------------------------------------------

def oracle_prompts(gt, n_pos=10, n_neg=10, r_outer=15, r_inner=13,
                   erode_iters=3, rng=None):
    """Ground-truth-derived prompts in SAM's (x, y) order.

    Positives from the eroded morphological core (the centroid of a crescent
    organ can fall outside it); negatives from FoB's differential dilation band.
    """
    import random as _random
    rng = rng or _random.Random(0)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)

    core = cv2.erode(gt, k, iterations=erode_iters)
    if core.sum() < n_pos:
        core = gt
    cy, cx = np.nonzero(core)
    if len(cy) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    sel = rng.sample(range(len(cy)), min(n_pos, len(cy)))
    pos = np.stack([cx[sel], cy[sel]], axis=1).astype(np.float32)

    band = cv2.dilate(gt, k, iterations=r_outer) - cv2.dilate(gt, k, iterations=r_inner)
    by, bx = np.nonzero(band)
    neg = np.zeros((0, 2), np.float32)
    if len(by) >= n_neg:
        sel = rng.sample(range(len(by)), n_neg)
        neg = np.stack([bx[sel], by[sel]], axis=1).astype(np.float32)
    return pos, neg


# ---------------------------------------------------------------------------
# Prompt hygiene
# ---------------------------------------------------------------------------

def sanitize_prompts(points, H, W, mode="drop", name="prompts", verbose=False):
    """Make a predicted point set safe to hand to SAM.

    Diagnostics showed the AdaFoB head emitting 85 of 120 background points
    OUTSIDE the image. SAM silently rescales whatever it is given, so such points
    become meaningless or actively harmful constraints instead of raising.

    mode='drop'  discard out-of-bounds points (recommended: an out-of-frame
                 background prompt carries no anatomical meaning)
    mode='clip'  clamp to the border (kept only for ablation; border points are
                 themselves poor background evidence)

    Also removes NaN/Inf and exact duplicates at pixel resolution, since repeated
    identical prompts waste the budget the whole method is about.
    Returns (clean_points, stats).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n_in = len(pts)
    stats = {"n_in": n_in, "n_nonfinite": 0, "n_oob": 0, "n_dup": 0, "n_out": 0}
    if n_in == 0:
        return pts, stats

    finite = np.isfinite(pts).all(axis=1)
    stats["n_nonfinite"] = int((~finite).sum())
    pts = pts[finite]

    oob = (pts[:, 0] < 0) | (pts[:, 0] > W - 1) | (pts[:, 1] < 0) | (pts[:, 1] > H - 1)
    stats["n_oob"] = int(oob.sum())
    if mode == "drop":
        pts = pts[~oob]
    else:
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    if len(pts):
        _, keep = np.unique(np.round(pts).astype(np.int32), axis=0, return_index=True)
        stats["n_dup"] = int(len(pts) - len(keep))
        pts = pts[np.sort(keep)]

    stats["n_out"] = int(len(pts))
    if verbose and (stats["n_oob"] or stats["n_dup"] or stats["n_nonfinite"]):
        print(f"    [{name}] {n_in} -> {stats['n_out']}  "
              f"(oob={stats['n_oob']} dup={stats['n_dup']} "
              f"nonfinite={stats['n_nonfinite']})")
    return pts, stats


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------

TRANSFORMS = {
    "identity":         lambda m: m,
    "flip_y":           lambda m: m[::-1, :],
    "flip_x":           lambda m: m[:, ::-1],
    "rot180":           lambda m: m[::-1, ::-1],
    "transpose":        lambda m: m.T,
    "transpose+flip_y": lambda m: m.T[::-1, :],
    "transpose+flip_x": lambda m: m.T[:, ::-1],
    "rot90":            lambda m: np.rot90(m, 1),
    "rot270":           lambda m: np.rot90(m, 3),
}


def grad_map(img_slice):
    img = np.asarray(img_slice, dtype=np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def contour_grad(grad, mask):
    m = np.asarray(mask).astype(np.uint8)
    if m.sum() < 10:
        return None
    k = np.ones((3, 3), np.uint8)
    contour = cv2.dilate(m, k, 1) - cv2.erode(m, k, 1)
    if contour.sum() < 10:
        return None
    return float(grad[contour.astype(bool)].mean())


def edge_z(grad, mask, body=None, n_null=40, rng=None, min_shift=8):
    """Alignment metric: does this outline sit on a real boundary HERE?

    Compares the gradient on the true mask contour with a null built by
    translating the SAME mask to other placements inside the body, scored with
    median/MAD.  Self-calibrating, so it transfers across datasets where a raw
    gradient threshold would not.  Aligned > 3, misaligned ~ 0.

    Two details matter: candidates must stay inside the body, or the null is
    dominated by the skin/air edge; and median/MAD, so one bone crossing cannot
    inflate the null.
    """
    rng = rng or np.random.default_rng(0)
    true = contour_grad(grad, mask)
    if true is None:
        return None

    m = np.asarray(mask).astype(bool)
    H, W = m.shape
    area = int(m.sum())
    null, tries = [], 0
    while len(null) < n_null and tries < n_null * 8:
        tries += 1
        dy = int(rng.integers(-H // 3, H // 3 + 1))
        dx = int(rng.integers(-W // 3, W // 3 + 1))
        if abs(dy) < min_shift and abs(dx) < min_shift:
            continue
        shifted = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        if body is not None and (shifted & body).sum() < 0.95 * area:
            continue
        g = contour_grad(grad, shifted)
        if g is not None:
            null.append(g)

    if len(null) < 8:
        return None
    med = float(np.median(null))
    mad = float(np.median(np.abs(np.array(null) - med))) * 1.4826
    if mad < 1e-6:
        return None
    return float((true - med) / mad)


def apply_label_transform(label_vol, transform="identity", z_shift=0):
    """Apply a detected correction to a label volume, in place of guessing."""
    lbl = np.asarray(label_vol)
    tf = TRANSFORMS[transform]
    out = np.stack([tf(lbl[z]) for z in range(lbl.shape[0])], axis=0)
    if z_shift:
        out = np.roll(out, -z_shift, axis=0)
    return np.ascontiguousarray(out)


def detect_alignment(volumes, organ, min_pixels=50, max_slices=60,
                     body_thresh=20.0, seed=0, verbose=True):
    """Choose the label transform that makes labels agree with images.

    `volumes` is a list of dicts with 'canon' (Z,H,W float [0,255]) and
    'label' (Z,H,W int).  Returns a dict with the winning transform, z-shift,
    the per-transform scores, and whether the result is trustworthy.

    This is deliberately measured rather than configured: a hard-coded flip is
    a guess that silently corrupts every downstream number if it is wrong.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for vi, v in enumerate(volumes):
        for z in range(v["label"].shape[0]):
            m = (v["label"][z] == organ)
            if m.sum() > min_pixels:
                samples.append((vi, z))
            if len(samples) >= max_slices:
                break
        if len(samples) >= max_slices:
            break
    if not samples:
        return {"transform": "identity", "z_shift": 0, "scores": {},
                "confident": False, "reason": "no annotated slices found"}

    grads, bodies = {}, {}
    for vi, z in samples:
        if (vi, z) not in grads:
            grads[(vi, z)] = grad_map(volumes[vi]["canon"][z])
            bodies[(vi, z)] = volumes[vi]["canon"][z] > body_thresh

    scores = {}
    for tname, tf in TRANSFORMS.items():
        acc = []
        for vi, z in samples:
            m = (volumes[vi]["label"][z] == organ)
            mt = tf(m)
            if mt.shape != m.shape:
                continue
            e = edge_z(grads[(vi, z)], mt, body=bodies[(vi, z)], rng=rng)
            if e is not None:
                acc.append(e)
        if acc:
            scores[tname] = float(np.mean(acc))
    if not scores:
        return {"transform": "identity", "z_shift": 0, "scores": {},
                "confident": False, "reason": "metric undefined on all slices"}

    best = max(scores, key=scores.get)
    id_score = scores.get("identity", 0.0)

    # Z-shift, evaluated under the winning in-plane transform
    tf = TRANSFORMS[best]
    z_scores = {}
    for k in range(-4, 5):
        acc = []
        for vi, z in samples:
            Z = volumes[vi]["label"].shape[0]
            zz = z + k
            if zz < 0 or zz >= Z:
                continue
            m = (volumes[vi]["label"][z] == organ)
            mt = tf(m)
            if mt.shape != m.shape:
                continue
            g = grads.get((vi, zz)) if (vi, zz) in grads else grad_map(volumes[vi]["canon"][zz])
            b = volumes[vi]["canon"][zz] > body_thresh
            e = edge_z(g, mt, body=b, rng=rng)
            if e is not None:
                acc.append(e)
        if acc:
            z_scores[k] = float(np.mean(acc))
    best_k = max(z_scores, key=z_scores.get) if z_scores else 0
    if z_scores and z_scores[best_k] <= z_scores.get(0, -1e9) + 0.5:
        best_k = 0                                  # not a convincing improvement

    chosen = best if scores[best] > id_score + 1.0 else "identity"
    confident = max(scores[chosen], scores.get("identity", 0.0)) >= 3.0

    if verbose:
        print("  alignment detection (edge-z, higher is better):")
        for t in sorted(scores, key=scores.get, reverse=True):
            mark = "  <-- chosen" if t == chosen else ""
            print(f"    {t:18s} {scores[t]:6.2f}{mark}")
        if best_k:
            print(f"    z-shift {best_k:+d} (score {z_scores[best_k]:.2f} "
                  f"vs {z_scores.get(0, float('nan')):.2f} at 0)")
        if not confident:
            print("    !! NO transform reaches edge-z >= 3.0: the labels do not follow")
            print("       any boundary in these images under any flip. Suspect wrong")
            print("       image/label PAIRING or averaged/template images, not orientation.")

    return {"transform": chosen, "z_shift": int(best_k), "scores": scores,
            "z_scores": z_scores, "confident": bool(confident),
            "reason": "ok" if confident else "low edge-z under all transforms"}
