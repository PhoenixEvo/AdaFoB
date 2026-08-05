"""Canonical preprocessing shared by train.py, eval.py and diagnose.py.

Having ONE implementation is the point: the whole Dice collapse came from
train.py and eval.py disagreeing about what an input image looks like.
Import from here; never re-implement normalisation in a script.
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
    if vol.shape[1:] == (size[1], size[0]):
        return vol
    out = np.zeros((vol.shape[0], size[1], size[0]), dtype=vol.dtype)
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    for z in range(vol.shape[0]):
        out[z] = cv2.resize(vol[z], size, interpolation=interp)
    return out


# ---------------------------------------------------------------------------
# Model-facing normalisation
# ---------------------------------------------------------------------------

def norm_zscore(vol_canon, mean, std):
    """FoB-style input: z-score of the canonical volume -> (Z, 3, H, W).

    This is what the published FoB checkpoints expect.  Any model initialised
    from those weights MUST use this, not the [0,1] transform below.
    """
    img = (np.asarray(vol_canon, dtype=np.float64) - mean) / (std + 1e-8)
    return np.stack(3 * [img], axis=1).astype(np.float32)


def norm_unit_legacy(vol_canon):
    """LEGACY [0,1] transform from the original train.py `_load_slice`:

        per-slice z-score -> clip(z * 50 + 128, 0, 255) -> /255

    Kept only so old checkpoints remain reproducible.  Do not use for new runs:
    it is incompatible with FoB-pretrained weights.
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

    Same min-max as FoB's SAM.pre_process, but on a *windowed* slice so
    soft-tissue contrast survives.  Note min-max is invariant to any affine
    transform, so the z-score never changes this -- only the window does.
    """
    s = np.asarray(slice_canon, dtype=np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-8:
        u8 = np.zeros_like(s, dtype=np.uint8)
    else:
        u8 = ((s - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack(3 * [u8], axis=-1)


# ---------------------------------------------------------------------------
# Oracle prompt construction (used by the sanity check and the diagnostics)
# ---------------------------------------------------------------------------

def oracle_prompts(gt, n_pos=10, n_neg=10, r_outer=15, r_inner=13, erode_iters=3, rng=None):
    """Ground-truth-derived prompts, in SAM's (x, y) order.

    Positives: sampled from the eroded morphological core (the centroid of a
    crescent-shaped organ can fall in the background).
    Negatives: FoB's differential dilation band, rho(M, r_outer) - rho(M, r_inner).
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
    if len(cy) >= n_pos:
        sel = rng.sample(range(len(cy)), n_pos)
    else:
        sel = list(range(len(cy)))
    pos = np.stack([cx[sel], cy[sel]], axis=1).astype(np.float32)

    band = cv2.dilate(gt, k, iterations=r_outer) - cv2.dilate(gt, k, iterations=r_inner)
    by, bx = np.nonzero(band)
    if len(by) >= n_neg:
        sel = rng.sample(range(len(by)), n_neg)
        neg = np.stack([bx[sel], by[sel]], axis=1).astype(np.float32)
    else:
        neg = np.zeros((0, 2), np.float32)

    return pos, neg
