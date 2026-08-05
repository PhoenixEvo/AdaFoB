"""End-to-end check that the alignment fix is the causal lever on oracle Dice.

Runs without SAM or a GPU. A deterministic stand-in segmenter (intensity region
growing seeded at the positive prompts, blocked by the negative prompts) plays
SAM's role: it is prompt-driven and image-driven, so it reproduces the failure
mode under test -- if the mask does not correspond to the image at the prompt
coordinates, the segmentation cannot match the mask.

Expectation:
  misaligned labels -> oracle Dice collapses to roughly the 0.5 we measured
  after detect_alignment + apply_label_transform -> oracle Dice recovers > 0.85
"""
import os
import sys
import importlib.util
import numpy as np
from scipy import ndimage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
spec = importlib.util.spec_from_file_location(
    "pp", os.path.join(_ROOT, "data", "preprocess.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

rng = np.random.default_rng(7)


def hu(v, window=(-125.0, 275.0)):
    lo, hi = window
    return (v - lo) / (hi - lo) * 255.0


def make_volume(z=10, flip=None):
    canon = np.zeros((z, 256, 256), np.float32)
    label = np.zeros((z, 256, 256), np.int32)
    yy, xx = np.mgrid[0:256, 0:256]
    for k in range(z):
        img = np.zeros((256, 256), np.float32)
        body = ((yy - 128) ** 2 / 100 ** 2 + (xx - 128) ** 2 / 110 ** 2) < 1
        img[body] = hu(-90) + rng.normal(0, 1.5, body.sum())
        inner = ((yy - 128) ** 2 / 88 ** 2 + (xx - 128) ** 2 / 96 ** 2) < 1
        img[inner] = hu(45) + rng.normal(0, 1.5, inner.sum())
        ry = 26 - abs(k - z // 2)
        organ = ((yy - 100) ** 2 / max(ry, 8) ** 2 + (xx - 186) ** 2 / 20 ** 2) < 1
        img[organ] = hu(70) + rng.normal(0, 1.5, organ.sum())
        img[((yy - 205) ** 2 / 16 ** 2 + (xx - 128) ** 2 / 30 ** 2) < 1] = 255.0
        canon[k] = img
        m = organ.astype(np.int32)
        label[k] = pp.TRANSFORMS[flip](m) if flip else m
    return {"canon": canon, "label": label}


def fake_segment(img, pos, neg, tol=5.0):
    """Prompt-driven stand-in for SAM: grow intensity-similar regions from the
    positive seeds, discard any component containing a negative prompt."""
    H, W = img.shape
    if len(pos) == 0:
        return np.zeros((H, W), np.uint8)
    py = np.clip(np.round(pos[:, 1]).astype(int), 0, H - 1)
    px = np.clip(np.round(pos[:, 0]).astype(int), 0, W - 1)
    seed_val = float(np.median(img[py, px]))
    band = (np.abs(img - seed_val) < tol)
    lab, n = ndimage.label(band)
    keep = set(lab[py, px].tolist()) - {0}
    if len(neg):
        ny = np.clip(np.round(neg[:, 1]).astype(int), 0, H - 1)
        nx = np.clip(np.round(neg[:, 0]).astype(int), 0, W - 1)
        keep -= set(lab[ny, nx].tolist())
    return np.isin(lab, list(keep)).astype(np.uint8) if keep else np.zeros((H, W), np.uint8)


def dice(a, b):
    a, b = (a > 0), (b > 0)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def oracle_dice(vols, organ=1, n_pos=10, n_neg=10):
    ds = []
    for v in vols:
        for z in range(v["label"].shape[0]):
            gt = (v["label"][z] == organ).astype(np.uint8)
            if gt.sum() < 50:
                continue
            pos, neg = oracle_prompts_np(gt, n_pos, n_neg)
            if len(pos) == 0:
                continue
            ds.append(dice(fake_segment(v["canon"][z], pos, neg), gt))
    return float(np.mean(ds)) if ds else 0.0


def oracle_prompts_np(gt, n_pos, n_neg):
    import random
    return pp.oracle_prompts(gt, n_pos=n_pos, n_neg=n_neg, rng=random.Random(0))


print("=" * 74)
print("Oracle Dice with a prompt-driven stand-in segmenter")
print("=" * 74)
print(f"  {'label state':34s} {'oracle Dice':>12s} {'gate (>=0.85)':>15s}")

aligned = [make_volume(flip=None) for _ in range(2)]
d_ok = oracle_dice(aligned)
print(f"  {'correctly aligned':34s} {d_ok:12.4f} {'PASS' if d_ok >= 0.85 else 'FAIL':>15s}")
assert d_ok >= 0.85, d_ok

for induced in ("flip_y", "rot180", "transpose"):
    bad = [make_volume(flip=induced) for _ in range(2)]
    d_bad = oracle_dice(bad)

    res = pp.detect_alignment(bad, organ=1, verbose=False)
    for v in bad:
        v["label"] = pp.apply_label_transform(v["label"], res["transform"], res["z_shift"])
    d_fix = oracle_dice(bad)

    print(f"  {'misaligned (' + induced + ')':34s} {d_bad:12.4f} "
          f"{'FAIL' if d_bad < 0.85 else 'PASS':>15s}")
    print(f"  {'  -> after auto_align (' + res['transform'] + ')':34s} {d_fix:12.4f} "
          f"{'PASS' if d_fix >= 0.85 else 'FAIL':>15s}")
    assert d_bad < 0.85, f"{induced} should have failed the gate"
    assert d_fix >= 0.85, f"{induced} not recovered: {d_fix}"

print("""
Interpretation
--------------
A misaligned label reproduces exactly the signature measured on the real data:
prompts that are inside GT by construction, yet a prompt-driven segmenter
recovers only part of the mask. detect_alignment + apply_label_transform restores
it above the gate WITHOUT touching either model.

This does not prove the real data has this defect -- only check_alignment.py on
the actual volumes can establish that. It proves the mechanism is sufficient to
cause a ~0.5 oracle, and that the committed fix removes it when it is the cause.
""")
print("ALL INTEGRATION TESTS PASSED")
