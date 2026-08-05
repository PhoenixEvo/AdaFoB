"""Validate the alignment detector on synthetic data with a KNOWN misalignment.

CPU only. Run from the repo root:  python tests/test_alignment_detector.py
"""
import os, sys, types, importlib.util
import numpy as np

sys.modules.setdefault("SimpleITK", types.ModuleType("SimpleITK"))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
spec = importlib.util.spec_from_file_location(
    "ca", os.path.join(_ROOT, "experiments", "check_alignment.py"))
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

rng = np.random.default_rng(0)


def make_slice():
    """Synthetic abdominal slice in canonical [0,255]: fat rim, soft-tissue
    interior, air outside, a bone structure, and a spleen blob on the right."""
    img = np.zeros((256, 256), np.float32)                       # air -> 0
    yy, xx = np.mgrid[0:256, 0:256]
    body = ((yy - 128) ** 2 / 100 ** 2 + (xx - 128) ** 2 / 110 ** 2) < 1
    img[body] = ca.hu_to_canon(-90) + rng.normal(0, 2, body.sum())    # fat
    inner = ((yy - 128) ** 2 / 88 ** 2 + (xx - 128) ** 2 / 96 ** 2) < 1
    img[inner] = ca.hu_to_canon(45) + rng.normal(0, 3, inner.sum())   # soft tissue
    spleen = ((yy - 100) ** 2 / 26 ** 2 + (xx - 185) ** 2 / 18 ** 2) < 1
    img[spleen] = ca.hu_to_canon(55) + rng.normal(0, 3, spleen.sum())
    bone = ((yy - 205) ** 2 / 16 ** 2 + (xx - 128) ** 2 / 30 ** 2) < 1
    img[bone] = 255.0
    return img, spleen.astype(np.uint8)


img, gt = make_slice()
_G = ca.grad_map(img)
_body = img > 20.0

print("=" * 70)
print("TEST A - aligned label scores high, flipped label scores low")
print("=" * 70)
for name, tf in [("identity", lambda m: m), ("flip_y", lambda m: m[::-1, :]),
                 ("flip_x", lambda m: m[:, ::-1]), ("rot180", lambda m: m[::-1, ::-1])]:
    z = ca.edge_z(_G, tf(gt), body=_body)
    e = ca.edge_score(img, tf(gt))
    c = ca.contrast_score(img, tf(gt))
    t = ca.tissue_score(img, tf(gt))
    print(f"  label {name:10s}: EDGE-Z={z:7.2f}  edge/ref={e:.3f}  "
          f"contrast={c:.3f}  soft-frac={t:.3f}")
print("\n  NOTE soft-frac is 1.000 for EVERY transform: the abdomen is soft tissue")
print("  almost everywhere, so an intensity test cannot detect a flip. That is")
print("  why EDGE-Z (does the outline sit on a real boundary?) is the decision")
print("  metric, calibrated against translations of the same mask inside the body.")
assert ca.edge_z(_G, gt, body=_body) > 3.0, "aligned label must have high edge-z"
assert ca.edge_z(_G, gt[::-1, :], body=_body) < 2.0, "flipped label must have low edge-z"

print()
print("=" * 70)
print("TEST B - sweep recovers the induced transform")
print("=" * 70)
for induced_name, induced in [("flip_y", lambda m: m[::-1, :]),
                              ("flip_x", lambda m: m[:, ::-1]),
                              ("rot180", lambda m: m[::-1, ::-1]),
                              ("identity", lambda m: m)]:
    stored = induced(gt)                        # what is on disk
    scores = {}
    for tname, tf in ca.TRANSFORMS.items():
        m = tf(stored)
        if m.shape != img.shape:
            continue
        s = ca.edge_z(_G, m, body=_body)
        if s is not None:
            scores[tname] = s
    best = max(scores, key=scores.get)
    ok = abs(scores[best] - ca.edge_z(_G, gt, body=_body)) < 0.5
    print(f"  induced {induced_name:9s} -> sweep picks {best:18s} "
          f"edge-z={scores[best]:.2f}  identity={scores['identity']:.2f}  "
          f"{'RECOVERED' if ok else 'FAILED'}")
    assert ok, f"sweep failed to recover {induced_name}"

print()
print("=" * 70)
print("TEST C - separation margin")
print("=" * 70)
aligned = ca.edge_z(_G, gt, body=_body)
mis = [ca.edge_z(_G, t(gt), body=_body) for t in
       [lambda m: m[::-1, :], lambda m: m[:, ::-1], lambda m: m[::-1, ::-1]]]
print(f"  aligned EDGE-Z    : {aligned:.2f}")
print(f"  misaligned EDGE-Z : min={min(mis):.2f} max={max(mis):.2f}")
print(f"  separation        : {aligned - max(mis):.2f} sigma")
print("  script thresholds : aligned edge-z > 3.0, (best - identity) > 1.0")

print("\nALL ALIGNMENT-DETECTOR TESTS PASSED")
