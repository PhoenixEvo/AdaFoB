"""Tests for prompt hygiene and automatic alignment detection (CPU only)."""
import os
import sys
import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
spec = importlib.util.spec_from_file_location(
    "pp", os.path.join(_ROOT, "data", "preprocess.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
import numpy as np

rng = np.random.default_rng(0)


def make_volume(z=12, flip=None, organ=1):
    """Synthetic abdominal volume + label, optionally with a misaligned label."""
    canon = np.zeros((z, 256, 256), np.float32)
    label = np.zeros((z, 256, 256), np.int32)
    yy, xx = np.mgrid[0:256, 0:256]
    for k in range(z):
        img = np.zeros((256, 256), np.float32)
        body = ((yy - 128) ** 2 / 100 ** 2 + (xx - 128) ** 2 / 110 ** 2) < 1
        img[body] = pp_hu(-90) + rng.normal(0, 2, body.sum())
        inner = ((yy - 128) ** 2 / 88 ** 2 + (xx - 128) ** 2 / 96 ** 2) < 1
        img[inner] = pp_hu(45) + rng.normal(0, 3, inner.sum())
        ry = 24 - abs(k - z // 2)                      # organ shrinks toward the ends
        spleen = ((yy - 100) ** 2 / max(ry, 6) ** 2 + (xx - 185) ** 2 / 18 ** 2) < 1
        img[spleen] = pp_hu(58) + rng.normal(0, 3, spleen.sum())
        bone = ((yy - 205) ** 2 / 16 ** 2 + (xx - 128) ** 2 / 30 ** 2) < 1
        img[bone] = 255.0
        canon[k] = img
        m = spleen.astype(np.int32)
        label[k] = (pp.TRANSFORMS[flip](m) if flip else m) * organ
    return {"canon": canon, "label": label}


def pp_hu(hu, window=(-125.0, 275.0)):
    lo, hi = window
    return (hu - lo) / (hi - lo) * 255.0


print("=" * 72)
print("TEST 1 - sanitize_prompts (the 85/120 out-of-bounds negatives bug)")
print("=" * 72)
pts = np.array([[10, 20], [10, 20], [300, 50], [-5, 10], [np.nan, 3],
                [255, 255], [128.4, 128.6]], np.float32)
clean, st = pp.sanitize_prompts(pts, 256, 256, mode="drop", name="neg", verbose=True)
print(f"  in={st['n_in']} out={st['n_out']} oob={st['n_oob']} dup={st['n_dup']} "
      f"nonfinite={st['n_nonfinite']}")
print(f"  kept:\n{clean}")
assert st["n_oob"] == 2 and st["n_dup"] == 1 and st["n_nonfinite"] == 1
assert st["n_out"] == 3
assert clean.min() >= 0 and clean[:, 0].max() <= 255 and clean[:, 1].max() <= 255
clipped, st2 = pp.sanitize_prompts(pts, 256, 256, mode="clip")
assert st2["n_out"] == 5, st2
print(f"  clip mode keeps {st2['n_out']} (border-clamped instead of dropped)")

print()
print("=" * 72)
print("TEST 2 - detect_alignment recovers an induced label flip")
print("=" * 72)
for induced in [None, "flip_y", "flip_x", "rot180"]:
    vols = [make_volume(flip=induced) for _ in range(2)]
    res = pp.detect_alignment(vols, organ=1, verbose=False)
    fixed = pp.apply_label_transform(vols[0]["label"], res["transform"], res["z_shift"])
    # after correction the label must sit on the organ again
    g = pp.grad_map(vols[0]["canon"][6])
    body = vols[0]["canon"][6] > 20.0
    z_before = pp.edge_z(g, vols[0]["label"][6] == 1, body=body)
    z_after = pp.edge_z(g, fixed[6] == 1, body=body)
    ok = z_after > 3.0
    print(f"  induced={str(induced):8s} -> detected={res['transform']:10s} "
          f"z_shift={res['z_shift']:+d} confident={res['confident']}  "
          f"edge-z {z_before:6.2f} -> {z_after:6.2f}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"failed to correct {induced}"

print()
print("=" * 72)
print("TEST 3 - detect_alignment refuses to guess when nothing fits")
print("=" * 72)
noise = [{"canon": rng.normal(120, 5, (8, 256, 256)).astype(np.float32),
          "label": np.zeros((8, 256, 256), np.int32)}]
noise[0]["label"][:, 100:130, 100:130] = 1          # a square on featureless noise
res = pp.detect_alignment(noise, organ=1, verbose=False)
print(f"  detected={res['transform']} confident={res['confident']} reason={res['reason']}")
print("  -> a low-confidence result must NOT be applied silently; eval.py aborts.")
assert res["confident"] is False

print()
print("=" * 72)
print("TEST 4 - dtype-safe label resize")
print("=" * 72)
lbl = np.zeros((3, 300, 300), np.int32); lbl[:, 100:200, 100:200] = 6
out = pp.resize_volume(lbl, (256, 256), is_label=True)
print(f"  int32 (3,300,300) -> {out.shape} dtype={out.dtype} "
      f"labels={np.unique(out)}")
assert out.shape == (3, 256, 256) and set(np.unique(out)) == {0, 6}
img = np.zeros((3, 300, 300), np.float32)
outi = pp.resize_volume(img, (256, 256))
assert outi.shape == (3, 256, 256)
print("  float32 image resize OK")

print("\nALL TESTS PASSED")
