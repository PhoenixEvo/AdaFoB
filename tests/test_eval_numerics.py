"""Unit-test the numeric core of eval_new.py with torch/FoB/SAM stubbed out."""
import sys, types, importlib.util
import numpy as np

# ---- stub heavy deps so the module imports in a CPU-only sandbox ----
torch = types.ModuleType("torch")
torch.is_tensor = lambda x: False
torch.from_numpy = lambda x: x
torch.manual_seed = lambda s: None
torch.load = lambda *a, **k: {}
torch.no_grad = lambda: __import__("contextlib").nullcontext()
sys.modules["torch"] = torch
for name in ["models", "models.FoB", "segment_anything", "SimpleITK"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["models.FoB"].FewShotSeg = object
sys.modules["segment_anything"].sam_model_registry = {}
sys.modules["segment_anything"].SamPredictor = object

spec = importlib.util.spec_from_file_location("ev", "/home/user/work/eval_new.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

print("=" * 68)
print("TEST 1 - windowing rescues soft-tissue contrast for SAM")
print("=" * 68)
rng = np.random.default_rng(0)
# synthetic abdominal CT slice: air -1000, fat -90, soft tissue 40, spleen 55, bone 1000
sl = np.full((256, 256), -1000.0)
sl[40:220, 40:220] = 40.0 + rng.normal(0, 3, (180, 180))     # soft tissue
sl[90:150, 90:150] = 55.0 + rng.normal(0, 3, (60, 60))       # spleen
sl[200:215, 60:200] = 1000.0                                  # vertebra / bone
vol = sl[None]

# OLD pipeline: z-score then min-max to uint8, no windowing
z = (vol - vol.mean()) / (vol.std() + 1e-8)
old = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)[0]
# NEW pipeline: HU window -> canonical -> min-max
canon, domain = ev.to_canonical(vol, (-125.0, 275.0))
new = ev.sam_uint8_from_canonical(canon[0])[..., 0]

def contrast(img):
    spleen = img[95:145, 95:145].astype(float).mean()
    tissue = img[50:80, 50:80].astype(float).mean()
    return abs(spleen - tissue)

print(f"detected domain          : {domain}")
print(f"OLD spleen-vs-tissue gap : {contrast(old):6.2f} grey levels")
print(f"NEW spleen-vs-tissue gap : {contrast(new):6.2f} grey levels")
print(f"OLD dynamic range used   : {old.max()-old.min():3d} / 255")
print(f"NEW dynamic range used   : {new.max()-new.min():3d} / 255")
assert contrast(new) > contrast(old) * 3, "windowing should sharply raise contrast"

print()
print("=" * 68)
print("TEST 2 - min-max is invariant to any affine z-score (their diagnosis)")
print("=" * 68)
a = ev.sam_uint8_from_canonical(canon[0])
shifted = (canon[0] - 37.0) / 4.5          # arbitrary affine, i.e. any z-score
b = ev.sam_uint8_from_canonical(shifted)
print(f"identical uint8 images   : {np.array_equal(a, b)}")
assert np.array_equal(a, b)
print("=> z-score choice cannot change SAM's input; only windowing can.")

print()
print("=" * 68)
print("TEST 3 - per-model normalisation ranges match training")
print("=" * 68)
fob_in = ev.norm_fob(canon, 94.0, 62.0)
ada_in = ev.norm_adafob_trainstyle(canon)
print(f"FoB    input: shape={fob_in.shape} range=[{fob_in.min():.3f}, {fob_in.max():.3f}] "
      f"mean={fob_in.mean():.3f}")
print(f"AdaFoB input: shape={ada_in.shape} range=[{ada_in.min():.3f}, {ada_in.max():.3f}] "
      f"mean={ada_in.mean():.3f}  (train.py produced [0,1])")
assert ada_in.min() >= 0.0 and ada_in.max() <= 1.0, "AdaFoB input must be [0,1]"
assert fob_in.shape[1] == 3 and ada_in.shape[1] == 3

print()
print("=" * 68)
print("TEST 4 - HD95 uses surface points (old version allocated a huge matrix)")
print("=" * 68)
gt = np.zeros((256, 256), np.uint8); gt[60:200, 60:200] = 1
pred = np.zeros((256, 256), np.uint8); pred[65:205, 65:205] = 1
n_fg = int(gt.sum())
n_surf = len(ev._surface_points(gt))
print(f"foreground voxels        : {n_fg}")
print(f"surface voxels           : {n_surf}")
print(f"old dense matrix entries : {n_fg*int(pred.sum()):,} (~{n_fg*int(pred.sum())*8/1e9:.1f} GB)")
print(f"new dense matrix entries : {n_surf*len(ev._surface_points(pred)):,}")
print(f"HD95 (5px shift)         : {ev.compute_hd95(pred, gt):.3f}")
print(f"HD95 identical masks     : {ev.compute_hd95(gt, gt):.3f}")
print(f"HD95 empty prediction    : {ev.compute_hd95(np.zeros_like(gt), gt):.1f} (sentinel)")
assert abs(ev.compute_hd95(gt, gt)) < 1e-6
print(f"Dice identical masks     : {ev.compute_dice(gt, gt):.3f}")

print()
print("=" * 68)
print("TEST 5 - domain auto-detection")
print("=" * 68)
for name, v in [("raw HU", vol),
                ("[0,1] floats", np.clip((vol + 1000) / 2000, 0, 1)),
                ("[0,255] bytes", np.clip((vol + 1000) / 2000 * 255, 0, 255))]:
    c, d = ev.to_canonical(v, (-125.0, 275.0))
    print(f"  {name:15s} -> domain={d:12s} canonical range=[{c.min():.1f}, {c.max():.1f}]")

print("\nALL TESTS PASSED")
