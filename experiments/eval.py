"""
Phase 4 Evaluation: AdaFoB-GAP vs FoB-pretrained on Abd-CT.

FIXES vs the previous version
-----------------------------
1. CANONICAL INTENSITY DOMAIN.  Volumes are first mapped into a single canonical
   [0, 255] domain that mimics Ouyang et al.'s `sabs_CT_normalized` output:
       raw HU  -> clip to --hu_window (default [-125, 275]) -> rescale to [0,255]
       [0,1]   -> x255
       [0,255] -> unchanged
   Without the HU window, min-max to uint8 over a slice that contains bone
   (+1000 HU) and air (-1000 HU) compresses all abdominal soft tissue into a
   ~10-grey-level band, i.e. SAM literally sees a flat grey image.  That, not
   the z-score, is what makes the spleen invisible to SAM.

2. PER-MODEL NORMALISATION.  The two networks were trained on different input
   distributions, so they each get the input distribution they were trained on:
       baseline FoB : z-score of the canonical volume.  Statistics come from
                      --baseline_norm  (dataset | volume | fixed).  `dataset`
                      (default) pools statistics over every volume, which
                      reconstructs full-volume SABS statistics far better than
                      per-volume statistics computed on a 12-slice crop.
       AdaFoB       : the exact transform used in experiments/train.py
                      `_load_slice`, i.e. per-slice z-score -> clip(z*50+128,
                      0, 255) -> /255, giving a [0,1] input.  Feeding raw
                      z-scores to a model trained on [0,1] inputs is the reason
                      AdaFoB scored ~0.00.

3. SAM SEES THE SAME IMAGE FOR BOTH MODELS.  SAM's uint8 image is built from the
   canonical volume, not from either model's normalised tensor.  Note that
   min-max is invariant to any affine transform, so a z-score can never change
   SAM's input anyway -- only the windowing does.  Decoupling makes the
   comparison isolate prompt quality, which is the actual claim under test.

4. CHECKPOINT LOADING IS VERIFIED.  `strict=False` silently tolerated a
   completely unloaded model (a `{'state_dict': ...}` wrapper or a `module.`
   prefix loads *nothing* and yields ~0.00 Dice).  Checkpoints are now
   unwrapped, de-prefixed, and the matched-parameter fraction is asserted.

5. SANITY CHECK (--sanity_check).  Runs SAM with ground-truth-derived prompts.
   If that oracle Dice is low the fault is in the image pipeline, not in the
   models -- run this before trusting any comparison.

6. SAM's image encoder runs ONCE per query slice instead of twice (~2x faster).

7. HD95 uses surface points, not all foreground voxels.  The old version built
   an N x M dense distance matrix that reaches tens of GB on large masks.

Pipeline details preserved from FoB's original test.py: SimpleITK loading
(Z, H, W), 3-channel replication via np.stack(3 * [img], axis=1), masks[0] for
non-ISIC, deep-copied inputs before each model call, middle-slice support.
"""

import os
import sys
import glob
import copy
import csv
import json
import random
import argparse
import urllib.request
import zipfile

import numpy as np
import torch
import cv2
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

# Add FoB_SAM to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "FoB_SAM")))
from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def _surface_points(mask, max_points=4000):
    """Boundary voxels of a binary mask (interior points cannot set HD95)."""
    mask = mask.astype(bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    pts = np.argwhere(mask & ~eroded)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def compute_hd95(pred, gt, empty_value=256.0, max_points=4000):
    """Symmetric 95th-percentile Hausdorff distance in pixels.

    Returns `empty_value` when either mask is empty (documented sentinel --
    report the empty-prediction rate alongside HD95 in the paper).
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0 or gt.sum() == 0:
        return float(empty_value)

    pred_pts = _surface_points(pred, max_points)
    gt_pts = _surface_points(gt, max_points)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(empty_value)

    d = cdist(pred_pts, gt_pts)
    all_min = np.concatenate([d.min(axis=1), d.min(axis=0)])
    return float(np.percentile(all_min, 95))


# ---------------------------------------------------------------------------
# Intensity handling
# ---------------------------------------------------------------------------

def to_canonical(vol, hu_window=(-125.0, 275.0)):
    """Map an arbitrary CT volume into the canonical [0, 255] domain.

    Mirrors Ouyang et al.'s SABS preprocessing, which is what the FoB
    checkpoints were trained on.  Returns (volume, detected_domain).
    """
    vol = vol.astype(np.float64)
    vmin, vmax = float(vol.min()), float(vol.max())

    if vmin < -20.0:                       # raw Hounsfield units
        lo, hi = float(hu_window[0]), float(hu_window[1])
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo) * 255.0
        domain = "raw_hu"
    elif vmax <= 1.5:                      # already scaled to [0, 1]
        vol = vol * 255.0
        domain = "unit"
    elif vmax <= 260.0:                    # already scaled to [0, 255]
        domain = "byte"
    else:                                  # unknown positive range
        lo, hi = np.percentile(vol, [0.5, 99.5])
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo + 1e-8) * 255.0
        domain = "percentile"

    return vol.astype(np.float32), domain


def norm_fob(vol_canon, mean, std):
    """Baseline FoB input: z-score of the canonical volume -> (Z, 3, H, W)."""
    img = (vol_canon.astype(np.float64) - mean) / (std + 1e-8)
    return np.stack(3 * [img], axis=1).astype(np.float32)


def norm_adafob_trainstyle(vol_canon):
    """AdaFoB input: replicates experiments/train.py `_load_slice` exactly.

        per-slice z-score -> clip(z * 50 + 128, 0, 255) -> /255  ->  [0, 1]
    """
    z_dim = vol_canon.shape[0]
    out = np.empty((z_dim, 3) + vol_canon.shape[1:], dtype=np.float32)
    for z in range(z_dim):
        s = vol_canon[z].astype(np.float64)
        s = (s - s.mean()) / (s.std() + 1e-8)
        s = np.clip(s * 50.0 + 128.0, 0, 255).astype(np.uint8)
        s = s.astype(np.float32) / 255.0
        out[z] = np.stack(3 * [s], axis=0)
    return out


def sam_uint8_from_canonical(slice_canon):
    """(H, W) canonical float -> (H, W, 3) uint8 for SAM.

    Same min-max as FoB's SAM.pre_process, but applied to a *windowed* slice so
    soft-tissue contrast survives.
    """
    s = slice_canon.astype(np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-8:
        u8 = np.zeros_like(s, dtype=np.uint8)
    else:
        u8 = ((s - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack(3 * [u8], axis=-1)


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def _resize_volume(vol, size=(256, 256), is_label=False):
    if vol.shape[1:] == (size[1], size[0]):
        return vol
    out = np.zeros((vol.shape[0], size[1], size[0]), dtype=vol.dtype)
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    for z in range(vol.shape[0]):
        out[z] = cv2.resize(vol[z], size, interpolation=interp)
    return out


def _collect_pairs(data_root):
    """Return list of (image_path, label_path)."""
    import re

    fob_dir = os.path.join(data_root, "sabs_CT_normalized")
    fob_imgs = sorted(glob.glob(os.path.join(fob_dir, "image_*.nii.gz"))) if os.path.isdir(fob_dir) else []
    if fob_imgs:
        pairs = []
        for ip in fob_imgs:
            lp = ip.replace("image_", "label_")
            if os.path.exists(lp):
                pairs.append((ip, lp))
        print(f"Found FoB preprocessed SABS layout: {len(pairs)} volume pairs")
        return pairs

    img_dict, lbl_dict = {}, {}
    for root, _, files in os.walk(data_root):
        for f in sorted(files):
            if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                continue
            path = os.path.join(root, f)
            fl = f.lower()
            match = re.search(r"(\d+)", f)
            if not match:
                continue
            pid = match.group(1)
            if "label" in fl or "seg" in fl:
                lbl_dict[pid] = path
            elif "image" in fl or "img" in fl or "avg" in fl:
                img_dict[pid] = path

    common = sorted(set(img_dict) & set(lbl_dict))
    print(f"Found raw layout: {len(img_dict)} images, {len(lbl_dict)} labels, {len(common)} paired")
    return [(img_dict[p], lbl_dict[p]) for p in common]


def load_volumes(data_root, hu_window=(-125.0, 275.0), limit=None):
    """Load volumes into the canonical domain.

    Returns (volumes, stats) where each volume is a dict:
        canon : (Z, 256, 256) float32 in [0, 255]
        label : (Z, 256, 256) int
    and stats holds the pooled dataset mean/std of the canonical intensities.
    """
    pairs = _collect_pairs(data_root)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"No volumes found in {data_root}")

    volumes = []
    total_sum = 0.0
    total_sqsum = 0.0
    total_n = 0
    domains = {}

    for ip, lp in pairs:
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip))     # (Z, H, W)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lp))
        if img.shape != lbl.shape:
            print(f"  WARNING: shape mismatch {os.path.basename(ip)} "
                  f"img {img.shape} vs lbl {lbl.shape}, skipping")
            continue

        raw_min, raw_max = float(img.min()), float(img.max())
        img = _resize_volume(img.astype(np.float64), (256, 256), is_label=False)
        lbl = _resize_volume(lbl.astype(np.int32), (256, 256), is_label=True)
        canon, domain = to_canonical(img, hu_window)
        domains[domain] = domains.get(domain, 0) + 1

        total_sum += float(canon.sum())
        total_sqsum += float((canon.astype(np.float64) ** 2).sum())
        total_n += canon.size

        volumes.append({
            "canon": canon,
            "label": lbl,
            "path": ip,
            "raw_range": (raw_min, raw_max),
            "domain": domain,
        })

    if not volumes:
        raise ValueError(f"No usable volume pairs in {data_root}")

    ds_mean = total_sum / total_n
    ds_std = float(np.sqrt(max(total_sqsum / total_n - ds_mean ** 2, 1e-12)))
    stats = {"dataset_mean": ds_mean, "dataset_std": ds_std, "domains": domains}

    print(f"\nIntensity domains detected: {domains}")
    print(f"Canonical dataset stats: mean={ds_mean:.3f}  std={ds_std:.3f}")
    for i, v in enumerate(volumes):
        ul = np.unique(v["label"])
        print(f"  Vol {i}: Z={v['canon'].shape[0]:3d}  raw[{v['raw_range'][0]:.0f},"
              f"{v['raw_range'][1]:.0f}] -> {v['domain']:10s}  labels={ul[:12]}")
    return volumes, stats


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------

def available_organs(volumes, organ_map, min_pixels=50, min_slices=2):
    """Organs that actually have enough annotated slices in this dataset."""
    ok = []
    for cls, name in organ_map.items():
        n_vols = 0
        for v in volumes:
            valid = ((v["label"] == cls).sum(axis=(1, 2)) > min_pixels).sum()
            if valid >= min_slices:
                n_vols += 1
        if n_vols >= 1:
            ok.append(cls)
            print(f"  organ {cls} ({name}): usable in {n_vols}/{len(volumes)} volumes")
        else:
            print(f"  organ {cls} ({name}): NOT present -- excluded from episodes")
    return ok


def sample_episode(volumes, organ_cls, min_pixels=50):
    """1-way 1-shot episode.  Support = middle slice of the valid sequence."""
    candidates = []
    for vi, v in enumerate(volumes):
        valid = np.where((v["label"] == organ_cls).sum(axis=(1, 2)) > min_pixels)[0]
        if len(valid) >= 2:
            candidates.append((vi, valid))
    if not candidates:
        return None

    if len(candidates) >= 2:
        si, qi = random.sample(range(len(candidates)), 2)
    else:
        si = qi = 0

    sv_idx, s_valid = candidates[si]
    qv_idx, q_valid = candidates[qi]

    # FoB TestDataset.get_support_index: 1-shot -> the middle slice (pct = 0.5)
    mid_idx = int(0.5 * len(s_valid))
    s_slices = [int(s_valid[mid_idx])]

    q_pool = [int(s) for s in q_valid if (sv_idx != qv_idx or int(s) not in s_slices)]
    if not q_pool:
        q_pool = [int(s) for s in q_valid]
    q_slice = random.choice(q_pool)

    return {
        "support_vol": sv_idx, "support_slices": s_slices,
        "query_vol": qv_idx, "query_slice": q_slice,
        "organ_cls": organ_cls,
    }


def build_inputs(volumes, ep, norm_fn):
    """Materialise FoB-format tensors for one episode under a given normaliser.

        supp_imgs  : way x shot x [1 x 3 x H x W]
        supp_masks : way x shot x [1 x H x W]
        qry_imgs   : N x [1 x 3 x H x W]
        qry_labels : 1 x H x W
    """
    sv = volumes[ep["support_vol"]]
    qv = volumes[ep["query_vol"]]
    cls = ep["organ_cls"]

    supp_imgs, supp_masks = [], []
    for s in ep["support_slices"]:
        img3 = norm_fn(sv["canon"][s:s + 1])                       # (1, 3, H, W)
        supp_imgs.append(torch.from_numpy(img3).float())
        m = (sv["label"][s] == cls).astype(np.float32)
        supp_masks.append(torch.from_numpy(m).unsqueeze(0).float())

    q = ep["query_slice"]
    qry_img3 = norm_fn(qv["canon"][q:q + 1])                       # (1, 3, H, W)
    qry_tensor = torch.from_numpy(qry_img3).float()
    qry_mask_np = (qv["label"][q] == cls).astype(np.int64)
    qry_label = torch.from_numpy(qry_mask_np).unsqueeze(0).long()

    return {
        "support_images":    [supp_imgs],
        "support_fg_labels": [supp_masks],
        "query_images":      [qry_tensor],
        "query_labels":      qry_label,
        "query_mask_np":     qry_mask_np,
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, name, min_match=0.5, strict=False):
    """Load a checkpoint, unwrapping common containers, and VERIFY it landed.

    `strict=False` alone silently produces a randomly-initialised network when
    the keys do not line up, which looks exactly like a model that scores 0.00.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"{name}: unsupported checkpoint object {type(obj)}")

    # strip DataParallel / compile prefixes
    cleaned = {}
    for k, v in obj.items():
        nk = k
        for pref in ("module.", "_orig_mod."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    model_keys = set(model.state_dict().keys())
    matched = [k for k in cleaned if k in model_keys]
    frac = len(matched) / max(len(model_keys), 1)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  {name}: matched {len(matched)}/{len(model_keys)} params ({frac:.1%}), "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected[:5]:
        print(f"    first unexpected keys: {list(unexpected)[:5]}")
    if missing[:5]:
        print(f"    first missing keys:    {list(missing)[:5]}")

    if frac < min_match:
        msg = (f"{name}: only {frac:.1%} of parameters were loaded from {path}. "
               f"The model is effectively random -- this is the classic cause of "
               f"~0.00 Dice. Inspect the checkpoint key names above.")
        if strict:
            raise RuntimeError(msg)
        print(f"  !! WARNING: {msg}")
    return frac


def run_model(model, sample, **fwd_kwargs):
    """Run FoB/AdaFoB; deep-copies inputs because FoB mutates lists in place."""
    supp_imgs = [[t.clone().cuda() for t in way] for way in sample["support_images"]]
    supp_masks = [[t.clone().cuda() for t in way] for way in sample["support_fg_labels"]]
    qry_imgs = [t.clone().cuda() for t in sample["query_images"]]
    qry_labels = sample["query_labels"].clone().cuda()

    with torch.no_grad():
        neg, pos = model(supp_imgs, supp_masks, qry_imgs, qry_labels, **fwd_kwargs)
    return neg, pos


def _as_points(p):
    if p is None:
        return np.zeros((0, 2), dtype=np.float32)
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float32)
    return p.reshape(-1, 2)


def predict_sam_from_points(predictor, pos_pts, neg_pts):
    """Predict with an image already registered via predictor.set_image()."""
    pos = _as_points(pos_pts)
    neg = _as_points(neg_pts)
    if len(pos) == 0 and len(neg) == 0:
        return None
    all_pts = np.concatenate([pos, neg], axis=0)
    all_lbls = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)

    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_lbls,
        multimask_output=True,
    )
    return masks[0]      # best_pred_idx = 0 for non-ISIC (FoB SAM.py)


# ---------------------------------------------------------------------------
# Sanity check: is the image pipeline itself usable?
# ---------------------------------------------------------------------------

def gt_prompt_sanity_check(predictor, volumes, organs, organ_map, n_cases=8, n_neg=10, seed=0):
    """Feed SAM ground-truth-derived prompts (centroid + FoB-style ring band).

    This removes both networks from the loop.  If Dice here is low, the fault is
    in loading/normalisation/coordinates -- fixing the models cannot help.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK - SAM with ground-truth prompts (no FoB, no AdaFoB)")
    print("=" * 60)
    rng = random.Random(seed)
    dices = []
    for i in range(n_cases):
        cls = organs[i % len(organs)]
        ep = sample_episode(volumes, cls)
        if ep is None:
            continue
        qv = volumes[ep["query_vol"]]
        q = ep["query_slice"]
        gt = (qv["label"][q] == cls).astype(np.uint8)
        if gt.sum() < 20:
            continue

        ys, xs = np.nonzero(gt)
        pos = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)   # (x, y)

        k = np.ones((3, 3), np.uint8)
        band = cv2.dilate(gt, k, iterations=15) - cv2.dilate(gt, k, iterations=13)
        by, bx = np.nonzero(band)
        if len(by) >= n_neg:
            sel = rng.sample(range(len(by)), n_neg)
            neg = np.stack([bx[sel], by[sel]], axis=1).astype(np.float32)
        else:
            neg = np.zeros((0, 2), dtype=np.float32)

        predictor.set_image(sam_uint8_from_canonical(qv["canon"][q]))
        pred = predict_sam_from_points(predictor, pos, neg)
        d = compute_dice(pred, gt) if pred is not None else 0.0
        dices.append(d)
        print(f"  case {i} ({organ_map[cls]:>8s}): Dice={d:.4f}  "
              f"gt_px={int(gt.sum())}  neg_pts={len(neg)}")

    if dices:
        m = float(np.mean(dices))
        print(f"\n  Oracle-prompt mean Dice: {m:.4f}")
        if m < 0.60:
            print("  !! The image pipeline is broken (windowing / resize / coordinate order).")
            print("     Fix this before interpreting any FoB vs AdaFoB comparison.")
        else:
            print("  OK - SAM can segment these images. Remaining error is prompt quality.")
    print("=" * 60 + "\n")
    return dices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    parser = argparse.ArgumentParser(description="Phase 4: AdaFoB vs FoB")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/adafob_abdct.pth")
    parser.add_argument("--sam_ckpt", type=str,
                        default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--limit_volumes", type=int, default=None)

    # --- normalisation policy -------------------------------------------------
    parser.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0],
                        help="HU clip window applied to raw-HU volumes before rescaling")
    parser.add_argument("--baseline_norm", choices=["dataset", "volume", "fixed"],
                        default="dataset",
                        help="z-score statistics for the SABS-trained baseline. "
                             "'dataset' pools all volumes (best proxy for full-volume "
                             "SABS stats on Z-cropped data); 'volume' is FoB's original "
                             "per-volume behaviour; 'fixed' uses --fixed_mean/--fixed_std")
    parser.add_argument("--fixed_mean", type=float, default=94.0)
    parser.add_argument("--fixed_std", type=float, default=62.0)
    parser.add_argument("--adafob_norm", choices=["train_slice", "dataset", "volume"],
                        default="train_slice",
                        help="'train_slice' replicates experiments/train.py _load_slice "
                             "(per-slice z-score -> clip(z*50+128,0,255) -> /255)")

    parser.add_argument("--organs", type=str, default="1,2,3,6")
    parser.add_argument("--min_pixels", type=int, default=50)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--strict_ckpt", action="store_true",
                        help="abort if a checkpoint fails to load most parameters")
    parser.add_argument("--out_csv", type=str, default="results/phase4_validation.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- baseline checkpoint --------------------------------------------------
    fob_dir = "/kaggle/working/baseline_fob"
    os.makedirs(fob_dir, exist_ok=True)
    existing = glob.glob(f"{fob_dir}/**/*.pth", recursive=True)
    if existing:
        baseline_ckpt = existing[0]
    else:
        print("Downloading baseline FoB SABS checkpoint from HuggingFace...")
        zp = os.path.join(fob_dir, "SABS_FSMIS_FoB.zip")
        urllib.request.urlretrieve(
            "https://huggingface.co/PrimeBo1/FoB_SAM/resolve/main/"
            "exps_train_on_SABS_FSMIS_FoB.zip", zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(fob_dir)
        baseline_ckpt = glob.glob(f"{fob_dir}/**/*.pth", recursive=True)[0]
    print(f"Baseline FoB checkpoint: {baseline_ckpt}")
    print(f"AdaFoB checkpoint:       {args.ckpt}")

    # --- data root ------------------------------------------------------------
    data_root = args.data_root
    if not data_root:
        for pat in ["/kaggle/input/**/*sabs_CT_normalized*", "/kaggle/input/**/*abd*ct*"]:
            for h in glob.glob(pat, recursive=True):
                if os.path.isdir(h):
                    data_root = h
                    break
            if data_root:
                break
        if not data_root:
            data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct/abd-ct"
    print(f"Data root: {data_root}")

    # --- load volumes ---------------------------------------------------------
    volumes, stats = load_volumes(data_root, tuple(args.hu_window), args.limit_volumes)

    # --- resolve normalisers --------------------------------------------------
    if args.baseline_norm == "dataset":
        b_mean, b_std = stats["dataset_mean"], stats["dataset_std"]
    elif args.baseline_norm == "fixed":
        b_mean, b_std = args.fixed_mean, args.fixed_std
    else:
        b_mean = b_std = None                       # per-volume, resolved lazily

    def make_baseline_norm(vol):
        if b_mean is not None:
            return lambda sl: norm_fob(sl, b_mean, b_std)
        mu = float(vol["canon"].mean())
        sd = float(vol["canon"].std())
        return lambda sl: norm_fob(sl, mu, sd)

    def make_adafob_norm(vol):
        if args.adafob_norm == "train_slice":
            return norm_adafob_trainstyle
        if args.adafob_norm == "dataset":
            return lambda sl: norm_fob(sl, stats["dataset_mean"], stats["dataset_std"])
        mu = float(vol["canon"].mean())
        sd = float(vol["canon"].std())
        return lambda sl: norm_fob(sl, mu, sd)

    print(f"\nBaseline FoB normalisation: {args.baseline_norm}"
          + (f" (mean={b_mean:.3f}, std={b_std:.3f})" if b_mean is not None else " (per-volume)"))
    print(f"AdaFoB normalisation:       {args.adafob_norm}")

    # --- SAM ------------------------------------------------------------------
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)

    # --- organs present -------------------------------------------------------
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver"}
    requested = [int(x) for x in args.organs.split(",") if x.strip()]
    organ_map = {k: v for k, v in organ_map.items() if k in requested}
    print("\nChecking organ availability:")
    organs = available_organs(volumes, organ_map, args.min_pixels)
    if not organs:
        raise ValueError("No requested organ has enough annotated slices.")

    # --- sanity check ---------------------------------------------------------
    if args.sanity_check:
        gt_prompt_sanity_check(predictor, volumes, organs, organ_map, seed=args.seed)

    # --- models ---------------------------------------------------------------
    dummy = type("A", (), {})()
    print("\nLoading models:")
    adafob = FewShotSeg(dummy).cuda().eval()
    if os.path.exists(args.ckpt):
        load_checkpoint(adafob, args.ckpt, "AdaFoB", strict=args.strict_ckpt)
    else:
        print(f"  !! AdaFoB ckpt not found: {args.ckpt} -- results will be meaningless")

    fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(fob, baseline_ckpt, "FoB baseline", strict=args.strict_ckpt)

    # AdaFoB's GAP path is opt-in; the baseline must never receive it.
    ada_kwargs = {"train": False, "use_skeleton": True}
    base_kwargs = {"train": False, "use_skeleton": False}

    # --- run ------------------------------------------------------------------
    results = []
    skipped = 0
    for ep_i in range(args.n_episodes):
        cls = organs[ep_i % len(organs)]
        ep = sample_episode(volumes, cls, args.min_pixels)
        if ep is None:
            skipped += 1
            continue

        sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
        base_sample = build_inputs(volumes, ep, make_baseline_norm(sv))
        ada_sample = build_inputs(volumes, ep, make_adafob_norm(sv))

        if base_sample["support_fg_labels"][0][0].max() == 0 or base_sample["query_labels"].max() == 0:
            skipped += 1
            continue

        try:
            ada_neg, ada_pos = run_model(adafob, ada_sample, **ada_kwargs)
        except TypeError:
            ada_neg, ada_pos = run_model(adafob, ada_sample, train=False)
        except Exception as e:
            print(f"  Ep {ep_i}: AdaFoB error: {e}")
            skipped += 1
            continue

        try:
            base_neg, base_pos = run_model(fob, base_sample, **base_kwargs)
        except TypeError:
            base_neg, base_pos = run_model(fob, base_sample, train=False)
        except Exception as e:
            print(f"  Ep {ep_i}: FoB error: {e}")
            skipped += 1
            continue

        # One SAM image for both models -> the comparison isolates prompt quality.
        gt = base_sample["query_mask_np"]
        predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))

        ada_pred = predict_sam_from_points(predictor, ada_pos, ada_neg)
        base_pred = predict_sam_from_points(predictor, base_pos, base_neg)
        if ada_pred is None or base_pred is None:
            skipped += 1
            continue

        results.append({
            "ep": ep_i, "organ": organ_map[cls],
            "ada_dice": compute_dice(ada_pred, gt),
            "ada_hd95": compute_hd95(ada_pred, gt),
            "base_dice": compute_dice(base_pred, gt),
            "base_hd95": compute_hd95(base_pred, gt),
            "ada_n_pos": len(_as_points(ada_pos)), "ada_n_neg": len(_as_points(ada_neg)),
            "base_n_pos": len(_as_points(base_pos)), "base_n_neg": len(_as_points(base_neg)),
        })

        if (ep_i + 1) % 10 == 0 and results:
            last10 = results[-min(10, len(results)):]
            print(f"Episode {ep_i+1}/{args.n_episodes} | "
                  f"Ada Dice(last10): {np.mean([r['ada_dice'] for r in last10]):.4f} | "
                  f"FoB Dice(last10): {np.mean([r['base_dice'] for r in last10]):.4f}")

    # --- CSV ------------------------------------------------------------------
    csv_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["ep", "organ", "ada_dice", "ada_hd95", "base_dice", "base_hd95",
              "ada_n_pos", "ada_n_neg", "base_n_pos", "base_n_neg"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nCSV written to: {csv_path}")
    print(f"Completed {len(results)} episodes ({skipped} skipped)")

    # run configuration, so a CSV is never ambiguous later
    meta = {
        "hu_window": args.hu_window, "baseline_norm": args.baseline_norm,
        "adafob_norm": args.adafob_norm, "dataset_mean": stats["dataset_mean"],
        "dataset_std": stats["dataset_std"], "domains": stats["domains"],
        "n_volumes": len(volumes), "organs": organs, "seed": args.seed,
    }
    with open(csv_path.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not results:
        print("No results collected!")
        return

    # --- summary --------------------------------------------------------------
    ada_d = [r["ada_dice"] for r in results]
    base_d = [r["base_dice"] for r in results]
    ada_h = [r["ada_hd95"] for r in results]
    base_h = [r["base_hd95"] for r in results]

    print(f"\n{'='*60}")
    print(f"AdaFoB  Mean Dice: {np.mean(ada_d):.4f}   Mean HD95: {np.mean(ada_h):.4f}")
    print(f"FoB     Mean Dice: {np.mean(base_d):.4f}   Mean HD95: {np.mean(base_h):.4f}")

    print("\nPer-organ breakdown:")
    for name in organ_map.values():
        sub = [r for r in results if r["organ"] == name]
        if sub:
            print(f"  {name:>8s}:  Ada={np.mean([r['ada_dice'] for r in sub]):.4f}   "
                  f"FoB={np.mean([r['base_dice'] for r in sub]):.4f}   (n={len(sub)})")

    try:
        _, p_d = wilcoxon(ada_d, base_d)
        _, p_h = wilcoxon(ada_h, base_h)
        print(f"\nWilcoxon  Dice p={p_d:.4e}   HD95 p={p_h:.4e}")
    except ValueError as e:
        print(f"Wilcoxon failed: {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate()
