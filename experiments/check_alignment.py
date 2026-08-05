"""Image/label alignment checker for the Abd-CT crops.

WHY
---
diagnose.py showed an oracle ceiling of ~0.50: SAM prompted with 10 points taken
from INSIDE the ground-truth mask recovers only half of that mask.  Those points
are inside GT by construction, so the mask is self-consistent -- which means the
mask does not describe the IMAGE at those coordinates.  That is an image/label
misalignment, and it also explains the two other symptoms (support prototypes
pooled over the wrong pixels -> 30.8% of positives inside GT, negatives drifting
to 69.7 px from the organ).

WHAT THIS DOES  (CPU only unless --with_sam)
--------------------------------------------
1. HEADER CHECK      - SimpleITK size / spacing / origin / direction for each
                       image-label pair.  A direction-cosine mismatch is the
                       most common cause of a silent flip.
2. LABEL INVENTORY   - unique label values, per-organ area, connected components.
                       Also flags a likely organ-id mapping error (liver should
                       be several times larger than spleen).
3. TISSUE TEST       - fraction of GT pixels whose canonical intensity falls in
                       the soft-tissue band.  A correctly aligned abdominal
                       organ scores > 0.85; a misaligned one scores far lower
                       because the mask covers fat, air or bowel gas.
4. TRANSFORM SWEEP   - re-scores the tissue test under flips / transpose / z
                       reversal and under a Z-shift, and reports which
                       transform makes the label agree with the image.
5. OPTIONAL SAM      - oracle Dice under the identity vs the best transform,
                       stratified by GT area, to confirm the ceiling lifts.

    python experiments/check_alignment.py --data_root <root>
    python experiments/check_alignment.py --data_root <root> --with_sam \
        --sam_ckpt /kaggle/working/checkpoints/sam_vit_h_4b8939.pth
"""

import os
import re
import sys
import glob
import json
import random
import argparse

import numpy as np
import cv2
import SimpleITK as sitk
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from data.preprocess import to_canonical, resize_volume, sam_uint8_from_canonical, oracle_prompts  # noqa: E402

# Canonical intensity of soft tissue.  With the default window [-125, 275] HU
# mapped onto [0, 255]:  canon = (HU + 125) / 400 * 255
#   HU  20 -> 92    HU  60 -> 118    HU  90 -> 137
SOFT_LO, SOFT_HI = 85.0, 145.0


def hu_to_canon(hu, window=(-125.0, 275.0)):
    lo, hi = window
    return (hu - lo) / (hi - lo) * 255.0


# ---------------------------------------------------------------------------
# label transforms
# ---------------------------------------------------------------------------

TRANSFORMS = {
    "identity":       lambda m: m,
    "flip_y":         lambda m: m[::-1, :],
    "flip_x":         lambda m: m[:, ::-1],
    "rot180":         lambda m: m[::-1, ::-1],
    "transpose":      lambda m: m.T,
    "transpose+flip_y": lambda m: m.T[::-1, :],
    "transpose+flip_x": lambda m: m.T[:, ::-1],
    "rot90":          lambda m: np.rot90(m, 1),
    "rot270":         lambda m: np.rot90(m, 3),
}


def tissue_score(img_slice, mask, soft_lo=SOFT_LO, soft_hi=SOFT_HI):
    """Fraction of masked pixels lying in the soft-tissue intensity band."""
    m = mask.astype(bool)
    if m.sum() < 10:
        return None
    vals = img_slice[m]
    return float(((vals >= soft_lo) & (vals <= soft_hi)).mean())


def contrast_score(img_slice, mask, ring_px=8):
    """|mean_inside - mean_ring| / pooled std.  High when the mask outlines a
    genuinely distinct structure in THIS image."""
    m = mask.astype(np.uint8)
    if m.sum() < 10:
        return None
    k = np.ones((3, 3), np.uint8)
    ring = cv2.dilate(m, k, iterations=ring_px) - m
    if ring.sum() < 10:
        return None
    a, b = img_slice[m.astype(bool)], img_slice[ring.astype(bool)]
    pooled = np.sqrt(0.5 * (a.var() + b.var())) + 1e-6
    return float(abs(a.mean() - b.mean()) / pooled)


def grad_map(img_slice):
    """Sobel gradient magnitude, computed once per slice and reused."""
    img = img_slice.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def contour_grad(grad, mask):
    """Mean image gradient along the mask contour."""
    m = mask.astype(np.uint8)
    if m.sum() < 10:
        return None
    k = np.ones((3, 3), np.uint8)
    contour = cv2.dilate(m, k, 1) - cv2.erode(m, k, 1)
    if contour.sum() < 10:
        return None
    return float(grad[contour.astype(bool)].mean())


def edge_z(grad, mask, body=None, n_null=40, rng=None, min_shift=8):
    """PRIMARY alignment metric, self-calibrating and robust.

    Compares the gradient on the TRUE mask contour against a null distribution
    built by randomly translating the SAME mask to other placements INSIDE the
    body, scored with a median/MAD z-score.

    Why a null instead of an absolute threshold: raw contour gradient depends on
    how much bone and body-air boundary a slice contains, so no fixed cutoff
    transfers across datasets.  Translating the identical shape holds size and
    perimeter fixed and isolates the only question that matters -- does this
    outline sit on a real boundary HERE rather than anywhere else?

    Two details that matter in practice:
      * candidate placements must stay inside the body, otherwise the null is
        dominated by the skin/air edge and swamps the true signal;
      * median/MAD rather than mean/std, so a single bone crossing cannot
        inflate the null.

    Aligned masks score z > 3.  Misaligned masks score z ~ 0.
    """
    rng = rng or np.random.default_rng(0)
    true = contour_grad(grad, mask)
    if true is None:
        return None

    m = mask.astype(bool)
    H, W = m.shape
    area = int(m.sum())
    null = []
    tries = 0
    while len(null) < n_null and tries < n_null * 8:
        tries += 1
        dy = int(rng.integers(-H // 3, H // 3 + 1))
        dx = int(rng.integers(-W // 3, W // 3 + 1))
        if abs(dy) < min_shift and abs(dx) < min_shift:
            continue
        shifted = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        if body is not None and (shifted & body).sum() < 0.95 * area:
            continue                      # keep the null inside the abdomen
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


def edge_score(img_slice, mask, body_thresh=20.0):
    """Raw contour gradient normalised by mean body gradient (context only)."""
    grad = grad_map(img_slice)
    c = contour_grad(grad, mask)
    if c is None:
        return None
    body = img_slice > body_thresh
    ref = float(grad[body].mean()) if body.sum() > 100 else float(grad.mean())
    return float(c / ref) if ref > 1e-6 else None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def collect_pairs(root):
    fob_dir = os.path.join(root, "sabs_CT_normalized")
    imgs = sorted(glob.glob(os.path.join(fob_dir, "image_*.nii.gz"))) if os.path.isdir(fob_dir) else []
    if imgs:
        return [(p, p.replace("image_", "label_")) for p in imgs
                if os.path.exists(p.replace("image_", "label_"))]

    idict, ldict = {}, {}
    for r, dirs, files in os.walk(root, followlinks=True):
        for entry in sorted(files + dirs):
            if not (entry.endswith(".nii") or entry.endswith(".nii.gz")):
                continue
            
            path = os.path.join(r, entry)
            if os.path.isdir(path):
                inner = [f for f in os.listdir(path) if (f.endswith(".nii") or f.endswith(".nii.gz")) and os.path.isfile(os.path.join(path, f))]
                if not inner:
                    continue
                path = os.path.join(path, inner[0])
                
            fl = entry.lower()
            m = re.search(r"(\d+)", entry)
            if not m:
                continue
            pid = m.group(1)
            if "label" in fl or "seg" in fl:
                ldict[pid] = path
            elif "image" in fl or "img" in fl or "avg" in fl:
                idict[pid] = path
    return [(idict[p], ldict[p]) for p in sorted(set(idict) & set(ldict))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0])
    ap.add_argument("--organ", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=8)
    ap.add_argument("--min_pixels", type=int, default=50)
    ap.add_argument("--with_sam", action="store_true")
    ap.add_argument("--sam_ckpt", type=str,
                    default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--out_json", type=str, default="results/alignment_check.json")
    args = ap.parse_args()

    pairs = collect_pairs(args.data_root)[:args.max_pairs]
    if not pairs:
        raise SystemExit(f"no image/label pairs under {args.data_root}")
    print(f"Checking {len(pairs)} image/label pairs\n")

    report = {"pairs": [], "soft_band_canonical": [SOFT_LO, SOFT_HI]}

    # ---------------- 1. header check -------------------------------------
    print("=" * 78)
    print("1. HEADER CHECK  (a direction/origin mismatch means a silent flip)")
    print("=" * 78)
    header_bad = False
    for ip, lp in pairs:
        ii, li = sitk.ReadImage(ip), sitk.ReadImage(lp)
        same_size = ii.GetSize() == li.GetSize()
        same_dir = np.allclose(ii.GetDirection(), li.GetDirection(), atol=1e-4)
        same_org = np.allclose(ii.GetOrigin(), li.GetOrigin(), atol=1e-3)
        same_spc = np.allclose(ii.GetSpacing(), li.GetSpacing(), atol=1e-4)
        flag = "" if (same_size and same_dir and same_org and same_spc) else "   <-- MISMATCH"
        if flag:
            header_bad = True
        print(f"  {os.path.basename(ip)[:38]:38s} size={same_size} dir={same_dir} "
              f"origin={same_org} spacing={same_spc}{flag}")
        if not same_dir:
            print(f"      image dir : {np.round(ii.GetDirection(), 3)}")
            print(f"      label dir : {np.round(li.GetDirection(), 3)}")
    report["header_mismatch"] = header_bad
    if not header_bad:
        print("  All headers agree -- a flip, if present, is baked into the voxel data.")

    # ---------------- load volumes ----------------------------------------
    vols = []
    for ip, lp in pairs:
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float64)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lp)).astype(np.int32)
        if img.shape != lbl.shape:
            print(f"  !! shape mismatch, skipping {os.path.basename(ip)}")
            continue
        img = resize_volume(img, (256, 256), False)
        lbl = resize_volume(lbl, (256, 256), True)
        canon, domain = to_canonical(img, tuple(args.hu_window))
        vols.append({"canon": canon, "label": lbl, "domain": domain,
                     "name": os.path.basename(ip)})

    # ---------------- 2. label inventory ----------------------------------
    print("\n" + "=" * 78)
    print("2. LABEL INVENTORY")
    print("=" * 78)
    domains = {}
    all_vals = {}
    for v in vols:
        domains[v["domain"]] = domains.get(v["domain"], 0) + 1
        for val, cnt in zip(*np.unique(v["label"], return_counts=True)):
            all_vals[int(val)] = all_vals.get(int(val), 0) + int(cnt)
    print(f"  detected intensity domains : {domains}")
    print(f"  canonical soft-tissue band : [{SOFT_LO:.0f}, {SOFT_HI:.0f}] "
          f"(HU {args.hu_window[0]:.0f}..{args.hu_window[1]:.0f} -> 0..255)")
    print(f"  label values (value: voxels):")
    for val in sorted(all_vals):
        if val == 0:
            continue
        print(f"    {val:3d}: {all_vals[val]:>9,d}")
    if set(all_vals) & {200, 500, 600}:
        print("    !! CHAOS-style values detected (200/500/600) -- remap to 1/2/3")

    # organ-size sanity: liver (6) should dwarf spleen (1)
    if 1 in all_vals and 6 in all_vals:
        ratio = all_vals[6] / max(all_vals[1], 1)
        print(f"  liver(6)/spleen(1) voxel ratio = {ratio:.2f}")
        if ratio < 1.0:
            print("    !! label 1 is LARGER than label 6 -- your organ_map may be "
                  "inverted (1=liver, 6=spleen) for this dataset")
    report["label_values"] = all_vals

    # per-slice area + components for the organ under test
    areas, ncomp = [], []
    for v in vols:
        for z in range(v["label"].shape[0]):
            m = (v["label"][z] == args.organ)
            a = int(m.sum())
            if a > args.min_pixels:
                areas.append(a)
                ncomp.append(int(ndimage.label(m)[1]))
    if areas:
        areas = np.array(areas)
        print(f"\n  organ {args.organ}: {len(areas)} annotated slices")
        print(f"    area px  min={areas.min()} p25={np.percentile(areas,25):.0f} "
              f"median={np.median(areas):.0f} p75={np.percentile(areas,75):.0f} "
              f"max={areas.max()}")
        print(f"    slices under 300 px: {(areas < 300).sum()} "
              f"({(areas < 300).mean():.0%})  <- SAM and Dice both degrade badly here")
        print(f"    mean connected components per slice: {np.mean(ncomp):.2f}")
        report["organ_area_stats"] = {
            "n_slices": int(len(areas)), "min": int(areas.min()),
            "median": float(np.median(areas)), "max": int(areas.max()),
            "frac_under_300px": float((areas < 300).mean()),
            "mean_components": float(np.mean(ncomp))}

    # ---------------- 3+4. tissue test and transform sweep -----------------
    print("\n" + "=" * 78)
    print("3-4. TISSUE TEST AND TRANSFORM SWEEP")
    print("=" * 78)
    print("  PRIMARY metric is EDGE-Z: gradient on the mask contour vs a null")
    print("  built by translating the SAME mask elsewhere in the slice.")
    print("  Aligned > 3.0, misaligned ~ 0.0. Self-calibrating, so it transfers")
    print("  across datasets. soft-frac is context only: the abdomen is soft")
    print("  tissue everywhere, so a misplaced mask still scores ~1.0 there.\n")
    print(f"  {'transform':20s} {'EDGE-Z':>8s} {'edge/ref':>9s} {'contrast':>10s} {'soft':>7s}")

    # precompute one gradient map per slice
    grads = [[grad_map(v["canon"][z]) for z in range(v["canon"].shape[0])] for v in vols]
    bodies = [[(v["canon"][z] > 20.0) for z in range(v["canon"].shape[0])] for v in vols]

    sweep = {}
    for tname, tf in TRANSFORMS.items():
        zsc, es, ts, cs = [], [], [], []
        rng = np.random.default_rng(0)
        for vi, v in enumerate(vols):
            for z in range(v["label"].shape[0]):
                m = (v["label"][z] == args.organ)
                if m.sum() <= args.min_pixels:
                    continue
                mt = tf(m)
                if mt.shape != v["canon"][z].shape:
                    continue
                zz = edge_z(grads[vi][z], mt, body=bodies[vi][z], rng=rng)
                e = edge_score(v["canon"][z], mt)
                t = tissue_score(v["canon"][z], mt)
                c = contrast_score(v["canon"][z], mt)
                for val, acc in ((zz, zsc), (e, es), (t, ts), (c, cs)):
                    if val is not None:
                        acc.append(val)
        if zsc:
            sweep[tname] = {"edge_z": float(np.mean(zsc)), "edge": float(np.mean(es or [0])),
                            "soft": float(np.mean(ts or [0])),
                            "contrast": float(np.mean(cs or [0]))}
            print(f"  {tname:20s} {np.mean(zsc):8.2f} {np.mean(es or [0]):9.3f} "
                  f"{np.mean(cs or [0]):10.3f} {np.mean(ts or [0]):7.3f}")
    best_t = max(sweep, key=lambda k: sweep[k]["edge_z"])
    print(f"\n  best transform by EDGE-Z: {best_t} ({sweep[best_t]['edge_z']:.2f})")
    report["transform_sweep"] = sweep
    report["best_transform"] = best_t

    # Z-shift sweep
    print("\n  Z-SHIFT SWEEP (label slice z scored against image slice z+k, EDGE-Z)")
    zs = {}
    for k in range(-5, 6):
        acc = []
        rng = np.random.default_rng(0)
        for vi, v in enumerate(vols):
            Z = v["label"].shape[0]
            for z in range(Z):
                zz = z + k
                if zz < 0 or zz >= Z:
                    continue
                m = (v["label"][z] == args.organ)
                if m.sum() <= args.min_pixels:
                    continue
                e = edge_z(grads[vi][zz], m, body=bodies[vi][zz], rng=rng)
                if e is not None:
                    acc.append(e)
        if acc:
            zs[k] = float(np.mean(acc))
    zmax = max(zs.values()) if zs else 1.0
    for k in sorted(zs):
        bar = "#" * int(zs[k] / max(zmax, 1e-6) * 40)
        print(f"    k={k:+d}: {zs[k]:.3f}  {bar}")
    best_k = max(zs, key=zs.get) if zs else 0
    print(f"  best z-shift: k={best_k:+d}")
    report["z_shift_sweep"] = zs
    report["best_z_shift"] = int(best_k)

    id_edge = sweep.get("identity", {}).get("edge_z", 0.0)
    best_edge = sweep[best_t]["edge_z"]
    print("\n" + "-" * 78)
    if best_t == "identity" and best_k == 0 and id_edge >= 3.0:
        print(f"  VERDICT: labels ARE aligned (edge-z={id_edge:.2f}).")
        print("  The 0.50 oracle ceiling is therefore NOT a flip. Next suspect, in order:")
        print(f"    a) tiny-area slices -- see the area table above; rerun diagnose.py")
        print(f"       with --min_pixels 300 and compare the oracle.")
        print("    b) multi-component masks (mean components printed above): SAM cannot")
        print("       return several disconnected blobs from one point set.")
        print("    c) label id mapping: confirm organ 1 really is the spleen here.")
    elif (best_edge > id_edge + 1.0) or best_k != 0:
        print("  VERDICT: MISALIGNMENT DETECTED.")
        print(f"  Apply '{best_t}' to the label (or its inverse to the image)"
              + (f" and a Z-shift of {best_k:+d}." if best_k else "."))
        print(f"  edge-z {id_edge:.2f} (identity) -> {best_edge:.2f} (fixed)")
        print("  Fix this in the loader, then re-run diagnose.py. The oracle must clear")
        print("  0.85 before any FoB vs AdaFoB number means anything.")
    else:
        print(f"  VERDICT: no transform helps (best edge-z {best_edge:.2f} vs identity "
              f"{id_edge:.2f}).")
        print("  If identity is also below ~3, the mask outlines do not follow ANY")
        print("  boundary in these images. Suspect wrong image/label PAIRING -- the")
        print("  filename regex takes the FIRST number in the name, so 'abd_ct_2020_p01'")
        print("  would pair every volume on '2020'. Print the pairs and verify by eye.")
    print("-" * 78)

    # ---------------- 5. optional SAM confirmation -------------------------
    if args.with_sam:
        sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "third_party", "FoB_SAM")))
        from segment_anything import sam_model_registry, SamPredictor
        print("\n" + "=" * 78)
        print("5. SAM ORACLE UNDER identity VS best transform, STRATIFIED BY AREA")
        print("=" * 78)
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
        pr = SamPredictor(sam)
        rng = random.Random(0)

        buckets = {"<300": [], "300-1000": [], ">1000": []}
        buckets_fixed = {k: [] for k in buckets}
        tf = TRANSFORMS[best_t]
        n = 0
        for v in vols:
            for z in range(v["label"].shape[0]):
                if n >= 24:
                    break
                m = (v["label"][z] == args.organ)
                if m.sum() <= args.min_pixels:
                    continue
                n += 1
                pr.set_image(sam_uint8_from_canonical(v["canon"][z]))
                key = "<300" if m.sum() < 300 else ("300-1000" if m.sum() < 1000 else ">1000")
                for mask, store in ((m, buckets), (tf(m), buckets_fixed)):
                    mask = mask.astype(np.uint8)
                    if mask.shape != v["canon"][z].shape or mask.sum() < 20:
                        continue
                    pos, neg = oracle_prompts(mask, rng=rng)
                    if len(pos) == 0:
                        continue
                    pts = np.concatenate([pos, neg], 0)
                    lbs = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], 0)
                    mk, _, _ = pr.predict(point_coords=pts, point_labels=lbs,
                                          multimask_output=True)
                    inter = (mk[0].astype(bool) & mask.astype(bool)).sum()
                    tot = mk[0].sum() + mask.sum()
                    store[key].append(2.0 * inter / tot if tot else 1.0)

        print(f"  {'area bucket':14s} {'n':>4s} {'identity':>10s} {'+'+best_t:>18s}")
        for k in buckets:
            if buckets[k]:
                f = np.mean(buckets_fixed[k]) if buckets_fixed[k] else float("nan")
                print(f"  {k:14s} {len(buckets[k]):4d} {np.mean(buckets[k]):10.4f} {f:18.4f}")
        report["sam_oracle_by_area"] = {k: float(np.mean(v)) for k, v in buckets.items() if v}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out_json}")


if __name__ == "__main__":
    main()
