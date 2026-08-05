"""Attribution diagnostics for the FoB / AdaFoB Dice collapse.

Runs a ladder of controlled experiments that localise WHERE the failure is,
instead of guessing.  Reuses experiments/eval.py so it exercises exactly the
same code paths as the real evaluation.

    python experiments/diagnose.py --ckpt <adafob.pth> --n_cases 12

Stages
------
0. ORACLE + MASK-INDEX SWEEP
   SAM with ground-truth prompts, scoring masks[0], masks[1], masks[2].
   FoB hardcodes index 0.  If index 1 or 2 is much better on your data, the
   mask-selection convention is wrong for these crops.
   Verdict: oracle Dice < 0.60 => image pipeline is still broken; stop here.

1. PROMPT GEOMETRY AUDIT
   Fraction of predicted POSITIVE points that land inside GT, and of predicted
   NEGATIVE points that land outside GT, per model.
   Verdict: pos-inside ~0 while oracle is high => the prompt coordinates are
   wrong (convention or transpose), not the segmentation.

2. COORDINATE-ORDER TEST
   Same prompts fed as (x, y) and as (y, x).  If the swap is markedly better,
   the model emits (row, col) and eval.py must transpose.

3. PROMPT-SWAP LADDER  (the decisive experiment)
   oracle_pos + oracle_neg   -> upper bound for this image pipeline
   model_pos  + oracle_neg   -> isolates the POSITIVE prompt head
   oracle_pos + model_neg    -> isolates the NEGATIVE prompt head (FoB's claim)
   model_pos  + model_neg    -> the full model
   Whichever row collapses tells you which half of the prompt generator to fix.

4. MISSING-KEY IMPACT TEST
   The baseline checkpoint is missing `encoder.reduce2.weight`, i.e. that layer
   is RANDOM.  This re-randomises the missing parameters and checks whether the
   model's prompts change.  If they do, the "baseline" is not published FoB and
   every baseline number is invalid.
"""

import os
import sys
import copy
import json
import random
import argparse
import importlib.util

import numpy as np
import torch
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "third_party", "FoB_SAM")))


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("adafob_eval", os.path.join(_HERE, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load_eval_module()
from models.FoB import FewShotSeg                      # noqa: E402
from segment_anything import sam_model_registry, SamPredictor   # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def oracle_prompts(gt, n_pos=10, n_neg=10, r_outer=15, r_inner=13, erode_iters=3, rng=None):
    """GT-derived prompts in SAM's (x, y) order (morphological core positives)."""
    rng = rng or random.Random(0)
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


def predict_all(predictor, pos, neg):
    """Return all 3 SAM masks for the currently registered image."""
    pos = EV._as_points(pos)
    neg = EV._as_points(neg)
    if len(pos) == 0 and len(neg) == 0:
        return None, None
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl,
                                         multimask_output=True)
    return masks, scores


def dice_of(predictor, pos, neg, gt, idx=0):
    masks, _ = predict_all(predictor, pos, neg)
    if masks is None:
        return 0.0
    return EV.compute_dice(masks[idx], gt)


def prompt_stats(pos, neg, gt):
    """How many prompts are where they should be."""
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    H, W = gt.shape
    out = {}

    def _inside(pts):
        pts = EV._as_points(pts)
        if len(pts) == 0:
            return np.zeros(0, bool), 0
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        oob = int(((pts[:, 0] < 0) | (pts[:, 0] >= W) |
                   (pts[:, 1] < 0) | (pts[:, 1] >= H)).sum())
        return gt[ys, xs].astype(bool), oob

    pin, pos_oob = _inside(pos)
    nin, neg_oob = _inside(neg)
    out["n_pos"] = len(EV._as_points(pos))
    out["n_neg"] = len(EV._as_points(neg))
    out["pos_inside_gt"] = float(pin.mean()) if len(pin) else float("nan")
    out["neg_outside_gt"] = float((~nin).mean()) if len(nin) else float("nan")
    out["pos_oob"] = pos_oob
    out["neg_oob"] = neg_oob

    # how far are the negatives from the organ boundary (FoB targets ~15 px)
    dist_bg = cv2.distanceTransform((1 - gt).astype(np.uint8), cv2.DIST_L2, 5)
    npts = EV._as_points(neg)
    if len(npts):
        xs = np.clip(np.round(npts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(npts[:, 1]).astype(int), 0, H - 1)
        out["neg_dist_to_organ"] = float(dist_bg[ys, xs].mean())
    else:
        out["neg_dist_to_organ"] = float("nan")
    return out


def fwd(model, sample, use_skeleton):
    """Model forward, tolerant of forks whose signature lacks use_skeleton."""
    try:
        return EV.run_model(model, sample, train=False, use_skeleton=use_skeleton)
    except TypeError:
        return EV.run_model(model, sample, train=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/adafob_abdct.pth")
    ap.add_argument("--baseline_ckpt", type=str, default=None,
                    help="defaults to the downloaded HuggingFace FoB checkpoint")
    ap.add_argument("--sam_ckpt", type=str,
                    default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--n_cases", type=int, default=12)
    ap.add_argument("--organs", type=str, default="1")
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0])
    ap.add_argument("--baseline_norm", choices=["dataset", "volume", "fixed"], default="dataset")
    ap.add_argument("--adafob_norm", choices=["train_slice", "dataset", "volume"],
                    default="train_slice")
    ap.add_argument("--n_pos", type=int, default=10)
    ap.add_argument("--n_neg", type=int, default=10)
    ap.add_argument("--out_json", type=str, default="results/diagnosis.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root or "/kaggle/input/datasets/nhatphatnguyen/abd-ct/abd-ct"
    volumes, stats = EV.load_volumes(data_root, tuple(args.hu_window))

    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver"}
    want = [int(x) for x in args.organs.split(",") if x.strip()]
    organ_map = {k: v for k, v in organ_map.items() if k in want}
    organs = EV.available_organs(volumes, organ_map)
    if not organs:
        raise SystemExit("no usable organ")

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)

    # normalisers
    if args.baseline_norm == "dataset":
        b_mu, b_sd = stats["dataset_mean"], stats["dataset_std"]
    elif args.baseline_norm == "fixed":
        b_mu, b_sd = 94.0, 62.0
    else:
        b_mu = b_sd = None

    def base_norm(vol):
        if b_mu is not None:
            return lambda sl: EV.norm_fob(sl, b_mu, b_sd)
        mu, sd = float(vol["canon"].mean()), float(vol["canon"].std())
        return lambda sl: EV.norm_fob(sl, mu, sd)

    def ada_norm(vol):
        if args.adafob_norm == "train_slice":
            return EV.norm_adafob_trainstyle
        if args.adafob_norm == "dataset":
            return lambda sl: EV.norm_fob(sl, stats["dataset_mean"], stats["dataset_std"])
        mu, sd = float(vol["canon"].mean()), float(vol["canon"].std())
        return lambda sl: EV.norm_fob(sl, mu, sd)

    # models
    dummy = type("A", (), {})()
    print("\nLoading models:")
    ada = FewShotSeg(dummy).cuda().eval()
    if os.path.exists(args.ckpt):
        EV.load_checkpoint(ada, args.ckpt, "AdaFoB")
    else:
        print(f"  !! AdaFoB ckpt missing: {args.ckpt}")

    base_ckpt = args.baseline_ckpt
    if not base_ckpt:
        import glob
        hits = glob.glob("/kaggle/working/baseline_fob/**/*.pth", recursive=True)
        base_ckpt = hits[0] if hits else None
    base = FewShotSeg(dummy).cuda().eval()
    base_missing = []
    if base_ckpt:
        obj = torch.load(base_ckpt, map_location="cpu")
        for key in ("state_dict", "model", "net"):
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in obj.items()}
        missing, unexpected = base.load_state_dict(cleaned, strict=False)
        base_missing = list(missing)
        print(f"  FoB baseline: missing={len(missing)} unexpected={len(unexpected)}")
        if base_missing:
            print(f"    RANDOMLY INITIALISED in the baseline: {base_missing}")
    else:
        print("  !! no baseline checkpoint found")

    # build the case list once so every stage uses identical episodes
    cases = []
    for i in range(args.n_cases):
        cls = organs[i % len(organs)]
        ep = EV.sample_episode(volumes, cls)
        if ep is None:
            continue
        qv = volumes[ep["query_vol"]]
        gt = (qv["label"][ep["query_slice"]] == cls).astype(np.uint8)
        if gt.sum() < 20:
            continue
        cases.append((ep, gt))
    print(f"\nUsing {len(cases)} diagnostic cases\n")

    report = {"n_cases": len(cases), "baseline_missing_keys": base_missing,
              "dataset_mean": stats["dataset_mean"], "dataset_std": stats["dataset_std"]}

    # ---------------- stage 0: oracle + mask index sweep -------------------
    print("=" * 72)
    print("STAGE 0 - oracle prompts, SAM mask-index sweep")
    print("=" * 72)
    idx_scores = {0: [], 1: [], 2: []}
    for ep, gt in cases:
        qv = volumes[ep["query_vol"]]
        predictor.set_image(EV.sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
        pos, neg = oracle_prompts(gt, args.n_pos, args.n_neg, rng=random.Random(args.seed))
        masks, scores = predict_all(predictor, pos, neg)
        if masks is None:
            continue
        for i in range(min(3, len(masks))):
            idx_scores[i].append(EV.compute_dice(masks[i], gt))
    for i in sorted(idx_scores):
        if idx_scores[i]:
            print(f"  masks[{i}] oracle Dice: {np.mean(idx_scores[i]):.4f}")
    report["oracle_dice_by_mask_idx"] = {str(i): (float(np.mean(v)) if v else None)
                                         for i, v in idx_scores.items()}
    oracle0 = float(np.mean(idx_scores[0])) if idx_scores[0] else 0.0
    best_idx = max((i for i in idx_scores if idx_scores[i]),
                   key=lambda i: np.mean(idx_scores[i]))
    print(f"\n  FoB uses masks[0]; best on your data is masks[{best_idx}]")
    if oracle0 < 0.60:
        print("  !! ORACLE IS LOW - the image pipeline, labels or prompt convention are")
        print("     still wrong. Nothing downstream is interpretable until this passes.")
    else:
        print("  OK - SAM can segment these slices from good prompts.")

    # ---------------- stages 1-3 over both models --------------------------
    rows = {}
    for tag, model, norm_factory, use_skel in [
            ("FoB-baseline", base, base_norm, False),
            ("AdaFoB", ada, ada_norm, True)]:
        print("\n" + "=" * 72)
        print(f"STAGES 1-3 - {tag}")
        print("=" * 72)
        st = {k: [] for k in ["pos_inside_gt", "neg_outside_gt", "neg_dist_to_organ",
                              "n_pos", "n_neg", "pos_oob", "neg_oob"]}
        d_full, d_swap, d_oracle, d_mpos_oneg, d_opos_mneg = [], [], [], [], []

        for ep, gt in cases:
            sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
            sample = EV.build_inputs(volumes, ep, norm_factory(sv))
            try:
                neg_p, pos_p = fwd(model, sample, use_skel)
            except Exception as e:
                print(f"  forward failed: {e}")
                break

            s = prompt_stats(pos_p, neg_p, gt)
            for k in st:
                st[k].append(s[k])

            predictor.set_image(EV.sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
            o_pos, o_neg = oracle_prompts(gt, args.n_pos, args.n_neg,
                                          rng=random.Random(args.seed))

            d_full.append(dice_of(predictor, pos_p, neg_p, gt))
            sw_p = EV._as_points(pos_p)[:, ::-1].copy()
            sw_n = EV._as_points(neg_p)[:, ::-1].copy()
            d_swap.append(dice_of(predictor, sw_p, sw_n, gt))
            d_oracle.append(dice_of(predictor, o_pos, o_neg, gt))
            d_mpos_oneg.append(dice_of(predictor, pos_p, o_neg, gt))
            d_opos_mneg.append(dice_of(predictor, o_pos, neg_p, gt))

        if not d_full:
            continue

        print(f"  prompts per episode      : pos={np.mean(st['n_pos']):.1f}  "
              f"neg={np.mean(st['n_neg']):.1f}")
        print(f"  out-of-bounds points     : pos={int(np.sum(st['pos_oob']))}  "
              f"neg={int(np.sum(st['neg_oob']))}")
        print(f"  POSITIVE inside GT       : {np.nanmean(st['pos_inside_gt']):.1%}   "
              f"(want > 80%)")
        print(f"  NEGATIVE outside GT      : {np.nanmean(st['neg_outside_gt']):.1%}   "
              f"(want > 90%)")
        print(f"  NEGATIVE dist to organ   : {np.nanmean(st['neg_dist_to_organ']):.1f} px  "
              f"(FoB targets ~15)")
        print()
        print(f"  [3] oracle_pos + oracle_neg : {np.mean(d_oracle):.4f}   <- pipeline ceiling")
        print(f"  [3] model_pos  + oracle_neg : {np.mean(d_mpos_oneg):.4f}   <- tests POS head")
        print(f"  [3] oracle_pos + model_neg  : {np.mean(d_opos_mneg):.4f}   <- tests NEG head")
        print(f"  [3] model_pos  + model_neg  : {np.mean(d_full):.4f}   <- full model")
        print(f"  [2] coords swapped (y,x)    : {np.mean(d_swap):.4f}")
        if np.mean(d_swap) > np.mean(d_full) + 0.10:
            print("      !! SWAP IS BETTER -> this fork emits (row, col); transpose in eval.py")

        rows[tag] = {
            "pos_inside_gt": float(np.nanmean(st["pos_inside_gt"])),
            "neg_outside_gt": float(np.nanmean(st["neg_outside_gt"])),
            "neg_dist_to_organ": float(np.nanmean(st["neg_dist_to_organ"])),
            "n_pos": float(np.mean(st["n_pos"])), "n_neg": float(np.mean(st["n_neg"])),
            "dice_oracle": float(np.mean(d_oracle)),
            "dice_modelpos_oracleneg": float(np.mean(d_mpos_oneg)),
            "dice_oraclepos_modelneg": float(np.mean(d_opos_mneg)),
            "dice_full": float(np.mean(d_full)),
            "dice_coords_swapped": float(np.mean(d_swap)),
        }
    report["models"] = rows

    # ---------------- stage 4: missing-key impact --------------------------
    if base_missing and base_ckpt:
        print("\n" + "=" * 72)
        print("STAGE 4 - does the randomly-initialised baseline layer matter?")
        print("=" * 72)
        ep, gt = cases[0]
        sv = volumes[ep["support_vol"]]
        sample = EV.build_inputs(volumes, ep, base_norm(sv))
        neg_a, pos_a = fwd(base, sample, False)

        sd = base.state_dict()
        saved = {k: sd[k].clone() for k in base_missing if k in sd}
        with torch.no_grad():
            for k in saved:
                sd[k].normal_(mean=0.0, std=float(saved[k].std().item() or 0.02) * 3.0)
        neg_b, pos_b = fwd(base, sample, False)
        with torch.no_grad():
            for k, v in saved.items():
                sd[k].copy_(v)

        dp = float(np.abs(EV._as_points(pos_a) - EV._as_points(pos_b)).max()) \
            if len(EV._as_points(pos_a)) == len(EV._as_points(pos_b)) else float("inf")
        dn = float(np.abs(EV._as_points(neg_a) - EV._as_points(neg_b)).max()) \
            if len(EV._as_points(neg_a)) == len(EV._as_points(neg_b)) else float("inf")
        print(f"  max prompt shift after re-randomising {base_missing}:")
        print(f"    positives: {dp:.3f} px    negatives: {dn:.3f} px")
        changed = (dp > 1e-3) or (dn > 1e-3)
        report["missing_key_affects_baseline"] = bool(changed)
        if changed:
            print("  !! THE MISSING LAYER IS IN THE BASELINE FORWARD PATH.")
            print("     Your 'baseline FoB' contains a random layer, so it is NOT published")
            print("     FoB and every baseline number is invalid. Evaluate the baseline with")
            print("     a pristine upstream clone of FoB_SAM instead.")
        else:
            print("  OK - the missing layer does not affect the baseline's output.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out_json}")


if __name__ == "__main__":
    main()
