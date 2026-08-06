"""AdaFoB training - fine-tunes the published FoB checkpoint with the GAP module.

WHY THIS WAS REWRITTEN
---------------------
The previous version built `FewShotSeg(args)` with RANDOM prompt heads and
trained for epochs(10) x iters_per_epoch(100) = 1,000 iterations.  FoB's paper
trains 36,000 iterations.  So AdaFoB was a from-scratch network at 2.8% of the
reference schedule -- which is why it scored ~0.01 Dice, far below the baseline.
That is not a scientific result about the GAP module, it is an undertrained net.

Three changes:
  1. --init_from  initialises from the FoB SABS checkpoint, so training is
     FINE-TUNING an 86-Dice model rather than starting from noise.
  2. --freeze_encoder keeps the ResNet-101 trunk fixed, so the few thousand
     iterations that Kaggle affords are all spent on the prompt heads.
  3. Inputs now use the SAME canonical z-score pipeline as the FoB checkpoint
     (data/preprocess.py).  The old [0,1] transform was incompatible with
     FoB-pretrained weights and with eval.py.

Also: volumes are cached in memory (the old `_load_slice` re-read an entire
.nii.gz from disk for every single slice access), plus checkpoint/resume so a
Kaggle session dying at 9-12 h does not lose the run.

    python experiments/train.py --config configs/adafob_abdct.yaml \
        --init_from /kaggle/working/baseline_fob/<...>.pth --iters 6000
"""

import os
import sys
import glob
import json
import time
import random
import argparse

import numpy as np
import torch
import yaml
import SimpleITK as sitk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "third_party", "FoB_SAM")))

from data.preprocess import (                                                   # noqa: E402
    to_canonical, resize_volume, norm_zscore, norm_unit_legacy,
    detect_alignment, apply_label_transform,
)
from models.FoB import FewShotSeg                                                        # noqa: E402


class DummyArgs:
    pass


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_volumes(root, hu_window=(-125.0, 275.0), limit=None):
    """Cache every volume once, in the canonical [0, 255] domain."""
    import re
    imgs, lbls = {}, {}
    for r, dirs, files in os.walk(root, followlinks=True):
        for entry in sorted(files + dirs):
            path = os.path.join(r, entry)
            if os.path.isdir(path):
                inner = [f for f in os.listdir(path) if (f.endswith(".nii") or f.endswith(".nii.gz")) and os.path.isfile(os.path.join(path, f))]
                if not inner:
                    continue
                path = os.path.join(path, inner[0])
            elif not (entry.endswith(".nii") or entry.endswith(".nii.gz")):
                continue
                
            fl = entry.lower()
            m = re.search(r"(\d+)", entry)
            if not m:
                continue
            pid = m.group(1)
            
            # Skip corrupted 0-byte files
            if os.path.getsize(path) == 0:
                print(f"[WARN] Skipping corrupted 0-byte file: {path}")
                continue

            if "label" in fl or "seg" in fl:
                lbls[pid] = path
            elif "image" in fl or "img" in fl or "avg" in fl:
                imgs[pid] = path

    common = sorted(set(imgs) & set(lbls))
    if limit:
        common = common[:limit]
    if not common:
        raise ValueError(f"no image/label pairs under {root}")

    vols = []
    tot_sum = tot_sq = tot_n = 0.0
    for pid in common:
        img = sitk.GetArrayFromImage(sitk.ReadImage(imgs[pid])).astype(np.float64)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbls[pid])).astype(np.int32)
        if img.shape != lbl.shape:
            print(f"  skip {pid}: shape {img.shape} vs {lbl.shape}")
            continue
        img = resize_volume(img, (256, 256), is_label=False)
        lbl = resize_volume(lbl, (256, 256), is_label=True)
        canon, domain = to_canonical(img, hu_window)
        tot_sum += float(canon.sum())
        tot_sq += float((canon.astype(np.float64) ** 2).sum())
        tot_n += canon.size
        vols.append({"canon": canon, "label": lbl, "pid": pid, "domain": domain})

    mean = tot_sum / tot_n
    std = float(np.sqrt(max(tot_sq / tot_n - mean ** 2, 1e-12)))
    print(f"Loaded {len(vols)} volumes | canonical mean={mean:.3f} std={std:.3f} "
          f"| domains={set(v['domain'] for v in vols)}")
    return vols, mean, std


def usable_organs(vols, candidates=(1, 2, 3, 6), min_px=50, min_slices=2):
    ok = []
    for c in candidates:
        n = sum(1 for v in vols
                if ((v["label"] == c).sum(axis=(1, 2)) > min_px).sum() >= min_slices)
        if n:
            ok.append(c)
        print(f"  organ {c}: usable in {n}/{len(vols)} volumes")
    return ok


def sample_episode(vols, organs, norm_fn, n_shot=1, min_px=50, rng=random):
    """One 1-way n-shot episode as FoB-format tensors, or None."""
    cls = rng.choice(organs)
    cands = []
    for vi, v in enumerate(vols):
        valid = np.where((v["label"] == cls).sum(axis=(1, 2)) > min_px)[0]
        if len(valid) >= n_shot + 1:
            cands.append((vi, valid))
    if not cands:
        return None

    vi, valid = cands[rng.randrange(len(cands))]
    picks = rng.sample(list(valid), n_shot + 1)
    supp_slices, q_slice = picks[:-1], picks[-1]
    v = vols[vi]

    supp_imgs, supp_masks = [], []
    for s in supp_slices:
        supp_imgs.append(torch.from_numpy(norm_fn(v["canon"][s:s + 1])).float())
        m = (v["label"][s] == cls).astype(np.float32)
        supp_masks.append(torch.from_numpy(m).unsqueeze(0).float())

    qry = torch.from_numpy(norm_fn(v["canon"][q_slice:q_slice + 1])).float()
    qry_lbl = torch.from_numpy((v["label"][q_slice] == cls).astype(np.int64)).unsqueeze(0)

    return {"support_images": [supp_imgs], "support_fg_labels": [supp_masks],
            "query_images": [qry], "query_labels": qry_lbl, "cls": cls}


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def init_from_checkpoint(model, path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    cleaned = {}
    for k, v in obj.items():
        nk = k
        for pref in ("module.", "_orig_mod."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    keys = set(model.state_dict().keys())
    matched = [k for k in cleaned if k in keys]
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Init from {path}")
    print(f"  matched {len(matched)}/{len(keys)} ({len(matched)/max(len(keys),1):.1%})  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  randomly initialised (new AdaFoB params): {list(missing)[:10]}")
    if len(matched) / max(len(keys), 1) < 0.5:
        raise RuntimeError("initialisation failed - check checkpoint key names")
    return list(missing)


def find_baseline_ckpt():
    hits = glob.glob("/kaggle/working/baseline_fob/**/*.pth", recursive=True)
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/adafob_abdct.yaml")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--init_from", type=str, default="auto",
                    help="FoB checkpoint to fine-tune from; 'auto' finds the HF download, "
                         "'none' trains from scratch (not recommended on a Kaggle budget)")
    ap.add_argument("--freeze_encoder", type=int, default=1)
    ap.add_argument("--iters", type=int, default=None, help="overrides epochs*iters_per_epoch")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out", type=str, default="outputs/checkpoints/adafob_abdct.pth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--norm", choices=["zscore", "unit_legacy"], default="zscore",
                    help="zscore matches the FoB checkpoints; unit_legacy reproduces "
                         "the old [0,1] pipeline and is incompatible with --init_from")
    ap.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0])
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--auto_align", type=int, default=1,
                    help="detect and apply the label transform that makes labels agree "
                         "with images (edge-z). MUST match eval.py or train and test "
                         "will disagree on geometry.")
    ap.add_argument("--force_transform", type=str, default=None)
    ap.add_argument("--force_z_shift", type=int, default=0)
    ap.add_argument("--baseline_only", type=int, default=0, help="Disable allocator for baseline training")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    total_iters = args.iters or cfg.get("epochs", 10) * cfg.get("iters_per_epoch", 100)
    lr = args.lr or cfg.get("lr", 1e-4)
    n_shot = cfg.get("n_shot", 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    root = args.data_root or os.environ.get("ABDCT_ROOT")
    if not root:
        cands = glob.glob("/kaggle/input/**/*abd*ct*", recursive=True)
        root = next((c for c in cands if os.path.isdir(c)), None)
    if not root:
        raise ValueError("set --data_root or ABDCT_ROOT")
    print(f"Data root: {root}")

    vols, ds_mean, ds_std = load_volumes(vols_root := root, tuple(args.hu_window))

    # Image/label alignment -- identical policy to eval.py. Training against
    # misaligned labels teaches the prompt head to fit masks that do not match
    # the images, which no amount of fine-tuning recovers.
    align = {"transform": "identity", "z_shift": 0, "confident": True}
    if args.force_transform:
        align = {"transform": args.force_transform, "z_shift": args.force_z_shift,
                 "confident": True}
        print(f"Alignment FORCED: {align['transform']} z_shift={align['z_shift']:+d}")
    elif args.auto_align:
        print("Detecting image/label alignment...")
        align = detect_alignment(vols, 1)
    if align["transform"] != "identity" or align["z_shift"]:
        for v in vols:
            v["label"] = apply_label_transform(v["label"], align["transform"],
                                               align["z_shift"])
        print(f"  applied: {align['transform']} z_shift={align['z_shift']:+d}")
    if not align.get("confident", True):
        raise SystemExit(
            "Alignment is not trustworthy: no transform makes the labels follow an "
            "image boundary. Fix the data (pairing / per-patient images) before "
            "spending GPU hours. Run experiments/check_alignment.py for detail.")

    print("Organ availability:")
    organs = usable_organs(vols)
    if not organs:
        raise ValueError("no usable organs in the training data")

    if args.norm == "zscore":
        norm_fn = lambda sl: norm_zscore(sl, ds_mean, ds_std)   # noqa: E731
    else:
        norm_fn = norm_unit_legacy
        if args.init_from not in ("none", None):
            print("!! WARNING: unit_legacy inputs do not match FoB-pretrained weights")

    model = FewShotSeg(DummyArgs()).cuda()

    init_path = args.init_from
    if init_path == "auto":
        init_path = find_baseline_ckpt()
        if not init_path:
            print("!! no baseline checkpoint found; run eval.py once to download it, "
                  "or pass --init_from none")
    new_params = []
    if init_path and init_path != "none":
        new_params = init_from_checkpoint(model, init_path)

    if args.freeze_encoder:
        n_frozen = 0
        for name, p in model.named_parameters():
            if name.startswith("refine.") or name in new_params:
                # Always train the AdaFoB-specific modules (refine) and any missing params
                continue
            
            p.requires_grad_(False)
            n_frozen += 1
        print(f"Froze {n_frozen} pretrained parameters (fine-tuning only AdaFoB modules)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train/1e6:.2f}M / {n_all/1e6:.2f}M params")

    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    start_iter = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    resume_path = args.out.replace(".pth", "_last.pth")
    if args.resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_iter = state["iter"]
        print(f"Resumed from iteration {start_iter}")

    if args.baseline_only:
        print("=> Training BASELINE FoB (Allocator disabled)")
        model.allocator = None

    model.train()
    print(f"\nTraining {start_iter} -> {total_iters} iterations (lr={lr})")
    t0 = time.time()
    running, n_run, skipped = 0.0, 0, 0
    log = []

    for it in range(start_iter, total_iters):
        sample = sample_episode(vols, organs, norm_fn, n_shot=n_shot, rng=rng)
        if sample is None:
            skipped += 1
            continue

        supp = [[t.cuda() for t in way] for way in sample["support_images"]]
        smask = [[t.cuda() for t in way] for way in sample["support_fg_labels"]]
        qry = [t.cuda() for t in sample["query_images"]]
        qlbl = sample["query_labels"].cuda()

        optimizer.zero_grad(set_to_none=True)
        try:
            out = model(supp, smask, qry, qlbl, train=True)
        except Exception as e:
            print(f"  iter {it}: forward failed: {e}")
            skipped += 1
            continue

        losses = out if isinstance(out, (tuple, list)) else (out,)
        losses = [l for l in losses if torch.is_tensor(l) and l.numel() == 1]
        loss = sum(losses)
        if not torch.is_tensor(loss) or not loss.requires_grad:
            skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        scheduler.step()

        running += float(loss.item())
        n_run += 1

        if (it + 1) % 50 == 0:
            avg = running / max(n_run, 1)
            el = time.time() - t0
            eta = el / max(it + 1 - start_iter, 1) * (total_iters - it - 1)
            parts = "  ".join(f"{float(l.item()):.4f}" for l in losses)
            print(f"iter {it+1}/{total_iters}  loss={avg:.4f}  [{parts}]  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={el/60:.1f}m  eta={eta/60:.1f}m")
            log.append({"iter": it + 1, "loss": avg})
            running, n_run = 0.0, 0

        if (it + 1) % args.save_every == 0 or (it + 1) == total_iters:
            torch.save(model.state_dict(), args.out)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "iter": it + 1},
                       resume_path)
            with open(args.out.replace(".pth", "_trainlog.json"), "w") as f:
                json.dump({"log": log, "dataset_mean": ds_mean, "dataset_std": ds_std,
                           "norm": args.norm, "init_from": init_path,
                           "freeze_encoder": bool(args.freeze_encoder),
                           "total_iters": total_iters, "organs": organs,
                           "alignment": align}, f, indent=2)
            print(f"  saved {args.out} @ iter {it+1}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min ({skipped} episodes skipped)")
    print(f"Checkpoint: {args.out}")
    print(f"IMPORTANT: evaluate with matching normalisation -> "
          f"--adafob_norm {'dataset' if args.norm == 'zscore' else 'train_slice'}")


if __name__ == "__main__":
    train()
