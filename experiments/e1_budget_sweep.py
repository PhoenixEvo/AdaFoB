import sys
import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from segment_anything import sam_model_registry, SamPredictor

# Reuse eval.py functions
from experiments.eval import (
    load_volumes, available_organs, 
    sample_episode, build_inputs,
    run_model, predict_sam_from_points,
    load_checkpoint
)

# Inline make_baseline_norm since it's nested in evaluate() in eval.py
def make_baseline_norm(vol, dataset_mean=35.577, dataset_std=59.635):
    # This is a simplified version of what evaluate() does. 
    # Just returning a function that applies dataset norm.
    return lambda arr: (arr - dataset_mean) / dataset_std

def sam_uint8_from_canonical(img):
    # Normalize img to 0-255 uint8
    if img.max() == img.min():
        return np.zeros((*img.shape, 3), dtype=np.uint8)
    norm = (img - img.min()) / (img.max() - img.min())
    uint8 = (norm * 255).astype(np.uint8)
    return np.stack([uint8, uint8, uint8], axis=-1)

def sweep_episode(predictor, fob, base_sample, Np_list, H, W, dummy_model, qry_img_canonical):
    """Run sweep of Np for a single episode."""
    
    # 1. Run FoB baseline to get 24 negative points (using fixed 24 budget)
    # We will slice these 24 points to get top-N for the sweep.
    fob.allocator = None
    fob.max_points = 24
    
    base_kwargs = {"train": False, "use_skeleton": False, "budget_Np": 24}
    
    try:
        # run_model returns (neg_points, pos_points)
        # We intercept FoB forward pass directly to extract intermediate predictions
        supp_imgs = base_sample['support_images']
        supp_masks = base_sample['support_fg_labels']
        qry_imgs = base_sample['query_images']
        qry_labels = base_sample['query_labels']
        
        # We need the FoB predictions
        neg_p, pos_p = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=24)
        
        # Run PBA allocator just to get Ambiguity Score (a)
        if not hasattr(fob, 'temp_allocator'):
            fob.temp_allocator = PromptBudgetAllocator(max_points=24).cuda()
            
        with torch.no_grad():
            spt_fts = [fob.getFeatures(supp_imgs[0][0], supp_masks[0][0])]
            spt_fg_proto = fob.getPrototype([spt_fts])[0]
            qry_fts = fob.encoder(qry_imgs)
            qry_pred = torch.stack([fob.getPred(qry_fts[0], spt_fg_proto[0], fob.t)], dim=1)
            qry_pred_coarse = F.interpolate(qry_pred, size=qry_imgs.shape[-2:], mode='bilinear', align_corners=True)
            
            # Extract a
            a, _, _ = fob.temp_allocator.get_ambiguity_score(qry_imgs, qry_pred_coarse, spt_fg_proto, supp_masks, fob, fob.encoder(supp_imgs)[0][0])
            
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return None, None
        
    # We have neg_p (shape 1, 24, 2) or (24, 2) and pos_p (10, 2)
    # 2. Predict with SAM for each Np
    results = {}
    
    if neg_p is not None:
        neg_p_all = np.array(neg_p).reshape(-1, 2)
    else:
        neg_p_all = np.zeros((0, 2))
        
    for np_val in Np_list:
        neg_p_subset = neg_p_all[:np_val] if np_val > 0 else np.zeros((0, 2))
        
        try:
            dice, hd95 = predict_sam_from_points(
                predictor, pos_p, neg_p_subset, H, W, mask_select="max_area"
            )
            results[np_val] = {"dice": dice, "hd95": hd95}
        except Exception as e:
            results[np_val] = {"dice": 0.0, "hd95": 100.0}
            
    return a, results

def main():
    print("Running E1 Budget Sweep...")
    device = "cuda"
    
    # Configuration
    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    ckpt_path = "/kaggle/working/baseline_fob/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv2/ckpt.pth"
    out_csv = "results/e1_budget_sweep.csv"
    
    Np_list = [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
    
    # 1. Load data
    volumes, stats = load_volumes(data_root, (-1024, 3072), None)
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver", 11: "pancreas"}
    organs = available_organs(volumes, organ_map, 200)
    
    # 2. Load models
    class DummyArgs:
        pass
    dummy = DummyArgs()
    dummy.n_ways = 1
    dummy.n_shots = 1
    
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(fob, ckpt_path, "FoB baseline", strict=False)
    
    # Add a helper function to PBA to just extract 'a'
    def get_ambiguity_score(self, qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
        import cv2
        M_tilde = (qry_pred_coarse[0, 0] > 0.9).cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(M_tilde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        a_proto = 0.5 * (1 + torch.nn.functional.cosine_similarity(spt_fg_proto, supp_fts.mean(dim=(1,2)).unsqueeze(0), dim=-1).item())
        if len(contours) > 0:
            grad_x = cv2.Sobel(qry_img[0][0].cpu().numpy(), cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(qry_img[0][0].cpu().numpy(), cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            contour_mask = np.zeros_like(M_tilde)
            cv2.drawContours(contour_mask, contours, -1, 1, 1)
            grad_contour = grad_mag[contour_mask == 1].mean() if contour_mask.sum() > 0 else 0
            grad_body = grad_mag[M_tilde == 1].mean() if M_tilde.sum() > 0 else 1e-5
            grad_body = max(grad_body, 1e-5)
            ratio = grad_contour / grad_body
            a_edge = 1.0 - min(ratio / 5.0, 1.0)
        else:
            a_edge = 0.0
        C_map = qry_pred_coarse[0, 0].cpu().numpy()
        a_conf = min(np.logical_and(C_map > 0.5, C_map < 0.9).sum() / (M_tilde.sum() + 1e-5), 1.0)
        a = self.w[0] * a_proto + self.w[1] * a_edge + self.w[2] * a_conf
        return a, contours, M_tilde
        
    PromptBudgetAllocator.get_ambiguity_score = get_ambiguity_score
    
    # 3. Sweep
    results = []
    skipped = 0
    episodes = 250
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fields = ["ep", "organ", "a_score", "optimal_Np", "oracle_dice", "fob_10_dice"] + [f"dice_{n}" for n in Np_list]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for ep_i in tqdm(range(episodes), desc="Sweeping Np"):
            cls = organs[ep_i % len(organs)]
            ep = sample_episode(volumes, cls, 200)
            if ep is None:
                skipped += 1
                continue
                
            sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
            base_sample = build_inputs(volumes, ep, make_baseline_norm(sv))
            
            # Encode SAM image once
            predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
            H, W = base_sample["query_mask_np"].shape
            
            a_score, sweep_res = sweep_episode(predictor, fob, base_sample, Np_list, H, W, dummy, qv["canon"][ep["query_slice"]])
            
            if sweep_res is None:
                skipped += 1
                continue
                
            # Find N* (highest Dice, tie break to smaller Np)
            best_dice = -1.0
            best_np = 0
            for np_val in Np_list:
                if sweep_res[np_val]["dice"] > best_dice + 1e-4:
                    best_dice = sweep_res[np_val]["dice"]
                    best_np = np_val
                    
            row = {
                "ep": ep_i,
                "organ": organ_map[cls],
                "a_score": a_score,
                "optimal_Np": best_np,
                "oracle_dice": best_dice,
                "fob_10_dice": sweep_res.get(10, {"dice": 0.0})["dice"]
            }
            
            for np_val in Np_list:
                row[f"dice_{np_val}"] = sweep_res[np_val]["dice"]
                
            writer.writerow(row)
            f.flush()
            
    print(f"Sweep complete. Wrote {episodes - skipped} valid episodes to {out_csv}")

if __name__ == "__main__":
    main()
