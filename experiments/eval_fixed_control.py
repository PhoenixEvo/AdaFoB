import sys
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from segment_anything import sam_model_registry, SamPredictor
from models.lora import LoRA_Sam
from experiments.eval import (
    load_volumes, available_organs, sample_episode, build_inputs,
    compute_volume_dice, compute_hd95, find_path, predict_with_sam, run_model
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lora_ckpt_dir", type=str, required=True)
    parser.add_argument("--np_spleen", type=int, required=True)
    parser.add_argument("--np_rk", type=int, required=True)
    parser.add_argument("--np_lk", type=int, required=True)
    parser.add_argument("--np_liver", type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fixed_budgets = {
        1: args.np_spleen,
        2: args.np_rk,
        3: args.np_lk,
        6: args.np_liver
    }

    SAM_B_CKPT = os.path.join(_ROOT, "checkpoints", "sam_vit_b_01ec64.pth")
    if not os.path.exists(SAM_B_CKPT):
        SAM_B_CKPT = find_path("sam_vit_b_01ec64.pth", is_file=True)
        
    SAM_H_CKPT = os.path.join(_ROOT, "checkpoints", "sam_vit_h_4b8939.pth")
    if not os.path.exists(SAM_H_CKPT):
        SAM_H_CKPT = find_path("sam_vit_h_4b8939.pth", is_file=True)

    sam_h = sam_model_registry["vit_h"](checkpoint=SAM_H_CKPT)
    predictor_h = SamPredictor(sam_h)
    predictor_h.model.to(device)

    sam_b = sam_model_registry["vit_b"](checkpoint=SAM_B_CKPT)
    lora_model = LoRA_Sam(sam_b, r=4, lora_alpha=8)
    
    results_data = []

    for fold in range(5):
        print(f"\n{'='*70}\nEVAL Fixed Control: Fold {fold} (GPU {args.gpu})\n{'='*70}")
        ckpt_path = os.path.join(args.lora_ckpt_dir, f"lora_fold{fold}_best.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            lora_model.load_lora_parameters(state_dict)
            print(f"[LoRA-SAM] Loaded LoRA checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: LoRA checkpoint not found: {ckpt_path}")
            
        predictor_b = SamPredictor(lora_model.sam)
        lora_model.sam.to(device)
        
        _, val_vols, val_spacings, val_labels = load_volumes(fold)
        
        for organ_id, target_np in fixed_budgets.items():
            for vi, vol in enumerate(val_vols):
                label = val_labels[vi]
                if organ_id not in np.unique(label):
                    continue
                    
                sample = sample_episode(vol, label, organ_id, n_pos=5, seed=vi*100+organ_id)
                if sample is None:
                    continue
                    
                support_images, support_fg_labels, query_images, query_labels = build_inputs(sample, device)
                
                # Get identical points as baseline
                uni_neg_p, pos_p = run_model(
                    predictor_h, None, support_images, support_fg_labels,
                    query_images, query_labels, return_pts=True, n_neg=24
                )
                
                pred_vol = np.zeros_like(vol, dtype=np.uint8)
                
                q_imgs = sample['query_images']
                for j in range(len(q_imgs)):
                    img_t = q_imgs[j].permute(1, 2, 0).numpy()
                    img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)
                    lbl_t = sample['query_fg_labels'][j].numpy()
                    
                    if lbl_t.sum() == 0:
                        continue
                        
                    pos_arr = np.array(pos_p[j]).reshape(-1, 2)
                    uni_neg = np.array(uni_neg_p[j]).reshape(-1, 2)
                    
                    predictor_b.set_image(img_t)
                    pred_vol[j] = predict_with_sam(predictor_b, pos_arr, uni_neg[:target_np], img_t.shape)

                gt = (label == organ_id).astype(np.uint8)
                spacing = val_spacings[vi] if val_spacings is not None else (1.0, 1.0, 1.0)
                d = compute_volume_dice(pred_vol, gt)
                h = compute_volume_hd95(pred_vol, gt, spacing)
                
                results_data.append({
                    "fold": fold,
                    "organ": available_organs()[organ_id],
                    "vol_id": vi,
                    "dice_fixed_control": d,
                    "hd95_fixed_control": h
                })
                print(f"Organ {organ_id} Vol {vi}: Fixed(Np={target_np}) Dice={d*100:.2f}%")
                
    df = pd.DataFrame(results_data)
    out_csv = os.path.join(_ROOT, "results", f"control_fixed_mean_budget_gpu{args.gpu}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    main()
