import sys
import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from experiments.eval import load_volumes, available_organs, sample_episode, build_inputs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    global_params_path = os.path.join(_ROOT, "checkpoints", "allocator_global_params.json")
    with open(global_params_path, "r") as f:
        params = json.load(f)
    print(f"Loaded Global Params: {params}")

    model = FewShotSeg().to(device)
    allocator = PromptBudgetAllocator(
        model, max_points=10,
        a0=params['a0'], scale=params['scale']
    )
    model.allocator = allocator
    model.eval()
    
    target_organs = {1: "SPLEEN", 2: "RK", 3: "LK", 6: "LIVER"}
    VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10]
    
    budget_stats = {o: [] for o in target_organs.keys()}

    with torch.no_grad():
        for fold in range(5):
            _, val_vols, _, val_labels = load_volumes(fold)
            for organ_id in target_organs.keys():
                for vi, vol in enumerate(val_vols):
                    label = val_labels[vi]
                    if organ_id not in np.unique(label):
                        continue
                        
                    sample = sample_episode(vol, label, organ_id, n_pos=5, seed=vi*100+organ_id)
                    if sample is None:
                        continue
                        
                    support_images, support_fg_labels, query_images, query_labels = build_inputs(sample, device)
                    
                    for j in range(query_images.shape[1]):
                        q_img_s = query_images[[0], [j]]
                        q_lbl_s = query_labels[[0], [j]]
                        if q_lbl_s.sum() == 0:
                            continue
                            
                        try:
                            model(
                                [support_images], [support_fg_labels],
                                [q_img_s], q_lbl_s, None
                            )
                            slice_budget = getattr(model.allocator, 'last_budget', 10)
                            budget_np = min(VALID_NPS, key=lambda x: abs(x - slice_budget))
                            budget_stats[organ_id].append(budget_np)
                        except:
                            pass

    print("\n" + "="*50)
    print("Mean Adaptive Budget (Np) Per Organ")
    print("="*50)
    organ_means = {}
    for o, name in target_organs.items():
        arr = budget_stats[o]
        if len(arr) > 0:
            mean_val = np.mean(arr)
            organ_means[o] = round(mean_val)
            print(f"{name} (label={o}): Mean Np = {mean_val:.2f} (from {len(arr)} slices) -> Rounded to {round(mean_val)}")
        else:
            print(f"{name} (label={o}): No data")

if __name__ == "__main__":
    main()
