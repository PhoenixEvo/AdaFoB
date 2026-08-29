"""
Extract Mean Adaptive Budget Per Organ
=======================================
Runs AdaFoB inference (using FoB model with PromptBudgetAllocator) across
all 5 folds to compute the empirical mean Np allocated per organ.

Uses the same eval pipeline as eval_lora.py for consistency.

Output: Printed mean Np values (to be used as --np_X args in eval_fixed_control.py)
"""
import os
import sys
import json
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "/kaggle/working/AdaFoB" not in sys.path:
    sys.path.insert(0, "/kaggle/working/AdaFoB")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "FoB_SAM"))
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "segment-anything"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names
from torch.utils.data import DataLoader

from experiments.eval_lora import (
    find_path, load_fob_checkpoint,
    NORMALIZED_DIR, CKPT_DIR, GLOBAL_PARAMS_PATH,
    TEST_LABELS, N_PART, SUPP_IDX, VALID_NPS
)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    global_params = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    if os.path.exists(GLOBAL_PARAMS_PATH):
        with open(GLOBAL_PARAMS_PATH, "r") as f:
            global_params.update(json.load(f))
        print(f"Loaded Global Params: {global_params}")

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    organ_names = {1: "SPLEEN", 2: "RK", 3: "LK", 6: "LIVER"}
    budget_stats = {o: [] for o in TEST_LABELS}

    for eval_fold in range(5):
        print(f"\n--- Fold {eval_fold} ---")

        ckpt_path = load_fob_checkpoint(eval_fold, CKPT_DIR)

        class DummyArgs:
            pass
        args_fob = DummyArgs()
        args_fob.dataset = "SABS"
        args_fob.max_points = 24

        model = FewShotSeg(args_fob)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model.eval()

        data_config = {
            'data_dir': data_dir, 'dataset': 'SABS', 'n_shot': 1, 'n_way': 1, 'n_query': 1,
            'n_sv': 5000, 'max_iter': 3000, 'eval_fold': eval_fold, 'min_size': 200,
            'max_slices': 3, 'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True, drop_last=False,
        )

        for label_val, label_name in labels.items():
            if label_name == 'BG' or label_val not in TEST_LABELS:
                continue

            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda()
                             for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda()
                               for i in range(support_sample['image'].shape[0])]

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    query_image = [sample['image'][j].float().cuda()
                                   for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()
                    C_q = sample['image'].shape[1]
                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')

                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            # Setup allocator
                            allocator = PromptBudgetAllocator(max_points=24).cuda()
                            allocator.nu = global_params.get("nu", 0.02)
                            allocator.lam = global_params.get("lam", 0.5)
                            allocator.a0 = global_params.get("a0", 0.6)
                            allocator.tau = global_params.get("tau", 0.05)
                            model.allocator = allocator

                            original_allocate = model.allocator.allocate
                            def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto,
                                                   supp_m, mdl, supp_fts):
                                a, contours, M_tilde = mdl.allocator.get_ambiguity_score(
                                    qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                                budget = mdl.allocator.compute_budget(a, contours, M_tilde)
                                mdl.allocator.last_budget = budget
                                r = mdl.allocator.get_scale_adaptive_offset(M_tilde)
                                pts = mdl.allocator.sample_placement(
                                    qry_img, M_tilde, contours, 24, r)
                                return pts, budget
                            model.allocator.allocate = capturing_allocate

                            try:
                                model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                slice_budget = getattr(model.allocator, 'last_budget', 10)
                                budget_np = min(VALID_NPS, key=lambda x: abs(x - slice_budget))
                                budget_stats[label_val].append(budget_np)
                            except:
                                pass

            n_slices = len(budget_stats[label_val])
            print(f"  {label_name}: {n_slices} slices collected so far")

    # Print final results
    print("\n" + "="*60)
    print("MEAN ADAPTIVE BUDGET (Np) PER ORGAN")
    print("="*60)
    print(f"{'Organ':<10} {'Mean Np':>10} {'Std':>8} {'Median':>8} {'Rounded':>10} {'N slices':>10}")
    for o in TEST_LABELS:
        arr = budget_stats[o]
        if len(arr) > 0:
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            med_val = np.median(arr)
            rounded = round(mean_val)
            print(f"{organ_names[o]:<10} {mean_val:>10.2f} {std_val:>8.2f} {med_val:>8.1f} {rounded:>10} {len(arr):>10}")
        else:
            print(f"{organ_names[o]:<10} {'NO DATA':>10}")

    print("\n--- Command for eval_fixed_control.py ---")
    parts = []
    for o in TEST_LABELS:
        arr = budget_stats[o]
        r = round(np.mean(arr)) if len(arr) > 0 else 3
        name = organ_names[o].lower()
        parts.append(f"--np_{name} {r}")
    print(f"python experiments/eval_fixed_control.py --gpu 0 {' '.join(parts)}")


if __name__ == "__main__":
    main()
