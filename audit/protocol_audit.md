# Protocol Audit: Setting I Compliance & Integrity Verification

Date: 2026-10-21  
Auditor: Lead Research Engineer & Independent Methodological Auditor

---

## 1. Executive Summary: GATE 1 Decision

**GATE 1 STATUS: BLOCKED**

Do not train improved models or launch new confirmation sweeps until the protocol violations identified below have agreed repairs. 

The investigation has established that the reported performance gains of AdaFoB+LoRA (Dice ~89.0%) over the FoB Baseline (Dice 85.74%) are fundamentally compromised by severe methodological violations:
1. **Target-Class Supervision Leakage (Violation V1):** LoRA fine-tuning directly trained on manual ground-truth masks of the evaluated target organs.
2. **Test-Fold Checkpoint Selection (Violation V2):** The `*_best.pth` checkpoints were chosen based on validation performance on the evaluated test patients themselves.
3. **Test-Set Hyperparameter Overfitting (Violation V3):** The adaptive prompt budget parameters were selected by maximizing Dice on the evaluation test folds.

Under strict Few-Shot Medical Image Segmentation (FSMIS) Setting I rules, the reported AdaFoB+LoRA numbers do not represent novel-class few-shot segmentation; they represent in-domain supervised fine-tuning with test-set model selection.

---

## 2. Setting I Definition & Requirements

Per the primary FoB literature (*Bo et al., CVPR 2026*) and the SSL-ALPNet foundation (*Ouyang et al., TMI 2022*):
- **Setting I Definition:** Target test classes (Spleen, Right Kidney, Left Kidney, Liver) are novel classes during evaluation. During training, the background prompt generator is trained either:
  1. Without any manual semantic masks, utilizing unsupervised 3D supervoxels (`USE_GT=False` in `third_party/FoB_SAM_upstream/scripts/train_SABS.sh`), or
  2. On disjoint base classes, where target organ regions are either unlabeled or excluded.
- **Novel Class Ground Truth:** Under no circumstances may ground-truth segmentation masks of novel classes be utilized for gradient updates, hyperparameter grid search, or model checkpoint selection prior to inference.

---

## 3. Detailed Audit of Violations

### Violation V1: Target Organ Ground Truth in LoRA Training (BLOCKING)
- **Code Evidence:** [`experiments/train_lora.py`](file:///d:/HD-NCKH27/AdaFoB/experiments/train_lora.py) lines 88–91:
  ```python
  ALL_ORGANS = {
      1: "SPLEEN", 2: "RK", 3: "LK", 6: "LIVER",
  }
  ```
  Lines 140–155 sample slices containing these exact organs from the training fold patients, and Line 280 computes binary targets `mask_slice = (lbl_vol[z] == organ_id)`.
- **Consequence:** LoRA adapters and the SAM Mask Decoder learned domain-specific representation of the exact 4 target organs from hundreds of ground-truth annotated slices. This is standard supervised fine-tuning, not few-shot segmentation.
- **Required Repair:** To remain in Setting I, LoRA must either:
  - Option A: Be trained exclusively on unsupervised supervoxels (matching FoB's training pipeline).
  - Option B: Be trained on non-test base organs (e.g. pancreas, gallbladder, stomach, aorta) with test organs masked out.
  - Option C: Be eliminated from the primary claim, restricting the paper's contribution to zero-shot training-free adaptive prompting on frozen SAM.

### Violation V2: Checkpoint Selection on Test-Fold Queries (BLOCKING)
- **Code Evidence:** [`experiments/train_lora.py`](file:///d:/HD-NCKH27/AdaFoB/experiments/train_lora.py) lines 146–147:
  ```python
  if not is_train:
      self.image_paths = [p for i, p in enumerate(all_image_paths) if i in test_indices]
      self.label_paths = [p for i, p in enumerate(all_label_paths) if i in test_indices]
  ```
  Lines 613–621 evaluate validation Dice every 5 epochs on this set and save `lora_fold{fold}_best.pth` when `val_dice > best_dice`.
- **Consequence:** `test_indices` contains the exact query patients evaluated for that fold. Selecting the best checkpoint on the test fold overfits epoch selection to the test set.
- **Required Repair:** Split the training folds into train/validation (e.g. 19 train patients, 4 val patients), or fix the training duration a priori (e.g. fixed 20 epochs without early stopping).

### Violation V3: Allocator Hyperparameters Tuned on Test Dice (BLOCKING)
- **Code Evidence:** [`experiments/r3_3d_global_tune.py`](file:///d:/HD-NCKH27/AdaFoB/experiments/r3_3d_global_tune.py) lines 262–295. While line 266 contains a comment claiming cross-validation, the code actually iterates:
  ```python
  for test_fold in folds:
      test_df = df[df['fold'] == test_fold]
      for _, row in test_df.iterrows():
          assigned_np = compute_budget_simple(mean_a, scale, a0, tau)
  ```
  It averages `test_df` Dice directly across all 5 test folds and selects `best_overall_params = (scale, a0, tau)` that maximizes test Dice.
- **Consequence:** The dynamic prompt allocator was directly tuned to maximize test set performance.
- **Required Repair:** Fit allocator hyperparameters nested strictly within training folds, or use fixed, pre-declared heuristic defaults without test-set tuning.

### Violation V4: Silent Config Key Mismatch (SERIOUS)
- **Code Evidence:** `r3_global_params.json` exports:
  ```json
  {"scale": 6, "a0": 0.3, "tau": 0.02}
  ```
  However, [`experiments/eval_lora.py`](file:///d:/HD-NCKH27/AdaFoB/experiments/eval_lora.py) lines 201–205 and 353–356 read:
  ```python
  allocator.nu = global_params.get("nu", 0.02)
  allocator.lam = global_params.get("lam", 0.5)
  allocator.a0 = global_params.get("a0", 0.6)
  allocator.tau = global_params.get("tau", 0.05)
  ```
- **Consequence:** The deployed evaluator ignored `scale` and silently fell back to hard-coded defaults (`nu=0.02, lam=0.5`), while `a0` and `tau` were not found in `r3_global_params.json` under their expected keys and defaulted to 0.6 and 0.05. The evaluated model was an accidental hybrid never verified by R3.
- **Required Repair:** Implement strict schema validation in configuration loaders that raises fatal exceptions on missing or unexpected keys.

### Violation V5: Fixed-Budget Control Derived Post-Hoc from Test Data (SERIOUS)
- **Code Evidence:** [`experiments/extract_mean_budget.py`](file:///d:/HD-NCKH27/AdaFoB/experiments/extract_mean_budget.py) runs inference on all 5 evaluation folds to compute mean allocated $N_p$ per organ, yielding $\{6, 3, 2, 2\}$.
- **Consequence:** This control was extracted from test outputs. It cannot be presented as a prior baseline.
- **Required Repair:** Label strictly as a post-hoc descriptive control.

### Violation V6: Query Ground Truth in Model Forward Signature (MODERATE)
- **Code Evidence:** [`third_party/FoB_SAM/models/FoB.py`](file:///d:/HD-NCKH27/AdaFoB/third_party/FoB_SAM/models/FoB.py) line 417:
  ```python
  if supp_mask[[0], 0, 0].max() > 0. and qry_labels.max() > 0.:
  ```
- **Consequence:** Query GT is accepted into the forward pass. While at test time `qry_labels.max() > 0` is always true due to slice filtering (V7), passing query labels to inference violates clean software architecture and risks accidental leakage.
- **Required Repair:** Remove `qry_labels` from `FewShotSeg.forward(..., train=False)`.

### Violation V7: Annotated Slice Filtering Assistance (DECLARED PROTOCOL)
- **Code Evidence:** `third_party/FoB_SAM/dataloaders/datasets.py` line 49:
  ```python
  idx = lbl.sum(axis=(1, 2)) > 0
  sample['image'] = torch.from_numpy(img[idx])
  sample['label'] = torch.from_numpy(lbl[idx])
  ```
- **Consequence:** Evaluation is restricted to the axial z-range containing the target organ. False positives in distant slices are ignored.
- **Status:** Established benchmark convention from Ouyang et al., but must be explicitly disclosed in the manuscript when reporting 3D metrics.
