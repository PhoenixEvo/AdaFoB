# Specification: Setting I Evaluation Protocol for Few-Shot Abdominal CT Segmentation

**Primary Sources:**
- Paper: *"Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting"* (Bo et al., CVPR 2026, arXiv:2603.21287)
- Upstream Repository: `https://github.com/primebo1/FoB_SAM` (vendored locally at `third_party/FoB_SAM_upstream/`)
- Reference Lineage: SSL-ALPNet (Ouyang et al., IEEE TMI 2022) / PANet (Wang et al., ICCV 2019)

---

## 1. Protocol Specifications (Items A - L)

### A. Patient-Level Partitions
- **Dataset:** Abd-CT / SABS (Multi-Atlas Abdomen Labeling Challenge, Synapse `syn3193805`).
- **Total Physical Patients:** 30 3D contrast-enhanced abdominal CT scans (`image_0.nii` through `image_29.nii`).
- **5-Fold Cross-Validation Splitting (`dataloaders/dataset_specifics.py` L16-25):**
  - Fold 0: `[0, 1, 2, 3, 4, 5, 6]` (7 patients)
  - Fold 1: `[6, 7, 8, 9, 10, 11, 12]` (7 patients)
  - Fold 2: `[12, 13, 14, 15, 16, 17, 18]` (7 patients)
  - Fold 3: `[18, 19, 20, 21, 22, 23, 24]` (7 patients)
  - Fold 4: `[24, 25, 26, 27, 28, 29, 0]` (7 patients, circular wrap to 0)
- **Training vs. Testing Patient Isolation:** For evaluation on fold $k$, the test set comprises the 7 patients in `FOLD[k]`. All training components may only access the remaining 23–24 patient volumes (`idx not in FOLD[k]`).

### B. Base and Novel Classes
- **Evaluated Target Classes (4 abdominal organs):**
  - Label 1: Spleen
  - Label 2: Right Kidney
  - Label 3: Left Kidney
  - Label 6: Liver
- **Setting I Class Definition:**
  - In strict few-shot segmentation Setting I (inherited from Ouyang et al., TMI 2022), the target classes (Spleen, Kidneys, Liver) are treated as **novel, unseen classes at inference time**.
  - During the training phase, the model is **NOT permitted to access manual ground-truth segmentation masks** of these target classes.

### C. Training, Validation, Support, and Query Roles
- **Training Role:** Volumes from folds $\neq k$. Used for learning background prompt generation (FoB feature extractor, structure priors).
- **Validation Role:** Held-out subset strictly from training folds $\neq k$. Never the test fold.
- **Support Role:** Exactly 1 patient per test fold (`supp_idx = 2`, `datasets.py` L36-37) acts as the 1-shot reference:
  - Fold 0 Support: `image_2.nii`
  - Fold 1 Support: `image_8.nii`
  - Fold 2 Support: `image_14.nii`
  - Fold 3: `image_20.nii`
  - Fold 4: `image_25.nii`
- **Query Role:** The remaining 6 patients in `FOLD[k]` act as test query volumes (total 30 evaluation episodes across 5 folds).

### D. Definition of One "Shot" ($N=1$)
- One shot means a single 3D reference patient scan with manual segmentation mask of the target organ provided at test time.
- To simulate standard clinical interaction, the 3D support volume is sampled into $N_{\text{part}} = 3$ representative 2D slices (`getSupport` in `datasets.py` L55-70) spanning the lower, middle, and upper thirds of the organ.

### E. Support Selection Mechanism
- Fixed deterministic index across all runs: `supp_idx = 2` within each sorted fold's patient list (`datasets.py` L36).
- Support slices are deterministically chosen at uniform depth percentiles ($p = 0.5$ for 1-shot chunk, or $1/(2N)$ intervals).

### F. Query Slices & Volume Reconstruction
- Slices from query volumes containing the target organ are processed sequentially.
- Predicted 2D masks from SAM are stacked into a 3D binary volume $\hat{Y} \in \{0, 1\}^{D \times H \times W}$.
- Evaluation metrics are computed on the reconstructed 3D volume against the 3D ground-truth volume.

### G. Label Access Restrictions by Component
- **FoB Prompt Generator:** In official FoB (`train_SABS.sh`), `USE_GT=False`. The model trains on supervoxels (`sprvxl = sitk.ReadImage(self.sprvxl_dirs[pat_idx])`). It has zero access to manual organ labels during training.
- **LoRA / Decoder Adaptation:** Must adhere to the same restriction: no ground-truth masks of target classes (Spleen, Kidneys, Liver) may be used as training targets.
- **Query Inference:** Zero label access. Query ground-truth masks must never enter the model forward pass, prompt selection, or budget allocation.

### H. Unlabeled Image Usage
- Training images from folds $\neq k$ may contain target organ anatomy in the background, provided no manual ground-truth target mask is used.

### I. Pseudo-labels & Generation
- Pseudo-labels must be unsupervised or self-supervised (e.g. Felzenszwalb 3D supervoxels, SLIC 2D superpixels) computed solely from image voxel intensities without target semantic annotations (`use_gt=False`, `n_sv=5000`).

### J. Checkpoint Selection & Tuning
- Checkpoint selection must be based on a fixed schedule (e.g. 39,000 steps in official FoB) or validation performance on base classes / pseudo-labels on training patients.
- Hyperparameter tuning (budget parameters, thresholds) must be nested strictly within training folds. Test-fold query Dice must never guide parameter selection.

### K. Benchmark-Specific Evaluation Assistance (Annotated Slice Filtering)
- In FoB and SSL-ALPNet (`TestDataset.__getitem__` in `datasets.py` L49-51):
  ```python
  idx = lbl.sum(axis=(1, 2)) > 0
  sample['image'] = torch.from_numpy(img[idx])
  sample['label'] = torch.from_numpy(lbl[idx])
  ```
  Evaluation only iterates through slices where the ground truth is non-empty. Slices outside the axial extent of the organ are excluded from prediction and scoring.
- This protocol assistance is an established convention in the Ouyang et al. benchmark lineage, but **must be explicitly declared** in the methodology, because it restricts evaluation to the known axial bounding slab of the organ.

### L. Metric Definitions & Aggregation
- **3D Volume Dice:**
  $$\text{Dice} = \frac{2 |\hat{Y} \cap Y|}{|\hat{Y}| + |Y|}$$
  Scale is strictly $[0, 1]$ or percentage $[0, 100\%]$.
- **3D HD95 (95th Percentile Hausdorff Distance):**
  Physical boundary distance in millimeters, accounting for anisotropic voxel spacing $(s_x, s_y, s_z)$ via MedPy `hd95(pred, gt, voxelspacing=(sz, sy, sx))`.
- **Primary Aggregation Unit:** Physical volume instance ($N = 120$ volume-organ measurements grouped by $K = 25$ physical subjects). Macro-average over the 4 organs per volume.
