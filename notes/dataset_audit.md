# Dataset Audit

## a. Abd-CT (SABS) and Abd-MRI (CHAOS T2-SPIR)
- **Abd-CT (SABS):** Beyond the Cranial Vault (Synapse).
  - URL: https://www.synapse.org/#!Synapse:syn3193805
  - Requirements: Requires Synapse registration and signing a data use agreement.
  - SSL-ALPNet Protocol: CT images are windowed to HU [-125, 275] and then normalized to [0, 1].
- **Abd-MRI (CHAOS T2-SPIR):** CHAOS challenge.
  - URL: https://chaos.grand-challenge.org/
  - Requirements: Requires Grand-Challenge registration.
  - SSL-ALPNet Protocol: MRI images are z-score normalized per volume.
- **Both Datasets Protocol:**
  - Slices are resized to 256x256.
  - 5-fold Cross-Validation splits.
  - Folder structure expected after preprocessing:
    ```
    data/CHAOST2/chaos_MR_T2_normalized/
    data/SABS/sabs_CT_normalized/
    ```
  - Evaluation uses Setting I (images) and Setting II (supervoxels/SSL).

## b. BraTS2020 (Irregular-Structure Benchmark)
- What is needed: BraTS provides 3D multimodal MRI (T1, T1c, T2, FLAIR) and tumor masks.
- To use in FoB's framework, we must extract 2D slices containing the tumor core mask.
- Slices need to be resized/padded to 256x256 and intensity-normalized similar to the MRI protocol (z-score per volume).
- Data must be structured into image and label `.nii.gz` pairs per slice or volume for compatibility with the existing dataloaders.

## c. DRIVE
- Note: Reserved for a cross-domain appendix experiment only. No immediate integration required.
