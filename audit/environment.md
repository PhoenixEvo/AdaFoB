# Environment and Execution Infrastructure Audit

Date: 2026-10-21
Auditor: Lead Research Engineer & Independent Methodological Auditor

## 1. Version Control and Working State
- **Repository Commit (HEAD):** `88914bc216595926fc10d070d8bb47cc1908acd0`
- **Branch:** `main` (ahead of `origin/main` by 1 commit)
- **Vendored Submodules / Upstream:**
  - `third_party/FoB_SAM_upstream`: Official FoB_SAM codebase vendored without `.git` metadata.
  - `third_party/FoB_SAM`: Modified project fork containing AdaFoB allocator and modified FoB model.
  - `third_party/segment-anything`: Official Meta SAM repository.
  - `third_party/SSL_ALPNet`: Empty submodule reference (no `.gitmodules` mapping found).

## 2. Local Hardware & Software Stack
- **Operating System:** Windows (Shell: PowerShell)
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
  - Dedicated VRAM: 6,141 MiB (6.0 GB)
  - Driver / CUDA: CUDA 12.8 support
- **Python Version:** 3.11.9
- **PyTorch Version:** 2.8.0+cu128 (`torch.cuda.is_available() == True`)
- **Key Local Packages:** `SimpleITK`, `numpy`, `scipy`, `pandas`, `torch`, `torchvision`, `cv2`, `medpy`
- **Local Data Presence:**
  - `data/` contains data loaders and documentation (`preprocess.py`, `README.md`, `download_instructions.md`).
  - Raw/Normalized medical volumes (`sabs_CT_normalized`) are **NOT stored locally** on this development machine.

## 3. Remote / Production Execution Environment
- **Platform:** Kaggle Notebooks
- **Compute Instance:** Dual GPU (2x NVIDIA Tesla T4, 16,384 MiB VRAM each; Total 32 GB VRAM)
- **Session Constraints:**
  - Maximum runtime limit: 12.0 hours per interactive / batch commit.
  - Non-persistent disk: Scratch outputs are lost unless committed to a Kaggle Output Dataset or pushed to GitHub.
- **Remote Dataset Locations:**
  - Abd-CT (SABS): `/kaggle/input/datasets/nhatphatnguyen/adafob-env/sabs_CT_normalized/`
  - Upstream FoB Checkpoints: HuggingFace `PrimeBo1/FoB_SAM` (`exps_train_on_SABS_FSMIS_FoB.zip`)
  - SAM Base Checkpoint: `/kaggle/working/sam_vit_b_01ec64.pth`
  - SAM Huge Checkpoint: `/kaggle/working/sam_vit_h_4b8939.pth`
