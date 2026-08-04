# AdaFoB: Adaptive Background-Prompt Generation

This is the research repository for AdaFoB, extending FoB (Focus on Background) for few-shot medical image segmentation.

## Project Structure
- `configs/`: YAML configurations for experiments.
- `data/`: Dataset storage (ignored by git). See `data/download_instructions.md`.
- `notes/`: Audit logs and research notes.
- `third_party/`: External codebases cloned for reference (FoB_SAM, segment-anything, SSL_ALPNet).

## Environment Setup
Run `conda env create -f environment.yml` and `conda activate adafob` to set up the environment.
