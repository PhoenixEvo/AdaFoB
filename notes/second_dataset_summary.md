# Second Dataset Transferability: Partial Mitigation

## Status: [NOT COMPLETED: Infeasible within compute/time budget]

Fully evaluating AdaFoB+LoRA on a second out-of-domain dataset (e.g., CHAOS-MRI) requires 
significant engineering (MRI data loaders, spacing normalization, intensity clipping) and 
cannot be rigorously completed prior to the MICCAI submission deadline.

Per the task specification constraints, we honestly document this as a limitation 
rather than attempting to suppress or guess the performance.

## Suggested Manuscript Draft (Limitations & Discussion)

> **Cross-Modality Transferability:** While AdaFoB+LoRA achieves highly competitive 
> performance on the SABS CT dataset, its generalization to out-of-domain modalities 
> (e.g., T2-weighted MRI from CHAOS) remains partially validated. The FoB baseline 
> explicitly leverages uniform point sampling optimized for CT intensity profiles. 
> Because our LoRA adaptation bridges the specific domain gap between the SAM ViT-H 
> source distribution and abdominal CT, applying the current checkpoint directly to MRI 
> without an intermediate MRI-specific LoRA fine-tuning phase is expected to yield 
> sub-optimal boundary delineation. Future work will extend the adaptive ambiguity score 
> to trigger dynamic modality-specific LoRA routing.

## Visual Overlay (Partial Mitigation)

A visualization script (`experiments/visualize_overlays.py`) has been provided to 
generate qualitative overlays for the worst-performing slices (e.g., the `image_7` LK 
outlier identified in Experiment 5). 

Reviewers often appreciate visual proof of *why* a model fails in the worst cases 
(e.g., disconnected components vs boundary bleeding) more than a poorly-tuned second 
dataset number. 
