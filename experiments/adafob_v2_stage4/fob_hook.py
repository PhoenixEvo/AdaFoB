"""Read-only hook around FoB's ResNet-101 encoder to capture coarse prior and medical features."""
from __future__ import annotations

import numpy as np

from analysis.rebaseline_v1.common import generate


def generate_with_fob_prior(
    model,
    case: dict,
    native_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FoB without altering (neg, pos) coordinates while extracting medical prior & features.

    Returns:
        neg: (N, 2) float32 negative coordinates matching generate(model, case) bit-for-bit.
        pos: (M, 2) float32 positive coordinates matching generate(model, case) bit-for-bit.
        p_fob_hw: (H, W) float32 foreground probability map in [0, 1] on native_hw.
        fob_supp_feat: (C, Hf, Wf) float32 FoB medical encoder features for the support slice.
        fob_qry_feat: (C, Hf, Wf) float32 FoB medical encoder features for the query slice.
    """
    import torch
    import torch.nn.functional as F

    captured: dict[str, torch.Tensor] = {}

    def _enc_hook(module, inputs, output):
        captured["img_fts"] = output[0].detach()
        captured["tao"] = output[1].detach()

    handle = model.encoder.register_forward_hook(_enc_hook)
    try:
        neg, pos = generate(model, case)
    finally:
        handle.remove()

    h, w = int(native_hw[0]), int(native_hw[1])
    if (
        case["supp"].max() == 0
        or case["support_mask"].max() == 0
        or "img_fts" not in captured
        or captured["img_fts"].shape[0] < 2
    ):
        feat_dim = int(getattr(model, "feature_dim", 512))
        return (
            neg,
            pos,
            np.zeros((h, w), dtype=np.float32),
            np.zeros((feat_dim, 64, 64), dtype=np.float32),
            np.zeros((feat_dim, 64, 64), dtype=np.float32),
        )

    img_fts = captured["img_fts"]
    tao = captured["tao"]
    with torch.inference_mode():
        supp_mask_t = torch.from_numpy(np.asarray(case["support_mask"], dtype=np.float32)).to(
            device=img_fts.device
        )
        supp_mask_256 = model.pre_process(supp_mask_t)
        spt_feat = model.getFeatures(img_fts[0:1], supp_mask_256)
        spt_fg_proto = model.getPrototype([[spt_feat]])[0]
        qry_pred = model.getPred(img_fts[1:2], spt_fg_proto[0], tao[1:])
        p_native = F.interpolate(qry_pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        p_fob_hw = p_native.detach().cpu().numpy().astype(np.float32)
        fob_supp_feat = img_fts[0].detach().cpu().numpy().astype(np.float32)
        fob_qry_feat = img_fts[1].detach().cpu().numpy().astype(np.float32)

    return neg, pos, p_fob_hw, fob_supp_feat, fob_qry_feat
