"""
LoRA Adapter for SAM (Segment Anything Model)
==============================================
Implements Low-Rank Adaptation (LoRA) for SAM's ViT image encoder,
following the SAMed architecture (Zhang & Liu, arXiv:2304.13785).

Only the Q and V projections in each transformer block's attention
layer receive LoRA adapters. The mask decoder is fully trainable.
All other parameters remain frozen.

Usage:
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, r=4)
    # lora_sam.sam is the modified SAM model
    # lora_sam.get_trainable_parameters() returns optimizer param groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


class _LoRA_qkv(nn.Module):
    """Drop-in replacement for SAM's fused QKV Linear layer with LoRA on Q and V.

    SAM's Attention module has a single nn.Linear that produces Q, K, V
    concatenated along the last dimension: output shape (..., 3*dim).
    This wrapper adds low-rank perturbations ΔQ = B_q @ A_q and ΔV = B_v @ A_v
    while keeping the original QKV weights frozen.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Linear,
        linear_b_q: nn.Linear,
        linear_a_v: nn.Linear,
        linear_b_v: nn.Linear,
    ):
        super().__init__()
        self.qkv = qkv  # Frozen original
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen QKV: (..., 3*dim)
        qkv = self.qkv(x)
        # LoRA perturbations for Q and V
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # Non-in-place addition (autograd safe)
        q = qkv[..., : self.dim] + new_q
        k = qkv[..., self.dim : 2 * self.dim]
        v = qkv[..., 2 * self.dim :] + new_v
        return torch.cat([q, k, v], dim=-1)


class LoRA_Sam(nn.Module):
    """Wraps a SAM model with LoRA adapters injected into the image encoder.

    Architecture:
        - Image Encoder (ViT): FROZEN backbone + TRAINABLE LoRA on Q,V
        - Prompt Encoder: TRAINABLE (tiny, ~7K params)
        - Mask Decoder: FULLY TRAINABLE (~4M params)

    Args:
        sam_model: A SAM model instance from sam_model_registry.
        r: LoRA rank (default: 4, optimal for medical CT per SAMed ablation).
        lora_alpha: LoRA scaling factor (default: 2*r = 8).
        lora_dropout: Dropout on LoRA A matrices (default: 0.05).
    """

    def __init__(self, sam_model, r: int = 4, lora_alpha: int = 8, lora_dropout: float = 0.05):
        super().__init__()
        self.sam = sam_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # ── Step 1: Freeze everything ──────────────────────────────────────
        for param in self.sam.parameters():
            param.requires_grad = False

        # ── Step 2: Inject LoRA into each ViT block's attention QKV ────────
        self.lora_layers = []  # Track for saving/loading
        self.A_weights = nn.ParameterList()
        self.B_weights = nn.ParameterList()

        for i_block, block in enumerate(self.sam.image_encoder.blocks):
            attn = block.attn
            dim = attn.qkv.in_features

            # LoRA matrices: A down-projects (dim → r), B up-projects (r → dim)
            linear_a_q = nn.Linear(dim, r, bias=False)
            linear_b_q = nn.Linear(r, dim, bias=False)
            linear_a_v = nn.Linear(dim, r, bias=False)
            linear_b_v = nn.Linear(r, dim, bias=False)

            # Initialization: A with Kaiming, B with zeros (LoRA starts as identity)
            nn.init.kaiming_uniform_(linear_a_q.weight, a=np.sqrt(5))
            nn.init.zeros_(linear_b_q.weight)
            nn.init.kaiming_uniform_(linear_a_v.weight, a=np.sqrt(5))
            nn.init.zeros_(linear_b_v.weight)

            # Optional dropout on A
            if lora_dropout > 0:
                linear_a_q = nn.Sequential(nn.Dropout(lora_dropout), linear_a_q)
                linear_a_v = nn.Sequential(nn.Dropout(lora_dropout), linear_a_v)

            # Track parameters
            self.A_weights.extend([
                linear_a_q[-1].weight if isinstance(linear_a_q, nn.Sequential) else linear_a_q.weight,
                linear_a_v[-1].weight if isinstance(linear_a_v, nn.Sequential) else linear_a_v.weight,
            ])
            self.B_weights.extend([linear_b_q.weight, linear_b_v.weight])

            # Replace the QKV projection with LoRA-augmented version
            lora_qkv = _LoRA_qkv(
                attn.qkv,
                linear_a_q,
                linear_b_q,
                linear_a_v,
                linear_b_v,
            )
            block.attn.qkv = lora_qkv
            self.lora_layers.append(lora_qkv)

        # ── Step 3: Unfreeze mask decoder and prompt encoder ───────────────
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True

        # Count trainable params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LoRA-SAM] Total params: {total:,}")
        print(f"[LoRA-SAM] Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        lora_only = sum(p.numel() for p in self.A_weights) + sum(p.numel() for p in self.B_weights)
        print(f"[LoRA-SAM] LoRA params: {lora_only:,}")

    def forward(self, batched_input, multimask_output: bool = True):
        """Forward pass through the LoRA-adapted SAM model.

        This replicates SAM's forward but allows gradients through LoRA.

        Args:
            batched_input: List of dicts, each with:
                - 'image': (3, H, W) tensor, preprocessed
                - 'point_coords': (N, 2) tensor of point prompts
                - 'point_labels': (N,) tensor of point labels
                - 'original_size': (H, W) tuple
            multimask_output: Whether to return 3 mask candidates.

        Returns:
            List of dicts with 'masks', 'iou_predictions', 'low_res_logits'.
        """
        outputs = []
        for inp in batched_input:
            image = inp['image'].unsqueeze(0)  # (1, 3, H, W)

            # Image embedding (through LoRA-adapted encoder)
            image_embedding = self.sam.image_encoder(image)

            # Prompt encoding
            points = (inp['point_coords'].unsqueeze(0), inp['point_labels'].unsqueeze(0))
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

            # Mask decoding
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # Upsample to input resolution (1024x1024)
            masks = F.interpolate(
                low_res_masks,
                (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )

            outputs.append({
                'masks': masks.squeeze(0),           # (num_masks, 1024, 1024)
                'iou_predictions': iou_predictions.squeeze(0),  # (num_masks,)
                'low_res_logits': low_res_masks.squeeze(0),     # (num_masks, 256, 256)
            })

        return outputs

    def get_trainable_parameters(self):
        """Returns parameter groups for the optimizer.

        Uses differential learning rates:
        - LoRA params: higher LR (they're randomly initialized)
        - Mask decoder: standard LR (pre-trained weights being fine-tuned)
        """
        lora_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'A_weights' in name or 'B_weights' in name or 'lora' in name.lower():
                lora_params.append(param)
            else:
                decoder_params.append(param)

        # Also collect LoRA module parameters not in A/B lists
        for layer in self.lora_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad and param not in lora_params:
                    lora_params.append(param)

        # Deduplicate
        seen = set()
        unique_lora = []
        for p in lora_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_lora.append(p)
        unique_decoder = []
        for p in decoder_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_decoder.append(p)

        return [
            {'params': unique_lora, 'lr_scale': 1.0, 'name': 'lora'},
            {'params': unique_decoder, 'lr_scale': 1.0, 'name': 'decoder'},
        ]

    def save_lora_parameters(self, filepath: str):
        """Save only LoRA weights + mask decoder weights (~17MB for ViT-B, r=4)."""
        state_dict = {}

        # LoRA weights
        for i, layer in enumerate(self.lora_layers):
            state_dict[f'lora.{i}.a_q'] = layer.linear_a_q.state_dict()
            state_dict[f'lora.{i}.b_q'] = layer.linear_b_q.state_dict()
            state_dict[f'lora.{i}.a_v'] = layer.linear_a_v.state_dict()
            state_dict[f'lora.{i}.b_v'] = layer.linear_b_v.state_dict()

        # Mask decoder weights
        state_dict['mask_decoder'] = self.sam.mask_decoder.state_dict()

        # Prompt encoder weights
        state_dict['prompt_encoder'] = self.sam.prompt_encoder.state_dict()

        # Metadata
        state_dict['meta'] = {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'num_blocks': len(self.lora_layers),
        }

        torch.save(state_dict, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[LoRA-SAM] Saved LoRA checkpoint: {filepath} ({size_mb:.1f} MB)")

    def load_lora_parameters(self, filepath: str):
        """Load LoRA weights + mask decoder weights from checkpoint."""
        state_dict = torch.load(filepath, map_location='cpu')

        # Verify compatibility
        meta = state_dict.get('meta', {})
        assert meta.get('r', self.r) == self.r, \
            f"Rank mismatch: checkpoint r={meta['r']}, model r={self.r}"

        # Load LoRA weights
        for i, layer in enumerate(self.lora_layers):
            layer.linear_a_q.load_state_dict(state_dict[f'lora.{i}.a_q'])
            layer.linear_b_q.load_state_dict(state_dict[f'lora.{i}.b_q'])
            layer.linear_a_v.load_state_dict(state_dict[f'lora.{i}.a_v'])
            layer.linear_b_v.load_state_dict(state_dict[f'lora.{i}.b_v'])

        # Load decoder and prompt encoder
        self.sam.mask_decoder.load_state_dict(state_dict['mask_decoder'])
        if 'prompt_encoder' in state_dict:
            self.sam.prompt_encoder.load_state_dict(state_dict['prompt_encoder'])

        print(f"[LoRA-SAM] Loaded LoRA checkpoint: {filepath}")


import os  # needed for save_lora_parameters
