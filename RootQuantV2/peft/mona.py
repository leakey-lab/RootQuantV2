"""Mona — multi-cognitive visual adapter for frozen ViT blocks.

Parallel depthwise conv adapter (3x3 / 5x5 / 7x7) on patch tokens only.
CLS and register tokens bypass the spatial conv path unchanged.

Near-no-op start: up-projection initialized to zero with ``mona_init_scale=1.0``
so forward output is zero at step 0 but gradients reach ``up`` immediately.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MonaAdapter(nn.Module):
    """Mona-style conv adapter on patch tokens in a token sequence.

    Parameters
    ----------
    dim : int
        Token embedding dimension (1024 for ViT-L).
    bottleneck : int
        Channel width inside the conv bottleneck (Mona paper uses 64).
    num_prefix : int
        Number of leading non-spatial tokens (CLS + registers) to bypass.
    dropout : float
        Dropout after the up-projection.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        num_prefix: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.num_prefix = num_prefix

        self.down = nn.Linear(dim, bottleneck, bias=True)
        self.dw3 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=True)
        self.dw5 = nn.Conv2d(bottleneck, bottleneck, 5, padding=2, groups=bottleneck, bias=True)
        self.dw7 = nn.Conv2d(bottleneck, bottleneck, 7, padding=3, groups=bottleneck, bias=True)
        # 1x1 conv expressed as Linear: same math, but its grad is always
        # contiguous, avoiding the DDP "grad strides do not match bucket view"
        # copy that cuDNN's conv backward triggers.
        self.pw = nn.Linear(bottleneck, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, dim, bias=True)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        for conv in (self.dw3, self.dw5, self.dw7, self.pw):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            nn.init.zeros_(conv.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, N, D) with prefix tokens first."""
        if tokens.shape[1] <= self.num_prefix:
            return torch.zeros_like(tokens)

        prefix = tokens[:, : self.num_prefix, :]
        patches = tokens[:, self.num_prefix :, :]
        B, N, D = patches.shape
        side = int(math.isqrt(N))
        if side * side != N:
            raise ValueError(
                f"MonaAdapter expects a square patch grid; got {N} patch tokens."
            )

        h = self.down(patches)
        x = h.transpose(1, 2).reshape(B, self.bottleneck, side, side)
        x = (self.dw3(x) + self.dw5(x) + self.dw7(x)) / 3.0
        x = x.reshape(B, self.bottleneck, N).transpose(1, 2)
        x = F.gelu(self.pw(x))
        x = self.drop(self.up(x))

        return torch.cat([torch.zeros_like(prefix), x], dim=1)


def _patch_block_with_mona(
    block: nn.Module,
    mona: MonaAdapter,
    scale: nn.Parameter,
    drop_path_rate: float,
) -> None:
    """Wrap ``block.forward`` to add Mona in parallel on the MLP branch.

    DINOv3 ``SelfAttentionBlock.forward`` signature is
    ``(x_or_x_list, rope_or_rope_list=None)`` — rope is positional, not a kwarg.
    """
    drop_path = float(drop_path_rate)

    def _mona_single(x: Tensor, rope) -> Tensor:
        x_attn = x + block.ls1(block.attn(block.norm1(x), rope=rope))
        x_norm = block.norm2(x_attn)
        delta = mona(x_norm) * scale
        if drop_path > 0.0 and mona.training and torch.is_grad_enabled():
            keep = 1.0 - drop_path
            # Per-sample stochastic depth: one Bernoulli per batch element,
            # broadcast over (N, D) — NOT a single batch-global coin flip.
            shape = (delta.shape[0],) + (1,) * (delta.dim() - 1)
            mask = torch.empty(shape, device=x.device, dtype=delta.dtype).bernoulli_(keep) / keep
            delta = delta * mask
        return x_attn + block.ls2(block.mlp(x_norm)) + delta

    def forward(x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return _mona_single(x_or_x_list, rope_or_rope_list)
        if isinstance(x_or_x_list, list):
            ropes = rope_or_rope_list
            if ropes is None:
                ropes = [None] * len(x_or_x_list)
            return [_mona_single(x, rope) for x, rope in zip(x_or_x_list, ropes)]
        raise AssertionError(f"Unexpected block input type: {type(x_or_x_list)}")

    block.forward = forward
    block.mona_adapter = mona
    block.mona_scale = scale


def apply_mona_to_block(
    block: nn.Module,
    dim: int,
    bottleneck: int = 64,
    num_prefix: int = 5,
    dropout: float = 0.1,
    init_scale: float = 0.0,
    drop_path_rate: float = 0.0,
) -> MonaAdapter:
    """Attach a Mona adapter to one transformer block (MLP branch, parallel)."""
    mona = MonaAdapter(
        dim=dim,
        bottleneck=bottleneck,
        num_prefix=num_prefix,
        dropout=dropout,
    )
    scale = nn.Parameter(torch.tensor(float(init_scale)))
    _patch_block_with_mona(block, mona, scale, drop_path_rate)
    return mona


def apply_mona_to_backbone(
    backbone_model: nn.Module,
    dim: int = 1024,
    bottleneck: int = 64,
    num_prefix: int = 5,
    blocks: str | Tuple[int, ...] = "all",
    dropout: float = 0.1,
    init_scale: float = 0.0,
    drop_path_rate: float = 0.0,
) -> list[MonaAdapter]:
    """Apply Mona to selected blocks in ``backbone_model.blocks``."""
    if not hasattr(backbone_model, "blocks"):
        raise AttributeError("backbone_model has no `.blocks` attribute.")
    blks = backbone_model.blocks
    if blocks == "all":
        idxs = range(len(blks))
    else:
        idxs = blocks
    adapters: list[MonaAdapter] = []
    for i in idxs:
        adapters.append(
            apply_mona_to_block(
                blks[i],
                dim=dim,
                bottleneck=bottleneck,
                num_prefix=num_prefix,
                dropout=dropout,
                init_scale=init_scale,
                drop_path_rate=drop_path_rate,
            )
        )
    return adapters


__all__ = ["MonaAdapter", "apply_mona_to_block", "apply_mona_to_backbone"]
