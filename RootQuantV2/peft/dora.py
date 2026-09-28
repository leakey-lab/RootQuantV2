"""DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024, arxiv 2402.09353).

DoRA decomposes a frozen linear weight ``W in R^{out x in}`` into

    magnitude  m = ||W||_col                        (trainable, shape (out,))
    direction  V = W / ||W||_col                    (frozen, shape (out, in))

and re-adapts the direction with a low-rank update::

    V_adapted = V_frozen + B @ A          # B in R^{out x r},  A in R^{r x in}
    W_eff     = m * V_adapted / ||V_adapted||_col

At initialisation B = 0, so V_adapted = V_frozen and ``W_eff = m * V`` exactly
recovers the original weight ``W`` to floating-point round-off.  Training thus
starts as a no-op and the adapter is incrementally learned.

This module is fp32 only — no autocast inside.  No imports from
``RootQuantV2.config``; everything is taken via explicit kwargs.

LinearKMaskedBias trade-off
---------------------------
DINOv3 ViT-L/16's ``attn.qkv`` is a ``LinearKMaskedBias`` — an nn.Linear
subclass that applies a structured mask to bias entries belonging to register
tokens.  When we replace its forward path with the DoRA computation we lose
that masking.  Acceptable because (a) the underlying weight is frozen via the
``V_frozen`` buffer, (b) register tokens carry minimal regression signal in
this downstream task, and (c) the mask was a pre-training-stability device,
not a downstream requirement.  See the in-line comment in ``DoRALinear``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────────
# DoRALinear
# ───────────────────────────────────────────────────────────────────────────────


class DoRALinear(nn.Module):
    """DoRA replacement for an ``nn.Linear`` (or compatible) layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension of the original linear layer.
    out_features : int
        Output feature dimension of the original linear layer.
    rank : int, default 16
        Low-rank update rank ``r``.  Trainable params per layer:
        ``out + r*in + out*r`` (magnitude + A + B).
    bias : bool, default True
        Whether the original layer had a bias.  The bias is kept as a
        non-trainable buffer (frozen) — see the ``LinearKMaskedBias`` note in
        the module docstring.
    original_weight : Tensor of shape ``(out, in)``, required
        The original weight tensor.  Used to compute the frozen direction
        buffer ``V_frozen`` and to initialise the magnitude vector
        ``dora_m = ||W||_col``.
    original_bias : Tensor of shape ``(out,)`` or None
        The original bias tensor.  Stored as a frozen buffer.

    Forward
    -------
    ``x: Tensor[..., in_features] -> Tensor[..., out_features]``

    The effective weight is recomputed every forward pass::

        V_adapted = V_frozen + dora_B @ dora_A     # (out, in)
        W_eff     = (dora_m / ||V_adapted||_col) * V_adapted   # (out, in)

    Trainable parameter names use the suffixes ``dora_m``, ``dora_A``,
    ``dora_B`` so the trainer's name-prefix LR grouping (``*.dora_*``) matches.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if original_weight is None:
            raise ValueError(
                "DoRALinear requires `original_weight` at construction so that "
                "magnitude m and frozen direction V can be initialised from the "
                "pretrained linear layer."
            )
        if original_weight.shape != (out_features, in_features):
            raise ValueError(
                f"original_weight has shape {tuple(original_weight.shape)}, "
                f"expected ({out_features}, {in_features})."
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Detach + clone + cast to fp32 so that the buffer is independent of the
        # source tensor's autograd graph and dtype.
        W = original_weight.detach().clone().float()  # (out, in)
        col_norm = W.norm(dim=1, keepdim=True)        # (out, 1) — per-output-channel L2

        # Frozen direction matrix V = W / ||W||_col.  Stored as a buffer so it
        # rides along with .to(device) / state_dict but is not a Parameter.
        self.register_buffer("V_frozen", W / (col_norm + 1e-9))

        # Trainable magnitude m, initialised from ||W||_col.  Shape (out,).
        self.dora_m = nn.Parameter(col_norm.squeeze(-1).clone())

        # Low-rank decomposition: B @ A = 0 at init so that V_adapted == V_frozen.
        # A: kaiming-uniform with gain sqrt(5) — same convention as nn.Linear.
        # B: zeros so the initial output is exactly W_orig @ x (+ b).
        self.dora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.dora_A, a=math.sqrt(5))

        self.dora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Bias: frozen non-trainable buffer.  We store it as a buffer (not a
        # Parameter with requires_grad=False) so it does not appear in
        # .parameters() and cannot be picked up by the optimizer.
        # NOTE: for LinearKMaskedBias modules (qkv) this drops the structured
        # register-bias mask — see module docstring for why this is acceptable.
        if bias and original_bias is not None:
            self.register_buffer("bias_buffer", original_bias.detach().clone().float())
        else:
            self.register_buffer("bias_buffer", None)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)."""
        # Reconstruct the effective weight in fp32 even under bf16 autocast: the
        # magnitude/direction decomposition (low-rank B@A, per-row norm, division)
        # is precision-sensitive. Only the large F.linear below should run in low
        # precision — and it does, since we exit the disabled-autocast block first.
        # In fp32 runs this is a no-op (.float() returns self), so behavior is
        # byte-identical to before.
        ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if torch.is_autocast_enabled()
            else nullcontext()
        )
        with ctx:
            V_adapted = self.V_frozen + (self.dora_B.float() @ self.dora_A.float())
            norm = V_adapted.norm(dim=1, keepdim=True) + 1e-9
            W_eff = (self.dora_m.float().unsqueeze(-1) / norm) * V_adapted
        return F.linear(x, W_eff, self.bias_buffer)

    # ──────────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias_buffer is not None}"
        )


# ───────────────────────────────────────────────────────────────────────────────
# Block-level swap
# ───────────────────────────────────────────────────────────────────────────────


_DEFAULT_TARGETS: Tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _set_submodule(parent_block: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at ``dotted_name`` (relative to ``parent_block``).

    Splits on the last dot, navigates to the parent via
    ``parent_block.get_submodule(parent_path)``, then setattr's the leaf.
    Empty parent_path means the leaf is a direct child of parent_block.
    """
    if "." in dotted_name:
        parent_path, leaf = dotted_name.rsplit(".", 1)
        parent = parent_block.get_submodule(parent_path)
    else:
        parent, leaf = parent_block, dotted_name
    setattr(parent, leaf, new_module)


def apply_dora_to_block(
    block: nn.Module,
    rank: int = 16,
    target_names: Iterable[str] = _DEFAULT_TARGETS,
) -> nn.Module:
    """In-place replace the named Linear-equivalent submodules of ``block`` with DoRALinear.

    Parameters
    ----------
    block : nn.Module
        A single transformer block (e.g. ``model.blocks[i]``).
    rank : int, default 16
        DoRA rank.
    target_names : iterable of str
        Dotted submodule paths within ``block`` to swap.  Defaults to the four
        DINOv3 ViT-L/16 paths confirmed by Phase 1 inspection:
        ``("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")``.

    Notes
    -----
    Uses ``block.get_submodule(name)`` so dotted paths work.  For
    ``LinearKMaskedBias`` (the qkv module) we treat any object with ``.weight``
    and (optionally) ``.bias`` as Linear-equivalent — we do NOT use
    ``isinstance(module, nn.Linear)`` as the guard.

    After swapping, defensively sets ``requires_grad=False`` on every remaining
    parameter in the block whose name does NOT contain ``dora_`` — this
    re-asserts that only DoRA's m, A, B are trainable.
    """
    targets = tuple(target_names)
    for name in targets:
        mod = block.get_submodule(name)

        if not hasattr(mod, "weight"):
            raise TypeError(
                f"apply_dora_to_block: submodule '{name}' is a {type(mod).__name__} "
                f"with no .weight attribute — cannot apply DoRA."
            )
        weight = mod.weight  # type: ignore[attr-defined]
        bias = getattr(mod, "bias", None)

        out_features, in_features = weight.shape

        dora = DoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias is not None,
            original_weight=weight.data,
            original_bias=bias.data if bias is not None else None,
        )
        # Match the dtype/device of the original module so the swap is seamless.
        dora = dora.to(device=weight.device, dtype=weight.dtype)
        _set_submodule(block, name, dora)

    # Defensive freeze: anything not named dora_* should remain frozen.
    for pname, p in block.named_parameters():
        if "dora_" in pname:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    return block


def apply_dora_to_backbone(
    backbone_model: nn.Module,
    rank: int = 16,
    target_names: Iterable[str] = _DEFAULT_TARGETS,
) -> nn.Module:
    """Convenience: apply DoRA to every block in ``backbone_model.blocks``.

    ``backbone_model`` is the *inner* DINOv3 model (e.g. the result of
    ``torch.hub.load(...)``), not the ``DINOv3Backbone`` wrapper.  Iterates
    ``backbone_model.blocks`` and calls ``apply_dora_to_block`` on each.
    Returns the same model for chaining.
    """
    if not hasattr(backbone_model, "blocks"):
        raise AttributeError(
            "apply_dora_to_backbone: backbone_model has no `.blocks` attribute. "
            "Pass the inner DINOv3 ViT model (e.g. wrapper.model), not the "
            "DINOv3Backbone wrapper."
        )
    for block in backbone_model.blocks:
        apply_dora_to_block(block, rank=rank, target_names=target_names)
    return backbone_model


__all__ = [
    "DoRALinear",
    "apply_dora_to_block",
    "apply_dora_to_backbone",
]
