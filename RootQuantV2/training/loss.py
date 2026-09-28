"""Presence-balanced regression loss (Huber / SmoothL1 / MSE) for RootQuantV2.

Two standardized targets (length, area). Most frames in a minirhizotron series
are empty (their target maps to a fixed negative value after z-scoring on
present rows), so presence is heavily imbalanced. Empty and present
rows are both kept in the loss — predicting ~0 for empty images is correct and
learnable — but present rows can be up-weighted so they are not drowned out.

No Kendall uncertainty, no presence head, no gating: those were the main
sources of training instability in the old model. This is a plain weighted
Huber over both targets.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RootRegressionLoss(nn.Module):
    """Presence-balanced regression loss over two standardized targets.

    ``loss_type`` controls the residual penalty (metric note below):
      * ``smooth_l1`` (legacy): ``F.smooth_l1_loss(beta=delta)`` ≡ Huber(δ=delta);
        at delta=1 the L1 knee sits at 1 std, so it CAPS the gradient on the
        heavy-tail large roots — the very samples that dominate R².
      * ``huber``: ``F.huber_loss(delta=delta)`` — same L2 gradient as MSE up to
        ``delta`` std, clipping only pathological residuals. With delta≈3-4 this is
        ≈MSE for R² while keeping an outlier guard. Preferred for accuracy.
      * ``mse``: ``0.5·(pred-target)²`` — directly maximizes R² (R²_raw = 1 − MSE_z
        on present rows), no robustness clip.

    NOTE: ``smooth_l1`` and ``huber`` differ by a 1/delta scale — switch the TYPE,
    do not just enlarge delta under ``smooth_l1`` (that also shrinks the gradient).
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        w_present: float = 1.0,
        w_empty: float = 1.0,
        loss_type: str = "smooth_l1",
        task_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        self.delta = float(huber_delta)
        self.w_present = float(w_present)
        self.w_empty = float(w_empty)
        self.loss_type = str(loss_type)
        # Per-target (length, area) weight. The combination is NORMALIZED by the
        # weight sum, so the overall loss scale (and thus effective LR) is
        # unchanged — only the length↔area balance shifts. (1, 1) is the legacy
        # equal-mean behavior; (1, 1.5) emphasizes the harder area target.
        tw = torch.tensor([float(task_weights[0]), float(task_weights[1])], dtype=torch.float32)
        self.register_buffer("task_weights", tw)

    def forward(
        self,
        preds: Dict[str, Tensor],
        targets: Tensor,   # (B, 2) standardized: col0=length, col1=area
        mask: Tensor,      # (B,) 1.0=present, 0.0=empty
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        pred = torch.stack([preds["length"], preds["area"]], dim=-1).float()  # (B,2)
        targets = targets.float()
        mask_f = mask.float()

        if self.loss_type == "mse":
            # 0.5·e² → L2 gradient = e, matching the others' quadratic region (no LR shift).
            per_task = 0.5 * (pred - targets) ** 2                                      # (B,2)
        elif self.loss_type == "huber":
            per_task = F.huber_loss(pred, targets, reduction="none", delta=self.delta)  # (B,2)
        else:  # "smooth_l1" — Huber(δ=1)≡SmoothL1(β=1); caps tail-residual gradients
            per_task = F.smooth_l1_loss(pred, targets, reduction="none", beta=self.delta)
        # Weighted mean over the two targets (normalized → loss scale preserved).
        tw = self.task_weights.to(per_task.dtype)
        per_sample = (per_task * tw).sum(dim=1) / tw.sum()                              # (B,)

        w = mask_f * self.w_present + (1.0 - mask_f) * self.w_empty                    # (B,)
        loss = (per_sample * w).sum() / w.sum().clamp_min(1e-6)

        # Logging: per-task loss on present rows only (what we ultimately care about).
        n_present = mask_f.sum().clamp_min(1.0)
        comps: Dict[str, Tensor] = {
            "loss": loss.detach(),
            "loss_length": (per_task[:, 0] * mask_f).sum().detach() / n_present,
            "loss_area": (per_task[:, 1] * mask_f).sum().detach() / n_present,
            "mask_pos_rate": mask_f.mean().detach(),
        }
        return loss, comps


__all__ = ["RootRegressionLoss"]
