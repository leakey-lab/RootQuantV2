"""DinoV3RootRegressor — frozen DINOv3 ViT-L/16 + DoRA + Mona + MLP head.

Image-level regression of (length_mm, area_mm2). No dense/segmentation path.

Flow
----
    RGB ─► DINOv3 ViT-L/16 (FROZEN, DoRA + optional Mona on blocks)
        ─► forward_features → CLS + patch tokens
        ─► pool  [CLS ⊕ attn(patch) ⊕ GeM(patch)]  or legacy mean/max
        ─► MLP head → (length, area)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from torch import Tensor

from RootQuantV2.backbone import DINOv3Backbone
from RootQuantV2.peft import apply_dora_to_block, apply_mona_to_backbone


_POOL_FACTOR = {
    "cls": 1,
    "mean": 1,
    "cls_mean": 2,
    "cls_mean_max": 3,
    "cls_attn_gem": 3,
}


def _patch_dinov3_sdpa_contiguous() -> None:
    """Make DINOv3's SelfAttention q/k/v contiguous before SDPA."""
    try:
        from dinov3.layers.attention import SelfAttention
    except Exception:
        return
    if getattr(SelfAttention, "_rqv2_compile_patched", False):
        return

    def _patched_compute_attention(self, qkv, attn_bias=None, rope=None):
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        import torch.nn.functional as _F
        x = _F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    SelfAttention.compute_attention = _patched_compute_attention
    SelfAttention._rqv2_compile_patched = True


class AttentionPool(nn.Module):
    """Single learnable query attends over patch tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, patches: Tensor, patch_mask: Optional[Tensor] = None) -> Tensor:
        logits = torch.einsum("bnd,d->bn", patches, self.query)
        if patch_mask is not None:
            logits = logits.masked_fill(patch_mask <= 0.0, -1e4)
        w = F.softmax(logits, dim=-1)
        return torch.einsum("bn,bnd->bd", w, patches)


class GeMPool(nn.Module):
    """Generalized mean pooling with learnable exponent p.

    ``mode='legacy'``   : ``clamp(min=eps).pow(p)`` — assumes NON-NEGATIVE inputs.
                          The patch tokens here are post-final-LayerNorm (signed,
                          ~zero-mean), so this floors ~half of every channel to
                          ``eps`` and collapses negative-dominant channels. Kept
                          only for backward-compat with existing checkpoints.
    ``mode='softplus'`` : GeM over ``softplus(tokens)`` — a smooth, strictly
                          positive map — so the generalized mean is well-defined
                          on signed features. Preferred.
    """

    def __init__(
        self,
        dim: int,
        init_p: float = 3.0,
        eps: float = 1e-6,
        mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(init_p)))
        self.eps = eps
        self.mode = str(mode)

    def _nonneg(self, patches: Tensor) -> Tensor:
        if self.mode == "softplus":
            return F.softplus(patches)
        return patches

    def forward(self, patches: Tensor, patch_mask: Optional[Tensor] = None) -> Tensor:
        p = self.p.clamp(min=1.0)
        x = self._nonneg(patches).clamp(min=self.eps).pow(p)
        if patch_mask is not None:
            m = patch_mask.unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            mean_p = (x * m).sum(dim=1) / denom
            return mean_p.pow(1.0 / p)
        return x.mean(dim=1).pow(1.0 / p)


class PatchPooler(nn.Module):
    """Dispatch patch pooling modes used by the regression head."""

    def __init__(
        self,
        dim: int,
        mode: str,
        gem_init_p: float = 3.0,
        gem_mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "cls_attn_gem":
            self.attn_pool = AttentionPool(dim)
            self.gem_pool = GeMPool(dim, init_p=gem_init_p, mode=gem_mode)
        else:
            self.attn_pool = None
            self.gem_pool = None

    def _masked_mean(self, patches: Tensor, patch_mask: Optional[Tensor]) -> Tensor:
        if patch_mask is None:
            return patches.mean(dim=1)
        m = patch_mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (patches * m).sum(dim=1) / denom

    def _masked_max(self, patches: Tensor, patch_mask: Optional[Tensor]) -> Tensor:
        if patch_mask is None:
            return patches.amax(dim=1)
        masked = patches.masked_fill(patch_mask.unsqueeze(-1) <= 0.0, -1e4)
        return masked.amax(dim=1)

    def forward(
        self,
        cls: Tensor,
        patches: Tensor,
        patch_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.mode == "cls":
            return cls
        if self.mode == "mean":
            return self._masked_mean(patches, patch_mask)
        if self.mode == "cls_mean":
            return torch.cat([cls, self._masked_mean(patches, patch_mask)], dim=-1)
        if self.mode == "cls_mean_max":
            return torch.cat([
                cls,
                self._masked_mean(patches, patch_mask),
                self._masked_max(patches, patch_mask),
            ], dim=-1)
        if self.mode == "cls_attn_gem":
            assert self.attn_pool is not None and self.gem_pool is not None
            return torch.cat([
                cls,
                self.attn_pool(patches, patch_mask),
                self.gem_pool(patches, patch_mask),
            ], dim=-1)
        raise ValueError(f"Unknown pool mode '{self.mode}'.")


class RegressionHead(nn.Module):
    """LayerNorm → [Linear → GELU → Dropout] × N → Linear(2).

    ``hidden`` is either an int (single hidden layer, legacy behavior) or a
    sequence of ints giving one hidden width per layer, e.g. ``(1024, 256)`` for
    a 2-hidden-layer head with a gentler taper from the 3072-d pooled feature.
    A LayerNorm is interleaved before each hidden block for training stability of
    the deeper variant; with a single layer this reduces to the original head.
    """

    def __init__(self, c_in: int, hidden=512, dropout: float = 0.2) -> None:
        super().__init__()
        dims = (int(hidden),) if isinstance(hidden, (int, float)) else tuple(int(h) for h in hidden)
        layers = [nn.LayerNorm(c_in)]
        prev = c_in
        for i, h in enumerate(dims):
            if i > 0:
                # Stabilize deeper heads; absent for the single-layer legacy case.
                layers.append(nn.LayerNorm(prev))
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        out = nn.Linear(prev, 2)
        nn.init.zeros_(out.bias)
        nn.init.normal_(out.weight, std=1e-3)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DensityReadout(nn.Module):
    """Extensive (sum-pooling) readout for length / area.

    Every other pool here (cls/mean/max/attn/GeM) is INTENSIVE — an average — but
    root length and area are EXTENSIVE: they scale with how much root is in the
    image. This head predicts a NON-NEGATIVE per-patch density for each target,
    zeroes letterbox padding, and SUMS over patches (a crowd-counting-style
    readout), giving a statistic that grows with root extent.

    Metric-aligned by construction: the masked sum is a raw-unit estimate, and it
    is mapped to the standardized target space with FIXED mean/std buffers (from
    target_stats) rather than learnable affines — so an empty image (≈0 density)
    maps to ``-mean/std`` exactly, which is the empty-row target z. Only the
    per-patch MLP and a per-target ``gain`` are learned; ``gain`` starts small so
    the summed density is calibrated, not exploded, at init.

    NOTE: assumes an additive/extensive target space (target_transform ∈
    {zscore, none}); it is not meaningful under a log transform.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
        mean: tuple = (0.0, 0.0),
        std: tuple = (1.0, 1.0),
        gain_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        # Small last layer → near-uniform low density at init (live gradients).
        nn.init.normal_(self.mlp[-1].weight, std=1e-2)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gain = nn.Parameter(torch.full((2,), float(gain_init)))
        self.register_buffer("t_mean", torch.tensor([float(mean[0]), float(mean[1])]))
        self.register_buffer("t_std", torch.tensor([float(std[0]), float(std[1])]))

    def forward(self, patches: Tensor, patch_mask: Optional[Tensor] = None) -> Tensor:
        dens = F.softplus(self.mlp(patches)).float()     # (B, N, 2) ≥ 0, fp32 for the sum
        if patch_mask is not None:
            dens = dens * patch_mask.float().unsqueeze(-1)
        total = dens.sum(dim=1) * self.gain              # (B, 2) raw-unit estimate (fp32 accum)
        return (total - self.t_mean) / self.t_std        # → standardized space


class DinoV3RootRegressor(nn.Module):
    """Frozen DINOv3 + DoRA + optional Mona + pooled MLP regression head."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.pool_mode = str(cfg.get("pool", "cls_mean_max"))
        if self.pool_mode not in _POOL_FACTOR:
            raise ValueError(
                f"Unknown pool '{self.pool_mode}'. Options: {list(_POOL_FACTOR)}"
            )
        self.R = int(cfg["backbone_num_register_tokens"])
        D = int(cfg["backbone_dim"])

        # Weights file: DINOV3_CHECKPOINT_PATH, else cfg's path if it exists here,
        # else RootQuantV2/dinov3/ (see backbone.dinov3.resolve_weights_path).
        self.backbone = DINOv3Backbone(
            weights_path=cfg.get("backbone_weights_path"),
            hub_repo=cfg["backbone_hub_repo"],
            hub_model="dinov3_vitl16",
            num_register_tokens=self.R,
            embed_dim=D,
            patch_size=int(cfg["backbone_patch_size"]),
            pretrained=cfg.get("pretrained", True),
        )
        for h in self.backbone._hooks:
            h.remove()
        self.backbone._hooks.clear()

        _patch_dinov3_sdpa_contiguous()

        blocks = self.backbone.model.blocks
        which_dora = cfg.get("dora_blocks", "all")
        dora_idxs = range(len(blocks)) if which_dora == "all" else tuple(which_dora)

        if cfg.get("use_dora", True):
            for i in dora_idxs:
                apply_dora_to_block(
                    blocks[i],
                    rank=int(cfg["dora_rank"]),
                    target_names=tuple(cfg["dora_target_names"]),
                )

        if cfg.get("use_mona", False):
            which_mona = cfg.get("mona_blocks", "all")
            mona_idxs = range(len(blocks)) if which_mona == "all" else tuple(which_mona)
            apply_mona_to_backbone(
                self.backbone.model,
                dim=D,
                bottleneck=int(cfg.get("mona_bottleneck", 64)),
                num_prefix=1 + self.R,
                blocks=mona_idxs if which_mona != "all" else "all",
                dropout=float(cfg.get("mona_dropout", 0.1)),
                init_scale=float(cfg.get("mona_init_scale", 0.0)),
                drop_path_rate=float(cfg.get("mona_drop_path", 0.0)),
            )

        # Optional: fully unfreeze the last N transformer blocks for extra
        # adaptation capacity beyond DoRA/Mona. Their pretrained Linear/Norm
        # weights become trainable and are picked up by the trainer's dedicated
        # low-LR "backbone" optimizer group (see _build_optimizer). 0 = keep the
        # backbone fully frozen (default, legacy behavior). Done AFTER DoRA/Mona
        # so it overrides their defensive freeze on the selected blocks.
        n_unfreeze = int(cfg.get("unfreeze_last_n_blocks", 0))
        if n_unfreeze > 0:
            n_unfreeze = min(n_unfreeze, len(blocks))
            for block in blocks[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad_(True)

        if bool(cfg.get("use_grad_checkpointing", True)):
            for block in blocks:
                self._wrap_checkpoint(block)

        # readout ∈ {global (pooled MLP), density (extensive sum), both (blend)}
        self.readout = str(cfg.get("readout", "global"))
        if self.readout not in ("global", "density", "both"):
            raise ValueError(
                f"Unknown readout '{self.readout}'. Options: global, density, both"
            )

        pool_dim = D * _POOL_FACTOR[self.pool_mode]
        if self.readout in ("global", "both"):
            self.pooler = PatchPooler(
                dim=D,
                mode=self.pool_mode,
                gem_init_p=float(cfg.get("gem_init_p", 3.0)),
                gem_mode=str(cfg.get("gem_mode", "legacy")),
            )
            _hh = cfg.get("head_hidden", 512)
            head_hidden = _hh if isinstance(_hh, (list, tuple)) else int(_hh)
            self.head = RegressionHead(
                c_in=pool_dim,
                hidden=head_hidden,
                dropout=float(cfg.get("head_dropout", 0.2)),
            )
        else:
            self.pooler = None
            self.head = None

        if self.readout in ("density", "both"):
            stats = cfg.get("target_stats", {}) or {}
            self.density_head = DensityReadout(
                dim=D,
                hidden=int(cfg.get("density_hidden", 256)),
                dropout=float(cfg.get("density_dropout", 0.1)),
                mean=(float(stats.get("length_mean", 0.0)), float(stats.get("area_mean", 0.0))),
                std=(float(stats.get("length_std", 1.0)), float(stats.get("area_std", 1.0))),
            )
        else:
            self.density_head = None

        # Per-target convex blend (sigmoid → weight on the global head); only for "both".
        self.readout_blend = nn.Parameter(torch.zeros(2)) if self.readout == "both" else None

    @staticmethod
    def _wrap_checkpoint(block: nn.Module) -> None:
        orig = block.forward

        def _ckpt_forward(*args, **kwargs):
            if torch.is_grad_enabled():
                return _ckpt.checkpoint(orig, *args, use_reentrant=False, **kwargs)
            return orig(*args, **kwargs)

        block.forward = _ckpt_forward

    def _get_patches(self, ff: Dict[str, Tensor]) -> Tensor:
        if "x_norm_patchtokens" in ff:
            return ff["x_norm_patchtokens"]
        return ff["x_prenorm"][:, 1 + self.R :, :]

    def train(self, mode: bool = True) -> "DinoV3RootRegressor":
        """Keep frozen backbone in eval; train Mona adapters + head separately."""
        super().train(mode)
        if self.cfg.get("use_mona", False):
            for block in self.backbone.model.blocks:
                if hasattr(block, "mona_adapter"):
                    block.mona_adapter.train(mode)
        return self

    def forward(
        self,
        x: Tensor,
        patch_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected (B,3,H,W); got {tuple(x.shape)}")
        ff = self.backbone.model.forward_features(x)
        if isinstance(ff, list):
            ff = ff[0]
        cls = ff["x_norm_clstoken"]
        patches = self._get_patches(ff)
        if patch_mask is not None and patch_mask.shape[1] != patches.shape[1]:
            raise ValueError(
                f"patch_mask length {patch_mask.shape[1]} != "
                f"num patches {patches.shape[1]}"
            )
        out_global = None
        out_density = None
        if self.head is not None:
            feat = self.pooler(cls, patches, patch_mask=patch_mask)
            out_global = self.head(feat)
        if self.density_head is not None:
            out_density = self.density_head(patches, patch_mask=patch_mask)

        if self.readout == "global":
            out = out_global
        elif self.readout == "density":
            out = out_density
        else:  # both: per-target convex blend of the two readouts
            a = torch.sigmoid(self.readout_blend)
            out = a * out_global + (1.0 - a) * out_density
        return {"length": out[:, 0], "area": out[:, 1]}

    def summarize_params(self) -> Dict[str, Any]:
        trainable, total = 0, 0
        dora, mona, head, pool, density = 0, 0, 0, 0, 0
        for name, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                if "dora_" in name:
                    dora += p.numel()
                elif "mona" in name:
                    mona += p.numel()
                elif "density_head" in name or name == "readout_blend":
                    density += p.numel()
                elif name.startswith("head."):
                    head += p.numel()
                elif name.startswith("pooler."):
                    pool += p.numel()
        for bname, buf in self.named_buffers():
            if bname.startswith("backbone."):
                total += buf.numel()
        return {
            "trainable": trainable,
            "total": total,
            "fraction": trainable / total if total else 0.0,
            "dora": dora,
            "mona": mona,
            "head": head,
            "pool": pool,
            "density": density,
        }

    def print_summary(self) -> None:
        s = self.summarize_params()
        print(
            f"DinoV3RootRegressor — trainable {s['trainable']:,} / {s['total']:,} "
            f"= {s['fraction']:.2%}  "
            f"(dora={s['dora']:,}  mona={s['mona']:,}  "
            f"pool={s['pool']:,}  head={s['head']:,}  density={s['density']:,})"
        )


@torch.no_grad()
def tta_predict_mean(
    model: nn.Module,
    x: Tensor,
    patch_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Test-time augmentation: average predictions over the symmetry group of the
    INPUT tensor. Flips/rotations preserve total length and area exactly, so this
    is a strictly variance-reducing, label-consistent ensemble — a near-free
    eval-time gain. The image and its letterbox ``patch_mask`` get the identical
    group element each view.

    The group depends on the tensor's aspect, because 90°/270° rotations TRANSPOSE
    H↔W and are only valid on a square tensor (and only such inputs were seen in
    training):

    * **square** input (letterbox modes, trained with ``RandomD4``)
      → full **D4**, 8 views: ``{id, rot90, rot180, rot270} × {·, hflip}``.
    * **non-square** input (``native_rect``, trained with ``RandomD2``)
      → **D2**, 4 views: ``{id, hflip, vflip, rot180}`` — the shape-preserving subset.

    Returns standardized predictions ({"length", "area"}); the caller inverts.
    """
    B, _, H, W = x.shape

    m2d: Optional[Tensor] = None
    if patch_mask is not None:
        n = int(patch_mask.shape[1])
        grid = int(math.isqrt(n))
        if grid * grid != n:
            return model(x, patch_mask=patch_mask)  # can't map mask to a grid → no TTA
        m2d = patch_mask.reshape(B, grid, grid)

    # group elements as (hflip, vflip, k_rot90)
    if H == W:
        elems = [(h, False, k) for h in (False, True) for k in (0, 1, 2, 3)]   # D4 (8)
    else:
        elems = [(False, False, 0), (True, False, 0), (False, True, 0), (False, False, 2)]  # D2 (4)

    lsum: Optional[Tensor] = None
    asum: Optional[Tensor] = None
    for fh, fv, k in elems:
        xt, mt = x, m2d
        if fh:
            xt = torch.flip(xt, dims=[3])
            mt = None if mt is None else torch.flip(mt, dims=[2])
        if fv:
            xt = torch.flip(xt, dims=[2])
            mt = None if mt is None else torch.flip(mt, dims=[1])
        if k:
            xt = torch.rot90(xt, k=k, dims=[2, 3])
            mt = None if mt is None else torch.rot90(mt, k=k, dims=[1, 2])
        pm = None if mt is None else mt.reshape(B, -1).contiguous()
        out = model(xt.contiguous(), patch_mask=pm)
        lsum = out["length"] if lsum is None else lsum + out["length"]
        asum = out["area"] if asum is None else asum + out["area"]
    nv = float(len(elems))
    return {"length": lsum / nv, "area": asum / nv}


__all__ = [
    "DinoV3RootRegressor",
    "RegressionHead",
    "PatchPooler",
    "AttentionPool",
    "GeMPool",
    "DensityReadout",
    "tta_predict_mean",
]
