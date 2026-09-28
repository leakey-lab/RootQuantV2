"""visualize_attention.py — "where is the fine-tuned DINOv3 looking?" maps.

Renders, for one or more images, a panel of complementary saliency maps from a
trained RootQuantV2 checkpoint (DINOv3 ViT-L/16 frozen + DoRA/Mona + regression
head). Mirrors infer.py's checkpoint loading (cfg + target_stats live in the .pt).

Two families of maps
--------------------
**Backbone (self-supervised style — what the DINO paper shows).** These are
computed on the adapted tokens (DoRA and Mona are in the forward path), but
they are task-agnostic: attention, PCA and cosine similarity organise any
structured texture, soil included, so they answer "what does the representation
group together", not "what does the regressor measure". For the same PCA view
on un-adapted DINOv3 tokens, see ``viz_root_panel.py --compare_frozen``.
  * ``cls_attn``     — last-block [CLS] self-attention, per head + mean (the
                        iconic DINO map). DINOv3 uses RoPE + SDPA, which hides
                        the attention matrix, so we monkeypatch
                        ``SelfAttention.compute_attention`` to recover the
                        softmax probabilities while preserving the exact output.
  * ``rollout``      — attention rollout (Abnar & Zuidema 2020) across all 24
                        blocks; usually the cleanest single object mask.
  * ``pca``          — PCA of patch tokens; PC1 (foreground) + top-3 PCs as RGB.
                        This is the canonical DINOv2/v3 "segmentation" view.
  * ``cosine``       — cosine similarity of every patch to a query patch (the
                        patch the regression head attends to most).

**Head (task-faithful — where OUR model looks to measure roots).** These come
from the trained head and are the honest answer for a regression model:
  * ``head_attn``    — the learnable AttentionPool query's softmax weights over
                        patches (``pool=cls_attn_gem``): where the global head
                        pools from.
  * ``density``      — the DensityReadout's per-patch non-negative density for
                        length and area (``readout=both``). The prediction is the
                        masked SUM of this map, so it is a learned, prediction-
                        coupled "root mass" map — the closest thing to a learned
                        root segmentation.

Usage (from the repository root)
--------------------------------
    # default: sample present-root images from the test split (needs
    # ROOTQUANT_DATA_DIR, or --csv / --data_path), headline checkpoint
    python -m RootQuantV2.visualize_attention

    # specific images (no dataset needed)
    python -m RootQuantV2.visualize_attention \
        --images /path/a.jpg /path/b.jpg --out_dir RootQuantV2/viz_attention

    # different checkpoint / block / no rollout (faster)
    python -m RootQuantV2.visualize_attention \
        --checkpoint RootQuantV2/runs/checkpoints/rootquant-v2-soybean-only/best.pt --block -1 --no-rollout
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Headless rendering.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import get_transforms, load_split
from RootQuantV2.data.transforms import LetterboxToSquare
from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.model import DinoV3RootRegressor
from RootQuantV2.training import load, load_into


# --------------------------------------------------------------------------- #
# Attention capture: replace DINOv3's SDPA with an explicit-softmax path that
# records the attention probabilities. Mathematically identical at eval (no
# dropout, no mask), so the forward output (and thus the predictions/pooling)
# is unchanged — we only gain visibility into the attention matrix.
# --------------------------------------------------------------------------- #
_CAP: Dict[str, object] = {
    "on": False,        # capture toggle (off during prediction passes)
    "cls_idx": 0,       # [CLS] token index
    "cls_rows": [],     # per-block [CLS] attention rows: list of (B, H, N)
    "rollout": None,    # running rollout matrix (B, N, N)
    "do_rollout": True,
}


def _install_attention_capture() -> None:
    """Monkeypatch ``SelfAttention.compute_attention`` to optionally record
    attention. Layers on top of (and supersedes) the repo's contiguity patch."""
    from dinov3.layers.attention import SelfAttention

    def _compute_attention(self, qkv, attn_bias=None, rope=None):
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        H = self.num_heads
        hd = C // H
        qkv = qkv.reshape(B, N, 3, H, hd)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]   # (B, H, N, hd)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        if not _CAP["on"]:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            x = F.scaled_dot_product_attention(q, k, v)
            return x.transpose(1, 2).reshape(B, N, C)

        # Explicit softmax attention (== SDPA at eval) so we can read it off.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B,H,N,N)
        probs = scores.softmax(dim=-1)
        ci = int(_CAP["cls_idx"])
        _CAP["cls_rows"].append(probs[:, :, ci, :].detach().float().cpu())  # (B,H,N)
        if _CAP["do_rollout"]:
            A = probs.mean(dim=1)                                    # (B,N,N)
            eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = A + eye
            A = A / A.sum(dim=-1, keepdim=True)
            _CAP["rollout"] = A if _CAP["rollout"] is None else torch.matmul(A, _CAP["rollout"])
        x = torch.matmul(probs, v)                                   # (B,H,N,hd)
        return x.transpose(1, 2).reshape(B, N, C)

    SelfAttention.compute_attention = _compute_attention
    SelfAttention._rqv2_compile_patched = True   # also satisfies model.py's guard


def _reset_capture(cls_idx: int, do_rollout: bool) -> None:
    _CAP["cls_rows"] = []
    _CAP["rollout"] = None
    _CAP["cls_idx"] = cls_idx
    _CAP["do_rollout"] = do_rollout


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def _upsample(grid_map: np.ndarray, side: int) -> np.ndarray:
    """Bilinearly resize a (g, g) map to (side, side)."""
    t = torch.from_numpy(np.ascontiguousarray(grid_map, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _letterbox_display(pil: Image.Image, side: int) -> np.ndarray:
    """Letterbox the original image to a (side, side) RGB array for display —
    same geometry as the model input but WITHOUT ImageNet normalization."""
    sq = LetterboxToSquare(target=side, fill=0).letterbox(pil)[0]
    return np.asarray(sq.convert("RGB"), dtype=np.float32) / 255.0


def _grid_from_mask(patch_mask: Optional[torch.Tensor], n_patch: int) -> int:
    g = int(math.isqrt(n_patch))
    if g * g != n_patch:
        raise ValueError(f"{n_patch} patch tokens is not a perfect square.")
    return g


# --------------------------------------------------------------------------- #
# Map computation (single image)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_maps(
    model: DinoV3RootRegressor,
    x: torch.Tensor,                    # (1, 3, S, S) normalized model input
    patch_mask: Optional[torch.Tensor], # (1, N_patch) or None
    cfg: dict,
    stats: dict,
    block: int,
    do_rollout: bool,
) -> Dict[str, object]:
    device = x.device
    R = int(cfg["backbone_num_register_tokens"])
    prefix = 1 + R                       # CLS + registers
    bb = model.backbone.model

    # --- single backbone pass, with attention capture on -------------------
    _reset_capture(cls_idx=0, do_rollout=do_rollout)
    _CAP["on"] = True
    ff = bb.forward_features(x)
    _CAP["on"] = False
    if isinstance(ff, list):
        ff = ff[0]
    cls = ff["x_norm_clstoken"]                      # (1, D)
    patches = ff["x_norm_patchtokens"]               # (1, N, D)
    n_patch = patches.shape[1]
    g = _grid_from_mask(patch_mask, n_patch)

    pm = patch_mask if (patch_mask is not None and patch_mask.shape[1] == n_patch) else None
    valid = (pm[0] > 0).cpu().numpy() if pm is not None else np.ones(n_patch, bool)

    maps: Dict[str, object] = {"grid": g, "valid": valid}

    # --- CLS self-attention (chosen block): per-head + mean ----------------
    cls_rows: List[torch.Tensor] = _CAP["cls_rows"]              # 24 × (1,H,N)
    blk = block if block >= 0 else len(cls_rows) + block
    blk = max(0, min(blk, len(cls_rows) - 1))
    cls_attn = cls_rows[blk][0]                                  # (H, N)
    cls_attn_p = cls_attn[:, prefix:]                            # (H, N_patch)
    H = cls_attn_p.shape[0]
    per_head = cls_attn_p.reshape(H, g, g).numpy()
    maps["cls_block"] = blk
    maps["cls_attn_heads"] = per_head                            # (H, g, g)
    maps["cls_attn_mean"] = per_head.mean(0)                     # (g, g)

    # --- attention rollout (all blocks) ------------------------------------
    if do_rollout and _CAP["rollout"] is not None:
        roll = _CAP["rollout"][0, 0, prefix:].float().cpu().numpy()   # CLS row → patches
        maps["rollout"] = roll.reshape(g, g)

    # --- PCA of patch tokens (foreground PC1 + top-3 RGB) ------------------
    feats = patches[0].float().cpu().numpy()                    # (N, D)
    fv = feats[valid]
    if fv.shape[0] >= 3:
        from sklearn.decomposition import PCA
        comp = PCA(n_components=3, random_state=0).fit(fv)
        proj_valid = comp.transform(fv)                         # (n_valid, 3)
        proj = np.zeros((n_patch, 3), np.float32)
        proj[valid] = proj_valid
        # PC1 oriented so foreground (high |attn|) tends bright.
        pc1 = proj[:, 0].copy()
        if np.corrcoef(pc1[valid], maps["cls_attn_mean"].reshape(-1)[valid])[0, 1] < 0:
            proj[:, 0] = -proj[:, 0]
            pc1 = proj[:, 0]
        rgb = np.zeros((n_patch, 3), np.float32)
        for c in range(3):
            col = proj[:, c]
            col_v = _norm01(col[valid])
            rgb[valid, c] = col_v
        maps["pca_rgb"] = rgb.reshape(g, g, 3)
        m = np.full(n_patch, np.nan, np.float32)
        m[valid] = pc1[valid]
        maps["pca_pc1"] = m.reshape(g, g)
        maps["pca_var"] = comp.explained_variance_ratio_

    # --- cosine similarity to the head's top-attended patch ----------------
    # query = patch the regression head's AttentionPool focuses on most.
    head_attn_w = None
    if getattr(model, "pooler", None) is not None and getattr(model.pooler, "attn_pool", None) is not None:
        qvec = model.pooler.attn_pool.query                     # (D,)
        logits = torch.einsum("bnd,d->bn", patches, qvec)
        if pm is not None:
            logits = logits.masked_fill(pm <= 0.0, -1e4)
        head_attn_w = logits.softmax(dim=-1)[0].float().cpu().numpy()   # (N_patch,)
        maps["head_attn"] = head_attn_w.reshape(g, g)

    feats_t = F.normalize(patches[0].float(), dim=-1)           # (N, D)
    if head_attn_w is not None:
        q_idx = int(np.argmax(np.where(valid, head_attn_w, -np.inf)))
    else:
        q_idx = int(np.argmax(valid))                            # first valid
    sim = (feats_t @ feats_t[q_idx]).cpu().numpy()              # (N_patch,)
    sim_m = np.full(n_patch, np.nan, np.float32)
    sim_m[valid] = sim[valid]
    maps["cosine"] = sim_m.reshape(g, g)
    maps["cosine_query_rc"] = (q_idx // g, q_idx % g)

    # --- DensityReadout per-patch density (length & area) ------------------
    if getattr(model, "density_head", None) is not None:
        dens = F.softplus(model.density_head.mlp(patches)).float()      # (1,N,2)
        if pm is not None:
            dens = dens * pm.float().unsqueeze(-1)
        dens = dens[0].cpu().numpy()                                    # (N, 2)
        dl = np.where(valid, dens[:, 0], np.nan)
        da = np.where(valid, dens[:, 1], np.nan)
        maps["density_length"] = dl.reshape(g, g)
        maps["density_area"] = da.reshape(g, g)

    # --- prediction (mm), reusing the trained head -------------------------
    pred = _predict(model, cls, patches, pm, cfg, stats)
    maps["pred"] = pred                                          # (length_mm, area_mm)
    return maps


@torch.no_grad()
def _predict(model, cls, patches, pm, cfg, stats) -> Tuple[float, float]:
    """Replicate DinoV3RootRegressor.forward's readout from cached features."""
    out_g = out_d = None
    if getattr(model, "head", None) is not None and getattr(model, "pooler", None) is not None:
        feat = model.pooler(cls, patches, patch_mask=pm)
        out_g = model.head(feat)
    if getattr(model, "density_head", None) is not None:
        out_d = model.density_head(patches, patch_mask=pm)
    if model.readout == "global":
        out = out_g
    elif model.readout == "density":
        out = out_d
    else:
        a = torch.sigmoid(model.readout_blend)
        out = a * out_g + (1.0 - a) * out_d
    if str(cfg.get("target_transform", "zscore")) != "none" and stats:
        out = _invert_targets(out, stats)
    out = out.clamp_min(0.0)[0]
    return float(out[0]), float(out[1])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _overlay(ax, base, heat_grid, side, title, cmap="turbo", alpha=0.55,
             mark_rc=None, patch=16):
    ax.imshow(base)
    heat = _upsample(_norm01(np.nan_to_num(heat_grid, nan=float(np.nanmin(heat_grid)))), side)
    ax.imshow(heat, cmap=cmap, alpha=alpha, extent=(0, side, side, 0))
    if mark_rc is not None:
        r, c = mark_rc
        ax.plot((c + 0.5) * patch, (r + 0.5) * patch, marker="+",
                ms=14, mew=2.5, color="white")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_main(maps, base, side, name, pred, gt, out_path, alpha, cmap, patch):
    panels = []
    panels.append(("input", "image", base))
    panels.append((f"CLS attn (mean, blk {maps['cls_block']})", "overlay", maps["cls_attn_mean"]))
    if "rollout" in maps:
        panels.append(("attention rollout", "overlay", maps["rollout"]))
    if "pca_rgb" in maps:
        var = maps.get("pca_var")
        vtxt = f" ({100*var[:3].sum():.0f}% var)" if var is not None else ""
        panels.append((f"PCA top-3 → RGB{vtxt}", "rgb", maps["pca_rgb"]))
        panels.append(("PCA PC1 (foreground)", "overlay", maps["pca_pc1"]))
    panels.append(("patch cosine sim (★=head query)", "cosine", maps["cosine"]))
    if "head_attn" in maps:
        panels.append(("HEAD AttentionPool weights", "overlay", maps["head_attn"]))
    if "density_length" in maps:
        panels.append(("HEAD density · length", "overlay", maps["density_length"]))
    if "density_area" in maps:
        panels.append(("HEAD density · area", "overlay", maps["density_area"]))

    n = len(panels)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes).reshape(-1)
    for ax, (title, kind, data) in zip(axes, panels):
        if kind == "image":
            ax.imshow(data); ax.set_title(title, fontsize=9); ax.axis("off")
        elif kind == "rgb":
            ax.imshow(np.kron(data, np.ones((patch, patch, 1))))   # nearest upsample
            ax.set_title(title, fontsize=9); ax.axis("off")
        elif kind == "cosine":
            _overlay(ax, base, data, side, title, cmap=cmap, alpha=alpha,
                     mark_rc=maps.get("cosine_query_rc"), patch=patch)
        else:
            _overlay(ax, base, data, side, title, cmap=cmap, alpha=alpha, patch=patch)
    for ax in axes[n:]:
        ax.axis("off")

    gt_txt = ""
    if gt is not None and not (math.isnan(gt[0]) and math.isnan(gt[1])):
        gt_txt = f"   |   true: len={gt[0]:.1f}mm  area={gt[1]:.1f}mm²"
    fig.suptitle(f"{name}    pred: len={pred[0]:.1f}mm  area={pred[1]:.1f}mm²{gt_txt}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_heads(maps, base, side, name, out_path, alpha, cmap, patch):
    heads = maps["cls_attn_heads"]
    H = heads.shape[0]
    ncol = int(math.ceil(math.sqrt(H)))
    nrow = int(math.ceil(H / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow))
    axes = np.atleast_2d(axes).reshape(-1)
    for h in range(H):
        _overlay(axes[h], base, heads[h], side, f"head {h}", cmap=cmap,
                 alpha=alpha, patch=patch)
    for ax in axes[H:]:
        ax.axis("off")
    fig.suptitle(f"{name} — last-block [CLS] attention per head", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Image selection
# --------------------------------------------------------------------------- #
def _pick_images(args, cfg) -> List[str]:
    if args.images:
        return list(args.images)
    csv = args.csv or os.path.join(os.path.dirname(cfg.get("data_path", "")), "test_data.csv")
    paths, labels = load_split(csv, csv, csv, args.data_path)["train"]
    present = (labels[:, 0] > 0) & (labels[:, 1] > 0)
    idx = torch.nonzero(present, as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(len(paths))
    # deterministic spread across the split
    sel = idx[torch.linspace(0, idx.numel() - 1, steps=min(args.n, idx.numel())).long()]
    return [paths[int(i)] for i in sel]


def _gt_lookup(csv, data_path) -> Dict[str, Tuple[float, float]]:
    try:
        paths, labels = load_split(csv, csv, csv, data_path)["train"]
    except Exception:
        return {}
    return {os.path.basename(p): (float(labels[i, 0]), float(labels[i, 1]))
            for i, p in enumerate(paths)}


# --------------------------------------------------------------------------- #
def load_model(checkpoint: str, device: str = "cuda", install_capture: bool = True):
    """Build a RootQuantV2 model from a checkpoint (cfg + weights live in the .pt).

    Mirrors infer.py: local backbone weights (DINOV3_CHECKPOINT_PATH or
    RootQuantV2/dinov3/), strict=False (frozen backbone is rebuilt at
    construction). Returns (model, cfg, stats, resolved_checkpoint).
    Set ``install_capture=False`` to skip the explicit-softmax attention hook
    (faster — only needed for the CLS-attention / rollout maps)."""
    repo = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(checkpoint):
        cand = os.path.join(repo, checkpoint)
        if os.path.isfile(cand):
            checkpoint = cand
    payload = load(checkpoint, map_location="cpu")
    cfg = dict(payload["cfg"])
    cfg["use_torch_compile"] = False                 # never compile for viz/eval
    stats = payload.get("target_stats", {})
    dev = torch.device(device)
    model = DinoV3RootRegressor(cfg).to(dev).eval()
    load_into(checkpoint, model=model, map_location="cpu", strict=False)
    if install_capture:
        # Install AFTER construction: the `dinov3` package is only importable once
        # the backbone is built via torch.hub. Overrides the repo's contiguity patch.
        _install_attention_capture()
    return model, cfg, stats, checkpoint


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Explicit image paths (overrides --csv sampling).")
    ap.add_argument("--csv", default=_CFG["test_csv"])
    ap.add_argument("--data_path", default=_CFG["data_path"])
    ap.add_argument("--n", type=int, default=4, help="How many images to sample.")
    ap.add_argument("--out_dir", default="RootQuantV2/viz_attention")
    ap.add_argument("--block", type=int, default=-1,
                    help="Block for the per-head/mean CLS attention (-1 = last).")
    ap.add_argument("--no-rollout", action="store_true", help="Skip attention rollout.")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg, stats, args.checkpoint = load_model(
        args.checkpoint, device=args.device, install_capture=True)
    patch = int(cfg.get("backbone_patch_size", 16))
    side = int(cfg.get("target_size", 896))
    device = torch.device(args.device)

    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    gt = _gt_lookup(args.csv, args.data_path)
    images = _pick_images(args, cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"checkpoint : {args.checkpoint}")
    print(f"profile    : {cfg.get('profile')}  | {side}px  | pool={cfg.get('pool')} "
          f"readout={cfg.get('readout')}")
    print(f"images     : {len(images)} → {args.out_dir}/")

    for ip in images:
        name = os.path.basename(ip)
        try:
            pil = Image.open(ip).convert("RGB")
        except Exception as e:
            print(f"  !! skip {name}: {e}")
            continue
        out = tf(pil)
        if isinstance(out, (tuple, list)):
            x, patch_mask = out[0], out[1]
            patch_mask = patch_mask.unsqueeze(0).to(device)
        else:
            x, patch_mask = out, None
        x = x.unsqueeze(0).to(device)
        base = _letterbox_display(pil, side)

        maps = compute_maps(model, x, patch_mask, cfg, stats,
                            block=args.block, do_rollout=not args.no_rollout)
        pred = maps["pred"]
        stem = os.path.splitext(name)[0]
        render_main(maps, base, side, name, pred, gt.get(name),
                    os.path.join(args.out_dir, f"{stem}__maps.png"),
                    args.alpha, args.cmap, patch)
        render_heads(maps, base, side, name,
                     os.path.join(args.out_dir, f"{stem}__heads.png"),
                     args.alpha, args.cmap, patch)
        print(f"  ✓ {name}  pred len={pred[0]:.1f}mm area={pred[1]:.1f}mm²")

    print("done.")


if __name__ == "__main__":
    main()
