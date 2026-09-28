"""viz_mona_pair.py — paired Mona-ON vs Mona-OFF density comparison.

Question being probed: *does the model internally localize the root (solve the
"root puzzle") and is Mona — the locality-injecting conv adapter — what builds
that internal spatial representation?*

The task-faithful signal is the trained ``DensityReadout`` per-patch density
(its masked SUM is the prediction), NOT the backbone's attention/PCA maps
(task-agnostic: they segment any texture, soil included — see
docs/visualization.md). So this
script renders ONLY the head density, for two checkpoints that differ by exactly
one bit — Mona on vs off (both DoRA-on, frozen ViT-L, v3 profile) — on the SAME
images, with:

  * a SHARED per-image color scale across the two models (a fair magnitude+shape
    comparison; per-map min-max would stretch a weak map to look bright), and
  * a Δ = (ON − OFF) panel on a diverging scale — red = density Mona ADDS,
    blue = density Mona removes. This panel is the direct picture of "what Mona
    contributes to the internal root localization".

Run from the repository root so RootQuantV2 imports as a package. Sampling
images from the test split needs ROOTQUANT_DATA_DIR (or --csv / --data_path);
--images works without a dataset:

    python -m RootQuantV2.viz_mona_pair \
        --n_present 4 --map length --out RootQuantV2/viz_mona_pair

Default checkpoints: the frozen-v3 Mona ablation pair
  ON : RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt
  OFF: RootQuantV2/runs/checkpoints/rootquant-v2-dora-only/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import get_transforms
from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.visualize_attention import (
    load_model, _letterbox_display, _pick_images, _gt_lookup, _upsample,
)

TARGET_IDX = {"length": 0, "area": 1}


@torch.no_grad()
def density_and_pred(model, x, pm, cfg, stats, which: int):
    """Per-patch density grid (g,g) for one target + the density head's own
    (length, area) prediction in mm. Pure forward_features path (no attention
    capture) so it keeps the fast SDPA kernel."""
    bb = model.backbone.model
    ff = bb.forward_features(x)
    if isinstance(ff, list):
        ff = ff[0]
    patches = ff["x_norm_patchtokens"]                       # (1, N, D)
    dens = F.softplus(model.density_head.mlp(patches)).float()  # (1, N, 2)
    if pm is not None:
        dens = dens * pm.float().unsqueeze(-1)
    grid = dens[0].cpu().numpy()                             # (N, 2)
    n_patch = grid.shape[0]
    g = int(math.isqrt(n_patch))
    valid = (pm[0] > 0).cpu().numpy() if pm is not None else np.ones(n_patch, bool)
    m = np.where(valid, grid[:, which], np.nan).reshape(g, g)

    out_d = model.density_head(patches, patch_mask=pm)
    if str(cfg.get("target_transform", "zscore")) != "none" and stats:
        out_d = _invert_targets(out_d, stats)
    out_d = out_d.clamp_min(0.0)[0]
    return m, (float(out_d[0]), float(out_d[1]))


def _overlay_scaled(ax, base, grid, side, vmin, vmax, cmap, alpha, title):
    heat = _upsample(np.nan_to_num(grid, nan=0.0), side)
    ax.imshow(base)
    ax.imshow(heat, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
              extent=(0, side, side, 0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render(rows, side, which_name, out_path, alpha):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4, 4 * n), squeeze=False)
    for r, row in enumerate(rows):
        base = row["base"]
        on, off = row["on"], row["off"]
        shared = float(np.nanmax([np.nanmax(on), np.nanmax(off), 1e-9]))
        d = np.nan_to_num(on, nan=0.0) - np.nan_to_num(off, nan=0.0)
        dmax = float(np.max(np.abs(d))) or 1e-9

        axes[r][0].imshow(base)
        gt = row["gt"]
        gt_txt = ("EMPTY (gt=0)" if gt is not None and gt[0] == 0 and gt[1] == 0
                  else (f"gt len={gt[0]:.1f}" if gt is not None else "gt ?"))
        axes[r][0].set_title(f"{row['name'][:34]}\n{gt_txt}", fontsize=8)
        axes[r][0].axis("off")

        _overlay_scaled(axes[r][1], base, on, side, 0.0, shared, "turbo", alpha,
                        f"Mona-ON · {which_name}\nŷ={row['on_pred']:.1f}mm")
        _overlay_scaled(axes[r][2], base, off, side, 0.0, shared, "turbo", alpha,
                        f"Mona-OFF · {which_name}\nŷ={row['off_pred']:.1f}mm")
        _overlay_scaled(axes[r][3], base, d, side, -dmax, dmax, "RdBu_r", alpha,
                        "Δ (ON−OFF)\nred=Mona adds")

    fig.suptitle(f"Internal root localization — head density · {which_name}: "
                 f"Mona ON vs OFF (shared scale per row)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_on",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--ckpt_off",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-dora-only/best.pt")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Explicit image paths (overrides sampling).")
    ap.add_argument("--extra_images", nargs="*", default=None,
                    help="Extra images appended after the sampled present ones "
                         "(e.g. mislabeled-empty tubes).")
    ap.add_argument("--csv", default=_CFG["test_csv"])
    ap.add_argument("--data_path", default=_CFG["data_path"])
    ap.add_argument("--n_present", type=int, default=4,
                    help="How many present-root images to sample from the split.")
    ap.add_argument("--map", choices=["length", "area", "both"], default="length")
    ap.add_argument("--out", default="RootQuantV2/viz_mona_pair",
                    help="Output directory.")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # No attention capture needed — density uses forward_features only.
    m_on, cfg, stats_on, _ = load_model(args.ckpt_on, device=args.device, install_capture=False)
    m_off, cfg_off, stats_off, _ = load_model(args.ckpt_off, device=args.device, install_capture=False)
    for m in (m_on, m_off):
        if getattr(m, "density_head", None) is None:
            raise SystemExit("A checkpoint has no density_head (readout != both/density).")

    side = int(cfg.get("target_size", 896))
    device = torch.device(args.device)
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    gt = _gt_lookup(args.csv, args.data_path)

    class _A: pass
    a = _A(); a.images = args.images; a.n = args.n_present
    a.csv = args.csv; a.data_path = args.data_path
    images = list(args.images) if args.images else _pick_images(a, cfg)
    if args.extra_images:
        images += list(args.extra_images)
    os.makedirs(args.out, exist_ok=True)

    print(f"ON  : {args.ckpt_on}")
    print(f"OFF : {args.ckpt_off}")
    print(f"{side}px | pool={cfg.get('pool')} readout={cfg.get('readout')} | {len(images)} images")

    targets = ["length", "area"] if args.map == "both" else [args.map]
    collected = {t: [] for t in targets}
    for ip in images:
        name = os.path.basename(ip)
        try:
            pil = Image.open(ip).convert("RGB")
        except Exception as e:
            print(f"  !! skip {name}: {e}")
            continue
        out = tf(pil)
        if isinstance(out, (tuple, list)):
            x, pm = out[0].unsqueeze(0).to(device), out[1].unsqueeze(0).to(device)
        else:
            x, pm = out.unsqueeze(0).to(device), None
        base = _letterbox_display(pil, side)
        for t in targets:
            wi = TARGET_IDX[t]
            on_m, on_pred = density_and_pred(m_on, x, pm, cfg, stats_on, wi)
            off_m, off_pred = density_and_pred(m_off, x, pm, cfg_off, stats_off, wi)
            collected[t].append(dict(name=name, base=base, gt=gt.get(name),
                                     on=on_m, off=off_m,
                                     on_pred=on_pred[wi], off_pred=off_pred[wi]))
        op = collected[targets[0]][-1]
        print(f"  ✓ {name[:40]:40}  ON {op['on_pred']:7.1f} | OFF {op['off_pred']:7.1f}  (Δ {op['on_pred']-op['off_pred']:+.1f}mm {targets[0]})")

    for t in targets:
        render(collected[t], side, t, os.path.join(args.out, f"mona_pair_{t}.png"), args.alpha)
    print("done.")


if __name__ == "__main__":
    main()
