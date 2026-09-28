"""montage_density.py — montage of likely-MISLABELED empty tubes.

Builds ONE montage of tubes whose ground truth is empty (length == area == 0)
but where the model says a root is present — i.e. probable annotation errors.
Candidates and their order come from `empty_density_ranking.csv` (produced by
`rank_empty_density.py`, which only scores gt-empty rows). Each cell shows the
head DensityReadout map over the input, so you literally see the root the model
found on a tube the label calls empty.

WHY filter on `density_length_mm` (documented choice)
-----------------------------------------------------
The CSV offers four numeric columns: density_{length,area} and pred_{length,area}.
We filter/rank on **density_length_mm**:

* density vs pred — the model runs `readout="both"`: the final `pred_*` is a blend
  of a global CLS-pooled head and the DensityReadout, and the learned blend
  *down-weights* density. So `pred_*` is a damped view of the spatial evidence.
  For *surfacing* missed roots we want maximum sensitivity to "structure was
  found", which the density branch gives — its value on a flagged tube typically
  runs well above the blended `pred_*`. It is also what this montage VISUALIZES — a
  masked sum of the per-patch density map — so a tube is shown *because* its
  displayed map is bright (criterion == picture), and density is spatially
  grounded (localized root mass, not a diffuse global guess).
* length vs area — root *presence* is a 1-D/length phenomenon (a root is a curve;
  length measures traced extent). Area conflates extent with thickness and is more
  easily inflated by diffuse bright blobs (condensation, scratches, soil mottling),
  making it a noisier presence detector. Length is also the better-calibrated of
  the two targets, so a high length signal is the more trustworthy "real root".

Threshold: density_length_mm >= 5.0 mm == "model predicts a root present".
The montage shows the top-N candidates above that threshold. Needs a checkpoint
with ``readout="both"``.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.montage_density \
        --ranking_csv RootQuantV2/viz_analysis/empty_density_ranking.csv \
        --out RootQuantV2/viz_analysis/montage_mislabeled.png --n 15
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Tuple

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
from RootQuantV2.visualize_attention import load_model, _letterbox_display, _upsample


@torch.no_grad()
def _density_and_pred(model, x, pm, cfg, stats):
    bb = model.backbone.model
    ff = bb.forward_features(x)
    if isinstance(ff, list):
        ff = ff[0]
    cls = ff["x_norm_clstoken"]
    patches = ff["x_norm_patchtokens"]
    n = patches.shape[1]
    g = int(round(n ** 0.5))
    dens = F.softplus(model.density_head.mlp(patches)).float()
    if pm is not None:
        dens = dens * pm.float().unsqueeze(-1)
    d = dens[0].cpu().numpy()
    valid = (pm[0] > 0).cpu().numpy() if pm is not None else np.ones(n, bool)
    dl = np.where(valid, d[:, 0], 0.0).reshape(g, g)
    da = np.where(valid, d[:, 1], 0.0).reshape(g, g)
    feat = model.pooler(cls, patches, patch_mask=pm)
    gg = model.head(feat)
    dstd = model.density_head(patches, patch_mask=pm)
    a = torch.sigmoid(model.readout_blend)
    out = a * gg + (1.0 - a) * dstd
    if str(cfg.get("target_transform", "zscore")) != "none" and stats:
        out = _invert_targets(out, stats)
    out = out.clamp_min(0.0)[0]
    return dl, da, (float(out[0]), float(out[1]))


def _overlay_density(ax, base, grid, side, title, vmax, cmap, max_alpha=0.9):
    """Overlay with a SHARED vmax and density-proportional opacity, so genuinely
    weak maps stay transparent rather than being stretched to look bright."""
    ax.imshow(base)
    heat = _upsample(np.nan_to_num(grid, nan=0.0), side)
    alpha = np.clip(heat / max(vmax, 1e-6), 0.0, 1.0) * max_alpha
    ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=vmax, alpha=alpha,
              extent=(0, side, side, 0))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _short(path: str) -> str:
    p = os.path.basename(path).split("_")
    return f"{p[0][:9]} {p[1]} {p[2]}" if len(p) >= 3 else os.path.basename(path)


def _select(ranking_csv: str, rank_col: str, threshold: float, n: int):
    rows = []
    with open(ranking_csv) as f:
        rd = csv.DictReader(f)
        if rank_col not in rd.fieldnames:
            raise SystemExit(f"--rank_col {rank_col!r} not in CSV columns {rd.fieldnames}")
        for r in rd:
            v = float(r[rank_col])
            if v >= threshold:
                rows.append((int(r["rank"]), r["image"], v,
                             float(r["density_length_mm"]), float(r["pred_length_mm"])))
    rows.sort(key=lambda x: -x[2])
    return rows[:n], len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--ranking_csv", default="viz_analysis/empty_density_ranking.csv")
    ap.add_argument("--rank_col", default="density_length_mm",
                    help="CSV column used to filter/rank (documented default: density_length_mm).")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="Min rank_col value to count as 'model predicts a root'.")
    ap.add_argument("--n", type=int, default=15, help="How many top candidates to show.")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--map", default="length", choices=["length", "area"],
                    help="Which density map to display (default length, matching rank_col).")
    ap.add_argument("--data_path", default=_CFG["data_path"])
    ap.add_argument("--out", default="RootQuantV2/viz_analysis/montage_mislabeled.png")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    repo = os.path.dirname(os.path.abspath(__file__))
    rk = args.ranking_csv if os.path.isfile(args.ranking_csv) else os.path.join(repo, args.ranking_csv)
    if not os.path.isfile(rk):
        raise SystemExit(f"ranking csv not found: {args.ranking_csv}")
    sel, n_total = _select(rk, args.rank_col, args.threshold, args.n)
    print(f"selection  : {args.rank_col} >= {args.threshold} → {n_total} mislabeled "
          f"candidates; showing top {len(sel)}")
    if not sel:
        raise SystemExit(f"no rows with {args.rank_col} >= {args.threshold}; nothing to draw.")

    model, cfg, stats, _ = load_model(args.checkpoint, device=args.device, install_capture=False)
    if cfg.get("readout") != "both":
        raise SystemExit(f"checkpoint readout={cfg.get('readout')!r}; this montage needs "
                         f"readout='both' (density head + global head + blend).")
    side = int(cfg.get("target_size", 896))
    device = torch.device(args.device)
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)

    cells = []
    for rank, img, val, dlen, plen in sel:
        path = img if os.path.isabs(img) else os.path.join(args.data_path, img)
        pil = Image.open(path).convert("RGB")
        o = tf(pil)
        if isinstance(o, (tuple, list)):
            x, pm = o[0].unsqueeze(0).to(device), o[1].unsqueeze(0).to(device)
        else:
            x, pm = o.unsqueeze(0).to(device), None
        dl, da, pred = _density_and_pred(model, x, pm, cfg, stats)
        grid = dl if args.map == "length" else da
        cells.append((rank, _short(path), _letterbox_display(pil, side), grid, dlen, plen))
        print(f"  ✓ #{rank:<4} dlen={dlen:6.1f}  plen={plen:6.1f}  {os.path.basename(path)}")

    # shared absolute scale across all cells (robust 99th pct of positive density)
    allv = np.concatenate([np.nan_to_num(c[3], nan=0.0).ravel() for c in cells])
    vmax = float(np.percentile(allv[allv > 0], 99)) if (allv > 0).any() else 1.0

    cols = args.cols
    rows = math.ceil(len(cells) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.05 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes).reshape(-1)
    for ax, (rank, short, base, grid, dlen, plen) in zip(axes, cells):
        _overlay_density(ax, base, grid, side,
                         f"#{rank} {short}\ndlen={dlen:.0f}  plen={plen:.0f}",
                         vmax, args.cmap)
    for ax in axes[len(cells):]:
        ax.axis("off")

    fig.suptitle(
        f"Likely-MISLABELED empty tubes — gt length=area=0 but model finds root\n"
        f"selected & ranked by {args.rank_col} ≥ {args.threshold:g}mm "
        f"({n_total} such empties; top {len(cells)} shown) · overlay = head density·{args.map}, "
        f"shared scale vmax={vmax:.1f}, opacity ∝ density",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out                                   # relative to the working directory
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
