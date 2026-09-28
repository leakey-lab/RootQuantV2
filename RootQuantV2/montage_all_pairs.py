"""montage_all_pairs.py — full contact sheet of ALL likely-mislabeled empties.

For every gt-empty tube (length == area == 0) whose model density says a root is
present, render **[ input | head density·length overlay ]** side by side. Unlike
`montage_density.py` (a small top-N summary slide), this is the complete review
sheet for re-annotation, paginated into one multi-page PDF.

Selection uses the SAME documented column as `montage_density.py`:
`density_length_mm >= --threshold` (default 5 mm).
Rationale (see montage_density.py / docs/visualization.md): density over pred
(`readout="both"` down-weights density, and density is the drawn, spatially
grounded quantity); length over area (presence is 1-D; length is better calibrated).

All overlays share one absolute color scale, opacity ∝ density.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.montage_all_pairs \
        --out RootQuantV2/viz_analysis/montage_mislabeled_all.pdf
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import get_transforms
from RootQuantV2.data.transforms import LetterboxToSquare
from RootQuantV2.visualize_attention import load_model
from RootQuantV2.montage_density import _density_and_pred, _short, _select


def _disp_base(pil: Image.Image, side: int, disp: int) -> np.ndarray:
    """Letterbox to the model canvas, then shrink to a compact display tile."""
    sq = LetterboxToSquare(target=side, fill=0).letterbox(pil)[0]
    return np.asarray(sq.convert("RGB").resize((disp, disp)), dtype=np.uint8)


def _pair_overlay(ax, base_disp, grid, disp, vmax, cmap, max_alpha=0.9):
    """Overlay the density grid on the (already disp×disp) base; matched sizes so
    default extents align. Opacity ∝ density, shared absolute vmax."""
    ax.imshow(base_disp)
    heat = F.interpolate(torch.from_numpy(np.ascontiguousarray(grid))[None, None].float(),
                         size=(disp, disp), mode="bilinear", align_corners=False)[0, 0].numpy()
    alpha = np.clip(heat / max(vmax, 1e-6), 0.0, 1.0) * max_alpha
    ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=vmax, alpha=alpha)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--ranking_csv", default="viz_analysis/empty_density_ranking.csv")
    ap.add_argument("--rank_col", default="density_length_mm")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--map", default="length", choices=["length", "area"])
    ap.add_argument("--pairs_per_row", type=int, default=3)
    ap.add_argument("--rows_per_page", type=int, default=7)
    ap.add_argument("--disp", type=int, default=384, help="Display tile size (px).")
    ap.add_argument("--limit", type=int, default=0, help="Cap candidates (0 = all).")
    ap.add_argument("--data_path", default=_CFG["data_path"])
    ap.add_argument("--out", default="RootQuantV2/viz_analysis/montage_mislabeled_all.pdf")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    repo = os.path.dirname(os.path.abspath(__file__))

    def _resolve(p):
        # Input path as given, else relative to the package dir (tolerating a
        # leading "RootQuantV2/").
        if os.path.isabs(p) or os.path.isfile(p):
            return p
        parts = [q for q in Path(p).parts if q]
        if parts and parts[0] == os.path.basename(repo):
            parts = parts[1:]
        return os.path.join(repo, *parts) if parts else p

    rk = _resolve(args.ranking_csv)
    if not os.path.isfile(rk):
        raise SystemExit(f"ranking csv not found: {args.ranking_csv}")
    sel, n_total = _select(rk, args.rank_col, args.threshold, 10 ** 9)
    if args.limit > 0:
        sel = sel[:args.limit]
    print(f"selection  : {args.rank_col} >= {args.threshold} → {n_total} mislabeled "
          f"empties; rendering {len(sel)} as [input | density·{args.map}] pairs")
    if not sel:
        raise SystemExit(f"no rows with {args.rank_col} >= {args.threshold}; nothing to draw.")

    model, cfg, stats, _ = load_model(args.checkpoint, device=args.device, install_capture=False)
    if cfg.get("readout") != "both":
        raise SystemExit(f"checkpoint readout={cfg.get('readout')!r}; this contact sheet "
                         f"needs readout='both' (density head + global head + blend).")
    side = int(cfg.get("target_size", 896))
    device = torch.device(args.device)
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)

    # Pass 1: compute every density map + prediction; cache compact display tiles.
    cells = []
    for i, (rank, img, val, dlen, plen) in enumerate(sel):
        path = img if os.path.isabs(img) else os.path.join(args.data_path, img)
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  !! skip #{rank} {os.path.basename(path)}: {e}")
            continue
        o = tf(pil)
        if isinstance(o, (tuple, list)):
            x, pm = o[0].unsqueeze(0).to(device), o[1].unsqueeze(0).to(device)
        else:
            x, pm = o.unsqueeze(0).to(device), None
        dl, da, pred = _density_and_pred(model, x, pm, cfg, stats)
        grid = dl if args.map == "length" else da
        cells.append((rank, _short(path), _disp_base(pil, side, args.disp), grid, dlen, plen))
        if (i + 1) % 25 == 0 or (i + 1) == len(sel):
            print(f"  computed {i + 1}/{len(sel)}")

    if not cells:
        raise SystemExit("no candidate image could be opened; nothing to draw.")
    allv = np.concatenate([np.nan_to_num(c[3], nan=0.0).ravel() for c in cells])
    vmax = float(np.percentile(allv[allv > 0], 99)) if (allv > 0).any() else 1.0

    # Pass 2: paginate into a single multi-page PDF.
    ppr = args.pairs_per_row
    rpp = args.rows_per_page
    per_page = ppr * rpp
    ncols = ppr * 2
    npages = math.ceil(len(cells) / per_page)
    out = args.out                                   # relative to the working directory
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    with PdfPages(out) as pdf:
        for pg in range(npages):
            chunk = cells[pg * per_page:(pg + 1) * per_page]
            fig, axes = plt.subplots(rpp, ncols, figsize=(2.0 * ncols, 2.35 * rpp))
            axes = np.atleast_2d(axes)
            for ax in axes.ravel():
                ax.axis("off")
            for i, (rank, short, base, grid, dlen, plen) in enumerate(chunk):
                r = i // ppr
                c = (i % ppr) * 2
                ax_img, ax_ovl = axes[r, c], axes[r, c + 1]
                ax_img.imshow(base)
                ax_img.set_title(f"#{rank}  {short}", fontsize=6.5)
                ax_img.axis("off")
                _pair_overlay(ax_ovl, base, grid, args.disp, vmax, args.cmap)
                ax_ovl.set_title(f"density·{args.map}\ndl={dlen:.0f}  pl={plen:.0f}", fontsize=6.5)
            fig.suptitle(
                f"Likely-MISLABELED empties (gt length=area=0, {args.rank_col} ≥ "
                f"{args.threshold:g}mm) — {n_total} total · page {pg + 1}/{npages}\n"
                f"left = input · right = head density·{args.map} (shared vmax={vmax:.1f}, opacity ∝ density)",
                fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
            print(f"  page {pg + 1}/{npages} written")

    print(f"wrote {out}  ({len(cells)} candidates, {npages} pages)")


if __name__ == "__main__":
    main()
