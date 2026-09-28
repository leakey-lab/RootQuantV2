#!/usr/bin/env python3
"""gen_letterbox_augment.py — the preprocessing / augmentation figure (paper Fig. 2).

Model-free: needs one input frame, no checkpoint. Draws the raw -> letterbox top
row and the training-augmentation panels, including the tile-shuffle
augmentation family at grids 2, 4, 8.

Three augmentation families, colour-coded by panel border:
    D4 dihedral   (green) : identity, rot 90, hflip
    photometric   (grey)  : colour jitter + occasional blur
    tile shuffle  (amber) : 2x2, 4x4, 8x8 tile permutation

Every panel shows ONE augmentation in isolation on the same letterboxed 896x896
frame, faithful to data/transforms.py (tile shuffle reuses TileShuffle's exact
reshape/permute; D4 reuses torch.rot90/hflip; photometric reuses Photometric).

Run from the repo root:
    python -m RootQuantV2.gen_letterbox_augment --image frame.jpg
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))                    # import data.transforms directly

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from data.transforms import (
    LetterboxToSquare,
    content_patch_mask_2d,
    Photometric,
)

# ---- patch-mask colours (per-token content/padding coding, all panels) ------
# Content (mask=1): green border, transparent inside so the root shows through.
# Padding (mask=0): red border with black zero-pad inside.
C_CONTENT = (0.16, 0.80, 0.24, 1.0)
C_PAD = (0.90, 0.18, 0.18, 1.0)
AMBER = "#e08214"      # tile-shuffle lattice lines
# Coarsen the true 56x56 patch mask by this factor for a LEGIBLE grid. This is a
# visualization only, so a 14x14 (64px) lattice reads cleanly without hazing the
# roots; the model still uses the full 16px mask.
MASK_VIZ_FACTOR = 4

TARGET = 896           # letterbox canvas side
PATCH = 16             # ViT-L/16 patch size -> 56x56 token grid
TILE_GRIDS = (2, 4, 8) # tile-shuffle scales to depict


# ---------------------------------------------------------------------------
# faithful tile-shuffle application (mirrors TileShuffle._shuffle_at exactly)
# ---------------------------------------------------------------------------
def apply_tile_perm(img: torch.Tensor, grid: int, perm: torch.Tensor) -> torch.Tensor:
    """Permute a grid x grid tile lattice of ``img`` [C,H,W] by ``perm``.

    Identical reshape/permute algebra to data/transforms.py::TileShuffle._shuffle_at,
    but the permutation is injected (so the figure is reproducible and legible)
    instead of drawn inside the call.
    """
    c, h, w = img.shape
    if h % grid or w % grid:
        raise ValueError(f"{h}x{w} not divisible by tile grid {grid}")
    th, tw = h // grid, w // grid
    tiles = (
        img.reshape(c, grid, th, grid, tw)
           .permute(1, 3, 0, 2, 4)
           .contiguous()
           .reshape(grid * grid, c, th, tw)
    )
    tiles = tiles[perm]
    return (
        tiles.reshape(grid, grid, c, th, tw)
             .permute(2, 0, 3, 1, 4)
             .contiguous()
             .reshape(c, h, w)
    )


def pick_tile_permutation(grid: int, gen: torch.Generator) -> torch.Tensor:
    """Choose which permutation of the grid*grid tiles to DISPLAY for this scale.

    Returns a 1-D ``torch.LongTensor`` of length ``grid*grid`` that is a
    permutation of ``range(grid*grid)``, reproducible from ``gen``.

    The training sampler is just ``torch.randperm(grid*grid)``
    (data/transforms.py::TileShuffle), but a uniform random draw can render a 2x2
    grid as a near-identity (e.g. one adjacent swap), which reads poorly in a
    figure. For legibility the figure therefore shows a derangement: no tile
    keeps its slot.
    """
    n = grid * grid
    ident = torch.arange(n)
    # Rejection-sample a derangement: no tile may keep its slot, so even the 2x2
    # grid visibly scrambles. Bounded loop; fall back to a single-step cyclic
    # shift (always a derangement for n>=2) if the draws are unlucky.
    for _ in range(1000):
        perm = torch.randperm(n, generator=gen)
        if not torch.any(perm == ident):
            return perm
    return torch.roll(ident, shifts=1)


# ---------------------------------------------------------------------------
def to_disp(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] in [0,1] -> HWC uint-safe float array for imshow."""
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def draw_lattice(ax, size: int, step: int, color: str, lw: float, alpha: float):
    """Overlay a square lattice (every ``step`` px) on an [0,size] extent panel."""
    for k in range(0, size + 1, step):
        ax.plot([k, k], [0, size], color=color, lw=lw, alpha=alpha)
        ax.plot([0, size], [k, k], color=color, lw=lw, alpha=alpha)


def draw_mask_grid(ax, m: np.ndarray, lw: float = 0.9) -> None:
    """Overlay the per-token validity grid on an [0,TARGET] panel.

    ``m`` is the (56, 56) patch mask (True = content). It is coarsened by
    ``MASK_VIZ_FACTOR`` (a coarse cell is content if ANY of its patches are, so
    no root pixel is blacked out) and every coarse cell is drawn: content ->
    green border, transparent inside (root shows); padding -> red border, black
    zero-pad inside. Coarsening keeps the lattice sparse so it does not haze the
    roots the way the true 56x56 grid did.
    """
    m = np.asarray(m, bool)
    F = MASK_VIZ_FACTOR
    g = m.shape[0] // F
    m = m[:g * F, :g * F].reshape(g, F, g, F).any(axis=(1, 3))   # coarse content-if-any
    cs = TARGET // g
    # Black padding fill sits LOW (z=2) so the amber tiling lattice (z=4) draws
    # over it; the green/red edges sit HIGH (z=5) so they overlay the amber.
    fills, edge_rects, edge_cols = [], [], []
    for i in range(g):
        for j in range(g):
            edge_rects.append(Rectangle((j * cs, i * cs), cs, cs))
            if m[i, j]:
                edge_cols.append(C_CONTENT)                            # content, transparent
            else:
                edge_cols.append(C_PAD)
                fills.append(Rectangle((j * cs, i * cs), cs, cs))      # padding, black
    if fills:
        fc = PatchCollection(fills, match_original=False)
        fc.set_facecolor((0, 0, 0, 1)); fc.set_edgecolor("none"); fc.set_zorder(2)
        ax.add_collection(fc)
    ec = PatchCollection(edge_rects, match_original=False)
    ec.set_facecolor("none"); ec.set_edgecolor(edge_cols); ec.set_linewidth(lw)
    ec.set_zorder(5)
    ax.add_collection(ec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Minirhizotron frame to draw.")
    ap.add_argument("--out",
        default=str(REPO / "viz_analysis" / "letterbox_augment.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # ---- letterbox the raw frame to 896x896 (faithful geometry) -------------
    pil = Image.open(args.image).convert("RGB")
    W, H = pil.size
    lb = LetterboxToSquare(target=TARGET, fill=0)
    sq_pil, new_w, new_h, left, top = lb.letterbox(pil)
    mask2d = content_patch_mask_2d(new_w, new_h, left, top, TARGET, PATCH)  # (56,56)

    base_t = T.ToTensor()(sq_pil)                        # [3,896,896] in [0,1]
    raw_arr = np.asarray(pil, np.float32) / 255.0
    lb_arr = to_disp(base_t)

    # ---- augmentation panels (each an isolated single augmentation) ---------
    # Panel tuple: (name, display image, per-token mask, tile-grid or None). The
    # mask is transformed by the SAME op as the image so the green/red grid tracks
    # where content and padding land; tile-grid draws the amber tiling lattice.
    m_base = mask2d
    aug_d4 = [
        ("identity", to_disp(base_t), m_base, None),
        ("rot 90",   to_disp(torch.rot90(base_t, 1, dims=[1, 2])),
                     torch.rot90(m_base, 1, dims=[0, 1]), None),
        ("hflip",    to_disp(TF.hflip(base_t)), torch.flip(m_base, dims=[1]), None),
    ]
    photo_pil = Photometric()(sq_pil)                    # jitter on PIL, as in training
    aug_photo = [("photometric", to_disp(T.ToTensor()(photo_pil)), m_base, None)]

    aug_tiles = []
    for g in TILE_GRIDS:
        perm = pick_tile_permutation(g, gen)
        shuf = apply_tile_perm(base_t, g, perm)
        # permute the mask with the SAME tile perm so green tracks the roots and
        # red tracks the black zero-pad tiles (figure aid; training does not
        # permute the mask — see config.py tile-shuffle note).
        m_shuf = apply_tile_perm(m_base.unsqueeze(0).float(), g, perm)[0] > 0.5
        aug_tiles.append((f"tile {g}x{g}", to_disp(shuf), m_shuf, g))

    row = aug_d4 + aug_photo + aug_tiles     # 7 panels, single row

    # ======================= deterministic inch layout =======================
    LM = RM = BM = 0.22
    TM = 0.55                                    # room for top-row titles
    TH = 3.00                                    # top-row panel side (square)
    raw_w = TH                                     # raw panel same square as letterbox
    lb_w = TH                                     # letterbox is square
    arrow_gap = 1.25
    top_w = raw_w + arrow_gap + lb_w

    AP = 1.72                                     # aug panel side
    gxa = 0.48                                    # inter-panel gap (room for big labels)
    lab_h = 0.58                                  # label strip under each aug panel
    row_w = 7 * AP + 6 * gxa                       # all augmentations in one row

    g_top_hdr, hdr_h, g_hdr_r1 = 0.50, 0.15, 0.52

    content_w = max(top_w, row_w)
    figw = LM + content_w + RM
    figh = (TM + TH + g_top_hdr + hdr_h + g_hdr_r1
            + (AP + lab_h) + BM)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(figw, figh))

    def add_ax(x, y, w, h):
        return fig.add_axes([x / figw, y / figh, w / figw, h / figh])

    # ---- top row: raw frame + letterbox with mask overlay -------------------
    top_x0 = LM + (content_w - top_w) / 2
    y_top = figh - TM - TH
    ax_raw = add_ax(top_x0, y_top, raw_w, TH)
    ax_lb = add_ax(top_x0 + raw_w + arrow_gap, y_top, lb_w, TH)
    top_ymid = y_top + TH / 2

    # Show the raw frame at the SAME scale and position as the letterbox content:
    # scale to width TARGET and centre vertically in the same 896-square data frame.
    # The root content then renders identically in both panels; only the padding
    # treatment differs (blank here, red mask=0 next).
    raw_h = TARGET * H / W
    raw_y0 = (TARGET - raw_h) / 2
    ax_raw.imshow(raw_arr, extent=(0, TARGET, raw_y0 + raw_h, raw_y0))
    ax_raw.set_xlim(0, TARGET); ax_raw.set_ylim(TARGET, 0)
    ax_raw.set_aspect("equal")
    ax_raw.set_xticks([]); ax_raw.set_yticks([])
    for s in ax_raw.spines.values():           # no box around the raw frame
        s.set_visible(False)
    ax_raw.set_title("Raw frame", fontsize=24, pad=8)
    # dimension caption in the blank margin directly under the raw image
    ax_raw.text(TARGET / 2, (raw_y0 + raw_h + TARGET) / 2, f"{W}×{H} px  (rectangular)",
                color="0.15", fontsize=20, va="center", ha="center")

    # letterbox panel in pixel coords (origin upper-left)
    ax_lb.imshow(lb_arr, extent=(0, TARGET, TARGET, 0))
    ax_lb.set_xlim(0, TARGET); ax_lb.set_ylim(TARGET, 0)
    ax_lb.set_xticks([]); ax_lb.set_yticks([])
    ax_lb.set_title("Letterbox → 896²  (zero-pad)", fontsize=24, pad=8)
    # --- per-token validity grid --------------------------------------------
    # content (mask=1) -> green border, transparent inside (root shows);
    # padding (mask=0) -> red border, black zero-pad inside. Exactly the
    # content_patch_mask_2d the DataLoader builds; masked tokens are dropped from
    # the pool and density sum (model.py), so they never contribute to training.
    mask_np = mask2d.numpy()
    draw_mask_grid(ax_lb, mask_np, lw=0.5)

    # ---- augmentation sub-rows ---------------------------------------------
    def place_row(items, row_w, y_bottom):
        x0 = LM + (content_w - row_w) / 2
        for k, (name, arr, mask, tgrid) in enumerate(items):
            ax = add_ax(x0 + k * (AP + gxa), y_bottom, AP, AP)
            ax.imshow(arr, extent=(0, TARGET, TARGET, 0))
            ax.set_xlim(0, TARGET); ax.set_ylim(TARGET, 0)
            draw_mask_grid(ax, np.asarray(mask))       # green content / red padding
            if tgrid is not None:                      # amber tiling lattice (under mask edges)
                step = TARGET // tgrid
                for t in range(step, TARGET, step):
                    ax.plot([t, t], [0, TARGET], color=AMBER, lw=1.8, alpha=0.95, zorder=4)
                    ax.plot([0, TARGET], [t, t], color=AMBER, lw=1.8, alpha=0.95, zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(name, fontsize=26, labelpad=8)

    top_used = TM + TH + g_top_hdr + hdr_h + g_hdr_r1
    y_row = figh - top_used - AP
    place_row(row, row_w, y_row)

    # ---- overlay: arrows, header text, colour key ---------------------------
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    ov.axis("off"); ov.set_zorder(20); ov.patch.set_alpha(0)
    fx = lambda xin: xin / figw
    fy = lambda yin_from_top: 1 - yin_from_top / figh

    # "letterbox" arrow between the two top panels
    ax1 = top_x0 + raw_w + 0.12
    ax2 = top_x0 + raw_w + arrow_gap - 0.12
    ov.annotate("", xy=(fx(ax2), top_ymid / figh), xytext=(fx(ax1), top_ymid / figh),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="0.3"))

    # per-token mask key to the right of the letterbox panel (first row)
    kx = fx(top_x0 + top_w + 0.30)
    swx, swy = 0.34 / figw, 0.34 / figh
    yc = fy(TM + TH / 2)
    dy = 0.65 / figh
    key = [(yc + dy, C_CONTENT, "content (mask=1)\n→ trained"),
           (yc - dy, C_PAD, "padding (mask=0)\n→ dropped")]
    for yy, color, label in key:
        ov.add_patch(Rectangle((kx, yy - swy / 2), swx, swy,
                               fc=(color[0], color[1], color[2], 1.0), ec="0.3", lw=0.6))
        ov.text(kx + swx + 0.006, yy, label, ha="left", va="center",
                fontsize=19, color="0.15", linespacing=1.2)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight", pad_inches=0.12)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print(f"letterbox: raw {W}x{H} -> content {new_w}x{new_h} at (left={left}, top={top})")


if __name__ == "__main__":
    main()
