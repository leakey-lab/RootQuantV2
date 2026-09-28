#!/usr/bin/env python3
"""viz_root_panel.py — tight "does the model look at roots?" figure for the paper.

One grid: each row an image, columns
    [ input | backbone PCA features | HEAD density.length | HEAD density.area ].

- The HEAD density columns are TASK-FAITHFUL: per-patch non-negative density whose
  masked SUM is (close to) the prediction (model.py DensityReadout). softplus(mlp(
  patches))[...,ti] * gain[ti], masked — identical to viz_tta_d4_density.py.
- The feature PCA column is a TASK-AGNOSTIC readout of RootQuant-V2's ADAPTED
  representation: the patch tokens from forward_features (frozen backbone weights +
  trained DoRA on attention + Mona on the MLP branch) projected to their top-3
  principal components. PCA is unsupervised (no task labels), so it organises ANY
  structured texture, incl. bare soil — that is the foil, not the signal. NOTE these
  are NOT pretrained DINOv3 tokens: DoRA/Mona are in the forward path.

The contrast is the proof: on a root-free frame the adapted representation still
segments soil texture while the head density stays dark — only the trained readout
measures roots, not substrate.

Density columns use ONE shared absolute color scale (default fit on present rows),
so a near-empty map is NOT stretched to look bright (per docs/visualization.md).

Ground truth for the "true" chips is read from --csv (default: the test split,
$ROOTQUANT_DATA_DIR/test_data.csv); without it the chips show 0.

Run from the repo root:
    python -m RootQuantV2.viz_root_panel --images /path/a.jpg /path/b.jpg \
        --csv "$ROOTQUANT_DATA_DIR/test_data.csv" --out RootQuantV2/viz_analysis/root_panel.png
"""
import argparse, csv as _csv, math, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO.parent))            # make `import RootQuantV2` work

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import get_transforms
from RootQuantV2.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from RootQuantV2.data.target_norm import invert as invert_targets
from RootQuantV2.model import DinoV3RootRegressor
from RootQuantV2.backbone import DINOv3Backbone
from RootQuantV2.training import load, load_into


def denorm(t):                                   # [C,H,W] normalized -> HWC [0,1]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _norm01(a):
    a = np.asarray(a, np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def pca_feats(feats, valid, grid):
    """Top-3 PCA of valid patch tokens (unsupervised, so task-agnostic; applied to
    the adapted RootQuant-V2 tokens, or to un-adapted DINOv3 tokens for
    --compare_frozen). Returns
      rgb   (grid,grid,3) per-component RGB (legacy rainbow view),
      pc1   (grid,grid)   normalized 1st component, oriented foreground-bright [0,1],
      pca3  (grid,grid)   variance-weighted L2 saliency over the top-3 components
                          [0,1] -- a single colorblind-safe scalar that still uses
                          PC1+PC2+PC3 energy, not just PC1.
    """
    n = feats.shape[0]
    fv = feats[valid]
    rgb = np.zeros((n, 3), np.float32)
    pc1 = np.full(n, np.nan, np.float32)
    pca3 = np.full(n, np.nan, np.float32)
    if fv.shape[0] >= 3:
        mu = fv.mean(0, keepdims=True)
        X = fv - mu
        _, S, Vt = np.linalg.svd(X, full_matrices=False)   # economy SVD
        proj = X @ Vt[:3].T                                 # (m,3)
        for c in range(3):
            rgb[valid, c] = _norm01(proj[:, c])
        # PC1, oriented so the border ring (usually background/tube) is dark
        gv = valid.reshape(grid, grid)
        border = np.zeros((grid, grid), bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        full = np.full(n, np.nan, np.float32); full[valid] = proj[:, 0]
        f2 = full.reshape(grid, grid)
        bvals = f2[border & gv]; ivals = f2[~border & gv]
        p = proj[:, 0].copy()
        if bvals.size and ivals.size and np.nanmean(bvals) > np.nanmean(ivals):
            p = -p
        pc1[valid] = _norm01(p)
        # variance-weighted saliency across all 3 components -> one cividis scale
        vr = (S[:3] ** 2) / float((S ** 2).sum())
        pca3[valid] = _norm01(np.sqrt((proj ** 2 * vr[None, :]).sum(1)))
    return rgb.reshape(grid, grid, 3), pc1.reshape(grid, grid), pca3.reshape(grid, grid)


@torch.no_grad()
def analyze(model, dh, blend, stats, tf, path, device, frozen_model=None):
    pil = Image.open(path).convert("RGB")
    img_t, patch_mask = tf(pil)                            # [C,H,W], [N]
    n = int(patch_mask.numel()); grid = int(math.isqrt(n))
    if grid * grid != n:
        raise SystemExit(f"{path}: patch grid {n} not square")
    pm = patch_mask.reshape(1, -1).to(device)
    xb = img_t.unsqueeze(0).to(device)

    ff = model.backbone.model.forward_features(xb)
    if isinstance(ff, list): ff = ff[0]
    cls = ff["x_norm_clstoken"]
    patches = model._get_patches(ff)                      # (1,N,D)

    # --- HEAD: per-patch density (raw units), masked -> the task-faithful maps ---
    dens = F.softplus(dh.mlp(patches)).float()            # (1,N,2)
    dens = dens * pm.float().unsqueeze(-1)
    gain = dh.gain.detach().float().cpu()
    dL = (dens[0, :, 0].cpu() * gain[0]).reshape(grid, grid).numpy()
    dA = (dens[0, :, 1].cpu() * gain[1]).reshape(grid, grid).numpy()

    # --- BACKBONE: task-agnostic PCA segmentation view ---
    valid = (pm[0] > 0).cpu().numpy()
    feats = patches[0].float().cpu().numpy()
    rgb, pc1, pca3 = pca_feats(feats, valid, grid)

    # --- FROZEN off-the-shelf DINOv3: same PCA on un-adapted tokens (no DoRA/Mona) ---
    pca3_frozen = None
    if frozen_model is not None:
        ff0 = frozen_model.forward_features(xb)
        if isinstance(ff0, list):
            ff0 = ff0[0]
        p0 = ff0.get("x_norm_patchtokens")
        if p0 is None:
            p0 = ff0["x_prenorm"][:, 1 + model.R:, :]
        _, _, pca3_frozen = pca_feats(p0[0].float().cpu().numpy(), valid, grid)

    # --- blended prediction (raw mm / mm^2), mirrors model.forward ---
    out_d = dh(patches, patch_mask=pm)
    if model.readout == "both":
        feat = model.pooler(cls, patches, patch_mask=pm)
        out = blend * model.head(feat) + (1 - blend) * out_d
    elif model.readout == "density":
        out = out_d
    else:
        out = model.head(model.pooler(cls, patches, patch_mask=pm))
    pred = invert_targets(out[0].cpu(), stats).clamp_min(0).numpy()

    p = img_t.shape[1] // grid
    return dict(disp=denorm(img_t), rgb=rgb, pc1=pc1, pca3=pca3, pca3_frozen=pca3_frozen,
                dL=dL, dA=dA, valid2d=valid.reshape(grid, grid), grid=grid, p=p,
                pred=pred, sumL=float(dL.sum()), sumA=float(dA.sum()))


def species_of(name):
    u = name.upper()
    if "SOY" in u: return "Soybean"
    if "MAIZE" in u or "MR" in u or "MUTANT" in u: return "Maize"
    return "Root"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
        default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--csv", default=_CFG["test_csv"],
                    help="Labelled CSV for the ground-truth chips "
                         "(default: $ROOTQUANT_DATA_DIR/test_data.csv).")
    ap.add_argument("--out", default="RootQuantV2/viz_analysis/root_panel.png")
    ap.add_argument("--backbone", choices=["pca3", "pca1", "pca_rgb", "both"],
                    default="pca3")
    ap.add_argument("--compare_frozen", action="store_true",
                    help="add an off-the-shelf (frozen) DINOv3 PCA column left of RootQuant-V2")
    ap.add_argument("--scale", choices=["present", "all"], default="present",
                    help="rows whose density sets the shared vmax")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # gt lookup (columns by name, as configured in config.py)
    gt = {}
    if args.csv and os.path.isfile(args.csv):
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                gt[row[_CFG["image_col"]].strip()] = (
                    float(row[_CFG["length_col"]]), float(row[_CFG["area_col"]]))
    else:
        print(f"WARNING: ground-truth CSV not found ({args.csv or '<unset>'}); "
              f"set ROOTQUANT_DATA_DIR or pass --csv. 'true' chips will show 0.")

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    cfg["use_torch_compile"] = False
    stats = payload.get("target_stats", {})
    if cfg.get("readout") not in ("density", "both"):
        raise SystemExit(f"readout={cfg.get('readout')!r}: no density head")

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    load_into(args.checkpoint, model=model, map_location="cpu", strict=False)
    dh = model.density_head
    blend = torch.sigmoid(model.readout_blend) if model.readout == "both" else None
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)

    # pristine off-the-shelf DINOv3: identical construction MINUS DoRA/Mona
    frozen_model = None
    if args.compare_frozen:
        fb = DINOv3Backbone(
            weights_path=cfg.get("backbone_weights_path"),
            hub_repo=cfg["backbone_hub_repo"], hub_model="dinov3_vitl16",
            num_register_tokens=int(cfg["backbone_num_register_tokens"]),
            embed_dim=int(cfg["backbone_dim"]),
            patch_size=int(cfg["backbone_patch_size"]),
            pretrained=cfg.get("pretrained", True),
        ).to(device).eval()
        for h in fb._hooks:
            h.remove()
        fb._hooks.clear()
        frozen_model = fb.model

    rows = []
    for ip in args.images:
        name = os.path.basename(ip)
        r = analyze(model, dh, blend, stats, tf, ip, device, frozen_model=frozen_model)
        r["name"] = name
        r["gt"] = gt.get(name)
        r["species"] = species_of(name)
        rows.append(r)
        g = r["gt"]
        print(f"{name}\n  pred L={r['pred'][0]:7.1f}  A={r['pred'][1]:7.1f} | "
              f"gt {('L=%.1f A=%.1f' % g) if g else 'n/a':>18} | "
              f"Sdens L={r['sumL']:7.1f} A={r['sumA']:7.1f}")

    # shared absolute density scale (so empties are not stretched bright)
    pick = [r for r in rows if (args.scale == "all" or (r["gt"] and r["gt"][0] > 0))] or rows
    vmaxL = float(np.percentile(np.concatenate([r["dL"].ravel() for r in pick]), 99.5)) or 1.0
    vmaxA = float(np.percentile(np.concatenate([r["dA"].ravel() for r in pick]), 99.5)) or 1.0

    # Perceptually-uniform, colorblind-safe colormaps on standard [0,1] scales:
    # cividis (CVD-optimized) for the backbone feature saliency, magma for density.
    bb_cmap = plt.get_cmap("cividis").with_extremes(bad="black")
    dn_cmap = plt.get_cmap("magma")

    # --- crop every panel to the common letterbox-free region (drop padding) ---
    grid = rows[0]["grid"]; p = rows[0]["p"]
    R0, C0, R1, C1 = 0, 0, grid, grid
    for r in rows:
        vr = np.where(r["valid2d"].any(1))[0]; vc = np.where(r["valid2d"].any(0))[0]
        R0 = max(R0, int(vr[0])); R1 = min(R1, int(vr[-1]) + 1)
        C0 = max(C0, int(vc[0])); C1 = min(C1, int(vc[-1]) + 1)
    cropm = lambda m: m[R0:R1, C0:C1]
    cropi = lambda im: im[R0 * p:R1 * p, C0 * p:C1 * p]
    cropH, cropW = (R1 - R0) * p, (C1 - C0) * p
    ext = (0, cropW, cropH, 0)

    if args.compare_frozen:
        bb_cols = ["frozen", "pca3"]
    elif args.backbone == "both":
        bb_cols = ["pca_rgb", "pca3"]
    else:
        bb_cols = [args.backbone]
    bb_head = {"pca3": "RootQuant-V2\nfeatures", "pca1": "RootQuant-V2\nPC-1",
               "pca_rgb": "RootQuant-V2\nPCA-RGB", "frozen": "DINOv3\n(pretrained)"}
    bb_key = {"pca3": "pca3", "pca1": "pc1", "frozen": "pca3_frozen"}
    headers = (["Input"] + [bb_head[b] for b in bb_cols]
               + ["Length\ndensity", "Area\ndensity"])
    ncol = len(headers); nrow = len(rows)

    # ------------------------------------------------------------------ layout
    # Deterministic inch-based layout: even column gutters, a Root-bearing /
    # Root-free group split with side brackets, uniform pred/true chips, dimmed
    # base under the density overlays, and one moderate colorbar per quantity.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    H_HEAD, H_ROW, H_GROUP, H_CHIP, H_CBL, H_CBT = 20, 17, 18, 13, 16, 15

    # present rows first, root-free (empty) rows last (input order already does this)
    is_empty = [not (r["gt"] and r["gt"][0] > 0) for r in rows]
    n_present = sum(1 for e in is_empty if not e)
    has_split = 0 < n_present < nrow

    cw = 2.6                              # image cell width (in)
    ch = cw * cropH / cropW              # image cell height (in), preserves aspect
    gx = gy = 0.05                        # inter-column / inter-row gap (in)
    ggap = 0.34                           # extra gap at the present/root-free split
    left = 1.15                           # bracket + rotated species label
    right = 0.12
    top = 0.62                            # column headers
    bot = 1.55                            # two colorbars + labels (room below the bars)

    grid_w = ncol * cw + (ncol - 1) * gx
    grid_h = nrow * ch + (nrow - 1) * gy + (ggap - gy if has_split else 0.0)
    figw = left + grid_w + right
    figh = top + grid_h + bot
    fig = plt.figure(figsize=(figw, figh))

    def add_ax(x_in, y_in, w_in, h_in):     # rect from bottom-left inches
        return fig.add_axes([x_in / figw, y_in / figh, w_in / figw, h_in / figh])

    def row_top_off(i):                     # offset (in) from grid top to row i's top
        off = i * (ch + gy)
        if has_split and i >= n_present:
            off += ggap - gy
        return off

    col_x = lambda j: left + j * (cw + gx)
    row_y = lambda i: figh - top - row_top_off(i) - ch      # bottom-left y (in)

    bb_im = dn_im = None
    dim = 0.55
    for i, r in enumerate(rows):
        disp = cropi(r["disp"]); y = row_y(i)
        # col 0: input + uniform monospaced chip
        ax = add_ax(col_x(0), y, cw, ch); ax.imshow(disp, extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        pp = r["pred"]; tl, ta = (r["gt"] if r["gt"] else (0.0, 0.0))
        chip = (f"pred  L{pp[0]:>4.0f}  A{pp[1]:>4.0f}\n"
                f"true  L{tl:>4.0f}  A{ta:>4.0f}")
        ax.text(0.035, 0.96, chip, transform=ax.transAxes, va="top", ha="left",
                color="white", fontsize=H_CHIP, family="monospace", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="black", alpha=0.62,
                          edgecolor="white", linewidth=0.7))
        ax.set_ylabel(r["species"], fontsize=H_ROW, labelpad=5)
        if i == 0:
            ax.set_title(headers[0], fontsize=H_HEAD, pad=8)
        # feature column(s)
        c = 1
        for b in bb_cols:
            ax = add_ax(col_x(c), y, cw, ch)
            if b == "pca_rgb":
                ax.imshow(cropm(r["rgb"]), extent=ext, interpolation="nearest")
            else:
                m = np.ma.masked_invalid(cropm(r[bb_key[b]]))
                bb_im = ax.imshow(m, cmap=bb_cmap, vmin=0, vmax=1, extent=ext,
                                  interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[c], fontsize=H_HEAD, pad=8, linespacing=0.95)
            c += 1
        # density columns: dimmed base + magma overlay for contrast
        for m, vmax in ((r["dL"], vmaxL), (r["dA"], vmaxA)):
            ax = add_ax(col_x(c), y, cw, ch); ax.imshow(disp * dim, extent=ext)
            mn = np.clip(cropm(m) / vmax, 0, 1)
            dn_im = ax.imshow(mn, cmap=dn_cmap, vmin=0, vmax=1,
                              alpha=mn ** 0.6 * 0.95, extent=ext,
                              interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[c], fontsize=H_HEAD, pad=8, linespacing=0.95)
            c += 1

    # ---- group brackets + divider (Root-bearing / Root-free) ----
    def bracket(y0_in, y1_in, label):
        axb = add_ax(0.12, y1_in, 0.62, y0_in - y1_in)     # spans the group vertically
        axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
        xs = 0.9
        axb.plot([xs, xs], [0.02, 0.98], color="0.25", lw=2.0, clip_on=False)
        axb.plot([xs - 0.16, xs], [0.98, 0.98], color="0.25", lw=2.0, clip_on=False)
        axb.plot([xs - 0.16, xs], [0.02, 0.02], color="0.25", lw=2.0, clip_on=False)
        axb.text(0.34, 0.5, label, rotation=90, va="center", ha="center",
                 fontsize=H_GROUP, fontweight="bold", color="0.15")

    if has_split:
        top_present = row_y(0) + ch
        bot_present = row_y(n_present - 1)
        bracket(top_present, bot_present, "Root-bearing")
        top_empty = row_y(n_present) + ch
        bot_empty = row_y(nrow - 1)
        bracket(top_empty, bot_empty, "Root-free")
        # thin divider line across the image columns at the split
        yd = 0.5 * (bot_present + top_empty)
        axd = add_ax(col_x(0), yd - 0.006, grid_w, 0.012); axd.axis("off")
        axd.set_xlim(0, 1); axd.set_ylim(0, 1)
        axd.axhline(0.5, color="0.55", lw=1.1)

    # ---- two moderate colorbars along the bottom, narrowed & centered under their
    #      column groups with a clear gap so the boundary ticks never collide ----
    cb_h = 0.16
    cb_y = bot - 0.52
    if bb_im is not None:
        if args.compare_frozen:                       # shared bar under both feature cols
            span = 2 * cw + gx
            wF = 0.85 * span
            xF = col_x(1) + (span - wF) / 2
            labF = "feature saliency (norm.)"
        else:
            wF = 0.85 * cw
            xF = col_x(1) + (cw - wF) / 2
            labF = "RootQuant-V2 feature saliency (norm.)"
        caxF = add_ax(xF, cb_y, wF, cb_h)
        cbF = fig.colorbar(bb_im, cax=caxF, orientation="horizontal", ticks=[0, 0.5, 1])
        cbF.set_label(labF, fontsize=H_CBL, labelpad=6)
        cbF.ax.tick_params(labelsize=H_CBT)
    if dn_im is not None:
        span = 2 * cw + gx
        wD = 0.85 * span
        caxD = add_ax(col_x(ncol - 2) + (span - wD) / 2, cb_y, wD, cb_h)
        cbD = fig.colorbar(dn_im, cax=caxD, orientation="horizontal", ticks=[0, 0.5, 1])
        cbD.set_label("root density (norm.)", fontsize=H_CBL, labelpad=6)
        cbD.ax.tick_params(labelsize=H_CBT)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(str(out.with_suffix(".pdf")))
    print(f"\nwrote {out}  and  {out.with_suffix('.pdf')}")
    print(f"crop rows {R0}:{R1} cols {C0}:{C1}  panel {cropW}x{cropH}px  "
          f"fig {figw:.1f}x{figh:.1f}in  vmax L={vmaxL:.3f} A={vmaxA:.3f}")


if __name__ == "__main__":
    main()
