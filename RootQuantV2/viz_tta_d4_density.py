#!/usr/bin/env python3
"""viz_tta_d4_density.py — per-view "where the model looks" for D4 TTA.

For each of the 8 dihedral views (the ones tta_predict_mean averages), overlay
the TASK-FAITHFUL DensityReadout per-patch *length* density on the view image.
The model's length prediction is (close to) the masked SUM of this map, so it is
the prediction-coupled "root mass" map — what the fine-tuned head actually uses.

Why density, not CLS-attention: the backbone's attention runs on the adapted
tokens (DoRA and Mona are in the forward path) but is task-agnostic — it
segments any texture incl. soil and is NOT task-faithful (docs/visualization.md).
The head density map is.

Extraction mirrors model.DensityReadout.forward:
    dens = softplus(mlp(patches))[..., 0] * gain[0]   # per-patch length density

Run from the repo root:
    python -m RootQuantV2.viz_tta_d4_density --image frame.jpg
"""
import argparse, math, os, sys
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

from RootQuantV2.data import get_transforms
from RootQuantV2.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from RootQuantV2.data.target_norm import invert as invert_targets
from RootQuantV2.model import DinoV3RootRegressor
from RootQuantV2.training import load, load_into

D4_NAMES = ["id", "rot90", "rot180", "rot270",
            "hflip", "hflip+rot90", "hflip+rot180", "hflip+rot270"]
D4_ELEMS = [(h, False, k) for h in (False, True) for k in (0, 1, 2, 3)]


def apply_d4_img(t, hflip, vflip, k):       # t: [C,H,W]
    if hflip: t = torch.flip(t, dims=[2])
    if vflip: t = torch.flip(t, dims=[1])
    if k:     t = torch.rot90(t, k=k, dims=[1, 2])
    return t


def apply_d4_mask(m, hflip, vflip, k):      # m: [gh,gw]
    if hflip: m = torch.flip(m, dims=[1])
    if vflip: m = torch.flip(m, dims=[0])
    if k:     m = torch.rot90(m, k=k, dims=[0, 1])
    return m


def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
        default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--image", required=True, help="Minirhizotron frame to visualize.")
    ap.add_argument("--out", default="RootQuantV2/viz_analysis/tta_d4_density.png")
    ap.add_argument("--target", choices=["length", "area"], default="length")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ti = 0 if args.target == "length" else 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    stats = payload.get("target_stats", {})
    if cfg.get("readout") not in ("density", "both"):
        raise SystemExit(f"checkpoint readout={cfg.get('readout')!r} has no density head")

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    load_into(args.checkpoint, model=model, map_location="cpu", strict=False)
    dh = model.density_head
    blend = torch.sigmoid(model.readout_blend) if model.readout == "both" else None

    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    pil = Image.open(args.image).convert("RGB")
    img_t, patch_mask = tf(pil)
    n = int(patch_mask.numel()); grid = int(math.isqrt(n))
    mask2d = patch_mask.reshape(grid, grid)

    disps, maps, dtot, preds = [], [], [], []
    with torch.no_grad():
        for (hf, vf, k) in D4_ELEMS:
            xt = apply_d4_img(img_t, hf, vf, k)
            mt = apply_d4_mask(mask2d, hf, vf, k)
            pm = mt.reshape(1, -1).to(device)
            xb = xt.unsqueeze(0).to(device)

            ff = model.backbone.model.forward_features(xb)
            if isinstance(ff, list): ff = ff[0]
            patches = model._get_patches(ff)                 # (1,N,D)

            # per-patch density for the chosen target (raw units), masked
            dens = F.softplus(dh.mlp(patches))[0, :, ti] * dh.gain[ti]
            dens = (dens * pm.reshape(-1)).float().cpu()
            dtot.append(float(dens.sum()))
            maps.append(dens.reshape(grid, grid).numpy())

            # blended prediction (== viz_tta_d4.py), for the title
            out_dens = dh(patches, patch_mask=pm)
            if model.readout == "both":
                feat = model.pooler(ff["x_norm_clstoken"], patches, patch_mask=pm)
                out = blend * model.head(feat) + (1 - blend) * out_dens
            else:
                out = out_dens
            preds.append(invert_targets(out[0].cpu(), stats).numpy())
            disps.append(denorm(xt))

    vmax = float(np.percentile(np.stack(maps), 99.5)) or 1.0   # shared absolute scale
    H, W = disps[0].shape[:2]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), layout="constrained")
    for ax, name, disp, m, tot, p in zip(axes.ravel(), D4_NAMES, disps, maps, dtot, preds):
        ax.imshow(disp)
        alpha = np.clip(m / vmax, 0, 1) ** 0.7 * 0.9          # opacity ∝ density
        ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, alpha=alpha,
                  extent=(0, W, H, 0), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n$\\Sigma$ density {args.target[0].upper()}={tot:.0f}   "
                     f"(blended {p[ti]:.0f})", fontsize=10)
    fig.suptitle(
        f"D4 TTA — where the model looks: DensityReadout per-patch {args.target} "
        f"(task-faithful; masked sum $\\approx$ the prediction). Shared color scale.",
        fontsize=13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"wrote {args.out}")
    for name, tot, p in zip(D4_NAMES, dtot, preds):
        print(f"  {name:<14} density_{args.target}={tot:8.2f}  blended={p[ti]:8.2f}")


if __name__ == "__main__":
    main()
