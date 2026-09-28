#!/usr/bin/env python3
"""viz_tta_d4.py — visualize D4 test-time augmentation for RootQuantV2.

Renders the 8 dihedral views that model.tta_predict_mean averages at eval, each
with the model's per-view (length_mm, area_mm2), plus the averaged final pred and
the inter-view spread (free UQ proxy). Faithful to model.tta_predict_mean: same
8 D4 group elements applied identically to image + letterbox patch_mask.

Run from the repo root:
    python -m RootQuantV2.viz_tta_d4 --image frame.jpg
"""
import argparse, math, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO.parent))            # make `import RootQuantV2` work

import numpy as np
import torch
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
# (hflip, vflip, k_rot90) — identical ordering to model.tta_predict_mean square branch
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


def denorm(t):                              # [C,H,W] normalized -> HWC [0,1]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
        default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--image", required=True, help="Minirhizotron frame to visualize.")
    ap.add_argument("--out", default="RootQuantV2/viz_analysis/tta_d4_montage.png")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    stats = payload.get("target_stats", {})
    if not stats:
        raise SystemExit("checkpoint has no target_stats; cannot invert to raw units")

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    load_into(args.checkpoint, model=model, map_location="cpu", strict=False)

    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    pil = Image.open(args.image).convert("RGB")
    img_t, patch_mask = tf(pil)                        # [C,H,W], [N]
    n = int(patch_mask.numel()); grid = int(math.isqrt(n))
    if grid * grid != n:
        raise SystemExit(f"patch_mask len {n} not square; TTA needs square grid")
    mask2d = patch_mask.reshape(grid, grid)

    disps, preds = [], []
    with torch.no_grad():
        for (hf, vf, k) in D4_ELEMS:
            xt = apply_d4_img(img_t, hf, vf, k)
            mt = apply_d4_mask(mask2d, hf, vf, k)
            pm = mt.reshape(1, -1).to(device)
            out = model(xt.unsqueeze(0).to(device), patch_mask=pm)
            std_pred = torch.tensor([float(out["length"][0]), float(out["area"][0])])
            raw = invert_targets(std_pred, stats)      # affine -> raw mm, mm^2
            preds.append(raw.numpy())
            disps.append(denorm(xt))
    preds = np.stack(preds)                            # [8,2]
    final = np.clip(preds.mean(0), 0, None)            # pipeline order: clamp(mean)
    spread = preds.std(0)                              # inter-view std (UQ proxy)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), layout="constrained")
    for ax, name, disp, p in zip(axes.ravel(), D4_NAMES, disps, preds):
        ax.imshow(disp); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nL={p[0]:.0f} mm   A={p[1]:.0f} mm$^2$", fontsize=11)
    fig.suptitle(
        f"D4 test-time augmentation — 8 views averaged   |   "
        f"final  length={final[0]:.1f} mm   area={final[1]:.1f} mm$^2$   "
        f"(view spread ±{spread[0]:.1f} / ±{spread[1]:.1f})",
        fontsize=14)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"final length={final[0]:.2f} mm  area={final[1]:.2f} mm^2")
    for name, p in zip(D4_NAMES, preds):
        print(f"  {name:<14} L={p[0]:8.2f}  A={p[1]:8.2f}")


if __name__ == "__main__":
    main()
