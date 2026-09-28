"""rank_empty_density.py — auto-surface mislabeled "empty" tubes.

For every row whose ground truth is empty (length == 0 AND area == 0), run the
trained model and record the DensityReadout's summed per-patch density (in raw
mm / mm²). The prediction's density branch is a spatially-grounded "root mass"
estimate, so a high summed density on a gt-empty image means the model sees a
real root the annotation marked as 0 — i.e. a likely LABEL ERROR. Ranking the
empties by this signal yields a re-annotation queue (bright = mislabeled,
dark = genuinely empty).

Outputs a CSV sorted by ``density_length_mm`` descending:
    rank, image, density_length_mm, density_area_mm2, pred_length_mm, pred_area_mm2

Needs a checkpoint with ``readout="both"`` (all released 768/896 px models) and
a labelled CSV (default: $ROOTQUANT_DATA_DIR/test_data.csv).

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.rank_empty_density \
        --out_csv RootQuantV2/viz_analysis/empty_density_ranking.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import RootDataset, get_transforms, load_split
from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.visualize_attention import load_model


@torch.no_grad()
def _signals(model, img, pmask, cfg, stats) -> Tuple[np.ndarray, np.ndarray]:
    """Return (density_only_mm, full_pred_mm), each (B, 2) in raw units.

    density_only_mm = gain * Σ_patch softplus(mlp(patch))   (the density branch
    alone, == invert of its standardized output). full_pred_mm = the model's
    actual blended readout, inverted to mm."""
    bb = model.backbone.model
    ff = bb.forward_features(img)
    if isinstance(ff, list):
        ff = ff[0]
    cls = ff["x_norm_clstoken"]
    patches = ff["x_norm_patchtokens"]
    pm = pmask if pmask is not None else None

    dens = F.softplus(model.density_head.mlp(patches)).float()       # (B,N,2)
    if pm is not None:
        dens = dens * pm.float().unsqueeze(-1)
    dens_mm = (dens.sum(dim=1) * model.density_head.gain).clamp_min(0.0)   # (B,2)

    feat = model.pooler(cls, patches, patch_mask=pm)
    g = model.head(feat)
    d_std = model.density_head(patches, patch_mask=pm)
    a = torch.sigmoid(model.readout_blend)
    out = a * g + (1.0 - a) * d_std
    if str(cfg.get("target_transform", "zscore")) != "none" and stats:
        out = _invert_targets(out, stats)
    full_mm = out.clamp_min(0.0)
    return dens_mm.float().cpu().numpy(), full_mm.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt")
    ap.add_argument("--csv", default=_CFG["test_csv"])
    ap.add_argument("--data_path", default=_CFG["data_path"])
    ap.add_argument("--out_csv", default="RootQuantV2/viz_analysis/empty_density_ranking.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # No attention hook needed (density only) → keep the fast SDPA path.
    model, cfg, stats, ckpt = load_model(args.checkpoint, device=args.device,
                                         install_capture=False)
    if cfg.get("readout") != "both":
        raise SystemExit(f"Checkpoint readout={cfg.get('readout')!r}; this ranking needs "
                         f"readout='both' (density head + global head + blend).")

    paths, labels = load_split(args.csv, args.csv, args.csv, args.data_path)["train"]
    empty = (labels[:, 0] == 0) & (labels[:, 1] == 0)
    idx = torch.nonzero(empty, as_tuple=False).flatten().tolist()
    paths_e = [paths[i] for i in idx]
    labels_e = labels[idx]
    print(f"checkpoint : {ckpt}")
    print(f"empties    : {len(paths_e)} / {len(paths)} rows (gt length==0 & area==0)")

    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    ds = RootDataset(paths_e, labels_e, transform=tf, target_norm_stats=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    device = torch.device(args.device)
    rows = []
    cursor = 0
    for batch in tqdm(loader, desc="scoring empties", unit="batch", dynamic_ncols=True):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            img, _, pmask = batch
            pmask = pmask.to(device)
        else:
            img, _ = batch
            pmask = None
        img = img.to(device)
        dens_mm, full_mm = _signals(model, img, pmask, cfg, stats)
        for b in range(img.shape[0]):
            rows.append((
                paths_e[cursor + b],
                float(dens_mm[b, 0]), float(dens_mm[b, 1]),
                float(full_mm[b, 0]), float(full_mm[b, 1]),
            ))
        cursor += img.shape[0]

    rows.sort(key=lambda r: -r[1])                     # by density_length_mm desc

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "image", "density_length_mm", "density_area_mm2",
                    "pred_length_mm", "pred_area_mm2"])
        for i, (img, dl, da, pl, pa) in enumerate(rows, 1):
            w.writerow([i, img, f"{dl:.4f}", f"{da:.4f}", f"{pl:.4f}", f"{pa:.4f}"])
    print(f"wrote {args.out_csv}  ({len(rows)} rows)")

    print("\nTop 20 likely-mislabeled empties (high summed density, gt=0):")
    print(f"{'rank':>4} {'dens_len':>9} {'dens_area':>10} {'pred_len':>9} {'pred_area':>10}  image")
    for i, (img, dl, da, pl, pa) in enumerate(rows[:20], 1):
        print(f"{i:>4} {dl:9.2f} {da:10.2f} {pl:9.2f} {pa:10.2f}  {os.path.basename(img)}")

    n_hi = sum(1 for r in rows if r[1] >= 5.0)
    print(f"\nempties with density_length_mm >= 5.0 (re-annotation candidates): {n_hi}")


if __name__ == "__main__":
    main()
