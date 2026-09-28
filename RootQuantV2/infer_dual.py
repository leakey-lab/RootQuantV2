"""infer_dual.py — evaluate one RootQuantV2 checkpoint on the MIXED test set,
producing BOTH vanilla (no-TTA) and TTA predictions in a single pass, tagging
every row with species (maize / soy), and writing a per-image CSV.

This assumes the mixed test set ($ROOTQUANT_DATA_DIR/test_data.csv) is the
disjoint union of the two per-species test splits:
    soy   = $ROOTQUANT_SOY_DIR/test_data.csv
    maize = $ROOTQUANT_MAIZE_DIR/test_data.csv
so species is assigned by membership of the image basename in those name sets.

For every image we run:
    * vanilla = model(img)                    (1 forward)
    * tta     = tta_predict_mean(model, img)  (D4 / 8 forwards for square inputs)
both inverted from z-score back to raw units (clamped >= 0), then written side
by side so downstream aggregation can score either mode per species.

Usage (run from the repository root):
    CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.infer_dual \
        --checkpoint RootQuantV2/runs/checkpoints/rootquant-v2-soybean-only/best.pt \
        --model_name rootquant-v2-soybean-only \
        --mixed_csv "$ROOTQUANT_DATA_DIR/test_data.csv" \
        --data_path "$ROOTQUANT_DATA_DIR/images" \
        --soy_csv   "$ROOTQUANT_SOY_DIR/test_data.csv" \
        --maize_csv "$ROOTQUANT_MAIZE_DIR/test_data.csv" \
        --output_csv RootQuantV2/runs/inference_eval/preds/rootquant-v2-soybean-only.csv \
        --batch_size 96
"""

import argparse
import csv
import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import RootDataset, get_transforms, load_split
from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.model import DinoV3RootRegressor, tta_predict_mean
from RootQuantV2.training import load, load_into


def _invert(y: torch.Tensor, cfg: Dict, stats: Dict) -> torch.Tensor:
    if str(cfg.get("target_transform", "zscore")) == "none" or not stats:
        return y
    return _invert_targets(y, stats)


def _metrics(pred: np.ndarray, true: np.ndarray):
    """R2 / RMSE / MAE for one target column over a selected row subset."""
    if len(true) == 0:
        return float("nan"), float("nan"), float("nan")
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    rmse = math.sqrt(float(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return r2, rmse, mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", required=True, help="Short tag for output rows/logs.")
    ap.add_argument("--mixed_csv", required=True, help="Combined (mixed-species) test CSV.")
    ap.add_argument("--data_path", required=True, help="Shared image dir for the mixed set.")
    ap.add_argument("--soy_csv", required=True, help="Per-species CSV defining soy names.")
    ap.add_argument("--maize_csv", required=True, help="Per-species CSV defining maize names.")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    stats = payload.get("target_stats", {})

    # Build dataset from the mixed split (reuse load_split's robust CSV parsing).
    # Column names come from this checkout's config.py.
    image_col = _CFG["image_col"]
    paths, labels = load_split(
        args.mixed_csv, args.mixed_csv, args.mixed_csv, args.data_path,
        image_col=image_col, length_col=_CFG["length_col"], area_col=_CFG["area_col"],
    )["train"]
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    ds = RootDataset(paths, labels, transform=tf, target_norm_stats=None)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    load_into(args.checkpoint, model=model, map_location="cpu", strict=False)

    van_chunks, tta_chunks = [], []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=args.model_name,
                          unit="batch", dynamic_ncols=True, mininterval=2.0):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                img, _, patch_mask = batch
            else:
                img, _ = batch
                patch_mask = None
            img = img.to(device, non_blocking=True)
            if patch_mask is not None:
                patch_mask = patch_mask.to(device, non_blocking=True)

            ov = model(img, patch_mask=patch_mask)
            van_chunks.append(torch.stack([ov["length"], ov["area"]], -1).float().cpu())

            ot = tta_predict_mean(model, img, patch_mask)
            tta_chunks.append(torch.stack([ot["length"], ot["area"]], -1).float().cpu())

    van = _invert(torch.cat(van_chunks), cfg, stats).clamp_min(0.0).numpy()
    tta = _invert(torch.cat(tta_chunks), cfg, stats).clamp_min(0.0).numpy()
    tgt = labels.float().numpy()

    # Species tagging by membership of basename in the per-species name sets.
    soy_names = set(pd.read_csv(args.soy_csv)[image_col].astype(str))
    maize_names = set(pd.read_csv(args.maize_csv)[image_col].astype(str))
    species = []
    for p in paths:
        n = os.path.basename(p)
        species.append("soy" if n in soy_names else ("maize" if n in maize_names else "unknown"))
    sp = np.array(species)
    n_unknown = int((sp == "unknown").sum())
    if n_unknown:
        print(f"[{args.model_name}] WARNING: {n_unknown} rows could not be classified soy/maize")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "image", "species",
            "pred_length_vanilla", "pred_area_vanilla",
            "pred_length_tta", "pred_area_tta",
            "true_length", "true_area",
        ])
        for i, p in enumerate(paths):
            w.writerow([
                args.model_name, os.path.basename(p), species[i],
                f"{van[i, 0]:.4f}", f"{van[i, 1]:.4f}",
                f"{tta[i, 0]:.4f}", f"{tta[i, 1]:.4f}",
                f"{tgt[i, 0]:.4f}", f"{tgt[i, 1]:.4f}",
            ])
    print(f"[{args.model_name}] wrote {args.output_csv} ({len(paths)} rows)")

    # Per-species, per-mode metrics on present rows (length>0 AND area>0).
    print(f"\n=== {args.model_name}: R2/RMSE/MAE on present rows (length>0 & area>0) ===")
    for sname in ("all", "maize", "soy"):
        sel = np.ones(len(paths), bool) if sname == "all" else (sp == sname)
        present = (tgt[:, 0] > 0) & (tgt[:, 1] > 0) & sel
        n = int(present.sum())
        for mode, preds in (("vanilla", van), ("tta", tta)):
            line = f"  {sname:<6} {mode:<8} n={n:>5}"
            for j, tn in enumerate(("length", "area")):
                r2, rmse, mae = _metrics(preds[present, j], tgt[present, j])
                line += f" | {tn}: R2={r2:.4f} RMSE={rmse:.3f} MAE={mae:.3f}"
            print(line)


if __name__ == "__main__":
    main()
