"""infer.py — evaluate a trained RootQuantV2 checkpoint on a CSV split.

Usage (from the repository root):
    python -m RootQuantV2.infer \
        --checkpoint RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt \
        --csv  "$ROOTQUANT_DATA_DIR/test_data.csv" \
        --data_path "$ROOTQUANT_DATA_DIR/images" \
        --output_csv predictions.csv

Loads the checkpoint's own cfg + target_stats, rebuilds the model, runs the
split, inverts z-score, and reports per-task R2 / RMSE / MAE on present rows.
"""

import argparse
import csv
import math
from typing import Dict

import numpy as np
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


def _r2(pred, true):
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True, help="Split CSV to evaluate.")
    ap.add_argument("--data_path", default=None,
                    help="Image dir (default: $ROOTQUANT_IMAGES, else $ROOTQUANT_DATA_DIR/images).")
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Load checkpoint with strict=False (prints missing/unexpected keys).",
    )
    args = ap.parse_args()

    # Eval was running without tensor cores; enable TF32 + the cuDNN autotuner.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    stats = payload.get("target_stats", {})
    if cfg.get("readout") in ("density", "both") and not cfg.get("target_stats"):
        raise ValueError(
            f"Checkpoint readout={cfg.get('readout')!r} but cfg.target_stats is empty; "
            f"the density head cannot be rebuilt from a slim checkpoint without it."
        )
    # The checkpoint's own cfg["data_path"] points at the training machine; use
    # this machine's config (environment) instead.
    data_path = args.data_path or _CFG["data_path"]
    if not data_path:
        raise SystemExit("no image directory: pass --data_path, or set "
                         "ROOTQUANT_IMAGES / ROOTQUANT_DATA_DIR.")

    # Reuse load_split for robust CSV parsing (point all three at the eval CSV).
    # Column names come from this checkout's config.py.
    paths, labels = load_split(
        args.csv, args.csv, args.csv, data_path,
        image_col=_CFG["image_col"], length_col=_CFG["length_col"], area_col=_CFG["area_col"],
    )["train"]
    tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    ds = RootDataset(paths, labels, transform=tf, target_norm_stats=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    # strict=False: slim checkpoints omit the frozen backbone (rebuilt at
    # construction). Old full checkpoints also load cleanly (0 missing/unexpected).
    load_into(
        args.checkpoint,
        model=model,
        map_location="cpu",
        strict=False,
    )

    preds = []
    n_batches = len(loader)
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="infer",
                          unit="batch", dynamic_ncols=True, mininterval=2.0):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                img, _, patch_mask = batch
            else:
                img, _ = batch
                patch_mask = None
            img = img.to(device)
            if patch_mask is not None:
                patch_mask = patch_mask.to(device)
            if bool(cfg.get("eval_tta", False)):
                out = tta_predict_mean(model, img, patch_mask)
            else:
                out = model(img, patch_mask=patch_mask)
            preds.append(torch.stack([out["length"], out["area"]], -1).float().cpu())
    preds = _invert(torch.cat(preds), cfg, stats).clamp_min(0.0)   # raw units
    targets = labels.float()                                       # raw units
    mask = (targets[:, 0] > 0) & (targets[:, 1] > 0)

    p, t = preds[mask].numpy(), targets[mask].numpy()
    print(f"n_total={len(targets)}  n_present={int(mask.sum())}")
    for i, name in enumerate(("length", "area")):
        err = p[:, i] - t[:, i]
        print(f"  {name:<7} R2={_r2(p[:, i], t[:, i]):.4f}  "
              f"RMSE={math.sqrt(float(np.mean(err**2))):.3f}  "
              f"MAE={float(np.mean(np.abs(err))):.3f}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "pred_length", "pred_area", "true_length", "true_area"])
            for path, pr, tr in zip(paths, preds.numpy(), targets.numpy()):
                w.writerow([path, f"{pr[0]:.4f}", f"{pr[1]:.4f}", f"{tr[0]:.4f}", f"{tr[1]:.4f}"])
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
