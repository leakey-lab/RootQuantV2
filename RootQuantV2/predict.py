"""predict.py — run a trained RootQuant-V2 checkpoint on images with NO labels.

``infer.py`` scores a labelled split; this is the plain "give me the numbers"
entry point: point it at image files, a directory, or a CSV column of names and
it writes predicted root length (mm) and surface area (mm2) per image.

Usage (from the repository root)
--------------------------------
    # a directory of images
    python -m RootQuantV2.predict --images /path/to/images --out predictions.csv

    # explicit files
    python -m RootQuantV2.predict --images a.jpg b.jpg c.png

    # a CSV that has an image-name column (labels not required)
    python -m RootQuantV2.predict --csv my_images.csv --data_path /path/to/images

The checkpoint carries its own config and target statistics, so nothing else
needs to match how it was trained. D4 test-time augmentation follows the
checkpoint's own ``eval_tta`` setting unless you override it with
``--tta`` / ``--no-tta``.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from RootQuantV2.config import CONFIG as _CFG
from RootQuantV2.data import RootDataset, get_transforms
from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.model import DinoV3RootRegressor, tta_predict_mean
from RootQuantV2.training import load, load_into

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

DEFAULT_CHECKPOINT = os.path.join(
    "RootQuantV2", "runs", "checkpoints", "rootquant-v2-weights", "best.pt"
)


def collect_images(items: List[str]) -> List[str]:
    """Expand a mix of files and directories into a sorted list of image paths."""
    paths: List[str] = []
    for item in items:
        if os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                paths.extend(
                    os.path.join(root, f)
                    for f in files
                    if f.lower().endswith(IMAGE_EXTS)
                )
        elif os.path.isfile(item):
            paths.append(item)
        else:
            raise SystemExit(f"no such file or directory: {item}")
    if not paths:
        raise SystemExit("no images found.")
    return sorted(paths)


def paths_from_csv(csv_path: str, data_path: str, image_col: str) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{csv_path} has no rows.")
    if image_col not in rows[0]:
        raise SystemExit(
            f"column {image_col!r} not in {csv_path}. Columns: {list(rows[0])}"
        )
    return [os.path.join(data_path, str(r[image_col]).strip()) for r in rows]


def _invert(y: torch.Tensor, cfg: Dict, stats: Dict) -> torch.Tensor:
    if str(cfg.get("target_transform", "zscore")) == "none" or not stats:
        return y
    return _invert_targets(y, stats)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Trained RootQuant-V2 checkpoint (default: the headline model).")
    src = ap.add_argument_group("inputs (choose one)")
    src.add_argument("--images", nargs="+", default=None,
                     help="Image files and/or directories to predict on.")
    src.add_argument("--csv", default=None,
                     help="CSV listing image names in --image_col (labels not needed).")
    ap.add_argument("--data_path", default="",
                    help="Directory prepended to names read from --csv.")
    ap.add_argument("--image_col", default=_CFG["image_col"])
    ap.add_argument("--out", default=None, help="Write predictions to this CSV.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tta", dest="tta", default=None, action=argparse.BooleanOptionalAction,
                    help="Force D4 test-time augmentation on/off (default: the "
                         "checkpoint's own eval_tta setting; 8 forwards per image).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if bool(args.images) == bool(args.csv):
        raise SystemExit("give exactly one of --images or --csv.")

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            f"Download it first:  python tools/download_checkpoints.py"
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    payload = load(args.checkpoint, map_location="cpu")
    cfg = payload["cfg"]
    stats = payload.get("target_stats", {})
    if cfg.get("readout") in ("density", "both") and not stats:
        raise SystemExit(
            f"checkpoint readout={cfg.get('readout')!r} but it carries no "
            f"target_stats; the density head cannot be rebuilt."
        )

    paths = (
        collect_images(args.images) if args.images
        else paths_from_csv(args.csv, args.data_path, args.image_col)
    )
    use_tta = bool(cfg.get("eval_tta", False)) if args.tta is None else bool(args.tta)
    print(
        f"model      : {os.path.basename(os.path.dirname(args.checkpoint))} "
        f"(profile {cfg.get('profile')}, {cfg.get('target_size')} px, "
        f"readout {cfg.get('readout')})\n"
        f"images     : {len(paths)}\n"
        f"device     : {args.device}\n"
        f"D4 TTA     : {'on (8 views)' if use_tta else 'off'}"
    )

    transform = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    dataset = RootDataset(paths, torch.zeros(len(paths), 2), transform=transform,
                          target_norm_stats=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    device = torch.device(args.device)
    model = DinoV3RootRegressor(cfg).to(device).eval()
    # strict=False: slim checkpoints omit the frozen backbone, which is rebuilt
    # from the local DINOv3 weights at construction time.
    load_into(args.checkpoint, model=model, map_location="cpu", strict=False)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", unit="batch", dynamic_ncols=True):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                img, _, patch_mask = batch
            else:
                img, _ = batch
                patch_mask = None
            img = img.to(device)
            if patch_mask is not None:
                patch_mask = patch_mask.to(device)
            out = (tta_predict_mean(model, img, patch_mask) if use_tta
                   else model(img, patch_mask=patch_mask))
            preds.append(torch.stack([out["length"], out["area"]], -1).float().cpu())

    # Back to raw mm / mm2; a negative prediction is physically meaningless.
    preds = _invert(torch.cat(preds), cfg, stats).clamp_min(0.0).numpy()

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "pred_length_mm", "pred_area_mm2"])
            for path, pred in zip(paths, preds):
                writer.writerow([os.path.basename(path), f"{pred[0]:.4f}", f"{pred[1]:.4f}"])
        print(f"\nwrote {args.out}")
    else:
        print(f"\n{'image':<60} {'length_mm':>10} {'area_mm2':>10}")
        for path, pred in zip(paths, preds):
            print(f"{os.path.basename(path):<60} {pred[0]:>10.3f} {pred[1]:>10.3f}")


if __name__ == "__main__":
    main()
