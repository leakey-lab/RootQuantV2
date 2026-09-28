"""
Dataset and split-loading utilities for RootQuantV2.

Re-implements CSV loading from the legacy RootQuant data layer without
importing from it. Column names are passed explicitly so the caller controls
the schema.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image

from .target_norm import apply as apply_norm

logger = logging.getLogger(__name__)


class RootDataset(torch.utils.data.Dataset):
    """Dataset for root-phenotyping images and (length_mm, area_mm2) targets.

    Parameters
    ----------
    image_paths : List[str]
        Absolute paths to image files (or data_path-relative; caller handles
        joining before passing in).
    labels : Tensor[N, 2]
        Columns (length_mm, area_mm2). Should be float32.
    transform : optional callable (PIL.Image) -> Tensor[C,H,W]
        Applied to each image on load. Must include ToTensor.
    target_norm_stats : optional dict
        If given, z-score is applied in __getitem__ via
        ``data.target_norm.apply``. Keys: length_mean, length_std,
        area_mean, area_std.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: torch.Tensor,
        transform: Optional[Callable] = None,
        target_norm_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) != "
                f"labels length ({len(labels)})"
            )
        self.image_paths = image_paths
        self.labels = labels.float()
        self.transform = transform
        self.target_norm_stats = target_norm_stats
        # Precompute z-scored labels once (apply is vectorized over all rows), so
        # __getitem__ is an O(1) index instead of a per-call normalize + import.
        # None when no stats (e.g. infer path) → __getitem__ returns raw labels.
        self._labels_norm = (
            apply_norm(self.labels, target_norm_stats)
            if target_norm_stats is not None else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open image '{path}' (index {idx}): {exc}"
            ) from exc

        patch_mask: Optional[torch.Tensor] = None
        if self.transform is not None:
            out = self.transform(img)
            if isinstance(out, tuple) and len(out) == 2:
                img, patch_mask = out
            else:
                img = out
        else:
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(img)

        if self._labels_norm is not None:
            label = self._labels_norm[idx].clone()
        else:
            label = self.labels[idx].clone()

        if patch_mask is not None:
            return img, label, patch_mask
        return img, label


def load_split(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    data_path: str,
    image_col: str = "ImageName",
    length_col: str = "AliveLength(mm)",
    area_col: str = "AliveSurfArea(mm2)",
    check_images_exist: bool = False,
) -> Dict[str, Tuple[List[str], torch.Tensor]]:
    """Load train / val / test CSV splits.

    Parameters
    ----------
    train_csv, val_csv, test_csv : str
        Absolute paths to the three CSV files.
    data_path : str
        Root directory that holds the image files. Each image name from
        ``image_col`` is joined with ``data_path`` via ``os.path.join``.
    image_col, length_col, area_col : str
        Column names in the CSV.

    Returns
    -------
    dict mapping split name to (image_paths: List[str], labels: Tensor[N, 2]).
    Labels are float32. Missing images are warned about but included with their
    path intact (so downstream errors surface clearly).

    NaN rows (in any of the three columns) are skipped with a warning.
    """
    splits = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
    }
    result: Dict[str, Tuple[List[str], torch.Tensor]] = {}

    for split_name, csv_path in splits.items():
        if not csv_path or not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"[{split_name}] CSV not found: {csv_path or '<unset>'}. Set "
                f"ROOTQUANT_DATA_DIR or pass the CSV path explicitly "
                f"(see 'Data layout' in the README)."
            )
        df = pd.read_csv(csv_path)

        # Validate required columns.
        for col in (image_col, length_col, area_col):
            if col not in df.columns:
                raise ValueError(
                    f"[{split_name}] CSV '{csv_path}' missing column '{col}'. "
                    f"Available: {list(df.columns)}"
                )

        # Drop NaN rows.
        nan_mask = df[[image_col, length_col, area_col]].isnull().any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            logger.warning(
                "[%s] Skipping %d row(s) with NaN in columns [%s, %s, %s].",
                split_name,
                n_nan,
                image_col,
                length_col,
                area_col,
            )
            df = df[~nan_mask].reset_index(drop=True)

        # Vectorized path build — df.iterrows() over ~90k rows is slow; pandas
        # ops + a list comprehension are far faster. The per-row os.path.isfile()
        # NFS-stat loop is skipped by default (check_images_exist=False): a missing
        # file raises a clear error at Image.open in __getitem__ anyway.
        names = df[image_col].astype(str).tolist()
        image_paths = [os.path.join(data_path, n) for n in names]
        if check_images_exist:
            for p in image_paths:
                if not os.path.isfile(p):
                    logger.warning("[%s] Image not found on disk: %s", split_name, p)

        labels = torch.tensor(
            df[[length_col, area_col]].to_numpy(dtype="float32"), dtype=torch.float32
        )
        result[split_name] = (image_paths, labels)
        logger.info(
            "[%s] Loaded %d rows (NaN skipped: %d).", split_name, len(image_paths), n_nan
        )

    return result
