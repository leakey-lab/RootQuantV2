"""
Target normalization utilities for RootQuantV2.

Fit z-score stats on the positive-label train split, apply and invert during
training / inference. Stats are JSON-serializable dicts of four float scalars.
"""

import torch
from typing import Dict


# Upper bound on the standardized→raw value in log space before expm1, so a
# single exploded prediction cannot overflow / dominate R². log1p(~22000mm) ≈ 10,
# far above any real root, so this never clips a plausible value.
_LOG_SPACE_CLAMP = 10.0
# Same idea for sqrt space: cap the standardized→raw value before squaring so an
# exploded prediction cannot dominate R². sqrt(~250000mm²) ≈ 500, far above any
# real surface area, so this never clips a plausible value.
_SQRT_SPACE_CLAMP = 500.0

# Per-column forward / inverse transforms applied BEFORE z-scoring. "none" keeps
# raw units (legacy), "log" uses log1p (heavy-tail / relative-error regime), and
# "sqrt" linearizes the ~quadratic area target without log's relative-error bias.
_VALID_TRANSFORMS = ("none", "log", "sqrt")


def _fwd_transform(col: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "log":
        return torch.log1p(col.clamp_min(0.0))
    if kind == "sqrt":
        return torch.sqrt(col.clamp_min(0.0))
    return col  # "none"


def _inv_transform(col: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "log":
        return torch.expm1(col.clamp_max(_LOG_SPACE_CLAMP))
    if kind == "sqrt":
        return col.clamp(min=0.0, max=_SQRT_SPACE_CLAMP) ** 2
    return col  # "none"


def _resolve_transforms(stats: Dict[str, float]) -> tuple:
    """Return (length_transform, area_transform) from a stats dict.

    Backward-compatible: a stats dict written before per-column transforms
    existed carries only the global ``log`` flag, which maps to log/log (its
    old meaning) or none/none. New dicts carry explicit
    ``length_transform`` / ``area_transform`` keys, which take precedence.
    """
    if "length_transform" in stats or "area_transform" in stats:
        return (
            str(stats.get("length_transform", "none")),
            str(stats.get("area_transform", "none")),
        )
    legacy = "log" if stats.get("log") else "none"
    return (legacy, legacy)


def fit_zscore(
    labels: torch.Tensor,
    log: bool = False,
    transforms: tuple = None,
) -> Dict[str, float]:
    """Fit per-task z-score stats over rows where both length > 0 and area > 0.

    Parameters
    ----------
    labels : Tensor[N, 2]
        Columns are (length_mm, area_mm2). May include zero rows (no-root images).
    log : bool, default False
        Legacy global flag: if True (and ``transforms`` is None), fit BOTH columns
        in ``log1p`` space. Kept for backward compatibility with existing configs
        and checkpoints.
    transforms : tuple(str, str) or None
        Per-column transform applied before z-scoring, one of
        ``("none", "log", "sqrt")`` each, e.g. ``("none", "sqrt")`` to leave
        length raw but linearize the ~quadratic area target. Takes precedence
        over ``log`` when given.

    Returns
    -------
    dict with keys: length_mean, length_std, area_mean, area_std, log,
    length_transform, area_transform.

    Raises
    ------
    ValueError if no rows pass the positivity filter or a transform is unknown.
    """
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"Expected labels shape (N, 2), got {tuple(labels.shape)}")

    if transforms is None:
        legacy = "log" if log else "none"
        len_tf, area_tf = legacy, legacy
    else:
        len_tf, area_tf = str(transforms[0]), str(transforms[1])
    for tf in (len_tf, area_tf):
        if tf not in _VALID_TRANSFORMS:
            raise ValueError(f"Unknown transform '{tf}'. Options: {_VALID_TRANSFORMS}")

    mask = (labels[:, 0] > 0) & (labels[:, 1] > 0)
    if mask.sum() == 0:
        raise ValueError(
            "fit_zscore: no rows with (length > 0 AND area > 0). "
            "Cannot compute normalization stats."
        )

    pos = labels[mask].clone().float()
    col0 = _fwd_transform(pos[:, 0], len_tf)
    col1 = _fwd_transform(pos[:, 1], area_tf)

    length_mean = col0.mean().item()
    length_std = max(col0.std(unbiased=True).item(), 1e-6)
    area_mean = col1.mean().item()
    area_std = max(col1.std(unbiased=True).item(), 1e-6)

    return {
        "length_mean": length_mean,
        "length_std": length_std,
        "area_mean": area_mean,
        "area_std": area_std,
        # Legacy flag: True only when BOTH columns are log (old semantics).
        "log": bool(len_tf == "log" and area_tf == "log"),
        "length_transform": len_tf,
        "area_transform": area_tf,
    }


def apply(labels: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    """Standardize labels per task: (t(label) - mean) / std, with t=log1p if fit
    in log space.

    Applied to ALL rows. Empty rows (zero targets) map to fixed negative values
    (``-mean/std``); they stay in the regression loss, weighted by ``w_empty``,
    and that value is the target the model learns for an empty frame. Returns a
    new tensor; input is never mutated.

    Parameters
    ----------
    labels : Tensor[N, 2]
    stats : dict from fit_zscore

    Returns
    -------
    Tensor[N, 2] normalized.
    """
    out = labels.clone().float()
    len_tf, area_tf = _resolve_transforms(stats)
    out[:, 0] = _fwd_transform(out[:, 0], len_tf)
    out[:, 1] = _fwd_transform(out[:, 1], area_tf)
    out[:, 0] = (out[:, 0] - stats["length_mean"]) / stats["length_std"]
    out[:, 1] = (out[:, 1] - stats["area_mean"]) / stats["area_std"]
    return out


def invert(preds: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    """Invert standardization: pred * std + mean per task, then expm1 if fit in
    log space.

    Returns a new tensor; input is never mutated.

    Parameters
    ----------
    preds : Tensor[N, 2]  (or Tensor[2] for a single sample)
    stats : dict from fit_zscore

    Returns
    -------
    Tensor same shape as preds, in original units.
    """
    out = preds.clone().float()
    len_tf, area_tf = _resolve_transforms(stats)
    if out.ndim == 1:
        out[0] = out[0] * stats["length_std"] + stats["length_mean"]
        out[1] = out[1] * stats["area_std"] + stats["area_mean"]
        out[0] = _inv_transform(out[0], len_tf)
        out[1] = _inv_transform(out[1], area_tf)
    else:
        out[:, 0] = out[:, 0] * stats["length_std"] + stats["length_mean"]
        out[:, 1] = out[:, 1] * stats["area_std"] + stats["area_mean"]
        out[:, 0] = _inv_transform(out[:, 0], len_tf)
        out[:, 1] = _inv_transform(out[:, 1], area_tf)
    return out
