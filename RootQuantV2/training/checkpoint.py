"""Minimal checkpoint save / load for RootQuantV2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def save(state: Dict[str, Any], path: str) -> None:
    """Save a checkpoint dict atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def is_base_backbone_key(name: str) -> bool:
    """True for pretrained DINOv3 weights (rebuilt from the DINOv3 file), False
    for DoRA / Mona adapters, the pooler, the heads and the readout blend."""
    return name.startswith("backbone.") and "dora_" not in name and "mona" not in name


def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}.")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_into(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and populate model / optimizer / scheduler in place."""
    payload = load(path, map_location=map_location)
    if model is not None and payload.get("model_state") is not None:
        missing, unexpected = model.load_state_dict(
            payload["model_state"], strict=strict,
        )
        if not strict:
            # Slim checkpoints omit the frozen base backbone and the buffers that
            # are rebuilt at construction (DoRA V_frozen, RoPE periods, density
            # t_mean/t_std), so those keys are expected to be missing. Any other
            # missing key keeps its initial value.
            buffers = {n for n, _ in model.named_buffers()}
            other = [k for k in missing if k not in buffers and not is_base_backbone_key(k)]
            if len(missing) > len(other):
                print(
                    f"[checkpoint] {len(missing) - len(other)} frozen-backbone / "
                    f"buffer key(s) not stored in the checkpoint (rebuilt at "
                    f"construction, expected)"
                )
            if other:
                print(
                    f"[checkpoint] WARNING: {len(other)} missing key(s) outside the "
                    f"frozen backbone, e.g. {other[:5]}. They keep their initial "
                    f"values; the checkpoint is incomplete for this model."
                )
            if unexpected:
                print(
                    f"[checkpoint] {len(unexpected)} unexpected key(s), "
                    f"e.g. {list(unexpected)[:3]}"
                )
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def validate_checkpoint_cfg(
    ckpt_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    """Fail loudly when resuming with an incompatible architecture/config."""
    keys = (
        "profile",
        "pool",
        "readout",
        "use_mona",
        "use_dora",
        "target_size",
        "input_mode",
        "dora_target_names",
        "mona_blocks",
        "dora_blocks",
    )
    mismatches: List[Tuple[str, Any, Any]] = []
    for key in keys:
        if key not in ckpt_cfg:
            continue
        old, new = ckpt_cfg.get(key), cfg.get(key)
        if old != new:
            mismatches.append((key, old, new))
    if mismatches:
        lines = [f"  {k}: checkpoint={ov!r} current={nv!r}" for k, ov, nv in mismatches]
        raise ValueError(
            "Checkpoint cfg is incompatible with the current run config:\n"
            + "\n".join(lines)
        )


__all__ = ["save", "load", "load_into", "validate_checkpoint_cfg", "is_base_backbone_key"]
