"""Trainer for DinoV3RootRegressor — DDP + grad-accum + cosine schedule.

Pure regression of (length, area). No SAM / SWA / Kendall / presence gate —
those were the main instability sources in the old pipeline. Optional EMA of the
trainable weights (``use_ema``; best.pt then holds the EMA weights). Single
optimizer step per accumulation cycle, AdamW with one LR group per parameter
family (DoRA, Mona, unfrozen backbone blocks, head), cosine LR with linear
warmup, optional bf16 autocast (no GradScaler needed).

Multi-GPU: launch with torchrun; the trainer detects the process group and
wraps the model in DDP when world_size > 1. Validation runs on rank 0 over the
full val set (the val loader is not sharded).
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:    # pragma: no cover
    SummaryWriter = None    # type: ignore[assignment,misc]
    _TB_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
    _SKLEARN_AVAILABLE = True
except Exception:    # pragma: no cover
    _sk_roc_auc_score = None    # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except Exception:    # pragma: no cover
    tqdm = None    # type: ignore[assignment]
    _TQDM_AVAILABLE = False

from RootQuantV2.data.target_norm import invert as _invert_targets
from RootQuantV2.model import tta_predict_mean
from RootQuantV2.training.checkpoint import is_base_backbone_key, save as _ckpt_save
from RootQuantV2.training.loss import RootRegressionLoss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper: attach presence_mask derived from RAW labels
# ─────────────────────────────────────────────────────────────────────────────


class _PresenceWrappedDataset(Dataset):
    """Wrap a (image, std_target) dataset and attach presence_mask from raw labels."""

    def __init__(self, base: Dataset, raw_labels: Tensor) -> None:
        if len(base) != raw_labels.shape[0]:
            raise ValueError(
                f"dataset length {len(base)} != raw_labels rows {raw_labels.shape[0]}"
            )
        self.base = base
        self.raw = raw_labels.float()
        # Precompute presence once (kills two .item() syncs per __getitem__).
        self._present = ((self.raw[:, 0] > 0) & (self.raw[:, 1] > 0)).float()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = self.base[idx]
        if len(item) == 3:
            img, label, patch_mask = item
        else:
            img, label = item
            patch_mask = None
        out: Dict[str, Tensor] = {
            "image": img,
            "target": label,
            "presence_mask": self._present[idx],
            # Global sample index for exact dedup after sharded-val all_gather
            # (DistributedSampler pads the last shard by repeating early indices).
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if patch_mask is not None:
            out["patch_mask"] = patch_mask
        return out


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


class ParamEMA:
    """Exponential moving average over the TRAINABLE parameters only.

    Keyed by the unwrapped model's parameter names, so it is agnostic to DDP /
    torch.compile wrapping. The frozen backbone is never touched — only the
    trainable adapter+head params (11.90 M for the headline model) get a
    smoothed copy that is used for validation and saved into ``best.pt``. Cheap
    (one extra fp32 copy of the trainable set) and unrelated to the old
    Kendall/presence-gate instability.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                s = self.shadow.get(n)
                if s is not None:
                    s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> Dict[str, Tensor]:
        """Copy EMA weights into the live model; return a backup to restore."""
        backup: Dict[str, Tensor] = {}
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                backup[n] = p.detach().clone()
                p.copy_(s)
        return backup

    @torch.no_grad()
    def swap_out(self, model: nn.Module, backup: Dict[str, Tensor]) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.copy_(backup[n])

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """Train a DinoV3RootRegressor: presence-balanced loss, AdamW + cosine
    schedule, optional EMA, per-epoch validation on present rows, and slim
    ``best.pt`` / ``last.pt`` checkpoints under ``checkpoint_dir/run_name``."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        raw_train_labels: Tensor,
        raw_val_labels: Tensor,
    ) -> None:
        self.cfg = cfg
        self.model = model

        # ── Device & DDP ─────────────────────────────────────────────────
        self._ddp = bool(
            getattr(dist, "is_available", lambda: False)()
            and getattr(dist, "is_initialized", lambda: False)()
        )
        if self._ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(cfg.get("local_rank", self.rank))
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, int(cfg.get("local_rank", 0))
        self.is_main = self.rank == 0

        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # ── AMP ──────────────────────────────────────────────────────────
        self.amp_dtype = str(cfg.get("amp_dtype", "fp32")).lower()
        self._use_bf16 = self.amp_dtype == "bf16" and torch.cuda.is_available()

        # ── DDP wrap (only when >1 rank) ─────────────────────────────────
        if self._ddp and self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        # ── torch.compile (after DDP wrap; does not change precision) ────
        if bool(cfg.get("use_torch_compile", False)) and hasattr(torch, "compile"):
            mode = str(cfg.get("torch_compile_mode", "default"))
            dynamic = bool(cfg.get("torch_compile_dynamic", False))
            self.model = torch.compile(self.model, mode=mode, dynamic=dynamic)
            if self.is_main:
                print(f"[Trainer] torch.compile enabled (mode={mode}, dynamic={dynamic})")

        # ── Datasets / loaders ───────────────────────────────────────────
        self._raw_train_labels = raw_train_labels.float()
        self._train_ds = _PresenceWrappedDataset(train_dataset, raw_train_labels)
        self._val_ds = _PresenceWrappedDataset(val_dataset, raw_val_labels)

        self.train_loader, self._train_sampler = self._build_loader(
            self._train_ds, int(cfg["per_gpu_batch"]), shuffle=True,
            num_workers=int(cfg.get("train_num_workers", 8)), drop_last=True,
        )
        self.val_loader, self._val_sampler = self._build_loader(
            self._val_ds, int(cfg.get("val_batch", cfg["per_gpu_batch"])), shuffle=False,
            num_workers=int(cfg.get("eval_num_workers", 4)), drop_last=False,
            force_distributed=True,
        )

        # ── Loss (presence-balanced) ─────────────────────────────────────
        w_present, w_empty = self._balance_weights()
        self.loss_fn = RootRegressionLoss(
            huber_delta=float(cfg.get("huber_delta", 1.0)),
            w_present=w_present,
            w_empty=w_empty,
            loss_type=str(cfg.get("loss_type", "smooth_l1")),
            task_weights=tuple(cfg.get("task_weights", (1.0, 1.0))),
        ).to(self.device)

        # ── Optimizer / scheduler ────────────────────────────────────────
        self.optimizer = self._build_optimizer()

        # Cache the trainable set once (computed on the UNWRAPPED model so the
        # names match state_dict keys for slim checkpointing). The param objects
        # are shared with the DDP wrapper, so grad-clip sees the populated .grad.
        _unwrapped = self._unwrap()
        self._trainable_params = [p for p in _unwrapped.parameters() if p.requires_grad]
        # Parameters written to checkpoints: everything except the frozen base
        # backbone, whether or not it trains in this run. finetune_mode=fc_only
        # freezes DoRA / Mona / the pooler, but they are still part of the model
        # and must be saved (unfrozen backbone blocks are trainable, so included).
        self._ckpt_names = {
            n for n, p in _unwrapped.named_parameters()
            if p.requires_grad or not is_base_backbone_key(n)
        }

        # ── EMA over trainable params (opt-in) ───────────────────────────
        self.ema: Optional[ParamEMA] = None
        if bool(cfg.get("use_ema", False)):
            self.ema = ParamEMA(self._unwrap(), float(cfg.get("ema_decay", 0.9995)))
            if self.is_main:
                print(f"[Trainer] EMA enabled (decay={self.ema.decay})")

        self.steps_per_epoch = max(1, math.ceil(
            len(self.train_loader) / max(1, int(cfg["grad_accum_steps"]))
        ))
        self.total_steps = self.steps_per_epoch * int(cfg["num_epochs"])
        self.scheduler = self._build_scheduler()

        # ── TensorBoard / bookkeeping ────────────────────────────────────
        if self.is_main and _TB_AVAILABLE:
            os.makedirs(cfg["logs_dir"], exist_ok=True)
            self.tb = SummaryWriter(log_dir=os.path.join(cfg["logs_dir"], cfg.get("run_name", "v2")))
        else:
            self.tb = None
        self.global_step = 0
        self.epoch = 0
        self.start_epoch = 0
        self.best_combined_r2 = -float("inf")
        self._history: List[Dict[str, float]] = []

        if self.is_main:
            self._log_groups()

    # ── construction helpers ─────────────────────────────────────────────

    def _build_loader(self, ds, batch_size, shuffle, num_workers, drop_last,
                      force_distributed=False):
        sampler = None
        if self._ddp and self.world_size > 1 and (shuffle or force_distributed):
            sampler = DistributedSampler(
                ds, num_replicas=self.world_size, rank=self.rank,
                shuffle=shuffle, drop_last=drop_last,
            )
            shuffle = False
        kwargs: Dict[str, Any] = dict(
            batch_size=batch_size, sampler=sampler, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last,
            pin_memory=bool(self.cfg.get("pin_memory", True)) and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.cfg.get("prefetch_factor", 2))
        return DataLoader(ds, **kwargs), sampler

    def _balance_weights(self) -> Tuple[float, float]:
        labels = self._raw_train_labels
        mask = (labels[:, 0] > 0) & (labels[:, 1] > 0)
        present_rate = float(mask.float().mean().item())

        w_present_cfg = self.cfg.get("w_present")
        w_empty_cfg = self.cfg.get("w_empty")
        if w_present_cfg is not None or w_empty_cfg is not None:
            w_present = float(w_present_cfg if w_present_cfg is not None else 1.0)
            w_empty = float(w_empty_cfg if w_empty_cfg is not None else 1.0)
        elif bool(self.cfg.get("balance_presence", True)) and 0.0 < present_rate < 1.0:
            w_present = 0.5 / present_rate
            w_empty = 0.5 / (1.0 - present_rate)
        else:
            w_present, w_empty = 1.0, 1.0

        if self.is_main:
            print(f"[Trainer] present_rate={present_rate:.4f}  "
                  f"w_present={w_present:.3f}  w_empty={w_empty:.3f}")
        return w_present, w_empty

    def _build_optimizer(self):
        cfg = self.cfg
        wd_head = float(cfg.get("weight_decay", 1e-2))
        wd_mona = float(cfg.get("mona_weight_decay", wd_head))
        dora_params: List[Tensor] = []
        mona_decay: List[Tensor] = []
        mona_no_decay: List[Tensor] = []
        backbone_params: List[Tensor] = []
        head_params: List[Tensor] = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "dora_" in name:
                dora_params.append(p)
            elif "mona" in name:
                if name.endswith("mona_scale") or name.endswith(".bias"):
                    mona_no_decay.append(p)
                else:
                    mona_decay.append(p)
            elif "backbone" in name and ".blocks." in name:
                # Trainable backbone-block weights that are NOT DoRA/Mona adapters
                # come from unfreeze_last_n_blocks>0 — give them a dedicated tiny
                # LR group so they fine-tune gently without destabilizing the
                # pretrained representation.
                backbone_params.append(p)
            else:
                head_params.append(p)

        groups: List[Dict[str, Any]] = []
        if dora_params:
            groups.append({
                "params": dora_params,
                "lr": float(cfg["lr_dora"]),
                "weight_decay": 0.0,
                "name": "dora",
            })
        if mona_decay:
            groups.append({
                "params": mona_decay,
                "lr": float(cfg.get("lr_mona", cfg["lr_dora"])),
                "weight_decay": wd_mona,
                "name": "mona",
            })
        if mona_no_decay:
            groups.append({
                "params": mona_no_decay,
                "lr": float(cfg.get("lr_mona", cfg["lr_dora"])),
                "weight_decay": 0.0,
                "name": "mona_no_decay",
            })
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": float(cfg.get("lr_backbone", 1e-5)),
                "weight_decay": wd_head,
                "name": "backbone",
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": float(cfg["lr_head"]),
                "weight_decay": wd_head,
                "name": "head",
            })
        if not groups:
            raise RuntimeError("No trainable parameters found.")
        self._group_names = [g["name"] for g in groups]
        return optim.AdamW(groups, betas=tuple(cfg.get("betas", (0.9, 0.999))))

    def _build_scheduler(self):
        warmup = max(1, int(round(float(self.cfg.get("warmup_frac", 0.05)) * self.total_steps)))
        total = max(1, self.total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return LambdaLR(self.optimizer, lr_lambda)

    def _log_groups(self) -> None:
        print("[Trainer] optimizer groups:")
        for g in self.optimizer.param_groups:
            n = sum(p.numel() for p in g["params"])
            print(f"  {g.get('name','?'):<6} lr={g['lr']:.2e} wd={g['weight_decay']:.1e} params={n:,}")

    def _autocast(self):
        if self._use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    # ── train epoch ──────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        cfg = self.cfg
        n_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
        grad_clip = float(cfg.get("grad_clip_norm", 0.0))
        is_ddp = isinstance(self.model, DDP)

        # Accumulate on-GPU to avoid a host<->device sync every micro-batch
        # (the old .item()×3/step forced ~33k syncs/epoch). One sync at epoch end.
        loss_sum = torch.zeros((), device=self.device)
        len_sum = torch.zeros((), device=self.device)
        area_sum = torch.zeros((), device=self.device)
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        it = self.train_loader
        if _TQDM_AVAILABLE and self.is_main:
            it = tqdm(self.train_loader, total=len(self.train_loader),
                      desc=f"train ep {epoch}", leave=False, dynamic_ncols=True)

        accum = 0
        batch_idx = 0
        for batch_idx, batch in enumerate(it):
            x = batch["image"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            mask = batch["presence_mask"].float().to(self.device, non_blocking=True)
            patch_mask = batch.get("patch_mask")
            if patch_mask is not None:
                patch_mask = patch_mask.to(self.device, non_blocking=True)

            is_last = (accum + 1) == n_accum
            sync_ctx = nullcontext() if (is_last or not is_ddp) else self.model.no_sync()
            with sync_ctx:
                with self._autocast():
                    preds = self.model(x, patch_mask=patch_mask)
                    loss, comps = self.loss_fn(preds, target, mask)
                (loss / n_accum).backward()

            loss_sum += loss.detach()
            len_sum += comps["loss_length"].detach()
            area_sum += comps["loss_area"].detach()
            n_batches += 1
            accum += 1

            if accum < n_accum:
                continue

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self._trainable_params, grad_clip)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self._unwrap())
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_step += 1
            accum = 0

        # Global mean-of-batch-means across ranks. drop_last=True (loader AND
        # DistributedSampler) gives every rank an identical micro-batch count, so
        # sum/(n_batches*world_size) is the exact global mean. NOTE: each per-batch
        # term is itself a presence-weighted mean, so this is mean-of-batch-means
        # (the pre-existing definition), now global instead of rank-0-local.
        if self._ddp and self.world_size > 1:
            dist.all_reduce(loss_sum)
            dist.all_reduce(len_sum)
            dist.all_reduce(area_sum)
            denom = max(1, n_batches * self.world_size)
        else:
            denom = max(1, n_batches)
        sums = torch.stack([loss_sum, len_sum, area_sum]).cpu().tolist()  # single sync
        metrics = {
            "train/loss": sums[0] / denom,
            "train/loss_length": sums[1] / denom,
            "train/loss_area": sums[2] / denom,
            "train/seconds": time.time() - t0,
        }
        if self.is_main and self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, self.global_step)
            for g in self.optimizer.param_groups:
                self.tb.add_scalar(f"lr/{g.get('name','?')}", g["lr"], self.global_step)
        return metrics

    # ── validation ───────────────────────────────────────────────────────

    @contextmanager
    def _ema_weights(self):
        """Temporarily swap EMA weights into the live model (no-op if EMA off)."""
        if self.ema is None:
            yield
            return
        m = self._unwrap()
        backup = self.ema.swap_in(m)
        try:
            yield
        finally:
            self.ema.swap_out(m, backup)

    def _invert(self, y: Tensor) -> Tensor:
        """Standardized → raw units. y: (N, 2). Log-aware via target_stats."""
        if str(self.cfg.get("target_transform", "zscore")) == "none":
            return y
        return _invert_targets(y, self.cfg["target_stats"])

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        # Under DDP each rank scores its own shard (val_loader has a
        # DistributedSampler); results are gathered to rank 0. Single-GPU runs
        # the full loader on the one rank.
        sharded = self._ddp and self.world_size > 1 and self._val_sampler is not None

        preds_all, targets_all, masks_all, idx_all = [], [], [], []
        it = self.val_loader
        if _TQDM_AVAILABLE and self.is_main:
            it = tqdm(self.val_loader, total=len(self.val_loader),
                      desc=f"val ep {epoch}", leave=False, dynamic_ncols=True)
        use_tta = bool(self.cfg.get("val_tta", False))
        # EMA swap runs on EVERY rank — shadows are bit-identical across ranks
        # (the update runs on all ranks under DDP param sync), so each shard is
        # scored with the same EMA weights.
        with self._ema_weights():
            for batch in it:
                x = batch["image"].to(self.device, non_blocking=True)
                patch_mask = batch.get("patch_mask")
                if patch_mask is not None:
                    patch_mask = patch_mask.to(self.device, non_blocking=True)
                with self._autocast():
                    if use_tta:
                        out = tta_predict_mean(self.model, x, patch_mask)
                    else:
                        out = self.model(x, patch_mask=patch_mask)
                preds_all.append(torch.stack([out["length"], out["area"]], -1).float().cpu())
                targets_all.append(batch["target"].float())
                masks_all.append(batch["presence_mask"].float())
                idx_all.append(batch["index"].long())

        local = {
            "preds": torch.cat(preds_all) if preds_all else torch.zeros(0, 2),
            "targets": torch.cat(targets_all) if targets_all else torch.zeros(0, 2),
            "masks": torch.cat(masks_all) if masks_all else torch.zeros(0),
            "index": torch.cat(idx_all) if idx_all else torch.zeros(0, dtype=torch.long),
        }

        if sharded:
            gathered: List[Optional[Dict[str, Tensor]]] = [None] * self.world_size
            dist.all_gather_object(gathered, local)
            if not self.is_main:
                return {}
            preds = torch.cat([g["preds"] for g in gathered])
            targets = torch.cat([g["targets"] for g in gathered])
            masks = torch.cat([g["masks"] for g in gathered])
            index = torch.cat([g["index"] for g in gathered])
            # DistributedSampler(drop_last=False) pads the last shard by repeating
            # early indices; keep the first occurrence per index to recover the
            # exact val set (metrics are order-independent).
            seen: set = set()
            keep: List[int] = []
            for pos, ix in enumerate(index.tolist()):
                if ix not in seen:
                    seen.add(ix)
                    keep.append(pos)
            keep_t = torch.tensor(keep, dtype=torch.long)
            preds, targets, masks = preds[keep_t], targets[keep_t], masks[keep_t]
        else:
            if not self.is_main:
                return {}
            preds, targets, masks = local["preds"], local["targets"], local["masks"]

        if preds.numel() == 0:
            return {}
        preds = self._invert(preds).clamp_min(0.0)   # raw units, ≥0
        targets = self._invert(targets)              # raw units
        return self._metrics(preds, targets, masks.bool())

    def _metrics(self, preds: Tensor, targets: Tensor, masks: Tensor) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n_present = int(masks.sum().item())
        out["val/n_present"] = float(n_present)

        if n_present == 0:
            for k in ("r2_length", "r2_area", "combined_r2", "rmse_length",
                      "rmse_area", "mae_length", "mae_area"):
                out[f"val/{k}"] = float("nan")
        else:
            p = preds[masks].numpy().astype(np.float64)
            t = targets[masks].numpy().astype(np.float64)
            for i, name in enumerate(("length", "area")):
                err = p[:, i] - t[:, i]
                out[f"val/mae_{name}"] = float(np.mean(np.abs(err)))
                out[f"val/rmse_{name}"] = float(np.sqrt(np.mean(err ** 2)))
                out[f"val/r2_{name}"] = _r2(p[:, i], t[:, i])
            r2l, r2a = out["val/r2_length"], out["val/r2_area"]
            out["val/combined_r2"] = (
                float("nan") if (math.isnan(r2l) or math.isnan(r2a)) else 0.5 * (r2l + r2a)
            )

        # Derived presence (from predicted magnitude — no presence head exists).
        score = (preds[:, 0] + preds[:, 1]).numpy().astype(np.float64)
        y = masks.numpy().astype(np.int64)
        out["val/presence_acc"] = float(np.mean((score > 1.0).astype(np.int64) == y))
        try:
            if _SKLEARN_AVAILABLE and len(np.unique(y)) == 2:
                out["val/presence_auroc"] = float(_sk_roc_auc_score(y, score))
            else:
                out["val/presence_auroc"] = float("nan")
        except Exception:
            out["val/presence_auroc"] = float("nan")
        return out

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, Any]:
        cfg = self.cfg
        history: List[Dict[str, float]] = list(self._history)
        validate_every = max(1, int(cfg.get("validate_every_n_epochs", 1)))
        run_ckpt_dir = os.path.join(cfg["checkpoint_dir"], cfg.get("run_name", "v2"))
        if self.is_main:
            os.makedirs(run_ckpt_dir, exist_ok=True)

        start = int(self.start_epoch)
        if self.is_main and start > 0:
            print(f"[Trainer] resuming training from epoch {start}")

        for epoch in range(start, int(cfg["num_epochs"])):
            self.epoch = epoch
            train_metrics = self.train_epoch(epoch)

            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(epoch)
                if self.is_main:
                    history.append({"epoch": float(epoch), **train_metrics, **val_metrics})
                    if self.tb is not None:
                        for k, v in val_metrics.items():
                            try:
                                self.tb.add_scalar(k, float(v), self.global_step)
                            except Exception:
                                pass
                    self._print_log(epoch, train_metrics, val_metrics)

                    comb = val_metrics.get("val/combined_r2", float("nan"))
                    if not math.isnan(comb) and comb > self.best_combined_r2:
                        self.best_combined_r2 = comb
                        # best.pt holds the EMA weights (what was validated) when EMA is on.
                        self._save(os.path.join(run_ckpt_dir, "best.pt"), epoch,
                                   history, val_metrics, ema_weights=True,
                                   include_optimizer=False)
                    self._save(os.path.join(run_ckpt_dir, "last.pt"), epoch, history, val_metrics)

            if self._ddp and self.world_size > 1:
                dist.barrier()

        if self.is_main and self.tb is not None:
            self.tb.flush()
        return {"history": history, "best_combined_r2": self.best_combined_r2}

    # ── logging / checkpoint ─────────────────────────────────────────────

    def _print_log(self, epoch, tr, va) -> None:
        print(
            f"[ep {epoch:03d}] train_loss={tr.get('train/loss', float('nan')):.4f} "
            f"L={tr.get('train/loss_length', float('nan')):.4f} "
            f"A={tr.get('train/loss_area', float('nan')):.4f} "
            f"| val r2_L={va.get('val/r2_length', float('nan')):.4f} "
            f"r2_A={va.get('val/r2_area', float('nan')):.4f} "
            f"comb={va.get('val/combined_r2', float('nan')):.4f} "
            f"mae_L={va.get('val/mae_length', float('nan')):.3f} "
            f"mae_A={va.get('val/mae_area', float('nan')):.3f} "
            f"step={self.global_step}"
        )

    def _unwrap(self) -> nn.Module:
        """Peel DDP (.module) and torch.compile (._orig_mod) in any order."""
        m = self.model
        for _ in range(4):
            if isinstance(m, DDP):
                m = m.module
            elif hasattr(m, "_orig_mod"):
                m = m._orig_mod
            else:
                break
        return m

    def _save(self, path, epoch, history, metrics, ema_weights: bool = False,
              include_optimizer: bool = True) -> None:
        if not self.is_main:
            return
        m = self._unwrap()
        # Optionally serialize the EMA weights as model_state (used for best.pt so
        # inference loads exactly what was validated). torch.save reads live storage,
        # so swap in → save → swap out captures EMA without an extra full-model copy.
        backup = self.ema.swap_in(m) if (ema_weights and self.ema is not None) else None
        try:
            # Slim checkpoint: save the adapter / pooler / head params (11.90 M for
            # the headline model, ~91 MB with the EMA copy) plus any unfrozen
            # backbone blocks, not the full ~315M-param state (~1.2 GB). The frozen
            # backbone, DoRA V_frozen / bias buffers, RoPE periods, and density
            # t_mean/t_std are all rebuilt at construction (the last from
            # cfg["target_stats"], saved below), so a strict=False load fully
            # restores the model.
            slim_state = {
                k: v for k, v in m.state_dict().items() if k in self._ckpt_names
            }
            ema_state = None
            if self.ema is not None:
                # The shadow covers trainable params only; store the frozen
                # adapters' fixed values alongside so it has model_state's keys.
                ema_state = self.ema.state_dict()
                ema_state["shadow"] = {
                    **{k: v for k, v in slim_state.items() if k not in self.ema.shadow},
                    **self.ema.shadow,
                }
            _ckpt_save(
                {
                    "model_state": slim_state,
                    "ema_state": ema_state,
                    # best.pt drops the optimizer (resume uses last.pt); AdamW state
                    # for the trainable set is only needed to continue training.
                    "optimizer_state": (
                        self.optimizer.state_dict() if include_optimizer else None
                    ),
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": int(epoch),
                    "global_step": int(self.global_step),
                    "best_combined_r2": float(self.best_combined_r2),
                    "target_stats": self.cfg.get("target_stats", {}),
                    "cfg": self.cfg,
                    "metrics": metrics,
                    "history": history,
                },
                path,
            )
        finally:
            if backup is not None:
                self.ema.swap_out(m, backup)


__all__ = ["Trainer"]
