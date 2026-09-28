"""RootQuantV2 training CLI — DINOv3 + DoRA root length/area regression.

Single-GPU:
    python -m RootQuantV2.train

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=4 -m RootQuantV2.train

The trainer auto-detects the torch.distributed process group created below and
wraps the model in DDP when world_size > 1.
"""

import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

from RootQuantV2.config import check_data_paths, get_config, list_ablations, list_profiles
from RootQuantV2.data import RootDataset, fit_zscore, get_transforms, load_split
from RootQuantV2.model import DinoV3RootRegressor
from RootQuantV2.training import Trainer, load, load_into, validate_checkpoint_cfg


def _maybe_init_distributed(cfg: Dict[str, Any]) -> None:
    """Create the process group when launched by torchrun."""
    if dist.is_available() and not dist.is_initialized() \
            and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
        cfg["local_rank"] = local_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m RootQuantV2.train")
    p.add_argument("--profile", default="dinov3_dora", choices=list(list_profiles()))
    p.add_argument("--ablate", default=None, choices=list(list_ablations()))
    p.add_argument("--target_size", type=int, default=None,
                   help="Letterbox canvas side length (must divide patch size 16).")
    p.add_argument("--pool", default=None,
                   help="Pooling mode override (e.g. cls_attn_gem, cls_mean_max).")
    p.add_argument("--readout", default=None, choices=["global", "density", "both"],
                   help="Output readout: global (pooled MLP), density (extensive sum), both.")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--per_gpu_batch", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--lr_dora", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--lr_mona", type=float, default=None)
    p.add_argument("--head_dropout", type=float, default=None)
    p.add_argument("--target_transform", default=None,
                   choices=["zscore", "log_zscore", "none"],
                   help="log_zscore fits the z-score in log1p space (heavy-tailed targets).")
    p.add_argument("--gem_mode", default=None, choices=["legacy", "softplus"],
                   help="softplus = sound GeM on signed tokens (only affects cls_attn_gem).")
    p.add_argument("--use_ema", default=None, action=argparse.BooleanOptionalAction,
                   help="EMA over trainable params; best.pt holds the EMA weights.")
    p.add_argument("--ema_decay", type=float, default=None)
    p.add_argument("--eval_tta", default=None, action=argparse.BooleanOptionalAction,
                   help="Average predictions over the 8 D4 views at val/inference.")
    p.add_argument("--loss_type", default=None, choices=["smooth_l1", "huber", "mse"],
                   help="Residual penalty. huber(δ≈3)/mse maximize R²; smooth_l1 caps the tail.")
    p.add_argument("--huber_delta", type=float, default=None,
                   help="delta for huber / beta for smooth_l1 (z-space).")
    p.add_argument("--w_empty", type=float, default=None,
                   help="Loss weight for empty rows (0 = present-only training).")
    p.add_argument("--w_present", type=float, default=None)
    p.add_argument("--amp", choices=["fp32", "bf16"], default=None,
                   help="Override amp_dtype (bf16 needs no GradScaler).")
    p.add_argument("--grad_checkpointing", type=int, choices=[0, 1], default=None,
                   help="1=activation checkpointing ON (less VRAM); 0=OFF (faster, more VRAM).")
    p.add_argument("--use_tile_shuffle", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="Train-time tile-shuffle augmentation (locality-disrupting).")
    p.add_argument("--tile_shuffle_grids", default=None,
                   help="Comma-separated tile grid sizes, e.g. 2,4,8.")
    p.add_argument("--tile_shuffle_p", default=None,
                   help="Tile-shuffle probability: one float (one grid sampled per "
                        "image) OR comma-separated per-grid probs matching "
                        "--tile_shuffle_grids (each scale fires independently), "
                        "e.g. 0.2,0.2,0.2.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--resume_check", action="store_true",
                   help="Build the trainer, run the full resume/LR-re-anchor path, "
                        "print the resumed step/LR diagnostics, and exit BEFORE "
                        "fit(). Use to verify a cross-GPU-count resume is sound "
                        "without committing to a real run.")
    p.add_argument("--init_from", default=None,
                   help="Warm-start: load the adapter / pooler / head weights from this "
                        "checkpoint (slim, strict=False) WITHOUT restoring optimizer/scheduler/"
                        "epoch. target_stats are refit on THIS run's train split. "
                        "Use for cross-dataset fine-tuning (e.g. soy->maize). "
                        "Mutually exclusive with --resume.")
    p.add_argument("--finetune_mode", default=None, choices=["fc_only", "full"],
                   help="fc_only: freeze the feature extractor (backbone/DoRA/Mona/"
                        "pooler) and train ONLY the regression readout "
                        "(head/density_head/readout_blend). full: train the whole "
                        "warm-started trainable set. Only meaningful with --init_from.")
    return p.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("train_csv", "val_csv", "test_csv", "data_path", "run_name"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
        cfg["checkpoint_dir"] = os.path.join(args.output_root, "checkpoints")
        cfg["logs_dir"] = os.path.join(args.output_root, "logs")
    for key in ("num_epochs", "per_gpu_batch", "grad_accum_steps", "lr_dora", "lr_head"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if args.amp is not None:
        cfg["amp_dtype"] = args.amp
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    if args.target_size is not None:
        cfg["target_size"] = args.target_size
        if cfg.get("input_mode") == "letterbox_square_640":
            cfg["input_mode"] = "letterbox_square"
        if (
            args.target_size >= 768
            and not cfg.get("use_grad_checkpointing")
            and cfg.get("profile") == "dinov3_dora"
        ):
            cfg["use_grad_checkpointing"] = True
    if args.pool is not None:
        cfg["pool"] = args.pool
    if args.readout is not None:
        cfg["readout"] = args.readout
    if args.lr_mona is not None:
        cfg["lr_mona"] = args.lr_mona
    if args.head_dropout is not None:
        cfg["head_dropout"] = args.head_dropout
    for key in ("target_transform", "gem_mode", "ema_decay", "loss_type", "huber_delta"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if args.use_ema is not None:
        cfg["use_ema"] = bool(args.use_ema)
    if args.eval_tta is not None:
        cfg["eval_tta"] = bool(args.eval_tta)
    if args.w_empty is not None:
        cfg["w_empty"] = args.w_empty
        if args.w_empty == 0.0:
            cfg["balance_presence"] = False
    if args.w_present is not None:
        cfg["w_present"] = args.w_present
    # Explicit override wins over the profile default AND the target_size>=768
    # auto-enable above (which only targets the base dinov3_dora profile).
    if args.grad_checkpointing is not None:
        cfg["use_grad_checkpointing"] = bool(args.grad_checkpointing)
    if args.use_tile_shuffle is not None:
        cfg["use_tile_shuffle"] = bool(args.use_tile_shuffle)
    if args.tile_shuffle_grids is not None:
        cfg["tile_shuffle_grids"] = tuple(
            int(g) for g in args.tile_shuffle_grids.split(",") if g.strip()
        )
    if args.tile_shuffle_p is not None:
        probs = [float(x) for x in args.tile_shuffle_p.split(",") if x.strip()]
        # One value → scalar (legacy) mode; several → per-grid independent probs.
        cfg["tile_shuffle_p"] = probs[0] if len(probs) == 1 else tuple(probs)


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_from is not None:
        raise ValueError(
            "--resume and --init_from are mutually exclusive: --resume continues an "
            "interrupted run (restores optimizer/epoch/stats); --init_from warm-starts "
            "a NEW run from another checkpoint's weights."
        )
    cfg = get_config(args.profile, ablation=args.ablate)
    apply_overrides(cfg, args)
    check_data_paths(cfg)

    _maybe_init_distributed(cfg)
    is_main = (not dist.is_initialized()) or dist.get_rank() == 0

    seed = int(cfg.get("random_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "high"))
    # Fixed input shapes (letterbox to a constant square) + drop_last=True let the
    # cuDNN autotuner lock in optimal kernels after a one-batch warmup. Free win.
    if cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    if is_main:
        print(f"[TrainCLI] profile={cfg['profile']} ablation={cfg.get('ablation')} "
              f"run_name={cfg['run_name']} seed={seed} amp={cfg['amp_dtype']}")

    # ── Data ─────────────────────────────────────────────────────────────
    splits = load_split(
        cfg["train_csv"], cfg["val_csv"], cfg["test_csv"], cfg["data_path"],
        image_col=cfg.get("image_col", "ImageName"),
        length_col=cfg.get("length_col", "AliveLength(mm)"),
        area_col=cfg.get("area_col", "AliveSurfArea(mm2)"),
    )
    train_paths, train_labels_raw = splits["train"]
    val_paths, val_labels_raw = splits["val"]
    if is_main:
        print(f"[TrainCLI] train={len(train_paths)} val={len(val_paths)}")

    resume_payload = None
    if args.resume is not None:
        resume_payload = load(args.resume, map_location="cpu")
        validate_checkpoint_cfg(resume_payload.get("cfg", {}), cfg)

    # z-score stats: prefer checkpoint stats when resuming. "log_zscore" fits
    # the same z-score in log1p space (heavy right-tailed root targets).
    target_transform = cfg.get("target_transform")
    if target_transform in ("zscore", "log_zscore"):
        if resume_payload is not None and resume_payload.get("target_stats"):
            stats = resume_payload["target_stats"]
            cfg["target_stats"] = stats
            if is_main:
                print(f"[TrainCLI] restored target_stats from checkpoint: {stats}")
        else:
            col_tf = cfg.get("target_col_transforms")  # e.g. ("none", "sqrt")
            stats = fit_zscore(
                train_labels_raw,
                log=(target_transform == "log_zscore"),
                transforms=tuple(col_tf) if col_tf else None,
            )
            cfg["target_stats"] = stats
            if is_main:
                print(f"[TrainCLI] target_stats ({target_transform}): {stats}")
    else:
        stats = None

    # The density head's t_mean/t_std come from cfg["target_stats"] and are dropped
    # from slim checkpoints (rebuilt at construction). Fail loudly if a density
    # readout is requested without stats, rather than silently using (0,0)/(1,1).
    if cfg.get("readout") in ("density", "both") and not cfg.get("target_stats"):
        raise ValueError(
            f"readout={cfg.get('readout')!r} requires non-empty target_stats "
            f"(density head t_mean/t_std). Use target_transform zscore/log_zscore, "
            f"or set readout=global."
        )

    train_tf = get_transforms(cfg["input_mode"], train=True, cfg=cfg)
    eval_tf = get_transforms(cfg["input_mode"], train=False, cfg=cfg)
    train_ds = RootDataset(train_paths, train_labels_raw, transform=train_tf, target_norm_stats=stats)
    val_ds = RootDataset(val_paths, val_labels_raw, transform=eval_tf, target_norm_stats=stats)

    # ── Model ────────────────────────────────────────────────────────────
    model = DinoV3RootRegressor(cfg)

    # Warm-start (cross-dataset fine-tuning): load the adapter / pooler / head
    # weights from a prior checkpoint WITHOUT touching the optimizer/scheduler/epoch (those are
    # what --resume restores). target_stats were refit on THIS run's train split
    # above, so the density head buffers / z-score inversion are calibrated to the
    # new dataset while the adapters+head start from the source model.
    if args.init_from is not None:
        init_payload = load(args.init_from, map_location="cpu")
        validate_checkpoint_cfg(init_payload.get("cfg", {}), cfg)
        load_into(args.init_from, model=model, map_location="cpu", strict=False)
        if is_main:
            print(f"[TrainCLI] warm-started adapter / pooler / head weights from {args.init_from}")

    # FC-only fine-tuning: freeze the (soy-trained) feature extractor — backbone,
    # DoRA, Mona, and the learned pooler — and train ONLY the regression readout
    # (global MLP head + extensive density head + per-target readout blend).
    # 'full' (or None) leaves the warm-started trainable set untouched.
    if args.finetune_mode == "fc_only":
        if args.init_from is None and is_main:
            print("[TrainCLI] WARNING: --finetune_mode fc_only without --init_from — "
                  "freezing a freshly-initialized feature extractor.")
        head_prefixes = ("head.", "density_head.")
        for name, p in model.named_parameters():
            keep = name.startswith(head_prefixes) or name == "readout_blend"
            if not keep:
                p.requires_grad_(False)
        if is_main:
            print("[TrainCLI] finetune_mode=fc_only — froze all but the regression "
                  "readout (head / density_head / readout_blend).")

    if is_main:
        model.print_summary()

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg=cfg, model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        raw_train_labels=train_labels_raw, raw_val_labels=val_labels_raw,
    )

    if resume_payload is not None:
        # Slim checkpoints omit the frozen base backbone (rebuilt at
        # construction), so its keys show as "missing" — expected.
        load_into(
            args.resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            map_location="cpu",
            strict=False,
        )
        # The EMA shadow was built from the freshly initialized weights in
        # Trainer.__init__; restore it (or, if the checkpoint has none, re-seed
        # it from the weights just loaded).
        if trainer.ema is not None:
            saved_shadow = (resume_payload.get("ema_state") or {}).get("shadow") or {}
            live = dict(model.named_parameters())
            with torch.no_grad():
                for name, s in trainer.ema.shadow.items():
                    src = saved_shadow.get(name, live[name].detach())
                    s.copy_(src.to(device=s.device, dtype=s.dtype))
            if is_main:
                print(f"[TrainCLI] EMA shadow restored ({len(saved_shadow)} saved tensors)")
        last_epoch = int(resume_payload.get("epoch", -1))
        trainer.epoch = last_epoch
        trainer.start_epoch = last_epoch + 1

        # ── LR-schedule re-anchoring (world-size / batch invariant) ──────────
        # The cosine schedule is parameterized by total_steps = steps_per_epoch
        # * num_epochs, and steps_per_epoch scales as ~1/world_size because the
        # DistributedSampler shards the train set across ranks (so it also moves
        # with per-GPU batch / grad_accum). The checkpoint's raw global_step is
        # on the ORIGINAL run's step basis; replaying it against a recomputed
        # total_steps (e.g. resuming a 1-GPU run on 4 GPUs) would push the cosine
        # past its end and collapse the LR toward 0. Re-derive the step counter
        # from the COMPLETED-EPOCH FRACTION — invariant to world size / batch —
        # and re-seed the (already correctly-built-for-this-run) scheduler to
        # that point. For a same-world resume this reproduces the saved
        # global_step exactly, so it is a no-op there.
        ckpt_global_step = int(resume_payload.get("global_step", 0))
        trainer.global_step = trainer.start_epoch * trainer.steps_per_epoch
        sched = trainer.scheduler
        reanchored_lrs = [
            base * lam(trainer.global_step)
            for base, lam in zip(sched.base_lrs, sched.lr_lambdas)
        ]
        sched.last_epoch = trainer.global_step
        for group, lr in zip(sched.optimizer.param_groups, reanchored_lrs):
            group["lr"] = lr
        sched._last_lr = list(reanchored_lrs)

        trainer.best_combined_r2 = float(
            resume_payload.get("best_combined_r2", -float("inf"))
        )
        trainer._history = list(resume_payload.get("history", []))
        if is_main:
            lr_str = ", ".join(
                f"{g.get('name', '?')}={lr:.3e}"
                for g, lr in zip(sched.optimizer.param_groups, reanchored_lrs)
            )
            print(
                f"[TrainCLI] resumed from {args.resume} after epoch {last_epoch}; "
                f"next epoch={trainer.start_epoch}"
            )
            print(
                f"[TrainCLI] LR re-anchor: ckpt global_step={ckpt_global_step} -> "
                f"{trainer.global_step} (start_epoch={trainer.start_epoch} x "
                f"steps_per_epoch={trainer.steps_per_epoch}); "
                f"total_steps={trainer.total_steps}; LRs: {lr_str}"
            )

    try:
        if getattr(args, "resume_check", False):
            if is_main:
                print("[TrainCLI] --resume_check: trainer built and resume path "
                      "ran cleanly; exiting before fit().")
        else:
            out = trainer.fit()
            if is_main:
                print(f"[TrainCLI] done. best combined R2: {out['best_combined_r2']:.4f}")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
