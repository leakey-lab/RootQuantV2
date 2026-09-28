"""Single source of truth for RootQuantV2 hyperparameters.

Simplified task: predict (root length_mm, area_mm2) from an RGB image with a
FROZEN DINOv3 ViT-L/16 backbone + DoRA (simple PEFT) + a small MLP regression
head. No segmentation masks exist for this dataset, so there is no dense /
segmentation path — only two scalar regression targets per image.

Profiles:
  - ``dinov3_dora``             : legacy 640px baseline
  - ``dinov3_dora_mona_768``    : 768px, DoRA on attention + Mona MLP adapters,
                                  learned pooling
  - ``dinov3_dora_mona_768_v2`` : + Huber(3) loss, softplus GeM, EMA, D4 TTA,
                                  readout="both"
  - ``dinov3_dora_mona_768_v3`` : headline — 896px, DoRA rank 32, task weights
                                  (1, 1.5), deeper head

Ablations toggle subsystems via ``get_config(profile, ablation)``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset location ────────────────────────────────────────────────────────
# Set ROOTQUANT_DATA_DIR to the directory that holds
#     train_data.csv  val_data.csv  test_data.csv  images/
# Set ROOTQUANT_IMAGES separately if the images live somewhere else (e.g. a
# local tmpfs/SSD copy staged for fast DataLoader reads).
# Both are only defaults: --train_csv / --val_csv / --test_csv / --data_path on
# the CLI still win. There is no fallback location: when neither is set the
# paths below are empty, and commands that read the dataset stop with a message
# (check_data_paths / data.load_split). Commands that need no dataset, such as
# predict.py, are unaffected.
DATASET_DIR = os.environ.get("ROOTQUANT_DATA_DIR", "")
IMAGES_DIR = os.environ.get("ROOTQUANT_IMAGES") or (
    os.path.join(DATASET_DIR, "images") if DATASET_DIR else ""
)


def _dataset_file(name: str) -> str:
    return os.path.join(DATASET_DIR, name) if DATASET_DIR else ""


CONFIG: Dict[str, Any] = {
    # ───── identity / IO ──────────────────────────────────────────────
    "profile": "dinov3_dora",
    "ablation": None,
    "run_name": "v2_dinov3_dora_reg",

    # ───── data ───────────────────────────────────────────────────────
    "data_path": IMAGES_DIR,
    "train_csv": _dataset_file("train_data.csv"),
    "val_csv": _dataset_file("val_data.csv"),
    "test_csv": _dataset_file("test_data.csv"),
    # CSV column names, read by train.py / infer.py / infer_dual.py.
    "image_col": "ImageName",
    "length_col": "AliveLength(mm)",
    "area_col": "AliveSurfArea(mm2)",
    # input_mode ∈ {letterbox_square, letterbox_square_640, native_rect_640x480}
    # Native images are non-square; letterbox modes preserve aspect ratio and
    # pad to a square canvas of target_size for the ViT.
    "input_mode": "letterbox_square_640",
    "target_size": 640,

    # ───── targets ────────────────────────────────────────────────────
    # target_transform ∈ {zscore, none}
    "target_transform": "zscore",
    "target_stats": {},   # filled at runtime: {length_mean/std, area_mean/std}
    # Per-column pre-z-score transform, each ∈ {none, log, sqrt}; None → legacy
    # (none/none, or log/log if target_transform==log_zscore). ("none","sqrt")
    # leaves length raw but linearizes the ~quadratic area target.
    "target_col_transforms": None,

    # ───── backbone (FROZEN) ──────────────────────────────────────────
    "backbone_name": "dinov3_vitl16",
    "backbone_dim": 1024,
    "backbone_patch_size": 16,
    "backbone_num_register_tokens": 4,
    # Optional explicit path to the DINOv3 weights file. Left None so no
    # machine-specific path is saved into checkpoints; the backbone reads env
    # DINOV3_CHECKPOINT_PATH or RootQuantV2/dinov3/ (backbone/dinov3.py).
    "backbone_weights_path": None,
    "backbone_hub_repo": "facebookresearch/dinov3",
    "pretrained": True,

    # ───── PEFT (DoRA) ────────────────────────────────────────────────
    "use_dora": True,
    "dora_rank": 16,
    "dora_target_names": ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
    "dora_blocks": "all",
    # Optional extra capacity: fully unfreeze the last N transformer blocks
    # (their pretrained Linear/Norm weights become trainable, fine-tuned by a
    # dedicated low-LR optimizer group). 0 = backbone fully frozen (default).
    "unfreeze_last_n_blocks": 0,
    "lr_backbone": 1e-5,

    # ───── Mona conv adapter (optional, parallel on MLP branch) ─────────
    "use_mona": False,
    "mona_bottleneck": 64,
    "mona_blocks": "all",
    "mona_dropout": 0.1,
    # Parallel-adapter recipe: scale=1 with zero-init up → no-op forward, nonzero grads.
    "mona_init_scale": 1.0,
    "mona_drop_path": 0.0,
    "lr_mona": 1e-4,
    "mona_weight_decay": 1e-2,

    # ───── regression head ────────────────────────────────────────────
    # readout ∈ {global (pooled MLP), density (extensive per-patch sum), both}.
    # length/area are EXTENSIVE (scale with root amount); every `pool` below is
    # intensive (an average), so the density readout gives a complementary,
    # metric-aligned signal. "both" learns a per-target blend of the two.
    "readout": "global",
    "density_hidden": 256,
    "density_dropout": 0.1,
    # pool ∈ {cls, mean, cls_mean, cls_mean_max, cls_attn_gem}
    "pool": "cls_mean_max",
    # head_hidden: int → single hidden layer (legacy); tuple/list → one width per
    # hidden layer, e.g. (1024, 256) for a deeper, gentler-taper regression head.
    "head_hidden": 512,
    "head_dropout": 0.2,
    "gem_init_p": 3.0,
    # gem_mode ∈ {legacy, softplus}. legacy clamps signed tokens to eps (loses
    # ~half the signal); softplus runs GeM over a sound non-negative map.
    "gem_mode": "legacy",

    # ───── eval / smoothing (opt-in; defaults preserve legacy behavior) ─
    "use_ema": False,          # EMA over trainable params; best.pt = EMA weights
    "ema_decay": 0.9995,
    "eval_tta": False,         # 8-view D4 TTA at FINAL inference (infer.py)
    "val_tta": False,          # 8-view D4 TTA during in-loop validation (slow; off)

    # ───── loss ───────────────────────────────────────────────────────
    # loss_type ∈ {smooth_l1, huber, mse}. smooth_l1 (β=delta) is legacy and caps
    # the gradient on the heavy-tail large roots that dominate R²; huber(delta≈3)
    # is ≈MSE-with-outlier-guard and metric-aligned (R²_raw = 1 − MSE_z). Switch
    # the TYPE — do not just enlarge delta under smooth_l1 (1/delta gradient shift).
    "loss_type": "smooth_l1",
    "huber_delta": 1.0,
    "balance_presence": True,
    # Optional explicit overrides (None → auto-balance from present rate).
    "w_present": None,
    "w_empty": None,
    # Per-target (length, area) loss weight, normalized internally so loss scale
    # is unchanged — only the length↔area balance shifts. (1,1) = legacy equal
    # weighting; raise the second entry to emphasize the harder area target.
    "task_weights": (1.0, 1.0),

    # ───── optimizer ──────────────────────────────────────────────────
    "lr_dora": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "grad_clip_norm": 1.0,

    # ───── precision / memory ─────────────────────────────────────────
    "amp_dtype": "fp32",
    "use_grad_checkpointing": False,
    "use_torch_compile": True,
    "torch_compile_mode": "default",
    "torch_compile_dynamic": False,
    "allow_tf32": True,
    "float32_matmul_precision": "high",

    # ───── batching / accumulation ────────────────────────────────────
    "per_gpu_batch": 16,
    "grad_accum_steps": 1,

    # ───── schedule ───────────────────────────────────────────────────
    "num_epochs": 30,
    "warmup_frac": 0.05,
    "validate_every_n_epochs": 1,

    # ───── DDP / runtime ──────────────────────────────────────────────
    "distributed": False,
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
    "ddp_find_unused_parameters": False,
    "deterministic_training": False,

    # ───── data loader ────────────────────────────────────────────────
    # 4 ranks × 16 train workers = 64 of 96 cores; images staged on /dev/shm so
    # workers are decode-bound (no NFS wait). Tune down if host CPU saturates.
    "train_num_workers": 16,
    "eval_num_workers": 8,
    "prefetch_factor": 4,
    "pin_memory": True,

    # ───── output paths ───────────────────────────────────────────────
    "output_root": os.path.join(BASE_DIR, "runs"),
    "checkpoint_dir": os.path.join(BASE_DIR, "runs", "checkpoints"),
    "logs_dir": os.path.join(BASE_DIR, "runs", "logs"),

    # ───── reproducibility ────────────────────────────────────────────
    "random_seed": 42,

    # ───── augmentation: tile shuffle (train-only, opt-in) ────────────
    # Split the letterboxed image into a grid×grid tile lattice and permute the
    # tiles — a locality-disrupting regularizer. OFF by default. tile_shuffle_p
    # may be a single float (one grid sampled per image, uniform over
    # tile_shuffle_grids) OR a per-grid sequence matching tile_shuffle_grids
    # (each scale fires INDEPENDENTLY at its own probability and may stack, e.g.
    # grids (2,4,8) with p (0.2,0.2,0.2) = 2×2/4×4/8×8 each at p=0.2).
    # NOTE: the content patch_mask is NOT permuted with the tiles, so shuffling
    # breaks mask↔image correspondence for the learned pooler / density readout.
    "use_tile_shuffle": False,
    "tile_shuffle_grids": (2, 4, 8),
    "tile_shuffle_p": (0.2, 0.2, 0.2),
}


# Rigorous 768px accuracy training profile.
MONA_768_CONFIG: Dict[str, Any] = {
    "profile": "dinov3_dora_mona_768",
    "run_name": "v2_dinov3_mona768",
    "input_mode": "letterbox_square",
    "target_size": 768,
    "use_mona": True,
    # Hybrid: DoRA on attention only; Mona restores locality on MLP branch.
    "dora_target_names": ("attn.qkv", "attn.proj"),
    "pool": "cls_attn_gem",
    "head_dropout": 0.35,
    "mona_bottleneck": 64,
    "mona_dropout": 0.15,
    "mona_drop_path": 0.05,
    "mona_init_scale": 1.0,
    "use_grad_checkpointing": True,
    "use_torch_compile": False,
    "per_gpu_batch": 8,
    # Tile-shuffle locality-disrupting regularizer — part of the 768+ recipe
    # (grids 2/4/8, each firing INDEPENDENTLY at 0.3/0.2/0.1; see paper §3.5).
    # OFF in the 640px DoRA baseline (base CONFIG); ON from the Mona-768 hybrid
    # onward, so v2/v3 inherit it. Previously supplied only via run_train.sh env
    # overrides; encoded here so the committed profile reproduces the paper run.
    "use_tile_shuffle": True,
    "tile_shuffle_grids": (2, 4, 8),
    "tile_shuffle_p": (0.3, 0.2, 0.1),
}


# Improved 768px profile: same architecture as ``dinov3_dora_mona_768`` plus the
# reviewed, metric-aligned accuracy wins — a sound GeM on signed tokens, EMA of
# the trainable weights, and eval-time D4 TTA. A/B against the current Mona-768
# checkpoint to attribute the gains.
#
# NOTE: target_transform stays plain "zscore" (NOT log). The eval metric is R² in
# raw mm units, which rewards fitting the large roots; log1p targets minimize
# relative error instead and empirically did MUCH worse on RootQuantV1. The
# log_zscore knob still exists (--target_transform / config) for relative-error or
# presence-detection experiments, but it is off here on purpose.
MONA_768_V2_CONFIG: Dict[str, Any] = {
    "profile": "dinov3_dora_mona_768_v2",
    "run_name": "v2_dinov3_mona768_v2",
    # Native full-precision (fp32) training — project preference, no autocast.
    "amp_dtype": "fp32",
    # R²-aligned loss: huber(δ=3) ≈ MSE across the realistic residual range (R²_raw
    # = 1 − MSE_z on present rows) but clips pathological outliers — vs the legacy
    # smooth_l1(β=1) that caps gradients on the heavy-tail roots R² cares about most.
    "loss_type": "huber",
    "huber_delta": 3.0,
    "gem_mode": "softplus",
    "use_ema": True,
    "ema_decay": 0.9995,
    "eval_tta": True,
    # Blend the proven global head with the extensive sum-density head. "both"
    # can't regress below global-only (blend can down-weight density); run
    # readout="density" vs "global" to measure the head's standalone power.
    "readout": "both",
}


# Headline profile. Builds on v2 with the levers aimed at the area-R² ceiling
# (area is the laggard target; length converges higher and earlier):
#   #3 resolution 768 → 896      → thin roots that drive surface area survive at
#      higher res; batch 8, grad-checkpointing off (see below).
#   #4 DoRA rank 16 → 32         → more adaptation capacity (safe lever). The
#      higher-ceiling "unfreeze last blocks" lever is wired but OFF here; enable
#      via the `unfreeze-last2` ablation to A/B it.
#   #1 task_weights = (1.0, 1.5)  → up-weight the harder area target. With sqrt
#      OFF (option C) area is still the laggard, so a 1.5× area emphasis is on by
#      default here. Normalized internally, so loss scale / LR are unchanged —
#      only the length↔area balance shifts. Trade-off: combined R² averages the
#      two targets, so pushing area too hard can steal from length; 1.5 is a
#      moderate start, dial up/down via this knob.
#
# NOTE: #2 (sqrt-transform area) is deliberately NOT used here — option C. The
# density readout (readout="both", inherited from v2) SUMS per-patch densities,
# which assumes an additive/extensive target (total area = Σ patch area). sqrt is
# non-additive — √(a+b) ≠ √a+√b — and would also standardize a raw-unit sum with
# √-space mean/std, breaking the density head exactly as log does. Targets stay
# raw z-score so the density branch remains valid. The per-column sqrt path is
# still available in the code for a global-only profile if revisited later.
MONA_768_V3_CONFIG: Dict[str, Any] = {
    "profile": "dinov3_dora_mona_768_v3",
    "run_name": "v2_dinov3_mona768_v3",
    "target_size": 896,
    "per_gpu_batch": 8,
    "dora_rank": 32,
    "task_weights": (1.0, 1.5),
    "unfreeze_last_n_blocks": 0,
    "lr_backbone": 1e-5,
    # Speed: grad-checkpointing OFF (recompute-free backward) + torch.compile ON.
    # Overridable from run_train.sh (GRAD_CKPT env). Note: ckpt-off raises
    # activation memory substantially at 896px — drop batch if it OOMs.
    "use_grad_checkpointing": False,
    "use_torch_compile": True,
    # Deeper regression head: gentler taper from the 3072-d pooled feature
    # (3072→1024→256→2 instead of the legacy single 3072→512→2 squeeze), with
    # the inherited head_dropout=0.35 between each hidden block as a guard
    # against overfitting the empty-dominated / heavy-tail targets.
    "head_hidden": (1024, 256),
}


_LAST6_BLOCKS = (18, 19, 20, 21, 22, 23)

# Minimal ablations for quick comparisons.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    # Disables PEFT adapters only; pooler + regression head remain trainable.
    "frozen": {"use_dora": False, "use_mona": False},
    # Mona OFF, DoRA untouched — isolates Mona's marginal value over DoRA.
    "no-mona": {"use_mona": False},
    "dora-last6": {
        "dora_blocks": _LAST6_BLOCKS,
        "mona_blocks": _LAST6_BLOCKS,
    },
    "cls-only": {"pool": "cls"},
    "learned-pool": {"pool": "cls_attn_gem"},
    # High-ceiling #4: fully unfreeze the last 2 ViT blocks at a tiny LR.
    "unfreeze-last2": {"unfreeze_last_n_blocks": 2},
    # Mona-OFF control for the unfreeze-last2 matched pair (last 2 blocks
    # unfrozen + Mona removed, DoRA untouched) — the paper's Mona-value control
    # against the existing Mona-ON run v2_dinov3_mona768_v3_unfreeze-last2. Pure
    # science delta (Mona off) only; inherits the v3 profile's grad-ckpt OFF +
    # compile ON so the compute path is byte-identical to that Mona-ON run.
    "unfreeze-last2-no-mona": {"unfreeze_last_n_blocks": 2, "use_mona": False},
    # #1 follow-up: emphasize the harder area target (use only if area still trails).
    "area-weight": {"task_weights": (1.0, 1.25)},
    "present-only": {"w_empty": 0.0, "balance_presence": False},
    "768-only": {
        "input_mode": "letterbox_square",
        "target_size": 768,
        "use_grad_checkpointing": True,
        "use_torch_compile": False,
        "per_gpu_batch": 8,
    },
}


PROFILES: Dict[str, Dict[str, Any]] = {
    "dinov3_dora": deepcopy(CONFIG),
    "dinov3_dora_mona_768": {**deepcopy(CONFIG), **deepcopy(MONA_768_CONFIG)},
    "dinov3_dora_mona_768_v2": {
        **deepcopy(CONFIG),
        **deepcopy(MONA_768_CONFIG),
        **deepcopy(MONA_768_V2_CONFIG),
    },
    "dinov3_dora_mona_768_v3": {
        **deepcopy(CONFIG),
        **deepcopy(MONA_768_CONFIG),
        **deepcopy(MONA_768_V2_CONFIG),
        **deepcopy(MONA_768_V3_CONFIG),
    },
}


def list_ablations() -> Iterable[str]:
    return tuple(ABLATIONS.keys())


def list_profiles() -> Iterable[str]:
    return tuple(PROFILES.keys())


def get_config(profile: str = "dinov3_dora", ablation: Optional[str] = None) -> Dict[str, Any]:
    """Return a deep-copied config for the given profile + optional ablation."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(PROFILES)}")
    cfg = deepcopy(PROFILES[profile])

    if ablation is not None:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation '{ablation}'. Available: {list(ABLATIONS)}")
        cfg["ablation"] = ablation
        cfg["run_name"] = f"{cfg['run_name']}_{ablation}"
        cfg.update(deepcopy(ABLATIONS[ablation]))

    # Sanity: target_size must be divisible by patch size.
    patch = int(cfg.get("backbone_patch_size", 16))
    ts = int(cfg.get("target_size", 640))
    if ts % patch != 0:
        raise ValueError(f"target_size={ts} must be divisible by patch_size={patch}.")

    return cfg


def check_data_paths(
    cfg: Dict[str, Any],
    keys: Iterable[str] = ("train_csv", "val_csv", "test_csv", "data_path"),
) -> None:
    """Exit with a clear message if a dataset path a command needs is unset or missing."""
    bad = [k for k in keys if not cfg.get(k) or not os.path.exists(cfg[k])]
    if bad:
        lines = "\n".join(f"  {k}: {cfg.get(k) or '<unset>'}" for k in bad)
        raise SystemExit(
            f"Dataset path(s) not found:\n{lines}\n"
            "Set ROOTQUANT_DATA_DIR to the directory holding train_data.csv, "
            "val_data.csv, test_data.csv and images/ (ROOTQUANT_IMAGES if the images "
            "live elsewhere), or pass --train_csv / --val_csv / --test_csv / "
            "--data_path. See 'Data layout' in the README."
        )


__all__ = [
    "BASE_DIR",
    "DATASET_DIR",
    "IMAGES_DIR",
    "CONFIG",
    "MONA_768_CONFIG",
    "MONA_768_V2_CONFIG",
    "MONA_768_V3_CONFIG",
    "ABLATIONS",
    "PROFILES",
    "get_config",
    "check_data_paths",
    "list_profiles",
    "list_ablations",
]
