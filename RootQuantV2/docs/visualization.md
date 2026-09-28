# RootQuantV2 — Saliency / "where is the model looking" visualization

Interpretability tooling for the fine-tuned DINOv3 root regressor. Every script
except `gen_letterbox_augment.py` is driven by a trained checkpoint (the
checkpoint carries its own `cfg` + `target_stats`, exactly like `infer.py`):

| script | produces | purpose |
|---|---|---|
| `visualize_attention.py` | per-image map boards + per-head grids | inspect *where* the model attends / measures roots on any image |
| `rank_empty_density.py` | a CSV ranking all gt-empty tubes by summed head-density | auto-surface **mislabeled** empties (a re-annotation queue) |
| `montage_density.py` | a top-N mislabeled montage (PNG) | one slide of the worst gt=0 tubes the model says have roots |
| `montage_all_pairs.py` | full contact sheet (multi-page PDF) | every mislabeled empty as `[input | density overlay]` pairs, for re-annotation review |
| `viz_mona_pair.py` | one board per target (PNG) | head density of two checkpoints (default: Mona on vs. off) on the same images, shared scale + difference panel |
| `viz_root_panel.py` | a figure grid (PNG + PDF) | input, feature PCA and head density side by side; `--compare_frozen` adds the same PCA on un-adapted DINOv3 tokens |
| `viz_tta_d4.py` | 8-view montage (PNG + PDF) | the 8 D4 views test-time augmentation averages, with per-view predictions |
| `viz_tta_d4_density.py` | 8-view montage (PNG + PDF) | the head density map of each D4 view |
| `gen_letterbox_augment.py` | preprocessing figure (PNG + PDF) | letterboxing, patch mask and the training augmentations on one frame (no checkpoint) |

> **TL;DR of the science.** The backbone weights are frozen, but DoRA and Mona
> sit in its forward path, so the attention / PCA / cosine maps are computed on
> *adapted* tokens. They are still task-agnostic: they segment *any* structured
> texture, including soil, and are **not** a faithful picture of what the
> regressor measures. The **head** is what learned the task. Trust the head maps —
> especially the `DensityReadout` per-patch density, whose masked sum *is* the
> prediction. To see the same PCA on un-adapted DINOv3 tokens, use
> `viz_root_panel.py --compare_frozen`.

---

## 1. Quick start

All commands run from the **repository root** so `RootQuantV2` imports as a
package. Pick an idle GPU first (`nvidia-smi`); everything also runs on CPU,
slowly. Modes that sample images from a labelled split read
`$ROOTQUANT_DATA_DIR/test_data.csv` and `$ROOTQUANT_DATA_DIR/images` (or
`--csv` / `--data_path`); modes that take `--images` / `--image` need no dataset.

```bash
cd /path/to/RootQuantV2

# (1) map board + per-head attention for sampled present-root images
CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.visualize_attention \
    --n 3 --out_dir RootQuantV2/viz_attention

# explicit images
CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.visualize_attention \
    --images /path/a.jpg /path/b.jpg --out_dir RootQuantV2/viz_attention

# (2) rank every gt-empty tube by summed head-density  →  CSV
CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.rank_empty_density \
    --out_csv RootQuantV2/viz_analysis/empty_density_ranking.csv

# (3) slide montage of the top-N likely-mislabeled empties
CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.montage_density \
    --ranking_csv RootQuantV2/viz_analysis/empty_density_ranking.csv \
    --out RootQuantV2/viz_analysis/montage_mislabeled.png --n 15

# (4) full contact sheet of every flagged empty (multi-page PDF)
CUDA_VISIBLE_DEVICES=0 python -m RootQuantV2.montage_all_pairs \
    --ranking_csv RootQuantV2/viz_analysis/empty_density_ranking.csv \
    --out RootQuantV2/viz_analysis/montage_mislabeled_all.pdf
```

Default checkpoint: `RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt`
(profile `dinov3_dora_mona_768_v3`, 896px, `pool=cls_attn_gem`, `readout=both`).
Override with `--checkpoint`. `rank_empty_density.py`, `montage_density.py` and
`montage_all_pairs.py` need a `readout=both` checkpoint (every released 768/896 px
model); `viz_*` density views need `readout` `density` or `both`. Output paths
(`--out`, `--out_dir`, `--out_csv`) are relative to the working directory.

---

## 2. The maps (`visualize_attention.py`)

Per image it writes `<stem>__maps.png` (a board of up to 9 panels) and
`<stem>__heads.png` (16 per-head CLS-attention maps). Inputs are letterboxed to
the model's square canvas; overlays share that geometry.

### Backbone family — *self-supervised style, task-agnostic*
- **CLS self-attention** (per head + mean), last block — the iconic DINO map.
- **Attention rollout** (Abnar & Zuidema 2020) across all 24 blocks — usually the
  cleanest single object mask.
- **PCA(3) of patch tokens → RGB** + **PC1** — the canonical DINOv2/v3
  "segmentation" view (foreground/background pops out).
- **Patch cosine similarity** to a query patch (auto-set to the patch the head
  attends to most, marked `+`).

### Head family — *task-faithful, what we actually fine-tuned* ⭐
- **AttentionPool weights** — softmax of the learnable query (`pool=cls_attn_gem`)
  over patches: where the global head pools from.
- **Per-patch density · length** and **· area** — `DensityReadout` predicts a
  non-negative density per patch; the prediction is the *masked sum* of it. This
  is a prediction-coupled "root mass" map — the closest thing to a learned root
  segmentation. **Lead with these in any figure.**

### How attention is recovered (the one tricky bit)
DINOv3 attention uses **RoPE + `scaled_dot_product_attention`**, which never
materializes the attention matrix. `visualize_attention.py` monkeypatches
`dinov3.layers.attention.SelfAttention.compute_attention` with an explicit-softmax
path (mathematically identical at eval — no dropout, no mask), recovering the
probabilities while leaving the forward output (and predictions) unchanged. It is
installed **after** model construction, because the `dinov3` package only lands on
`sys.path` once the backbone is built via `torch.hub`. Register tokens (4) + CLS
are stripped before reshaping the patch grid (896px → 56×56).

---

## 3. Label-error detection on empty tubes

Most rows in a minirhizotron series are labeled empty
(`AliveLength=0, AliveSurfArea=0`). Two questions:

1. **Worst false positives** — empties the model scores *high*. Rendering the
   density map for those frames separates two cases: the map traces a clear root
   structure (a **label error** — the model found a root the annotation missed),
   or it is diffuse (a genuine model failure on condensation, scratches or soil
   mottling).
2. **Genuine empties** — empties the model scores ~0. The head density maps are
   **dark / flat** (model agrees: no root), while the *backbone* maps still fire
   on soil texture and illumination gradients (task-agnostic structure).

This gives a free **label-error detector**: rank all empties by summed head
density.

```
HEAD density on a gt=0 image
   bright, structured  →  a real root is there      →  LABEL ERROR (re-annotate)
   dark, flat          →  genuinely empty            →  correct
```

`rank_empty_density.py` runs the model over every gt-empty row and records the
`DensityReadout` total (`gain · Σ_patch softplus(mlp(patch))`, in raw mm/mm²)
alongside the full blended prediction. Output CSV is sorted by
`density_length_mm` descending — the top is the re-annotation queue.

`montage_density.py` turns this into one slide of **only the mislabeled tubes** —
gt-empty rows the model says contain a root. Each cell overlays the head density
map on the input, so you see the root the model found on a tube labeled empty.
`montage_all_pairs.py` renders every such tube, paginated, for a full review.

**Filtering column — documented choice: `density_length_mm`.** Of the CSV's four
numeric columns we filter/rank on the density-head *length* signal:
- *density over pred* — `readout="both"` blends the density head with a global
  CLS-pooled head and the learned blend **down-weights** density, so `pred_*` is a
  damped view. `density_*` is more sensitive, is the **exact quantity the montage
  draws** (selection criterion == the picture), and is spatially grounded (a sum
  of localized per-patch root mass, not a global guess).
- *length over area* — root *presence* is a 1-D/length phenomenon (a root is a
  curve; length = traced extent). Area conflates extent with thickness and is more
  easily inflated by diffuse bright blobs (condensation, scratches, soil mottling),
  a noisier presence detector. Length is also the better-calibrated of the two.

Default rule: `density_length_mm ≥ 5mm` ("a root is present"), top-N shown. All
cells share one absolute color scale with opacity ∝ density (per-image min-max is
avoided — it would stretch weak maps to look bright). Both knobs are overridable
(`--rank_col`, `--threshold`, `--map`) to A/B against area or the blended prediction.

---

## 4. Other figures

```bash
# head density of two checkpoints on the same images (default: headline vs. dora-only)
python -m RootQuantV2.viz_mona_pair --n_present 4 --map length \
    --out RootQuantV2/viz_mona_pair

# input | feature PCA | length / area density grid; add --compare_frozen for
# the same PCA on un-adapted DINOv3 tokens
python -m RootQuantV2.viz_root_panel --images /path/a.jpg /path/b.jpg \
    --csv "$ROOTQUANT_DATA_DIR/test_data.csv" --compare_frozen \
    --out RootQuantV2/viz_analysis/root_panel.png

# the 8 D4 views test-time augmentation averages, with per-view predictions
python -m RootQuantV2.viz_tta_d4 --image frame.jpg \
    --out RootQuantV2/viz_analysis/tta_d4_montage.png

# the head density map of each D4 view
python -m RootQuantV2.viz_tta_d4_density --image frame.jpg --target length \
    --out RootQuantV2/viz_analysis/tta_d4_density.png

# preprocessing / augmentation figure (no checkpoint needed)
python -m RootQuantV2.gen_letterbox_augment --image frame.jpg \
    --out RootQuantV2/viz_analysis/letterbox_augment.png
```

`viz_root_panel.py` reads the ground-truth chips from `--csv` (column names from
`config.py`); without it they show 0.

---

## 5. Output locations

Defaults, relative to the repository root (all under `RootQuantV2/`, which
`.gitignore` excludes):

| path | produced by | contents |
|---|---|---|
| `RootQuantV2/viz_attention/<stem>__maps.png`, `<stem>__heads.png` | `visualize_attention.py` | map board and per-head attention grid per image |
| `RootQuantV2/viz_analysis/empty_density_ranking.csv` | `rank_empty_density.py` | all empties ranked by summed density |
| `RootQuantV2/viz_analysis/montage_mislabeled.png` | `montage_density.py` | montage of top-N mislabeled empties (gt=0, model finds root) |
| `RootQuantV2/viz_analysis/montage_mislabeled_all.pdf` | `montage_all_pairs.py` | full contact sheet — every flagged empty as `[input \| overlay]` pairs, paginated |
| `RootQuantV2/viz_mona_pair/mona_pair_<target>.png` | `viz_mona_pair.py` | paired density boards |
| `RootQuantV2/viz_analysis/root_panel.{png,pdf}` | `viz_root_panel.py` | root panel figure |
| `RootQuantV2/viz_analysis/tta_d4_montage.{png,pdf}` | `viz_tta_d4.py` | D4 view montage |
| `RootQuantV2/viz_analysis/tta_d4_density.{png,pdf}` | `viz_tta_d4_density.py` | D4 density montage |
| `RootQuantV2/viz_analysis/letterbox_augment.{png,pdf}` | `gen_letterbox_augment.py` | preprocessing figure |

## 6. Notes / caveats
- Rankings are checkpoint-specific — they agree at the top but differ on borderline
  images. `rank_empty_density.py` always uses the checkpoint you pass, so its CSV is
  self-consistent; do not mix rankings from two checkpoints.
- Precision: **fp32**. These scripts enable TF32 matmul for inference speed (as
  `infer.py` does); they never use bf16/AMP.
- `visualize_attention.py`'s explicit-softmax attention is heavier than SDPA
  (materializes the N×N matrix); fine for a handful of images. Use `--no-rollout`
  to skip the all-block fold. `rank_empty_density.py`/`montage_density.py` do
  **not** install the hook (density needs only `forward_features`), so they keep
  the fast SDPA path.
