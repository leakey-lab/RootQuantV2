# RootQuant-V2

**Segmentation-free root-trait regression from minirhizotron imagery with a frozen DINOv3 backbone.**

[Paper (arXiv:2609.25567)](https://arxiv.org/abs/2609.25567) · CVPPA Workshop, ECCV 2026

RootQuant-V2 predicts living root **length** (mm) and **surface area** (mm²)
directly from one RGB minirhizotron frame. It needs no segmentation and no
pixel-level labels: it trains on the per-image scalar totals that
minirhizotron archives already hold.

The DINOv3 ViT-L/16 backbone stays frozen. DoRA adapts the attention, a Mona
convolutional adapter adapts the MLP branch, and a density readout sums
per-patch predictions. The headline model trains 11.90 M parameters (3.78 %).

This repository holds the code and trained weights. The dataset is not
distributed. For accuracy numbers, see the paper.

<p align="center">
  <img src="docs/assets/architecture.svg" alt="RootQuant-V2 architecture" width="100%">
  <br><em><a href="docs/assets/architecture.pdf">PDF</a> · <a href="docs/assets/architecture.tex">TikZ source</a></em>
</p>

## Quickstart

```bash
git clone https://github.com/leakey-lab/RootQuantV2.git && cd RootQuantV2
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126  # match your CUDA
pip install -r requirements.txt
```

1. **Backbone.** Download `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
   (~1.2 GB, DINOv3 License) into `RootQuantV2/dinov3/`, or set
   `DINOV3_CHECKPOINT_PATH`. Keep the file name. See
   [`RootQuantV2/dinov3/README.md`](RootQuantV2/dinov3/README.md).
2. **Checkpoint.** `python tools/download_checkpoints.py` fetches the headline
   model. See [Checkpoints](#checkpoints).
3. **Predict.**

   ```bash
   python -m RootQuantV2.predict --images /path/to/frames --out predictions.csv
   ```

   Output columns: `image, pred_length_mm, pred_area_mm2`. The default
   checkpoint is `RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt`;
   use `--checkpoint` for another. `--csv` reads image names from a CSV column.
   `--no-tta` disables D4 test-time augmentation (8× faster, slightly less
   accurate).

Run every command from the repository root. No `pip install -e` is needed.

## Checkpoints

All files are in one [Google Drive folder](https://drive.google.com/drive/folders/1dPkiepx5tDsctXCahvv0E8LYo2zkyaVQ?usp=sharing).
[`checkpoints.json`](checkpoints.json) is the manifest (Drive IDs, byte sizes,
SHA-256, training run names).

| name | size | description |
|---|---|---|
| [`rootquant-v2-weights`](https://drive.google.com/file/d/1Mam81wHZUfyq40PhaGm2B8xmf3EazcRL/view) | 91 MB | **Headline.** 896 px, DoRA r=32 + Mona, mixed soybean + maize. Default download. |
| [`rootquant-v2-768px`](https://drive.google.com/file/d/1rMUbDPnZyAwc9T7NeEpJk-U2LEvU5Fdu/view) | 59 MB | 768 px, DoRA r=16 + Mona. † |
| [`rootquant-v2-dora-only`](https://drive.google.com/file/d/1ysSemsNqGAnV0zC9TvX7e8aNK9qKYYJq/view) | 65 MB | Headline recipe without Mona (ablation). |
| [`rootquant-v2-unfreeze-last2`](https://drive.google.com/file/d/1S2-W256CVmsH0fZD54OcwEqgPCEC7Mlc/view) | 219 MB | Headline plus MLP/LayerNorm/LayerScale of the last 2 ViT blocks unfrozen. † |
| [`rootquant-v2-soybean-only`](https://drive.google.com/file/d/1KyGAS2MCIdrJ2VL9s1-UWAcJHzmYYTKk/view) | 91 MB | Headline architecture, soybean only. † |
| [`rootquant-v2-soybean-to-maize-fcft`](https://drive.google.com/file/d/1jUQOjucCwBRob5YsHDI9LkZZaf9HqT9P/view) | 91 MB | Soybean-only model, readout fine-tuned on maize (FCFT). † |
| [`rootquant-v2-soybean-to-maize-fft`](https://drive.google.com/file/d/16h3iFqcnJOe2GmLdXKo_3mLhTsG1pIOb/view) | 91 MB | Soybean-only model, fully fine-tuned on maize (FFT). † |
| `rootquant-v2-dora-baseline-640px` | 1.2 GB | 640 px DoRA baseline, no Mona. Optional; not hosted yet. Stores the backbone inline. |

† Trained without tile-shuffle. To reproduce, set `TILE_SHUFFLE=0`
(`run_cross_species.sh` and `run_maize_ft.sh` already do).

```bash
python tools/download_checkpoints.py --list                  # show all
python tools/download_checkpoints.py rootquant-v2-768px      # one by name
python tools/download_checkpoints.py --all                   # all except the optional baseline
python tools/download_checkpoints.py <name> --verify-only    # check a manual download
```

Files install to `RootQuantV2/runs/checkpoints/<name>/best.pt` and are
SHA-256 verified. For a manual download, save `<name>.pt` to that path.

Each checkpoint stores its config, target statistics and EMA weights, so
`predict.py` and `infer.py` need no extra flags. Slim checkpoints omit the
frozen backbone and rebuild it from your local DINOv3 file. To see a
checkpoint's config:

```bash
python -c "import torch; print(torch.load('RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt', map_location='cpu', weights_only=False)['cfg'])"
```

## Data

The loader expects this layout:

```
$ROOTQUANT_DATA_DIR/
├── train_data.csv, val_data.csv, test_data.csv
└── images/
```

CSV columns: `ImageName`, `AliveLength(mm)`, `AliveSurfArea(mm2)` (change
`image_col`, `length_col` and `area_col` in `RootQuantV2/config.py` if yours
differ). A row is **present** when length > 0 and area > 0, else **empty**.
Empty rows stay in training. Metrics use present rows only.

```bash
export ROOTQUANT_DATA_DIR=/abs/path/to/dataset
export ROOTQUANT_IMAGES=/abs/path/to/images        # optional
export ROOTQUANT_SOY_DIR=/abs/path/to/soy_split    # cross-species only
export ROOTQUANT_MAIZE_DIR=/abs/path/to/maize_split
```

CLI flags (`--train_csv`, `--val_csv`, `--test_csv`, `--data_path`) override
these variables. The per-species directories use the same three CSVs. The mixed
`test_data.csv` must be the union of the two species test splits.

## Training

```bash
GPUS=0,1,2,3 bash RootQuantV2/run_train.sh                   # headline recipe
ABLATE=no-mona GPUS=0,1,2,3 bash RootQuantV2/run_train.sh    # ablation
PROFILE=dinov3_dora_mona_768_v2 GPUS=0 bash RootQuantV2/run_train.sh
```

`run_train.sh` needs tmux (and rsync for optional `/dev/shm` staging). The
default is profile `dinov3_dora_mona_768_v3`, 20 epochs, per-GPU batch 4, seed
42, fp32 with TF32. Common overrides: `BATCH`, `NUM_EPOCHS`, `LR_*`, `SEED`,
`TARGET_SIZE`, `AMP`, `TILE_SHUFFLE`, `RUN_NAME`, `RESUME`, `INIT_FROM`,
`FINETUNE_MODE`, `STAGE_SHM=0`. Read `run_train.sh` for the full list.

Outputs go to `RootQuantV2/runs/checkpoints/<run_name>/{best,last}.pt` and
TensorBoard logs go to `RootQuantV2/runs/logs/<run_name>/`.

`python -m RootQuantV2.train` has its own defaults (640 px, 30 epochs). To
match the headline run, pass `--profile dinov3_dora_mona_768_v3 --num_epochs 20
--per_gpu_batch 4` and launch 4 processes with `torchrun`.

Profiles and ablations are in `RootQuantV2/config.py`. Ablations: `frozen`,
`no-mona`, `dora-last6`, `cls-only`, `learned-pool`, `unfreeze-last2`,
`unfreeze-last2-no-mona`, `area-weight`, `present-only`, `768-only`.

Reference hardware: 4× A100 80 GB. The 896 px model peaks near 37 GB per GPU
at batch 8.

### Fine-tune on new data

```bash
INIT_FROM=RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt \
FINETUNE_MODE=fc_only ROOTQUANT_DATA_DIR=/path/to/data \
GPUS=0 bash RootQuantV2/run_train.sh
```

`fc_only` trains only the readout. Try it first on small datasets. `full`
trains all adapters: it is stronger in-domain but worse on the source species.
Target statistics are refit on the new split.

## Evaluation

```bash
python -m RootQuantV2.infer \
  --checkpoint RootQuantV2/runs/checkpoints/rootquant-v2-weights/best.pt \
  --csv "$ROOTQUANT_DATA_DIR/test_data.csv" --output_csv preds.csv
```

The script prints R², RMSE and MAE on present rows.

| script | purpose |
|---|---|
| `infer_dual.py` | mixed test set, vanilla and TTA, split by species |
| `aggregate_metrics.py` | combines `infer_dual.py` CSVs into one table |
| `run_infer_all.sh` | runs `infer_dual.py` on every downloaded checkpoint |
| `plot_regression.py` | predicted-vs-measured scatter from `infer.py` output |

## Interpretability

The per-patch `DensityReadout` map is the faithful explanation: its masked sum
is the prediction. Bright density on a frame labelled empty often marks a missed
root, so `rank_empty_density.py` gives a re-annotation queue. Backbone
attention and PCA maps are task-agnostic.

```bash
python -m RootQuantV2.rank_empty_density --out_csv empty_density_ranking.csv
python -m RootQuantV2.visualize_attention --n 4 --out_dir viz_attention
```

See [`RootQuantV2/docs/visualization.md`](RootQuantV2/docs/visualization.md)
for all visualization scripts.

## Citation

```bibtex
@misc{parth2026rootquantv2,
  title         = {RootQuantV2: Adapting a Vision Foundation Model for Root-Trait
                   Regression from Minirhizotron Imagery},
  author        = {Parth, Kinjalk and Varela, Sebastian and Leakey, Andrew D. B.},
  year          = {2026},
  eprint        = {2609.25567},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.25567},
  url           = {https://arxiv.org/abs/2609.25567},
  note          = {Accepted to the CVPPA Workshop at ECCV 2026}
}
```

## License

The code and the RootQuant-V2 checkpoints are dual-licensed:

- [AGPL-3.0-only](LICENSE)
- a [commercial license](LICENSE-COMMERCIAL.md) for use that cannot meet AGPL
  terms. Contact <kinjalk2@illinois.edu>.

The DINOv3 backbone weights are Meta's and fall under the
[DINOv3 License](https://github.com/facebookresearch/dinov3). Neither license
above covers them. The optional 640 px baseline contains them inline. See
[NOTICE](NOTICE).

Built on [DINOv3](https://github.com/facebookresearch/dinov3) (Siméoni et al.,
2025), DoRA (Liu et al., ICML 2024) and
[Mona](https://doi.org/10.1109/CVPR52734.2025.01869) (Yin et al., CVPR 2025).
