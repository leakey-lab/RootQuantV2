# DINOv3 backbone weights go here

RootQuant-V2 adapts a **frozen DINOv3 ViT-L/16** pretrained on LVD-1689M. Those
weights belong to Meta and are **not redistributed** in this repository — you
download them yourself under Meta's DINOv3 License.

## 1. Get the checkpoint

Request access at <https://github.com/facebookresearch/dinov3> (Meta sends the
download links after you accept the license) and download the ViT-L/16
LVD-1689M backbone. The file is ~1.2 GB and is named:

```
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The HuggingFace model `facebook/dinov3-vitl16-pretrain-lvd1689m` ships the same
weights only in the `transformers` format, which this code cannot load. Use the
`.pth` file from Meta.

## 2. Put it in this directory

```
RootQuantV2/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Every entry point (training, `predict.py`, `infer.py`, the visualization
scripts) finds it there automatically. To keep it elsewhere, export the
variable instead:

```bash
export DINOV3_CHECKPOINT_PATH=/abs/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The backbone loader (`backbone/dinov3.py`) looks in this order:

1. `DINOV3_CHECKPOINT_PATH`, if set;
2. the path stored in the checkpoint's config, if that file exists on this
   machine;
3. `RootQuantV2/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.

It prints the path it used. If none of them exists it stops with an error that
says where to put the file; it never tries to download the weights.

## 3. Keep the file name

**Keep the `-8aa4cbdd.pth` suffix.** The loader passes the path to the DINOv3
torch.hub entry point, which reads the 8-character hash from the file name. A
renamed file (e.g. `dinov3.pth`) is rejected with an error.

## 4. Network and disk notes

The weights are always read from the local file, but the architecture is built
by `torch.hub.load('facebookresearch/dinov3', ..., source='github')`, which
needs the `facebookresearch/dinov3` **code** in the torch.hub cache
(`~/.cache/torch/hub/facebookresearch_dinov3_main`). The first run downloads it
from GitHub; for an offline machine, run once on a networked one and copy that
directory across. The DINOv3 code imports `torchmetrics` and `termcolor`, which
are in `requirements.txt`.

On first use torch.hub also copies the weights file into
`~/.cache/torch/hub/checkpoints/` (another 1.2 GB) and loads that copy on later
runs.
