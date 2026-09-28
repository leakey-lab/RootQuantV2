"""
Augmentation transforms for RootQuantV2.

Pipeline convention (letterbox modes)
--------------------------------------
  1. Spatial geometry on PIL (letterbox to square)
  2. Photometric on PIL (train only)
  3. PIL -> Tensor via torchvision.transforms.ToTensor
  4. Normalize (ImageNet stats)
  5. Build per-patch validity mask (excludes letterbox padding)
  6. Tile shuffle on Tensor (optional, train only)
  7. Geometric augmentation on Tensor + mask (RandomD4, train only)

Returns ``(image_tensor, patch_mask)`` for letterbox modes so learned pooling
can ignore padded regions. Native-rect mode returns image only.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# ImageNet mean and std — match DINOv3 pretraining.
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def content_patch_mask_2d(
    new_w: int,
    new_h: int,
    left: int,
    top: int,
    target: int,
    patch_size: int,
) -> torch.Tensor:
    """Bool mask (grid, grid) for patch tokens overlapping letterbox content.

    Vectorized: a patch overlaps the content box iff its [start, start+ps)
    interval overlaps the content interval on BOTH axes, so the 2-D mask is the
    outer-AND of the two 1-D axis-overlap vectors. This replaces a per-image
    grid×grid Python double loop (2304 iterations at 768px) that ran in every
    DataLoader __getitem__ and bottlenecked CPU once the GPU is bf16-fast.
    """
    grid = target // patch_size
    ps = patch_size
    starts = torch.arange(grid) * ps          # patch pixel offsets
    ends = starts + ps
    col_ov = (ends > left) & (starts < left + new_w)   # x-axis (width)
    row_ov = (ends > top) & (starts < top + new_h)     # y-axis (height)
    return row_ov.unsqueeze(1) & col_ov.unsqueeze(0)   # (grid[i=y], grid[j=x])


def _apply_d4_pair(
    img: torch.Tensor,
    mask2d: torch.Tensor,
    choice: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same D4 element to image [C,H,W] and mask [H,W]."""
    if choice >= 4:
        img = TF.hflip(img)
        mask2d = torch.flip(mask2d, dims=[1])
        choice -= 4
    if choice == 1:
        img = torch.rot90(img, k=1, dims=[1, 2])
        mask2d = torch.rot90(mask2d, k=1, dims=[0, 1])
    elif choice == 2:
        img = torch.rot90(img, k=2, dims=[1, 2])
        mask2d = torch.rot90(mask2d, k=2, dims=[0, 1])
    elif choice == 3:
        img = torch.rot90(img, k=3, dims=[1, 2])
        mask2d = torch.rot90(mask2d, k=3, dims=[0, 1])
    return img, mask2d


class LetterboxToSquare:
    """Zero-pad a PIL image to a square, then resize to (target, target)."""

    def __init__(self, target: int = 640, fill: int = 0) -> None:
        self.target = target
        self.fill = fill

    def letterbox(self, img: Image.Image) -> Tuple[Image.Image, int, int, int, int]:
        """Return letterboxed PIL image and content geometry (new_w, new_h, left, top)."""
        w, h = img.size
        scale = self.target / max(w, h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.target - new_w
        pad_h = self.target - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = TF.pad(img, padding=[left, top, right, bottom], fill=self.fill)
        return img, new_w, new_h, left, top

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.letterbox(img)[0]


class LetterboxTransform:
    """Letterbox pipeline returning ``(image, patch_mask)``."""

    def __init__(
        self,
        target: int,
        patch_size: int = 16,
        train: bool = True,
        photometric: Optional[Photometric] = None,
        tile_shuffle: Optional["TileShuffle"] = None,
    ) -> None:
        self.letterbox = LetterboxToSquare(target=target, fill=0)
        self.target = target
        self.patch_size = patch_size
        self.train = train
        self.photometric = photometric
        self.tile_shuffle = tile_shuffle
        self._to_tensor = T.ToTensor()
        # inplace: the tensor from ToTensor is freshly allocated each call, so
        # normalizing in place is safe and avoids an extra copy.
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=True)

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img, new_w, new_h, left, top = self.letterbox.letterbox(img)
        if self.train and self.photometric is not None:
            img = self.photometric(img)
        tensor = self._to_tensor(img)
        tensor = self._normalize(tensor)
        mask2d = content_patch_mask_2d(
            new_w, new_h, left, top, self.target, self.patch_size,
        )
        if self.tile_shuffle is not None:
            tensor = self.tile_shuffle(tensor)
        if self.train:
            choice = random.randint(0, 7)
            tensor, mask2d = _apply_d4_pair(tensor, mask2d, choice)
        patch_mask = mask2d.reshape(-1).float()
        return tensor, patch_mask


class RandomD4:
    """Uniformly sample one of the 8 dihedral-group elements and apply it."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 7)
        if choice >= 4:
            img = TF.hflip(img)
            choice -= 4
        if choice == 1:
            img = torch.rot90(img, k=1, dims=[1, 2])
        elif choice == 2:
            img = torch.rot90(img, k=2, dims=[1, 2])
        elif choice == 3:
            img = torch.rot90(img, k=3, dims=[1, 2])
        return img


class RandomD2:
    """Uniformly sample one of the 4 D2-group elements and apply it."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 3)
        if choice == 1:
            img = TF.hflip(img)
        elif choice == 2:
            img = TF.vflip(img)
        elif choice == 3:
            img = torch.rot90(img, k=2, dims=[1, 2])
        return img


class TileShuffle:
    """Split the image into a ``grid x grid`` lattice of tiles and randomly
    permute the tile order.

    ``p`` controls how the grid scales are sampled:
      - **float** (legacy): with probability ``p`` apply ONE shuffle at a grid
        drawn uniformly from ``grids``.
      - **per-grid sequence** (``len(p) == len(grids)``): each grid scale fires
        INDEPENDENTLY with its own probability; multiple scales can stack and are
        composed coarse→fine in ``grids`` order. E.g. ``grids=(2, 4, 8)`` with
        ``p=(0.2, 0.2, 0.2)`` shuffles 2×2, 4×4, and 8×8 tiles each at p=0.2.
    """

    def __init__(
        self,
        grids: Sequence[int] = (2, 4),
        p: Union[float, Sequence[float]] = 0.2,
    ) -> None:
        if not grids:
            raise ValueError("grids must be a non-empty sequence")
        grids_t = tuple(int(g) for g in grids)
        if any(g < 2 for g in grids_t):
            raise ValueError(f"every grid size must be >= 2, got {grids_t}")
        self.grids = grids_t

        if isinstance(p, (int, float)):
            p_scalar = float(p)
            if not 0.0 <= p_scalar <= 1.0:
                raise ValueError(f"p must be in [0, 1], got {p}")
            self.p = p_scalar              # scalar (legacy) mode
            self.per_grid_p: Optional[Tuple[float, ...]] = None
        else:
            probs = tuple(float(x) for x in p)
            if len(probs) != len(grids_t):
                raise ValueError(
                    f"per-grid p must match grids length: got {len(probs)} "
                    f"prob(s) for {len(grids_t)} grid(s)"
                )
            if any(not 0.0 <= x <= 1.0 for x in probs):
                raise ValueError(f"every per-grid p must be in [0, 1], got {probs}")
            self.p = None
            self.per_grid_p = probs

    def _shuffle_at(self, img: torch.Tensor, grid: int) -> torch.Tensor:
        c, h, w = img.shape
        if h % grid != 0 or w % grid != 0:
            raise ValueError(
                f"image size ({h}x{w}) is not divisible by tile grid {grid}"
            )
        th, tw = h // grid, w // grid
        tiles = (
            img.reshape(c, grid, th, grid, tw)
               .permute(1, 3, 0, 2, 4)
               .contiguous()
               .reshape(grid * grid, c, th, tw)
        )
        perm = torch.randperm(grid * grid, device=img.device)
        tiles = tiles[perm]
        return (
            tiles.reshape(grid, grid, c, th, tw)
                 .permute(2, 0, 3, 1, 4)
                 .contiguous()
                 .reshape(c, h, w)
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 3:
            raise ValueError(
                f"TileShuffle expects Tensor[C, H, W]; got shape {tuple(img.shape)}"
            )
        if self.per_grid_p is not None:
            for grid, pg in zip(self.grids, self.per_grid_p):
                if random.random() < pg:
                    img = self._shuffle_at(img, grid)
            return img
        # scalar (legacy) mode: gate once, then one grid uniformly.
        if random.random() >= self.p:
            return img
        return self._shuffle_at(img, random.choice(self.grids))


class Photometric:
    """ColorJitter followed by optional Gaussian blur."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.02,
        blur_p: float = 0.1,
        blur_sigma: tuple = (0.1, 1.0),
    ) -> None:
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self._blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=3, sigma=blur_sigma)],
            p=blur_p,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self._jitter(img)
        img = self._blur(img)
        return img


def _resolve_target_size(profile: str, cfg: Optional[dict]) -> int:
    """Resolve letterbox canvas side length from cfg or legacy profile name."""
    if cfg is not None and "target_size" in cfg:
        return int(cfg["target_size"])
    if profile == "letterbox_square_640":
        return 640
    if profile == "letterbox_square_768":
        return 768
    if profile == "letterbox_square":
        return int(cfg.get("target_size", 640)) if cfg else 640
    raise ValueError(f"Cannot resolve target_size for profile '{profile}'.")


def get_transforms(
    profile: str,
    train: bool,
    cfg: Optional[dict] = None,
) -> Union[LetterboxTransform, T.Compose]:
    """Build the image-preprocessing pipeline for a given input mode."""
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    use_tile_shuffle = bool(train and cfg is not None and cfg.get("use_tile_shuffle"))
    tile_shuffle: Optional[TileShuffle] = None
    if use_tile_shuffle:
        # tile_shuffle_p may be a float (one grid sampled per image) or a per-grid
        # sequence (each grid fires independently); TileShuffle validates both.
        tile_shuffle = TileShuffle(
            grids=cfg.get("tile_shuffle_grids", (2, 4)),
            p=cfg.get("tile_shuffle_p", 0.2),
        )

    letterbox_profiles = (
        "letterbox_square",
        "letterbox_square_640",
        "letterbox_square_768",
    )
    if profile in letterbox_profiles:
        target = _resolve_target_size(profile, cfg)
        patch_size = int(cfg.get("backbone_patch_size", 16)) if cfg else 16
        photometric = Photometric() if train else None
        return LetterboxTransform(
            target=target,
            patch_size=patch_size,
            train=train,
            photometric=photometric,
            tile_shuffle=tile_shuffle if train else None,
        )

    if profile == "native_rect_640x480":
        resize = T.Resize((480, 640))
        if train:
            steps = [
                resize,
                Photometric(),
                T.ToTensor(),
                normalize,
            ]
            if tile_shuffle is not None:
                steps.append(tile_shuffle)
            steps.append(RandomD2())
            return T.Compose(steps)
        else:
            return T.Compose([
                resize,
                T.ToTensor(),
                normalize,
            ])

    raise ValueError(
        f"Unknown transform profile '{profile}'. "
        f"Expected one of {letterbox_profiles} or 'native_rect_640x480'."
    )
