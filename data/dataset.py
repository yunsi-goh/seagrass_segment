"""
PyTorch Dataset for seagrass segmentation.

Loads source RGB image-mask pairs from data/rgb/ with online augmentation.
"""
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A

from utils.normalization import normalize_np


# ── Augmentation pipelines ─────────────────────────────────────────────────────

def get_train_augmentation(
    crop_size: int = 512,
    rescale_min: float = 0.75,
    rescale_max: float = 1.25,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    rotate_p: float = 0.5,
    rotate_limit: int = 30,
    elastic_p: float = 0.3,
    grid_p: float = 0.3,
    brightness_p: float = 0.4,
    contrast_p: float = 0.4,
    hue_p: float = 0.2,
    blur_p: float = 0.2,
) -> "A.Compose":
    """Training augmentation pipeline (spatial + colour)."""
    return A.Compose([

        A.RandomScale(
            scale_limit=(rescale_min - 1.0, rescale_max - 1.0),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=crop_size,
            min_width=crop_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=rotate_limit, p=rotate_p,
                 border_mode=cv2.BORDER_REFLECT_101),
        A.ElasticTransform(alpha=120, sigma=6.0, p=elastic_p,
                           border_mode=cv2.BORDER_REFLECT_101),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=grid_p,
                         border_mode=cv2.BORDER_REFLECT_101),

        # ── Colour ───────────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2,
            p=max(brightness_p, contrast_p),
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
            p=hue_p,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=blur_p),
    ])


def get_val_augmentation(crop_size: int = 512) -> "A.Compose":
    """Validation/test preprocessing for fixed-size batching."""
    return A.Compose([
        A.SmallestMaxSize(
            max_size=crop_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=crop_size,
            min_width=crop_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────

class SeagrassDataset(Dataset):
    """
    Loads source RGB image-mask pairs.

    Layout: data_dir/images/*.{jpg,png,tif}  +  data_dir/masks/*.png
    """

    def __init__(
        self,
        data_dir: str | Path,
        normalization: str = "minmax",
        augmentation: Optional["A.Compose"] = None,
        sample_stems: Optional[Iterable[str]] = None,
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.norm_method  = normalization
        self.augmentation = augmentation

        img_dir  = self.data_dir / "images"
        mask_dir = self.data_dir / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Expected images/ and masks/ under {self.data_dir}")

        selected = set(sample_stems) if sample_stems is not None else None
        image_paths: List[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            image_paths.extend(sorted(img_dir.glob(pattern)))
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {img_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for ip in image_paths:
            if selected is not None and ip.stem not in selected:
                continue
            mp = mask_dir / f"{ip.stem}.png"
            if mp.exists():
                self.samples.append((ip, mp))

        if not self.samples:
            raise FileNotFoundError(
                f"No image files matched mask stems in {self.data_dir}"
            )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> np.ndarray:
        """Return H×W×3 float32 in [0, 255]."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read {path}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image with 3 channels: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Return H×W uint8 binary mask {0, 1}."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Cannot read {path}")
        return (mask > 0).astype(np.uint8)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        img  = self._load_image(img_path)     # H×W×C float32
        mask = self._load_mask(mask_path)     # H×W uint8 {0,1}

        if self.augmentation is not None:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            augmented = self.augmentation(image=img_uint8, mask=mask)
            img       = augmented["image"].astype(np.float32)
            mask      = augmented["mask"]

        img = normalize_np(img, method=self.norm_method)

        img_t  = torch.from_numpy(img.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t

    def __repr__(self) -> str:
        return (f"SeagrassDataset(n={len(self)}, norm={self.norm_method}, "
                f"aug={'yes' if self.augmentation else 'no'}, ch=3)")


# ── Train/val/test split ───────────────────────────────────────────────────────

def split_source_dir(
    data_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split source images into train/val/test by filename stem.

    Returns:
        Three lists of source-image stems.
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    stems = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for ip in sorted(img_dir.glob(pattern)):
            if (mask_dir / f"{ip.stem}.png").exists():
                stems.append(ip.stem)
    keys = sorted(set(stems))
    if not keys:
        raise FileNotFoundError(f"No matched image/mask pairs found in {data_dir}")

    rng  = random.Random(seed)
    rng.shuffle(keys)

    n       = len(keys)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    return (
        keys[:n_train],
        keys[n_train:n_train + n_val],
        keys[n_train + n_val:],
    )
