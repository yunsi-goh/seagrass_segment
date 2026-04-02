"""
Normalization utilities.

Paper (Jeon et al. 2021) Eqs. (1)-(2):
  Z-score  : z = (x - μ) / σ          — per-band, per-image
  Min-Max  : X = (x - x_min) / (x_max - x_min)  — per-band, per-image

Key finding: Min-Max is strong for RGB optical images when training from scratch.

Extension: ImageNet normalization is required when using a pretrained encoder
  (e.g. ResNet34 via segmentation-models-pytorch).
"""
import numpy as np

# ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── numpy helpers ─────────────────────────────────────────────────────────────

def minmax_norm_np(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-band Min-Max normalisation to [0, 1].
    img: H×W×C float32 array.
    """
    out = img.astype(np.float32)
    for c in range(img.shape[-1]):
        band = out[..., c]
        lo, hi = band.min(), band.max()
        out[..., c] = np.clip((band - lo) / (hi - lo + eps), 0.0, 1.0)
    return out


def zscore_norm_np(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-band Z-score normalisation.
    img: H×W×C float32 array.
    """
    out = img.astype(np.float32)
    for c in range(img.shape[-1]):
        band = out[..., c]
        mu  = band.mean()
        sig = band.std()
        out[..., c] = (band - mu) / (sig + eps)
    return out


def imagenet_norm_np(img: np.ndarray) -> np.ndarray:
    """
    Scale [0, 255] → [0, 1] then apply ImageNet mean/std normalisation.
    img: H×W×3 float32 array (uint8 values expected).
    Required when using a pretrained torchvision encoder.
    """
    out = img.astype(np.float32) / 255.0
    return (out - IMAGENET_MEAN) / IMAGENET_STD


def normalize_np(img: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Dispatch to the chosen normalisation method."""
    if method == "imagenet":
        return imagenet_norm_np(img)
    elif method == "minmax":
        return minmax_norm_np(img)
    elif method == "zscore":
        return zscore_norm_np(img)
    elif method == "none":
        return img.astype(np.float32) / 255.0   # scale to [0,1] as baseline
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

