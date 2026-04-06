"""
Image normalisation helpers: min-max, z-score, and ImageNet.
"""
import numpy as np

# ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── numpy helpers ─────────────────────────────────────────────────────────────

def minmax_norm_np(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-band min-max normalisation to [0, 1]."""
    out = img.astype(np.float32)
    for c in range(img.shape[-1]):
        band = out[..., c]
        lo, hi = band.min(), band.max()
        out[..., c] = np.clip((band - lo) / (hi - lo + eps), 0.0, 1.0)
    return out


def zscore_norm_np(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-band z-score normalisation."""
    out = img.astype(np.float32)
    for c in range(img.shape[-1]):
        band = out[..., c]
        mu  = band.mean()
        sig = band.std()
        out[..., c] = (band - mu) / (sig + eps)
    return out


def imagenet_norm_np(img: np.ndarray) -> np.ndarray:
    """Scale [0, 255] to [0, 1] then apply ImageNet mean/std."""
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
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

