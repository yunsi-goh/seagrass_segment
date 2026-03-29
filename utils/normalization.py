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
import torch

# ImageNet statistics — required by pretrained torchvision encoders
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD_T  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# ── numpy helpers (used during dataset and inference preprocessing) ────────────

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


# ── torch helpers (used inside Dataset.__getitem__) ─────────────────────────────

def minmax_norm_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-channel Min-Max on a C×H×W float tensor.
    """
    out = x.clone().float()
    for c in range(x.shape[0]):
        ch = out[c]
        lo = ch.min()
        hi = ch.max()
        out[c] = torch.clamp((ch - lo) / (hi - lo + eps), 0.0, 1.0)
    return out


def zscore_norm_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel Z-score on a C×H×W float tensor."""
    out = x.clone().float()
    for c in range(x.shape[0]):
        ch = out[c]
        mu  = ch.mean()
        sig = ch.std()
        out[c] = (ch - mu) / (sig + eps)
    return out


def imagenet_norm_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Scale [0, 255] → [0, 1] then apply ImageNet mean/std on a C×H×W float tensor.
    """
    out = x.float() / 255.0
    mean = IMAGENET_MEAN_T.view(3, 1, 1).to(out.device)
    std  = IMAGENET_STD_T .view(3, 1, 1).to(out.device)
    return (out - mean) / std


def normalize_tensor(x: torch.Tensor, method: str = "minmax") -> torch.Tensor:
    if method == "imagenet":
        return imagenet_norm_tensor(x)
    elif method == "minmax":
        return minmax_norm_tensor(x)
    elif method == "zscore":
        return zscore_norm_tensor(x)
    elif method == "none":
        return x.float() / 255.0
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
