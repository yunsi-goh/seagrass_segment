"""
Patch reconstruction with Gaussian-weighted blending.
"""
from typing import List, Tuple

import numpy as np


def _gaussian_weight(tile_size: int, sigma: float = 0.35) -> np.ndarray:
    """2-D Gaussian weight map peaking at 1.0 in the tile centre."""
    ax = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return w  # max = 1.0 at centre


def reconstruct_from_tiles(
    tile_preds: List[np.ndarray],
    tile_coords: List[Tuple[int, int]],
    original_shape: Tuple[int, int],
    tile_size: int = 512,
) -> np.ndarray:
    """Reconstruct a full prediction map from overlapping tiles using Gaussian blending."""
    height, width = original_shape
    accum  = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    full_w = _gaussian_weight(tile_size)  # tile_size × tile_size

    for pred, (y, x) in zip(tile_preds, tile_coords):
        pred_h, pred_w = pred.shape[:2]
        end_y = min(y + pred_h, height)
        end_x = min(x + pred_w, width)
        h_sl  = end_y - y
        w_sl  = end_x - x

        # Slice the weight map to match any edge tile that was unpadded.
        w_patch = full_w[:h_sl, :w_sl]

        accum [y:end_y, x:end_x] += pred[:h_sl, :w_sl] * w_patch
        weight[y:end_y, x:end_x] += w_patch

    weight = np.maximum(weight, 1e-6)
    return accum / weight
