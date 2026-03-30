"""
Patch reconstruction helpers for full-image inference.
"""
from typing import List, Tuple

import numpy as np


def _gaussian_weight(tile_size: int, sigma: float = 0.35) -> np.ndarray:
    """
    2-D Gaussian weight map of shape (tile_size, tile_size).

    Peaks at 1.0 in the centre and decays toward the tile edges so that
    edge pixels (which receive less context from the model) contribute less
    to the blended prediction than centre pixels.

    Args:
        tile_size: Side length of the square tile.
        sigma:     Standard deviation in normalised coordinates [-1, 1].
                   Smaller values give a tighter peak (more centre-biased).
    """
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
    """
    Reconstruct a full prediction map from (possibly overlapping) tile predictions
    using Gaussian-weighted blending.

    Each tile's contribution is weighted by a 2-D Gaussian centred on the tile,
    so model outputs near the tile boundaries (where context is limited) are
    down-weighted relative to outputs near the tile centre.  When tiles overlap
    (stride < tile_size) the blending is smooth and tiling artefacts disappear.

    Args:
        tile_preds:     List of H_i × W_i float32 probability arrays.
        tile_coords:    List of (y, x) top-left corners matching tile_preds.
        original_shape: (height, width) of the full image.
        tile_size:      Side length used when extracting tiles (needed to build
                        the weight map).

    Returns:
        Gaussian-blended probability map of shape original_shape, float32.
    """
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
