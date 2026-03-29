"""
Patch reconstruction helpers for full-image inference.
"""
from typing import List, Tuple

import numpy as np


def reconstruct_from_tiles(
    tile_preds: List[np.ndarray],
    tile_coords: List[Tuple[int, int]],
    original_shape: Tuple[int, int],
    tile_size: int = 512,
) -> np.ndarray:
    """
    Reconstruct a full prediction map from crop predictions.

    Overlapping crops are averaged when stride is smaller than tile size.
    """
    del tile_size  # retained in the signature for CLI/config compatibility

    height, width = original_shape
    accum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for pred, (y, x) in zip(tile_preds, tile_coords):
        pred_h, pred_w = pred.shape[:2]
        end_y = min(y + pred_h, height)
        end_x = min(x + pred_w, width)
        accum[y:end_y, x:end_x] += pred[: end_y - y, : end_x - x]
        count[y:end_y, x:end_x] += 1.0

    count = np.maximum(count, 1.0)
    return accum / count
