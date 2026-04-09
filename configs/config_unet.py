"""
EfficientNet-B4 U-Net configuration (v2).
Jeon et al. (2021) Ecological Informatics 66, 101430.
"""
from configs.config import *  # noqa: F401, F403

# ── Encoder ─────────────────────────────────────────────────────────────
ENCODER_NAME    = "efficientnet-b4"  # was resnet34; stronger ImageNet features
ENCODER_WEIGHTS = "imagenet"

# ── Image / tile size ────────────────────────────────────────────────────────
CROP_SIZE   = 512
TILE_SIZE   = CROP_SIZE
TILE_STRIDE = TILE_SIZE // 2

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 18
LR_ENCODER = 1e-4
LR_DECODER = 3e-4
USE_COSINE_LR = True  # cosine anneal instead of step decay