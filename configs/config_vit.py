"""
ViT-B/16 hyperparameters for seagrass segmentation.
SeagrassFinder: Elsässer et al. (2024/2025, Ecological Informatics).
"""

# ── ViT encoder ────────────────────────────────────────────────────────────────
# ImageNet-21k pretrained (stronger); swap to "vit_base_patch16_224" for vanilla.
VIT_MODEL_NAME  = "vit_base_patch16_224.augreg_in21k_ft_in1k"
VIT_PRETRAINED  = True

# Decoder MLP hidden width
DECODE_CHANNELS = 256

# ── Freeze warmup ──────────────────────────────────────────────────────────────
# epochs to train decoder-only before unfreezing the ViT encoder; 0 = disable
VIT_FREEZE_EPOCHS = 10  # was 5; longer warmup reduces encoder destabilisation

# ── Image / tile size ─────────────────────────────────────────────────────────
# must be a multiple of 16 (ViT patch size)
CROP_SIZE  = 512
TILE_STRIDE = CROP_SIZE // 2   # 50% overlap

# ── Learning rates ─────────────────────────────────────────────────────────────
# encoder LR much lower to avoid catastrophic forgetting
LR_ENCODER   = 1e-5
LR_DECODER   = 3e-4
WEIGHT_DECAY = 5e-4  # was 1e-4; stronger regularisation to reduce overfitting

# ── Scheduler ──────────────────────────────────────────────────────────
USE_COSINE_LR = True  # smoother than StepLR

# ── Batch size ────────────────────────────────────────────────────────────────

BATCH_SIZE = 4
