"""
ViT-B/16 binary image classifier hyperparameters.

Task: predict whether seagrass is present anywhere in the image (whole-image label).
Label is derived automatically from the mask: 1 if mask has any foreground pixel, else 0.
"""

# ── Encoder ───────────────────────────────────────────────────────────────────
# Same pretrained model as the segmentation ViT for consistency
VIT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"
VIT_PRETRAINED = True

# ── Image size ────────────────────────────────────────────────────────────────
# 224 matches the pretrained ViT-B/16 exactly — no interpolation needed
CROP_SIZE = 224

# ── Head regularisation ───────────────────────────────────────────────────────
DROPOUT = 0.3

# ── Freeze warmup ─────────────────────────────────────────────────────────────
# epochs to train head-only before unfreezing encoder
VIT_FREEZE_EPOCHS = 5

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32   # larger batch OK — single forward pass, no tiling
LR_ENCODER   = 1e-5
LR_HEAD      = 3e-4
WEIGHT_DECAY = 5e-4

# ── Class imbalance (87% present, 13% absent) ─────────────────────────────────
# pos_weight = n_negative / n_positive ≈ 0.15; down-weight majority class
POS_WEIGHT = 0.15

# ── Scheduler ────────────────────────────────────────────────────────────────
USE_COSINE_LR = True
