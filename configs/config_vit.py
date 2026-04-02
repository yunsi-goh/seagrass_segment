"""
ViT-specific hyperparameters for seagrass segmentation.

This file extends configs/config.py (base pipeline settings remain unchanged).
Only ViT-specific knobs live here.

SeagrassFinder reference model: ViT-B/16 pretrained on ImageNet-21k,
fine-tuned on underwater eelgrass images.

Tile/crop size must be a multiple of the ViT patch size (16).
Recommended: 512 (32×32 patches) — same as the base UNet config.
"""

# ── ViT encoder ────────────────────────────────────────────────────────────────
# "vit_base_patch16_224" accepts any multiple-of-16 img_size via timm.
# For ImageNet-21k pretraining (stronger, same as SeagrassFinder intent):
#   VIT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"
# For vanilla ImageNet-1k (lighter, faster):
#   VIT_MODEL_NAME = "vit_base_patch16_224"
VIT_MODEL_NAME  = "vit_base_patch16_224.augreg_in21k_ft_in1k"
VIT_PRETRAINED  = True

# Decoder MLP hidden width
DECODE_CHANNELS = 256

# ── Freeze warmup ──────────────────────────────────────────────────────────────
# Number of epochs to train ONLY the decoder while the ViT encoder is frozen.
# Helps with early instability when fine-tuning large ViT on small datasets.
# Set to 0 to disable (train everything from epoch 1).
VIT_FREEZE_EPOCHS = 5

# ── Image / tile size ─────────────────────────────────────────────────────────
# Must be a multiple of 16 (ViT patch size).
CROP_SIZE  = 512
TILE_STRIDE = CROP_SIZE // 2   # 50 % overlap → Gaussian blending

# ── Learning rates ─────────────────────────────────────────────────────────────
# ViT encoder needs a much smaller LR than the decoder to avoid catastrophic
# forgetting of ImageNet features.
LR_ENCODER   = 1e-5   # ~10× smaller than base UNet encoder LR
LR_DECODER   = 3e-4   # same as base UNet decoder LR
WEIGHT_DECAY = 1e-4   # AdamW regularisation

# ── Batch size ────────────────────────────────────────────────────────────────
# ViT-B/16 at 512×512 uses ~3× more GPU memory than ResNet34 UNet.
# Reduce batch size if you hit OOM; gradient accumulation in config.py applies.
# GPU     | VRAM  | recommended BATCH_SIZE
# RTX3090 | 24GB  | 8
# RTX3080 | 10GB  | 4
# A100    | 40GB  | 16
BATCH_SIZE = 4
