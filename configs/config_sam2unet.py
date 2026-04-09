"""
SAM2UNet hyperparameters for seagrass segmentation.

Huang S (2025) Advancing Seagrass Semantic Segmentation with SAM2 Models.
Journal of High School Science 9(2):235–47.
"""

# ── Backbone ───────────────────────────────────────────────────────────────────
# "tiny" | "small" | "base_plus" | "large"

SAM2_MODEL_SIZE = "small"  # was "tiny"; better pretrained features

# SAM2 checkpoint (.pt). None uses random weights; download from Meta/HuggingFace.
# https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
SAM2_CHECKPOINT = None   # e.g. "checkpoints/sam2.1_hiera_tiny.pt"

# RFB output channels
RFB_CHANNELS = 64

# adapter bottleneck ratio (dim/4)
ADAPTER_RATIO = 0.25

# ── Image / tile size ─────────────────────────────────────────────────────────
# matches hiera_*_224 pretrained size; no pos_embed interpolation needed
CROP_SIZE   = 224
TILE_STRIDE = CROP_SIZE // 2   # 50% overlap

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4
LR           = 3e-4
WEIGHT_DECAY = 1e-3   # was 5e-4; reduce heavy overfitting (train IoU 0.67 vs val 0.37)
BCE_WEIGHT   = 1.0     # λ1 in loss
IOU_WEIGHT   = 1.0     # λ2 in loss

# pos_weight for class imbalance (~7:1 bg:fg)
POS_WEIGHT   = 7.0

# freeze backbone for N warmup epochs
FREEZE_EPOCHS = 10  # was 5; more stable decoder warmup before backbone fine-tuning

# ── Scheduler ────────────────────────────────────────────────────────────────
USE_COSINE_LR = True
