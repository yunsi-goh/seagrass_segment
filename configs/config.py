"""
Central configuration for the seagrass U-Net pipeline.
Based on: Jeon et al. (2021) Ecological Informatics 66, 101430
Default pipeline: ImageNet-initialized ResNet34 U-Net with source-image
training and online augmentation.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"
PRED_DIR    = OUTPUT_DIR / "predictions"
LOG_DIR     = OUTPUT_DIR / "logs"

# ── Dataset ────────────────────────────────────────────────────────────────────
# Expected layout:
#   data/
#     rgb/
#       images/      *.jpg / *.png / *.tif  (RGB source images)
#       masks/       *.png                  (binary, same stem)

CROP_SIZE = 512       # training and validation crop size
TILE_SIZE = CROP_SIZE # inference crop size
TILE_STRIDE = TILE_SIZE // 2  # 50 % overlap → Gaussian blending removes tiling artefacts

# Train / Val / Test split (at the *source image* level, not tile level)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
RANDOM_SEED = 42

# ── Input channels ─────────────────────────────────────────────────────────────
IN_CHANNELS = 3

# ── Normalization ──────────────────────────────────────────────────────────────
# "imagenet" : ImageNet mean/std — required when using a pretrained encoder
# "minmax"   : per-image min-max (paper default, for training from scratch)
# "zscore"   : per-image z-score
# "none"     : scale to [0,1] only
NORMALIZATION = "imagenet"

# ── Model ──────────────────────────────────────────────────────────────────────
OUT_CHANNELS = 1
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE          = 18
GRAD_ACCUM_STEPS    = 1
EPOCHS              = 100
EARLY_STOP_PATIENCE = 20
LR_ENCODER    = 1e-4
LR_DECODER    = 3e-4
LR_DECAY_STEP = 20
LR_DECAY_RATE = 0.5
SAVE_EVERY    = 50
NUM_WORKERS   = 4
PIN_MEMORY    = True

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_THRESHOLD = 0.5

# ── Augmentation ───────────────────────────────────────────────────────────────
AUG_RESCALE_MIN  = 0.75
AUG_RESCALE_MAX  = 1.25
AUG_HFLIP_P      = 0.5
AUG_VFLIP_P      = 0.5
AUG_ROTATE_P     = 0.5
AUG_ROTATE_LIMIT = 30
AUG_ELASTIC_P    = 0.3
AUG_GRID_P       = 0.3
AUG_BRIGHTNESS_P = 0.4
AUG_CONTRAST_P   = 0.4
AUG_HUE_P        = 0.2
AUG_BLUR_P       = 0.2
