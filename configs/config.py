"""
Shared pipeline configuration (paths, training defaults, augmentation).
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"
PRED_DIR    = OUTPUT_DIR / "predictions"
LOG_DIR     = OUTPUT_DIR / "logs"

# ── Dataset split ──────────────────────────────────────────────────────────────
# train/val/test split at source-image level
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
RANDOM_SEED = 42

# ── Network I/O ────────────────────────────────────────────────────────────────
IN_CHANNELS  = 3
OUT_CHANNELS = 1

# ── Normalization ──────────────────────────────────────────────────────────────
# "imagenet" | "minmax" | "zscore" | "none"
NORMALIZATION = "imagenet"

# ── Training (shared defaults) ─────────────────────────────────────────────────
EPOCHS              = 100
EARLY_STOP_PATIENCE = 20
GRAD_ACCUM_STEPS    = 1
SAVE_EVERY          = 50
NUM_WORKERS         = 4
PIN_MEMORY          = True
LR_DECAY_STEP       = 20
LR_DECAY_RATE       = 0.5

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
