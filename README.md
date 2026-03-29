# Seagrass U-Net Pipeline

RGB U-Net pipeline for seagrass segmentation, adapted from:

> Jeon et al. (2021). *Semantic segmentation of seagrass habitat from drone imagery based on deep learning: A comparative study.* Ecological Informatics 66, 101430.

## What This Repo Does

- preprocesses a COCO-style seagrass dataset into RGB images and binary masks
- trains from source RGB images with standard spatial augmentation
- fine-tunes a U-Net with an ImageNet-initialized ResNet34 encoder
- runs full-image inference on high-resolution imagery
- evaluates predictions with accuracy, precision, recall, F1, and IoU

## Project Structure

```text
seagrass_unet/
├── configs/
│   └── config.py
├── data/
│   ├── dataset.py
│   └── coco_to_unet.py
├── models/
│   └── unet.py
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
├── utils/
│   ├── metrics.py
│   ├── normalization.py
│   └── tiling.py
└── outputs/
```

## Setup

```bash
uv sync
```

## Data Preparation

Prepare the COCO export into the expected layout:

```bash
# Default — outputs to data/rgb/
python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip

# Custom modality — outputs to data/<modality>/
python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip --modality rgb_nir
```

This creates:

```text
data/
  <modality>/        (default: rgb)
    images/   *.jpg
    masks/    *.png
```

Masks are binary PNGs with the same filename stem as the source image.

## Training

```bash
python scripts/train.py
```

Training reads directly from `data/rgb/images` and `data/rgb/masks`. Source
images are split by filename stem and augmented into fixed-size inputs during
training.

Outputs are written to:

- `outputs/checkpoints/rgb/best.pth`
- `outputs/checkpoints/rgb/epoch_XXXX.pth`
- `outputs/logs/train_rgb.jsonl`

## Inference

```bash
python scripts/infer.py \
    --checkpoint outputs/checkpoints/rgb/best.pth \
    --input path/to/image_or_folder \
    --output outputs/predictions/
```

Use `--save_prob` to also save float32 probability maps as `.npy`.

## Evaluation

Evaluate saved predictions:

```bash
python scripts/evaluate.py \
    --pred_dir outputs/predictions/ \
    --gt_dir data/rgb/masks/ \
    --output outputs/eval_results.csv
```

Or run inference and evaluation in one step:

```bash
python scripts/evaluate.py \
    --run_inference \
    --checkpoint outputs/checkpoints/rgb/best.pth \
    --image_dir path/to/images \
    --gt_dir path/to/masks
```

## Configuration

Main settings live in `configs/config.py`.

- `IN_CHANNELS = 3`
- `NORMALIZATION = "imagenet"`
- `ENCODER_NAME = "resnet34"`
- `ENCODER_WEIGHTS = "imagenet"`
- `CROP_SIZE = 512`
- `AUG_RESCALE_MIN = 0.75`
- `AUG_RESCALE_MAX = 1.25`
- `EPOCHS = 100`
- `LR_ENCODER = 1e-4`
- `LR_DECODER = 1e-3`
- `BATCH_SIZE = 18`
