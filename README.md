# Seagrass Segmentation Pipeline

Four-model seagrass pipeline combining pixel-level segmentation and image-level classification:
- **U-Net** + EfficientNet-B4 encoder — Jeon et al. (2021)
- **ViT-B/16** encoder + MLP decoder — SeagrassFinder (Elsässer et al. 2024/2025, default)
- **SAM2-UNet** — Hiera backbone + RFB + adapter modules (Huang 2025)
- **ViT-B/16 Classifier** — whole-image binary classification (seagrass present / absent)

## What This Repo Does

- converts COCO-format seagrass labels into `images/` + binary `masks/`
- trains U-Net, ViT, or SAM2-UNet **segmentation** models (pixel-level masks)
- trains a ViT-B/16 **classifier** (image-level: is seagrass present at all?)
- runs tiled inference on large images with overlap-aware Gaussian blending
- evaluates predictions with accuracy, precision, recall, F1, and IoU

## Project Structure

```text
seagrass_segment/
├── configs/
│   ├── config.py              # shared defaults (paths, training, augmentation)
│   ├── config_unet.py         # EfficientNet-B4 U-Net settings
│   ├── config_vit.py          # ViT segmentation overrides
│   ├── config_sam2unet.py     # SAM2UNet-specific settings
│   └── config_vit_cls.py      # ViT classifier hyperparameters
├── data/
│   ├── coco_to_unet.py
│   └── dataset.py
├── models/
│   ├── unet.py
│   ├── vit.py
│   ├── sam2unet.py
│   └── vit_classifier.py      # ViT-B/16 presence/absence classifier
├── scripts/
│   ├── train_unet.py
│   ├── train_vit.py
│   ├── train_sam2unet.py
│   ├── train_vit_cls.py       # classifier training
│   ├── infer_unet.py
│   ├── infer_vit.py
│   ├── infer_sam2unet.py
│   ├── infer_vit_cls.py       # classifier inference
│   └── evaluate.py
├── utils/
│   ├── metrics.py
│   ├── normalization.py
│   └── tiling.py
└── main.py
```

## Setup

```bash
uv sync
```

## Data Preparation

```bash
python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip
```

Optional custom modality output path:

```bash
python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip --modality rgb_nir
```

This creates:

```text
data/
  <modality>/
    images/   *.jpg
    masks/    *.png
```

## Train

Default (`vit` segmentation):

```bash
python main.py train
```

U-Net:

```bash
python main.py train --model unet
```

SAM2-UNet:

```bash
python main.py train --model sam2unet
```

ViT classifier (image-level presence/absence):

```bash
python main.py train --model vit_cls
```

Direct script usage:

```bash
python scripts/train_vit.py
python scripts/train_unet.py
python scripts/train_sam2unet.py
python scripts/train_vit_cls.py
```

## Infer

Default (`vit` segmentation — outputs a pixel mask):

```bash
python main.py infer --checkpoint outputs/vit__bs4__lrdec0.0003__wd0.0005/checkpoints/best.pth --input path/to/image_or_folder
```

U-Net:

```bash
python main.py infer --model unet --checkpoint outputs/unet__efficientnet-b4__bs18__lrdec0.0003/checkpoints/best.pth --input path/to/image_or_folder
```

SAM2-UNet:

```bash
python main.py infer --model sam2unet --checkpoint outputs/sam2unet__small__bs4__lr0.0003/checkpoints/best.pth --input path/to/image_or_folder
```

ViT classifier (returns probability + PRESENT/ABSENT label for a single image):

```bash
python main.py infer --model vit_cls --checkpoint outputs/vit_cls__bs32__lrhead0.0003/checkpoints/best.pth --input path/to/image.jpg
```

Use `--save_prob` to also save float32 probability maps as `.npy` (segmentation models only).

## Evaluate

Evaluate precomputed predictions:

```bash
python scripts/evaluate.py \
  --pred_dir outputs/predictions/ \
  --gt_dir data/rgb/masks/ \
  --output outputs/eval_results.csv
```

Run inference + evaluation:

```bash
python main.py evaluate --model vit \
  --run_inference \
  --checkpoint outputs/vit__bs4__lrdec0.0003/checkpoints/best.pth \
  --image_dir path/to/images \
  --gt_dir path/to/masks

python main.py evaluate --model unet \
  --run_inference \
  --checkpoint outputs/unet__bs18__lrdec0.0003/checkpoints/best.pth \
  --image_dir path/to/images \
  --gt_dir path/to/masks

python main.py evaluate --model sam2unet \
  --run_inference \
  --checkpoint outputs/sam2unet__tiny__bs4__lr0.0003/checkpoints/best.pth \
  --image_dir path/to/images \
  --gt_dir path/to/masks
```

## Configuration

| File | Purpose |
|---|---|
| `configs/config.py` | Shared defaults: paths, split ratios, augmentation, normalization, training params |
| `configs/config_unet.py` | EfficientNet-B4 encoder, crop size, batch size, LR, cosine schedule |
| `configs/config_vit.py` | ViT model name, crop size, batch size, LR, freeze warmup, cosine schedule |
| `configs/config_sam2unet.py` | SAM2 model size, checkpoint, RFB channels, adapter ratio, loss weights |
| `configs/config_vit_cls.py` | Classifier model name, crop size, dropout, batch size, LR, pos_weight |

Segmentation configs (ViT, SAM2UNet) only declare the settings that differ from the shared defaults.
The classifier config is independent (different task, different metrics).

