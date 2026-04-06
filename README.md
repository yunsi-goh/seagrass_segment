# Seagrass Segmentation Pipeline

Three-model seagrass segmentation pipeline:
- **U-Net** + ResNet34 encoder — Jeon et al. (2021) baseline
- **ViT-B/16** encoder + MLP decoder — SeagrassFinder (Elsässer et al. 2024/2025, default)
- **SAM2-UNet** — Hiera backbone + RFB + adapter modules (Huang 2025)

## What This Repo Does

- converts COCO-format seagrass labels into `images/` + binary `masks/`
- trains U-Net, ViT, or SAM2-UNet segmentation models
- runs tiled inference on large images with overlap-aware Gaussian blending
- evaluates predictions with accuracy, precision, recall, F1, and IoU

## Project Structure

```text
seagrass_segment/
├── configs/
│   ├── config.py              # shared defaults (paths, training, augmentation)
│   ├── config_unet.py         # re-exports config.py (U-Net uses shared defaults)
│   ├── config_vit.py          # ViT-specific overrides
│   └── config_sam2unet.py     # SAM2UNet-specific settings
├── data/
│   ├── coco_to_unet.py
│   └── dataset.py
├── models/
│   ├── unet.py
│   ├── vit.py
│   └── sam2unet.py
├── scripts/
│   ├── train_unet.py
│   ├── train_vit.py
│   ├── train_sam2unet.py
│   ├── infer_unet.py
│   ├── infer_vit.py
│   ├── infer_sam2unet.py
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

Default (`vit`):

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

Direct script usage:

```bash
python scripts/train_vit.py
python scripts/train_unet.py
python scripts/train_sam2unet.py
```

## Infer

Default (`vit`):

```bash
python main.py infer --checkpoint outputs/vit__bs4__lrdec0.0003/checkpoints/best.pth --input path/to/image_or_folder
```

U-Net:

```bash
python main.py infer --model unet --checkpoint outputs/unet__bs18__lrdec0.0003/checkpoints/best.pth --input path/to/image_or_folder
```

SAM2-UNet:

```bash
python main.py infer --model sam2unet --checkpoint outputs/sam2unet__tiny__bs4__lr0.0003/checkpoints/best.pth --input path/to/image_or_folder
```

Use `--save_prob` to also save float32 probability maps as `.npy`.

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
| `configs/config.py` | Shared defaults: paths, split ratios, augmentation, normalization, U-Net training params |
| `configs/config_vit.py` | ViT model name, crop size, batch size, LR, freeze warmup |
| `configs/config_sam2unet.py` | SAM2 model size, checkpoint, RFB channels, adapter ratio, loss weights |

ViT and SAM2-UNet configs only declare the settings that differ from the shared defaults.

