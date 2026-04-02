# Seagrass Segmentation Pipeline

Dual-model seagrass segmentation pipeline with:
- U-Net + ResNet34 encoder (baseline)
- ViT-B/16 encoder + lightweight segmentation decoder (default)

## What This Repo Does

- converts COCO-format seagrass labels into `images/` + binary `masks/`
- trains either U-Net or ViT segmentation models
- runs tiled inference on large images with overlap-aware blending
- evaluates predictions with accuracy, precision, recall, F1, and IoU

## Project Structure

```text
seagrass_segment/
|-- configs/
|   |-- config.py
|   |-- config_vit.py
|-- data/
|   |-- coco_to_unet.py
|   |-- dataset.py
|-- models/
|   |-- unet.py
|   |-- vit.py
|-- scripts/
|   |-- train_unet.py
|   |-- train_vit.py
|   |-- infer_unet.py
|   |-- infer_vit.py
|   |-- evaluate.py
|-- utils/
|   |-- metrics.py
|   |-- normalization.py
|   |-- tiling.py
|-- main.py
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

Direct script usage:

```bash
python scripts/train_vit.py
python scripts/train_unet.py
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

Use `--save_prob` to also save float32 probability maps as `.npy`.

## Evaluate

Evaluate precomputed predictions:

```bash
python scripts/evaluate.py \
  --pred_dir outputs/predictions/ \
  --gt_dir data/rgb/masks/ \
  --output outputs/eval_results.csv
```

Run inference + evaluation (ViT):

```bash
python main.py evaluate --model vit \
  --run_inference \
  --checkpoint outputs/vit__bs4__lrdec0.0003/checkpoints/best.pth \
  --image_dir path/to/images \
  --gt_dir path/to/masks
```

Run inference + evaluation (U-Net):

```bash
python main.py evaluate --model unet \
  --run_inference \
  --checkpoint outputs/unet__bs18__lrdec0.0003/checkpoints/best.pth \
  --image_dir path/to/images \
  --gt_dir path/to/masks
```

## Configuration

Base settings: `configs/config.py`
- paths, data split, augmentation, normalization
- U-Net model defaults
- shared training and evaluation parameters

ViT overrides: `configs/config_vit.py`
- ViT model name and pretrained setting
- ViT crop/tile size constraints
- ViT-specific batch size and optimization hyperparameters
- optional encoder freeze warmup

