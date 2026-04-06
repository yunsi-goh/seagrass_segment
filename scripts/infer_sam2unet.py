"""
Inference script for SAM2-UNet seagrass segmentation.

Handles any image size via patch-based tiling with Gaussian blending.

Usage:
    python scripts/infer_sam2unet.py \\
        --checkpoint outputs/sam2unet__tiny__bs4__lr0.0003/checkpoints/best.pth \\
        --input path/to/image.jpg \\
        --output outputs/predictions/

    # directory of images
    python scripts/infer_sam2unet.py \\
        --checkpoint outputs/sam2unet__tiny__bs4__lr0.0003/checkpoints/best.pth \\
        --input path/to/image_dir/ \\
        --output outputs/predictions/

    --save_prob  : also save float32 probability map as .npy
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs import config as cfg
from configs import config_sam2unet as scfg
from models.sam2unet import build_sam2unet
from utils.normalization import normalize_np
from utils.tiling import reconstruct_from_tiles

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ── Patch-based inference ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_image_sam2unet(
    model: torch.nn.Module,
    img: np.ndarray,
    tile_size: int = 352,
    stride: int = 176,
    normalization: str = "imagenet",
    threshold: float = 0.5,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tiled patch inference on a full image. Returns (prob_map, binary_map)."""
    model.eval()
    H, W = img.shape[:2]
    tile_preds:  List[np.ndarray]      = []
    tile_coords: List[Tuple[int, int]] = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tile = img[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

            tile_norm = normalize_np(tile, method=normalization)
            tensor = torch.from_numpy(
                tile_norm.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            logit = model(tensor)          # eval → 1×1×H×W  (finest prediction)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

            tile_preds.append(prob[:th, :tw])
            tile_coords.append((y, x))

    prob_map   = reconstruct_from_tiles(tile_preds, tile_coords, (H, W), tile_size)
    binary_map = (prob_map >= threshold).astype(np.uint8) * 255
    return prob_map, binary_map


# ── Load model ────────────────────────────────────────────────────────────────

def load_sam2unet_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model_size    = ckpt.get("model_size",    scfg.SAM2_MODEL_SIZE)
    rfb_channels  = ckpt.get("rfb_channels",  scfg.RFB_CHANNELS)
    adapter_ratio = ckpt.get("adapter_ratio", scfg.ADAPTER_RATIO)
    img_size      = ckpt.get("crop_size",     scfg.CROP_SIZE)

    model = build_sam2unet(
        model_size=model_size,
        pretrained_path=None,         # weights come from checkpoint
        rfb_channels=rfb_channels,
        adapter_ratio=adapter_ratio,
        freeze_backbone=False,        # no freezing during inference
        img_size=img_size,
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    print(
        f"SAM2UNet loaded from {checkpoint_path}  "
        f"(backbone=Hiera-{model_size}, rfb_ch={rfb_channels})"
    )
    return model


# ── Image loader ──────────────────────────────────────────────────────────────

def load_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run SAM2-UNet seagrass inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input",      type=str, required=True,
                        help="Image file or directory")
    parser.add_argument("--output",     type=str,
                        default=str(cfg.PRED_DIR / "sam2unet"))
    parser.add_argument("--tile_size",  type=int, default=scfg.CROP_SIZE)
    parser.add_argument("--stride",     type=int, default=scfg.TILE_STRIDE)
    parser.add_argument("--threshold",  type=float, default=cfg.EVAL_THRESHOLD)
    parser.add_argument("--norm",       type=str, default=cfg.NORMALIZATION)
    parser.add_argument("--save_prob",  action="store_true",
                        help="Also save float32 probability map as .npy")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_sam2unet_model(args.checkpoint)

    input_path = Path(args.input)
    if input_path.is_dir():
        img_paths = sorted(
            list(input_path.glob("*.jpg"))  +
            list(input_path.glob("*.png"))  +
            list(input_path.glob("*.tif"))  +
            list(input_path.glob("*.tiff"))
        )
    else:
        img_paths = [input_path]

    print(f"\nRunning SAM2UNet inference on {len(img_paths)} image(s) …")

    for img_path in tqdm(img_paths, desc="Predicting"):
        img = load_image(img_path)
        prob_map, binary_map = predict_image_sam2unet(
            model, img,
            tile_size=args.tile_size, stride=args.stride,
            normalization=args.norm,  threshold=args.threshold,
            device=DEVICE,
        )
        cv2.imwrite(str(output_dir / f"{img_path.stem}_pred.png"), binary_map)
        if args.save_prob:
            np.save(str(output_dir / f"{img_path.stem}_prob.npy"), prob_map)

    print(f"\nPredictions saved to {output_dir}")


if __name__ == "__main__":
    main()
