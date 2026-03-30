"""
Inference script for RGB models.

Handles any-size input images by:
  1. Running patch-based inference on fixed-size windows
  2. Applying the trained U-Net to each window
  3. Reconstructing the full prediction map
  4. Saving as a binary PNG mask

Usage:
    # Single image
    python scripts/infer.py \
        --checkpoint outputs/checkpoints/rgb/best.pth \
        --input path/to/image.jpg \
        --output outputs/predictions/

    # Directory of images
    python scripts/infer.py \
        --checkpoint outputs/checkpoints/rgb/best.pth \
        --input path/to/image_dir/ \
        --output outputs/predictions/

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
from models.unet import build_unet
from utils.normalization import normalize_np
from utils.tiling import reconstruct_from_tiles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")


# ── Patch-based inference for one large image ──────────────────────────────────

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    img: np.ndarray,              # H×W×3 float32 (already in [0,255])
    tile_size: int = 512,
    stride: int = 256,  # 50 % overlap by default; use Gaussian blending in reconstruct
    normalization: str = cfg.NORMALIZATION,
    threshold: float = 0.5,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a full image with patch-based prediction.

    Returns:
        prob_map:   H×W float32  probability map (0–1)
        binary_map: H×W uint8   binary mask      (0 or 255)
    """
    model.eval()
    H, W = img.shape[:2]

    tile_preds: List[np.ndarray] = []
    tile_coords: List[Tuple[int, int]] = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Crop tile (may be smaller near edges — pad to tile_size)
            tile = img[y:y+tile_size, x:x+tile_size]
            th, tw = tile.shape[:2]

            # Pad to full tile_size if needed
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                if tile.ndim == 2:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)))
                else:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

            # Normalise
            tile_norm = normalize_np(tile, method=normalization)  # H×W×C float32

            # To tensor: C×H×W
            tensor = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(device)

            logit = model(tensor)                       # 1×1×H×W
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # H×W

            # Unpad
            prob = prob[:th, :tw]

            tile_preds .append(prob)
            tile_coords.append((y, x))

    prob_map   = reconstruct_from_tiles(tile_preds, tile_coords, (H, W), tile_size)
    binary_map = (prob_map >= threshold).astype(np.uint8) * 255

    return prob_map, binary_map


# ── Load model from checkpoint ─────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    in_channels = ckpt.get("in_channels", cfg.IN_CHANNELS)
    out_channels = ckpt.get("out_channels", cfg.OUT_CHANNELS)
    encoder_name = ckpt.get("encoder_name", cfg.ENCODER_NAME)

    model = build_unet(
        in_channels=in_channels,
        out_channels=out_channels,
        encoder_name=encoder_name,
        # Checkpoints already contain encoder weights, so don't require a
        # fresh ImageNet download just to rebuild the module skeleton.
        encoder_weights=None,
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    print(
        f"Model loaded from {checkpoint_path}  "
        f"(encoder={encoder_name}, in_channels={in_channels})"
    )
    return model


# ── Load image ────────────────────────────────────────────────────────────────

def load_image(img_path: Path) -> np.ndarray:
    """Load an RGB image as H×W×3 float32 in [0,255]."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # H×W×3
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input",      type=str, required=True,
                        help="Image file or directory of images")
    parser.add_argument("--output",     type=str,
                        default=str(cfg.PRED_DIR))
    parser.add_argument("--tile_size",  type=int,   default=cfg.TILE_SIZE)
    parser.add_argument("--stride",     type=int,   default=cfg.TILE_STRIDE,
                        help="Stride for patch-based inference (default=tile_size//2 for 50%% overlap)")
    parser.add_argument("--threshold",  type=float, default=cfg.EVAL_THRESHOLD)
    parser.add_argument("--norm",       type=str,   default=cfg.NORMALIZATION)
    parser.add_argument("--save_prob",  action="store_true",
                        help="Also save the float32 probability map as .npy")
    args = parser.parse_args()

    output_dir    = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)

    # Collect inputs
    input_path = Path(args.input)
    if input_path.is_dir():
        img_paths = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.tif")) +
            list(input_path.glob("*.tiff"))
        )
    else:
        img_paths = [input_path]

    print(f"\nRunning inference on {len(img_paths)} image(s) …")

    for img_path in tqdm(img_paths, desc="Predicting"):
        img = load_image(img_path)

        prob_map, binary_map = predict_image(
            model, img,
            tile_size=args.tile_size,
            stride=args.stride,
            normalization=args.norm,
            threshold=args.threshold,
            device=DEVICE,
        )

        # Save binary mask
        out_mask = output_dir / f"{img_path.stem}_pred.png"
        cv2.imwrite(str(out_mask), binary_map)

        # Optionally save probability map
        if args.save_prob:
            np.save(str(output_dir / f"{img_path.stem}_prob.npy"), prob_map)

    print(f"\nPredictions saved to {output_dir}")


if __name__ == "__main__":
    main()
