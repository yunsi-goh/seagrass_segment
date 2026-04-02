"""
Inference script for ViT-based seagrass segmentation.

Mirrors scripts/infer_unet.py exactly — drop-in replacement.
Handles any-size input via patch-based tiling + Gaussian blending.

Usage:
    # Single image
    python scripts/infer_vit.py \
        --checkpoint outputs/checkpoints/vit/best.pth \
        --input path/to/image.jpg \
        --output outputs/predictions/

    # Directory
    python scripts/infer_vit.py \
        --checkpoint outputs/checkpoints/vit/best.pth \
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
from configs import config_vit as vcfg
from models.vit import build_vit_seg
from utils.normalization import normalize_np
from utils.tiling import reconstruct_from_tiles

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ── Patch-based inference ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_image_vit(
    model: torch.nn.Module,
    img: np.ndarray,
    tile_size: int = 512,
    stride: int = 256,
    normalization: str = cfg.NORMALIZATION,
    threshold: float = 0.5,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ViT inference on a full image with patch-based prediction.

    Args:
        model:         Loaded ViTSegNet in eval mode.
        img:           H×W×3 float32 in [0,255].
        tile_size:     Patch size (must match model img_size, default 512).
        stride:        Overlap stride (default 50 % → tile_size//2).
        normalization: Normalization strategy (matches training config).
        threshold:     Binarisation threshold.
        device:        Torch device.

    Returns:
        prob_map:   H×W float32 probability map (0–1).
        binary_map: H×W uint8 binary mask (0 or 255).
    """
    model.eval()
    H, W = img.shape[:2]

    tile_preds:  List[np.ndarray]       = []
    tile_coords: List[Tuple[int, int]]  = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tile = img[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            # Pad to tile_size
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

            tile_norm = normalize_np(tile, method=normalization)
            tensor = torch.from_numpy(
                tile_norm.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            logit = model(tensor)                            # 1×1×H×W
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # H×W

            prob = prob[:th, :tw]   # unpad
            tile_preds.append(prob)
            tile_coords.append((y, x))

    prob_map   = reconstruct_from_tiles(tile_preds, tile_coords, (H, W), tile_size)
    binary_map = (prob_map >= threshold).astype(np.uint8) * 255
    return prob_map, binary_map


# ── Load model from checkpoint ────────────────────────────────────────────────

def load_vit_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    vit_model_name  = ckpt.get("vit_model_name",  vcfg.VIT_MODEL_NAME)
    img_size        = ckpt.get("img_size",         vcfg.CROP_SIZE)
    decode_channels = ckpt.get("decode_channels",  vcfg.DECODE_CHANNELS)
    in_channels     = ckpt.get("in_channels",      cfg.IN_CHANNELS)
    out_channels    = ckpt.get("out_channels",      cfg.OUT_CHANNELS)

    model = build_vit_seg(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        model_name=vit_model_name,
        pretrained=False,       # weights come from checkpoint
        decode_channels=decode_channels,
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    print(
        f"ViT model loaded from {checkpoint_path}  "
        f"(encoder={vit_model_name}, img_size={img_size})"
    )
    return model


# ── Load image ────────────────────────────────────────────────────────────────

def load_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run ViT seagrass inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input",      type=str, required=True,
                        help="Image file or directory")
    parser.add_argument("--output",     type=str, default=str(cfg.PRED_DIR / "vit"))
    parser.add_argument("--tile_size",  type=int, default=vcfg.CROP_SIZE)
    parser.add_argument("--stride",     type=int, default=vcfg.TILE_STRIDE)
    parser.add_argument("--threshold",  type=float, default=cfg.EVAL_THRESHOLD)
    parser.add_argument("--norm",       type=str, default=cfg.NORMALIZATION)
    parser.add_argument("--save_prob",  action="store_true",
                        help="Also save float32 probability map as .npy")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_vit_model(args.checkpoint)

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

    print(f"\nRunning ViT inference on {len(img_paths)} image(s) …")

    for img_path in tqdm(img_paths, desc="Predicting"):
        img = load_image(img_path)
        prob_map, binary_map = predict_image_vit(
            model, img,
            tile_size=args.tile_size,
            stride=args.stride,
            normalization=args.norm,
            threshold=args.threshold,
            device=DEVICE,
        )
        cv2.imwrite(str(output_dir / f"{img_path.stem}_pred.png"), binary_map)
        if args.save_prob:
            np.save(str(output_dir / f"{img_path.stem}_prob.npy"), prob_map)

    print(f"\nPredictions saved to {output_dir}")


if __name__ == "__main__":
    main()

