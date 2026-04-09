"""
Inference script for ViT-B/16 seagrass presence/absence classifier.

Given an image, returns:
  - a probability (0–1) that seagrass is present
  - a binary label (0 = absent, 1 = present)

Usage:
    python scripts/infer_vit_cls.py --checkpoint outputs/vit_cls__bs32__lrhead0.0003/checkpoints/best.pth --input path/to/image.jpg
    python main.py infer --model vit_cls --checkpoint <ckpt> --input <img>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import config as cfg
from configs import config_vit_cls as ccfg
from models.vit_classifier import build_vit_classifier
from utils.normalization import normalize_np

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


def load_vit_classifier(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint, reconstructing architecture from saved metadata."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_vit_classifier(
        model_name=ckpt.get("vit_model_name", ccfg.VIT_MODEL_NAME),
        pretrained=False,
        img_size=ckpt.get("img_size", ccfg.CROP_SIZE),
        dropout=ckpt.get("dropout", ccfg.DROPOUT),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(
        f"ViT classifier loaded from {checkpoint_path}  "
        f"(val F1={ckpt.get('val_f1', '?'):.4f}, "
        f"val acc={ckpt.get('val_acc', '?'):.4f})"
    )
    return model


def load_image(path: str | Path) -> np.ndarray:
    """Return H×W×3 uint8 RGB image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def predict_image(
    model: torch.nn.Module,
    img: np.ndarray,
    img_size: int = 224,
    normalization: str = "imagenet",
    threshold: float = 0.5,
) -> tuple[float, int]:
    """
    Classify a single image.

    Returns:
        (probability, label)  where label is 0 or 1.
    """
    # Resize + centre-crop to model input size
    h, w = img.shape[:2]
    scale = img_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Centre crop
    y0 = (nh - img_size) // 2
    x0 = (nw - img_size) // 2
    crop = resized[y0:y0 + img_size, x0:x0 + img_size]

    crop_norm = normalize_np(crop.astype(np.float32), method=normalization)
    tensor = (
        torch.from_numpy(crop_norm.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(next(model.parameters()).device)
    )

    with torch.no_grad():
        logit = model(tensor)          # (1, 1)
        prob  = torch.sigmoid(logit).item()

    return prob, int(prob >= threshold)


def main():
    parser = argparse.ArgumentParser(description="ViT classifier inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--input",      required=True, help="Image file to classify")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Probability threshold for positive class (default 0.5)")
    args = parser.parse_args()

    model    = load_vit_classifier(args.checkpoint)
    img_size = int(torch.load(args.checkpoint, map_location="cpu").get("img_size",
                                                                        ccfg.CROP_SIZE))
    img = load_image(args.input)
    prob, label = predict_image(model, img, img_size=img_size,
                                normalization=cfg.NORMALIZATION,
                                threshold=args.threshold)

    result = "PRESENT" if label == 1 else "ABSENT"
    print(f"\nImage     : {args.input}")
    print(f"Seagrass  : {result}")
    print(f"Prob      : {prob:.4f}")


if __name__ == "__main__":
    main()
