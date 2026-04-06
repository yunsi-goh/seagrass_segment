"""
Evaluation script for seagrass segmentation.

Computes per-image and aggregate pixel accuracy, precision, recall, F1, IoU.

Usage:
    # Evaluate pre-existing predictions
    python scripts/evaluate.py \\
        --pred_dir outputs/predictions/ \\
        --gt_dir data/rgb/masks/

    # Run inference then evaluate
    python scripts/evaluate.py \\
        --run_inference --model unet \\
        --checkpoint outputs/.../best.pth \\
        --image_dir path/to/images \\
        --gt_dir path/to/masks
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from configs import config as cfg
from utils.metrics import compute_metrics, AggregateResults


def load_prediction(pred_path: Path) -> np.ndarray:
    """
    Load a prediction:
      - .npy -> float32 probability map
      - .png -> binary uint8 {0, 255}
    """
    if pred_path.suffix == ".npy":
        return np.load(str(pred_path)).astype(np.float32)
    img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read {pred_path}")
    return img.astype(np.float32) / 255.0


def load_gt(gt_path: Path) -> np.ndarray:
    """Load ground-truth mask as binary {0,1} uint8."""
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise IOError(f"Cannot read GT: {gt_path}")
    return (gt > 0).astype(np.uint8)


def find_gt(stem: str, gt_dir: Path) -> Path | None:
    """
    Find the ground-truth mask for a prediction stem.
    Prediction stems may have '_pred' suffix; GT stems are plain.
    """
    clean = stem.replace("_pred", "")
    for ext in (".png", ".jpg", ".tif", ".tiff"):
        p = gt_dir / f"{clean}{ext}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate seagrass segmentation")

    # Mode A: evaluate pre-existing predictions
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Directory of predicted masks (.png or .npy)",
    )

    # Mode B: run inference then evaluate
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument(
        "--model",
        choices=["unet", "vit", "sam2unet"],
        default="unet",
        help="Model used when --run_inference is set.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)

    # Common
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory of ground-truth masks",
    )
    parser.add_argument("--threshold", type=float, default=cfg.EVAL_THRESHOLD)
    parser.add_argument(
        "--output",
        type=str,
        default=str(cfg.OUTPUT_DIR / "eval_results.csv"),
        help="Path to save per-image CSV results",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)

    if not args.run_inference and not args.pred_dir:
        parser.error(
            "Provide --pred_dir or use --run_inference with --checkpoint and --image_dir"
        )

    if args.run_inference:
        if not args.checkpoint or not args.image_dir:
            parser.error("--run_inference requires --checkpoint and --image_dir")

        pred_dir = cfg.PRED_DIR / "eval_run" / args.model
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Import and call model-specific inference logic directly.
        if args.model == "vit":
            from configs import config_vit as vcfg
            from scripts.infer_vit import load_vit_model as load_model
            from scripts.infer_vit import load_image, predict_image_vit as predict_image
            tile_size = vcfg.CROP_SIZE
            stride = vcfg.TILE_STRIDE
        elif args.model == "sam2unet":
            from configs import config_sam2unet as scfg
            from scripts.infer_sam2unet import (
                load_sam2unet_model as load_model,
                load_image,
                predict_image_sam2unet as predict_image,
            )
            tile_size = scfg.CROP_SIZE
            stride = scfg.TILE_STRIDE
        else:
            from configs import config_unet as ucfg
            from scripts.infer_unet import load_model, load_image, predict_image
            tile_size = ucfg.TILE_SIZE
            stride = ucfg.TILE_STRIDE

        model = load_model(args.checkpoint)

        img_paths = sorted(
            list(Path(args.image_dir).glob("*.jpg"))
            + list(Path(args.image_dir).glob("*.png"))
            + list(Path(args.image_dir).glob("*.tif"))
            + list(Path(args.image_dir).glob("*.tiff"))
        )
        print(f"Running {args.model} inference on {len(img_paths)} image(s) ...")
        for ip in tqdm(img_paths):
            img = load_image(ip)
            _, binary = predict_image(
                model,
                img,
                tile_size=tile_size,
                stride=stride,
                normalization=cfg.NORMALIZATION,
                threshold=args.threshold,
            )
            cv2.imwrite(str(pred_dir / f"{ip.stem}_pred.png"), binary)

        args.pred_dir = str(pred_dir)

    pred_dir = Path(args.pred_dir)
    pred_paths = sorted(
        list(pred_dir.glob("*_pred.png"))
        + list(pred_dir.glob("*_pred.npy"))
        + list(pred_dir.glob("*.png"))
    )

    seen_stems = set()
    unique_preds = []
    for p in pred_paths:
        if p.stem not in seen_stems:
            seen_stems.add(p.stem)
            unique_preds.append(p)
    pred_paths = unique_preds

    if not pred_paths:
        raise FileNotFoundError(f"No predictions found in {pred_dir}")

    print(f"\nEvaluating {len(pred_paths)} predictions ...")
    print(f"GT directory : {gt_dir}")
    print(f"Threshold    : {args.threshold}\n")

    results = AggregateResults()
    rows = []

    for pred_path in tqdm(pred_paths, desc="Evaluating"):
        gt_path = find_gt(pred_path.stem, gt_dir)
        if gt_path is None:
            print(f"  [SKIP] no GT found for {pred_path.stem}")
            continue

        pred = load_prediction(pred_path)
        gt = load_gt(gt_path)

        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        m = compute_metrics(pred, gt, threshold=args.threshold)
        results.add(m)

        rows.append(
            {
                "image": pred_path.stem,
                "accuracy": f"{m.accuracy:.4f}",
                "precision": f"{m.precision:.4f}",
                "recall": f"{m.recall:.4f}",
                "f1": f"{m.f1:.4f}",
                "iou": f"{m.iou:.4f}",
            }
        )

        print(f"  {pred_path.stem:40s}  {m}")

    results.print_summary()

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = results.summary()
    rows.append(
        {
            "image": "MEAN",
            "accuracy": f"{summary.get('mean_accuracy', 0):.4f}",
            "precision": f"{summary.get('mean_precision', 0):.4f}",
            "recall": f"{summary.get('mean_recall', 0):.4f}",
            "f1": f"{summary.get('mean_f1', 0):.4f}",
            "iou": f"{summary.get('mean_iou', 0):.4f}",
        }
    )
    rows.append(
        {
            "image": "STD",
            "accuracy": f"{summary.get('std_accuracy', 0):.4f}",
            "precision": f"{summary.get('std_precision', 0):.4f}",
            "recall": f"{summary.get('std_recall', 0):.4f}",
            "f1": f"{summary.get('std_f1', 0):.4f}",
            "iou": f"{summary.get('std_iou', 0):.4f}",
        }
    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    out_json = out_csv.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump({"per_image": rows, "summary": summary}, f, indent=2)

    print(f"Results saved -> {out_csv}")
    print(f"              -> {out_json}")


if __name__ == "__main__":
    main()
