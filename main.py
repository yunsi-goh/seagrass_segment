"""
Seagrass segmentation pipeline entry point.

Usage:
    # Train ViT-B/16 model (default)
    python main.py train

    # Train U-Net / ResNet34 model
    python main.py train --model unet

    # Run inference with ViT (default)
    python main.py infer --checkpoint outputs/vit__bs4__lrdec0.0003/checkpoints/best.pth --input path/to/img.jpg

    # Run inference with UNet
    python main.py infer --model unet --checkpoint outputs/unet__bs18__lrdec0.0003/checkpoints/best.pth --input path/to/img.jpg

    # Evaluate predictions
    python main.py evaluate --pred_dir outputs/predictions/ --gt_dir data/rgb/masks/
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Seagrass segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["train", "infer", "evaluate"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--model",
        choices=["unet", "vit"],
        default="vit",
        help=(
            "'vit' = ViT-B/16 encoder (SeagrassFinder best model, default). "
            "'unet' = ResNet34 U-Net (original pipeline)."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "train":
        if args.model == "vit":
            from scripts.train_vit import main as run
        else:
            from scripts.train_unet import main as run
        run()
    elif args.command == "infer":
        if args.model == "vit":
            from scripts.infer_vit import main as run
        else:
            from scripts.infer_unet import main as run
        run()
    elif args.command == "evaluate":
        # Forward selected model into evaluate.py, which uses it for
        # --run_inference mode.
        sys.argv = [sys.argv[0], "--model", args.model] + remaining
        from scripts.evaluate import main as run
        run()


if __name__ == "__main__":
    main()
