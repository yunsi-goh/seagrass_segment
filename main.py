"""
Seagrass pipeline entry point — segmentation (pixel masks) and classification (presence/absence).

Usage:
    # Train ViT-B/16 segmentation model (default)
    python main.py train

    # Train U-Net / EfficientNet-B4 segmentation model
    python main.py train --model unet

    # Train SAM2-UNet segmentation model (Hiera backbone)
    python main.py train --model sam2unet

    # Train ViT-B/16 image classifier (seagrass present / absent)
    python main.py train --model vit_cls

    # Run segmentation inference with ViT (default)
    python main.py infer --checkpoint outputs/vit__bs4__lrdec0.0003__wd0.0005/checkpoints/best.pth --input path/to/img.jpg

    # Run segmentation inference with UNet
    python main.py infer --model unet --checkpoint outputs/unet__efficientnet-b4__bs18__lrdec0.0003/checkpoints/best.pth --input path/to/img.jpg

    # Run segmentation inference with SAM2-UNet
    python main.py infer --model sam2unet --checkpoint outputs/sam2unet__small__bs4__lr0.0003/checkpoints/best.pth --input path/to/img.jpg

    # Run classifier inference on a single image (returns probability + PRESENT/ABSENT label)
    python main.py infer --model vit_cls --checkpoint outputs/vit_cls__bs32__lrhead0.0003/checkpoints/best.pth --input path/to/img.jpg

    # Evaluate segmentation predictions (pixel-level metrics; not applicable to vit_cls)
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
        choices=["unet", "vit", "sam2unet", "vit_cls"],
        default="vit",
        help=(
            "'vit' = ViT-B/16 encoder (SeagrassFinder best model, default). "
            "'unet' = ResNet34 U-Net (original pipeline). "
            "'sam2unet' = SAM2-UNet with Hiera backbone (Huang 2025). "
            "'vit_cls' = ViT-B/16 image-level classifier (seagrass present/absent)."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "train":
        if args.model == "vit":
            from scripts.train_vit import main as run
        elif args.model == "sam2unet":
            from scripts.train_sam2unet import main as run
        elif args.model == "vit_cls":
            from scripts.train_vit_cls import main as run
        else:
            from scripts.train_unet import main as run
        run()
    elif args.command == "infer":
        if args.model == "vit":
            from scripts.infer_vit import main as run
        elif args.model == "sam2unet":
            from scripts.infer_sam2unet import main as run
        elif args.model == "vit_cls":
            from scripts.infer_vit_cls import main as run
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
