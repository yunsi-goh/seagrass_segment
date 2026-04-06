"""
Training script for ResNet34 U-Net seagrass segmentation.

Usage:
    python scripts/train_unet.py
    python main.py train --model unet

Outputs: outputs/unet__bs<N>__lrdec<LR>/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from configs import config_unet as cfg
from data.dataset import (
    SeagrassDataset,
    get_train_augmentation,
    split_source_dir,
)
from models.unet import CombinedLoss, build_unet, count_parameters
from utils.metrics import batch_iou
from utils.normalization import normalize_np
from utils.tiling import reconstruct_from_tiles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


def plot_training_metrics(history, out_path: Path):
    """Save a compact training dashboard as a PNG."""
    if not history["epoch"]:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_iou"], label="train")
    axes[1].plot(history["epoch"], history["val_iou"],   label="val")
    axes[1].set_title("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(history["epoch"], history["lr_encoder"], label="encoder")
    axes[2].plot(history["epoch"], history["lr_decoder"], label="decoder")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_test_inference(best_ckpt_path: Path, data_dir: Path, test_stems, pred_dir: Path):
    """Run inference for held-out test stems and save binary masks."""
    if not best_ckpt_path.exists():
        print(f"Skipping test inference: checkpoint not found at {best_ckpt_path}")
        return
    if not test_stems:
        print("Skipping test inference: no test samples from split")
        return

    from scripts.infer_unet import load_model, load_image, predict_image

    model = load_model(str(best_ckpt_path))

    img_dir = data_dir / "images"
    pred_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    saved = 0
    for stem in test_stems:
        img_path = None
        for ext in exts:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  [SKIP] test image missing for stem '{stem}'")
            continue

        img = load_image(img_path)
        _, binary_map = predict_image(
            model,
            img,
            tile_size=cfg.TILE_SIZE,
            stride=cfg.TILE_STRIDE,
            normalization=cfg.NORMALIZATION,
            threshold=cfg.EVAL_THRESHOLD,
            device=DEVICE,
        )
        out_mask = pred_dir / f"{stem}_pred.png"
        cv2.imwrite(str(out_mask), binary_map)
        saved += 1

    print(f"Test inference complete. Saved {saved} mask(s) → {pred_dir}")


def train_epoch(model, loader, criterion, optimizer, device, grad_accum_steps=1):
    model.train()
    loss_sum = iou_sum = 0.0
    optimizer.zero_grad()
    for step, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks) / grad_accum_steps
        loss.backward()
        loss_sum += loss.item() * grad_accum_steps
        iou_sum  += batch_iou(logits.detach(), masks).item()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    n = max(len(loader), 1)
    return loss_sum / n, iou_sum / n


@torch.no_grad()
def val_epoch_tiled(model, val_stems, data_dir, criterion, device,
                   tile_size, stride, normalization, threshold=0.5):
    """
    Validates on full-resolution images using the same tiled inference as the
    inference scripts, avoiding the ~5x scale mismatch of SmallestMaxSize(512).
    IoU and loss are computed at native image resolution.
    """
    model.eval()
    img_dir  = data_dir / "images"
    mask_dir = data_dir / "masks"
    exts     = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    total_loss = 0.0
    total_iou  = 0.0
    count      = 0

    for stem in val_stems:
        img_path = next(
            (img_dir / f"{stem}{ext}" for ext in exts
             if (img_dir / f"{stem}{ext}").exists()), None
        )
        mask_path = mask_dir / f"{stem}.png"
        if img_path is None or not mask_path.exists():
            continue

        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]

        tile_preds  = []
        tile_coords = []
        tile_losses = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile = img[y:y+tile_size, x:x+tile_size]
                gt   = mask[y:y+tile_size, x:x+tile_size]
                th, tw = tile.shape[:2]
                pad_h = tile_size - th
                pad_w = tile_size - tw
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))
                    gt   = np.pad(gt,   ((0, pad_h), (0, pad_w)))

                tile_norm = normalize_np(tile, method=normalization)
                t    = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
                gt_t = torch.from_numpy(
                    (gt > 127).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0).to(device)

                logit = model(t)
                tile_losses.append(criterion(logit, gt_t).item())

                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                tile_preds .append(prob[:th, :tw])
                tile_coords.append((y, x))

        prob_map    = reconstruct_from_tiles(tile_preds, tile_coords, (H, W), tile_size)
        gt_binary   = (mask > 127).astype(np.float32)
        pred_binary = (prob_map >= threshold).astype(np.float32)
        intersection = (pred_binary * gt_binary).sum()
        union        = pred_binary.sum() + gt_binary.sum() - intersection
        iou          = 1.0 if union < 1.0 else float(intersection / union)

        total_loss += float(np.mean(tile_losses))
        total_iou  += float(iou)
        count      += 1

    if count == 0:
        return 0.0, 0.0
    return total_loss / count, total_iou / count

def main():
    run_name = f"unet__bs{cfg.BATCH_SIZE}__lrdec{cfg.LR_DECODER}"
    run_dir  = cfg.OUTPUT_DIR / run_name
    log_dir  = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "predictions" / "test"

    print("U-Net seagrass segmentation training")
    print(f"Device  : {DEVICE}")
    print(f"Encoder : {cfg.ENCODER_NAME} ({cfg.ENCODER_WEIGHTS})")
    print(f"ImgSize : {cfg.CROP_SIZE}")
    print(f"Run dir : {run_dir}\n")
    data_dir = cfg.DATA_DIR / "rgb"

    # ── Split ───────────────────────────────────────────────────────────────────
    train_stems, val_stems, test_stems = split_source_dir(
        data_dir, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )
    print(
        f"Source images → train: {len(train_stems)}  "
        f"val: {len(val_stems)}  test: {len(test_stems)}"
    )

    # ── Config log ────────────────────────────────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)
    config_log_path = log_dir / "config.json"
    with open(config_log_path, "w") as f:
        json.dump({
            "model":               "unet",
            "encoder_name":        cfg.ENCODER_NAME,
            "encoder_weights":     cfg.ENCODER_WEIGHTS,
            "in_channels":         cfg.IN_CHANNELS,
            "out_channels":        cfg.OUT_CHANNELS,
            "crop_size":           cfg.CROP_SIZE,
            "normalization":       cfg.NORMALIZATION,
            "batch_size":          cfg.BATCH_SIZE,
            "grad_accum_steps":    cfg.GRAD_ACCUM_STEPS,
            "epochs":              cfg.EPOCHS,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
            "lr_encoder":          cfg.LR_ENCODER,
            "lr_decoder":          cfg.LR_DECODER,
            "lr_decay_step":       cfg.LR_DECAY_STEP,
            "lr_decay_rate":       cfg.LR_DECAY_RATE,
            "train_ratio":         cfg.TRAIN_RATIO,
            "val_ratio":           cfg.VAL_RATIO,
            "random_seed":         cfg.RANDOM_SEED,
            "device":              str(DEVICE),
            "n_train":             len(train_stems),
            "n_val":               len(val_stems),
            "n_test":              len(test_stems),
            "run_name":            run_name,
        }, f, indent=2)
    print(f"Config log    → {config_log_path}")

    # ── Datasets ────────────────────────────────────────────────────────────────
    aug = get_train_augmentation(
        crop_size=cfg.CROP_SIZE,
        rescale_min=cfg.AUG_RESCALE_MIN,
        rescale_max=cfg.AUG_RESCALE_MAX,
        hflip_p=cfg.AUG_HFLIP_P, vflip_p=cfg.AUG_VFLIP_P,
        rotate_p=cfg.AUG_ROTATE_P, rotate_limit=cfg.AUG_ROTATE_LIMIT,
        elastic_p=cfg.AUG_ELASTIC_P, grid_p=cfg.AUG_GRID_P,
        brightness_p=cfg.AUG_BRIGHTNESS_P, contrast_p=cfg.AUG_CONTRAST_P,
        hue_p=cfg.AUG_HUE_P, blur_p=cfg.AUG_BLUR_P,
    )
    train_ds = SeagrassDataset(data_dir, cfg.NORMALIZATION, aug, train_stems)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )

    model = build_unet(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        encoder_name=cfg.ENCODER_NAME,
        encoder_weights=cfg.ENCODER_WEIGHTS,
    ).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}\n")

    criterion = CombinedLoss(dice_w=0.5, bce_w=0.5, pos_weight=7.0)  # pos_weight compensates ~7:1 bg:fg ratio

    optimizer = optim.Adam([
        {"params": model.encoder.parameters(),          "lr": cfg.LR_ENCODER},
        {"params": model.decoder.parameters(),          "lr": cfg.LR_DECODER},
        {"params": model.segmentation_head.parameters(),"lr": cfg.LR_DECODER},
    ])
    scheduler = StepLR(optimizer, step_size=cfg.LR_DECAY_STEP,
                       gamma=cfg.LR_DECAY_RATE)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"

    best_iou = float("-inf")
    patience_counter = 0
    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_iou": [],  "val_iou": [],
        "lr_encoder": [], "lr_decoder": [],
    }
    print(f"Training for {cfg.EPOCHS} epochs (early stop patience={cfg.EARLY_STOP_PATIENCE}) …\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_iou = train_epoch(model, train_loader, criterion,
                                      optimizer, DEVICE, cfg.GRAD_ACCUM_STEPS)
        va_loss, va_iou = val_epoch_tiled(
            model, val_stems, data_dir, criterion, DEVICE,
            cfg.TILE_SIZE, cfg.TILE_STRIDE, cfg.NORMALIZATION, cfg.EVAL_THRESHOLD,
        )
        scheduler.step()
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]

        print(f"[{epoch:4d}/{cfg.EPOCHS}]  "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"IoU  {tr_iou:.4f}/{va_iou:.4f}  "
              f"lr enc={lr_enc:.2e} dec={lr_dec:.2e}  ({time.time()-t0:.1f}s)")

        with open(log_path, "a") as f:
            f.write(json.dumps(dict(
                epoch=epoch, train_loss=tr_loss, train_iou=tr_iou,
                val_loss=va_loss, val_iou=va_iou,
                lr_encoder=lr_enc, lr_decoder=lr_dec,
            )) + "\n")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)
        history["lr_encoder"].append(lr_enc)
        history["lr_decoder"].append(lr_dec)

        if va_iou > best_iou:
            best_iou = va_iou
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": va_iou,
                "in_channels": cfg.IN_CHANNELS,
                "out_channels": cfg.OUT_CHANNELS,
                "encoder_name": cfg.ENCODER_NAME,
                "encoder_weights": cfg.ENCODER_WEIGHTS,
            }, ckpt_dir / "best.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
                break

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "in_channels": cfg.IN_CHANNELS,
                "out_channels": cfg.OUT_CHANNELS,
                "encoder_name": cfg.ENCODER_NAME,
                "encoder_weights": cfg.ENCODER_WEIGHTS,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

    plot_path = log_dir / "metrics.png"
    plot_training_metrics(history, plot_path)

    # ── Best-epoch summary ────────────────────────────────────────────────────
    if history["epoch"]:
        rows = [
            dict(epoch=e, train_loss=tl, val_loss=vl, train_iou=ti, val_iou=vi)
            for e, tl, vl, ti, vi in zip(
                history["epoch"], history["train_loss"], history["val_loss"],
                history["train_iou"],  history["val_iou"],
            )
        ]
        best = max(rows, key=lambda r: r["val_iou"])
        with open(log_path, "a") as f:
            f.write(json.dumps({"type": "best_epoch", **best}) + "\n")

    run_test_inference(ckpt_dir / "best.pth", data_dir, test_stems, pred_dir)

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Run dir       → {run_dir}")
    print(f"Checkpoints   → {ckpt_dir}")
    print(f"Training plot → {plot_path}")


if __name__ == "__main__":
    main()
