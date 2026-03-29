"""
Training script for RGB seagrass segmentation.

Current default setup:
  Train on source RGB images with online augmentation.
  Fine-tune a U-Net with an ImageNet-initialized ResNet34 encoder.
  Use differential encoder/decoder learning rates, CombinedLoss,
  early stopping, and checkpointing.

Outputs: outputs/checkpoints/rgb/best.pth
         outputs/checkpoints/rgb/epoch_XXXX.pth
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from configs import config as cfg
from data.dataset import (
    SeagrassDataset,
    get_train_augmentation,
    get_val_augmentation,
    split_source_dir,
)
from models.unet import CombinedLoss, build_unet, count_parameters
from utils.metrics import batch_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


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
def val_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = iou_sum = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits, masks).item()
        iou_sum  += batch_iou(logits, masks).item()
    n = max(len(loader), 1)
    return loss_sum / n, iou_sum / n

def main():
    print("RGB training")
    print(f"Device : {DEVICE}\n")
    data_dir = cfg.DATA_DIR / "rgb"

    # ── Split ───────────────────────────────────────────────────────────────────
    train_stems, val_stems, _ = split_source_dir(
        data_dir, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )
    print(f"Source images -> train: {len(train_stems)}  val: {len(val_stems)}")

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
    val_aug = get_val_augmentation(crop_size=cfg.CROP_SIZE)
    train_ds = SeagrassDataset(data_dir, cfg.NORMALIZATION, aug, train_stems)
    val_ds   = SeagrassDataset(data_dir, cfg.NORMALIZATION, val_aug, val_stems)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    model = build_unet(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        encoder_name=cfg.ENCODER_NAME,
        encoder_weights=cfg.ENCODER_WEIGHTS,
    ).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}\n")

    criterion = CombinedLoss(dice_w=0.5, bce_w=0.5)

    optimizer = optim.Adam([
        {"params": model.encoder.parameters(),          "lr": cfg.LR_ENCODER},
        {"params": model.decoder.parameters(),          "lr": cfg.LR_DECODER},
        {"params": model.segmentation_head.parameters(),"lr": cfg.LR_DECODER},
    ])
    scheduler = StepLR(optimizer, step_size=cfg.LR_DECAY_STEP,
                       gamma=cfg.LR_DECAY_RATE)

    ckpt_dir = cfg.CKPT_DIR / "rgb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cfg.LOG_DIR / "train_rgb.jsonl"

    best_iou = 0.0
    patience_counter = 0
    print(f"Training for {cfg.EPOCHS} epochs (early stop patience={cfg.EARLY_STOP_PATIENCE}) …\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_iou = train_epoch(model, train_loader, criterion,
                                      optimizer, DEVICE, cfg.GRAD_ACCUM_STEPS)
        va_loss, va_iou = val_epoch  (model, val_loader,   criterion, DEVICE)
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

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    main()
