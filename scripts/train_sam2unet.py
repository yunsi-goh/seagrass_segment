"""
Training script for SAM2-UNet seagrass segmentation.

Huang S (2025) Advancing Seagrass Semantic Segmentation with SAM2 Models.
Journal of High School Science 9(2):235–47.

Usage:
    python scripts/train_sam2unet.py
    python main.py train --model sam2unet

Outputs: outputs/sam2unet__<size>__bs<N>__lr<LR>/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from configs import config as cfg
from configs import config_sam2unet as scfg
from data.dataset import SeagrassDataset, get_train_augmentation, split_source_dir
from models.sam2unet import SAM2UNetLoss, build_sam2unet, count_parameters
from utils.normalization import normalize_np
from utils.tiling import reconstruct_from_tiles
from utils.metrics import batch_iou

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def plot_training_metrics(history: dict, out_path: Path):
    if not history["epoch"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"],   label="val")
    axes[0].set(title="Loss",     xlabel="Epoch", ylabel="Loss");   axes[0].grid(alpha=.3); axes[0].legend()

    axes[1].plot(history["epoch"], history["train_iou"],  label="train")
    axes[1].plot(history["epoch"], history["val_iou"],    label="val")
    axes[1].set(title="IoU",      xlabel="Epoch", ylabel="IoU");    axes[1].grid(alpha=.3); axes[1].legend()

    axes[2].plot(history["epoch"], history["lr"], label="lr")
    axes[2].set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
    axes[2].set_yscale("log");  axes[2].grid(alpha=.3); axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_param_groups(model, backbone_lr: float, head_lr: float) -> list:
    """Split parameters into backbone vs head groups for differential LRs."""
    bb_ids = {id(p) for p in model.encoder.backbone.parameters()}
    backbone_p = [p for p in model.parameters() if id(p) in bb_ids and p.requires_grad]
    head_p     = [p for p in model.parameters() if id(p) not in bb_ids and p.requires_grad]
    groups = []
    if backbone_p:
        groups.append({"params": backbone_p, "lr": backbone_lr})
    if head_p:
        groups.append({"params": head_p, "lr": head_lr})
    return groups


# ── Train / Val ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, grad_accum_steps=1):
    model.train()
    loss_sum = iou_sum = 0.0
    optimizer.zero_grad()

    for step, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)   # tuple (S1, S2, S3) in train mode

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        loss = criterion(list(outputs), masks) / grad_accum_steps
        loss.backward()
        loss_sum += loss.item() * grad_accum_steps

        # IoU on the finest prediction (last output)
        iou_sum += batch_iou(outputs[-1].detach(), masks).item()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    n = max(len(loader), 1)
    return loss_sum / n, iou_sum / n


@torch.no_grad()
def val_epoch_tiled(model, val_stems, data_dir, criterion, device,
                    tile_size, stride, normalization, threshold=0.5):
    """Validate on full-resolution images with tiled inference (same as test time)."""
    model.eval()
    img_dir  = data_dir / "images"
    mask_dir = data_dir / "masks"
    exts     = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    total_loss = total_iou = count = 0.0

    for stem in val_stems:
        img_path = next(
            (img_dir / f"{stem}{e}" for e in exts if (img_dir / f"{stem}{e}").exists()),
            None,
        )
        mask_path = mask_dir / f"{stem}.png"
        if img_path is None or not mask_path.exists():
            continue

        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]

        tile_preds, tile_coords, tile_losses = [], [], []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile = img[y:y+tile_size, x:x+tile_size]
                gt   = mask[y:y+tile_size, x:x+tile_size]
                th, tw = tile.shape[:2]
                if th < tile_size or tw < tile_size:
                    tile = np.pad(tile, ((0, tile_size-th), (0, tile_size-tw), (0,0)))
                    gt   = np.pad(gt,   ((0, tile_size-th), (0, tile_size-tw)))

                tile_norm = normalize_np(tile, method=normalization)
                t    = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
                gt_t = torch.from_numpy((gt > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

                logit = model(t)  # eval → single tensor
                tile_losses.append(criterion([logit], gt_t).item())

                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                tile_preds.append(prob[:th, :tw])
                tile_coords.append((y, x))

        prob_map    = reconstruct_from_tiles(tile_preds, tile_coords, (H, W), tile_size)
        gt_binary   = (mask > 127).astype(np.float32)
        pred_binary = (prob_map >= threshold).astype(np.float32)
        intersection = (pred_binary * gt_binary).sum()
        union        = pred_binary.sum() + gt_binary.sum() - intersection
        iou = 1.0 if union < 1.0 else float(intersection / union)

        total_loss += float(np.mean(tile_losses))
        total_iou  += iou
        count      += 1

    if count == 0:
        return 0.0, 0.0
    return total_loss / count, total_iou / count


@torch.no_grad()
def run_test_inference(best_ckpt: Path, data_dir: Path, test_stems, pred_dir: Path):
    if not best_ckpt.exists() or not test_stems:
        print("Skipping test inference.")
        return
    from scripts.infer_sam2unet import load_sam2unet_model, load_image, predict_image_sam2unet

    model = load_sam2unet_model(str(best_ckpt))
    pred_dir.mkdir(parents=True, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    saved = 0
    for stem in test_stems:
        img_path = next(
            (data_dir / "images" / f"{stem}{e}" for e in exts
             if (data_dir / "images" / f"{stem}{e}").exists()), None,
        )
        if img_path is None:
            continue
        img = load_image(img_path)
        _, binary = predict_image_sam2unet(
            model, img,
            tile_size=scfg.CROP_SIZE, stride=scfg.TILE_STRIDE,
            normalization=cfg.NORMALIZATION, threshold=cfg.EVAL_THRESHOLD,
            device=DEVICE,
        )
        cv2.imwrite(str(pred_dir / f"{stem}_pred.png"), binary)
        saved += 1
    print(f"Test inference done. Saved {saved} mask(s) → {pred_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_name  = f"sam2unet__{scfg.SAM2_MODEL_SIZE}__bs{scfg.BATCH_SIZE}__lr{scfg.LR}"
    run_dir   = cfg.OUTPUT_DIR / run_name
    log_dir   = run_dir / "logs"
    ckpt_dir  = run_dir / "checkpoints"
    pred_dir  = run_dir / "predictions" / "test"

    print("SAM2-UNet seagrass segmentation training")
    print(f"  Best model from Huang 2025 (IoU 0.923 / 0.935 on LFSG subsets)")
    print(f"  Backbone : Hiera-{scfg.SAM2_MODEL_SIZE}")
    print(f"  Crop     : {scfg.CROP_SIZE}×{scfg.CROP_SIZE}")
    print(f"  Device   : {DEVICE}\n")

    data_dir = cfg.DATA_DIR / "rgb"
    train_stems, val_stems, test_stems = split_source_dir(
        data_dir, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )
    print(f"  Split → train:{len(train_stems)}  val:{len(val_stems)}  test:{len(test_stems)}\n")

    # Config log
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model":           "sam2unet",
            "model_size":      scfg.SAM2_MODEL_SIZE,
            "checkpoint":      scfg.SAM2_CHECKPOINT,
            "rfb_channels":    scfg.RFB_CHANNELS,
            "adapter_ratio":   scfg.ADAPTER_RATIO,
            "crop_size":       scfg.CROP_SIZE,
            "normalization":   cfg.NORMALIZATION,
            "batch_size":      scfg.BATCH_SIZE,
            "lr":              scfg.LR,
            "weight_decay":    scfg.WEIGHT_DECAY,
            "pos_weight":      scfg.POS_WEIGHT,
            "freeze_epochs":   scfg.FREEZE_EPOCHS,
            "epochs":          cfg.EPOCHS,
            "early_stop":      cfg.EARLY_STOP_PATIENCE,
            "n_train":         len(train_stems),
            "n_val":           len(val_stems),
            "n_test":          len(test_stems),
            "device":          str(DEVICE),
        }, f, indent=2)

    # Dataset
    aug = get_train_augmentation(
        crop_size=scfg.CROP_SIZE,
        rescale_min=cfg.AUG_RESCALE_MIN, rescale_max=cfg.AUG_RESCALE_MAX,
        hflip_p=cfg.AUG_HFLIP_P,        vflip_p=cfg.AUG_VFLIP_P,
        rotate_p=cfg.AUG_ROTATE_P,      rotate_limit=cfg.AUG_ROTATE_LIMIT,
        elastic_p=cfg.AUG_ELASTIC_P,    grid_p=cfg.AUG_GRID_P,
        brightness_p=cfg.AUG_BRIGHTNESS_P, contrast_p=cfg.AUG_CONTRAST_P,
        hue_p=cfg.AUG_HUE_P,            blur_p=cfg.AUG_BLUR_P,
    )
    train_ds = SeagrassDataset(data_dir, cfg.NORMALIZATION, aug, train_stems)
    train_loader = DataLoader(
        train_ds, batch_size=scfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )

    # Model
    model = build_sam2unet(
        model_size=scfg.SAM2_MODEL_SIZE,
        pretrained_path=scfg.SAM2_CHECKPOINT,
        rfb_channels=scfg.RFB_CHANNELS,
        adapter_ratio=scfg.ADAPTER_RATIO,
        freeze_backbone=(scfg.FREEZE_EPOCHS > 0),
        img_size=scfg.CROP_SIZE,
    ).to(DEVICE)

    total_p, trainable_p = count_parameters(model)
    print(f"  Parameters: {total_p:,} total  |  {trainable_p:,} trainable\n")

    criterion = SAM2UNetLoss(
        bce_w=scfg.BCE_WEIGHT, iou_w=scfg.IOU_WEIGHT,
        pos_weight=scfg.POS_WEIGHT,
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=scfg.LR, weight_decay=scfg.WEIGHT_DECAY,
    )
    if scfg.USE_COSINE_LR:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=scfg.LR * 0.01)
    else:
        scheduler = StepLR(optimizer, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_RATE)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"

    best_iou = float("-inf")
    patience_counter = 0
    history: dict = {"epoch": [], "train_loss": [], "val_loss": [],
                     "train_iou": [], "val_iou": [], "lr": []}

    print(f"Training for {cfg.EPOCHS} epochs  (patience={cfg.EARLY_STOP_PATIENCE}) …\n")

    for epoch in range(1, cfg.EPOCHS + 1):

        # Unfreeze backbone after warmup
        if scfg.FREEZE_EPOCHS > 0 and epoch == scfg.FREEZE_EPOCHS + 1:
            print(f"\nUnfreezing Hiera backbone at epoch {epoch}.")
            model.unfreeze_backbone()
            # Differential LRs: backbone at LR×0.1, head/decoder stays at LR
            groups = _make_param_groups(model, backbone_lr=scfg.LR * 0.1, head_lr=scfg.LR)
            optimizer = optim.AdamW(groups, weight_decay=scfg.WEIGHT_DECAY)
            if scfg.USE_COSINE_LR:
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=cfg.EPOCHS - epoch, eta_min=0.0
                )

        t0 = time.time()
        tr_loss, tr_iou = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, cfg.GRAD_ACCUM_STEPS
        )
        va_loss, va_iou = val_epoch_tiled(
            model, val_stems, data_dir, criterion, DEVICE,
            scfg.CROP_SIZE, scfg.TILE_STRIDE, cfg.NORMALIZATION, cfg.EVAL_THRESHOLD,
        )
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]  # head LR (last group)

        print(
            f"[{epoch:4d}/{cfg.EPOCHS}]  "
            f"loss {tr_loss:.4f}/{va_loss:.4f}  "
            f"IoU {tr_iou:.4f}/{va_iou:.4f}  "
            f"lr={lr:.2e}  ({time.time()-t0:.1f}s)"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps(dict(
                epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                train_iou=tr_iou, val_iou=va_iou, lr=lr,
            )) + "\n")

        for k, v in [("epoch", epoch), ("train_loss", tr_loss), ("val_loss", va_loss),
                     ("train_iou", tr_iou), ("val_iou", va_iou), ("lr", lr)]:
            history[k].append(v)

        if va_iou > best_iou:
            best_iou = va_iou
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state_dict": model.state_dict(),
                "val_iou":     va_iou,
                "model_size":  scfg.SAM2_MODEL_SIZE,
                "rfb_channels": scfg.RFB_CHANNELS,
                "adapter_ratio": scfg.ADAPTER_RATIO,
                "crop_size":   scfg.CROP_SIZE,
            }, ckpt_dir / "best.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_size": scfg.SAM2_MODEL_SIZE,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

    plot_training_metrics(history, log_dir / "metrics.png")

    if history["epoch"]:
        best_row = max(
            zip(history["epoch"], history["train_loss"], history["val_loss"],
                history["train_iou"], history["val_iou"]),
            key=lambda r: r[4],
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({"type": "best_epoch", "epoch": best_row[0],
                                "val_iou": best_row[4]}) + "\n")

    run_test_inference(ckpt_dir / "best.pth", data_dir, test_stems, pred_dir)

    print(f"\nTraining complete.  Best val IoU : {best_iou:.4f}")
    print(f"Run dir      → {run_dir}")
    print(f"Checkpoints  → {ckpt_dir}")


if __name__ == "__main__":
    main()
