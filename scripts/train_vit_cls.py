"""
Training script for ViT-B/16 seagrass presence/absence classifier.

Labels are derived from segmentation masks: an image is labelled 1 (seagrass
present) if its mask contains any foreground pixel, else 0 (absent).

Usage:
    python scripts/train_vit_cls.py
    python main.py train --model vit_cls

Outputs: outputs/vit_cls__bs<N>__lrhead<LR>/
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from configs import config as cfg
from configs import config_vit_cls as ccfg
from data.dataset import split_source_dir
from models.vit_classifier import build_vit_classifier, count_parameters
from utils.normalization import normalize_np

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeagrassClassificationDataset(Dataset):
    """
    Binary image-level dataset.
    Label = 1 if mask has any non-zero pixel, else 0.
    """

    def __init__(
        self,
        data_dir: Path,
        normalization: str,
        crop_size: int,
        stems: list[str],
        augment: bool = False,
    ):
        self.data_dir      = Path(data_dir)
        self.norm_method   = normalization
        self.crop_size     = crop_size
        self.augment       = augment
        self.samples: list[tuple[Path, int]] = []

        img_dir  = self.data_dir / "images"
        mask_dir = self.data_dir / "masks"
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        for stem in stems:
            img_path = next(
                (img_dir / f"{stem}{e}" for e in exts
                 if (img_dir / f"{stem}{e}").exists()), None
            )
            mask_path = mask_dir / f"{stem}.png"
            if img_path is None or not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            label = 1 if (mask is not None and mask.max() > 0) else 0
            self.samples.append((img_path, label))

        self._aug = self._build_aug() if augment else self._build_val_aug()

    def _build_aug(self) -> A.Compose:
        cs = self.crop_size
        return A.Compose([
            A.RandomResizedCrop(size=(cs, cs), scale=(0.7, 1.0),
                                ratio=(0.8, 1.2), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                                 val_shift_limit=15, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    def _build_val_aug(self) -> A.Compose:
        cs = self.crop_size
        return A.Compose([
            A.SmallestMaxSize(max_size=cs, p=1.0),
            A.CenterCrop(height=cs, width=cs, p=1.0),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.cvtColor(
            cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        ).astype(np.float32)

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_uint8 = self._aug(image=img_uint8)["image"]
        img_norm  = normalize_np(img_uint8.astype(np.float32), method=self.norm_method)

        img_t   = torch.from_numpy(img_norm.transpose(2, 0, 1))
        label_t = torch.tensor([label], dtype=torch.float32)
        return img_t, label_t


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """Returns accuracy, precision, recall, F1 as Python floats."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return acc, prec, rec, f1


# ── Training / validation loops ───────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    all_logits, all_labels = [], []
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        loss_sum += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    n = max(len(loader), 1)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = compute_metrics(all_logits, all_labels)
    return loss_sum / n, acc, f1


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    all_logits, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits, labels).item()
        all_logits.append(logits)
        all_labels.append(labels)

    n = max(len(loader), 1)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = compute_metrics(all_logits, all_labels)
    return loss_sum / n, acc, f1


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_metrics(history: dict, out_path: Path):
    if not history["epoch"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"],   label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE"); axes[0].grid(alpha=.3); axes[0].legend()

    axes[1].plot(history["epoch"], history["train_f1"], label="train")
    axes[1].plot(history["epoch"], history["val_f1"],   label="val")
    axes[1].set(title="F1", xlabel="Epoch", ylabel="F1"); axes[1].grid(alpha=.3); axes[1].legend()

    axes[2].plot(history["epoch"], history["lr"], label="lr")
    axes[2].set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
    axes[2].set_yscale("log"); axes[2].grid(alpha=.3); axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_name = f"vit_cls__bs{ccfg.BATCH_SIZE}__lrhead{ccfg.LR_HEAD}"
    run_dir  = cfg.OUTPUT_DIR / run_name
    log_dir  = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"

    print("ViT-B/16 seagrass presence classifier")
    print(f"  Encoder : {ccfg.VIT_MODEL_NAME}")
    print(f"  ImgSize : {ccfg.CROP_SIZE}×{ccfg.CROP_SIZE}")
    print(f"  Device  : {DEVICE}\n")

    data_dir = cfg.DATA_DIR / "rgb"
    train_stems, val_stems, test_stems = split_source_dir(
        data_dir, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )
    print(f"  Split → train:{len(train_stems)}  val:{len(val_stems)}  test:{len(test_stems)}\n")

    # ── Config log ────────────────────────────────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model":           "vit_cls",
            "vit_model_name":  ccfg.VIT_MODEL_NAME,
            "crop_size":       ccfg.CROP_SIZE,
            "dropout":         ccfg.DROPOUT,
            "freeze_epochs":   ccfg.VIT_FREEZE_EPOCHS,
            "batch_size":      ccfg.BATCH_SIZE,
            "lr_encoder":      ccfg.LR_ENCODER,
            "lr_head":         ccfg.LR_HEAD,
            "weight_decay":    ccfg.WEIGHT_DECAY,
            "pos_weight":      ccfg.POS_WEIGHT,
            "epochs":          cfg.EPOCHS,
            "early_stop":      cfg.EARLY_STOP_PATIENCE,
            "n_train":         len(train_stems),
            "n_val":           len(val_stems),
            "n_test":          len(test_stems),
            "device":          str(DEVICE),
        }, f, indent=2)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SeagrassClassificationDataset(
        data_dir, cfg.NORMALIZATION, ccfg.CROP_SIZE, train_stems, augment=True
    )
    val_ds = SeagrassClassificationDataset(
        data_dir, cfg.NORMALIZATION, ccfg.CROP_SIZE, val_stems, augment=False
    )
    test_ds = SeagrassClassificationDataset(
        data_dir, cfg.NORMALIZATION, ccfg.CROP_SIZE, test_stems, augment=False
    )

    # Log label distribution
    n_pos = sum(1 for _, l in train_ds.samples if l == 1)
    n_neg = len(train_ds) - n_pos
    print(f"  Train class balance — present:{n_pos}  absent:{n_neg}\n")

    train_loader = DataLoader(
        train_ds, batch_size=ccfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=ccfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=ccfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_vit_classifier(
        model_name=ccfg.VIT_MODEL_NAME,
        pretrained=ccfg.VIT_PRETRAINED,
        img_size=ccfg.CROP_SIZE,
        dropout=ccfg.DROPOUT,
    ).to(DEVICE)

    total_p, trainable_p = count_parameters(model), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Parameters: {total_p:,} total | {trainable_p:,} trainable\n")

    # Freeze encoder for warmup
    if ccfg.VIT_FREEZE_EPOCHS > 0:
        print(f"  Freezing ViT encoder for first {ccfg.VIT_FREEZE_EPOCHS} epoch(s).")
        for p in model.encoder.parameters():
            p.requires_grad = False

    # ── Loss ──────────────────────────────────────────────────────────────────
    pw = torch.tensor([ccfg.POS_WEIGHT], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ccfg.LR_ENCODER},
        {"params": model.head.parameters(),    "lr": ccfg.LR_HEAD},
    ], weight_decay=ccfg.WEIGHT_DECAY)

    if ccfg.USE_COSINE_LR:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=0.0)
    else:
        scheduler = StepLR(optimizer, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_RATE)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"

    best_f1 = float("-inf")
    patience_counter = 0
    history: dict = {"epoch": [], "train_loss": [], "val_loss": [],
                     "train_f1": [], "val_f1": [], "lr": []}

    print(f"Training for {cfg.EPOCHS} epochs  (patience={cfg.EARLY_STOP_PATIENCE}) …\n")

    for epoch in range(1, cfg.EPOCHS + 1):

        # Unfreeze encoder after warmup
        if ccfg.VIT_FREEZE_EPOCHS > 0 and epoch == ccfg.VIT_FREEZE_EPOCHS + 1:
            print(f"\nUnfreezing ViT encoder at epoch {epoch}.")
            for p in model.encoder.parameters():
                p.requires_grad = True

        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, va_f1 = val_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]  # head LR

        print(
            f"[{epoch:4d}/{cfg.EPOCHS}]  "
            f"loss {tr_loss:.4f}/{va_loss:.4f}  "
            f"F1 {tr_f1:.4f}/{va_f1:.4f}  "
            f"acc {tr_acc:.4f}/{va_acc:.4f}  "
            f"lr={lr:.2e}  ({time.time()-t0:.1f}s)"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps(dict(
                epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                train_f1=tr_f1, val_f1=va_f1,
                train_acc=tr_acc, val_acc=va_acc, lr=lr,
            )) + "\n")

        for k, v in [("epoch", epoch), ("train_loss", tr_loss), ("val_loss", va_loss),
                     ("train_f1", tr_f1), ("val_f1", va_f1), ("lr", lr)]:
            history[k].append(v)

        if va_f1 > best_f1:
            best_f1 = va_f1
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state_dict": model.state_dict(),
                "val_f1":      va_f1,
                "val_acc":     va_acc,
                "vit_model_name": ccfg.VIT_MODEL_NAME,
                "img_size":    ccfg.CROP_SIZE,
                "dropout":     ccfg.DROPOUT,
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
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

    plot_metrics(history, log_dir / "metrics.png")

    # ── Best epoch summary ────────────────────────────────────────────────────
    if history["epoch"]:
        rows = [{"epoch": e, "train_loss": tl, "val_loss": vl,
                 "train_f1": tf, "val_f1": vf}
                for e, tl, vl, tf, vf in zip(
                    history["epoch"], history["train_loss"], history["val_loss"],
                    history["train_f1"], history["val_f1"])]
        best = max(rows, key=lambda r: r["val_f1"])
        with open(log_path, "a") as f:
            f.write(json.dumps({"type": "best_epoch", **best}) + "\n")

    # ── Test set evaluation ───────────────────────────────────────────────────
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        te_loss, te_acc, te_f1 = val_epoch(model, test_loader, criterion, DEVICE)
        print(f"\nTest set  →  loss {te_loss:.4f}  acc {te_acc:.4f}  F1 {te_f1:.4f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "type": "test", "test_loss": te_loss,
                "test_acc": te_acc, "test_f1": te_f1,
            }) + "\n")

    print(f"\nTraining complete.  Best val F1 : {best_f1:.4f}")
    print(f"Run dir     → {run_dir}")
    print(f"Checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    main()
