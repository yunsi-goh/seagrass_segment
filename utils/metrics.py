"""
Evaluation metrics — exactly as defined in Jeon et al. (2021) Eqs. (3)–(7).

Metrics reported:
  - Pixel Accuracy  Eq.(3): (TP + TN) / (TP + TN + FP + FN)
  - Precision       Eq.(4): TP / (TP + FP)
  - Recall          Eq.(5): TP / (TP + FN)
  - F1 / Dice       Eq.(6): 2TP / (2TP + FP + FN)
  - IoU             Eq.(7): TP / (TP + FP + FN)

All computed at the pixel level over the full (reconstructed) predicted map.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import torch


# ── Dataclass to hold per-image results ────────────────────────────────────────

@dataclass
class SegMetrics:
    accuracy:  float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    iou:       float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"Acc={self.accuracy:.4f}  "
                f"Prec={self.precision:.4f}  "
                f"Rec={self.recall:.4f}  "
                f"F1={self.f1:.4f}  "
                f"IoU={self.iou:.4f}")


# ── Core computation ───────────────────────────────────────────────────────────

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> SegMetrics:
    """
    Compute segmentation metrics for a single image.

    Args:
        pred_mask: H×W float32 probability map  OR  binary uint8 {0,1} map.
        gt_mask:   H×W binary uint8 {0,1} ground-truth.
        threshold: Binarisation threshold (paper uses 0.5).

    Returns:
        SegMetrics dataclass.
    """
    if pred_mask.dtype != np.uint8:
        pred_binary = (pred_mask >= threshold).astype(np.uint8)
    else:
        pred_binary = (pred_mask > 0).astype(np.uint8)

    gt_binary = (gt_mask > 0).astype(np.uint8)

    tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (gt_binary == 0)).sum())
    fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())

    eps = 1e-8
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp        / (tp + fp + eps)
    recall    = tp        / (tp + fn + eps)
    f1        = 2 * tp    / (2 * tp + fp + fn + eps)
    iou       = tp        / (tp + fp + fn + eps)

    return SegMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        iou=float(iou),
    )


# ── Batch-level (for training loop) ────────────────────────────────────────────

def batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Fast IoU over a batch during training (no numpy, stays on GPU).

    Args:
        logits:  B×1×H×W raw logits.
        targets: B×1×H×W float binary.
    Returns:
        Scalar IoU tensor.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    return iou.mean()


# ── Aggregate results over a test set ─────────────────────────────────────────

@dataclass
class AggregateResults:
    per_image: List[SegMetrics] = field(default_factory=list)

    def add(self, m: SegMetrics) -> None:
        self.per_image.append(m)

    def summary(self) -> Dict[str, float]:
        if not self.per_image:
            return {}
        keys = ["accuracy", "precision", "recall", "f1", "iou"]
        result = {}
        for k in keys:
            vals = [getattr(m, k) for m in self.per_image]
            result[f"mean_{k}"] = float(np.mean(vals))
            result[f"std_{k}"]  = float(np.std(vals))
        return result

    def print_summary(self) -> None:
        s = self.summary()
        print("\n── Evaluation Summary ─────────────────────────────────────")
        print(f"  Images evaluated : {len(self.per_image)}")
        print(f"  Accuracy  : {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}")
        print(f"  Precision : {s['mean_precision']:.4f} ± {s['std_precision']:.4f}")
        print(f"  Recall    : {s['mean_recall']:.4f} ± {s['std_recall']:.4f}")
        print(f"  F1        : {s['mean_f1']:.4f} ± {s['std_f1']:.4f}")
        print(f"  IoU       : {s['mean_iou']:.4f} ± {s['std_iou']:.4f}")
        print("───────────────────────────────────────────────────────────\n")
