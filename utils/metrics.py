"""
Segmentation metrics: pixel accuracy, precision, recall, F1, IoU.
Jeon et al. (2021) Ecological Informatics 66, 101430. Eqs. (3)–(7).
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import torch


# ── Per-image metrics ────────────────────────────────────────────────────────────────

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
    """Compute pixel-level metrics for one image."""
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


# ── Batch IoU (training loop) ───────────────────────────────────────────────────────

def batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Fast scalar IoU over a batch (stays on GPU)."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    return iou.mean()


# ── Aggregate results ────────────────────────────────────────────────────────────────

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
