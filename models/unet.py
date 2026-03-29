"""
U-Net model helpers for seagrass segmentation.

This project uses segmentation-models-pytorch with a ResNet34 encoder
initialised from ImageNet weights.
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice loss operating on raw logits.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(logits.shape[0], -1)
        truth = targets.view(logits.shape[0], -1)
        numerator = 2.0 * (probs * truth).sum(1) + self.smooth
        denominator = probs.sum(1) + truth.sum(1) + self.smooth
        return (1.0 - numerator / denominator).mean()


class CombinedLoss(nn.Module):
    """
    Dice + BCE loss for sparse foreground segmentation.
    """

    def __init__(self, dice_w: float = 0.5, bce_w: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + self.bce_w * self.bce(logits, targets)


def build_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """
    Build a segmentation-models-pytorch U-Net with a ResNet encoder.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            "segmentation-models-pytorch is required for the ResNet34 ImageNet encoder. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from e

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
