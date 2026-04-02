"""
ViT-B/16 segmentation model for seagrass detection.

Based on SeagrassFinder (Elsässer et al., 2024/2025, Ecological Informatics).
Architecture: ViT-B/16 encoder + lightweight MLP decode head (SegFormer-inspired).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Losses (re-exported so train_unet.py only needs to import from one place)
# ---------------------------------------------------------------------------
from models.unet import CombinedLoss, DiceLoss  # noqa: F401


# ---------------------------------------------------------------------------
# Option A: ViT-B/16 encoder + MLP decode head
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    Wraps timm's ViT-B/16 pretrained on ImageNet-21k (or ImageNet-1k) and
    exposes intermediate patch tokens as spatial feature maps for decoding.

    The ViT does not produce multi-scale features natively, so we take the
    final sequence of patch tokens and reshape them into a 2-D feature map.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        img_size: int = 512,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for the ViT encoder. Install with: pip install timm"
            ) from e

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,        # remove classification head
        )
        self.embed_dim = self.vit.embed_dim   # 768 for ViT-B
        # Number of patches per side
        patch_size  = self.vit.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size   # e.g. 512//16 = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B×3×H×W
        Returns: B×C×h×w  spatial feature map (h = w = H/patch_size)
        """
        # forward_features returns (B, N+1, C) where N = num_patches, +1 = cls
        tokens = self.vit.forward_features(x)  # (B, N+1, C)
        patch_tokens = tokens[:, 1:, :]        # drop [CLS], (B, N, C)
        B, N, C = patch_tokens.shape
        h = w = int(N ** 0.5)
        # reshape to spatial map
        feat = patch_tokens.permute(0, 2, 1).view(B, C, h, w)  # B×C×h×w
        return feat


class MLPDecodeHead(nn.Module):
    """
    Simple MLP + bilinear upsample decode head for ViT features.
    Progressively doubles resolution until reaching the input size.

    decode_channels controls intermediate feature width.
    """

    def __init__(
        self,
        in_channels: int = 768,
        decode_channels: int = 256,
        out_channels: int = 1,
        input_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()
        self.upsample_factor = patch_size  # e.g. 16× to get back to full res

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, decode_channels, kernel_size=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
        )
        # Two refinement blocks at intermediate scale
        self.refine = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decode_channels, decode_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decode_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decode_channels // 2, out_channels, kernel_size=1)

    def forward(self, feat: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        """
        feat: B×C×h×w  (patch-level feature map)
        Returns: B×out_channels×H×W logits
        """
        x = self.proj(feat)
        # Upsample to full image resolution
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=self.upsample_factor,
                              mode="bilinear", align_corners=False)
        x = self.refine(x)
        return self.head(x)


class ViTSegNet(nn.Module):
    """
    Full segmentation network: ViT encoder + MLP decode head.
    Drop-in replacement for the smp U-Net returned by build_unet().
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 512,
        decode_channels: int = 256,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ViTSegNet currently supports only in_channels=3.")

        self.encoder = ViTEncoder(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
        )
        self.decoder = MLPDecodeHead(
            in_channels=self.encoder.embed_dim,
            decode_channels=decode_channels,
            out_channels=out_channels,
            input_size=img_size,
            patch_size=self.encoder.patch_size,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        feat   = self.encoder(x)                    # B×C×h×w
        logits = self.decoder(feat, target_size=(H, W))  # B×1×H×W
        return logits


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------

def build_vit_seg(
    in_channels: int = 3,
    out_channels: int = 1,
    img_size: int = 512,
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    decode_channels: int = 256,
) -> nn.Module:
    """
    Primary factory function — returns a ViTSegNet.
    Used by scripts/train_vit.py and scripts/infer_vit.py.
    """
    return ViTSegNet(
        model_name=model_name,
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        decode_channels=decode_channels,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

