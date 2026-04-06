"""
SAM2-UNet for seagrass segmentation.

Hiera backbone + lightweight adapters + RFB projections + 3-stage U-Net decoder.
Loss: BCE + soft-IoU summed over 3 decoder outputs (deep supervision).

References:
  Xiong et al. (2024) SAM2-UNet. arXiv:2408.08870
  Huang S (2025) Advancing Seagrass Semantic Segmentation with SAM2 Models.
  Journal of High School Science 9(2):235–47.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export losses used by existing training scripts so imports stay consistent
from models.unet import CombinedLoss, DiceLoss  # noqa: F401


# ── Losses ─────────────────────────────────────────────────────────────────────

class IoULoss(nn.Module):
    """Soft IoU loss (used in the paper instead of Dice)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(logits.shape[0], -1)
        truth = targets.view(logits.shape[0], -1)
        intersection = (probs * truth).sum(1)
        union = probs.sum(1) + truth.sum(1) - intersection
        return (1.0 - (intersection + self.smooth) / (union + self.smooth)).mean()


class SAM2UNetLoss(nn.Module):
    """BCE + soft-IoU loss averaged over all decoder outputs (deep supervision)."""
    def __init__(self, bce_w: float = 1.0, iou_w: float = 1.0,
                 pos_weight: float | None = 7.0):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.iou = IoULoss()
        self.bce_w = bce_w
        self.iou_w = iou_w

    def forward(
        self,
        outputs: List[torch.Tensor],  # [S_1, S_2, S_3], each B×1×H_i×W_i
        targets: torch.Tensor,        # B×1×H×W  (full resolution)
    ) -> torch.Tensor:
        # Move pos_weight to correct device
        if self.bce.pos_weight is not None and \
                self.bce.pos_weight.device != targets.device:
            self.bce.pos_weight = self.bce.pos_weight.to(targets.device)

        total = torch.tensor(0.0, device=targets.device)
        for out in outputs:
            H, W = out.shape[2], out.shape[3]
            gt_i = F.interpolate(targets, size=(H, W),
                                 mode="bilinear", align_corners=False)
            total = total + self.bce_w * self.bce(out, gt_i) \
                          + self.iou_w * self.iou(out, gt_i)
        return total / len(outputs)


# ── Building blocks ────────────────────────────────────────────────────────────

class Adapter(nn.Module):
    """Bottleneck adapter (Linear down → GELU → Linear up → GELU) with residual."""
    def __init__(self, dim: int, ratio: float = 0.25):
        super().__init__()
        mid = max(1, int(dim * ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Linear(mid, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x may be (B, C, H, W) or (B, N, C)
        if x.dim() == 4:
            B, C, H, W = x.shape
            flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            out  = self.net(flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            out = self.net(x)
        return x + out


class RFB(nn.Module):
    """Receptive Field Block: parallel dilated convolutions, projects to out_ch channels."""
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        mid = out_ch // 4

        # parallel branches at dilation rates 1, 3, 5, 7
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, dilation=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=3, dilation=3),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=5, dilation=5),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=7, dilation=7),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.fuse(out)
        return F.relu(out + self.shortcut(x), inplace=True)


class DecoderBlock(nn.Module):
    """
    One U-Net decoder stage:  skip + x → (Conv-BN-ReLU) × 2 → bilinear×2
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims if skip is slightly different (padding artefacts)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Hiera backbone wrapper ─────────────────────────────────────────────────────

class HieraEncoder(nn.Module):
    """SAM2 Hiera backbone with per-stage trainable adapters. Returns [f4, f8, f16, f32]."""

    # Known Hiera configs (embed_dim for each of 4 stages)
    _CHANNELS = {
        "tiny":      [96,  192,  384,  768],
        "small":     [96,  192,  384,  768],
        "base_plus": [112, 224,  448,  896],
        "large":     [144, 288,  576, 1152],
    }

    def __init__(
        self,
        model_size: str = "tiny",
        pretrained_path: str | None = None,
        adapter_ratio: float = 0.25,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_size = model_size.lower()
        if self.model_size not in self._CHANNELS:
            raise ValueError(
                f"model_size must be one of {list(self._CHANNELS)}, got {model_size}"
            )
        self.stage_channels = self._CHANNELS[self.model_size]
        self.img_size = img_size

        self._build_backbone(pretrained_path)
        self.adapters = nn.ModuleList([
            Adapter(ch, ratio=adapter_ratio) for ch in self.stage_channels
        ])

    # ------------------------------------------------------------------
    def _build_backbone(self, pretrained_path: str | None):
        """Try sam2 package first; fall back to timm if not installed."""
        try:
            self._load_via_sam2(pretrained_path)
        except Exception as e1:
            try:
                self._load_via_timm()
            except Exception as e2:
                raise ImportError(
                    "Could not load Hiera backbone. Install sam2 with:\n"
                    "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                    f"SAM2 error : {e1}\n"
                    f"timm error : {e2}"
                ) from e2

    def _load_via_sam2(self, pretrained_path: str | None):
        """Load Hiera via the sam2 package."""
        from sam2.modeling.backbones.hieradet import Hiera  # type: ignore

        size_cfgs = {
            "tiny":      dict(embed_dim=96,  num_heads=1,  stages=[1,2,7,2],
                              global_att_blocks=[5,7,9], window_pos_embed_bkg_spatial_size=[7,7]),
            "small":     dict(embed_dim=96,  num_heads=1,  stages=[1,2,11,2],
                              global_att_blocks=[7,10,13], window_pos_embed_bkg_spatial_size=[7,7]),
            "base_plus": dict(embed_dim=112, num_heads=2,  stages=[2,3,16,3],
                              global_att_blocks=[12,16,20], window_pos_embed_bkg_spatial_size=[7,7]),
            "large":     dict(embed_dim=144, num_heads=2,  stages=[2,6,36,4],
                              global_att_blocks=[23,33,43], window_pos_embed_bkg_spatial_size=[7,7]),
        }
        cfg = size_cfgs[self.model_size]
        self.backbone = Hiera(**cfg)
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("model", ckpt)
            # Keep only backbone keys
            bb_state = {k.replace("image_encoder.trunk.", ""): v
                        for k, v in state.items()
                        if k.startswith("image_encoder.trunk.")}
            if bb_state:
                missing, unexpected = self.backbone.load_state_dict(bb_state, strict=False)
                if missing:
                    print(f"  [HieraEncoder] Missing keys ({len(missing)}): {missing[:5]}…")
            else:
                # fallback: try the full checkpoint as backbone weights
                self.backbone.load_state_dict(state, strict=False)
        self._forward_fn = "sam2"

    def _load_via_timm(self):
        """Fallback: load Hiera via timm with pretrained ImageNet weights."""
        import timm  # type: ignore

        _timm_names = {
            "tiny":      "hiera_tiny_224",
            "small":     "hiera_small_224",
            "base_plus": "hiera_base_plus_224",
            "large":     "hiera_large_224",
        }
        model_name = _timm_names[self.model_size]
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, features_only=True,
            img_size=self.img_size,
        )
        self._forward_fn = "timm"
        fi = self.backbone.feature_info.info
        self.stage_channels = [f["num_chs"] for f in fi][-4:]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self._forward_fn == "sam2":
            return self._forward_sam2(x)
        else:
            return self._forward_timm(x)

    def _forward_sam2(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Hiera can return a dict or list depending on version
        out = self.backbone(x)
        if isinstance(out, dict):
            feats = out.get("backbone_fpn", out.get("multiscale_features", []))
        elif isinstance(out, (list, tuple)):
            feats = list(out)
        else:
            raise RuntimeError(f"Unexpected Hiera output type: {type(out)}")

        feats = feats[-4:]
        feats = sorted(feats, key=lambda f: -f.shape[2])  # largest H first (stride 4)
        feats = [self.adapters[i](f) for i, f in enumerate(feats)]
        return feats

    def _forward_timm(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Fallback: use timm feature extractor (features_only=True)."""
        feats = self.backbone(x)
        feats = feats[-4:]
        feats = [self.adapters[i](f) for i, f in enumerate(feats)]
        return feats


# ── Full SAM2-UNet ─────────────────────────────────────────────────────────────

class SAM2UNet(nn.Module):
    """
    SAM2-UNet: Hiera encoder + RFB projections + 3-stage U-Net decoder.

    Input: B×3×H×W (paper default 352×352; any multiple of 32 works).
    Training returns (S1, S2, S3) for deep supervision.
    Inference returns the finest logit map (S3) upsampled to input resolution.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        pretrained_path: str | None = None,
        rfb_channels: int = 64,
        adapter_ratio: float = 0.25,
        freeze_backbone: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_size    = model_size
        self.rfb_ch        = rfb_channels

        # ── Encoder ───────────────────────────────────────────────────────────────────
        self.encoder = HieraEncoder(
            model_size=model_size,
            pretrained_path=pretrained_path,
            adapter_ratio=adapter_ratio,
            img_size=img_size,
        )
        stage_ch = self.encoder.stage_channels   # [C0, C1, C2, C3]

        # freeze backbone; adapters stay trainable
        if freeze_backbone:
            for name, p in self.encoder.backbone.named_parameters():
                p.requires_grad = False
            for p in self.encoder.adapters.parameters():
                p.requires_grad = True

        # ── RFB projections ───────────────────────────────────────────────────
        R = rfb_channels
        self.rfb4  = RFB(stage_ch[0], R)   # stride-4  features
        self.rfb8  = RFB(stage_ch[1], R)   # stride-8
        self.rfb16 = RFB(stage_ch[2], R)   # stride-16
        self.rfb32 = RFB(stage_ch[3], R)   # stride-32

        # ── Decoder ──────────────────────────────────────────────────────────
        # in_ch = upsampled (R) + skip (R) = 2R
        self.dec1 = DecoderBlock(R + R, R)   # 11×11 → 22×22
        self.dec2 = DecoderBlock(R + R, R)   # 22×22 → 44×44
        self.dec3 = DecoderBlock(R + R, R)   # 44×44 → 88×88

        # ── Prediction heads (one per decoder stage) ─────────────────────────
        self.head1 = nn.Conv2d(R, 1, 1)
        self.head2 = nn.Conv2d(R, 1, 1)
        self.head3 = nn.Conv2d(R, 1, 1)

        self._init_decoder()

    def _init_decoder(self):
        for m in [self.rfb4, self.rfb8, self.rfb16, self.rfb32,
                  self.dec1, self.dec2, self.dec3,
                  self.head1, self.head2, self.head3]:
            for p in m.modules():
                if isinstance(p, nn.Conv2d):
                    nn.init.kaiming_normal_(p.weight, mode="fan_out")
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)
                elif isinstance(p, nn.BatchNorm2d):
                    nn.init.ones_(p.weight)
                    nn.init.zeros_(p.bias)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Training:  returns (S1, S2, S3) for deep supervision.
        Eval/inf:  returns only the full-resolution logit map.
        """
        H, W = x.shape[2], x.shape[3]

        # Encoder → [f4, f8, f16, f32]
        f4, f8, f16, f32 = self.encoder(x)

        # RFB channel reduction
        r4  = self.rfb4 (f4)
        r8  = self.rfb8 (f8)
        r16 = self.rfb16(f16)
        r32 = self.rfb32(f32)

        # Decoder with skip connections
        d1 = self.dec1(r32, r16)           # 11→22 (approx)
        d2 = self.dec2(d1,  r8)            # 22→44
        d3 = self.dec3(d2,  r4)            # 44→88

        # Prediction heads (raw logits, no sigmoid)
        s1 = self.head1(d1)
        s2 = self.head2(d2)
        s3 = self.head3(d3)

        # upsample all heads to input resolution
        s1 = F.interpolate(s1, size=(H, W), mode="bilinear", align_corners=False)
        s2 = F.interpolate(s2, size=(H, W), mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=(H, W), mode="bilinear", align_corners=False)

        if self.training:
            return s1, s2, s3
        else:
            return s3

    # ------------------------------------------------------------------
    def unfreeze_backbone(self):
        """Call after warmup epochs to fine-tune the full backbone."""
        for p in self.encoder.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        for p in self.encoder.backbone.parameters():
            p.requires_grad = False
        for p in self.encoder.adapters.parameters():
            p.requires_grad = True


# ── Factories + helpers ────────────────────────────────────────────────────────

def build_sam2unet(
    model_size: str = "tiny",
    pretrained_path: str | None = None,
    rfb_channels: int = 64,
    adapter_ratio: float = 0.25,
    freeze_backbone: bool = True,
    img_size: int = 224,
) -> SAM2UNet:
    """Build a SAM2UNet. Pass pretrained_path=None for random Hiera weights."""
    return SAM2UNet(
        model_size=model_size,
        pretrained_path=pretrained_path,
        rfb_channels=rfb_channels,
        adapter_ratio=adapter_ratio,
        freeze_backbone=freeze_backbone,
        img_size=img_size,
    )


def count_parameters(model: nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
