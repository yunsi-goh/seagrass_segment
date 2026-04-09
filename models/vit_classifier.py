"""
ViT-B/16 image classifier for seagrass presence/absence detection.

Reuses the ViTEncoder from the segmentation pipeline but replaces the
decode head with a single linear classifier on top of the [CLS] token.

Output: a single logit (positive = seagrass present).
"""

import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    """ViT-B/16 encoder + linear classifier head."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        img_size: int = 224,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required. Install with: pip install timm"
            ) from e

        # Load ViT with its built-in classification head removed
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,   # returns [CLS] token embedding, shape (B, embed_dim)
        )
        embed_dim = self.encoder.embed_dim  # 768 for ViT-B

        # Lightweight head: dropout → linear
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 1),
        )

        # Initialise head
        nn.init.normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns B×1 raw logits."""
        cls_token = self.encoder(x)   # (B, embed_dim) — [CLS] embedding
        return self.head(cls_token)   # (B, 1)


# ── Factory + helpers ─────────────────────────────────────────────────────────

def build_vit_classifier(
    model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained: bool = True,
    img_size: int = 224,
    dropout: float = 0.2,
) -> ViTClassifier:
    return ViTClassifier(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
