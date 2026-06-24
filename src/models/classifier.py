import torch
import torch.nn as nn
import timm

from src.label_map import NUM_CLASSES
from src.models.aggregations import (
    GlobalAveragePooling,
    AttentionPooling,
    GatedAttentionPooling,
)


ENCODER_ZOO = {
    # CNN baseline
    "resnet50": "resnet50.a1_in1k",

    # Efficient CNN
    "efficientnet_b3": "tf_efficientnet_b3.ns_jft_in1k",

    # Modern CNN
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",

    # Vision Transformer
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",

    # CLIP vision encoder, ImageNet fine-tuned version
    "clip_vit_b32_cls": "vit_base_patch32_clip_224.openai_ft_in1k",
}


class SidewalkClassifier(nn.Module):
    """
    Multi-label classifier for sidewalk encroachment detection.

    Input:
        x: image tensor [B, 3, H, W]

    Output:
        logits: [B, num_classes]

    Loss:
        BCEWithLogitsLoss / Weighted BCE / Focal Loss / ASL
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        aggregation: str = "gap",
    ):
        super().__init__()

        if encoder_name not in ENCODER_ZOO:
            raise ValueError(
                f"Unknown encoder_name: {encoder_name}. "
                f"Available encoders: {list(ENCODER_ZOO.keys())}"
            )

        self.encoder_name = encoder_name
        self.model_name = ENCODER_ZOO[encoder_name]
        self.aggregation_type = aggregation

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        in_features = self._infer_feature_dim()

        if aggregation == "gap":
            self.aggregation = GlobalAveragePooling()

        elif aggregation == "attention":
            self.aggregation = AttentionPooling(in_features)

        elif aggregation == "gated_attention":
            self.aggregation = GatedAttentionPooling(in_features)

        elif aggregation == "cls_token":
            self.aggregation = None

        else:
            raise ValueError(
                f"Unknown aggregation: {aggregation}. "
                f"Available: gap, attention, gated_attention, cls_token"
            )

        self.head = nn.Linear(in_features, num_classes)

    def _infer_feature_dim(self):
        self.backbone.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)

        if features.dim() == 4:
            # CNN-like feature map: [B, C, H, W]
            return features.shape[1]

        if features.dim() == 3:
            # Transformer tokens: [B, N, C]
            return features.shape[2]

        if features.dim() == 2:
            # Already pooled feature: [B, C]
            return features.shape[1]

        raise ValueError(f"Unsupported feature shape: {features.shape}")

    def forward(self, x):
        features = self.backbone(x)

        if self.aggregation_type == "cls_token":
            if features.dim() == 3:
                pooled = features[:, 0]
            elif features.dim() == 4:
                pooled = features.mean(dim=(2, 3))
            elif features.dim() == 2:
                pooled = features
            else:
                raise ValueError(f"Unsupported feature shape: {features.shape}")
        else:
            pooled = self.aggregation(features)

        logits = self.head(pooled)
        return logits


def create_model(
    encoder_name: str = "resnet50",
    aggregation: str = "gap",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
):
    return SidewalkClassifier(
        encoder_name=encoder_name,
        num_classes=num_classes,
        pretrained=pretrained,
        aggregation=aggregation,
    )