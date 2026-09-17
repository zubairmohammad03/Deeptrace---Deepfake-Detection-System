# ml/models/deepfake_model.py
"""
DeepTrace Deepfake Detection Model

Architecture:
  - EfficientNet-B4 backbone (pretrained on ImageNet)
  - Multi-head classification with dropout
  - Optional: dual-stream with frequency branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DeepTraceModel(nn.Module):
    """
    EfficientNet-B4 based deepfake detector.
    Achieves strong generalization across multiple deepfake types.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
        num_classes: int = 2,
    ):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # remove head
            global_pool="avg"
        )
        feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

        # Grad-CAM target layer
        self.target_layer = self.backbone.conv_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_feature_maps(self, x: torch.Tensor):
        """Return intermediate feature maps for XAI visualization."""
        return self.backbone.forward_features(x)


class FrequencyBranch(nn.Module):
    """
    Auxiliary branch that analyzes frequency domain (DCT) features.
    Catches GAN fingerprints invisible in pixel space.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_features)

    def dct_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate DCT via learned convolution on frequency channels."""
        # Simple high-pass filter to emphasize frequency artifacts
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.dct_transform(x)
        feat = self.conv(freq).squeeze(-1).squeeze(-1)
        return self.fc(feat)


class DeepTraceDualStream(nn.Module):
    """
    Dual-stream architecture combining spatial + frequency analysis.
    More powerful but slower than single-stream.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True, dropout: float = 0.4):
        super().__init__()

        # Spatial stream
        self.spatial = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        spatial_dim = self.spatial.num_features

        # Frequency stream
        self.freq_branch = FrequencyBranch(out_features=256)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.spatial(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion(combined)


def build_model(
    architecture: str = "single",
    backbone: str = "efficientnet_b4",
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to build the detection model.

    Args:
        architecture: "single" | "dual"
        backbone: any timm model name
        pretrained: use ImageNet weights
    """
    if architecture == "dual":
        return DeepTraceDualStream(backbone=backbone, pretrained=pretrained)
    return DeepTraceModel(backbone=backbone, pretrained=pretrained)


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M"
