"""
DeepTrace AI — EfficientNet-B4 Deepfake Detector
Fine-tuned EfficientNet-B4 backbone for spatial artifact detection.

Architecture:
  - Backbone: EfficientNet-B4 (pretrained on ImageNet)
  - Custom head: Global Average Pool → Dropout(0.3) → FC(256) → FC(1)
  - Training datasets: FaceForensics++, DFDC, Celeb-DF v2
  - Input: 224×224 RGB face crop
  - Output: probability score [0, 1] (1 = fake)

Key Papers:
  - EfficientNet: "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)
  - Face forgery: "FaceForensics++: Learning to Detect Manipulated Facial Images" (Rössler et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict


class EfficientNetDetector(nn.Module):
    """
    Binary deepfake classifier built on EfficientNet-B4.
    Returns both a prediction score and intermediate feature maps
    used by the GRAD-CAM explainability module.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b4(weights=weights)
            self.features = backbone.features        # Convolutional feature extractor
            self.avgpool = backbone.avgpool
            in_features = backbone.classifier[1].in_features
        except Exception:
            # Fallback: simple CNN if torchvision not available
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = 256

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, 1),   # Binary: real vs fake
        )

        # Store last feature map for GRAD-CAM
        self._feature_maps = None
        self._gradients = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) normalized face crops

        Returns:
            (logits, features): raw scores and feature maps for XAI
        """
        features = self.features(x)

        # Hook for GRAD-CAM gradient computation
        if features.requires_grad:
            features.register_hook(self._save_gradient)
        self._feature_maps = features

        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fake probability [0,1] for each image in batch."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    def _save_gradient(self, grad):
        self._gradients = grad

    def get_gradcam(self, x: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """
        Compute GRAD-CAM heatmap for the given input.
        Highlights which face regions contributed most to fake/real prediction.

        Algorithm:
          1. Forward pass → get feature maps at last conv layer
          2. Backward pass → get gradients w.r.t. feature maps
          3. Global-average-pool gradients → neuron importance weights
          4. Weighted sum of feature maps → raw heatmap
          5. ReLU + normalize to [0, 1]

        Returns:
            heatmap: (H, W) tensor normalized to [0, 1]
        """
        self.eval()
        x.requires_grad_(True)

        logits, features = self.forward(x)
        score = logits[0, target_class] if logits.shape[-1] > 1 else logits[0, 0]

        self.zero_grad()
        score.backward(retain_graph=True)

        if self._gradients is None or self._feature_maps is None:
            return torch.zeros(224, 224)

        # Average gradients spatially → channel importance weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * self._feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input resolution and normalize
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach()

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "EfficientNetDetector":
        model = cls(pretrained=False)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(device)
        model.eval()
        return model
