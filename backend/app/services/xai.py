"""
DeepTrace AI — GradCAM XAI (Explainability) Service

Implements Gradient-weighted Class Activation Mapping (GradCAM) to produce
saliency heatmaps showing WHICH regions of the face triggered the detection.

Why XAI matters for deepfake detection:
  - Legal/forensic use requires interpretable evidence, not just a probability
  - Helps researchers understand what artifacts the model is exploiting
  - Builds trust: "The model flagged the eye region as inconsistent"
  - Exposes bias: if model always flags dark skin tones, we can see that

Paper: "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)

Flow:
  1. Forward pass through model up to target layer
  2. Store activations (A^k) at target layer
  3. Backward pass of the classification score
  4. Compute gradient ∂y/∂A^k for each feature map
  5. Global-average-pool gradients → importance weights α^k
  6. Weighted sum of activations → raw CAM
  7. ReLU → resize to input dimensions → overlay on original image
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GradCAM:
    """
    GradCAM implementation compatible with EfficientNet, ViT, and XceptionNet.

    Usage:
        cam = GradCAM(model, target_layer_name="blocks.6.0")
        heatmap, overlay = cam(input_tensor, original_image_bgr)
        cam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer = self._find_layer(model, target_layer_name)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _find_layer(self, model, layer_name: str) -> torch.nn.Module:
        """Navigate model by dot-separated layer name."""
        parts = layer_name.split(".")
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        original_image_bgr: np.ndarray,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GradCAM heatmap and overlay.

        Args:
            input_tensor:       [1, 3, H, W] preprocessed input
            original_image_bgr: BGR image to overlay heatmap on
            alpha:              heatmap transparency (0=original, 1=heatmap only)

        Returns:
            heatmap:  [H, W] float32 in [0, 1] — raw CAM
            overlay:  [H, W, 3] uint8 BGR — colorized overlay on original image
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass on the positive class score
        self.model.zero_grad()
        score = output[0, 0]  # Binary: single output neuron
        score.backward()

        # GradCAM computation
        gradients = self._gradients    # [1, C, h, w]
        activations = self._activations  # [1, C, h, w]

        if gradients is None or activations is None:
            logger.warning("GradCAM: no gradients/activations — returning blank heatmap")
            h, w = original_image_bgr.shape[:2]
            return np.zeros((h, w)), original_image_bgr.copy()

        # Global average pool of gradients → importance weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)  # Only positive influences

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to original image dimensions
        h, w = original_image_bgr.shape[:2]
        heatmap = cv2.resize(cam, (w, h))

        # Colorize with JET colormap (blue=low, red=high attention)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Overlay on original image
        overlay = cv2.addWeighted(original_image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

        return heatmap, overlay


class SHAPExplainer:
    """
    Lightweight SHAP-inspired explainer for deepfake detection.
    Uses integrated gradients (faster than full SHAP, similar quality).

    Integrated Gradients:
        For each pixel, compute the average gradient along a straight path
        from a baseline (black image) to the actual input.
        This gives the contribution of each pixel to the final prediction.

    Returns a [H, W] importance map showing pixel-level contributions.
    """

    def __init__(self, model: torch.nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def explain(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute integrated gradients attribution map.

        Returns: [H, W] float32 importance map (positive = supports fake prediction)
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, self.n_steps).to(input_tensor.device)
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * (input_tensor - baseline)
        interpolated.requires_grad_(True)

        # Forward + backward for all interpolations
        self.model.eval()
        outputs = self.model(interpolated)  # [n_steps, 1]
        outputs.sum().backward()

        # Integrated gradients = mean gradient × (input - baseline)
        integrated_grads = interpolated.grad.mean(dim=0)  # [3, H, W]
        attributions = (integrated_grads * (input_tensor.squeeze(0) - baseline.squeeze(0)))

        # Collapse channels → importance map
        importance = attributions.abs().mean(dim=0).cpu().numpy()  # [H, W]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        return importance


def generate_xai_report(
    heatmap: np.ndarray,
    importance_map: Optional[np.ndarray],
    fake_probability: float,
    face_bbox: Tuple[int, int, int, int],
) -> dict:
    """
    Analyze heatmap to produce human-readable XAI findings.

    Divides face into anatomical regions and reports which regions
    contribute most to the fake prediction.

    Face region grid (normalized):
        Forehead:   top 25%
        Eyes:       25-45%
        Nose:       45-65%
        Mouth:      65-85%
        Chin/Jaw:   85-100%
        Left cheek: left 30%
        Right cheek: right 30%
    """
    h, w = heatmap.shape

    regions = {
        "forehead":    heatmap[:int(h*0.25), :],
        "eyes":        heatmap[int(h*0.25):int(h*0.45), :],
        "nose":        heatmap[int(h*0.45):int(h*0.65), int(w*0.3):int(w*0.7)],
        "mouth":       heatmap[int(h*0.65):int(h*0.85), int(w*0.2):int(w*0.8)],
        "jaw_chin":    heatmap[int(h*0.85):, :],
        "left_cheek":  heatmap[int(h*0.4):int(h*0.8), :int(w*0.3)],
        "right_cheek": heatmap[int(h*0.4):int(h*0.8), int(w*0.7):],
        "hair_border": np.concatenate([heatmap[:, :int(w*0.1)], heatmap[:, int(w*0.9):]], axis=1),
    }

    region_scores = {
        name: float(region.mean()) for name, region in regions.items()
    }

    # Sort by attention
    ranked = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

    highlights = []
    for region_name, score in ranked[:3]:
        if score > 0.4:
            label = region_name.replace("_", " ").title()
            if score > 0.7:
                highlights.append(f"HIGH attention in {label} region — strong manipulation signal")
            elif score > 0.5:
                highlights.append(f"Elevated attention in {label} — possible blending artifact")
            else:
                highlights.append(f"Moderate attention in {label} region")

    return {
        "region_scores": region_scores,
        "top_regions": [r[0] for r in ranked[:3]],
        "highlights": highlights,
        "overall_attention_score": float(heatmap.mean()),
        "attention_entropy": float(-(heatmap + 1e-8) * np.log(heatmap + 1e-8)).mean() if heatmap.size > 0 else 0,
    }
