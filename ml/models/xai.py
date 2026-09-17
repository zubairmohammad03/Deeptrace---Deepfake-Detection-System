# ml/models/xai.py
"""
Explainable AI (XAI) utilities for DeepTrace.

Features:
  - GRAD-CAM: gradient-weighted activation maps
  - SHAP integration
  - Overlay visualization
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional


class GradCAM:
    """
    GRAD-CAM implementation for CNN models.
    Highlights which image regions contributed most to the deepfake decision.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.conv_head)
        heatmap = cam(input_tensor, class_idx=1)  # class 1 = fake
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GRAD-CAM heatmap.

        Returns:
            heatmap: numpy array (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute weighted activation map
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: HxWx3 uint8 BGR image
            heatmap: HxW float [0,1]
            alpha: blend weight

        Returns:
            blended: HxWx3 uint8
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
        return blended


class IntegratedGradients:
    """
    Integrated Gradients — more faithful attribution than GRAD-CAM.
    Slower but gives pixel-level importance scores.
    """

    def __init__(self, model: torch.nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        baseline = torch.zeros_like(input_tensor)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, self.steps).to(input_tensor.device)
        interpolated = [baseline + a * (input_tensor - baseline) for a in alphas]
        interpolated = torch.cat(interpolated, dim=0)
        interpolated.requires_grad_(True)

        outputs = self.model(interpolated)
        scores = outputs[:, class_idx].sum()
        scores.backward()

        grads = interpolated.grad  # (steps, C, H, W)
        avg_grads = grads.mean(dim=0)  # (C, H, W)
        integrated = (input_tensor.squeeze(0) - baseline.squeeze(0)) * avg_grads

        # Aggregate channels
        attribution = integrated.abs().sum(dim=0).cpu().detach().numpy()
        attribution = attribution / (attribution.max() + 1e-8)
        return attribution
