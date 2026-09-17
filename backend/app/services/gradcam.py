# app/services/gradcam.py
"""
GRAD-CAM: Gradient-weighted Class Activation Mapping.
Generates a real heatmap overlay showing which pixels triggered the deepfake decision.
"""
import base64
import io
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._hooks = []
        self._register()

    def _register(self):
        def fwd(module, inp, out):
            self._activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def generate(self, tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """Generate CAM for class_idx (0 = fake). Returns H×W float array [0,1]."""
        self.model.eval()
        inp = tensor.unsqueeze(0).clone().requires_grad_(True)

        out = self.model(inp)
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def get_target_layer(model):
    """Get the last convolutional layer for GRAD-CAM from EfficientNet."""
    for name in ["conv_head", "features"]:
        layer = getattr(model, name, None)
        if layer is not None:
            return layer
    # Fallback: last Conv2d in the model
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def make_heatmap_overlay(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend JET colormap heatmap over original RGB image."""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (image_np * (1 - alpha) + colored_rgb * alpha).astype(np.uint8)
    return overlay


def np_to_base64(image_np: np.ndarray) -> str:
    """Convert RGB numpy array to base64 JPEG string."""
    img = Image.fromarray(image_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()
