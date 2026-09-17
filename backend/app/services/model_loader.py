# app/services/model_loader.py
"""
Loads the trained EfficientNet-B4 model once at startup.
If no model found, falls back to OpenRouter API automatically.
"""
import os
import torch
import torch.nn as nn

# Try importing timm — only available if torch is installed
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from app.config import settings

# Global state
_model = None
_device = None
_model_loaded = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def load_model():
    """
    Attempt to load trained EfficientNet weights.
    Called once at app startup.
    Returns (model, device, is_loaded)
    """
    global _model, _device, _model_loaded

    # Check if PyTorch + timm are available
    if not TIMM_AVAILABLE:
        print("[Model] timm not installed — skipping model load, using OpenRouter API")
        _model_loaded = False
        return None, None, False

    _device = get_device()
    model_path = settings.model_path

    if not os.path.exists(model_path):
        print(f"[Model] No weights found at '{model_path}'")
        print("[Model] → Train on Colab and place best_model.pt in backend/checkpoints/")
        print("[Model] → Falling back to OpenRouter API for all requests")
        _model_loaded = False
        return None, _device, False

    try:
        _model = timm.create_model(
            settings.model_backbone,
            pretrained=False,
            num_classes=2
        )

        checkpoint = torch.load(model_path, map_location=_device)

        # Handle both raw state_dict and full checkpoint dict
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            _model.load_state_dict(checkpoint["model_state"])
            epoch = checkpoint.get("epoch", "?")
            auc   = checkpoint.get("val_auc", "?")
            print(f"[Model] ✓ Loaded checkpoint — epoch={epoch}, val_auc={auc}")
        else:
            _model.load_state_dict(checkpoint)
            print(f"[Model] ✓ Loaded weights from {model_path}")

        _model = _model.to(_device)
        _model.eval()
        _model_loaded = True
        print(f"[Model] ✓ EfficientNet-B4 ready on {_device}")
        return _model, _device, True

    except Exception as e:
        print(f"[Model] ✗ Failed to load model: {e}")
        print("[Model] → Falling back to OpenRouter API")
        _model_loaded = False
        return None, _device, False


def get_model():
    """Return (model, device, is_loaded) — call after load_model()."""
    return _model, _device, _model_loaded
