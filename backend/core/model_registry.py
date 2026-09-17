"""
DeepTrace AI — Model Registry
Central manager for all ML models. Loads them once at startup and
provides thread-safe access throughout the application lifecycle.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from core.config import settings

logger = logging.getLogger("deeptrace.models")


class ModelRegistry:
    """
    Singleton-like registry that loads all ML models at startup.
    Models are loaded to GPU if available, else CPU.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Using device: {self.device}")

        # Model containers
        self.efficientnet: Optional[nn.Module] = None
        self.vit: Optional[nn.Module] = None
        self.xception: Optional[nn.Module] = None
        self.rawnet: Optional[nn.Module] = None
        self.face_detector = None
        self._loaded = False

    async def load_all_models(self):
        """Load all models. Falls back to lightweight stubs if weights not found."""
        try:
            self._load_visual_models()
            self._load_face_detector()
            self._loaded = True
            logger.info("✅ Model registry ready")
        except Exception as e:
            logger.warning(f"⚠️  Could not load pretrained weights: {e}. Using stub models.")
            self._load_stub_models()
            self._loaded = True

    def _load_visual_models(self):
        """Load EfficientNet-B4 and ViT for visual deepfake detection."""
        from models.efficientnet_detector import EfficientNetDetector
        from models.vit_detector import ViTDetector

        # Primary location: checkpoints/best_model.pt (trained output)
        # Fallback: models/weights/<EFFICIENTNET_WEIGHTS> (legacy path)
        base_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = base_dir / "checkpoints" / "best_model.pt"
        legacy_path = settings.MODEL_DIR / settings.EFFICIENTNET_WEIGHTS
        eff_path = checkpoint_path if checkpoint_path.exists() else legacy_path

        logger.info(f"EfficientNet path: {eff_path} | exists: {eff_path.exists()}")
        if eff_path.exists():
            try:
                self.efficientnet = EfficientNetDetector.load(eff_path, self.device)
                logger.info("✅ EfficientNet-B4 loaded from checkpoint")
            except Exception as e:
                logger.error(f"MODEL LOAD FAILED: {type(e).__name__}: {e}")
                raise
        else:
            logger.info(f"📦 EfficientNet weights not found at {eff_path} — demo mode")
            self.efficientnet = EfficientNetDetector(pretrained=False).to(self.device)

        vit_path = settings.MODEL_DIR / settings.VIT_WEIGHTS
        if vit_path.exists():
            self.vit = ViTDetector.load(vit_path, self.device)
            logger.info("✅ ViT loaded")
        else:
            logger.info("📦 ViT weights not found — using randomly initialized model (demo mode)")
            self.vit = ViTDetector(pretrained=False).to(self.device)

    def _load_face_detector(self):
        """Load MTCNN face detector."""
        try:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(
                keep_all=True,
                device=self.device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
            )
            logger.info("✅ MTCNN face detector loaded")
        except ImportError:
            logger.warning("⚠️  facenet_pytorch not installed. Using OpenCV face detection fallback.")
            self.face_detector = None

    def _load_stub_models(self):
        """Lightweight stub models for demo/testing without GPU."""
        from models.stub_models import StubVisualDetector
        self.efficientnet = StubVisualDetector()
        self.vit = StubVisualDetector()
        logger.info("🔧 Stub models loaded (for demo purposes)")

    async def cleanup(self):
        """Free GPU memory on shutdown."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Model registry cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def get_info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": {
                "efficientnet": self.efficientnet is not None,
                "vit": self.vit is not None,
                "face_detector": self.face_detector is not None,
                "rawnet": self.rawnet is not None,
            }
        }
