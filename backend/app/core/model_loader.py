"""
DeepTrace AI — Model Loader
Loads all ML models once at startup and keeps them in memory.
Each model is wrapped so it degrades gracefully when no checkpoint exists.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    efficientnet: Optional[torch.nn.Module] = None
    vit: Optional[torch.nn.Module] = None
    xception: Optional[torch.nn.Module] = None
    rawnet2: Optional[torch.nn.Module] = None
    face_detector: Optional[object] = None        # MTCNN
    device: str = "cpu"


class ModelLoader:
    """
    Singleton-style loader. Call load_all() once at startup.
    Models are stored as instance attributes and injected into services
    via FastAPI's request.app.state.model_loader.
    """

    def __init__(self):
        self.models = LoadedModels()
        self.models.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.models.device}")

    async def load_all(self):
        self._load_face_detector()
        self._load_efficientnet()
        self._load_vit()
        self._load_xception()
        self._load_rawnet2()

    async def unload_all(self):
        self.models = LoadedModels()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Individual Loaders ────────────────────────────────────────────────────

    def _load_face_detector(self):
        """MTCNN for face detection and alignment."""
        try:
            from facenet_pytorch import MTCNN
            self.models.face_detector = MTCNN(
                image_size=settings.IMAGE_SIZE,
                margin=int(settings.IMAGE_SIZE * settings.FACE_CROP_MARGIN),
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                keep_all=True,
                device=self.models.device,
            )
            logger.info("✅ MTCNN face detector loaded")
        except ImportError:
            logger.warning("⚠️  facenet_pytorch not installed — using fallback face detector")
            self.models.face_detector = None

    def _load_efficientnet(self):
        """
        EfficientNet-B4 fine-tuned for deepfake detection.
        Architecture: EfficientNet-B4 backbone → GlobalAvgPool → Dropout(0.3) → Dense(1)
        Input:  [B, 3, 224, 224] face-crop tensor (normalized)
        Output: [B, 1] sigmoid probability (1 = fake)
        """
        try:
            import timm
            model = timm.create_model(
                "efficientnet_b4",
                pretrained=False,
                num_classes=1,
                drop_rate=0.3,
            )
            if settings.EFFICIENTNET_CHECKPOINT:
                state = torch.load(settings.EFFICIENTNET_CHECKPOINT, map_location=self.models.device)
                model.load_state_dict(state)
                logger.info("✅ EfficientNet-B4 checkpoint loaded")
            else:
                logger.warning("⚠️  EfficientNet-B4: no checkpoint — using untrained weights (set EFFICIENTNET_CHECKPOINT in .env)")

            model = model.to(self.models.device).eval()
            self.models.efficientnet = model
        except ImportError:
            logger.error("❌ timm not installed — pip install timm")

    def _load_vit(self):
        """
        Vision Transformer (ViT-B/16) for face-reenactment detection.
        Uses patch-level attention to catch subtle reenactment artifacts.
        Input:  [B, 3, 224, 224]
        Output: [B, 1] sigmoid probability
        """
        try:
            import timm
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=1,
            )
            if settings.VIT_CHECKPOINT:
                state = torch.load(settings.VIT_CHECKPOINT, map_location=self.models.device)
                model.load_state_dict(state)
                logger.info("✅ ViT-B/16 checkpoint loaded")
            else:
                logger.warning("⚠️  ViT-B/16: no checkpoint — using untrained weights")

            model = model.to(self.models.device).eval()
            self.models.vit = model
        except ImportError:
            logger.error("❌ timm not installed")

    def _load_xception(self):
        """
        XceptionNet — strong baseline for FaceForensics++ style detection.
        Input:  [B, 3, 299, 299]
        Output: [B, 1]
        """
        try:
            import timm
            model = timm.create_model(
                "xception",
                pretrained=False,
                num_classes=1,
            )
            if settings.XCEPTION_CHECKPOINT:
                state = torch.load(settings.XCEPTION_CHECKPOINT, map_location=self.models.device)
                model.load_state_dict(state)
                logger.info("✅ XceptionNet checkpoint loaded")
            else:
                logger.warning("⚠️  XceptionNet: no checkpoint — using untrained weights")

            model = model.to(self.models.device).eval()
            self.models.xception = model
        except ImportError:
            logger.error("❌ timm not installed")

    def _load_rawnet2(self):
        """
        RawNet2 for audio deepfake / voice cloning detection.
        Operates directly on raw waveform without hand-crafted features.
        Input:  [B, 1, T] raw waveform at 16kHz
        Output: [B, 1] sigmoid probability (1 = fake voice)
        """
        try:
            from app.models.rawnet2 import RawNet2
            model = RawNet2()
            if settings.RAWNET2_CHECKPOINT:
                state = torch.load(settings.RAWNET2_CHECKPOINT, map_location=self.models.device)
                model.load_state_dict(state)
                logger.info("✅ RawNet2 checkpoint loaded")
            else:
                logger.warning("⚠️  RawNet2: no checkpoint — using untrained weights")

            model = model.to(self.models.device).eval()
            self.models.rawnet2 = model
        except Exception as e:
            logger.warning(f"⚠️  RawNet2 not loaded: {e}")
