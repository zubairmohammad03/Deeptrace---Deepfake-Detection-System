"""
DeepTrace AI — Configuration & Settings
All tunable constants live here. Override via .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "DeepTrace AI"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "*"]

    # ── File Limits ───────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 20
    MAX_VIDEO_SIZE_MB: int = 200
    MAX_AUDIO_SIZE_MB: int = 50
    UPLOAD_DIR: str = "uploads"
    REPORTS_DIR: str = "reports"

    # ── Model Paths ───────────────────────────────────────────────────────────
    # In production, set these to real checkpoint paths.
    # On first run without checkpoints, models use random weights (for demo).
    EFFICIENTNET_CHECKPOINT: str = os.getenv("EFFICIENTNET_CHECKPOINT", "")
    VIT_CHECKPOINT: str = os.getenv("VIT_CHECKPOINT", "")
    RAWNET2_CHECKPOINT: str = os.getenv("RAWNET2_CHECKPOINT", "")
    XCEPTION_CHECKPOINT: str = os.getenv("XCEPTION_CHECKPOINT", "")

    # ── Detection Thresholds ──────────────────────────────────────────────────
    FAKE_THRESHOLD: float = 0.5       # Probability above which = FAKE
    SUSPICIOUS_THRESHOLD: float = 0.35  # Probability above which = SUSPICIOUS

    # ── Video Processing ──────────────────────────────────────────────────────
    VIDEO_SAMPLE_FRAMES: int = 16     # Number of frames to extract per video
    VIDEO_TEMPORAL_STRIDE: int = 5    # Frame stride for temporal sampling
    FACE_CONFIDENCE_MIN: float = 0.85 # Min MTCNN confidence to accept a face

    # ── Image Processing ──────────────────────────────────────────────────────
    IMAGE_SIZE: int = 224             # Resize target for CNN input
    FACE_CROP_MARGIN: float = 0.3     # Extra margin around face crop

    # ── GradCAM ───────────────────────────────────────────────────────────────
    GRADCAM_LAYER: str = "blocks[-1]"  # Target layer for EfficientNet
    GRADCAM_ALPHA: float = 0.5         # Overlay transparency

    # ── Frequency Domain ──────────────────────────────────────────────────────
    DCT_BLOCK_SIZE: int = 8            # DCT block size (JPEG-style analysis)
    FFT_MAGNITUDE_THRESHOLD: float = 0.15

    # ── Adversarial Defense ───────────────────────────────────────────────────
    ADVERSARIAL_SMOOTHING: bool = True    # Apply input smoothing as defense
    ADVERSARIAL_JPEG_COMPRESSION: bool = True  # Apply JPEG pre-processing

settings = Settings()
