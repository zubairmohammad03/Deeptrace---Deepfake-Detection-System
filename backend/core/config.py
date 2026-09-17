"""
DeepTrace AI — Configuration
All environment variables and constants are centralized here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow")
    
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "DeepTrace AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "*"]

    # ── Paths ─────────────────────────────────────────────────────────────────
    MODEL_DIR: Path = BASE_DIR / "models" / "weights"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    TEMP_DIR: Path = BASE_DIR / "temp"

    # ── Model Weights (downloaded from HuggingFace / local) ──────────────────
    # EfficientNet fine-tuned on FaceForensics++ + DFDC + Celeb-DF
    EFFICIENTNET_WEIGHTS: str = "efficientnet_b4_deepfake_v2.pth"
    # Vision Transformer for temporal artifacts
    VIT_WEIGHTS: str = "vit_base_patch16_deepfake.pth"
    # XceptionNet (baseline comparison)
    XCEPTION_WEIGHTS: str = "xception_deepfake.pth"
    # Audio deepfake detector (RawNet2 architecture)
    RAWNET_WEIGHTS: str = "rawnet2_asvspoof.pth"
    # SyncNet for lip-sync analysis
    SYNCNET_WEIGHTS: str = "syncnet_v2.pth"

    # ── Detection Thresholds ─────────────────────────────────────────────────
    FAKE_THRESHOLD: float = 0.55       # Above this → DEEPFAKE
    SUSPICIOUS_THRESHOLD: float = 0.35 # Above this → SUSPICIOUS
    HIGH_CONFIDENCE_MIN: float = 0.75  # Confidence required for HIGH verdict

    # ── Processing ───────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 200
    MAX_VIDEO_FRAMES: int = 32         # Sample 32 frames from video
    FACE_DETECTION_CONFIDENCE: float = 0.9
    IMAGE_SIZE: int = 224              # Model input size
    BATCH_SIZE: int = 8

    # ── Fusion Weights (how much each modality contributes) ──────────────────
    VISUAL_WEIGHT: float = 0.55
    FREQUENCY_WEIGHT: float = 0.25
    AUDIO_WEIGHT: float = 0.20

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60  # seconds

    


settings = Settings()

# Create directories if they don't exist
for path in [settings.MODEL_DIR, settings.UPLOAD_DIR, settings.REPORTS_DIR, settings.TEMP_DIR]:
    path.mkdir(parents=True, exist_ok=True)
