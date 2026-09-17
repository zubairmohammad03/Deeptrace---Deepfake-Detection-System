# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenRouter API
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.0-flash-001"
    ai_provider: str = "openrouter"

    # Custom trained model (EfficientNet-B4)
    # Place your best_model.pt from Colab training into backend/checkpoints/
    model_path: str = "checkpoints/best_model.pt"
    model_backbone: str = "efficientnet_b4"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origin: str = "http://localhost:5173"
    max_upload_mb: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"  # ignore unknown keys like ANTHROPIC_API_KEY etc.

settings = Settings()
