"""
DeepTrace AI — Main FastAPI Application
Entry point for the deepfake detection backend server.
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.routers.detection import router as detection_router
from contextlib import asynccontextmanager
import logging

from app.api.routes import detection
from app.api.routes import health
from app.api.routes import report
from app.api.routes import ai

from core.config import settings
from core.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("deeptrace")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    logger.info("🚀 DeepTrace AI starting up...")
    registry = ModelRegistry()
    await registry.load_all_models()
    app.state.model_registry = registry
    logger.info("✅ All models loaded and ready")
    yield
    logger.info("🛑 DeepTrace AI shutting down...")
    await registry.cleanup()


app = FastAPI(
    title="DeepTrace AI",
    description="Forensic-grade deepfake detection with multimodal AI and XAI explainability",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(detection.router)
app.include_router(health.router)
app.include_router(report.router)
app.include_router(ai.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,  # 1 worker to share loaded models in memory
    )
