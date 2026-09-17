# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.detection import router
from app.services.model_loader import load_model
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load EfficientNet model once at startup (if available)
    print("\n" + "="*55)
    print("  DeepTrace AI v3 — Starting")
    print("="*55)
    model, device, loaded = load_model()
    if loaded:
        print(f"  Engine : EfficientNet-B4 custom model ({device})")
    else:
        print(f"  Engine : OpenRouter API ({settings.openrouter_model})")
    print("="*55 + "\n")
    yield


app = FastAPI(
    title="DeepTrace AI",
    description="Forensic deepfake detection — EfficientNet-B4 + OpenRouter",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.cors_origin,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service":   "DeepTrace AI v3",
        "endpoints": {
            "detect": "POST /api/detect",
            "health": "GET /api/health",
            "docs":   "GET /docs",
        }
    }
