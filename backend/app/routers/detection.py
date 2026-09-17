# app/routers/detection.py
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.detection_service import analyze_media, extract_video_frame
from app.services.model_loader import get_model
from app.config import settings

router = APIRouter(prefix="/api", tags=["detection"])

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO = {"video/mp4", "video/quicktime", "video/avi", "video/x-msvideo"}
ALL_ALLOWED   = ALLOWED_IMAGE | ALLOWED_VIDEO


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Analyze uploaded image or video for deepfake manipulation."""
    if file.content_type not in ALL_ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: jpeg, png, webp, mp4, mov, avi"
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_upload_mb}MB")

    start = time.time()
    try:
        if file.content_type in ALLOWED_VIDEO:
            image_bytes, media_type = extract_video_frame(contents)
        else:
            image_bytes, media_type = contents, file.content_type

        result = analyze_media(image_bytes, media_type, filename=file.filename)
        print(f"[Response] verdict={result.get('verdict')} overall_score={result.get('overall_score')} confidence={result.get('confidence')} risk={result.get('risk_level')}")

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "media_type": file.content_type,
            "processing_time_ms": round((time.time() - start) * 1000, 1),
            "engine": result.get("model_source", "unknown"),
            "result": result,
        })

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/health")
async def health():
    """Shows which detection engine is active."""
    _, _, model_loaded = get_model()
    return {
        "status": "ok",
        "active_engine": "EfficientNet-B4 (custom model)" if model_loaded else "OpenRouter API",
        "model_loaded": model_loaded,
        "model_path": settings.model_path if model_loaded else None,
        "openrouter_model": settings.openrouter_model,
        "openrouter_key_set": bool(settings.openrouter_api_key),
        "version": "3.0.0",
    }
