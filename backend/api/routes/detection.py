"""
DeepTrace AI — Detection API Routes
REST endpoints for image and video deepfake detection.

Endpoints:
  POST /api/detect/image   — Analyze a single image
  POST /api/detect/video   — Analyze a video file
  POST /api/detect/url     — Analyze media from URL
  GET  /api/detect/status/{task_id} — Get async task status
"""

import uuid
import shutil
import aiofiles
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from services.orchestrator import DetectionOrchestrator
from core.config import settings

logger = logging.getLogger("deeptrace.api")
router = APIRouter()

# ── Request / Response Models ──────────────────────────────────────────────────

class URLAnalysisRequest(BaseModel):
    url: HttpUrl
    description: Optional[str] = None


class DetectionResponse(BaseModel):
    task_id: str
    verdict: str
    risk_level: str
    fake_probability: float
    confidence: float
    generation_method: str
    processing_time_seconds: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_orchestrator(request: Request) -> DetectionOrchestrator:
    registry = request.app.state.model_registry
    if not hasattr(request.app.state, "_orchestrator"):
        request.app.state._orchestrator = DetectionOrchestrator(registry)
    return request.app.state._orchestrator


def _validate_file_size(file_bytes: bytes):
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {settings.MAX_FILE_SIZE_MB}MB"
        )


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/mpeg"}


# ── Image Detection ───────────────────────────────────────────────────────────

@router.post("/image", summary="Analyze an image for deepfake detection")
async def detect_image(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
):
    """
    Perform comprehensive deepfake detection on an uploaded image.

    Pipeline:
      1. Validate file type and size
      2. Decode image and extract EXIF metadata
      3. Detect faces (MTCNN / OpenCV fallback)
      4. Run EfficientNet-B4 classifier + GRAD-CAM
      5. Run ViT classifier + attention rollout
      6. Run frequency domain analysis (DCT + DWT)
      7. Run spatial face analysis (texture, lighting, edges)
      8. Multimodal fusion with confidence calibration
      9. Generate XAI explanation
      10. Return comprehensive forensic report
    """
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        # Try to infer from extension
        ext = Path(file.filename or "").suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}. Allowed: JPEG, PNG, WEBP, BMP, TIFF"
            )

    image_bytes = await file.read()
    _validate_file_size(image_bytes)

    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] Image analysis request: {file.filename} ({len(image_bytes)/1024:.1f}KB)")

    try:
        orchestrator = _get_orchestrator(request)
        result = await orchestrator.analyze_image(image_bytes, file.filename or "upload.jpg")
        result["task_id"] = task_id
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{task_id}] Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


# ── Video Detection ───────────────────────────────────────────────────────────

@router.post("/video", summary="Analyze a video for deepfake detection")
async def detect_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI, WEBM)"),
):
    """
    Perform comprehensive deepfake detection on an uploaded video.

    Additional steps vs image:
      - Frame sampling (up to 32 frames uniformly distributed)
      - Per-frame analysis
      - Temporal consistency analysis
      - Flickering detection
      - Optical flow analysis
      - Aggregate multi-frame verdict
    """
    content_type = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()
    if content_type not in ALLOWED_VIDEO_TYPES and ext not in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Allowed: MP4, MOV, AVI, WEBM"
        )

    task_id = str(uuid.uuid4())
    video_path = settings.TEMP_DIR / f"{task_id}{ext or '.mp4'}"

    try:
        # Save to temp file (video processing requires file on disk)
        async with aiofiles.open(video_path, "wb") as f:
            content = await file.read()
            _validate_file_size(content)
            await f.write(content)

        logger.info(f"[{task_id}] Video analysis request: {file.filename} ({len(content)/1024/1024:.1f}MB)")

        orchestrator = _get_orchestrator(request)
        result = await orchestrator.analyze_video(video_path, file.filename or "video.mp4")
        result["task_id"] = task_id
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{task_id}] Video analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")
    finally:
        # Cleanup temp file
        background_tasks.add_task(_cleanup_file, video_path)


# ── URL Detection ─────────────────────────────────────────────────────────────

@router.post("/url", summary="Analyze media from a URL")
async def detect_url(
    request: Request,
    body: URLAnalysisRequest,
):
    """
    Download and analyze media from a URL.
    Supports direct image/video URLs.
    """
    import httpx

    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] URL analysis: {body.url}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(str(body.url), follow_redirects=True)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        image_bytes = response.content
        _validate_file_size(image_bytes)

        filename = str(body.url).split("/")[-1] or "url_media"
        orchestrator = _get_orchestrator(request)

        if content_type in ALLOWED_IMAGE_TYPES or any(
            filename.lower().endswith(e) for e in [".jpg", ".jpeg", ".png", ".webp"]
        ):
            result = await orchestrator.analyze_image(image_bytes, filename)
        else:
            raise HTTPException(status_code=415, detail=f"URL content type not supported: {content_type}")

        result["task_id"] = task_id
        return JSONResponse(content=result)

    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")
    except Exception as e:
        logger.error(f"[{task_id}] URL analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch Detection ───────────────────────────────────────────────────────────

@router.post("/batch", summary="Analyze multiple images in one request")
async def detect_batch(
    request: Request,
    files: list[UploadFile] = File(...),
):
    """
    Analyze up to 10 images in a single request.
    Returns list of results in the same order as input files.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch request")

    orchestrator = _get_orchestrator(request)
    results = []

    for file in files:
        try:
            image_bytes = await file.read()
            result = await orchestrator.analyze_image(image_bytes, file.filename or "batch.jpg")
            results.append({"filename": file.filename, "status": "success", "result": result})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    return JSONResponse(content={"batch_results": results, "total": len(results)})


# ── Cleanup ───────────────────────────────────────────────────────────────────

async def _cleanup_file(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning(f"Could not delete temp file {path}: {e}")
