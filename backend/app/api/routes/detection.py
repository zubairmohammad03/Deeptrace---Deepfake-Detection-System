"""
DeepTrace AI — Detection API Routes

Endpoints:
  POST /api/v1/detect/image   → analyze a single image
  POST /api/v1/detect/video   → analyze a video file
  POST /api/v1/detect/audio   → analyze audio for voice deepfakes
  POST /api/v1/detect/url     → analyze media from a URL
"""

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import logging

from app.core.config import settings
from app.services.detection_engine import DeepfakeDetector, DetectionResult

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Response Models ───────────────────────────────────────────────────────────

class ModelScoreResponse(BaseModel):
    name: str
    probability: float
    weight: float
    available: bool


class DetectionResponse(BaseModel):
    verdict: str
    confidence: float
    fake_probability: float
    risk_level: str
    generation_method: str
    model_scores: list
    facial_artifacts_score: int
    texture_consistency_score: int
    frequency_domain_score: int
    temporal_consistency_score: int
    lip_sync_score: int
    audio_score: int
    gradcam_heatmap_b64: str | None
    xai_region_scores: dict
    xai_highlights: list
    top_manipulated_regions: list
    faces_detected: int
    frames_analyzed: int
    processing_time_ms: float
    has_audio: bool
    has_video: bool


class UrlRequest(BaseModel):
    url: HttpUrl


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_detector(request: Request) -> DeepfakeDetector:
    loader = request.app.state.model_loader
    return DeepfakeDetector(loader.models)


def result_to_response(r: DetectionResult) -> dict:
    return {
        "verdict": r.verdict,
        "confidence": r.confidence,
        "fake_probability": r.fake_probability,
        "risk_level": r.risk_level,
        "generation_method": r.generation_method,
        "model_scores": [
            {"name": s.name, "probability": s.probability, "weight": s.weight, "available": s.available}
            for s in r.model_scores
        ],
        "facial_artifacts_score": r.facial_artifacts_score,
        "texture_consistency_score": r.texture_consistency_score,
        "frequency_domain_score": r.frequency_domain_score,
        "temporal_consistency_score": r.temporal_consistency_score,
        "lip_sync_score": r.lip_sync_score,
        "audio_score": r.audio_score,
        "gradcam_heatmap_b64": r.gradcam_heatmap_b64,
        "xai_region_scores": r.xai_region_scores,
        "xai_highlights": r.xai_highlights,
        "top_manipulated_regions": r.top_manipulated_regions,
        "faces_detected": r.faces_detected,
        "frames_analyzed": r.frames_analyzed,
        "processing_time_ms": r.processing_time_ms,
        "has_audio": r.has_audio,
        "has_video": r.has_video,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/detect/image", summary="Analyze an image for deepfake content")
async def detect_image(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
):
    """
    Accepts an image upload and runs the full deepfake detection pipeline:

    1. **Face detection** — MTCNN detects and aligns faces
    2. **Adversarial defense** — Input smoothing + JPEG pre-processing
    3. **EfficientNet-B4** — Face-swap artifact detection
    4. **Vision Transformer** — Face-reenactment detection
    5. **XceptionNet** — Texture-level manipulation detection
    6. **Frequency Analysis** — FFT + DCT spectral anomaly detection
    7. **Multimodal Fusion** — Weighted ensemble of all models
    8. **GradCAM XAI** — Saliency map showing manipulated regions
    """
    # Validate file type
    if file.content_type not in [
        "image/jpeg", "image/jpg", "image/png",
        "image/webp", "image/bmp", "image/tiff"
    ]:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")

    # Validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(400, f"Image too large: {size_mb:.1f}MB > {settings.MAX_IMAGE_SIZE_MB}MB limit")

    detector = get_detector(request)
    result = await detector.detect_image(contents)
    return JSONResponse(result_to_response(result))


@router.post("/detect/video", summary="Analyze a video for deepfake content")
async def detect_video(
    request: Request,
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI, MKV, WEBM)"),
):
    """
    Accepts a video upload and runs temporal deepfake detection:

    1. **Frame extraction** — Samples N frames with stride for efficiency
    2. **Per-frame analysis** — Full image pipeline on each frame
    3. **Temporal consistency** — Detects frame-to-frame flickering (deepfake signal)
    4. **Worst-frame GradCAM** — XAI on the most suspicious frame
    5. **Aggregation** — 60% max-frame + 40% mean-frame fusion (conservative)
    """
    if file.content_type not in [
        "video/mp4", "video/quicktime", "video/avi",
        "video/x-msvideo", "video/webm", "video/x-matroska"
    ]:
        raise HTTPException(400, f"Unsupported video type: {file.content_type}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise HTTPException(400, f"Video too large: {size_mb:.1f}MB > {settings.MAX_VIDEO_SIZE_MB}MB limit")

    detector = get_detector(request)
    result = await detector.detect_video(contents)
    return JSONResponse(result_to_response(result))


@router.post("/detect/audio", summary="Analyze audio for voice cloning / deepfake")
async def detect_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG)"),
):
    """
    Analyzes audio for AI-generated or voice-cloned speech using RawNet2.

    RawNet2 operates directly on raw waveforms (no MFCC or spectrograms),
    learning to detect subtle prosody and codec artifacts introduced by
    TTS/voice conversion systems.
    """
    if file.content_type not in [
        "audio/wav", "audio/x-wav", "audio/mpeg",
        "audio/mp3", "audio/flac", "audio/ogg"
    ]:
        raise HTTPException(400, f"Unsupported audio type: {file.content_type}")

    contents = await file.read()
    detector = get_detector(request)
    result = await detector.detect_audio(contents)
    return JSONResponse(result_to_response(result))


@router.post("/detect/url", summary="Analyze media from a public URL")
async def detect_url(
    request: Request,
    body: UrlRequest,
):
    """
    Downloads media from a public URL and runs appropriate detection pipeline.
    Supports images and videos. Content-Type header determines pipeline used.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(str(body.url))
            response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(400, f"Could not fetch URL: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"URL returned {e.response.status_code}")

    content_type = response.headers.get("content-type", "").split(";")[0].strip()
    contents = response.content

    detector = get_detector(request)

    if content_type.startswith("image/"):
        result = await detector.detect_image(contents)
    elif content_type.startswith("video/"):
        result = await detector.detect_video(contents)
    elif content_type.startswith("audio/"):
        result = await detector.detect_audio(contents)
    else:
        raise HTTPException(400, f"Unsupported content type from URL: {content_type}")

    return JSONResponse(result_to_response(result))
