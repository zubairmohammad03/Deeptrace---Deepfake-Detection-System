# app/services/detection_service.py
"""
Main detection orchestrator.

Priority:
  1. EfficientNet-B4 custom model  (if best_model.pt exists in checkpoints/)
  2. OpenRouter API                (always available as fallback)

This means:
  - If you've trained and placed best_model.pt → uses your model + real GRAD-CAM
  - If no model file → uses OpenRouter API seamlessly, no crashes
"""
import os
import io
import logging
import tempfile
import cv2
import numpy as np
from PIL import Image

from app.services.model_loader import get_model
from app.services.model_inference import run_model_inference
from app.services.openrouter_service import analyze_with_openrouter

logger = logging.getLogger(__name__)

_FALLBACK_DIMS = [
    "facial_artifacts", "texture_consistency", "lighting_shadows",
    "edge_blending", "frequency_domain", "metadata_integrity",
]


def _openrouter_fallback(error: str) -> dict:
    """Structured result returned when BOTH model and OpenRouter are unavailable."""
    print(f"[Detection] *** FALLBACK TRIGGERED *** — both model and OpenRouter failed.")
    print(f"[Detection] *** FALLBACK REASON: {error}")
    logger.error(f"[Detection] OpenRouter unavailable — returning fallback: {error}")
    return {
        "verdict":              "SUSPICIOUS",
        "confidence":           0,
        "risk_level":           "MEDIUM",
        "overall_score":        50,
        "analysis":             {d: {"score": 50, "findings": ["AI analysis unavailable"], "details": "AI explanation unavailable"} for d in _FALLBACK_DIMS},
        "manipulation_regions": [],
        "technical_indicators": ["AI explanation unavailable — analysis engine temporarily offline"],
        "generation_method":    "Unknown",
        "forensic_summary":     "AI explanation unavailable",
        "recommendations":      ["Please retry the analysis"],
        "xai_highlights":       [],
        "model_source":         "fallback",
        "model_used":           "none",
        "_processing_ms":       0,
    }


def _apply_explanation_fallback(result: dict, error: str) -> dict:
    """
    When OpenRouter explanation fails but the model already produced results,
    keep ALL model scores/verdict intact — only mark the LLM explanation fields
    as unavailable.
    """
    logger.warning(f"[Detection] OpenRouter explanation unavailable: {error}")
    result["forensic_summary"]     = "AI explanation unavailable"
    result["manipulation_regions"] = []
    return result


def extract_video_frame(video_bytes: bytes) -> tuple:
    """Extract frame at 25% of video duration."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(total * 0.25)))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError("Could not extract frame from video")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "image/jpeg"
    finally:
        os.unlink(tmp_path)


def analyze_media(image_bytes: bytes, media_type: str, filename: str = "unknown") -> dict:
    """
    Run detection using the best available engine.

    Flow:
      1. Model inference  → verdict, confidence, scores (ground truth)
      2. OpenRouter       → forensic_summary, manipulation_regions (explanation layer)
      3. If OpenRouter fails → keep all model scores, only mark explanation unavailable
      4. If model not loaded → OpenRouter as primary; if that fails → hardcoded fallback
    """
    print(f"[Detection] >>> Processing image: {filename} | {len(image_bytes)} bytes | type={media_type}")
    _, _, model_loaded = get_model()
    print(f"[Detection]     model_loaded={model_loaded}")

    if model_loaded:
        try:
            print("[Detection] Using EfficientNet-B4 custom model")
            result = run_model_inference(image_bytes)
        except RuntimeError as e:
            if "MODEL_NOT_LOADED" not in str(e):
                raise
            print("[Detection] Model runtime error, falling back to OpenRouter")
            model_loaded = False
        except Exception as e:
            print(f"[Detection] Model inference EXCEPTION ({type(e).__name__}: {e}), falling back to OpenRouter")
            model_loaded = False

    if model_loaded:
        # Model succeeded — save model scores as secondary, then use LLM as PRIMARY verdict.
        model_fake_score = result.get("overall_score", 50)
        model_verdict    = result.get("verdict", "UNKNOWN")
        result["model_score"] = {
            "fake_probability": result.get("model_info", {}).get("fake_probability", model_fake_score),
            "verdict":          model_verdict,
            "architecture":     "EfficientNet-B4",
        }

        try:
            print("[Detection] Calling OpenRouter for PRIMARY verdict (LLM has vision, model is secondary)")
            llm = analyze_with_openrouter(image_bytes)

            llm_score = llm.get("overall_score", 50)
            print(f"[Detection] LLM overall_score={llm_score}, verdict={llm.get('verdict')}")
            print(f"[Detection] Model overall_score={model_fake_score}, verdict={model_verdict}")

            # LLM is the authoritative verdict source — it can actually SEE the image
            result["verdict"]               = llm.get("verdict",              result["verdict"])
            result["confidence"]            = llm.get("confidence",           result["confidence"])
            result["risk_level"]            = llm.get("risk_level",           result["risk_level"])
            result["overall_score"]         = llm.get("overall_score",        result["overall_score"])
            result["analysis"]              = llm.get("analysis",             result["analysis"])
            result["forensic_summary"]      = llm.get("forensic_summary",     result.get("forensic_summary", ""))
            result["manipulation_regions"]  = llm.get("manipulation_regions", [])
            result["xai_highlights"]        = llm.get("xai_highlights",       result.get("xai_highlights", []))
            result["recommendations"]       = llm.get("recommendations",      result.get("recommendations", []))
            result["technical_indicators"]  = llm.get("technical_indicators", result.get("technical_indicators", []))
            result["generation_method"]     = llm.get("generation_method",    result.get("generation_method", "Unknown"))
            result["verdict_source"]        = "llm_primary"
        except Exception as e:
            print(f"[Detection] OpenRouter EXCEPTION ({type(e).__name__}: {e}) — falling back to model verdict")
            _apply_explanation_fallback(result, str(e))
            result["verdict_source"] = "model_fallback"

        print(f"[Detection] <<< COMPLETE RESULT:")
        import pprint
        pprint.pprint({k: v for k, v in result.items() if k not in ("gradcam_image", "original_image")})
        return result

    # No model loaded (or model failed) — use OpenRouter as primary engine.
    print("[Detection] Using OpenRouter API as primary engine")
    try:
        result = analyze_with_openrouter(image_bytes)
        print(f"[Detection] <<< Result: verdict={result.get('verdict')} confidence={result.get('confidence')} fake_score={result.get('overall_score')} engine=openrouter")
        return result
    except Exception as e:
        print(f"[Detection] OpenRouter PRIMARY EXCEPTION ({type(e).__name__}: {e})")
        result = _openrouter_fallback(str(e))
        print(f"[Detection] <<< Result: verdict={result.get('verdict')} confidence={result.get('confidence')} fake_score={result.get('overall_score')} engine=fallback")
        return result
