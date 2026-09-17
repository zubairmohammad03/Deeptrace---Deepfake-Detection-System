# app/services/openrouter_service.py
"""
Deepfake detection via OpenRouter API.
Used as primary engine when no trained model is loaded,
or as fallback if model inference fails.
"""
import base64
import json
import re
import time
import io
from PIL import Image
import httpx
from app.config import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DETECTION_PROMPT = """You are a forensic image analyst. Analyze this image OBJECTIVELY. It may be completely authentic OR it may be manipulated.

Do NOT assume manipulation. Many images are real photos with natural imperfections like JPEG compression, phone camera processing, lighting variations, and lens effects — these are NOT signs of manipulation.

Only flag manipulation if you see CLEAR, SPECIFIC deepfake artifacts such as:
- Impossible facial geometry (eyes at wrong angles, asymmetric ears in an unnatural way)
- Mismatched eye reflections (different light sources reflected in each eye)
- Blending boundaries between face and background (hard seams, color fringing, halo artifacts)
- GAN grid patterns visible at pixel level
- Impossibly smooth skin with absolutely NO pores, texture, or natural variation whatsoever

Normal photos often have these natural properties — do NOT count these as manipulation:
- Slight skin smoothing from phone camera beauty processing
- Slight bokeh or lens blur on background
- JPEG compression artifacts (blocking, ringing)
- Minor lighting variation across the face
- Natural facial asymmetry
- Slight motion blur or noise from camera shake

Score guidance — overall_score is FAKE probability (0=definitely real, 100=definitely fake):
- 0–30 = clearly authentic — most real photos should score here
- 30–50 = mostly authentic with minor concerns
- 50–70 = suspicious with notable artifacts
- 70–100 = likely deepfake with clear evidence

Respond ONLY with valid JSON, no markdown, no backticks, no extra text:

{
  "verdict": "DEEPFAKE" or "AUTHENTIC" or "SUSPICIOUS",
  "confidence": <0-100>,
  "risk_level": "CRITICAL" or "HIGH" or "MEDIUM" or "LOW",
  "overall_score": <0-100 where 100=definitely fake>,
  "analysis": {
    "facial_artifacts":    {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"},
    "texture_consistency": {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"},
    "lighting_shadows":    {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"},
    "edge_blending":       {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"},
    "frequency_domain":    {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"},
    "metadata_integrity":  {"score":<0-100>,"findings":["f1","f2","f3"],"details":"explanation"}
  },
  "manipulation_regions": [{"region":"name","severity":"HIGH" or "MEDIUM" or "LOW","description":"detail"}],
  "technical_indicators": ["i1","i2","i3","i4"],
  "generation_method": "Unknown" or "GAN-based" or "Diffusion Model" or "Face Swap (DeepFaceLab)" or "Face Reenactment" or "Neural Rendering" or "Authentic",
  "forensic_summary": "2-3 sentence expert forensic summary",
  "recommendations": ["r1","r2","r3"],
  "xai_highlights": ["primary signal","secondary factor","third factor"]
}"""


def preprocess(image_bytes: bytes, max_dim: int = 1024) -> tuple:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def parse_response(text) -> dict:
    if text is None:
        raise ValueError("Empty LLM response: model returned no content")
    # Strip control characters that break JSON (preserve \n \r \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            return json.loads(match.group())
        raise ValueError("Could not parse AI response. Please try again.")


def apply_verdict_correction(result: dict) -> dict:
    # The prompt instructs the LLM to use 100 = definitely fake for both
    # overall_score and per-dimension scores.  Do NOT invert them.
    # Re-derive verdict and confidence from overall_score so they are always
    # internally consistent regardless of what text the LLM put in "verdict".
    raw_verdict    = result.get("verdict")
    raw_score      = result.get("overall_score", 50)
    raw_confidence = result.get("confidence", 50)
    print(f"[OpenRouter] apply_verdict_correction: raw verdict={raw_verdict}, overall_score={raw_score}, confidence={raw_confidence}")

    fake_score = raw_score

    if fake_score > 70:
        result["verdict"]    = "DEEPFAKE"
        result["risk_level"] = "CRITICAL" if fake_score > 85 else "HIGH"
    elif fake_score > 50:
        result["verdict"]    = "SUSPICIOUS"
        result["risk_level"] = "MEDIUM"
    else:
        result["verdict"]    = "AUTHENTIC"
        result["risk_level"] = "LOW"

    result["confidence"] = 100 - fake_score

    print(f"[OpenRouter] apply_verdict_correction: corrected verdict={result['verdict']}, confidence={result['confidence']}")
    return result


def analyze_with_openrouter(image_bytes: bytes) -> dict:
    """Call OpenRouter API for deepfake analysis."""
    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to backend/.env and restart."
        )

    start = time.time()
    processed, proc_type = preprocess(image_bytes)
    b64      = base64.b64encode(processed).decode()
    data_url = f"data:{proc_type};base64,{b64}"

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:5173",
        "X-Title":       "DeepTrace AI",
    }
    payload = {
        "model":      settings.openrouter_model,
        "max_tokens": 1500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": DETECTION_PROMPT},
            ],
        }],
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)

    if resp.status_code == 401:
        raise ValueError("Invalid OpenRouter API key. Check OPENROUTER_API_KEY in backend/.env")
    if resp.status_code == 429:
        raise ValueError("OpenRouter rate limited. Please wait a moment and try again.")
    if resp.status_code == 402:
        raise ValueError("OpenRouter account has insufficient credits.")
    if not resp.is_success:
        try:
            msg = resp.json().get("error", {}).get("message", "Unknown error")
        except Exception:
            msg = resp.text[:200]
        raise ValueError(f"OpenRouter error {resp.status_code}: {msg}")

    data    = resp.json()
    choices = data.get("choices") or []
    text    = choices[0].get("message", {}).get("content") if choices else None
    result  = parse_response(text)
    result = apply_verdict_correction(result)

    result["model_source"] = "openrouter"
    result["model_used"]   = settings.openrouter_model
    result["_processing_ms"] = round((time.time() - start) * 1000, 1)
    return result
