"""
Vision analysis proxy (Gemini direct or OpenRouter).

Env:
  AI_PROVIDER=gemini|openrouter   (default: gemini)

  Gemini: GEMINI_API_KEY, optional GEMINI_MODELS
  OpenRouter: OPENROUTER_API_KEY, optional OPENROUTER_MODEL
    Use a real ID from https://openrouter.ai/models (vision + image input).
    Good defaults: openrouter/free  OR  qwen/qwen-2.5-vl-7b-instruct  (not qwen2-vl-7b — wrong slug)
"""

import json
import logging
import os
import asyncio
import httpx
import hashlib
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_FALLBACK_DIMS = [
    "facial_artifacts", "texture_consistency", "lighting_shadows",
    "edge_blending", "frequency_domain", "metadata_integrity",
]


def _llm_fallback(reason: str = "AI explanation unavailable") -> dict:
    """Valid response returned whenever the LLM call fails for any reason."""
    return {
        "verdict":              "SUSPICIOUS",
        "confidence":           50,
        "risk_level":           "MEDIUM",
        "overall_score":        50,
        "analysis":             {d: {"score": 50, "findings": ["AI analysis unavailable"], "details": reason} for d in _FALLBACK_DIMS},
        "manipulation_regions": [],
        "technical_indicators": [reason],
        "generation_method":    "Unknown",
        "forensic_summary":     "AI explanation unavailable",
        "recommendations":      ["Please retry the analysis"],
        "xai_highlights":       [],
    }

router = APIRouter()

# Order: try lighter/cheaper models first; override with GEMINI_MODELS=comma,separated
_DEFAULT_MODELS = "gemini-2.0-flash,gemini-1.5-flash-8b,gemini-1.5-flash"
# Picks a free, capability-matched model (including vision when you send images).
_DEFAULT_OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
CACHE_TTL_SECONDS = 15 * 60
_RESULT_CACHE: dict[str, tuple[float, dict]] = {}

DETECTION_PROMPT = """You are a forensic deepfake detection AI with expert-level analysis capabilities.
Analyze this image for signs of AI manipulation, synthetic generation, or deepfake artifacts.

Respond ONLY with a valid JSON object - no markdown, no backticks, no text outside JSON:

{
  "verdict": "DEEPFAKE" or "AUTHENTIC" or "SUSPICIOUS",
  "confidence": <number 0-100>,
  "risk_level": "CRITICAL" or "HIGH" or "MEDIUM" or "LOW",
  "overall_score": <number 0-100 where 100 means definitely fake>,
  "analysis": {
    "facial_artifacts": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Detailed explanation"
    },
    "texture_consistency": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Texture analysis explanation"
    },
    "lighting_shadows": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Lighting consistency analysis"
    },
    "edge_blending": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Edge boundary analysis"
    },
    "frequency_domain": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Frequency pattern analysis"
    },
    "metadata_integrity": {
      "score": <0-100>,
      "findings": ["finding1", "finding2", "finding3"],
      "details": "Metadata integrity analysis"
    }
  },
  "manipulation_regions": [
    { "region": "Region Name", "severity": "HIGH" or "MEDIUM" or "LOW", "description": "what was found" }
  ],
  "technical_indicators": ["indicator1", "indicator2", "indicator3", "indicator4"],
  "generation_method": "Unknown" or "GAN-based" or "Diffusion Model" or "Face Swap (DeepFaceLab)" or "Face Reenactment" or "Neural Rendering" or "Authentic",
  "forensic_summary": "2-3 sentence expert forensic summary",
  "recommendations": ["action1", "action2", "action3"],
  "xai_highlights": ["primary signal", "secondary factor", "third factor"]
}"""


class AnalyzeRequest(BaseModel):
    base64Data: str
    mediaType: str


def _cache_key(body: AnalyzeRequest) -> str:
    digest = hashlib.sha256(f"{body.mediaType}:{body.base64Data}".encode("utf-8")).hexdigest()
    return digest


def _get_cached_result(key: str) -> dict | None:
    hit = _RESULT_CACHE.get(key)
    if not hit:
        return None
    stored_at, value = hit
    if time.time() - stored_at > CACHE_TTL_SECONDS:
        _RESULT_CACHE.pop(key, None)
        return None
    return value


def _set_cached_result(key: str, value: dict) -> None:
    _RESULT_CACHE[key] = (time.time(), value)


def _models_list() -> list[str]:
    raw = os.getenv("GEMINI_MODELS", _DEFAULT_MODELS).strip()
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_retry_seconds(resp: httpx.Response) -> int | None:
    """Parse Retry-After header or google.rpc.RetryInfo retryDelay from JSON body."""
    ra = resp.headers.get("retry-after")
    if ra and ra.isdigit():
        return min(int(ra), 120)
    try:
        data = resp.json()
        err = data.get("error") or {}
        for d in err.get("details") or []:
            if d.get("@type", "").endswith("RetryInfo"):
                delay = d.get("retryDelay") or ""
                # e.g. "42s" or "42.5s"
                delay = delay.replace("s", "").strip()
                if delay:
                    sec = int(float(delay))
                    return min(max(sec, 1), 120)
    except Exception:
        pass
    return None


def _extract_json(text) -> dict:
    import re as _re
    if not text:
        raise ValueError("Empty response from model")
    # Strip control characters that break JSON (preserve \n \r \t)
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    clean = text.replace("```json", "").replace("```", "").strip()
    if not clean:
        raise ValueError("Empty response from model")
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean[start : end + 1])
    raise ValueError("Could not parse JSON from model response")


async def _analyze_openrouter(body: AnalyzeRequest, cache_key: str) -> dict:
    """OpenAI-compatible vision chat (OpenRouter aggregates many providers)."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.error("OPENROUTER_API_KEY not configured")
        return _llm_fallback("AI explanation unavailable: OPENROUTER_API_KEY not configured")

    model = os.getenv("OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL).strip()
    data_url = f"data:{body.mediaType};base64,{body.base64Data}"
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DETECTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:5173"),
        "X-Title": "DeepTrace AI",
    }
    last_error = None
    for attempt in range(1, 3):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if resp.status_code == 429:
                wait_seconds = _parse_retry_seconds(resp) or min(20 * attempt, 60)
                last_error = f"Rate limited (attempt {attempt})"
                if attempt < 2:
                    await asyncio.sleep(wait_seconds)
                continue
            if not resp.is_success:
                err = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                msg = err.get("error", {}).get("message") if isinstance(err.get("error"), dict) else str(err)
                last_error = f"OpenRouter error {resp.status_code}: {msg or resp.text[:300]}"
                break  # non-429 errors won't improve on retry
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            parsed = _extract_json(text)
            _set_cached_result(cache_key, parsed)
            return parsed
        except Exception as ex:
            last_error = str(ex)
            if attempt < 2:
                await asyncio.sleep(2)

    logger.error(f"OpenRouter unavailable: {last_error}")
    return _llm_fallback(f"AI explanation unavailable: {last_error}")


@router.post("/ai/analyze", summary="Analyze media (Gemini or OpenRouter)")
async def analyze_media(body: AnalyzeRequest):
    key = _cache_key(body)
    cached = _get_cached_result(key)
    if cached is not None:
        return cached

    provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
    if provider in ("openrouter", "or"):
        return await _analyze_openrouter(body, key)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GEMINI_API_KEY")
    if not api_key:
        logger.error("No Gemini API key configured")
        return _llm_fallback("AI explanation unavailable: API key not configured")

    last_error = None
    saw_429 = False
    models = _models_list()

    for model in models:
        # At most 2 attempts per model on 429 — long multi-minute waits were mostly retries, not "processing".
        for attempt in range(1, 3):
            try:
                url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{
                        "parts": [
                            {"inline_data": {"mime_type": body.mediaType, "data": body.base64Data}},
                            {"text": DETECTION_PROMPT},
                        ]
                    }],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700, "topP": 0.8},
                }

                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=payload)

                if resp.status_code == 429:
                    saw_429 = True
                    wait_seconds = _parse_retry_seconds(resp)
                    if wait_seconds is None:
                        wait_seconds = min(20 * attempt, 60)
                    last_error = "Rate limited"
                    if attempt < 2:
                        await asyncio.sleep(wait_seconds)
                    continue

                if not resp.is_success:
                    err = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    last_error = err.get("error", {}).get("message", f"Gemini error {resp.status_code}")
                    await asyncio.sleep(2)
                    continue

                data = resp.json()
                text = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [{}])[0].get("text", "")
                parsed = _extract_json(text)
                _set_cached_result(key, parsed)
                return parsed
            except Exception as ex:
                last_error = str(ex)
                if attempt < 2:
                    await asyncio.sleep(2)

    reason = (
        f"Gemini rate limited — retry in a few minutes (last: {last_error})"
        if saw_429
        else f"AI explanation unavailable: {last_error}"
    )
    logger.error(f"Gemini unavailable: {reason}")
    return _llm_fallback(reason)
