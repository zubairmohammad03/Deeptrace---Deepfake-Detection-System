"""
DeepTrace AI — Forensic Report Generator

Generates a structured forensic PDF report from detection results.
Suitable for academic submission, legal evidence, or news verification.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import datetime

router = APIRouter()


class ReportRequest(BaseModel):
    file_name: str
    verdict: str
    confidence: float
    fake_probability: float
    risk_level: str
    generation_method: str
    facial_artifacts_score: int
    texture_consistency_score: int
    frequency_domain_score: int
    temporal_consistency_score: int
    faces_detected: int
    frames_analyzed: int
    processing_time_ms: float
    xai_highlights: List[str] = []
    top_manipulated_regions: List[str] = []
    model_scores: List[dict] = []
    notes: Optional[str] = None


@router.post("/report/generate", summary="Generate forensic PDF report")
async def generate_report(body: ReportRequest):
    """
    Generates a forensic evidence report in plain text (PDF via reportlab if available).
    Returns a downloadable .txt file that documents the full analysis chain.
    """
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 72,
        "          DEEPTRACE AI — FORENSIC DEEPFAKE ANALYSIS REPORT",
        "=" * 72,
        "",
        f"  Generated:       {now}",
        f"  System Version:  DeepTrace AI v1.0.0",
        f"  Analysis Engine: Multimodal Deep Learning (EfficientNet-B4 + ViT + XceptionNet)",
        "",
        "─" * 72,
        "  SUBJECT MEDIA",
        "─" * 72,
        f"  File Name:       {body.file_name}",
        f"  Faces Detected:  {body.faces_detected}",
        f"  Frames Analyzed: {body.frames_analyzed}",
        f"  Processing Time: {body.processing_time_ms:.1f}ms",
        "",
        "─" * 72,
        "  FORENSIC VERDICT",
        "─" * 72,
        f"  VERDICT:         *** {body.verdict} ***",
        f"  Confidence:      {body.confidence:.1f}%",
        f"  Fake Probability:{body.fake_probability:.4f}",
        f"  Risk Level:      {body.risk_level}",
        f"  Generation Type: {body.generation_method}",
        "",
        "─" * 72,
        "  DIMENSION ANALYSIS SCORES  (0 = Authentic, 100 = Manipulated)",
        "─" * 72,
        f"  Facial Artifacts:       {body.facial_artifacts_score:3d}/100",
        f"  Texture Consistency:    {body.texture_consistency_score:3d}/100",
        f"  Frequency Domain:       {body.frequency_domain_score:3d}/100",
        f"  Temporal Consistency:   {body.temporal_consistency_score:3d}/100",
        "",
        "─" * 72,
        "  MODEL ENSEMBLE SCORES",
        "─" * 72,
    ]

    for ms in body.model_scores:
        bar = "█" * int(ms.get("probability", 0) * 20)
        status = "✓" if ms.get("available") else "✗ (not loaded)"
        lines.append(f"  {ms.get('name', '?'):20s} {ms.get('probability', 0):.3f}  [{bar:<20}] {status}")

    lines += [
        "",
        "─" * 72,
        "  XAI EXPLAINABILITY — WHY THIS DECISION?",
        "─" * 72,
    ]

    if body.xai_highlights:
        for i, h in enumerate(body.xai_highlights, 1):
            lines.append(f"  [{i}] {h}")
    else:
        lines.append("  XAI analysis not available for this media.")

    if body.top_manipulated_regions:
        lines.append("")
        lines.append(f"  Top manipulated regions: {', '.join(body.top_manipulated_regions)}")

    lines += [
        "",
        "─" * 72,
        "  METHODOLOGY",
        "─" * 72,
        "  1. Face detection via MTCNN with 5-point landmark alignment",
        "  2. Adversarial defense: Gaussian smoothing + JPEG re-compression",
        "  3. EfficientNet-B4: Face-swap artifact detection (ImageNet pretrained)",
        "  4. Vision Transformer ViT-B/16: Face-reenactment detection",
        "  5. XceptionNet: Texture-level manipulation detection",
        "  6. Frequency Analysis: FFT spectral anomaly + DCT block analysis",
        "  7. Weighted ensemble fusion (EfficientNet 35%, ViT 30%, Xception 20%, Freq 15%)",
        "  8. GradCAM XAI: Gradient-weighted Class Activation Mapping",
        "  9. MC-Dropout uncertainty estimation (5 forward passes)",
        "",
        "─" * 72,
        "  DATASETS USED FOR TRAINING",
        "─" * 72,
        "  • FaceForensics++ (FF++) — Face swap & reenactment",
        "  • Deepfake Detection Challenge (DFDC) — Wild deepfakes",
        "  • Celeb-DF v2 — High-quality celebrity deepfakes",
        "  • WildDeepfake — In-the-wild unconstrained deepfakes",
        "  • ASVspoof 2021 — Audio voice cloning detection",
        "",
        "─" * 72,
        "  LIMITATIONS & CAVEATS",
        "─" * 72,
        "  • Detection accuracy may decrease for heavily compressed media",
        "  • Novel GAN architectures not in training data may evade detection",
        "  • Audio analysis requires RawNet2 checkpoint to be loaded",
        "  • This report is for investigative purposes — not legal proof",
        "",
    ]

    if body.notes:
        lines += [
            "─" * 72,
            "  ANALYST NOTES",
            "─" * 72,
            f"  {body.notes}",
            "",
        ]

    lines += [
        "=" * 72,
        "  DeepTrace AI · Forensic Deepfake Detection System",
        "  github.com/your-repo/deeptrace · Contact: your@email.com",
        "=" * 72,
    ]

    report_text = "\n".join(lines)
    buf = io.BytesIO(report_text.encode("utf-8"))
    safe_name = body.file_name.replace(" ", "_").replace("/", "_")

    return StreamingResponse(
        buf,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="deeptrace_report_{safe_name}.txt"',
            "Content-Length": str(len(report_text.encode())),
        },
    )
