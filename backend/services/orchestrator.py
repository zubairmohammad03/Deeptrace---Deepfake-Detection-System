"""
DeepTrace AI — Detection Orchestrator
The main pipeline that coordinates all detection services.

INPUT → PREPROCESS → DETECT → FUSE → EXPLAIN → REPORT

This is the heart of the system. Every detection request flows through here.
"""

import asyncio
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import logging
import time
import io
import base64
from pathlib import Path
from typing import Dict, Optional, List
from PIL import Image

from services.frequency_analyzer import FrequencyDomainAnalyzer
from services.face_analyzer import FaceAnalysisService
from services.fusion_engine import FusionEngine
from services.video_processor import VideoProcessor
from utils.metadata_extractor import extract_image_metadata
from utils.xai_generator import XAIGenerator
from core.config import settings

logger = logging.getLogger("deeptrace.orchestrator")


# ImageNet normalization (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DetectionOrchestrator:
    """
    Coordinates the complete detection pipeline for images and videos.
    Instantiated once at startup and reused for all requests.
    """

    def __init__(self, model_registry):
        self.registry = model_registry
        self.frequency_analyzer = FrequencyDomainAnalyzer()
        self.face_analyzer = FaceAnalysisService()
        self.fusion_engine = FusionEngine()
        self.video_processor = VideoProcessor()
        self.xai_generator = XAIGenerator()
        logger.info("✅ Detection orchestrator ready")

    async def analyze_image(self, image_bytes: bytes, filename: str) -> Dict:
        """
        Full image analysis pipeline.

        Flow:
          1. Decode image bytes
          2. Extract EXIF metadata
          3. Detect & crop faces
          4. Run EfficientNet on face crop
          5. Run ViT on face crop
          6. Run frequency domain analysis
          7. Run spatial face analysis
          8. Fuse all scores
          9. Generate GRAD-CAM heatmap
          10. Build comprehensive result dict
        """
        start_time = time.time()
        logger.info(f"🔍 Analyzing image: {filename}")

        # ── Step 1: Decode ────────────────────────────────────────────────────
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # ── Step 2: Metadata ──────────────────────────────────────────────────
        metadata = extract_image_metadata(image_bytes, filename)

        # ── Step 3: Face Detection ────────────────────────────────────────────
        face_boxes = self.face_analyzer.detect_faces(bgr)
        if face_boxes:
            x, y, w, h = max(face_boxes, key=lambda b: b[2] * b[3])
            m = int(min(w, h) * 0.2)
            fh, fw = bgr.shape[:2]
            x1, y1 = max(0, x - m), max(0, y - m)
            x2, y2 = min(fw, x + w + m), min(fh, y + h + m)
            face_crop_bgr = cv2.resize(bgr[y1:y2, x1:x2], (224, 224))
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_crop_rgb)
            face_detected = True
        else:
            # No face detected — analyze full image
            face_crop_bgr = cv2.resize(bgr, (224, 224))
            face_pil = pil_img.resize((224, 224))
            face_detected = False
            logger.warning("⚠️  No face detected — analyzing full image")

        # ── Step 4: EfficientNet ──────────────────────────────────────────────
        eff_score, eff_conf, gradcam = await self._run_efficientnet(face_pil)

        # ── Step 5: ViT ───────────────────────────────────────────────────────
        vit_score, attn_rollout = await self._run_vit(face_pil)

        # ── Step 6: Frequency Domain ──────────────────────────────────────────
        freq_result = self.frequency_analyzer.analyze(face_crop_bgr)

        # ── Step 7: Spatial Face Analysis ────────────────────────────────────
        spatial_result = self.face_analyzer.analyze_face(face_crop_bgr)

        # ── Step 8: Fusion ────────────────────────────────────────────────────
        fusion = self.fusion_engine.fuse(
            efficientnet_score=eff_score,
            efficientnet_conf=eff_conf,
            vit_score=vit_score,
            frequency_score=freq_result["composite_score"],
            spatial_score=spatial_result["spatial_fake_score"],
            metadata=metadata,
        )

        # ── Step 9: XAI ───────────────────────────────────────────────────────
        xai_data = self.xai_generator.generate(
            image=face_crop_bgr,
            gradcam=gradcam,
            attn_rollout=attn_rollout,
            fusion=fusion,
        )

        elapsed = time.time() - start_time
        logger.info(f"✅ Analysis complete in {elapsed:.2f}s — {fusion['verdict']} ({fusion['fake_probability']:.2%})")

        # ── Step 10: Build result ─────────────────────────────────────────────
        return self._build_result(
            fusion=fusion,
            freq_result=freq_result,
            spatial_result=spatial_result,
            xai_data=xai_data,
            metadata=metadata,
            face_detected=face_detected,
            n_faces=len(face_boxes),
            processing_time=elapsed,
            filename=filename,
        )

    async def analyze_video(self, video_path: Path, filename: str) -> Dict:
        """
        Full video analysis pipeline.

        Additional steps vs image:
          - Frame sampling
          - Per-frame analysis (runs analyze_image on each frame)
          - Temporal consistency analysis
          - Flickering analysis
          - Optical flow analysis
          - Aggregate frame scores into video-level verdict
        """
        start_time = time.time()
        logger.info(f"🎬 Analyzing video: {filename}")

        # Process video
        video_data = self.video_processor.process(video_path)

        if not video_data["frames"]:
            raise ValueError("No processable frames found in video")

        # Run per-frame analysis
        frame_results = []
        face_crops = [(idx, crop) for idx, crop in video_data["frames"] if crop is not None][:16]

        for frame_idx, face_crop in face_crops:
            # Convert BGR crop to bytes for reuse of analyze_image logic
            _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buf.tobytes()

            frame_result = await self.analyze_image(frame_bytes, f"frame_{frame_idx}.jpg")
            frame_results.append({
                "frame": frame_idx,
                "fake_probability": frame_result["fake_probability"],
                "verdict": frame_result["verdict"],
            })

        if not frame_results:
            raise ValueError("Could not analyze any frames")

        # Aggregate frame scores
        frame_scores = [r["fake_probability"] for r in frame_results]
        mean_score = float(np.mean(frame_scores))
        max_score = float(np.max(frame_scores))
        # Weighted: max score has more influence (one bad frame = suspicious)
        video_score = mean_score * 0.6 + max_score * 0.4

        # Include temporal analysis in final score
        temporal_penalty = video_data["temporal"]["score"] * 0.15
        flickering_penalty = video_data["flickering"]["score"] * 0.10
        flow_penalty = video_data["optical_flow"]["score"] * 0.05
        final_score = min(1.0, video_score + temporal_penalty + flickering_penalty + flow_penalty)

        verdict, risk_level = FusionEngine()._make_verdict(final_score, 0.1)

        elapsed = time.time() - start_time
        logger.info(f"✅ Video analysis complete in {elapsed:.2f}s — {verdict}")

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "fake_probability": round(final_score, 4),
            "confidence": round((1.0 - final_score) * 100, 1),
            "generation_method": "Video Deepfake" if final_score > 0.5 else "Authentic Video",
            "video_metadata": video_data["metadata"],
            "frame_analysis": frame_results,
            "temporal_consistency": video_data["temporal"],
            "flickering": video_data["flickering"],
            "optical_flow": video_data["optical_flow"],
            "n_frames_analyzed": len(frame_results),
            "processing_time_seconds": round(elapsed, 2),
            "filename": filename,
        }

    # ── Model Inference Helpers ───────────────────────────────────────────────

    async def _run_efficientnet(self, face_pil: Image.Image):
        """Run EfficientNet inference in executor (non-blocking)."""
        model = self.registry.efficientnet
        device = self.registry.device

        def _infer():
            tensor = preprocess(face_pil).unsqueeze(0).to(device)
            tensor.requires_grad_(True)
            with torch.enable_grad():
                logits, _ = model(tensor)
                prob = torch.sigmoid(logits).item()

            # GRAD-CAM
            gradcam = None
            try:
                gradcam_map = model.get_gradcam(tensor)
                gradcam = gradcam_map.cpu().numpy()
            except Exception as e:
                logger.debug(f"GRAD-CAM failed: {e}")

            # Confidence: how far from decision boundary
            conf = min(0.99, abs(prob - 0.5) * 2)
            return float(prob), float(conf), gradcam

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer)

    async def _run_vit(self, face_pil: Image.Image):
        """Run ViT inference and attention rollout."""
        model = self.registry.vit
        device = self.registry.device

        def _infer():
            tensor = preprocess(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(tensor)
                prob = torch.sigmoid(logits).item()

            attn = None
            try:
                attn_map = model.get_attention_rollout(tensor)
                attn = attn_map.cpu().numpy()
            except Exception:
                pass

            return float(prob), attn

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer)

    # ── Result Builder ────────────────────────────────────────────────────────

    def _build_result(
        self,
        fusion: Dict,
        freq_result: Dict,
        spatial_result: Dict,
        xai_data: Dict,
        metadata: Dict,
        face_detected: bool,
        n_faces: int,
        processing_time: float,
        filename: str,
    ) -> Dict:
        """Assemble the final comprehensive result."""

        # Collect all findings
        all_findings = (
            freq_result.get("findings", []) +
            spatial_result.get("findings", [])
        )

        # Build per-dimension analysis breakdown
        analysis_breakdown = {
            "facial_artifacts": {
                "score": int(spatial_result["texture"]["score"] * 100),
                "findings": spatial_result["texture"]["findings"],
                "details": f"Laplacian variance: {spatial_result['texture'].get('laplacian_variance', 0):.0f}. Sharpness ratio: {spatial_result['texture'].get('sharpness_ratio', 1):.2f}"
            },
            "texture_consistency": {
                "score": int(spatial_result.get("symmetry", {}).get("score", 0) * 100),
                "findings": spatial_result.get("symmetry", {}).get("findings", []),
                "details": f"Facial symmetry score: {spatial_result.get('symmetry', {}).get('asymmetry_score', 0):.3f}"
            },
            "lighting_shadows": {
                "score": int(spatial_result["lighting"]["score"] * 100),
                "findings": spatial_result["lighting"]["findings"],
                "details": f"Brightness asymmetry: {spatial_result['lighting'].get('brightness_asymmetry', 0):.1f}"
            },
            "edge_blending": {
                "score": int(spatial_result["edge_blending"]["score"] * 100),
                "findings": spatial_result["edge_blending"]["findings"],
                "details": f"Channel misalignment: {spatial_result['edge_blending'].get('channel_misalignment', 0):.1f}"
            },
            "frequency_domain": {
                "score": int(freq_result["composite_score"] * 100),
                "findings": freq_result["findings"],
                "details": f"DCT: {freq_result['dct']['anomaly_score']:.2f} | DWT HH entropy: {freq_result['dwt'].get('hh_entropy', 0):.2f}"
            },
            "metadata_integrity": {
                "score": int(len(fusion.get("metadata_flags", [])) / 3 * 100),
                "findings": [f["description"] for f in fusion.get("metadata_flags", [])],
                "details": f"EXIF fields found: {len(metadata)} | Suspicious flags: {len(fusion.get('metadata_flags', []))}"
            }
        }

        # Manipulation regions
        regions = []
        if spatial_result["edge_blending"]["score"] > 0.4:
            regions.append({"region": "Face Boundary", "severity": "HIGH", "description": "Edge blending artifacts detected at face perimeter"})
        if spatial_result["lighting"]["score"] > 0.4:
            regions.append({"region": "Lighting/Shadows", "severity": "MEDIUM", "description": "Inconsistent lighting across face regions"})
        if freq_result["dct"]["anomaly_score"] > 0.5:
            regions.append({"region": "Skin Texture", "severity": "MEDIUM", "description": "Abnormal frequency patterns in skin texture region"})
        if spatial_result["texture"]["score"] > 0.4:
            regions.append({"region": "Facial Center", "severity": "HIGH" if spatial_result["texture"]["score"] > 0.7 else "MEDIUM", "description": "Texture inconsistency in face center region"})
        if not regions:
            regions.append({"region": "No Manipulation", "severity": "LOW", "description": "No specific manipulation regions identified"})

        # Technical indicators
        technical_indicators = []
        if freq_result["dwt"]["hh_entropy"] > 4:
            technical_indicators.append(f"DWT HH entropy {freq_result['dwt']['hh_entropy']:.2f} exceeds natural threshold (3.5)")
        if freq_result["dct"].get("high_freq_ratio", 0) > 0.15:
            technical_indicators.append(f"DCT high-frequency ratio {freq_result['dct']['high_freq_ratio']:.3f} anomalous")
        if spatial_result["edge_blending"].get("channel_misalignment", 0) > 5:
            technical_indicators.append(f"RGB channel misalignment score {spatial_result['edge_blending']['channel_misalignment']:.1f}")
        if not face_detected:
            technical_indicators.append("No face detected — full image analyzed (reduced accuracy)")
        technical_indicators.append(f"Modality disagreement index: {fusion['disagreement']:.3f}")
        technical_indicators.append(f"EfficientNet-B4: {fusion['modality_scores'].get('visual_efficientnet', 0):.3f}")
        technical_indicators.append(f"Frequency domain: {fusion['modality_scores'].get('frequency_domain', 0):.3f}")

        forensic_summary = self._generate_forensic_summary(fusion, all_findings, n_faces)

        return {
            # Core verdict
            "verdict": fusion["verdict"],
            "risk_level": fusion["risk_level"],
            "fake_probability": round(fusion["fake_probability"], 4),
            "confidence": round(fusion["confidence"] * 100, 1),
            "overall_score": int(fusion["fake_probability"] * 100),
            "generation_method": fusion["generation_method"],

            # Detailed breakdown
            "analysis": analysis_breakdown,
            "manipulation_regions": regions,
            "technical_indicators": technical_indicators,
            "feature_importance": fusion["feature_importance"],

            # XAI
            "xai": xai_data,
            "xai_highlights": xai_data.get("highlights", []),

            # Metadata
            "image_metadata": metadata,
            "metadata_flags": fusion.get("metadata_flags", []),

            # Summary & recommendations
            "forensic_summary": forensic_summary,
            "recommendations": self._generate_recommendations(fusion),

            # Debug info
            "face_detected": face_detected,
            "n_faces": n_faces,
            "processing_time_seconds": round(processing_time, 2),
            "filename": filename,
        }

    def _generate_forensic_summary(self, fusion: Dict, findings: List[str], n_faces: int) -> str:
        verdict = fusion["verdict"]
        prob = fusion["fake_probability"]
        method = fusion["generation_method"]
        conf = fusion["confidence"]

        if verdict == "AUTHENTIC":
            return (
                f"Analysis indicates this media is likely authentic with {conf:.0%} confidence. "
                f"All forensic modalities — spatial analysis, frequency domain, and neural network classifiers — "
                f"show no significant manipulation signals (fake probability: {prob:.1%})."
            )
        elif verdict == "SUSPICIOUS":
            top_findings = findings[:2] if findings else ["anomalies detected"]
            return (
                f"Analysis reveals suspicious patterns (fake probability: {prob:.1%}) but insufficient evidence for a definitive verdict. "
                f"Key concerns: {'; '.join(top_findings[:2])}. "
                f"Recommend additional verification with original source media."
            )
        else:
            return (
                f"Strong evidence of digital manipulation detected (fake probability: {prob:.1%}, confidence: {conf:.0%}). "
                f"Likely generated using {method}. "
                f"Multiple forensic indicators confirm artificial synthesis across spatial, frequency, and deep learning analyses."
            )

    def _generate_recommendations(self, fusion: Dict) -> List[str]:
        recs = []
        verdict = fusion["verdict"]

        if verdict == "DEEPFAKE":
            recs.extend([
                "Do not share or redistribute this media without disclosure",
                "Report to platform if found in a misleading context",
                "Cross-reference with known authentic sources of the subject",
                "Preserve original file and metadata as forensic evidence",
            ])
        elif verdict == "SUSPICIOUS":
            recs.extend([
                "Request the original high-resolution source file for re-analysis",
                "Verify provenance through trusted channels before sharing",
                "Cross-reference with reverse image search tools",
            ])
        else:
            recs.extend([
                "Media appears authentic — standard verification still recommended for high-stakes use",
                "Store with cryptographic hash for future integrity verification",
            ])

        if fusion.get("metadata_flags"):
            recs.append("Investigate missing or suspicious EXIF metadata")

        return recs
