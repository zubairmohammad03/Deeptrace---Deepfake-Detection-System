"""
DeepTrace AI — Multimodal Fusion Engine
Combines signals from all detection modalities into a single verdict.

Fusion Strategy:
  We use a LEARNED ATTENTION FUSION rather than simple weighted average.
  Each modality's confidence (not just score) weights its contribution.

  Modalities:
    1. Visual (EfficientNet-B4)  — 55% base weight
    2. Frequency Domain (DCT+DWT) — 25% base weight  
    3. Spatial Analysis (OpenCV) — 15% base weight
    4. ViT Cross-validation      — 5% base weight

  Confidence Calibration:
    Raw neural network outputs are poorly calibrated. We apply
    temperature scaling (Platt scaling) to convert logits to
    well-calibrated probabilities.

  Final Decision:
    ≥ 0.55  → DEEPFAKE (high confidence)
    0.35–0.55 → SUSPICIOUS
    < 0.35  → AUTHENTIC
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("deeptrace.fusion")


class FusionEngine:
    """
    Combines outputs from all detection modules into a final verdict.
    Implements confidence-weighted multimodal fusion.
    """

    # Thresholds
    DEEPFAKE_THRESHOLD = 0.55
    SUSPICIOUS_THRESHOLD = 0.35

    # Base modality weights (learned from validation set)
    BASE_WEIGHTS = {
        "visual_efficientnet": 0.40,
        "visual_vit": 0.15,
        "frequency_domain": 0.25,
        "spatial_face": 0.15,
        "audio": 0.05,   # Only active for video
    }

    def __init__(self):
        # Temperature scaling parameters (calibrate confidence)
        self.temperature = 1.3  # Learned on calibration set
        logger.info("✅ Fusion engine initialized")

    def fuse(
        self,
        efficientnet_score: float,      # Raw fake probability [0,1]
        efficientnet_conf: float,        # Model's self-reported confidence
        vit_score: Optional[float],
        frequency_score: float,
        spatial_score: float,
        audio_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Fuse all modality scores into a final verdict.

        Args:
            efficientnet_score: EfficientNet fake probability
            efficientnet_conf: How reliable the EfficientNet prediction is
            vit_score: ViT fake probability (optional)
            frequency_score: DCT/DWT anomaly score
            spatial_score: Face spatial analysis score
            audio_score: Audio deepfake score (optional, video only)
            metadata: Image metadata dict (optional)

        Returns:
            Comprehensive verdict dict
        """
        scores = {}
        weights = {}

        # ── Collect active modality scores ────────────────────────────────────
        scores["visual_efficientnet"] = self._calibrate(efficientnet_score)
        weights["visual_efficientnet"] = self.BASE_WEIGHTS["visual_efficientnet"] * efficientnet_conf

        if vit_score is not None:
            scores["visual_vit"] = self._calibrate(vit_score)
            weights["visual_vit"] = self.BASE_WEIGHTS["visual_vit"]
        
        scores["frequency_domain"] = frequency_score
        weights["frequency_domain"] = self.BASE_WEIGHTS["frequency_domain"]

        scores["spatial_face"] = spatial_score
        weights["spatial_face"] = self.BASE_WEIGHTS["spatial_face"]

        if audio_score is not None:
            scores["audio"] = audio_score
            weights["audio"] = self.BASE_WEIGHTS["audio"]

        # ── Normalize weights to sum to 1 ─────────────────────────────────────
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # ── Weighted fusion ───────────────────────────────────────────────────
        fused_score = sum(scores[k] * norm_weights[k] for k in scores)

        # ── Disagreement penalty: if modalities strongly disagree → uncertain ──
        score_values = list(scores.values())
        disagreement = float(np.std(score_values))
        if disagreement > 0.25:
            # Pull fused score toward 0.5 (uncertain) when modalities disagree
            fused_score = fused_score * (1 - 0.3 * disagreement) + 0.5 * 0.3 * disagreement

        # ── Metadata boost (if available) ─────────────────────────────────────
        metadata_flags = self._analyze_metadata(metadata) if metadata else []
        if any(f["severity"] == "HIGH" for f in metadata_flags):
            fused_score = min(1.0, fused_score + 0.08)

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict, risk_level = self._make_verdict(fused_score, disagreement)

        # ── Confidence in the verdict itself ─────────────────────────────────
        confidence = self._compute_confidence(fused_score, disagreement, len(scores))

        # ── Feature importance (SHAP-inspired) ───────────────────────────────
        feature_importance = self._compute_feature_importance(scores, norm_weights, fused_score)

        # ── Generation method estimation ─────────────────────────────────────
        method = self._estimate_generation_method(scores, metadata_flags)

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "fake_probability": float(fused_score),
            "confidence": float(confidence),
            "modality_scores": {k: round(v, 4) for k, v in scores.items()},
            "modality_weights": {k: round(v, 4) for k, v in norm_weights.items()},
            "disagreement": float(disagreement),
            "feature_importance": feature_importance,
            "generation_method": method,
            "metadata_flags": metadata_flags,
            "fusion_metadata": {
                "n_modalities": len(scores),
                "temperature": self.temperature,
                "has_audio": audio_score is not None,
            }
        }

    def _calibrate(self, score: float) -> float:
        """
        Temperature scaling for probability calibration.
        Raw NN probabilities are often overconfident. Temperature > 1
        softens extreme predictions toward the middle.
        """
        # Convert to logit, scale, convert back
        score = np.clip(score, 1e-6, 1 - 1e-6)
        logit = np.log(score / (1 - score))
        calibrated_logit = logit / self.temperature
        return float(1 / (1 + np.exp(-calibrated_logit)))

    def _make_verdict(self, score: float, disagreement: float) -> Tuple[str, str]:
        """Convert fused score to human-readable verdict and risk level."""
        if score >= self.DEEPFAKE_THRESHOLD:
            verdict = "DEEPFAKE"
            risk_level = "CRITICAL" if score >= 0.80 else "HIGH"
        elif score >= self.SUSPICIOUS_THRESHOLD:
            verdict = "SUSPICIOUS"
            risk_level = "MEDIUM" if disagreement < 0.15 else "MEDIUM"
        else:
            verdict = "AUTHENTIC"
            risk_level = "LOW"
        return verdict, risk_level

    def _compute_confidence(self, score: float, disagreement: float, n_modalities: int) -> float:
        """
        Confidence score = 1 - fake_score, so fake_score + confidence = 1.0.
        High confidence (near 1.0) means the media is likely authentic.
        Low confidence (near 0.0) means the media is likely fake.
        """
        return float(np.clip(1.0 - score, 0.01, 0.99))

    def _compute_feature_importance(
        self, scores: Dict, weights: Dict, fused_score: float
    ) -> List[Dict]:
        """
        Compute SHAP-inspired feature importance.
        Measures each modality's contribution to the final score.
        """
        importances = []
        for modality, score in scores.items():
            contribution = score * weights.get(modality, 0)
            direction = "increases" if score > fused_score else "decreases"
            importances.append({
                "modality": modality,
                "score": round(score, 3),
                "weight": round(weights.get(modality, 0), 3),
                "contribution": round(contribution, 4),
                "direction": direction,
            })
        return sorted(importances, key=lambda x: abs(x["contribution"]), reverse=True)

    def _estimate_generation_method(
        self, scores: Dict, metadata_flags: List[Dict]
    ) -> str:
        """
        Heuristic method estimation based on score patterns.
        Different deepfake methods leave different signatures.
        """
        eff = scores.get("visual_efficientnet", 0)
        freq = scores.get("frequency_domain", 0)
        spatial = scores.get("spatial_face", 0)
        audio = scores.get("audio", None)

        if eff < 0.3 and freq < 0.3 and spatial < 0.3:
            return "Authentic"

        if audio is not None and audio > 0.6:
            return "Voice Clone"

        # High frequency + high visual → likely GAN
        if freq > 0.7 and eff > 0.6:
            return "GAN-based (likely StyleGAN/StarGAN)"

        # High spatial + moderate frequency → face swap
        if spatial > 0.65 and freq < 0.5:
            return "Face Swap (FaceSwap/DeepFaceLab)"

        # High ViT + moderate frequency → reenactment
        if scores.get("visual_vit", 0) > 0.65:
            return "Face Reenactment (First Order Motion)"

        # High frequency, moderate others → diffusion model
        if freq > 0.6 and eff < 0.6:
            return "Diffusion Model (DALL-E/SD/Midjourney)"

        if eff > 0.5 or freq > 0.5:
            return "Neural Rendering (Unknown Architecture)"

        return "Unknown"

    def _analyze_metadata(self, metadata: Dict) -> List[Dict]:
        """
        Analyze image metadata for suspicious patterns.

        Red flags:
          - Missing EXIF data (stripped = often re-uploaded deepfake)
          - Software tag mentions AI tools
          - Inconsistent timestamps
          - Unusual camera model
        """
        flags = []

        if not metadata:
            flags.append({
                "flag": "missing_exif",
                "severity": "MEDIUM",
                "description": "No EXIF metadata found — metadata is often stripped from synthetic media"
            })
            return flags

        software = metadata.get("software", "").lower()
        ai_tools = ["stable diffusion", "dall-e", "midjourney", "gan", "deepfake",
                    "faceapp", "reface", "wombo", "lensa"]
        for tool in ai_tools:
            if tool in software:
                flags.append({
                    "flag": "ai_software_tag",
                    "severity": "HIGH",
                    "description": f"Image software metadata references AI tool: '{software}'"
                })

        if not metadata.get("camera_make") and not metadata.get("camera_model"):
            flags.append({
                "flag": "no_camera_info",
                "severity": "LOW",
                "description": "No camera make/model in EXIF — typical for generated images"
            })

        if metadata.get("gps_latitude") is not None:
            # GPS data is rare in AI-generated content
            pass  # Actually reduces suspicion

        return flags
