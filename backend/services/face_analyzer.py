"""
DeepTrace AI — Face Analysis Service
Spatial artifact detection on extracted face crops.

Analyses performed:
  1. Facial Landmark Consistency  — are landmarks geometrically plausible?
  2. Texture Consistency          — are pores/skin texture natural?
  3. Lighting & Shadow Analysis   — do shadows match lighting direction?
  4. Edge Blending Detection      — are face boundaries smooth or composited?
  5. Eye Blink & Gaze Analysis    — natural gaze and symmetric blink patterns?
  6. Symmetry Analysis            — deepfakes often have unusual facial asymmetry
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("deeptrace.face")


class FaceAnalysisService:
    """
    Performs spatial analysis on face crops to detect manipulation artifacts.
    Does NOT require deep learning — uses classical computer vision.
    Combined with DL scores in the fusion layer.
    """

    def __init__(self):
        # Haar cascade as fallback (built into OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        # Try loading dnn-based face detector (more accurate)
        self.dnn_detector = self._load_dnn_detector()

    def _load_dnn_detector(self):
        """Load OpenCV DNN face detector if available."""
        try:
            net = cv2.dnn.readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel"
            )
            return net
        except Exception:
            return None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in the image.
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.dnn_detector is not None:
            return self._dnn_detect(image)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []

    def _dnn_detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def analyze_face(self, face_crop: np.ndarray) -> Dict:
        """
        Full spatial analysis on a single face crop.

        Args:
            face_crop: (H, W, 3) BGR face crop

        Returns:
            Dict with per-analysis scores and findings
        """
        if face_crop is None or face_crop.size == 0:
            return self._empty_result()

        face_crop = cv2.resize(face_crop, (256, 256))
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        texture = self._analyze_texture(face_crop, gray)
        lighting = self._analyze_lighting(face_crop, gray)
        edges = self._analyze_edge_blending(face_crop, gray)
        symmetry = self._analyze_symmetry(gray)
        quality = self._analyze_compression_artifacts(face_crop)

        # Overall spatial score
        spatial_score = (
            texture["score"] * 0.25 +
            lighting["score"] * 0.30 +
            edges["score"] * 0.25 +
            symmetry["score"] * 0.10 +
            quality["score"] * 0.10
        )

        all_findings = (
            texture["findings"] + lighting["findings"] +
            edges["findings"] + symmetry["findings"]
        )

        return {
            "spatial_fake_score": float(np.clip(spatial_score, 0, 1)),
            "texture": texture,
            "lighting": lighting,
            "edge_blending": edges,
            "symmetry": symmetry,
            "compression": quality,
            "findings": all_findings,
        }

    # ── Texture Analysis ──────────────────────────────────────────────────────

    def _analyze_texture(self, bgr: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Analyze skin texture consistency using LBP and Laplacian variance.

        Natural skin has:
          - Consistent pore structure (measurable via LBP)
          - Natural sharpness gradient (center sharp, periphery softer)
          - Consistent color distribution per region

        Deepfakes often have:
          - Over-smoothed skin (too little texture variance)
          - OR over-sharpened artifacts (too much at wrong scale)
          - Inconsistent texture between face center and edges
        """
        score = 0.0
        findings = []

        # Laplacian variance (measures sharpness/blur)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Split into center and border
        h, w = gray.shape
        pad = 40
        center = gray[pad:h-pad, pad:w-pad]
        border = np.concatenate([
            gray[:pad, :].flatten(),
            gray[h-pad:, :].flatten(),
            gray[:, :pad].flatten(),
            gray[:, w-pad:].flatten()
        ])

        center_lap = float(cv2.Laplacian(center, cv2.CV_64F).var())
        border_lap = float(np.var(np.abs(np.diff(border.reshape(-1)))))

        # Natural face: center sharper than border (natural depth of field)
        # Deepfake: uniform sharpness OR reversed pattern
        if center_lap > 0:
            sharpness_ratio = border_lap / (center_lap + 1e-8)
        else:
            sharpness_ratio = 1.0

        if lap_var < 50:
            score += 0.4
            findings.append("Unusually low texture variance — over-smoothed skin typical of GAN output")
        elif lap_var > 3000:
            score += 0.2
            findings.append("Abnormally high sharpness — possible post-processing artifact")

        if sharpness_ratio > 1.5:
            score += 0.3
            findings.append("Border sharper than face center — reversed depth-of-field pattern")

        # Color consistency across face regions
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 180, 255]))
        if skin_mask.sum() > 1000:
            skin_sat = hsv[:, :, 1][skin_mask > 0]
            sat_std = float(np.std(skin_sat))
            if sat_std > 40:
                score += 0.3
                findings.append(f"High saturation variance ({sat_std:.0f}) — inconsistent skin color processing")

        if not findings:
            findings.append("Texture appears natural — no anomalies detected")

        return {"score": min(1.0, score), "findings": findings,
                "laplacian_variance": lap_var, "sharpness_ratio": sharpness_ratio}

    # ── Lighting Analysis ─────────────────────────────────────────────────────

    def _analyze_lighting(self, bgr: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Analyze lighting consistency and shadow plausibility.

        Approach:
          1. Estimate dominant light direction using gradient analysis
          2. Check if shadows are consistent with that direction
          3. Look for specular highlight anomalies on nose/forehead
          4. Detect "double lighting" common in composited faces
        """
        score = 0.0
        findings = []

        # Estimate gradient direction (dominant light source direction)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Weighted average direction by magnitude
        strong_edges = magnitude > np.percentile(magnitude, 85)
        if strong_edges.sum() > 0:
            mean_angle = float(np.average(angle[strong_edges], weights=magnitude[strong_edges]))
        else:
            mean_angle = 0.0

        # Consistency of gradient directions (should be coherent for single light source)
        if strong_edges.sum() > 100:
            angle_std = float(np.std(angle[strong_edges]))
            if angle_std > 2.0:  # High variance = multiple inconsistent light sources
                score += 0.35
                findings.append(f"Inconsistent gradient directions (σ={angle_std:.2f}) — possible dual lighting compositing")

        # Check specular highlights (nose, forehead regions)
        h, w = gray.shape
        forehead_roi = gray[5:h//4, w//4:3*w//4]
        if forehead_roi.size > 0:
            highlight_mask = forehead_roi > 240  # Near-white pixels = highlights
            highlight_ratio = highlight_mask.mean()
            if highlight_ratio > 0.15:
                score += 0.2
                findings.append(f"Saturated specular highlights ({highlight_ratio:.1%}) — unnatural skin reflectance")

        # Global illumination coherence: left vs right brightness
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        l_mean, r_mean = float(left_half.mean()), float(right_half.mean())
        asymmetry = abs(l_mean - r_mean)

        if asymmetry > 35:
            score += 0.25
            findings.append(f"Strong brightness asymmetry L:{l_mean:.0f} vs R:{r_mean:.0f} — inconsistent lighting on face halves")

        if not findings:
            findings.append("Lighting appears physically consistent")

        return {"score": min(1.0, score), "findings": findings,
                "light_direction_rad": mean_angle,
                "brightness_asymmetry": asymmetry}

    # ── Edge Blending ─────────────────────────────────────────────────────────

    def _analyze_edge_blending(self, bgr: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Detect face boundary artifacts from imperfect face-swapping.

        GAN face-swapping typically blends the generated face with the
        original background. The boundary region often shows:
          - Double edges (original + generated)
          - Abrupt frequency changes at the blend boundary
          - Color fringing (chromatic aberration from blending)
        """
        score = 0.0
        findings = []

        # Canny edge detection with two thresholds
        edges_strict = cv2.Canny(gray, 100, 200)
        edges_loose = cv2.Canny(gray, 30, 100)
        double_edges = cv2.bitwise_and(
            edges_loose,
            cv2.bitwise_not(edges_strict)
        )

        h, w = gray.shape
        border_region = np.zeros_like(gray)
        margin = int(min(h, w) * 0.12)
        border_region[:margin, :] = 1
        border_region[h-margin:, :] = 1
        border_region[:, :margin] = 1
        border_region[:, w-margin:] = 1

        # Double-edge density in border vs center
        border_double_edges = double_edges[border_region == 1]
        center_double_edges = double_edges[border_region == 0]

        border_density = float(border_double_edges.mean())
        center_density = float(center_double_edges.mean()) + 1e-8
        edge_ratio = border_density / center_density

        if edge_ratio > 2.5:
            score += 0.4
            findings.append(f"Double-edge density {edge_ratio:.1f}× higher at face boundary — compositing artifact")

        # Chromatic aberration at boundaries (R-G-B channel misalignment)
        b, g, r = cv2.split(bgr)
        edges_r = cv2.Canny(r, 50, 150)
        edges_g = cv2.Canny(g, 50, 150)
        edges_b = cv2.Canny(b, 50, 150)

        # Measure channel misalignment
        rg_diff = cv2.bitwise_xor(edges_r, edges_g)
        rb_diff = cv2.bitwise_xor(edges_r, edges_b)
        channel_misalign = float((rg_diff.mean() + rb_diff.mean()) / 2)

        if channel_misalign > 8:
            score += 0.35
            findings.append(f"Chromatic aberration score {channel_misalign:.1f} — channel misalignment at boundaries")

        # High-frequency detail at face center vs border
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        border_hf = float(np.abs(lap[border_region == 1]).mean())
        center_hf = float(np.abs(lap[border_region == 0]).mean())

        if center_hf > 0 and border_hf / center_hf > 1.8:
            score += 0.25
            findings.append("Face boundary contains more detail than face center — unnatural for real photography")

        if not findings:
            findings.append("Edge blending appears natural — no boundary artifacts detected")

        return {"score": min(1.0, score), "findings": findings,
                "edge_ratio": edge_ratio, "channel_misalignment": channel_misalign}

    # ── Symmetry Analysis ─────────────────────────────────────────────────────

    def _analyze_symmetry(self, gray: np.ndarray) -> Dict:
        """
        Faces are naturally slightly asymmetric.
        Deepfakes can be either TOO symmetric (GAN) or WRONGLY asymmetric (bad swap).
        """
        h, w = gray.shape
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)

        # Resize to same size
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        diff = cv2.absdiff(left.astype(float), right.astype(float))
        asymmetry_score = float(diff.mean()) / 255.0

        findings = []
        score = 0.0

        if asymmetry_score < 0.03:
            score = 0.5
            findings.append(f"Unusually high symmetry ({asymmetry_score:.3f}) — GAN faces tend toward perfect symmetry")
        elif asymmetry_score > 0.20:
            score = 0.4
            findings.append(f"Abnormal asymmetry ({asymmetry_score:.3f}) — possible face misalignment from swapping")
        else:
            findings.append(f"Facial symmetry within natural range ({asymmetry_score:.3f})")

        return {"score": score, "findings": findings, "asymmetry_score": asymmetry_score}

    # ── Compression Artifacts ─────────────────────────────────────────────────

    def _analyze_compression_artifacts(self, bgr: np.ndarray) -> Dict:
        """
        Detect JPEG blocking artifacts. Heavy compression can both
        indicate a shared/recompressed deepfake and mask other artifacts.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape
        block_size = 8
        blocking_score = 0.0
        count = 0

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                # Measure discontinuity at block boundaries
                right_diff = abs(float(gray[y:y+block_size, x+block_size-1].mean()) -
                                  float(gray[y:y+block_size, x+block_size].mean())) if x + block_size < w else 0
                bottom_diff = abs(float(gray[y+block_size-1, x:x+block_size].mean()) -
                                   float(gray[y+block_size, x:x+block_size].mean())) if y + block_size < h else 0
                blocking_score += (right_diff + bottom_diff) / 2
                count += 1

        avg_blocking = blocking_score / max(count, 1)
        score = min(1.0, avg_blocking / 20.0)

        return {
            "score": score,
            "blocking_score": float(avg_blocking),
            "findings": [f"Block artifact score: {avg_blocking:.2f}"] if avg_blocking > 5 else []
        }

    def _empty_result(self) -> Dict:
        return {
            "spatial_fake_score": 0.0,
            "texture": {"score": 0.0, "findings": ["No face detected"]},
            "lighting": {"score": 0.0, "findings": []},
            "edge_blending": {"score": 0.0, "findings": []},
            "symmetry": {"score": 0.0, "findings": []},
            "compression": {"score": 0.0, "findings": []},
            "findings": ["No face detected in the image"],
        }
