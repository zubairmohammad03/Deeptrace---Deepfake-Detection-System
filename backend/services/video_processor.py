"""
DeepTrace AI — Video Processing Pipeline
Handles video ingestion, frame extraction, and temporal consistency analysis.

Why temporal analysis matters:
  A deepfake might fool single-frame detectors but fail temporal consistency.
  Real video has:
    - Consistent identity across frames (same person throughout)
    - Natural optical flow (motion vectors follow physics)
    - Consistent lighting changes over time
    - Natural micro-expressions and blinking patterns

  Deepfakes often have:
    - Flickering artifacts between frames (inconsistent generation)
    - Abrupt identity shifts when the face moves to extreme angles
    - Unnatural optical flow at face boundaries
    - Missing or unrealistic blinks (older models)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
import logging
import tempfile
import os

logger = logging.getLogger("deeptrace.video")


class VideoProcessor:
    """
    Extracts frames from video files and performs temporal consistency analysis.
    """

    def __init__(
        self,
        target_fps: float = 2.0,       # Sample 2 frames per second
        max_frames: int = 32,           # Maximum frames to analyze
        face_margin: float = 0.3,       # Face crop margin (30%)
    ):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.face_margin = face_margin

    def process(self, video_path: Path) -> Dict:
        """
        Full video processing pipeline.

        Steps:
          1. Open video and extract metadata
          2. Uniformly sample frames at target FPS
          3. Detect and crop faces in each frame
          4. Extract temporal features (optical flow, flickering)
          5. Return frames + temporal analysis

        Returns:
            {
                "frames": List of (frame_idx, face_crop) tuples,
                "metadata": Video metadata dict,
                "temporal": Temporal consistency analysis,
                "face_tracks": Face tracking results,
            }
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        metadata = self._extract_metadata(cap)
        logger.info(f"📹 Video: {metadata['duration']:.1f}s @ {metadata['fps']:.1f}fps, {metadata['width']}×{metadata['height']}")

        # Compute frame sampling indices
        sample_indices = self._get_sample_indices(metadata)
        logger.info(f"📊 Sampling {len(sample_indices)} frames")

        # Extract sampled frames
        raw_frames = self._extract_frames(cap, sample_indices)
        cap.release()

        if not raw_frames:
            raise ValueError("No frames could be extracted from the video")

        # Face detection on all frames
        face_crops, face_boxes = self._extract_faces(raw_frames)

        # Temporal consistency analysis
        temporal = self._analyze_temporal_consistency(raw_frames, face_boxes)

        # Flickering analysis
        flickering = self._analyze_flickering(face_crops)

        # Optical flow analysis
        optical_flow = self._analyze_optical_flow(raw_frames[:min(16, len(raw_frames))])

        return {
            "frames": face_crops,         # List of (frame_idx, face_crop_bgr) tuples
            "raw_frames": raw_frames,     # List of (frame_idx, full_frame) tuples
            "metadata": metadata,
            "temporal": temporal,
            "flickering": flickering,
            "optical_flow": optical_flow,
            "n_faces_detected": sum(1 for _, f in face_crops if f is not None),
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _extract_metadata(self, cap: cv2.VideoCapture) -> Dict:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return {
            "fps": fps, "total_frames": total_frames,
            "width": width, "height": height,
            "duration": duration, "codec": codec,
            "resolution": f"{width}×{height}",
        }

    # ── Frame Sampling ────────────────────────────────────────────────────────

    def _get_sample_indices(self, metadata: Dict) -> List[int]:
        """
        Compute uniformly spaced sample indices.
        Skips first 10% and last 10% of video (often title/end cards).
        """
        total = metadata["total_frames"]
        fps = metadata["fps"]

        start_frame = int(total * 0.10)
        end_frame = int(total * 0.90)

        # Sample every N frames to hit target_fps
        step = max(1, int(fps / self.target_fps))
        indices = list(range(start_frame, end_frame, step))

        # Cap at max_frames, but keep them evenly distributed
        if len(indices) > self.max_frames:
            idx = np.linspace(0, len(indices) - 1, self.max_frames, dtype=int)
            indices = [indices[i] for i in idx]

        return indices

    # ── Frame Extraction ──────────────────────────────────────────────────────

    def _extract_frames(
        self, cap: cv2.VideoCapture, indices: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """Efficiently extract frames at specified indices using seek."""
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append((idx, frame))
        return frames

    # ── Face Detection & Cropping ─────────────────────────────────────────────

    def _extract_faces(
        self, frames: List[Tuple[int, np.ndarray]]
    ) -> Tuple[List[Tuple[int, Optional[np.ndarray]]], List[Optional[Tuple]]]:
        """
        Detect and crop faces from each frame.
        Returns aligned face crops for analysis.
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        crops = []
        boxes = []

        for frame_idx, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if len(detected) > 0:
                # Take the largest face
                x, y, w, h = max(detected, key=lambda f: f[2] * f[3])
                margin_x = int(w * self.face_margin)
                margin_y = int(h * self.face_margin)
                fh, fw = frame.shape[:2]
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(fw, x + w + margin_x)
                y2 = min(fh, y + h + margin_y)
                crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
                crops.append((frame_idx, crop))
                boxes.append((x, y, w, h))
            else:
                crops.append((frame_idx, None))
                boxes.append(None)

        return crops, boxes

    # ── Temporal Consistency ──────────────────────────────────────────────────

    def _analyze_temporal_consistency(
        self,
        frames: List[Tuple[int, np.ndarray]],
        face_boxes: List[Optional[Tuple]],
    ) -> Dict:
        """
        Analyze whether the face identity is consistent across frames.
        
        Method: Compare color histogram similarity between consecutive face crops.
        Real videos: histogram changes slowly and smoothly.
        Deepfakes: can show abrupt identity shifts.
        """
        if len(frames) < 2:
            return {"score": 0.0, "consistency": 1.0, "findings": []}

        similarities = []
        prev_hist = None

        for (_, frame), box in zip(frames, face_boxes):
            if box is None:
                continue
            x, y, w, h = box
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                similarities.append(float(sim))
            prev_hist = hist

        if not similarities:
            return {"score": 0.0, "consistency": 1.0, "findings": ["Insufficient frames for temporal analysis"]}

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))

        # Abrupt drops in similarity → identity inconsistency
        drops = [s for s in similarities if s < 0.6]
        n_drops = len(drops)

        score = 0.0
        findings = []
        if mean_sim < 0.75:
            score += 0.4
            findings.append(f"Low inter-frame identity consistency ({mean_sim:.2f}) — possible face flickering")
        if std_sim > 0.15:
            score += 0.3
            findings.append(f"High temporal variance (σ={std_sim:.2f}) — inconsistent appearance across frames")
        if n_drops > len(similarities) * 0.15:
            score += 0.3
            findings.append(f"{n_drops} abrupt identity drops detected — typical of video deepfakes at extreme angles")

        if not findings:
            findings.append(f"Temporal identity consistency: {mean_sim:.2f} (natural)")

        return {
            "score": float(min(1.0, score)),
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "n_abrupt_drops": n_drops,
            "consistency": mean_sim,
            "findings": findings,
        }

    # ── Flickering Analysis ───────────────────────────────────────────────────

    def _analyze_flickering(
        self, face_crops: List[Tuple[int, Optional[np.ndarray]]]
    ) -> Dict:
        """
        Detect high-frequency flickering in face crops.
        GANs that generate each frame independently produce subtle
        frame-to-frame noise not present in real video.
        """
        valid_crops = [crop for _, crop in face_crops if crop is not None]
        if len(valid_crops) < 3:
            return {"score": 0.0, "findings": []}

        # Compute per-pixel variance across frames
        stack = np.stack([c.astype(float) for c in valid_crops[:16]])
        pixel_var = float(stack.var(axis=0).mean())

        # High variance at face center (not edges) = flickering
        center_crops = [c[56:168, 56:168] for c in valid_crops[:16]]
        center_stack = np.stack([c.astype(float) for c in center_crops])
        center_var = float(center_stack.var(axis=0).mean())

        score = min(1.0, center_var / 200.0)
        findings = []
        if score > 0.5:
            findings.append(f"Frame-to-frame flickering detected (variance: {center_var:.1f}) — GAN temporal inconsistency")

        return {"score": score, "pixel_variance": pixel_var, "center_variance": center_var, "findings": findings}

    # ── Optical Flow ──────────────────────────────────────────────────────────

    def _analyze_optical_flow(
        self, frames: List[Tuple[int, np.ndarray]]
    ) -> Dict:
        """
        Compute dense optical flow between consecutive frames.
        Deepfakes can have unnatural motion vectors at face boundaries
        where the generated face doesn't perfectly track head motion.
        """
        valid_frames = [f for _, f in frames if f is not None]
        if len(valid_frames) < 2:
            return {"score": 0.0, "findings": []}

        flow_magnitudes = []
        boundary_anomalies = []

        for i in range(min(8, len(valid_frames) - 1)):
            gray1 = cv2.cvtColor(valid_frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(valid_frames[i+1], cv2.COLOR_BGR2GRAY)
            gray1 = cv2.resize(gray1, (128, 128))
            gray2 = cv2.resize(gray2, (128, 128))

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(float(magnitude.mean()))

            # Check flow consistency at face center vs boundary
            center_flow = magnitude[32:96, 32:96].mean()
            border_flow = np.concatenate([
                magnitude[:16, :].flatten(),
                magnitude[112:, :].flatten(),
                magnitude[:, :16].flatten(),
                magnitude[:, 112:].flatten()
            ]).mean()

            if center_flow > 0:
                boundary_ratio = border_flow / center_flow
                boundary_anomalies.append(boundary_ratio)

        if not flow_magnitudes:
            return {"score": 0.0, "findings": []}

        mean_flow = float(np.mean(flow_magnitudes))
        mean_boundary = float(np.mean(boundary_anomalies)) if boundary_anomalies else 1.0

        score = 0.0
        findings = []
        if mean_boundary > 2.0:
            score = min(1.0, (mean_boundary - 1.5) / 2.0)
            findings.append(f"Optical flow boundary anomaly ({mean_boundary:.1f}×) — face motion doesn't match background")

        return {
            "score": score,
            "mean_flow_magnitude": mean_flow,
            "boundary_flow_ratio": mean_boundary,
            "findings": findings,
        }
