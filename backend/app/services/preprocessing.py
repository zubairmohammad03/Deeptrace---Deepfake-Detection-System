"""
DeepTrace AI — Preprocessing Pipeline

Handles all input normalization before inference:
  1. Image/video → face crop (MTCNN)
  2. Face alignment (affine transform to canonical 5-point landmarks)
  3. Tensor normalization (ImageNet stats)
  4. Frequency domain feature extraction (FFT + DCT)
  5. Audio → raw waveform at 16kHz
  6. Adversarial input defense (smoothing + JPEG compression simulation)

This module is the critical bridge between raw bytes and model input.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import io
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── ImageNet normalization (standard for pretrained CNNs) ─────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

to_tensor = T.Compose([
    T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

to_tensor_299 = T.Compose([
    T.Resize((299, 299)),   # XceptionNet uses 299
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass
class FaceCrop:
    """Result of face detection."""
    image: np.ndarray          # BGR uint8 face crop
    bbox: Tuple[int,int,int,int]  # x1,y1,x2,y2 in original image
    confidence: float
    landmarks: Optional[np.ndarray]  # 5×2 facial landmarks
    face_index: int


# ── Face Detection & Alignment ────────────────────────────────────────────────

def detect_faces_opencv(image_bgr: np.ndarray) -> List[FaceCrop]:
    """
    Fallback face detector using OpenCV's DNN face detector.
    Used when facenet_pytorch (MTCNN) is not installed.
    """
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0),
    )
    try:
        net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
        net.setInput(blob)
        detections = net.forward()
    except Exception:
        # If model files not present, detect full frame
        return [FaceCrop(
            image=image_bgr, bbox=(0, 0, w, h),
            confidence=0.5, landmarks=None, face_index=0
        )]

    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < settings.FACE_CONFIDENCE_MIN:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        margin = int((x2 - x1) * settings.FACE_CROP_MARGIN)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        faces.append(FaceCrop(
            image=image_bgr[y1:y2, x1:x2],
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            landmarks=None,
            face_index=i,
        ))
    return faces


def detect_faces_mtcnn(image_rgb: np.ndarray, mtcnn) -> List[FaceCrop]:
    """
    Face detection using MTCNN (better accuracy, returns 5-point landmarks).
    Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    pil_img = Image.fromarray(image_rgb)
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)

    if boxes is None:
        return []

    h, w = image_rgb.shape[:2]
    faces = []
    for i, (box, prob, lmk) in enumerate(zip(boxes, probs, landmarks)):
        if prob < settings.FACE_CONFIDENCE_MIN:
            continue
        x1, y1, x2, y2 = box.astype(int)
        margin_x = int((x2 - x1) * settings.FACE_CROP_MARGIN)
        margin_y = int((y2 - y1) * settings.FACE_CROP_MARGIN)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        face_bgr = cv2.cvtColor(image_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        faces.append(FaceCrop(
            image=face_bgr,
            bbox=(x1, y1, x2, y2),
            confidence=float(prob),
            landmarks=lmk,
            face_index=i,
        ))
    return faces


def align_face(face_crop: FaceCrop, output_size: int = 224) -> np.ndarray:
    """
    Align face using 5-point landmarks via affine transform.
    Normalizes eye positions to canonical coordinates — removes pose variation
    so the model focuses on texture/frequency artifacts, not head orientation.

    Reference eye positions (from FFHQ alignment):
        left_eye  → (0.31, 0.46) * output_size
        right_eye → (0.69, 0.46) * output_size
    """
    if face_crop.landmarks is None:
        return cv2.resize(face_crop.image, (output_size, output_size))

    src = face_crop.landmarks[:2].astype(np.float32)  # left & right eye
    dst = np.array([
        [0.31 * output_size, 0.46 * output_size],
        [0.69 * output_size, 0.46 * output_size],
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(face_crop.image, M, (output_size, output_size))
    return aligned


# ── Adversarial Defense ───────────────────────────────────────────────────────

def apply_adversarial_defense(image_np: np.ndarray) -> np.ndarray:
    """
    Input pre-processing defenses against adversarial patch attacks (FGSM/PGD).

    Two techniques combined:
    1. Gaussian smoothing — removes high-frequency adversarial perturbations
    2. JPEG re-compression simulation — disrupts adversarial pixel patterns
       without significantly affecting natural image statistics

    Why this works: Adversarial perturbations are typically high-frequency,
    small-magnitude noise. Both operations suppress these while preserving
    the coarse semantic content needed for deepfake detection.
    """
    defended = image_np.copy()

    if settings.ADVERSARIAL_SMOOTHING:
        # Kernel size 3 = subtle smoothing; enough to destroy adversarial noise
        defended = cv2.GaussianBlur(defended, (3, 3), sigmaX=1.0)

    if settings.ADVERSARIAL_JPEG_COMPRESSION:
        # Simulate JPEG compression at quality=85
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, encoded = cv2.imencode(".jpg", defended, encode_params)
        defended = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return defended


# ── Frequency Domain Features ─────────────────────────────────────────────────

def extract_fft_features(image_np: np.ndarray) -> np.ndarray:
    """
    2D Fourier Transform analysis.

    GAN-generated images produce characteristic spectral artifacts:
    - Grid-like patterns in frequency domain (from upsampling/transposed conv)
    - Abnormal high-frequency energy distribution
    - Spectral peaks at regular intervals

    Returns: [H, W, 1] magnitude spectrum (log-scaled, normalized to [0,1])
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude[..., np.newaxis]


def extract_dct_features(image_np: np.ndarray) -> np.ndarray:
    """
    Block DCT analysis (JPEG-style 8×8 blocks).

    Deepfakes processed through JPEG compression show characteristic
    coefficient distributions. Face swaps often have discontinuities
    at block boundaries (blocking artifacts) that are invisible to
    the human eye but detectable in DCT domain.

    Returns: [H//8, W//8, 64] — DCT coefficients for each block
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    block_size = settings.DCT_BLOCK_SIZE

    # Crop to multiple of block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    gray = gray[:h_crop, :w_crop]

    n_blocks_h = h_crop // block_size
    n_blocks_w = w_crop // block_size
    dct_map = np.zeros((n_blocks_h, n_blocks_w, block_size * block_size))

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = cv2.dct(block)
            dct_map[i, j] = dct_block.flatten()

    return dct_map


# ── Image → Tensor Pipeline ───────────────────────────────────────────────────

def preprocess_image(
    image_bytes: bytes,
    mtcnn=None,
    target_size: int = 224,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray], List[FaceCrop]]:
    """
    Full preprocessing pipeline for a single image.

    Returns:
        face_tensor_224:  [1, 3, 224, 224] normalized tensor for EfficientNet/ViT
        face_tensor_299:  [1, 3, 299, 299] for XceptionNet
        fft_features:     [H, W, 1] FFT magnitude map
        face_crops:       List of detected FaceCrop objects
    """
    # Decode bytes → numpy BGR
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image bytes")

    # Apply adversarial defense
    image_bgr = apply_adversarial_defense(image_bgr)

    # Detect faces
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if mtcnn is not None:
        faces = detect_faces_mtcnn(image_rgb, mtcnn)
    else:
        faces = detect_faces_opencv(image_bgr)

    if not faces:
        # No face detected — use full image (e.g., GAN-generated landscape)
        logger.warning("No face detected — using full image for analysis")
        faces = [FaceCrop(image=image_bgr, bbox=(0,0,image_bgr.shape[1],image_bgr.shape[0]), confidence=0.5, landmarks=None, face_index=0)]

    # Use the most confident face
    primary_face = max(faces, key=lambda f: f.confidence)

    # Align face
    aligned_bgr = align_face(primary_face, output_size=target_size)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(aligned_rgb)

    face_tensor_224 = to_tensor(pil_face).unsqueeze(0)    # [1, 3, 224, 224]
    face_tensor_299 = to_tensor_299(pil_face).unsqueeze(0) # [1, 3, 299, 299]
    fft_feats = extract_fft_features(aligned_bgr)

    return face_tensor_224, face_tensor_299, fft_feats, faces


# ── Video → Frame Pipeline ────────────────────────────────────────────────────

def extract_video_frames(
    video_bytes: bytes,
    n_frames: int = 16,
    stride: int = 5,
) -> List[np.ndarray]:
    """
    Extract evenly-spaced frames from a video for temporal analysis.

    Strategy: Sample frames with stride to capture temporal artifacts.
    Temporal artifacts (flickering eyes, inconsistent head motion) are
    the strongest signal for face-reenactment deepfakes.

    Returns: List of BGR numpy arrays
    """
    # Write to temp buffer (OpenCV needs seekable stream)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample frames uniformly, respecting stride
    sampled_indices = list(range(0, min(total_frames, n_frames * stride), stride))[:n_frames]

    frames = []
    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    os.unlink(tmp_path)

    logger.info(f"Extracted {len(frames)} frames from video (total: {total_frames}, fps: {fps:.1f})")
    return frames


# ── Audio Preprocessing ───────────────────────────────────────────────────────

def preprocess_audio(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[torch.Tensor]:
    """
    Audio → normalized waveform tensor for RawNet2.

    Processing:
    1. Decode audio (WAV/MP3/FLAC via soundfile)
    2. Resample to 16kHz (RawNet2 standard)
    3. Convert to mono if stereo
    4. Normalize to [-1, 1]
    5. Pad/trim to fixed 4-second window (64000 samples)

    Returns: [1, 1, 64000] float32 tensor
    """
    try:
        import soundfile as sf
        import io
        waveform, sr = sf.read(io.BytesIO(audio_bytes))

        # Stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sr != sample_rate:
            import resampy
            waveform = resampy.resample(waveform, sr, sample_rate)

        # Normalize
        waveform = waveform.astype(np.float32)
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        # Fixed-length: 4 seconds = 64000 samples
        target_len = sample_rate * 4
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        else:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))

        tensor = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        return tensor

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return None
