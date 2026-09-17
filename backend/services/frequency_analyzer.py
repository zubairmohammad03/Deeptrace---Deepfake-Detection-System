"""
DeepTrace AI — Frequency Domain Analyzer
Detects deepfake artifacts in the frequency domain using DCT and DWT.

WHY FREQUENCY ANALYSIS?
  Most deepfake generators operate in pixel space. When you convert to
  frequency domain (DCT/DWT), GAN-generated images show characteristic
  spectral patterns — particularly at high frequencies — that don't appear
  in natural photographs. This approach is COMPRESSION-RESISTANT because
  spectral fingerprints survive JPEG/H.264 compression.

Key techniques:
  1. DCT (Discrete Cosine Transform): Same as used in JPEG compression.
     Deepfakes show abnormal energy distribution at high frequencies.
     
  2. DWT (Discrete Wavelet Transform): Multi-resolution analysis.
     HH subband (diagonal details) is particularly discriminative.

  3. Azimuthal Power Spectrum: The 1D power spectrum of deepfakes
     shows a characteristic "fingerprint" at specific frequencies.

References:
  - "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues" (Li et al., ECCV 2021)
  - "FRepGAN: Deepfake Detection Using Frequency Representation" 
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger("deeptrace.frequency")


class FrequencyDomainAnalyzer:
    """
    Analyzes images in the frequency domain to detect deepfake artifacts
    that survive lossy compression.
    """

    def __init__(self):
        self.dct_anomaly_threshold = 0.15
        self.high_freq_ratio_threshold = 0.22
        self.wavelet_entropy_threshold = 3.5

    def analyze(self, face_crop: np.ndarray) -> Dict:
        """
        Full frequency domain analysis pipeline.

        Args:
            face_crop: (H, W, 3) uint8 face crop

        Returns:
            Dict with scores, findings, and normalized spectral features
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_norm = gray / 255.0

        dct_result = self._analyze_dct(gray_norm)
        dwt_result = self._analyze_dwt(gray_norm)
        spectrum_result = self._analyze_power_spectrum(gray_norm)
        noise_result = self._analyze_noise_pattern(face_crop)

        # Weighted combination
        composite_score = (
            dct_result["anomaly_score"] * 0.35 +
            dwt_result["anomaly_score"] * 0.30 +
            spectrum_result["anomaly_score"] * 0.20 +
            noise_result["anomaly_score"] * 0.15
        )

        findings = []
        if dct_result["anomaly_score"] > 0.6:
            findings.append(f"Abnormal DCT coefficient distribution (score: {dct_result['anomaly_score']:.2f})")
        if dwt_result["anomaly_score"] > 0.6:
            findings.append(f"DWT HH subband entropy anomaly ({dwt_result['hh_entropy']:.2f} vs expected ≤3.5)")
        if spectrum_result["anomaly_score"] > 0.6:
            findings.append(f"GAN spectral fingerprint detected at {spectrum_result['peak_freq']:.1f} cycles/pixel")
        if noise_result["anomaly_score"] > 0.6:
            findings.append("PRNU noise pattern inconsistency — likely composited image")
        if not findings:
            findings.append("No significant frequency-domain anomalies detected")

        return {
            "composite_score": float(np.clip(composite_score, 0, 1)),
            "fake_probability": float(np.clip(composite_score, 0, 1)),
            "dct": dct_result,
            "dwt": dwt_result,
            "spectrum": spectrum_result,
            "noise": noise_result,
            "findings": findings,
            "spectral_features": self._extract_feature_vector(dct_result, dwt_result, spectrum_result),
        }

    # ── DCT Analysis ──────────────────────────────────────────────────────────

    def _analyze_dct(self, gray: np.ndarray) -> Dict:
        """
        Block-wise DCT analysis.
        
        Natural images follow a predictable distribution where energy
        concentrates in low frequencies. GANs disrupt this pattern,
        particularly in 8×8 DCT blocks (same as JPEG).
        """
        h, w = gray.shape
        block_size = 8
        high_freq_energies = []
        low_freq_energies = []
        dc_coefficients = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block.astype(np.float32))

                # DC coefficient (top-left)
                dc_coefficients.append(abs(dct_block[0, 0]))

                # Low-freq energy (top-left 4×4)
                low_freq_energies.append(np.sum(dct_block[:4, :4] ** 2))

                # High-freq energy (bottom-right 4×4)
                high_freq_energies.append(np.sum(dct_block[4:, 4:] ** 2))

        if not high_freq_energies:
            return {"anomaly_score": 0.0, "high_freq_ratio": 0.0}

        total_energy = np.array(low_freq_energies) + np.array(high_freq_energies) + 1e-8
        hf_ratio = np.array(high_freq_energies) / total_energy
        mean_hf_ratio = float(np.mean(hf_ratio))

        # Deepfakes typically have higher high-frequency energy due to
        # generator artifacts at sharp boundaries
        anomaly_score = min(1.0, max(0.0, (mean_hf_ratio - 0.08) / 0.20))

        # DCT coefficient variance — deepfakes show unusual variance patterns
        dc_variance = float(np.var(dc_coefficients))
        dc_anomaly = min(1.0, max(0.0, (dc_variance - 500) / 2000))

        return {
            "anomaly_score": float((anomaly_score * 0.7 + dc_anomaly * 0.3)),
            "high_freq_ratio": mean_hf_ratio,
            "dc_variance": dc_variance,
            "expected_hf_ratio": "0.05–0.12 (natural image)",
            "observed_hf_ratio": f"{mean_hf_ratio:.3f}",
        }

    # ── DWT Analysis ──────────────────────────────────────────────────────────

    def _analyze_dwt(self, gray: np.ndarray) -> Dict:
        """
        Haar DWT decomposition.
        
        Decomposes image into:
          LL: Low-Low (approximation) — smooth regions
          LH: Low-High (horizontal edges)
          HL: High-Low (vertical edges)
          HH: High-High (diagonal details) ← MOST DISCRIMINATIVE

        GAN-generated faces show abnormal entropy in HH subband
        because the generator must hallucinate fine diagonal textures.
        """
        # Manual Haar DWT (2-level)
        def haar_dwt(img):
            h, w = img.shape
            h, w = h - h % 2, w - w % 2
            img = img[:h, :w]
            rows_even, rows_odd = img[0::2, :], img[1::2, :]
            LL = (rows_even[:, 0::2] + rows_odd[:, 0::2] + rows_even[:, 1::2] + rows_odd[:, 1::2]) / 4
            LH = (rows_even[:, 0::2] + rows_odd[:, 0::2] - rows_even[:, 1::2] - rows_odd[:, 1::2]) / 4
            HL = (rows_even[:, 0::2] - rows_odd[:, 0::2] + rows_even[:, 1::2] - rows_odd[:, 1::2]) / 4
            HH = (rows_even[:, 0::2] - rows_odd[:, 0::2] - rows_even[:, 1::2] + rows_odd[:, 1::2]) / 4
            return LL, LH, HL, HH

        def shannon_entropy(subband):
            subband = np.abs(subband).flatten()
            subband = subband[subband > 1e-10]
            if len(subband) == 0:
                return 0.0
            p = subband / subband.sum()
            return float(-np.sum(p * np.log2(p + 1e-10)))

        LL, LH, HL, HH = haar_dwt(gray)

        hh_entropy = shannon_entropy(HH)
        lh_entropy = shannon_entropy(LH)
        hl_entropy = shannon_entropy(HL)

        # Second level decomposition on LL
        LL2, _, _, HH2 = haar_dwt(LL)
        hh2_entropy = shannon_entropy(HH2)

        # Natural images: HH entropy typically 2.5–3.5
        # Deepfakes: HH entropy often >4.0 (over-synthesized textures)
        hh_anomaly = max(0.0, min(1.0, (hh_entropy - 3.5) / 2.5))
        hh2_anomaly = max(0.0, min(1.0, (hh2_entropy - 3.0) / 2.0))

        anomaly_score = float(hh_anomaly * 0.6 + hh2_anomaly * 0.4)

        return {
            "anomaly_score": anomaly_score,
            "hh_entropy": hh_entropy,
            "hh2_entropy": hh2_entropy,
            "lh_entropy": lh_entropy,
            "hl_entropy": hl_entropy,
            "expected_hh_entropy": "2.5–3.5",
            "observed_hh_entropy": f"{hh_entropy:.2f}",
        }

    # ── Power Spectrum Analysis ───────────────────────────────────────────────

    def _analyze_power_spectrum(self, gray: np.ndarray) -> Dict:
        """
        Azimuthal power spectrum analysis.

        Natural images follow a 1/f power law — power decreases as
        frequency increases. GANs that learn to fool pixel-space
        discriminators still leave characteristic peaks in the
        2D Fourier spectrum, particularly at frequencies corresponding
        to the GAN's internal resolution boundaries (e.g., 4px, 8px).
        """
        # 2D FFT and power spectrum
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2
        power_db = 10 * np.log10(power + 1e-10)

        h, w = power_db.shape
        cy, cx = h // 2, w // 2

        # Compute radial (azimuthal average) profile
        y_idx, x_idx = np.mgrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x_idx**2 + y_idx**2).astype(int)
        r_max = min(cy, cx)

        radial_mean = np.zeros(r_max)
        for radius in range(r_max):
            mask = (r == radius)
            if mask.sum() > 0:
                radial_mean[radius] = power_db[mask].mean()

        # Fit 1/f power law (linear in log-log space)
        freqs = np.arange(1, r_max)
        if len(freqs) > 10:
            log_freqs = np.log(freqs + 1e-8)
            log_power = radial_mean[1:r_max]
            coeffs = np.polyfit(log_freqs, log_power, 1)
            slope = float(coeffs[0])
            # Natural images: slope ≈ -2 to -3
            # GANs: slope deviation and spectral peaks
            slope_anomaly = max(0.0, min(1.0, abs(slope + 2) / 2))
        else:
            slope = -2.0
            slope_anomaly = 0.0

        # Detect spectral peaks (GAN grid artifacts)
        if len(radial_mean) > 20:
            residual = radial_mean - np.convolve(radial_mean, np.ones(5)/5, mode='same')
            peak_freq_idx = int(np.argmax(np.abs(residual[5:-5])) + 5)
            peak_magnitude = float(np.abs(residual[peak_freq_idx]))
            peak_anomaly = min(1.0, peak_magnitude / 10.0)
            peak_freq = peak_freq_idx / max(h, w)
        else:
            peak_anomaly = 0.0
            peak_freq = 0.0

        anomaly_score = float(slope_anomaly * 0.5 + peak_anomaly * 0.5)

        return {
            "anomaly_score": anomaly_score,
            "spectral_slope": slope,
            "expected_slope": "-2.0 to -3.0 (natural)",
            "peak_freq": peak_freq,
            "peak_anomaly": peak_anomaly,
        }

    # ── Noise Pattern Analysis ────────────────────────────────────────────────

    def _analyze_noise_pattern(self, bgr: np.ndarray) -> Dict:
        """
        Photo-Response Non-Uniformity (PRNU) noise analysis.

        Every camera sensor has a unique noise fingerprint (PRNU).
        In composited deepfakes, different image regions may have
        inconsistent noise patterns — revealing boundaries where
        faces were blended.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract noise residual using Gaussian denoising
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - denoised

        h, w = noise.shape
        half_h, half_w = h // 2, w // 2

        if half_h < 4 or half_w < 4:
            return {"anomaly_score": 0.0, "variance_ratio": 1.0}

        # Split into quadrants
        quadrants = [
            noise[:half_h, :half_w],    # top-left
            noise[:half_h, half_w:],    # top-right
            noise[half_h:, :half_w],    # bottom-left
            noise[half_h:, half_w:],    # bottom-right
        ]
        variances = [float(np.var(q)) for q in quadrants]

        # In real photos, noise variance should be roughly uniform
        # In composited faces, there's a boundary where variance jumps
        if max(variances) > 0:
            variance_ratio = min(variances) / max(variances)
            inconsistency = 1 - variance_ratio
        else:
            variance_ratio = 1.0
            inconsistency = 0.0

        # Cross-correlation between opposite quadrants
        if quadrants[0].size > 0 and quadrants[3].size > 0:
            q0_flat = quadrants[0].flatten()[:100]
            q3_flat = quadrants[3].flatten()[:100]
            if len(q0_flat) == len(q3_flat) and np.std(q0_flat) > 0 and np.std(q3_flat) > 0:
                corr = float(np.corrcoef(q0_flat, q3_flat)[0, 1])
                corr_anomaly = max(0.0, 1 - abs(corr))
            else:
                corr_anomaly = 0.0
        else:
            corr_anomaly = 0.0

        anomaly_score = float(inconsistency * 0.6 + corr_anomaly * 0.4)

        return {
            "anomaly_score": min(1.0, anomaly_score),
            "variance_ratio": variance_ratio,
            "quadrant_variances": variances,
            "corr_anomaly": corr_anomaly,
        }

    # ── Feature Vector ────────────────────────────────────────────────────────

    def _extract_feature_vector(self, dct, dwt, spectrum) -> List[float]:
        """Extract a compact feature vector for downstream fusion."""
        return [
            dct.get("high_freq_ratio", 0),
            dct.get("dc_variance", 0) / 5000,
            dwt.get("hh_entropy", 0) / 10,
            dwt.get("lh_entropy", 0) / 10,
            abs(spectrum.get("spectral_slope", -2) + 2) / 4,
            spectrum.get("peak_anomaly", 0),
        ]
