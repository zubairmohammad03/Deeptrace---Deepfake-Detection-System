"""
DeepTrace AI — RawNet2 Architecture
Audio deepfake / voice clone detection operating on raw waveforms.

Paper: "RawNet2: Improved RawNet with feature map scaling" (Tak et al., 2021)
Input:  Raw waveform tensor [B, 1, T] at 16kHz
Output: [B, 1] probability score (1 = synthetic/fake voice)

Key design choices:
  - Sinc-conv filters in first layer (learnable bandpass filters)
  - Residual blocks with FMS (Feature Map Scaling)
  - GRU for temporal context
  - No hand-crafted features (MFCC etc.) — raw end-to-end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SincConv(nn.Module):
    """
    Sinc-based convolution layer (from SincNet).
    Instead of learning arbitrary filters, learns center-frequency and bandwidth
    of bandpass filters — physically meaningful and sample-efficient.
    """

    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        # Learnable: center frequencies and bandwidths (in Hz, constrained positive)
        low_hz, high_hz = 30.0, sample_rate / 2 - 100
        mel_points = torch.linspace(
            self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1
        )
        hz_points = self._to_hz(mel_points)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        # Hamming window for smooth filters
        n = (self.kernel_size - 1) / 2.0
        n_vec = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
        self.register_buffer("n_", n_vec)
        self.register_buffer("window_", torch.hamming_window(self.kernel_size)[:(self.kernel_size // 2)])

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.abs(self.low_hz_) + 50   # min 50 Hz
        high = low + torch.abs(self.band_hz_) + 50  # at least 50 Hz bandwidth
        high = torch.clamp(high, min=low + 50, max=self.sample_rate / 2 - 50)

        f_times_t_low = torch.matmul(low / self.sample_rate, self.n_)
        f_times_t_high = torch.matmul(high / self.sample_rate, self.n_)

        band_pass = (
            2 * high * torch.sinc(2 * f_times_t_high)
            - 2 * low * torch.sinc(2 * f_times_t_low)
        )
        band_pass = band_pass * self.window_
        # Symmetric filter
        filters = torch.cat([band_pass.flip(dims=[1]), band_pass], dim=1).unsqueeze(1)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, groups=1)


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with Feature Map Scaling (FMS).
    FMS learns a per-channel scale + shift after each BN, making the network
    more expressive for detecting subtle prosody artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(downsample)

        # Feature Map Scaling
        self.fms_scale = nn.Parameter(torch.ones(out_channels))
        self.fms_shift = nn.Parameter(torch.zeros(out_channels))

        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        out = self.bn2(self.conv2(out))

        # Apply FMS
        scale = self.fms_scale.unsqueeze(0).unsqueeze(-1)
        shift = self.fms_shift.unsqueeze(0).unsqueeze(-1)
        out = out * torch.sigmoid(scale) + shift

        out = F.leaky_relu(out + residual, 0.3)
        return self.pool(out)


class RawNet2(nn.Module):
    """
    Full RawNet2 model.

    Architecture:
        SincConv → ResBlock×6 → GRU(1024) → FC(1024) → Sigmoid
    
    Parameter count: ~25M
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()

        self.sinc = SincConv(128, kernel_size=1024, sample_rate=sample_rate)
        self.bn_sinc = nn.BatchNorm1d(128)
        self.pool_sinc = nn.MaxPool1d(3)

        channels = [128, 128, 256, 256, 256, 512, 512]
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(channels[i], channels[i + 1], downsample=3)
            for i in range(len(channels) - 1)
        ])

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T] raw waveform (T = sample_rate * duration)
        Returns: [B, 1] logit (apply sigmoid for probability)
        """
        # Sinc filter bank
        out = torch.abs(self.sinc(x))  # Abs = magnitude of bandpass response
        out = F.leaky_relu(self.bn_sinc(out), 0.3)
        out = self.pool_sinc(out)

        # Residual blocks
        for block in self.res_blocks:
            out = block(out)

        # GRU over time — captures temporal patterns in voice artifacts
        out = out.permute(0, 2, 1)  # [B, T', C]
        out, _ = self.gru(out)
        out = out[:, -1, :]  # Last hidden state

        return self.fc(out)
