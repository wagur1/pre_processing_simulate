"""
model.py — Preprocessor + Virtual Codec
=========================================
Implements the two core components from:
  "A Preprocessing Framework for Video Machine Vision under Compression"

1. **Preprocessor**: A multi-branch CNN with:
   - Temporal branch  (inter-frame, 2D convolutions on stacked frames)
   - Spatial branch   (single-frame, Conv2d + Residual Blocks)
   - Additive fusion  + global residual connection to the input frame.

2. **VirtualCodec**: A differentiable proxy for a video encoder, featuring:
   - Straight-Through Estimator (STE) quantization
   - Stochastic quantization factor f_q ~ U(30, 50)
   - Lightweight entropy estimation for rate (bpp) loss
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================== #
#                          Building Blocks                                 #
# ======================================================================== #

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block (Conv-BN-ReLU × 2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


# ======================================================================== #
#                    Temporal Branch  (inter-frame)                        #
# ======================================================================== #

class TemporalBranch(nn.Module):
    """Processes *stacked* frames (T consecutive frames concatenated along
    the channel dimension) through 2D convolutions to capture inter-frame
    (temporal) patterns.

    Input : (B, T*C, H, W)   — T RGB frames channel-stacked
    Output: (B, 3, H, W)     — single enhanced frame
    """

    def __init__(self, num_frames: int = 8, base_channels: int = 64):
        super().__init__()
        in_ch = num_frames * 3  # T × RGB

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, 3, padding=1),  # project to RGB
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================== #
#                     Spatial Branch  (intra-frame)                        #
# ======================================================================== #

class SpatialBranch(nn.Module):
    """Processes a *single* frame through Conv2d layers and residual blocks
    to enhance spatial features.

    Input : (B, 3, H, W)
    Output: (B, 3, H, W)
    """

    def __init__(self, base_channels: int = 64, num_res_blocks: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(base_channels))

        layers.append(nn.Conv2d(base_channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================== #
#                             Preprocessor                                 #
# ======================================================================== #

class Preprocessor(nn.Module):
    """Multi-branch preprocessor fusing temporal and spatial cues.

    Forward pass
    ------------
    1. Temporal branch receives all T frames stacked channel-wise.
    2. Spatial branch receives only the *current* (last) frame.
    3. Both outputs are added elementwise (fusion).
    4. A global residual connection adds the original current frame.

    Input : (B, T, C, H, W)   — a clip of T RGB frames
    Output: (B, 3, H, W)      — the enhanced current frame
    """

    def __init__(self, num_frames: int = 8, base_channels: int = 64):
        super().__init__()
        self.temporal = TemporalBranch(num_frames, base_channels)
        self.spatial = SpatialBranch(base_channels)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (B, T, C, H, W) float tensor in [0, 1]
        Returns:
            enhanced: (B, C, H, W) preprocessed current frame
        """
        B, T, C, H, W = clip.shape

        # Current frame = last frame in the temporal window
        current_frame = clip[:, -1]  # (B, C, H, W)

        # Temporal branch: stack all frames along the channel dim
        stacked = clip.view(B, T * C, H, W)  # (B, T*C, H, W)
        temporal_out = self.temporal(stacked)  # (B, 3, H, W)

        # Spatial branch: process only the current frame
        spatial_out = self.spatial(current_frame)  # (B, 3, H, W)

        # Fusion: elementwise addition + global residual connection
        enhanced = temporal_out + spatial_out + current_frame

        return enhanced


# ======================================================================== #
#                          Virtual Codec                                   #
# ======================================================================== #

class VirtualCodec(nn.Module):
    """Differentiable proxy for a video codec.

    Key Ideas
    ---------
    *Straight-Through Estimator (STE)*:
        During the forward pass we apply ``torch.round()`` to simulate
        integer quantization, but ``round`` has zero gradient almost
        everywhere.  The STE trick lets the gradient flow *as if* no
        rounding happened:

            x_quant = x + (round(x) - x).detach()

        Forward:  x_quant == round(x)  (exact integer values)
        Backward: d(x_quant)/dx == 1   (identity gradient)

    *Quantization factor f_q*:
        During training, f_q is randomly sampled from U(30, 50) at each
        forward call to simulate varying codec quality parameters (like
        QP in H.264/H.265).  At inference the caller may fix f_q.

    *Entropy / Rate estimation*:
        We approximate the bit rate of the quantized representation using
        a simple parametric entropy model (Laplacian assumption) over the
        quantized latent, yielding an estimated bits-per-pixel (bpp).

    Pipeline
    --------
        input → analysis_transform → scale by f_q → STE_round → scale by 1/f_q
                → synthesis_transform → reconstructed
        (rate loss is computed from the quantized latent)
    """

    def __init__(self, latent_channels: int = 48):
        super().__init__()
        self.latent_channels = latent_channels

        # Analysis transform: RGB → latent
        self.analysis = nn.Sequential(
            nn.Conv2d(3, latent_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # Synthesis transform: latent → RGB
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, latent_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 3, 3, padding=1),
        )

        # Learnable scale/bias for entropy model (Laplacian assumption)
        self.log_scale = nn.Parameter(torch.zeros(1, latent_channels, 1, 1))

    # ------------------------------------------------------------------ #
    #  STE Quantization                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ste_round(x: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator for rounding.

        Forward:  returns round(x)          — exact integer quantization
        Backward: gradient of round(x) ≈ 1  — as if no rounding happened

        Implementation detail:
            x_quant = x + (round(x) - x).detach()
            The .detach() stops the (round(x) - x) residual from
            contributing to the gradient, so ∂x_quant/∂x = 1.
        """
        return x + (torch.round(x) - x).detach()

    # ------------------------------------------------------------------ #
    #  Entropy estimation (Laplacian model)                                #
    # ------------------------------------------------------------------ #

    def _estimate_rate(self, y_quantized: torch.Tensor) -> torch.Tensor:
        """Estimate bits-per-pixel (bpp) of the quantized latent.

        We model each element of y as drawn from a Laplace distribution:
            p(y) = 1/(2b) · exp(-|y|/b)
        where b = exp(log_scale).

        The cross-entropy (= coding cost) under this model is:
            H ≈ |y| / b  +  log(2b)   (nats, converted to bits)

        We average over all spatial and channel dimensions to get
        bits-per-element, then scale to bits-per-pixel of the original
        image resolution.
        """
        scale = torch.exp(self.log_scale).clamp(min=1e-6)
        # Laplacian negative log-likelihood (nats)
        nll = torch.abs(y_quantized) / scale + torch.log(2.0 * scale)
        # Convert nats → bits
        bits_per_element = nll / math.log(2.0)
        # Average over batch to get mean bpp
        # y_quantized shape: (B, latent_channels, H/2, W/2)
        # Original pixels per image: latent_channels * H/2 * W/2 * 4 / 3
        # (factor 4 for stride-2 downsampling, /3 for RGB channels)
        total_bits = bits_per_element.sum(dim=(1, 2, 3))  # per sample
        # Approximate original pixel count from latent spatial dims
        _, _, Hl, Wl = y_quantized.shape
        num_pixels = Hl * 2 * Wl * 2  # original H * W (single channel)
        bpp = total_bits / num_pixels  # bits per pixel
        return bpp.mean()  # scalar

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        fq: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x  : (B, 3, H, W) enhanced frame from the Preprocessor.
            fq : quantization factor.  If None, sampled from U(30, 50).

        Returns:
            x_recon : (B, 3, H, W) reconstructed frame after virtual codec.
            rate    : scalar tensor — estimated bits-per-pixel (bpp).
        """
        # Sample quantization factor during training
        if fq is None:
            fq = torch.empty(1).uniform_(30.0, 50.0).item()

        # Analysis: RGB → latent
        y = self.analysis(x)  # (B, C_l, H/2, W/2)

        # Quantize with STE -------------------------------------------
        # Scale to quantization grid, round, scale back.
        y_scaled = y * fq
        y_quantized = self._ste_round(y_scaled)  # STE: gradients pass through
        y_dequant = y_quantized / fq

        # Rate estimation from the quantized (integer) representation
        rate = self._estimate_rate(y_quantized)

        # Synthesis: latent → RGB
        x_recon = self.synthesis(y_dequant)

        # Ensure output matches input spatial size (handles odd dims)
        if x_recon.shape[2:] != x.shape[2:]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return x_recon, rate


# ======================================================================== #
#                   Combined Forward Convenience                           #
# ======================================================================== #

class PreprocessingSystem(nn.Module):
    """End-to-end wrapper: Preprocessor → VirtualCodec.

    This is the full trainable system.  The downstream analyzer is *not*
    part of this module (its weights are frozen and it is called
    externally in the training loop).
    """

    def __init__(self, num_frames: int = 8, base_channels: int = 64,
                 latent_channels: int = 48):
        super().__init__()
        self.preprocessor = Preprocessor(num_frames, base_channels)
        self.codec = VirtualCodec(latent_channels)

    def forward(
        self, clip: torch.Tensor, fq: float | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            clip : (B, T, C, H, W) temporal tensor from the dataset.
            fq   : optional fixed quantization factor.

        Returns:
            enhanced : (B, 3, H, W) preprocessor output (before codec).
            recon    : (B, 3, H, W) reconstructed frame (after codec).
            rate     : scalar — estimated bpp.
        """
        enhanced = self.preprocessor(clip)        # (B, 3, H, W)
        recon, rate = self.codec(enhanced, fq)     # (B, 3, H, W), scalar
        return enhanced, recon, rate
