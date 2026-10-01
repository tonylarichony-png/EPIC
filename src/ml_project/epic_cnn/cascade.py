"""Coarse-to-fine CNN components for CNN-EXP-049.

The module is deliberately device agnostic.  It runs with ordinary PyTorch on
CPU or CUDA and does not depend on the DirectML-specific training helpers used
by earlier local experiments.

Network A consumes a long, strand-oriented DNA window and emits one activity
logit plus a latent vector for every exact genomic R16 bin.  Network B consumes
base-resolution DNA together with the aligned, broadcast R16 features.  The
regional features condition the local trunk through zero-initialised FiLM
adapters and a zero-initialised residual logit head.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


DNA_CHANNELS = 6
REGION_SIZE = 16
REGIONAL_LATENT_CHANNELS = 16
REGIONAL_CONDITION_CHANNELS = 1 + REGIONAL_LATENT_CHANNELS
RC_CHANNEL_PERMUTATION = (3, 2, 1, 0, 4, 5)


def encode_raw_dna_six_channel(raw_codes: torch.Tensor) -> torch.Tensor:
    """Encode case-preserving FASTA codes as A/C/G/T + lowercase + N.

    ``raw_codes`` must contain the repository FASTA representation:
    A/C/G/T=0..3, a/c/g/t=4..7 and N/other/contig padding=8.
    """

    if raw_codes.ndim != 2:
        raise ValueError("raw_codes must have shape [batch, length]")
    if raw_codes.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("raw_codes must contain integer FASTA codes")
    if raw_codes.numel() and (
        int(raw_codes.min().item()) < 0 or int(raw_codes.max().item()) > 8
    ):
        raise ValueError("raw_codes values must be in [0, 8]")

    batch, length = raw_codes.shape
    encoded = torch.zeros(
        (batch, DNA_CHANNELS, length),
        dtype=torch.float32,
        device=raw_codes.device,
    )
    nucleotide = raw_codes.remainder(4)
    known = raw_codes < 8
    encoded[:, :4, :].scatter_(
        1,
        nucleotide.clamp(max=3).unsqueeze(1).long(),
        known.unsqueeze(1).to(encoded.dtype),
    )
    encoded[:, 4, :] = ((raw_codes >= 4) & (raw_codes < 8)).to(encoded.dtype)
    encoded[:, 5, :] = (raw_codes == 8).to(encoded.dtype)
    return encoded


def reverse_complement_six_channel(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a six-channel DNA tensor of shape ``[B, 6, L]``."""

    if x.ndim != 3 or x.shape[1] != DNA_CHANNELS:
        raise ValueError("x must have shape [batch, 6, length]")
    return x[:, RC_CHANNEL_PERMUTATION, :].flip(-1).contiguous()


class GELUResidualBlock(nn.Module):
    """Post-activation residual block used by the regional detector."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(residual + x)


def _resize_exactly_to_regions(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize a feature map to the exact regional grid.

    Higher-resolution maps use non-overlapping average pooling, so local and
    stride-4 features respect exact R16 boundaries.  Coarser maps are linearly
    interpolated because they already summarize multiple R16 bins.
    """

    source_length = x.shape[-1]
    if source_length == target_length:
        return x
    if source_length > target_length:
        if source_length % target_length:
            raise ValueError(
                f"feature length {source_length} is not divisible by "
                f"regional length {target_length}"
            )
        factor = source_length // target_length
        return F.avg_pool1d(x, kernel_size=factor, stride=factor)
    return F.interpolate(x, size=target_length, mode="linear", align_corners=False)


class RegionalDetectorA(nn.Module):
    """Long-context multiscale detector with one output per exact R16 bin."""

    def __init__(
        self,
        *,
        input_channels: int = DNA_CHANNELS,
        local_channels: int = 64,
        tower_channels: Sequence[int] = (96, 128, 160, 192),
        latent_channels: int = REGIONAL_LATENT_CHANNELS,
        region_size: int = REGION_SIZE,
    ) -> None:
        super().__init__()
        if region_size != REGION_SIZE:
            raise ValueError("CNN-EXP-049 currently requires exact R16 outputs")
        if len(tower_channels) != 4:
            raise ValueError("the registered detector uses four stride-4 stages")
        self.region_size = region_size
        self.latent_channels = latent_channels

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, local_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(local_channels),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            GELUResidualBlock(local_channels, kernel_size=5, dilation=1),
            GELUResidualBlock(local_channels, kernel_size=5, dilation=2),
        )

        downs: list[nn.Module] = []
        blocks: list[nn.Module] = []
        projections: list[nn.Module] = [nn.Conv1d(local_channels, 64, 1)]
        previous = local_channels
        for channels in tower_channels:
            downs.append(
                nn.Sequential(
                    nn.Conv1d(
                        previous,
                        channels,
                        kernel_size=9,
                        stride=4,
                        padding=4,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
            )
            blocks.append(
                nn.Sequential(
                    GELUResidualBlock(channels, kernel_size=5, dilation=1),
                    GELUResidualBlock(channels, kernel_size=5, dilation=3),
                )
            )
            projections.append(nn.Conv1d(channels, 64, 1))
            previous = channels

        self.downs = nn.ModuleList(downs)
        self.blocks = nn.ModuleList(blocks)
        self.projections = nn.ModuleList(projections)
        fused_channels = 64 * len(self.projections)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(fused_channels, 64, kernel_size=1),
            nn.GELU(),
        )
        self.latent_head = nn.Conv1d(64, latent_channels, kernel_size=1)
        self.activity_head = nn.Conv1d(64, 1, kernel_size=1)
        nn.init.constant_(self.activity_head.bias, -6.0)

    @property
    def condition_channels(self) -> int:
        return 1 + self.latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != DNA_CHANNELS:
            raise ValueError("Network A expects [batch, 6, length]")
        if x.shape[-1] % self.region_size:
            raise ValueError("Network A input length must be divisible by 16")

        local = self.local(self.stem(x))
        scales = [local]
        tower = local
        for down, blocks in zip(self.downs, self.blocks, strict=True):
            tower = blocks(down(tower))
            scales.append(tower)

        regional_length = x.shape[-1] // self.region_size
        projected = [
            _resize_exactly_to_regions(project(feature), regional_length)
            for project, feature in zip(self.projections, scales, strict=True)
        ]
        return self.bottleneck(torch.cat(projected, dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.encode(x)
        return self.activity_head(fused), self.latent_head(fused)

    def condition(self, x: torch.Tensor) -> torch.Tensor:
        activity, latent = self(x)
        return torch.cat((activity, latent), dim=1)


class RegionalRateDetectorA(RegionalDetectorA):
    """Count-aware A whose supervised log-rate is computed through latent.

    Unlike :class:`RegionalDetectorA`, every latent-head parameter lies on the
    loss path.  The first returned tensor is a regional Poisson log-rate rather
    than an independently parameterised binary activity logit.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.activity_head = nn.Conv1d(self.latent_channels, 1, kernel_size=1)
        nn.init.constant_(self.activity_head.bias, -6.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.encode(x)
        latent = F.gelu(self.latent_head(fused))
        log_rate = self.activity_head(latent)
        return log_rate, latent


class PreActivationResidualBlock(nn.Module):
    """Full-resolution residual block used by the nucleotide localizer."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class FiLMAdapter(nn.Module):
    """Zero-initialised feature-wise linear modulation from regional context."""

    def __init__(
        self,
        condition_channels: int,
        feature_channels: int,
        hidden_channels: int = 64,
        strength: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.strength = float(strength)
        self.net = nn.Sequential(
            nn.Conv1d(condition_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 2 * feature_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != condition.shape[0] or x.shape[-1] != condition.shape[-1]:
            raise ValueError("FiLM feature and condition batch/length must match")
        gamma, beta = self.net(condition).chunk(2, dim=1)
        scale = 1.0 + self.strength * torch.tanh(gamma)
        shift = self.strength * torch.tanh(beta)
        return x * scale + shift


LOCAL_BLOCK_SETTINGS: tuple[tuple[int, int], ...] = (
    (7, 1),
    (3, 2),
    (3, 4),
    (3, 8),
    (3, 16),
    (3, 32),
    (3, 64),
    (3, 128),
)


def theoretical_receptive_field(
    block_settings: Iterable[tuple[int, int]] = LOCAL_BLOCK_SETTINGS,
) -> int:
    """Return the receptive field for two convolutions in every block."""

    return 1 + sum(2 * (kernel - 1) * dilation for kernel, dilation in block_settings)


class ConditionalLocalizerB(nn.Module):
    """Full-resolution localizer conditioned on frozen, broadcast R16 features."""

    def __init__(
        self,
        *,
        input_channels: int = DNA_CHANNELS,
        channels: int = 256,
        condition_channels: int = REGIONAL_CONDITION_CHANNELS,
        film_after_blocks: Sequence[int] = (2, 4, 6, 8),
        film_strength: float = 0.1,
        block_settings: Sequence[tuple[int, int]] = LOCAL_BLOCK_SETTINGS,
    ) -> None:
        super().__init__()
        if len(block_settings) != 8:
            raise ValueError("the registered localizer uses eight residual blocks")
        film_indices = tuple(sorted(set(int(index) for index in film_after_blocks)))
        if any(index < 1 or index > len(block_settings) for index in film_indices):
            raise ValueError("FiLM block indices are one-based and must be in [1, 8]")

        self.channels = channels
        self.condition_channels = condition_channels
        self.film_after_blocks = film_indices
        self.stem = nn.Conv1d(input_channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            PreActivationResidualBlock(channels, kernel, dilation)
            for kernel, dilation in block_settings
        )
        self.film = nn.ModuleDict(
            {
                str(index): FiLMAdapter(
                    condition_channels,
                    channels,
                    strength=film_strength,
                )
                for index in film_indices
            }
        )
        self.base_head = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv1d(channels + condition_channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )
        self.intensity_head = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def receptive_field(self) -> int:
        return theoretical_receptive_field()

    def forward(
        self,
        dna: torch.Tensor,
        regional_condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if dna.ndim != 3 or dna.shape[1] != DNA_CHANNELS:
            raise ValueError("Network B DNA must have shape [batch, 6, length]")
        if regional_condition.ndim != 3:
            raise ValueError("regional_condition must have shape [batch, channels, length]")
        if regional_condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} regional channels, got "
                f"{regional_condition.shape[1]}"
            )
        if dna.shape[0] != regional_condition.shape[0] or dna.shape[-1] != regional_condition.shape[-1]:
            raise ValueError("DNA and regional condition batch/length must match")

        x = self.stem(dna)
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            adapter = self.film[str(index)] if str(index) in self.film else None
            if adapter is not None:
                x = adapter(x, regional_condition)

        base_logit = self.base_head(x)
        delta_logit = self.residual_head(torch.cat((x, regional_condition), dim=1))
        final_logit = base_logit + delta_logit
        return {
            "final_logit": final_logit,
            "base_logit": base_logit,
            "delta_logit": delta_logit,
            "intensity": self.intensity_head(x),
            "features": x,
        }


def broadcast_r16_condition(
    regional_condition: torch.Tensor,
    *,
    region_size: int = REGION_SIZE,
    output_length: int | None = None,
) -> torch.Tensor:
    """Repeat every exact R16 feature vector over its nucleotide positions."""

    if regional_condition.ndim != 3:
        raise ValueError("regional_condition must have shape [B, C, R]")
    result = regional_condition.repeat_interleave(region_size, dim=-1)
    if output_length is not None:
        if output_length < 0 or output_length > result.shape[-1]:
            raise ValueError("output_length is outside the broadcast tensor")
        result = result[..., :output_length]
    return result


__all__ = [
    "ConditionalLocalizerB",
    "DNA_CHANNELS",
    "FiLMAdapter",
    "GELUResidualBlock",
    "LOCAL_BLOCK_SETTINGS",
    "PreActivationResidualBlock",
    "REGIONAL_CONDITION_CHANNELS",
    "REGIONAL_LATENT_CHANNELS",
    "REGION_SIZE",
    "RegionalDetectorA",
    "RegionalRateDetectorA",
    "broadcast_r16_condition",
    "encode_raw_dna_six_channel",
    "reverse_complement_six_channel",
    "theoretical_receptive_field",
]
