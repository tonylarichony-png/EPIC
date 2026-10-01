"""Compact strand-oriented CNN used by the EPIC baseline experiment."""

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_BLOCK_SETTINGS: tuple[tuple[int, int], ...] = (
    (7, 1),
    (3, 2),
    (3, 4),
    (3, 8),
    (3, 16),
    (3, 32),
    (3, 64),
    (3, 128),
)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with two same-resolution convolutions."""

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 7,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels должен быть положительным")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size должен быть положительным нечётным")
        if dilation <= 0:
            raise ValueError("dilation должен быть положительным")

        padding = dilation * (kernel_size - 1) // 2

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(torch.relu(self.bn1(x)))
        x = self.conv2(torch.relu(self.bn2(x)))
        return x + residual


class CNNBackbone(nn.Module):
    """Stem plus named residual stages with optional activation capture."""

    def __init__(self, stem: nn.Module, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.stem = stem
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ):
        activations: dict[str, torch.Tensor] = {}
        x = self.stem(x)

        if return_activations:
            activations["stem"] = x

        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if return_activations:
                activations[f"block{index}"] = x

        if return_activations:
            return x, activations
        return x


class CNNBaseline(nn.Module):
    """Backbone with one strand-oriented logit per sequence position."""

    def __init__(self, backbone: CNNBackbone, profile_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.profile_head = profile_head

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ):
        if return_activations:
            features, activations = self.backbone(
                x,
                return_activations=True,
            )
            logits = self.profile_head(features)
            activations["profile_logits"] = logits
            return logits, activations

        return self.profile_head(self.backbone(x))


class CNNPresenceIntensity(nn.Module):
    """Shared backbone with presence and positive-count intensity outputs."""

    def __init__(self, backbone: CNNBackbone, profile_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.profile_head = profile_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.profile_head(self.backbone(x))
        if outputs.ndim != 3 or outputs.shape[1] != 2:
            raise ValueError("profile_head должна вернуть тензор [B,2,L]")
        return outputs[:, 0:1, :], outputs[:, 1:2, :]


def create_cnn_baseline(
    *,
    input_channels: int = 5,
    channels: int = 64,
    output_channels: int = 1,
    block_settings: Sequence[tuple[int, int]] = DEFAULT_BLOCK_SETTINGS,
) -> CNNBaseline:
    """Create a fresh baseline model with independently initialized weights."""

    stem = nn.Conv1d(input_channels, channels, kernel_size=1)
    blocks = [
        ResidualBlock(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        for kernel_size, dilation in block_settings
    ]
    backbone = CNNBackbone(stem=stem, blocks=blocks)
    profile_head = nn.Conv1d(channels, output_channels, kernel_size=1)
    return CNNBaseline(backbone=backbone, profile_head=profile_head)


def create_cnn_presence_intensity(
    *,
    input_channels: int = 5,
    channels: int = 64,
    block_settings: Sequence[tuple[int, int]] = DEFAULT_BLOCK_SETTINGS,
) -> CNNPresenceIntensity:
    """Create the EXP-002 model while keeping the EXP-001 backbone unchanged."""

    baseline = create_cnn_baseline(
        input_channels=input_channels,
        channels=channels,
        block_settings=block_settings,
    )
    profile_head = nn.Conv1d(channels, 2, kernel_size=1)
    return CNNPresenceIntensity(
        backbone=baseline.backbone,
        profile_head=profile_head,
    )


__all__ = [
    "CNNBackbone",
    "CNNBaseline",
    "CNNPresenceIntensity",
    "DEFAULT_BLOCK_SETTINGS",
    "ResidualBlock",
    "create_cnn_baseline",
    "create_cnn_presence_intensity",
]
