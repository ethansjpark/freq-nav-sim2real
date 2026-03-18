from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    Compact convolutional encoder for RGB navigation observations.

    Notes:
    - Input expected in BCHW format with values in [0, 1].
    - Output is a fixed-size feature vector (default 512-dim).
    - Adaptive pooling makes it robust to varying image sizes.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
        x = self.backbone(x)
        return self.proj(x)
