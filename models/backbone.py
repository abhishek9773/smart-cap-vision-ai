
"""
Lightweight CNN backbone for the Smart Cap vision model.

Input:
    RGB tensor: [batch, 3, 192, 192]

Outputs:
    P3: [batch, 32, 24, 24]
    P4: [batch, 48, 12, 12]
    P5: [batch, 96, 6, 6]

This is an initial architecture prototype, not a trained model.
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Convolution followed by batch normalization and ReLU6."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        """Build a padded convolution block.

        Set ``groups`` to ``in_channels`` for a depthwise convolution.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution, normalization, and activation."""
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Apply a depthwise spatial convolution followed by pointwise channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        """Create a depthwise-separable block with optional spatial downsampling."""
        super().__init__()

        self.depthwise = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )

        self.pointwise = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial filtering and then mix information across channels."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SmartCapBackbone(nn.Module):
    """MobileNet-style backbone returning three progressively coarser features.

    The input is an RGB tensor shaped ``[B, 3, 192, 192]``. The returned
    ``(P3, P4, P5)`` feature maps have shapes ``[B, 32, 24, 24]``,
    ``[B, 48, 12, 12]``, and ``[B, 96, 6, 6]`` respectively.
    """

    def __init__(self) -> None:
        """Construct the stem and four depthwise-separable stages."""
        super().__init__()

        # Input: [B, 3, 192, 192]
        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)

        # 96 x 96 -> 48 x 48
        self.stage1 = DepthwiseSeparableConv(16, 24, stride=2)

        # 48 x 48 -> 24 x 24: P3
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(24, 32, stride=2),
            DepthwiseSeparableConv(32, 32, stride=1),
        )

        # 24 x 24 -> 12 x 12: P4
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 48, stride=1),
        )

        # 12 x 12 -> 6 x 6: P5
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(48, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return P3, P4, and P5 feature maps for a 192x192 RGB batch."""
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        return p3, p4, p5
