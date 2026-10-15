
"""
Lightweight Feature Pyramid Network (FPN) neck for Smart Cap.

Inputs from SmartCapBackbone:
    P3: [B, 32, 24, 24]
    P4: [B, 48, 12, 12]
    P5: [B, 96,  6,  6]

Outputs:
    N3: [B, 32, 24, 24]
    N4: [B, 32, 12, 12]
    N5: [B, 32,  6,  6]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ConvBNAct, DepthwiseSeparableConv


class SmartCapNeck(nn.Module):
    """Fuse backbone features using a lightweight top-down FPN."""

    def __init__(self, channels: int = 32):
        super().__init__()

        # Project all backbone feature maps to a common channel count.
        self.lateral_p3 = ConvBNAct(32, channels, kernel_size=1)
        self.lateral_p4 = ConvBNAct(48, channels, kernel_size=1)
        self.lateral_p5 = ConvBNAct(96, channels, kernel_size=1)

        # Refine features after concatenating information from two scales.
        self.fuse_p4 = DepthwiseSeparableConv(
            channels * 2, channels
        )
        self.fuse_p3 = DepthwiseSeparableConv(
            channels * 2, channels
        )

        # Refine the deepest feature map.
        self.refine_p5 = DepthwiseSeparableConv(
            channels, channels
        )

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse backbone features from coarse to fine resolutions.

        Returns:
            N3: high-resolution fused features
            N4: intermediate-resolution fused features
            N5: low-resolution deep features
        """

        # Project backbone features to 32 channels.
        lateral3 = self.lateral_p3(p3)
        lateral4 = self.lateral_p4(p4)
        lateral5 = self.lateral_p5(p5)

        # Refine the deepest features.
        n5 = self.refine_p5(lateral5)

        # Top-down: upsample P5 and combine with P4.
        up5 = F.interpolate(
            n5,
            size=lateral4.shape[-2:],
            mode="nearest",
        )

        n4 = self.fuse_p4(
            torch.cat([lateral4, up5], dim=1)
        )

        # Top-down: upsample fused P4 and combine with P3.
        up4 = F.interpolate(
            n4,
            size=lateral3.shape[-2:],
            mode="nearest",
        )

        n3 = self.fuse_p3(
            torch.cat([lateral3, up4], dim=1)
        )

        return n3, n4, n5
