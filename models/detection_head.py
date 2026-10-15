
"""Lightweight anchor-free detection head for Smart Cap."""

import torch
import torch.nn as nn

from models.backbone import ConvBNAct


class SmartCapDetectionHead(nn.Module):
    """Predict box coordinates and class scores at three scales."""

    def __init__(self, num_classes: int = 6, channels: int = 32):
        super().__init__()

        self.num_classes = num_classes

        # Four box values + one objectness score + class scores.
        output_channels = 5 + num_classes

        self.detect_p3 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, output_channels, kernel_size=1),
        )

        self.detect_p4 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, output_channels, kernel_size=1),
        )

        self.detect_p5 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, output_channels, kernel_size=1),
        )

    def forward(self, n3, n4, n5):
        """Return raw predictions at fine, medium and coarse scales."""
        p3 = self.detect_p3(n3)
        p4 = self.detect_p4(n4)
        p5 = self.detect_p5(n5)

        return p3, p4, p5
