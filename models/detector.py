
"""Complete Smart Cap object detector."""

import torch
import torch.nn as nn

from models.backbone import SmartCapBackbone
from models.neck import SmartCapNeck
from models.detection_head import SmartCapDetectionHead


class SmartCapDetector(nn.Module):
    """Combine backbone, neck, and detection head."""

    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.backbone = SmartCapBackbone()
        self.neck = SmartCapNeck()
        self.head = SmartCapDetectionHead(
            num_classes=num_classes
        )

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: RGB tensor shaped [B, 3, 192, 192].

        Returns:
            Three raw prediction tensors at different scales.
        """
        features = self.backbone(images)
        fused_features = self.neck(*features)
        predictions = self.head(*fused_features)

        return predictions
