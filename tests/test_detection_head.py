
import unittest

import torch

from models.backbone import SmartCapBackbone
from models.neck import SmartCapNeck
from models.detection_head import SmartCapDetectionHead


class TestSmartCapDetectionHead(unittest.TestCase):

    def test_detection_output_shapes(self):
        backbone = SmartCapBackbone().eval()
        neck = SmartCapNeck().eval()
        head = SmartCapDetectionHead(num_classes=6).eval()

        image = torch.randn(1, 3, 192, 192)

        with torch.inference_mode():
            features = backbone(image)
            fused = neck(*features)
            outputs = head(*fused)

        expected_shapes = [
            (1, 11, 24, 24),
            (1, 11, 12, 12),
            (1, 11, 6, 6),
        ]

        actual_shapes = [tuple(output.shape) for output in outputs]

        self.assertEqual(actual_shapes, expected_shapes)

    def test_custom_class_count(self):
        head = SmartCapDetectionHead(num_classes=4)

        self.assertEqual(head.num_classes, 4)

        self.assertEqual(head.detect_p3[-1].out_channels, 9)


if __name__ == "__main__":
    unittest.main()
