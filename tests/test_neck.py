
import unittest

import torch

from models.backbone import SmartCapBackbone
from models.neck import SmartCapNeck


class TestSmartCapNeck(unittest.TestCase):

    def test_neck_output_shapes(self):
        backbone = SmartCapBackbone().eval()
        neck = SmartCapNeck().eval()

        image = torch.randn(1, 3, 192, 192)

        with torch.inference_mode():
            p3, p4, p5 = backbone(image)
            n3, n4, n5 = neck(p3, p4, p5)

        self.assertEqual(tuple(n3.shape), (1, 32, 24, 24))
        self.assertEqual(tuple(n4.shape), (1, 32, 12, 12))
        self.assertEqual(tuple(n5.shape), (1, 32, 6, 6))

    def test_neck_batch_size(self):
        backbone = SmartCapBackbone().eval()
        neck = SmartCapNeck().eval()

        image = torch.randn(2, 3, 192, 192)

        with torch.inference_mode():
            features = neck(*backbone(image))

        self.assertEqual(features[0].shape[0], 2)
        self.assertEqual(features[1].shape[0], 2)
        self.assertEqual(features[2].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
