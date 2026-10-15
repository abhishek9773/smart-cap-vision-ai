
import unittest

import torch

from models.detector import SmartCapDetector


class TestSmartCapDetector(unittest.TestCase):

    def test_complete_detector_output_shapes(self):
        model = SmartCapDetector(num_classes=6)
        model.eval()

        images = torch.randn(1, 3, 192, 192)

        with torch.inference_mode():
            outputs = model(images)

        expected_shapes = [
            (1, 11, 24, 24),
            (1, 11, 12, 12),
            (1, 11, 6, 6),
        ]

        actual_shapes = [
            tuple(output.shape) for output in outputs
        ]

        self.assertEqual(actual_shapes, expected_shapes)

    def test_parameter_count(self):
        model = SmartCapDetector(num_classes=6)

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        self.assertGreater(total_parameters, 0)
        self.assertEqual(total_parameters, trainable_parameters)

        print(f"\nTotal parameters: {total_parameters:,}")
        print(
            f"FP32 parameter storage estimate: "
            f"{total_parameters * 4 / 1024:.2f} KiB"
        )


if __name__ == "__main__":
    unittest.main()
