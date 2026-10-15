"""Tests for the Smart Cap feature-extraction backbone."""

import unittest

import torch

from models.backbone import SmartCapBackbone


class SmartCapBackboneTests(unittest.TestCase):
    """Check the backbone's feature-map interface and model size."""

    def test_feature_shapes_and_trainable_parameter_count(self) -> None:
        model = SmartCapBackbone().eval()
        sample = torch.zeros(1, 3, 192, 192)

        with torch.no_grad():
            features = model(sample)

        self.assertEqual(
            [tuple(feature.shape) for feature in features],
            [(1, 32, 24, 24), (1, 48, 12, 12), (1, 96, 6, 6)],
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            19_288,
        )


if __name__ == "__main__":
    unittest.main()
