import unittest

import torch

from model_architecture import architecture_stages
from training import Config, parameter_count


class ArchitectureTests(unittest.TestCase):
    def test_diagram_matches_model_dimensions_and_parameters(self):
        for blocks in range(1, 5):
            for convs in range(1, 4):
                config = Config(blocks=blocks, convs_per_block=convs, base_channels=128)
                with self.subTest(blocks=blocks, convs=convs):
                    stages = architecture_stages(config)
                    self.assertEqual(len(stages), blocks + 3)
                    self.assertEqual(stages[0].shape, (3, 32, 32))
                    for index, stage in enumerate(stages[1:blocks + 1]):
                        self.assertEqual(stage.title, f"Block {index + 1}")
                        self.assertEqual(
                            stage.shape,
                            (min(128 * 2**index, 512), 32 // 2**(index + 1), 32 // 2**(index + 1)),
                        )
                        self.assertEqual(sum(line.startswith("Conv ") for line in stage.details), convs)
                    self.assertEqual(stages[-2].title, "Classifier head")
                    self.assertEqual(stages[-1].shape, (10,))
                    self.assertEqual(sum(stage.parameters for stage in stages), parameter_count(config))

    def test_snapshot_reflects_dropout(self):
        stages = architecture_stages(Config(dropout=0.65))
        self.assertIn("Dropout: p = 0.65", stages[-2].details)

    def test_preview_does_not_advance_training_rng(self):
        before = torch.random.get_rng_state()
        architecture_stages(Config())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            architecture_stages(Config(blocks=0))


if __name__ == "__main__":
    unittest.main()
