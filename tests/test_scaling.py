import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model

class TestScaling(unittest.TestCase):
    def test_presets_exist(self):
        for preset in ["100M", "1B", "3B", "7B"]:
            cfg = Config.from_preset(preset)
            self.assertIsNotNone(cfg)

    def test_invalid_preset(self):
        with self.assertRaises(ValueError):
            Config.from_preset("999B")

    def test_param_count_parity_with_meta_device(self):
        presets = ["100M", "1B", "3B", "7B"]
        for preset in presets:
            cfg = Config.from_preset(preset)
            calc = cfg.compute_param_count()
            
            with torch.device("meta"):
                m = Model(cfg)
                actual_total = sum(p.numel() for p in m.parameters())
            
            self.assertEqual(
                calc["total"],
                actual_total,
                f"Preset {preset}: computed {calc['total']} != meta {actual_total}"
            )
            self.assertTrue(calc["vocab_tax_pct"] < 5.0)

if __name__ == "__main__":
    unittest.main()
