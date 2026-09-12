import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model
from maba.optimizers import HybridOpt
from maba.optimizers.muon import newton_schulz5
from maba.hardware import (
    is_tpu_available,
    get_tpu_device,
    get_device,
    get_dtype,
    mark_step,
    optimizer_step,
    clip_grad_norm,
    wrap_loader,
    is_master_process,
    master_print,
    get_tpu_info,
    get_hardware_status
)

class TestTPUSupport(unittest.TestCase):
    def test_tpu_uninstalled_fallback(self):
        self.assertFalse(is_tpu_available())
        with self.assertRaises(RuntimeError):
            get_tpu_device()

    def test_device_resolution(self):
        dev_cpu = get_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")
        dev_auto = get_device("auto")
        self.assertIn(dev_auto.type, ("cpu", "cuda", "mps", "xla"))

    def test_dtype_selection(self):
        self.assertEqual(get_dtype(torch.device("cpu"), requested="bfloat16"), torch.bfloat16)
        self.assertEqual(get_dtype(torch.device("cpu"), requested="float16"), torch.float16)
        self.assertEqual(get_dtype(torch.device("cpu"), requested="float32"), torch.float32)

    def test_mocked_tpu_operations(self):
        mock_xm = MagicMock()
        mock_device = torch.device("cpu")
        mock_xm.xla_device.return_value = mock_device
        mock_xm.xrt_world_size.return_value = 8
        mock_xm.get_ordinal.return_value = 0
        mock_xm.is_master_ordinal.return_value = True

        mock_torch_xla = MagicMock()
        mock_torch_xla.__version__ = "2.6.0"
        mock_torch_xla.core.xla_model = mock_xm

        with patch.dict("sys.modules", {"torch_xla": mock_torch_xla, "torch_xla.core.xla_model": mock_xm}):
            with patch("maba.hardware.is_tpu_available", return_value=True):
                dev = get_device("tpu")
                self.assertEqual(dev, mock_device)

                mark_step()
                mock_xm.mark_step.assert_called_once()

                fake_opt = MagicMock()
                optimizer_step(fake_opt, barrier=True)
                mock_xm.optimizer_step.assert_called_once_with(fake_opt, barrier=True)

                self.assertTrue(is_master_process())

    def test_newton_schulz5_bfloat16(self):
        G = torch.randn(32, 64, dtype=torch.bfloat16)
        out = newton_schulz5(G, steps=5)
        self.assertEqual(out.shape, (32, 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())

    def test_hardware_status(self):
        status = get_hardware_status()
        self.assertIn("selected_device", status)
        self.assertIn("optimal_dtype", status)
        self.assertIn("tpu", status)
        self.assertIn("available", status["tpu"])

    def test_hybrid_opt_xla_step(self):
        model = Model(Config())
        opt = HybridOpt(model)

        mock_xm = MagicMock()
        mock_xla_module = MagicMock()
        mock_xla_module.core.xla_model = mock_xm

        fake_param = MagicMock()
        fake_param.device.type = "xla"
        opt.muon.param_groups[0]["params"] = [fake_param]

        with patch.dict("sys.modules", {"torch_xla": mock_xla_module, "torch_xla.core.xla_model": mock_xm}):
            opt.step(barrier=True)
            self.assertTrue(mock_xm.optimizer_step.called)

if __name__ == "__main__":
    unittest.main()
