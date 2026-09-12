import sys
import os
import unittest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model
from maba.optimizers import HybridOpt
from maba.dataset import Dataset
from maba.tokenizer import Tokenizer

torch.set_num_threads(min(8, os.cpu_count() or 4))

class TestEndToEndTraining(unittest.TestCase):
    def test_e2e_training(self):
        dev = "cpu"
        cfg = Config()
        model = Model(cfg).to(dev)

        tok = Tokenizer()
        ds = Dataset(tokenizer=tok, seq_len=32, repeat=10)
        dl = DataLoader(ds, batch_size=2, shuffle=True)
        opt = HybridOpt(model, lr_muon=0.03, wd_muon=0.01, lr_adamw=2e-3, wd_adamw=0.05)

        model.train()
        n_steps = 2
        losses = []
        data_iter = iter(dl)

        for s in range(n_steps):
            try:
                b = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                b = next(data_iter)

            x = b["input_ids"].to(dev)
            y = b["labels"].to(dev)

            opt.zero_grad()
            out = model(x, labels=y)
            tot_loss = out["loss"]["total_loss"]
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(tot_loss.item())
            del out, tot_loss
            opt.zero_grad(set_to_none=True)

        self.assertEqual(len(losses), n_steps)

        ckpt = "test_tmp_ckpt.pt"
        torch.save(model.state_dict(), ckpt)
        self.assertTrue(os.path.exists(ckpt))

        m2 = Model(cfg)
        m2.load_state_dict(torch.load(ckpt, weights_only=True))
        if os.path.exists(ckpt):
            os.remove(ckpt)

    def test_checkpoint_weights_only_payload_roundtrip(self):
        cfg = Config(vocab_size=100, dim=64, n_layers=2, n_passes=1, layer_types=[0, 1], n_heads=2, d_head=32, n_kv_heads=1, d_ffn=128)
        model = Model(cfg)
        ckpt = "test_full_payload_ckpt.pt"
        try:
            torch.save({
                "config": cfg,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "step": 3,
                "loss": 1.5,
            }, ckpt)
            loaded = torch.load(ckpt, weights_only=True)
            self.assertIn("config", loaded)
            self.assertIn("model_state_dict", loaded)
            self.assertEqual(loaded["config"].dim, 64)
            m2 = Model(loaded["config"])
            incompat = m2.load_state_dict(loaded["model_state_dict"])
            self.assertEqual(len(incompat.missing_keys), 0)
            self.assertEqual(len(incompat.unexpected_keys), 0)
        finally:
            if os.path.exists(ckpt):
                os.remove(ckpt)

if __name__ == "__main__":
    unittest.main()
