import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.layers.rms_norm import RMSNorm
from maba.layers.embeddings import EmbHead
from maba.layers.rope import RotaryEmbedding, apply_rope
from maba.layers.swiglu import SwiGLU
from maba.layers.gated_residual import GatedRes
from maba.layers.gdn2 import GDN2
from maba.layers.gqa import GQA
from maba.layers.transformer_block import Block
from maba.layers.mtp import MTPHead
from maba.optimizers.muon import newton_schulz5, Muon
from maba.optimizers.hybrid_optimizer import HybridOpt

class TestComponents(unittest.TestCase):
    def test_embeddings(self):
        head = EmbHead(vocab_size=32768, d_emb=128, dim=640)
        toks = torch.randint(0, 32768, (2, 16))
        h = head.forward_in(toks)
        self.assertEqual(h.shape, (2, 16, 640))
        lg = head.forward_out(h)
        self.assertEqual(lg.shape, (2, 16, 32768))
        fe = head.factor_emb(toks)
        self.assertEqual(fe.shape, (2, 16, 128))

    def test_rmsnorm(self):
        norm = RMSNorm(640)
        x = torch.randn(2, 16, 640)
        out = norm(x)
        self.assertEqual(out.shape, (2, 16, 640))
        v = out.pow(2).mean(-1)
        self.assertTrue(torch.allclose(v, torch.ones_like(v), atol=1e-2))

    def test_rope(self):
        rope = RotaryEmbedding(dim=64, max_len=128, theta=500000.0)
        dummy = torch.randn(1, 16, 640)
        cos, sin = rope(dummy, 16)
        self.assertEqual(cos.shape, (16, 64))
        self.assertEqual(sin.shape, (16, 64))

        q = torch.randn(2, 10, 16, 64)
        k = torch.randn(2, 10, 16, 64)
        qr, kr = apply_rope(q, k, cos, sin)
        self.assertEqual(qr.shape, q.shape)
        self.assertEqual(kr.shape, k.shape)
        self.assertTrue(torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))

    def test_swiglu(self):
        ffn = SwiGLU(dim=640, d_ffn=1728)
        x = torch.randn(2, 8, 640)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 8, 640))

    def test_gated_res(self):
        res = GatedRes(dim=640, bias_init=2.0)
        x = torch.randn(2, 4, 640)
        sub = torch.randn(2, 4, 640)
        out = res(x, sub)
        self.assertEqual(out.shape, (2, 4, 640))
        g = torch.sigmoid(torch.tensor(2.0))
        exp = g * x + (1.0 - g) * sub
        self.assertTrue(torch.allclose(out, exp, atol=1e-5))

    def test_gdn2(self):
        gdn = GDN2(dim=640, n_heads=10, d_head=64, k_size=4)
        x = torch.randn(2, 12, 640)
        out, S, cs = gdn(x)
        self.assertEqual(out.shape, (2, 12, 640))
        self.assertEqual(S.shape, (2, 10, 64, 64))

        x_step = torch.randn(2, 1, 640)
        out_s, S_next, _ = gdn(x_step, s_rec=S, cs_q=cs[0], cs_k=cs[1], cs_v=cs[2])
        self.assertEqual(out_s.shape, (2, 1, 640))
        self.assertEqual(S_next.shape, (2, 10, 64, 64))

    def test_gqa(self):
        gqa = GQA(dim=640, n_heads=10, n_kv_heads=2, d_head=64)
        rope = RotaryEmbedding(dim=64, max_len=32, theta=500000.0)
        x = torch.randn(2, 8, 640)
        cos, sin = rope(x, 8)
        out, (kc, vc) = gqa(x, cos, sin)
        self.assertEqual(out.shape, (2, 8, 640))
        self.assertEqual(kc.shape, (2, 2, 8, 64))
        self.assertEqual(vc.shape, (2, 2, 8, 64))

    def test_block(self):
        bgdn = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        x = torch.randn(2, 6, 640)
        out, _ = bgdn(x)
        self.assertEqual(out.shape, (2, 6, 640))

        bgqa = Block(dim=640, d_ffn=1728, is_gqa=True, n_passes=2)
        rope = RotaryEmbedding(dim=64, max_len=16)
        cos, sin = rope(x, 6)
        out_g, _ = bgqa(x, cos=cos, sin=sin)
        self.assertEqual(out_g.shape, (2, 6, 640))

    def test_mtp(self):
        head = EmbHead(vocab_size=32768, d_emb=128, dim=640)
        mtp = MTPHead(dim=640, d_emb=128)
        h = torch.randn(2, 8, 640)
        next_toks = torch.randint(0, 32768, (2, 8))
        fe = head.factor_emb(next_toks)
        lg = mtp(h, fe, head)
        self.assertEqual(lg.shape, (2, 8, 32768))

    def test_newton_schulz(self):
        G = torch.randn(64, 64)
        ortho = newton_schulz5(G, steps=5)
        self.assertEqual(ortho.shape, (64, 64))
        prod = ortho @ ortho.T
        diag = prod.diag()
        off = (prod - torch.diag(diag)).abs()
        self.assertTrue(diag.mean() > 0.6)
        self.assertTrue(off.max() < 0.25)

if __name__ == "__main__":
    unittest.main()
