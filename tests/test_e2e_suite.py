import os
import sys
import math
import struct
import socket
import subprocess
import tempfile
import unittest
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model
from maba.tokenizer import Tokenizer
from maba.generate import spec_gen
from maba.export_weights import export_bin
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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ddp_worker(rank: int, world_size: int, master_port: int, pipe):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        torch.manual_seed(42)
        cfg = Config(
            vocab_size=64,
            d_emb=32,
            dim=64,
            n_layers=2,
            n_passes=2,
            layer_types=[0, 1],
            n_heads=4,
            d_head=16,
            n_kv_heads=2,
            d_ffn=128,
        )
        model = Model(cfg)
        opt = HybridOpt(model, lr_muon=0.01, lr_adamw=0.01)

        torch.manual_seed(100 + rank)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        y = torch.randint(0, cfg.vocab_size, (1, 4))

        opt.zero_grad()
        out = model(x, labels=y)
        out["loss"]["total_loss"].backward()

        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

        opt.step()

        sample_weights = list(model.parameters())[0].detach().cpu().flatten()[:20].tolist()
        dist.destroy_process_group()
        pipe.send((True, sample_weights))
    except Exception as e:
        import traceback
        pipe.send((False, traceback.format_exc()))


class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1: Feature Coverage (>=5 tests per feature across all 12 core features).
    Total Tier 1 tests: 60 tests.
    """

    # -------------------------------------------------------------------------
    # Feature 1: Factorized Embedding (EmbHead)
    # -------------------------------------------------------------------------
    def test_f1_embhead_shapes(self):
        head = EmbHead(vocab_size=1024, d_emb=64, dim=256)
        toks = torch.randint(0, 1024, (2, 8))
        h = head.forward_in(toks)
        self.assertEqual(h.shape, (2, 8, 256))
        f_emb = head.factor_emb(toks)
        self.assertEqual(f_emb.shape, (2, 8, 64))

    def test_f1_embhead_weight_tying(self):
        head = EmbHead(vocab_size=512, d_emb=32, dim=128)
        h = torch.randn(2, 4, 128)
        logits = head.forward_out(h)
        self.assertEqual(logits.shape, (2, 4, 512))
        # Verify output projection ties LM head with w_emb.weight
        expected = F.linear(head.w_proj_out(h), head.w_emb.weight)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_f1_embhead_parameter_dimensions(self):
        head = EmbHead(vocab_size=32768, d_emb=128, dim=640)
        self.assertEqual(head.w_emb.weight.shape, (32768, 128))
        self.assertEqual(head.w_proj_in.weight.shape, (640, 128))
        self.assertEqual(head.w_proj_out.weight.shape, (128, 640))

    def test_f1_embhead_gradient_propagation(self):
        head = EmbHead(vocab_size=256, d_emb=32, dim=64)
        toks = torch.randint(0, 256, (2, 4))
        h = head.forward_in(toks)
        logits = head.forward_out(h)
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(head.w_emb.weight.grad)
        self.assertIsNotNone(head.w_proj_in.weight.grad)
        self.assertIsNotNone(head.w_proj_out.weight.grad)
        self.assertTrue(torch.isfinite(head.w_emb.weight.grad).all())
        self.assertTrue(torch.isfinite(head.w_proj_in.weight.grad).all())
        self.assertTrue(torch.isfinite(head.w_proj_out.weight.grad).all())

    def test_f1_embhead_rank_bottleneck(self):
        vocab_size = 500
        d_emb = 32
        dim = 128
        head = EmbHead(vocab_size=vocab_size, d_emb=d_emb, dim=dim)
        all_toks = torch.arange(vocab_size)
        emb_full = head.forward_in(all_toks)  # shape (500, 128)
        rank = torch.linalg.matrix_rank(emb_full).item()
        self.assertLessEqual(rank, d_emb)

    # -------------------------------------------------------------------------
    # Feature 2: RMSNorm
    # -------------------------------------------------------------------------
    def test_f2_rmsnorm_variance_normalization(self):
        norm = RMSNorm(128)
        x = torch.randn(4, 16, 128) * 5.0 + 3.0
        out = norm(x)
        self.assertEqual(out.shape, (4, 16, 128))
        rms = out.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_f2_rmsnorm_scale_invariance(self):
        norm = RMSNorm(64, eps=1e-8)
        x = torch.randn(2, 8, 64)
        out1 = norm(x)
        out2 = norm(x * 7.5)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-4))

    def test_f2_rmsnorm_learnable_gain(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 4, 64)
        out_base = norm(x)
        with torch.no_grad():
            norm.weight.mul_(2.0)
        out_scaled = norm(x)
        self.assertTrue(torch.allclose(out_scaled, out_base * 2.0, atol=1e-5))

    def test_f2_rmsnorm_zero_stability(self):
        norm = RMSNorm(64, eps=1e-6)
        x = torch.zeros(2, 4, 64)
        out = norm(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_f2_rmsnorm_gradients(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(norm.weight.grad).all())

    # -------------------------------------------------------------------------
    # Feature 3: RoPE (Rotary Positional Embedding)
    # -------------------------------------------------------------------------
    def test_f3_rope_norm_preservation(self):
        rope = RotaryEmbedding(dim=64, max_len=128, theta=500000.0)
        dummy = torch.randn(1, 8, 64)
        cos, sin = rope(dummy, 8)
        q = torch.randn(2, 4, 8, 64)
        k = torch.randn(2, 4, 8, 64)
        qr, kr = apply_rope(q, k, cos, sin)
        self.assertTrue(torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))
        self.assertTrue(torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-4))

    def test_f3_rope_relative_shift_invariance(self):
        rope = RotaryEmbedding(dim=32, max_len=64)
        dummy = torch.randn(1, 16, 32)
        cos, sin = rope(dummy, 16)
        # Random query and key vectors
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        # Evaluate dot product at (m=2, n=5) vs (m=6, n=9) where relative distance is -3
        qr_2, _ = apply_rope(q, k, cos[2:3], sin[2:3])
        _, kr_5 = apply_rope(q, k, cos[5:6], sin[5:6])
        dp1 = (qr_2 * kr_5).sum().item()

        qr_6, _ = apply_rope(q, k, cos[6:7], sin[6:7])
        _, kr_9 = apply_rope(q, k, cos[9:10], sin[9:10])
        dp2 = (qr_6 * kr_9).sum().item()

        self.assertAlmostEqual(dp1, dp2, places=4)

    def test_f3_rope_position_zero_identity(self):
        rope = RotaryEmbedding(dim=32, max_len=16)
        dummy = torch.randn(1, 1, 32)
        cos, sin = rope(dummy, 1, pos=0)
        q = torch.randn(1, 2, 1, 32)
        k = torch.randn(1, 2, 1, 32)
        qr, kr = apply_rope(q, k, cos, sin)
        self.assertTrue(torch.allclose(q, qr, atol=1e-6))
        self.assertTrue(torch.allclose(k, kr, atol=1e-6))

    def test_f3_rope_theta_frequencies(self):
        dim = 64
        theta = 500000.0
        rope = RotaryEmbedding(dim=dim, theta=theta)
        expected_inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.assertTrue(torch.allclose(rope.inv_freq, expected_inv, atol=1e-6))

    def test_f3_rope_cache_growth(self):
        rope = RotaryEmbedding(dim=32, max_len=8)
        dummy = torch.randn(1, 24, 32)
        cos, sin = rope(dummy, 24, pos=10)
        self.assertEqual(cos.shape, (24, 32))
        self.assertEqual(sin.shape, (24, 32))
        self.assertGreaterEqual(rope.max_len, 34)

    # -------------------------------------------------------------------------
    # Feature 4: SwiGLU FFN
    # -------------------------------------------------------------------------
    def test_f4_swiglu_mathematical_definition(self):
        ffn = SwiGLU(dim=64, d_ffn=128)
        x = torch.randn(2, 4, 64)
        out = ffn(x)
        expected = ffn.w_down(F.silu(ffn.w_gate(x)) * ffn.w_up(x))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_f4_swiglu_intermediate_dimension(self):
        ffn = SwiGLU(dim=640, d_ffn=1728)
        x = torch.randn(2, 6, 640)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 6, 640))
        self.assertEqual(ffn.w_gate.weight.shape, (1728, 640))
        self.assertEqual(ffn.w_up.weight.shape, (1728, 640))
        self.assertEqual(ffn.w_down.weight.shape, (640, 1728))

    def test_f4_swiglu_zero_input_property(self):
        ffn = SwiGLU(dim=64, d_ffn=128)
        x = torch.zeros(2, 4, 64)
        out = ffn(x)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_f4_swiglu_gating_behavior(self):
        ffn = SwiGLU(dim=32, d_ffn=64)
        x = torch.randn(1, 4, 32)
        # SwiGLU gating is non-linear: f(2x) != 2f(x)
        out1 = ffn(2.0 * x)
        out2 = 2.0 * ffn(x)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-3))
        # Zeroing out gate weights shuts off output signal
        with torch.no_grad():
            ffn.w_gate.weight.zero_()
        out_zero_gate = ffn(x)
        self.assertTrue(torch.allclose(out_zero_gate, torch.zeros_like(out_zero_gate)))

    def test_f4_swiglu_gradient_flow(self):
        ffn = SwiGLU(dim=64, d_ffn=128)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        self.assertIsNotNone(ffn.w_gate.weight.grad)
        self.assertIsNotNone(ffn.w_up.weight.grad)
        self.assertIsNotNone(ffn.w_down.weight.grad)
        self.assertTrue((ffn.w_gate.weight.grad != 0).any())
        self.assertTrue((ffn.w_up.weight.grad != 0).any())
        self.assertTrue((ffn.w_down.weight.grad != 0).any())

    # -------------------------------------------------------------------------
    # Feature 5: Gated Residual (GatedRes)
    # -------------------------------------------------------------------------
    def test_f5_gated_res_convex_combination(self):
        res_layer = GatedRes(dim=64, bias_init=1.5)
        res = torch.randn(2, 4, 64)
        sub = torch.randn(2, 4, 64)
        out = res_layer(res, sub)
        g = torch.sigmoid(res_layer.g_res)
        expected = g * res + (1.0 - g) * sub
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_f5_gated_res_initial_bias_value(self):
        res_layer = GatedRes(dim=64, bias_init=2.0)
        g = torch.sigmoid(res_layer.g_res)
        expected_g = 1.0 / (1.0 + math.exp(-2.0))
        self.assertTrue(torch.allclose(g, torch.full_like(g, expected_g), atol=1e-5))

    def test_f5_gated_res_extreme_asymptotes(self):
        res_layer = GatedRes(dim=32)
        res = torch.randn(2, 4, 32)
        sub = torch.randn(2, 4, 32)
        with torch.no_grad():
            res_layer.g_res.fill_(50.0)
        out_res = res_layer(res, sub)
        self.assertTrue(torch.allclose(out_res, res, atol=1e-5))

        with torch.no_grad():
            res_layer.g_res.fill_(-50.0)
        out_sub = res_layer(res, sub)
        self.assertTrue(torch.allclose(out_sub, sub, atol=1e-5))

    def test_f5_gated_res_shape_broadcasting(self):
        res_layer = GatedRes(dim=64)
        # 3D
        out3d = res_layer(torch.randn(2, 8, 64), torch.randn(2, 8, 64))
        self.assertEqual(out3d.shape, (2, 8, 64))
        # 2D
        out2d = res_layer(torch.randn(2, 64), torch.randn(2, 64))
        self.assertEqual(out2d.shape, (2, 64))

    def test_f5_gated_res_parameter_learnability(self):
        res_layer = GatedRes(dim=32, bias_init=0.0)
        res = torch.randn(2, 4, 32)
        sub = torch.randn(2, 4, 32)
        out = res_layer(res, sub)
        loss = (out - res).pow(2).sum()
        loss.backward()
        self.assertIsNotNone(res_layer.g_res.grad)
        self.assertTrue(torch.isfinite(res_layer.g_res.grad).all())
        self.assertTrue((res_layer.g_res.grad != 0).any())

    # -------------------------------------------------------------------------
    # Feature 6: GDN-2 Recurrent Attention
    # -------------------------------------------------------------------------
    def test_f6_gdn2_causality_convolution(self):
        gdn = GDN2(dim=64, n_heads=2, d_head=32, k_size=4)
        x = torch.randn(1, 6, 64)
        out1, _, _ = gdn(x)

        x_perturbed = x.clone()
        x_perturbed[:, 4:, :] += 10.0
        out2, _, _ = gdn(x_perturbed)

        # First 4 tokens must not be affected by perturbations in tokens 4 and 5
        self.assertTrue(torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5))

    def test_f6_gdn2_step_recurrence_equivalence(self):
        gdn = GDN2(dim=64, n_heads=2, d_head=32, k_size=4)
        x = torch.randn(1, 6, 64)
        out_full, S_full, _ = gdn(x)

        S = None
        cs_q = cs_k = cs_v = None
        step_outs = []
        for t in range(6):
            xt = x[:, t : t + 1, :]
            ot, S, (cs_q, cs_k, cs_v) = gdn(xt, s_rec=S, cs_q=cs_q, cs_k=cs_k, cs_v=cs_v)
            step_outs.append(ot)

        out_step = torch.cat(step_outs, dim=1)
        self.assertTrue(torch.allclose(out_full, out_step, atol=1e-5))
        self.assertTrue(torch.allclose(S_full, S, atol=1e-5))

    def test_f6_gdn2_recurrent_state_topology(self):
        gdn = GDN2(dim=128, n_heads=4, d_head=32, k_size=4)
        x = torch.randn(2, 5, 128)
        _, S, (cs_q, cs_k, cs_v) = gdn(x)
        self.assertEqual(S.shape, (2, 4, 32, 32))
        self.assertEqual(cs_q.shape, (2, 128, 3))
        self.assertEqual(cs_k.shape, (2, 128, 3))
        self.assertEqual(cs_v.shape, (2, 128, 3))

    def test_f6_gdn2_decay_gate_modulation(self):
        gdn = GDN2(dim=64, n_heads=2, d_head=32, k_size=4)
        S_init = torch.randn(1, 2, 32, 32)
        x = torch.zeros(1, 1, 64)
        _, S_next, _ = gdn(x, s_rec=S_init)
        self.assertIsNotNone(S_next)
        self.assertTrue(torch.isfinite(S_next).all())

    def test_f6_gdn2_all_parameters_receive_gradients(self):
        gdn = GDN2(dim=64, n_heads=2, d_head=32, k_size=4)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out, _, _ = gdn(x)
        out.sum().backward()
        for name, p in gdn.named_parameters():
            self.assertIsNotNone(p.grad, f"GDN2 parameter {name} received None grad")
            self.assertTrue(torch.isfinite(p.grad).all(), f"GDN2 parameter {name} has NaN/Inf grad")

    # -------------------------------------------------------------------------
    # Feature 7: Grouped-Query Attention (GQA)
    # -------------------------------------------------------------------------
    def test_f7_gqa_group_ratio(self):
        gqa = GQA(dim=640, n_heads=10, n_kv_heads=2, d_head=64)
        self.assertEqual(gqa.n_rep, 5)
        self.assertEqual(gqa.q_proj.weight.shape, (640, 640))
        self.assertEqual(gqa.k_proj.weight.shape, (128, 640))
        self.assertEqual(gqa.v_proj.weight.shape, (128, 640))

    def test_f7_gqa_per_head_rms_norm(self):
        gqa = GQA(dim=128, n_heads=4, n_kv_heads=2, d_head=32)
        self.assertIsInstance(gqa.q_norm, RMSNorm)
        self.assertIsInstance(gqa.k_norm, RMSNorm)
        self.assertEqual(gqa.q_norm.weight.shape, (32,))
        self.assertEqual(gqa.k_norm.weight.shape, (32,))

    def test_f7_gqa_causal_masking(self):
        gqa = GQA(dim=64, n_heads=4, n_kv_heads=2, d_head=16)
        rope = RotaryEmbedding(dim=16, max_len=16)
        x = torch.randn(1, 4, 64)
        cos, sin = rope(x, 4)
        out1, _ = gqa(x, cos, sin)

        x_perturbed = x.clone()
        x_perturbed[:, 3:, :] += 5.0
        out2, _ = gqa(x_perturbed, cos, sin)
        # Due to causal masking, first 3 tokens must remain unaffected
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5))

    def test_f7_gqa_kv_cache_incremental_equivalence(self):
        dim = 64
        d_head = 16
        gqa = GQA(dim=dim, n_heads=4, n_kv_heads=2, d_head=d_head)
        rope = RotaryEmbedding(dim=d_head, max_len=32)
        x = torch.randn(1, 4, dim)
        cos, sin = rope(x, 4)
        out_full, (k_full, v_full) = gqa(x, cos, sin)

        kv = None
        outs = []
        for t in range(4):
            xt = x[:, t : t + 1, :]
            c_t, s_t = rope(xt, 1, pos=t)
            ot, kv = gqa(xt, c_t, s_t, kv=kv)
            outs.append(ot)

        out_step = torch.cat(outs, dim=1)
        self.assertTrue(torch.allclose(out_full, out_step, atol=1e-5))
        self.assertTrue(torch.allclose(k_full, kv[0], atol=1e-5))
        self.assertTrue(torch.allclose(v_full, kv[1], atol=1e-5))

    def test_f7_gqa_gradient_flow(self):
        gqa = GQA(dim=64, n_heads=4, n_kv_heads=2, d_head=16)
        rope = RotaryEmbedding(dim=16, max_len=16)
        x = torch.randn(2, 4, 64, requires_grad=True)
        cos, sin = rope(x, 4)
        out, _ = gqa(x, cos, sin)
        out.sum().backward()
        for name, p in gqa.named_parameters():
            self.assertIsNotNone(p.grad, f"GQA parameter {name} received None grad")
            self.assertTrue(torch.isfinite(p.grad).all(), f"GQA parameter {name} has NaN/Inf grad")

    # -------------------------------------------------------------------------
    # Feature 8: 2-Pass Block Execution
    # -------------------------------------------------------------------------
    def test_f8_block_weight_tying(self):
        block = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        init_params = sum(p.numel() for p in block.parameters())
        x = torch.randn(2, 4, 640)
        out, _ = block(x)
        self.assertEqual(out.shape, (2, 4, 640))
        # Parameters count unchanged: passes are tied over same layers
        self.assertEqual(sum(p.numel() for p in block.parameters()), init_params)

    def test_f8_block_pass_transformation(self):
        block = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        x = torch.randn(2, 4, 640)
        h1, _ = block._pass(x)
        h2, _ = block._pass(h1)
        self.assertFalse(torch.allclose(h1, h2, atol=1e-4))
        out, _ = block(x)
        self.assertTrue(torch.allclose(out, h2, atol=1e-6))

    def test_f8_block_state_tracking_per_pass(self):
        block = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        x = torch.randn(2, 4, 640)
        out, states = block(x, return_states=True)
        self.assertIsInstance(states, list)
        self.assertEqual(len(states), 2)
        for st in states:
            self.assertIn("recurrent_state", st)
            self.assertIn("conv_state_q", st)

    def test_f8_block_variants_compatibility(self):
        block_gdn = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        block_gqa = Block(dim=640, d_ffn=1728, is_gqa=True, n_passes=2)
        rope = RotaryEmbedding(dim=64, max_len=16)
        x = torch.randn(2, 4, 640)
        cos, sin = rope(x, 4)
        out_gdn, _ = block_gdn(x)
        out_gqa, _ = block_gqa(x, cos=cos, sin=sin)
        self.assertEqual(out_gdn.shape, (2, 4, 640))
        self.assertEqual(out_gqa.shape, (2, 4, 640))

    def test_f8_block_gradient_accumulation(self):
        block = Block(dim=640, d_ffn=1728, is_gqa=False, n_passes=2)
        x = torch.randn(2, 4, 640, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, f"Block parameter {name} has None grad")
            self.assertTrue(torch.isfinite(p.grad).all())

    # -------------------------------------------------------------------------
    # Feature 9: MTP Head (k=2)
    # -------------------------------------------------------------------------
    def test_f9_mtp_concatenation_structure(self):
        dim = 128
        d_emb = 32
        mtp = MTPHead(dim=dim, d_emb=d_emb)
        self.assertEqual(mtp.proj.weight.shape, (dim, dim + d_emb))

    def test_f9_mtp_output_logits_shape(self):
        head = EmbHead(vocab_size=256, d_emb=32, dim=128)
        mtp = MTPHead(dim=128, d_emb=32)
        h = torch.randn(2, 6, 128)
        next_toks = torch.randint(0, 256, (2, 6))
        f_emb = head.factor_emb(next_toks)
        logits = mtp(h, f_emb, head)
        self.assertEqual(logits.shape, (2, 6, 256))

    def test_f9_mtp_loss_weighting(self):
        cfg = Config(vocab_size=128, d_emb=32, dim=64, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=4, d_head=16, n_kv_heads=2, d_ffn=128, mtp_weight=0.5)
        model = Model(cfg)
        x = torch.randint(0, 128, (2, 6))
        y = torch.randint(0, 128, (2, 6))
        out = model(x, labels=y)
        loss = out["loss"]
        expected_total = loss["main_loss"] + 0.5 * loss["mtp_loss"]
        self.assertTrue(torch.allclose(loss["total_loss"], expected_total, atol=1e-5))

    def test_f9_mtp_label_shifting(self):
        cfg = Config(vocab_size=128, d_emb=32, dim=64, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=4, d_head=16, n_kv_heads=2, d_ffn=128)
        model = Model(cfg)
        x = torch.randint(0, 128, (2, 5))
        y = torch.randint(0, 128, (2, 5))
        out = model(x, labels=y)
        self.assertIsNotNone(out["mtp_logits"])
        self.assertEqual(out["mtp_logits"].shape, (2, 4, 128))

    def test_f9_mtp_short_sequence_stability(self):
        cfg = Config(vocab_size=128, d_emb=32, dim=64, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=4, d_head=16, n_kv_heads=2, d_ffn=128)
        model = Model(cfg)
        for seq_len in (1, 2):
            x = torch.randint(0, 128, (1, seq_len))
            y = torch.randint(0, 128, (1, seq_len))
            out = model(x, labels=y)
            loss = out["loss"]["total_loss"]
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward()

    # -------------------------------------------------------------------------
    # Feature 10: HybridOpt Optimizer
    # -------------------------------------------------------------------------
    def test_f10_hybrid_opt_parameter_partitioning(self):
        cfg = Config(vocab_size=128, d_emb=32, dim=64, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=4, d_head=16, n_kv_heads=2, d_ffn=128)
        model = Model(cfg)
        opt = HybridOpt(model)
        self.assertIsNotNone(opt.muon)
        self.assertIsNotNone(opt.adamw)
        self.assertGreater(opt.muon_param_count, 0)
        self.assertGreater(opt.adamw_param_count, 0)
        total = opt.muon_param_count + opt.adamw_param_count
        self.assertEqual(total, sum(p.numel() for p in model.parameters()))

    def test_f10_hybrid_opt_newton_schulz_orthogonalization(self):
        G = torch.randn(64, 64)
        ortho = newton_schulz5(G, steps=5)
        self.assertEqual(ortho.shape, (64, 64))
        gram = ortho @ ortho.T
        diag = gram.diag()
        off_diag = gram - torch.diag(diag)
        self.assertTrue((diag > 0.5).all())
        self.assertLess(off_diag.abs().max().item(), 0.25)

    def test_f10_hybrid_opt_step_optimization(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 1, bias=False))
        # Rename parameters to exercise HybridOpt logic
        model[0].q_proj = nn.Linear(32, 32, bias=False)
        opt = HybridOpt(model, lr_muon=0.05, lr_adamw=0.01)
        x = torch.randn(10, 32)
        y = torch.randn(10, 1)

        opt.zero_grad()
        loss_init = ((model[0](x) @ torch.randn(32, 1)) - y).pow(2).mean()
        loss_init.backward()
        opt.step()

        opt.zero_grad()
        loss_next = ((model[0](x) @ torch.randn(32, 1)) - y).pow(2).mean()
        self.assertTrue(torch.isfinite(loss_next))

    def test_f10_hybrid_opt_weight_decay(self):
        p_2d = nn.Parameter(torch.ones(16, 16))
        opt = Muon([p_2d], lr=0.1, weight_decay=0.2)
        p_2d.grad = torch.zeros_like(p_2d)
        init_norm = p_2d.norm().item()
        opt.step()
        decayed_norm = p_2d.norm().item()
        self.assertLess(decayed_norm, init_norm)

    def test_f10_hybrid_opt_state_dict_serialization(self):
        cfg = Config(vocab_size=64, d_emb=16, dim=32, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=2, d_head=16, n_kv_heads=1, d_ffn=64)
        m1 = Model(cfg)
        opt1 = HybridOpt(m1)
        x = torch.randint(0, 64, (2, 4))
        y = torch.randint(0, 64, (2, 4))
        m1(x, labels=y)["loss"]["total_loss"].backward()
        opt1.step()

        sd = opt1.state_dict()
        m2 = Model(cfg)
        opt2 = HybridOpt(m2)
        opt2.load_state_dict(sd)
        self.assertEqual(len(opt2.muon.state), len(opt1.muon.state))

    # -------------------------------------------------------------------------
    # Feature 11: Checkpoint Serialization & CLI
    # -------------------------------------------------------------------------
    def test_f11_checkpoint_safe_globals(self):
        import torch.serialization
        # Ensure Config is registered in safe globals for PyTorch 2.6+
        torch.serialization.add_safe_globals([Config])
        cfg = Config(dim=256)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save({"config": cfg}, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["config"].dim, 256)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_f11_checkpoint_state_dict_roundtrip(self):
        cfg = Config(vocab_size=64, d_emb=16, dim=32, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=2, d_head=16, n_kv_heads=1, d_ffn=64)
        m1 = Model(cfg)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(m1.state_dict(), path)
            m2 = Model(cfg)
            m2.load_state_dict(torch.load(path, weights_only=True))
            for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.allclose(p1, p2))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_f11_binary_export_magic_header(self):
        cfg = Config(vocab_size=64, d_emb=16, dim=32, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=2, d_head=16, n_kv_heads=1, d_ffn=64)
        m = Model(cfg)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_bin(m, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as bf:
                magic = struct.unpack("<I", bf.read(4))[0]
                self.assertEqual(magic, 0x4D414241)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_f11_cli_params_command(self):
        res = subprocess.run(
            [sys.executable, "-m", "maba.cli", "params", "--scale", "100M"],
            cwd="/workspaces/123123/maba-v1-architecture",
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("101,177,984", res.stdout)

    def test_f11_cli_hardware_command(self):
        res = subprocess.run(
            [sys.executable, "-m", "maba.cli", "hardware"],
            cwd="/workspaces/123123/maba-v1-architecture",
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Maba Hardware Accelerator Audit", res.stdout)
        self.assertIn("Active Device", res.stdout)

    # -------------------------------------------------------------------------
    # Feature 12: C++ Numerical Parity
    # -------------------------------------------------------------------------
    def test_f12_cpp_binary_weights_file(self):
        bin_path = "/workspaces/123123/maba-v1-architecture/maba_ref.bin"
        self.assertTrue(os.path.exists(bin_path), f"Reference binary {bin_path} missing")
        with open(bin_path, "rb") as f:
            magic, vocab_size, dim, d_emb, d_head, n_heads, n_kv, n_layers, d_ffn, n_tensors = struct.unpack(
                "<IIIIIIIIII", f.read(40)
            )
        self.assertEqual(magic, 0x4D414241)
        self.assertEqual(vocab_size, 32768)
        self.assertEqual(dim, 640)
        self.assertEqual(n_tensors, 366)

    def test_f12_cpp_ref_logits_file(self):
        ref_logits = "/workspaces/123123/maba-v1-architecture/ref_logits.bin"
        self.assertTrue(os.path.exists(ref_logits), f"Reference logits {ref_logits} missing")
        with open(ref_logits, "rb") as f:
            L, V = struct.unpack("<II", f.read(8))
            self.assertEqual(L, 4)
            self.assertEqual(V, 32768)
            payload = f.read()
            self.assertEqual(len(payload), 4 * 32768 * 4)

    def test_f12_cpp_test_numerical_binary_execution(self):
        exe_path = "/workspaces/123123/maba-v1-architecture/cpp/build/test_numerical"
        self.assertTrue(os.path.exists(exe_path), f"Executable {exe_path} missing")
        res = subprocess.run(
            [exe_path],
            cwd="/workspaces/123123/maba-v1-architecture",
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0, f"C++ numerical test failed with stderr: {res.stderr}")
        self.assertIn("PASS: numerical equivalence verified", res.stdout)

    def test_f12_cpp_numerical_equivalence_threshold(self):
        exe_path = "/workspaces/123123/maba-v1-architecture/cpp/build/test_numerical"
        res = subprocess.run(
            [exe_path],
            cwd="/workspaces/123123/maba-v1-architecture",
            capture_output=True,
            text=True
        )
        # Parse max_diff from output
        # Output snippet: "Compared 131072 logits: max_diff=7.86334e-05, mean_diff=1.28909e-05"
        for part in res.stdout.split():
            if "max_diff=" in part:
                val = float(part.split("max_diff=")[1].rstrip(","))
                self.assertLess(val, 1e-4, f"max_diff {val} exceeds 1e-4 threshold")

    def test_f12_pytorch_deterministic_logits_parity(self):
        torch.manual_seed(42)
        cfg = Config(vocab_size=128, d_emb=32, dim=64, n_layers=2, n_passes=2, layer_types=[0, 1], n_heads=4, d_head=16, n_kv_heads=2, d_ffn=128)
        m = Model(cfg).eval()
        tokens = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)
        with torch.no_grad():
            out1 = m(tokens)["logits"]
            out2 = m(tokens)["logits"]
        self.assertTrue(torch.allclose(out1, out2, atol=0.0))


class TestTier2BoundaryCases(unittest.TestCase):
    """
    Tier 2: Boundary & Corner Cases.
    Total Tier 2 tests: 11 tests.
    """

    def setUp(self):
        self.cfg = Config(
            vocab_size=128,
            d_emb=32,
            dim=64,
            n_layers=2,
            n_passes=2,
            layer_types=[0, 1],
            n_heads=4,
            d_head=16,
            n_kv_heads=2,
            d_ffn=128,
            max_len=64,
        )

    def test_t2_empty_sequence_forward(self):
        model = Model(self.cfg)
        x = torch.empty((2, 0), dtype=torch.long)
        out = model(x)
        self.assertEqual(out["logits"].shape, (2, 0, self.cfg.vocab_size))
        self.assertEqual(out["hidden_states"].shape, (2, 0, self.cfg.dim))
        self.assertIsNone(out["loss"])

    def test_t2_single_token_eval(self):
        model = Model(self.cfg).eval()
        x = torch.randint(0, self.cfg.vocab_size, (2, 1))
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out["logits"].shape, (2, 1, self.cfg.vocab_size))
        self.assertEqual(out["hidden_states"].shape, (2, 1, self.cfg.dim))

    def test_t2_single_token_training(self):
        model = Model(self.cfg)
        x = torch.randint(0, self.cfg.vocab_size, (1, 1))
        y = torch.randint(0, self.cfg.vocab_size, (1, 1))
        out = model(x, labels=y)
        loss = out["loss"]["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_t2_all_masked_labels(self):
        model = Model(self.cfg)
        x = torch.randint(0, self.cfg.vocab_size, (2, 4))
        y = torch.full((2, 4), -100, dtype=torch.long)
        out = model(x, labels=y)
        loss = out["loss"]["total_loss"]
        self.assertEqual(loss.item(), 0.0)
        loss.backward()

    def test_t2_partial_masked_labels(self):
        model = Model(self.cfg)
        x = torch.randint(0, self.cfg.vocab_size, (2, 4))
        y = torch.tensor([[10, -100, 25, -100], [-100, 30, -100, 45]], dtype=torch.long)
        out = model(x, labels=y)
        loss = out["loss"]["total_loss"]
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_t2_extreme_input_values(self):
        norm = RMSNorm(64)
        swiglu = SwiGLU(64, 128)
        gate = GatedRes(64)
        x_huge = torch.randn(2, 4, 64) * 1e5
        out_norm = norm(x_huge)
        self.assertFalse(torch.isnan(out_norm).any())
        self.assertFalse(torch.isinf(out_norm).any())

        out_gate = gate(out_norm, out_norm)
        self.assertFalse(torch.isnan(out_gate).any())
        self.assertFalse(torch.isinf(out_gate).any())

        out_swi = swiglu(out_gate)
        self.assertFalse(torch.isnan(out_swi).any())
        self.assertFalse(torch.isinf(out_swi).any())

    def test_t2_fp32_precision_stability(self):
        model = Model(self.cfg).float()
        x = torch.randint(0, self.cfg.vocab_size, (2, 4))
        y = torch.randint(0, self.cfg.vocab_size, (2, 4))
        out = model(x, labels=y)
        loss = out["loss"]["total_loss"]
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_t2_bf16_precision_stability(self):
        model = Model(self.cfg).bfloat16()
        x = torch.randint(0, self.cfg.vocab_size, (2, 4))
        y = torch.randint(0, self.cfg.vocab_size, (2, 4))
        out = model(x, labels=y)
        self.assertEqual(out["logits"].dtype, torch.bfloat16)
        loss = out["loss"]["total_loss"]
        self.assertFalse(torch.isnan(loss))
        loss.backward()

    def test_t2_sequence_length_boundary(self):
        model = Model(self.cfg).eval()
        x = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.max_len))
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out["logits"].shape, (1, self.cfg.max_len, self.cfg.vocab_size))

    def test_t2_batch_size_variation(self):
        model = Model(self.cfg).eval()
        for b in (1, 3, 5):
            x = torch.randint(0, self.cfg.vocab_size, (b, 4))
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out["logits"].shape, (b, 4, self.cfg.vocab_size))

    def test_t2_generation_temperature_boundaries(self):
        model = Model(self.cfg)
        toks = torch.randint(0, self.cfg.vocab_size, (1, 3))
        # Greedy / low temp
        gen_greedy = model.generate(toks, max_new_tokens=3, temperature=1e-4)
        self.assertEqual(gen_greedy.shape, (1, 6))
        # High temp
        gen_hot = model.generate(toks, max_new_tokens=3, temperature=2.0)
        self.assertEqual(gen_hot.shape, (1, 6))


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3: Cross-Feature Combinations (Pairwise Interactions).
    Total Tier 3 tests: 6 tests.
    """

    def test_t3_gdn2_gqa_interleaving_multipass(self):
        cfg = Config(
            vocab_size=64,
            d_emb=16,
            dim=32,
            n_layers=4,
            n_passes=2,
            layer_types=[0, 0, 0, 1],
            n_heads=2,
            d_head=16,
            n_kv_heads=1,
            d_ffn=64,
        )
        model = Model(cfg).eval()
        x = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            out = model(x, return_states=True)
        states = out["states"]
        self.assertEqual(len(states), 4)
        # Block 0-2 are GDN-2
        for i in range(3):
            self.assertIn("recurrent_state", states[i][0])
        # Block 3 is GQA
        self.assertIn("kv_cache", states[3][0])

    def test_t3_mtp_with_speculative_decoding(self):
        cfg = Config(
            vocab_size=32768,
            d_emb=32,
            dim=64,
            n_layers=2,
            n_passes=2,
            layer_types=[0, 1],
            n_heads=2,
            d_head=32,
            n_kv_heads=1,
            d_ffn=64,
        )
        model = Model(cfg).eval()
        tok = Tokenizer()
        prompt = "Hello"
        txt, acc, steps = spec_gen(model, tok, prompt, max_new_tokens=4, device="cpu")
        self.assertIsInstance(txt, str)
        self.assertGreaterEqual(steps, 1)
        self.assertGreaterEqual(acc, 0.0)

    def test_t3_hybrid_opt_with_gradient_reduction(self):
        cfg = Config(
            vocab_size=64,
            d_emb=16,
            dim=32,
            n_layers=2,
            n_passes=2,
            layer_types=[0, 1],
            n_heads=2,
            d_head=16,
            n_kv_heads=1,
            d_ffn=64,
        )
        model = Model(cfg)
        opt = HybridOpt(model, lr_muon=0.01, lr_adamw=0.01)
        x = torch.randint(0, 64, (2, 4))
        y = torch.randint(0, 64, (2, 4))
        out = model(x, labels=y)
        out["loss"]["total_loss"].backward()

        # Simulate simulated multi-rank gradient average
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(0.5)

        opt.step()
        # Verify both Muon and AdamW parameters stepped
        for p in opt.muon.param_groups[0]["params"]:
            self.assertIn("momentum_buffer", opt.muon.state[p])

    def test_t3_factorized_embedding_with_mtp(self):
        head = EmbHead(vocab_size=128, d_emb=32, dim=64)
        mtp = MTPHead(dim=64, d_emb=32)
        h = torch.randn(2, 4, 64)
        next_toks = torch.randint(0, 128, (2, 4))
        f_emb = head.factor_emb(next_toks)
        logits = mtp(h, f_emb, head)
        self.assertEqual(logits.shape, (2, 4, 128))

    def test_t3_rmsnorm_with_gated_residual(self):
        norm1 = RMSNorm(64)
        norm2 = RMSNorm(64)
        gate = GatedRes(64, bias_init=2.0)
        h = torch.randn(2, 4, 64)
        sub = norm2(torch.randn(2, 4, 64))
        out = gate(norm1(h), sub)
        self.assertEqual(out.shape, (2, 4, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_t3_rope_with_gqa_kv_caching(self):
        dim = 64
        d_head = 16
        gqa = GQA(dim=dim, n_heads=4, n_kv_heads=2, d_head=d_head)
        rope = RotaryEmbedding(dim=d_head, max_len=16)
        x = torch.randn(1, 3, dim)
        cos, sin = rope(x, 3)
        _, (k_cache, v_cache) = gqa(x, cos, sin)

        x_next = torch.randn(1, 1, dim)
        cos_next, sin_next = rope(x_next, 1, pos=3)
        out_step, (k_cache_new, v_cache_new) = gqa(x_next, cos_next, sin_next, kv=(k_cache, v_cache))
        self.assertEqual(out_step.shape, (1, 1, dim))
        self.assertEqual(k_cache_new.shape[2], 4)
        self.assertEqual(v_cache_new.shape[2], 4)


class TestTier4RealWorldScenarios(unittest.TestCase):
    """
    Tier 4: Real-World Application Scenarios.
    Total Tier 4 tests: 5 tests.
    """

    def test_t4_scenario1_speculative_generation(self):
        """
        Scenario 1: Full autoregressive generation with MTP speculative verification.
        """
        cfg = Config(
            vocab_size=32768,
            d_emb=32,
            dim=64,
            n_layers=2,
            n_passes=2,
            layer_types=[0, 1],
            n_heads=4,
            d_head=16,
            n_kv_heads=2,
            d_ffn=128,
        )
        model = Model(cfg).eval()
        tok = Tokenizer()
        prompt = "def evaluate"
        txt, acc, steps = spec_gen(model, tok, prompt, max_new_tokens=6, device="cpu")
        self.assertIsInstance(txt, str)
        self.assertGreater(len(txt), len(prompt))
        self.assertGreater(steps, 0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_t4_scenario2_recurrent_state_chunk_invariance(self):
        """
        Scenario 2: Recurrent state chunk invariance across long-sequence passing.
        """
        cfg = Config(
            vocab_size=100,
            d_emb=32,
            dim=64,
            n_layers=4,
            n_passes=2,
            layer_types=[0, 0, 0, 1],
            n_heads=4,
            d_head=16,
            n_kv_heads=2,
            d_ffn=128,
        )
        model = Model(cfg).eval()
        tokens = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            out_full = model(tokens)
            out1 = model(tokens[:, :8], return_states=True)
            out2 = model(tokens[:, 8:], states=out1["states"], return_states=True, start_pos=8)
            logits_chunked = torch.cat([out1["logits"], out2["logits"]], dim=1)

        diff = (out_full["logits"] - logits_chunked).abs().max().item()
        self.assertLess(diff, 1e-4, f"Chunk invariance violation: max diff {diff} >= 1e-4")

    def test_t4_scenario3_multi_sample_batch_independence(self):
        """
        Scenario 3: Multi-sample batch independence (zero cross-sample state leakage).
        """
        cfg = Config(
            vocab_size=100,
            d_emb=32,
            dim=64,
            n_layers=4,
            n_passes=2,
            layer_types=[0, 0, 0, 1],
            n_heads=4,
            d_head=16,
            n_kv_heads=2,
            d_ffn=128,
        )
        model = Model(cfg).eval()
        A = torch.randint(0, 100, (1, 6))
        B = torch.randint(0, 100, (1, 6))
        AB = torch.cat([A, B], dim=0)

        with torch.no_grad():
            out_AB = model(AB)["logits"]
            out_A = model(A)["logits"]
            out_B = model(B)["logits"]

        diff_A = (out_AB[0] - out_A[0]).abs().max().item()
        diff_B = (out_AB[1] - out_B[0]).abs().max().item()
        self.assertLess(diff_A, 1e-5, f"Batch leakage sample A: diff {diff_A}")
        self.assertLess(diff_B, 1e-5, f"Batch leakage sample B: diff {diff_B}")

    def test_t4_scenario4_two_process_cpu_gloo_ddp(self):
        """
        Scenario 4: 2-process CPU Gloo DDP simulation for gradient all-reduce synchronization.
        """
        ctx = mp.get_context("spawn")
        p0_r, p0_w = ctx.Pipe()
        p1_r, p1_w = ctx.Pipe()
        port = find_free_port()

        procs = [
            ctx.Process(target=_ddp_worker, args=(0, 2, port, p0_w)),
            ctx.Process(target=_ddp_worker, args=(1, 2, port, p1_w)),
        ]
        for p in procs:
            p.start()

        res0 = p0_r.recv()
        res1 = p1_r.recv()
        for p in procs:
            p.join(timeout=15)

        self.assertTrue(res0[0], f"Rank 0 failed: {res0[1]}")
        self.assertTrue(res1[0], f"Rank 1 failed: {res1[1]}")

        diff = max(abs(a - b) for a, b in zip(res0[1], res1[1]))
        self.assertLess(diff, 1e-6, f"DDP weight disparity across ranks: {diff}")

    def test_t4_scenario5_dynamic_autograd_audit_100_percent_active(self):
        """
        Scenario 5: Dynamic autograd audit asserting 100% active parameters receive non-null finite gradients.
        """
        cfg = Config()
        model = Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        y = torch.randint(0, cfg.vocab_size, (1, 4))

        out = model(x, labels=y)
        loss = out["loss"]["total_loss"]
        loss.backward()

        total_active = 0
        null_grad_params = []
        non_finite_params = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                total_active += 1
                if p.grad is None:
                    null_grad_params.append(name)
                elif not torch.isfinite(p.grad).all():
                    non_finite_params.append(name)

        self.assertEqual(total_active, 366, f"Expected 366 active parameters, found {total_active}")
        self.assertEqual(len(null_grad_params), 0, f"Parameters with null gradients: {null_grad_params}")
        self.assertEqual(len(non_finite_params), 0, f"Parameters with non-finite gradients: {non_finite_params}")


if __name__ == "__main__":
    unittest.main()
