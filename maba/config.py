from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    vocab_size: int = 32768
    d_emb: int = 128
    dim: int = 640
    n_layers: int = 20
    n_passes: int = 2
    layer_types: List[int] = field(default_factory=lambda: [0, 0, 0, 1] * 5)
    n_heads: int = 10
    d_head: int = 64
    n_kv_heads: int = 2
    kernel_size: int = 4
    d_ffn: int = 1728
    gate_bias: float = 2.0
    rope_theta: float = 500000.0
    max_len: int = 4096
    mtp_k: int = 2
    mtp_weight: float = 0.3
    eps: float = 1e-6
    init_std: float = 0.02

    @property
    def d_model(self) -> int: return self.dim
    @property
    def num_layers(self) -> int: return self.n_layers
    @property
    def passes_per_block(self) -> int: return self.n_passes
    @property
    def num_heads(self) -> int: return self.n_heads
    @property
    def num_kv_heads(self) -> int: return self.n_kv_heads
    @property
    def conv_kernel_size(self) -> int: return self.kernel_size
    @property
    def gated_res_bias_init(self) -> float: return self.gate_bias
    @property
    def max_position_embeddings(self) -> int: return self.max_len
    @property
    def mtp_loss_weight(self) -> float: return self.mtp_weight
    @property
    def rms_norm_eps(self) -> float: return self.eps
    @property
    def initializer_range(self) -> float: return self.init_std

    @classmethod
    def from_preset(cls, scale: str = "100M") -> "Config":
        s = scale.upper().strip()
        if s in ("100M", "101M", "BASE"):
            return cls(
                vocab_size=32768,
                d_emb=128,
                dim=640,
                n_layers=20,
                n_passes=2,
                layer_types=[0, 0, 0, 1] * 5,
                n_heads=10,
                d_head=64,
                n_kv_heads=2,
                kernel_size=4,
                d_ffn=1728,
                gate_bias=2.0,
                rope_theta=500000.0,
                max_len=4096,
                mtp_k=2,
                mtp_weight=0.3,
            )
        elif s in ("1B", "1.0B"):
            return cls(
                vocab_size=64256,
                d_emb=256,
                dim=2048,
                n_layers=20,
                n_passes=2,
                layer_types=[0, 0, 0, 1] * 5,
                n_heads=16,
                d_head=128,
                n_kv_heads=4,
                kernel_size=4,
                d_ffn=5504,
                gate_bias=2.0,
                rope_theta=500000.0,
                max_len=8192,
                mtp_k=2,
                mtp_weight=0.3,
            )
        elif s in ("3B", "2.8B", "3.0B"):
            return cls(
                vocab_size=64256,
                d_emb=384,
                dim=2816,
                n_layers=32,
                n_passes=2,
                layer_types=[0, 0, 0, 1] * 8,
                n_heads=22,
                d_head=128,
                n_kv_heads=4,
                kernel_size=4,
                d_ffn=7488,
                gate_bias=2.0,
                rope_theta=500000.0,
                max_len=16384,
                mtp_k=2,
                mtp_weight=0.3,
            )
        elif s in ("7B", "7.1B", "7.0B"):
            return cls(
                vocab_size=64256,
                d_emb=512,
                dim=4096,
                n_layers=36,
                n_passes=2,
                layer_types=[0, 0, 0, 1] * 9,
                n_heads=32,
                d_head=128,
                n_kv_heads=8,
                kernel_size=4,
                d_ffn=11008,
                gate_bias=2.0,
                rope_theta=500000.0,
                max_len=32768,
                mtp_k=2,
                mtp_weight=0.3,
            )
        elif s in ("30B", "29B", "AGENTIC"):
            return cls(
                vocab_size=64256,
                d_emb=768,
                dim=6656,
                n_layers=52,
                n_passes=2,
                layer_types=[0, 0, 0, 1] * 13,
                n_heads=52,
                d_head=128,
                n_kv_heads=4,
                kernel_size=4,
                d_ffn=19968,
                gate_bias=2.0,
                rope_theta=500000.0,
                max_len=131072,
                mtp_k=2,
                mtp_weight=0.3,
            )
        else:
            raise ValueError(f"Unknown scale preset: {scale}. Available presets: 100M, 1B, 3B, 7B, 30B")

    get_scale = from_preset

    def compute_param_count(self) -> dict:
        w_emb = self.vocab_size * self.d_emb
        w_proj_in = self.d_emb * self.dim
        w_proj_out = self.dim * self.d_emb
        emb_total = w_emb + w_proj_in + w_proj_out

        gdn2_one = (
            4 * self.dim * self.dim
            + 3 * self.dim * self.d_ffn
            + 3 * self.dim * self.kernel_size
            + 3 * self.dim * self.n_heads
            + 6 * self.dim
        )
        gqa_one = (
            self.dim * self.dim
            + 2 * self.dim * (self.n_kv_heads * self.d_head)
            + self.dim * self.dim
            + 3 * self.dim * self.d_ffn
            + 8 * self.dim
        )

        n_gqa = sum(1 for t in self.layer_types if t == 1)
        n_gdn = len(self.layer_types) - n_gqa

        gdn_total = n_gdn * gdn2_one
        gqa_total = n_gqa * gqa_one
        core_total = gdn_total + gqa_total

        final_norm = self.dim
        mtp_head = self.dim * (self.dim + self.d_emb) + self.dim

        total = emb_total + core_total + final_norm + mtp_head
        return {
            "w_emb": w_emb,
            "w_proj_in": w_proj_in,
            "w_proj_out": w_proj_out,
            "embedding_total": emb_total,
            "gdn2_blocks": gdn_total,
            "gqa_blocks": gqa_total,
            "core_total": core_total,
            "final_norm": final_norm,
            "mtp_head": mtp_head,
            "total": total,
            "vocab_tax_pct": round(emb_total / total * 100, 2),
            "core_pct": round(core_total / total * 100, 2),
            "n_gdn_layers": n_gdn,
            "n_gqa_layers": n_gqa,
            "effective_depth": self.n_layers * self.n_passes,
        }

MabaConfig = Config
