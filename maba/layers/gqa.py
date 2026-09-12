import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rms_norm import RMSNorm
from .rope import apply_rope

class GQA(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        d_head: int = 64,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, dim, bias=False)

        self.q_norm = RMSNorm(d_head, eps=eps)
        self.k_norm = RMSNorm(d_head, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin)

        if kv is not None:
            pk, pv = kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        k_att = k.repeat_interleave(self.n_rep, dim=1)
        v_att = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.d_head)
        if mask is not None:
            attn = F.scaled_dot_product_attention(q, k_att, v_att, attn_mask=mask, scale=scale)
        elif L > 1:
            if kv is None:
                attn = F.scaled_dot_product_attention(q, k_att, v_att, is_causal=True, scale=scale)
            else:
                k_len = k_att.shape[2]
                c_mask = torch.ones(L, k_len, device=q.device, dtype=torch.bool).tril(diagonal=k_len - L)
                attn = F.scaled_dot_product_attention(q, k_att, v_att, attn_mask=c_mask, scale=scale)
        else:
            attn = F.scaled_dot_product_attention(q, k_att, v_att, is_causal=False, scale=scale)

        out = attn.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.o_proj(out), new_kv

GroupedQueryAttention = GQA
