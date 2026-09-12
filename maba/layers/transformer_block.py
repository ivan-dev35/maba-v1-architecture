import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from .rms_norm import RMSNorm
from .gated_residual import GatedRes
from .swiglu import SwiGLU
from .gdn2 import GDN2
from .gqa import GQA

class Block(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        d_ffn: int = 1728,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        d_head: int = 64,
        k_size: int = 4,
        is_gqa: bool = False,
        n_passes: int = 2,
        gate_bias: float = 2.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.is_gqa = is_gqa
        self.n_passes = n_passes

        if is_gqa:
            self.mixer = GQA(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, d_head=d_head, eps=eps)
        else:
            self.mixer = GDN2(dim=dim, n_heads=n_heads, d_head=d_head, k_size=k_size)

        self.input_norm = RMSNorm(dim, eps=eps)
        self.post_attn_norm = RMSNorm(dim, eps=eps)
        self.attn_gate = GatedRes(dim, bias_init=gate_bias)

        self.ffn = SwiGLU(dim=dim, d_ffn=d_ffn)
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.post_ffn_norm = RMSNorm(dim, eps=eps)
        self.ffn_gate = GatedRes(dim, bias_init=gate_bias)

    def _pass(
        self,
        h: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        st: Optional[Dict[str, Any]] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        nst = {} if (return_states or st is not None) else None
        h_norm = self.input_norm(h)

        if self.is_gqa:
            kv = st.get("kv_cache") if st else None
            m_out, n_kv = self.mixer(h_norm, cos, sin, kv=kv, mask=mask)
            if nst is not None:
                nst["kv_cache"] = n_kv
        else:
            sr = st.get("recurrent_state") if st else None
            cq = st.get("conv_state_q") if st else None
            ck = st.get("conv_state_k") if st else None
            cv = st.get("conv_state_v") if st else None
            m_out, n_sr, (ncq, nck, ncv) = self.mixer(h_norm, s_rec=sr, cs_q=cq, cs_k=ck, cs_v=cv)
            if nst is not None:
                nst["recurrent_state"] = n_sr
                nst["conv_state_q"] = ncq
                nst["conv_state_k"] = nck
                nst["conv_state_v"] = ncv

        m_out = self.post_attn_norm(m_out)
        h = self.attn_gate(h, m_out)

        f_out = self.ffn(self.ffn_norm(h))
        f_out = self.post_ffn_norm(f_out)
        h = self.ffn_gate(h, f_out)
        return h, nst

    def forward(
        self,
        h: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        block_states: Optional[List[Dict[str, Any]]] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        new_states = [] if (return_states or block_states is not None) else None
        for p in range(self.n_passes):
            st = block_states[p] if block_states is not None else None
            h, updated_st = self._pass(h, cos=cos, sin=sin, mask=mask, st=st, return_states=return_states)
            if new_states is not None:
                new_states.append(updated_st)
        return h, new_states

MabaBlock = Block
