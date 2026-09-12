import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GDN2(nn.Module):
    def __init__(self, dim: int = 640, n_heads: int = 10, d_head: int = 64, k_size: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.k_size = k_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.conv_q = nn.Conv1d(dim, dim, k_size, groups=dim, bias=False, padding=0)
        self.conv_k = nn.Conv1d(dim, dim, k_size, groups=dim, bias=False, padding=0)
        self.conv_v = nn.Conv1d(dim, dim, k_size, groups=dim, bias=False, padding=0)

        self.gate_alpha = nn.Linear(dim, n_heads, bias=False)
        self.gate_erase = nn.Linear(dim, n_heads, bias=False)
        self.gate_write = nn.Linear(dim, n_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _conv(
        self, x: torch.Tensor, conv: nn.Conv1d, cs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        xt = x.transpose(1, 2)
        k = self.k_size
        if cs is not None:
            pad = torch.cat([cs, xt], dim=2)
        else:
            pad = F.pad(xt, (k - 1, 0))
        ncs = pad[:, :, -(k - 1):]
        out = F.silu(conv(pad)).transpose(1, 2)
        return out, ncs

    def forward(
        self,
        x: torch.Tensor,
        s_rec: Optional[torch.Tensor] = None,
        cs_q: Optional[torch.Tensor] = None,
        cs_k: Optional[torch.Tensor] = None,
        cs_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head

        q, ncs_q = self._conv(self.q_proj(x), self.conv_q, cs_q)
        k, ncs_k = self._conv(self.k_proj(x), self.conv_k, cs_k)
        v, ncs_v = self._conv(self.v_proj(x), self.conv_v, cs_v)

        alpha = torch.sigmoid(self.gate_alpha(x)).unsqueeze(-1).unsqueeze(-1)
        b = torch.sigmoid(self.gate_erase(x)).unsqueeze(-1)
        w = torch.sigmoid(self.gate_write(x)).unsqueeze(-1)

        q = q.view(B, L, H, d)
        k = k.view(B, L, H, d)
        v = v.view(B, L, H, d)

        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + 1e-6)
        e = (b * k).unsqueeze(-2)
        z = (w * v).unsqueeze(-2)
        kt = k.unsqueeze(-1)
        qt = q.unsqueeze(-2)

        S = s_rec.clone() if s_rec is not None else torch.zeros(B, H, d, d, dtype=x.dtype, device=x.device)

        if L == 1:
            S = alpha[:, 0] * S
            delta = z[:, 0] - torch.matmul(e[:, 0], S)
            S = S + torch.matmul(kt[:, 0], delta)
            out = torch.matmul(qt[:, 0], S).squeeze(-2).unsqueeze(1).reshape(B, 1, D)
            return self.o_proj(out), S, (ncs_q, ncs_k, ncs_v)

        outs = []
        for t in range(L):
            S = alpha[:, t] * S
            delta = z[:, t] - torch.matmul(e[:, t], S)
            S = S + torch.matmul(kt[:, t], delta)
            outs.append(torch.matmul(qt[:, t], S).squeeze(-2))

        out = torch.stack(outs, dim=1).reshape(B, L, D)
        return self.o_proj(out), S, (ncs_q, ncs_k, ncs_v)

GatedDeltaNet2 = GDN2
