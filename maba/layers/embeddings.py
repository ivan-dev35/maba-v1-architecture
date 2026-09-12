import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbHead(nn.Module):
    def __init__(self, vocab_size: int = 32768, d_emb: int = 128, dim: int = 640):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.dim = dim

        self.w_emb = nn.Embedding(vocab_size, d_emb)
        self.w_proj_in = nn.Linear(d_emb, dim, bias=False)
        self.w_proj_out = nn.Linear(dim, d_emb, bias=False)

    def factor_emb(self, idx: torch.Tensor) -> torch.Tensor:
        return self.w_emb(idx)

    def forward_in(self, idx: torch.Tensor) -> torch.Tensor:
        return self.w_proj_in(self.w_emb(idx))

    def forward_out(self, h: torch.Tensor) -> torch.Tensor:
        comp = self.w_proj_out(h)
        return F.linear(comp, self.w_emb.weight)

    get_factor_embedding = factor_emb
    forward_input = forward_in
    forward_output = forward_out

FactorizedEmbeddingHead = EmbHead
