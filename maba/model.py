import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from .config import Config
from .layers.rms_norm import RMSNorm
from .layers.embeddings import EmbHead
from .layers.rope import RotaryEmbedding
from .layers.mtp import MTPHead
from .layers.transformer_block import Block

class Model(nn.Module):
    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.config = cfg or Config()
        c = self.config

        self.embeddings = EmbHead(c.vocab_size, c.d_emb, c.dim)
        self.rotary_emb = RotaryEmbedding(c.d_head, c.max_len, c.rope_theta)
        self.layers = nn.ModuleList([
            Block(
                dim=c.dim,
                d_ffn=c.d_ffn,
                n_heads=c.n_heads,
                n_kv_heads=c.n_kv_heads,
                d_head=c.d_head,
                k_size=c.kernel_size,
                is_gqa=(c.layer_types[i] == 1),
                n_passes=c.n_passes,
                gate_bias=c.gate_bias,
                eps=c.eps
            )
            for i in range(c.n_layers)
        ])
        self.final_norm = RMSNorm(c.dim, eps=c.eps)
        self.mtp_head = MTPHead(c.dim, c.d_emb)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        std = self.config.init_std
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, 0.0, std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            "w_emb": self.embeddings.w_emb.weight.numel(),
            "w_proj_in": self.embeddings.w_proj_in.weight.numel(),
            "w_proj_out": self.embeddings.w_proj_out.weight.numel(),
            "gdn2_blocks": 0,
            "gqa_blocks": 0,
            "final_norm": self.final_norm.weight.numel(),
            "mtp_head": sum(p.numel() for p in self.mtp_head.parameters()),
            "total": 0
        }
        for layer in self.layers:
            n = sum(p.numel() for p in layer.parameters())
            if layer.is_gqa:
                counts["gqa_blocks"] += n
            else:
                counts["gdn2_blocks"] += n
        counts["embedding_total"] = counts["w_emb"] + counts["w_proj_in"] + counts["w_proj_out"]
        counts["core_total"] = counts["gdn2_blocks"] + counts["gqa_blocks"]
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    count_params = count_parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[List[List[Dict[str, Any]]]] = None,
        return_states: bool = False,
        start_pos: Optional[int] = None
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        dev = input_ids.device

        if start_pos is None and states is not None:
            for s in states:
                if s:
                    for ps in s:
                        if ps and ps.get("kv_cache") is not None:
                            start_pos = ps["kv_cache"][0].shape[2]
                            break
                    if start_pos is not None:
                        break
        pos = start_pos or 0

        h = self.embeddings.forward_in(input_ids)
        cos, sin = self.rotary_emb(h, L, pos=pos)

        mask = None
        if L > 1 and states is None:
            mask = torch.full((L, L), float("-inf"), device=dev)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        new_states = [] if return_states else None
        for i, layer in enumerate(self.layers):
            st = states[i] if states is not None else None
            h, updated_st = layer(h, cos=cos, sin=sin, mask=mask, block_states=st)
            if return_states:
                new_states.append(updated_st)

        normed_h = self.final_norm(h)
        logits = self.embeddings.forward_out(normed_h)

        mtp_logits = None
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            if L > 2:
                next_toks = labels[..., 1:].clamp(min=0)
                f_emb = self.embeddings.factor_emb(next_toks)
                mtp_logits = self.mtp_head(normed_h[..., :-1, :], f_emb, self.embeddings)
                shift_mtp_logits = mtp_logits[..., :-1, :].contiguous()
                shift_mtp_labels = labels[..., 2:].contiguous().clone()
                shift_mtp_labels[labels[..., 1:-1] == -100] = -100
                mtp_loss = F.cross_entropy(
                    shift_mtp_logits.view(-1, self.config.vocab_size),
                    shift_mtp_labels.view(-1),
                    ignore_index=-100
                )
            else:
                mtp_loss = torch.tensor(0.0, device=dev)

            total_loss = main_loss + self.config.mtp_weight * mtp_loss
            loss = {
                "total_loss": total_loss,
                "main_loss": main_loss,
                "mtp_loss": mtp_loss
            }

        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "hidden_states": normed_h,
            "loss": loss,
            "states": new_states
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        use_cache: bool = True,
        **kwargs
    ) -> torch.Tensor:
        self.eval()
        _, prompt_len = input_ids.shape
        gen = input_ids.clone()
        temp = max(temperature, 1e-5)

        if use_cache:
            out = self.forward(input_ids, return_states=True)
            states = out["states"]
            lg = out["logits"][:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = float("-inf")
            tok = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
            gen = torch.cat([gen, tok], dim=1)

            for step in range(1, max_new_tokens):
                out = self.forward(tok, states=states, return_states=True, start_pos=prompt_len + step - 1)
                states = out["states"]
                lg = out["logits"][:, -1, :] / temp
                if top_k > 0:
                    v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                    lg[lg < v[:, [-1]]] = float("-inf")
                tok = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
                gen = torch.cat([gen, tok], dim=1)
        else:
            for _ in range(max_new_tokens):
                out = self.forward(gen)
                lg = out["logits"][:, -1, :] / temp
                if top_k > 0:
                    v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                    lg[lg < v[:, [-1]]] = float("-inf")
                tok = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
                gen = torch.cat([gen, tok], dim=1)
        return gen

MabaLM = Model
MabaModel = Model
