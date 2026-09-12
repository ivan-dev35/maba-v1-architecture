import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Any, Tuple
from .muon import Muon

class HybridOpt:
    def __init__(
        self,
        model: nn.Module,
        lr_muon: float = 0.02,
        wd_muon: float = 0.01,
        lr_adamw: float = 1.5e-3,
        wd_adamw: float = 0.1,
        betas_adamw: Tuple[float, float] = (0.9, 0.95)
    ):
        self.model = model
        p_muon = []
        p_adamw = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "embeddings" in name:
                p_adamw.append(p)
            elif p.ndim == 2 and any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down")):
                p_muon.append(p)
            else:
                p_adamw.append(p)

        self.muon = Muon(p_muon, lr=lr_muon, weight_decay=wd_muon) if p_muon else None
        self.adamw = AdamW(p_adamw, lr=lr_adamw, betas=betas_adamw, weight_decay=wd_adamw) if p_adamw else None

        self.muon_param_count = sum(p.numel() for p in p_muon)
        self.adamw_param_count = sum(p.numel() for p in p_adamw)

    def zero_grad(self, set_to_none: bool = True):
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None, barrier: bool = True):
        loss = closure() if closure is not None else None

        is_xla = False
        for opt in (self.muon, self.adamw):
            if opt and opt.param_groups and opt.param_groups[0]["params"]:
                p = opt.param_groups[0]["params"][0]
                if p.device.type == "xla":
                    is_xla = True
                    break

        if is_xla:
            try:
                import torch_xla.core.xla_model as xm
                if self.muon:
                    xm.optimizer_step(self.muon, barrier=False)
                if self.adamw:
                    xm.optimizer_step(self.adamw, barrier=barrier)
                elif barrier:
                    xm.mark_step()
            except ImportError:
                if self.muon:
                    self.muon.step()
                if self.adamw:
                    self.adamw.step()
        else:
            if self.muon:
                self.muon.step()
            if self.adamw:
                self.adamw.step()
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon.state_dict() if self.muon else None,
            "adamw": self.adamw.state_dict() if self.adamw else None
        }

    def load_state_dict(self, d: Dict[str, Any]):
        if self.muon and d.get("muon"):
            self.muon.load_state_dict(d["muon"])
        if self.adamw and d.get("adamw"):
            self.adamw.load_state_dict(d["adamw"])

MabaOptimizer = HybridOpt
