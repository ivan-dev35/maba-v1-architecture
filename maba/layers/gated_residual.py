import torch
import torch.nn as nn

class GatedRes(nn.Module):
    def __init__(self, dim: int = 640, bias_init: float = 2.0):
        super().__init__()
        self.g_res = nn.Parameter(torch.full((dim,), bias_init, dtype=torch.float32))

    def forward(self, res: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.g_res).type_as(res)
        return g * res + (1.0 - g) * sub

GatedResidual = GatedRes
