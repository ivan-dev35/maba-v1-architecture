import torch
from torch.optim.optimizer import Optimizer

def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    norm = torch.linalg.vector_norm(X)
    X = X / (norm + eps)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.type_as(G)

zeropower_via_newtonschulz5 = newton_schulz5

class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-7
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(grad)

                if p.ndim == 2:
                    upd = newton_schulz5(buf, steps=steps, eps=eps)
                    rms = (torch.linalg.vector_norm(p.float()) / (p.numel() ** 0.5)).clamp(min=1e-3)
                    upd = upd * rms
                else:
                    upd = buf

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(upd, alpha=-lr)

        return loss
