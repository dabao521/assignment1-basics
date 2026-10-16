from collections.abc import Callable, Iterable
import torch
import math
from torch import nn
from typing import Optional

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        betas: tuple[float],
        eps: float,
        weight_decay: float,
    ):
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable] = None, # noqa: UP045
    ):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                # initialize state only once
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                m = state["m"]
                v = state["v"]
                g = p.grad
                t = state["t"] + 1

                # Bias-corrected learning rate
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # update moments
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g.square()

                # update paramter
                # work around torch doesn't allow updating the paramter
                with torch.no_grad():
                    p.mul_(1 - lr * weight_decay)
                    p.addcdiv_(m, v.sqrt() + eps, value=-lr_t)

                state["m"]=m
                state["v"]=v
                state["t"]=t

        return loss
