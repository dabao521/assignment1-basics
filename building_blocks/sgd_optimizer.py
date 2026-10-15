from collections.abc import Callable, Iterable
import torch
import math
from typing import Optional

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr=1e-3,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)


    def step(
        self,
        closure: Optional[Callable] = None, # noqa: UP045
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # get state associated with p.
                t = state.get("t", 0) # Get iteration number from the sate, or 0.
                grad = p.grad.data # get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # update weight tensor in-place
                state["t"] = t+1
        return loss
