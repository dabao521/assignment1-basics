from building_blocks.sgd_optimizer import SGDOptimizer
import torch
from torch import nn

def run_optimization(
    lr: float,
    num_iterations: int = 10,
) -> None:
    print(f"run optimization with lr={lr} and {num_iterations} iterations...")
    weights = nn.Parameter(5 * torch.randn(10, 10))
    opt = SGDOptimizer([weights], lr=lr)

    for t in range(num_iterations):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()



def main() -> None:
    run_optimization(lr=1)
    run_optimization(lr=1e1)
    run_optimization(lr=1e2)
    run_optimization(lr=1e3)

if __name__ == "__main__":
    main()
