import torch
from jaxtyping import Bool, Float, Int
from einops import einsum
import math
from torch import Tensor
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np
import os
import typing

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    apply softmax to the i-th dimesnion of the input tensor.
    The output tensor should have the same shape as the input tensor,
    but i-th dimension will now hve a normalized probability districution.
    """
    x_max: torch.Tensor = x.amax(dim=i, keepdim=True)
    x_norm: torch.Tensor = x - x_max
    x_exp: torch.Tensor = torch.exp(x_norm)
    x_sum: torch.Tensor = x_exp.sum(dim=i, keepdim=True)
    return x_exp / x_sum

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    d_k: int = Q.shape[-1]
    Q_K: Float[torch.Tensor, " ... queries keys"] = einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    )
    Q_K_sqrt = Q_K / math.sqrt(d_k)

    # apply the mask if there is any
    # we add a -INF for those False element since e^INF = 0
    # print("shape 1:", Q_K_sqrt.shape)
    # print("shape 2:", mask.shape)
    if mask is not None:
        Q_K_sqrt = Q_K_sqrt.masked_fill(~mask, float("-inf"))

    soft_max = softmax(Q_K_sqrt, -1)
    output = einsum(
        soft_max,
        V,
        " ... queries keys, ... keys values -> ... queries values"
    )
    return output

def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    # find vecab dimension's max value and keep that dimension
    # later this will be deducted from all logits to keep the
    # computation stable.
    logits_max = inputs.amax(dim=-1, keepdim=True)

    # normalize by deucting from max
    logits_norm = inputs - logits_max

    logits_exp = torch.exp(logits_norm)

    # sumarize all logits for the vocab dimension and keep its dimension
    log_sum = torch.log(logits_exp.sum(dim=-1, keepdim=True))

    # select targetd indexed logits
    # unsequzze is required to match the same input dimention
    target_shift = torch.gather(
        logits_norm,
        dim=-1,
        index=targets.unsqueeze(-1),
    )

    L = log_sum - target_shift
    return L.mean()

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (float(it) / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        lt = min_learning_rate
        cos = math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        lt = lt + 0.5 * (1 + cos) * (max_learning_rate - min_learning_rate)
        return lt

    return min_learning_rate

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
    ) -> None:
    valid_params = [
        p for p in parameters if p.grad is not None
    ]

    l2_norm = sum(
        p.grad.square().sum()
        for p in valid_params
    )
    l2_norm = torch.sqrt(l2_norm)
    if l2_norm < max_l2_norm:
        return

    scale = max_l2_norm / (l2_norm + eps)

    for p in valid_params:
        with torch.no_grad():
            p.grad.mul_(scale)

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    first randomly pick batch_size number of start indexes for each sentence.
    Random start is in the range of [0, n-contenxt_length). Exclusive here is for right shift

    Returns:
        a tuple of two matrixs (batch_size, context_length)
        first matrix:  the matrix sliced based on the sampled start points
        second matrix: first matrix + 1 right shift -> used as the target / expected next token
    """

    # create a random array
    n = len(dataset)
    # shape (batch_size,)
    start_indexes = np.random.randint(0, n-context_length, dtype=int, size=batch_size)

    # shape (context_length,)
    ordinals = np.arange(context_length)

    # shape (batch_size, context_length)
    # expansion trick
    # each row is [starti, starti+1, starti+2... starti+m-1]
    indexes = start_indexes[:,None] + ordinals[None,:]

    sampled = torch.as_tensor(
        dataset[indexes],
        dtype=torch.long,
        device=device,
    )
    # shfit by 1 to find next token for each input token
    shift = indexes + 1
    target = torch.as_tensor(
        dataset[shift],
        dtype=torch.long,
        device=device,
        )
    
    return (sampled, target)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    ) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)

def load_checkoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ) -> int:
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

