import torch
from jaxtyping import Bool, Float, Int
from einops import einsum
import math
from torch import Tensor

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

