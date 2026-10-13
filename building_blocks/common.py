import torch
from jaxtyping import Bool, Float
from einops import einsum
import math

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
    if mask is not None:
        Q_K_sqrt = Q_K_sqrt.masked_fill(~mask, float("-inf"))

    soft_max = softmax(Q_K_sqrt, -1)
    output = einsum(
        soft_max,
        V,
        " ... queries keys, ... keys values -> ... queries values"
    )
    return output

