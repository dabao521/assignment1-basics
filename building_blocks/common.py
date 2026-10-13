import torch

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
