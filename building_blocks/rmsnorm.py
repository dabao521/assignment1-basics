import torch
from torch import nn
from einops import reduce

class RMSNormModule(nn.Module):
    eps: float
    d_model: int
    g: nn.Parameter

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        proces an input tensor of shape (batch_size, sequence_length, d_model) and return
        a tensor of the same shape.
        """

        # upcast to F32 to prevent overflow when you square the input
        in_dtype: torch.dtype = x.dtype
        x = x.to(torch.float32)

        # get the sqaure of the x
        x_square: torch.Tensor = torch.square(x)

        # calculate the mean matrics
        # we have to keep the last dimension as 1 so that later matrix
        # element-wise multiplication can expand.
        x_mean: torch.Tensor = reduce(
            x_square,
            "... hidden -> ... 1",
            "mean",
        )

        x_mean: torch.Tensor = x_mean + self.eps

        x_square_root: torch.Tensor = torch.sqrt(x_mean)

        output: torch.Tensor = (x / x_square_root) * self.g
        return output.to(in_dtype)


