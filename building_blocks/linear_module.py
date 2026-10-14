from torch import nn
import torch

class LinearModule(nn.Module):
    weight: nn.Parameter
    """
    Performs a linear transformation.
    """

    def __init__(
        self,
        in_featurers: int,
        out_features: int,
        device: torch.device | None  = None,
        dtype: torch.dtype | None = None):
        """
        Construct a linear transformation module.

        Args:
            in_features: final dimension of the input.
            out_features: final dimension of the putput.
            device: Device to store the parameters on.
            dtype: Data type of the parameters

        Returns:
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            out_features,
            in_featurers,
            device=device,
            dtype=dtype
        ))
        
        std: float = (2.0 / (in_featurers + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        x = x @ self.weight.T
        return x

