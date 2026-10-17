import torch
from torch import nn
from building_blocks.linear_module import LinearModule

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLUModule(nn.Module):
    w1_weight: LinearModule
    w2_weight: LinearModule
    w3_weight: LinearModule
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1_weight = LinearModule(d_model, d_ff, device=device)
        self.w2_weight = LinearModule(d_ff, d_model, device=device)
        self.w3_weight = LinearModule(d_model, d_ff, device=device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # shape (seq_length, d_ff)
        silu_part: torch.Tensor = silu(self.w1_weight(x))

        # shape (seq_legnth, d_ff)
        w3_part: torch.Tensor = self.w3_weight(x)

        # shape (seq_length, d_ff)
        dot_product: torch.Tensor = silu_part * w3_part

        # shape (seg_length, d_model)
        return self.w2_weight(dot_product)
