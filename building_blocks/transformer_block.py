import torch
from torch import nn, Tensor
from jaxtyping import Float
from building_blocks.multihead_attenion import MultiheadSelfAttentionModule
from building_blocks.rmsnorm import RMSNormModule
from building_blocks.swiglu import SwiGLUModule

class TransformerBlock(nn.Module):
    # sub-layer 1: multihead self attention
    multi_head_self_attention: MultiheadSelfAttentionModule
    rmsnorm1: RMSNormModule

    # sub-layer 2: SwiGLU feed-forward netowrk
    swiglu: SwiGLUModule
    rmsnorm2: RMSNormModule

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: dimensionality of the transformer block
            d_ff: dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        # walk through the shapes
        # in sublayer1:
        #  input: (... batch seq_len d_model)
        #  rms1 output: (... batch seq_len d_model)
        #  MHSA output: (... batch seq_len d_model)
        #
        # in sublayer2:
        #  rms2 output: (... batch seq_len d_model)
        #  swiGlu output: ( ... batch seq_len d_model)
        self.multi_head_self_attention = MultiheadSelfAttentionModule(
            d_model,
            num_heads,
            device,
            dtype,
            theta,
            max_seq_len
        )
        self.rmsnorm1 = RMSNormModule(d_model, device=device, dtype=dtype)
        
        self.rmsnorm2 = RMSNormModule(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLUModule(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: Float[Tensor, " batch seq_len d_model"],
    ) -> Float[Tensor, " batch seq_len d_model"]:
        # sub layer 1
        norm1: Float[Tensor, " batch seq_len d_model"] = self.rmsnorm1(x)
        mhsa: Float[Tensor, " batch seq_len d_model"] = self.multi_head_self_attention(norm1)

        output: Float[Tensor, " batch seq_len d_model"] = x + mhsa

        # sub layer 2
        norm2: Float[Tensor, " batch seq_len d_model"] = self.rmsnorm2(output)
        swiglu: Float[Tensor, " batch seq_len d_model"] = self.swiglu(norm2)

        output = output + swiglu

        return output
