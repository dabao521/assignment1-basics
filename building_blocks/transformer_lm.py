import torch 
from torch import nn, Tensor
from jaxtyping import Float
from building_blocks.transformer_block import TransformerBlock
from building_blocks.embedding import Embedding
from building_blocks.rmsnorm import RMSNormModule
from building_blocks.linear_module import LinearModule
from building_blocks.rope import RoPEModule

class TransformerLM(nn.Module):
    # all layers for a e2e transformer.
    input_embedding: Embedding
    blocks: nn.ModuleList
    rms_norm: RMSNormModule
    output_embedding: LinearModule

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # create a share ROPE module to be shared across layers
        rope = RoPEModule(theta, d_model // num_heads, context_length, device)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=None,
                max_seq_len=None,
                device=device,
                dtype=dtype,
                shared_rope=rope,
            )
            for _ in range(num_layers)
        ])

        self.input_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.rms_norm = RMSNormModule(d_model, device=device, dtype=dtype)
        self.output_embedding = LinearModule(d_model, vocab_size, device, dtype)
    
    def forward(
        self,
        x: Float[Tensor, " batch_size seq_len"],
    ) -> Float[Tensor, " batch_size seq_len vocab_size"]:
        x_embedded: Float[Tensor, " batch_size seq_len d_model"] = self.input_embedding(x)

        for block in self.blocks:
            x_embedded = block(x_embedded)

        x_embedded = self.rms_norm(x_embedded)

        output: Float[Tensor, " batch_szie seq_len vocab_size"] = self.output_embedding(x_embedded)
        return output

