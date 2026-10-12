import torch
from torch import nn

class Embedding(nn.Module):
    embedding: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.embedding = nn.Parameter(torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        ))

        nn.init.trunc_normal_(
            self.embedding,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        The forard method should select the embedding vector for each tokenID by
        indexing into an embedding matrix of shape (vocab_size, embedding_dim or d_model)
        using a torch.LongTensor of token IDs with shape (batch_size, sequence_length)
        """
        # simply lookup the embedding matrix from token id
        return self.embedding[token_ids]
