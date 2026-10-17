import torch
from torch import nn, Tensor
from jaxtyping import Float, Int, Bool
from building_blocks.linear_module import LinearModule
from building_blocks.rope import RoPEModule
from einops import rearrange
from building_blocks.common import scaled_dot_product_attention

class MultiheadSelfAttentionModule(nn.Module):
    Wq: LinearModule
    Wk: LinearModule
    Wv: LinearModule
    Wo: LinearModule
    rope: RoPEModule | None
    num_heads: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        shared_rope: RoPEModule | None = None,
    ):
        super().__init__()
        self.Wq = LinearModule(d_model, d_model, device, dtype)
        self.Wk = LinearModule(d_model, d_model, device, dtype)
        self.Wv = LinearModule(d_model, d_model, device, dtype)
        self.Wo = LinearModule(d_model, d_model, device, dtype)

        self.rope = None
        if theta is not None:
            # setup rope
            assert max_seq_len is not None
            assert shared_rope is None
            self.rope = RoPEModule(theta, d_model // num_heads, max_seq_len, device)
        elif shared_rope is not None:
            self.rope = shared_rope

        self.num_heads = num_heads

    def forward(
        self,
        x: Float[Tensor, " seq_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        seq_len: int = x.shape[-2]
        # 1. compute the Q, K , V 
        Q: Float[Tensor, " seq_length d_model"] = self.Wq(x)
        K: Float[Tensor, " seq_length d_model"] = self.Wk(x)
        V: Float[Tensor, " seq_length d_model"] = self.Wv(x)

        # 2. rearrange the Q, K , V to have num_heads heads
        # move h before seqlen to make downstream matrix mul easier
        Q_heads: Float[Tensor, " h seq_length dh"] = rearrange(
            Q,
            " ... seqlen (h dh) -> ... h seqlen dh",
            h=self.num_heads,
        )
        K_heads: Float[Tensor, " h seq_length dh"] = rearrange(
            K,
            " ... seqlen (h dh) -> ... h seqlen dh",
            h=self.num_heads,
        )
        V_heads: Float[Tensor, " h seq_length dh"] = rearrange(
            V,
            " ... seqlen (h dh) -> ... h seqlen dh",
            h=self.num_heads,
        )

        # 3. apply rope Q and K only if thea set before set
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            K_heads = self.rope(K_heads, token_positions)
            Q_heads = self.rope(Q_heads, token_positions)

        # 4. crete a mask with shape [seq_len, seq_len] and mask lower diagonal part of it
        mask: Bool[Tensor, " seq_len seq_len"] = self.create_mask(seq_len, Q_heads.device)

        # 5 apply scaled_dot_product_attention
        # print("shape0: ", Q_heads.shape, K_heads.shape, self.num_heads, Q.shape)
        attention: Float[Tensor, " h seq_len dh"] = scaled_dot_product_attention(
            Q_heads,
            K_heads,
            V_heads,
            mask,
        )
        # 6. concat all heads
        attention = rearrange(
            attention,
            "... h seqlen dh -> ... seqlen (h dh)",
            h=self.num_heads,
        )
        # 7. apply Wo projection and return
        return self.Wo(attention)

    def create_mask(self, seq_len: int, device: torch.device) -> Bool[Tensor, " seq_len seq_len"]:
        # (seq_len,)
        x = torch.arange(seq_len, device=device)
        # (seq_len,)
        y = torch.arange(seq_len, device=device)
        # rearrange x and y
        # 
        # now x is (seq_len, 1)
        x = x[:,None]

        # now y is (1, seq_len)
        y = y[None,:]

        # with expansion both x and y are (seq_len, seq_len)
        return x >= y


