import torch
from torch import nn
from einops import rearrange, einsum

class RoPEModule(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # pre-compute a COS and SIN matrix with shape (... max_seq_len, d_k // 2)
        # d_k // 2 because each pair shares the same theta_i_k for token at position i
        #
        # 1. build a position vector starting with 0, 1, 2.. with shape (max_seq_len,)
        i_vec: torch.Tensor = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # 2. build a k vector starting with 0, 1... , with shape (d_k//2,)
        k_vec: torch.Tensor = torch.arange(d_k // 2, device=device, dtype=torch.float32) * 2 / d_k
        # 3. now reduce to the inverse vector 1 / theta^(2k//d_k) with shape (d_k//2,)
        k_inverse: torch.Tensor = 1 / torch.pow(theta, k_vec)
        # 4. to create (max_seq_len, dk//2) matrix
        # shape (max_seq_len, 1)
        i_matrix = rearrange(i_vec, "d -> d 1")
        # shape(1, d_k//2)
        k_matrix = rearrange(k_inverse, "d -> 1 d")
        theta_matrix = einsum(i_matrix, k_matrix, "seq one, one d -> seq d")

        COS: torch.Tensor = torch.cos(theta_matrix)
        SIN: torch.Tensor = torch.sin(theta_matrix)

        # store into the buffer so that later can be shared among layers
        # this also creates a instance member fields.
        self.register_buffer("COS", COS, persistent=False)
        self.register_buffer("SIN", SIN, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        ) -> torch.Tensor:
        """
        process an input tensor of shape (..., seq_len, d_k) and retrn a tensor
        of the same shape. token_positions are a tensor of shape (... seq_len) specifying
        the token positoins of x along the sequence dimension.
        """

        original_dtype = x.dtype
        x = x.to(torch.float32)

        # find the index cos and sin sub matrix
        C = self.COS[token_positions]
        S = self.SIN[token_positions]

        # slice x into two matrixes with odd and even positions 
        # x has shape(..., seq_len, d_k)
        # pairs has shape (... seq_len pairs 2)
        pairs: torch.Tensor = rearrange(
            x, "... seq_len (pairs two) -> ... seq_len pairs two", two=2
        )
        # slice even and odd into X and Y
        # shape (... seq_len d_k/2)
        X = pairs[...,0]
        Y = pairs[...,1]

        # X = [x0,x1,x2,x3...x_d_k//2]
        # Y = [y0,y1,y2,y3...y_d_k//2]
        # C and S has shape (seq_len, d_k // 2)
        # based on rotation formula:
        # token i, at position k , (xk,yk) is roated to
        # xk' = cos_i_k * xk - sin_i_k * yk
        # yk' = sin_i_k * xk + cos_i_k * yk
        X_prime = C * X - S * Y
        Y_prime = S * X + C * Y

        # now stack back X_prime and Y_prime to single matrix
        # by stackcing Y_prime at the last dimension.
        # shape (... seq_len pairs two)
        pairs_prime = torch.stack((X_prime, Y_prime), dim=-1)
        # then arrange it back to orignal shape (... seq_len d_k)
        x_prime = rearrange(
            pairs_prime, "... seq_len pairs two -> ... seq_len (pairs two)"
        )
        return x_prime.to(original_dtype)


