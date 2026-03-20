import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, dim/2]
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # [T, dim]
        sin = freqs.sin().repeat_interleave(2, dim=-1)  # [T, dim]
        return cos[None, None, :, :].to(dtype), sin[None, None, :, :].to(dtype)

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (x * cos) + (rotate_half(x) * sin)


# class RotaryPositionalEmbeddings(nn.Module):
#     def __init__(self, d: int, max_seq_len: int = 2048, base: int = 10_000):
#         super().__init__()
#         # 1. Generate the matrix once
#         theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
#         m = torch.arange(max_seq_len).float()
#         idx_theta = torch.einsum("n,d->nd", m, theta)
#         idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

#         # 2. Store them as buffers (Fixed forever)
#         self.register_buffer("cos_cached", idx_theta2.cos()[None, None, :, :])
#         self.register_buffer("sin_cached", idx_theta2.sin()[None, None, :, :])

#     def forward(self, x):
#         # Just slice what you need from the fixed cache
#         seq_len = x.shape[2]
#         cos = self.cos_cached[:, :, :seq_len, :]
#         sin = self.sin_cached[:, :, :seq_len, :]

#         return (x * cos) + (self._neg_half(x) * sin)

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(RotaryPositionalEmbedding, self).__init__()

#         # Create a rotation matrix.
#         self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cuda"))
#         for i in range(d_model):
#             for j in range(d_model):
#                 self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

#         # Create a positional embedding matrix.
#         self.positional_embedding = torch.zeros(max_seq_len, d_model, device=torch.device("cuda"))
#         for i in range(max_seq_len):
#             for j in range(d_model):
#                 self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

#     def forward(self, x):
#         """
#         Args:
#             x: A tensor of shape (batch_size, seq_len, d_model).

#         Returns:
#             A tensor of shape (batch_size, seq_len, d_model).
#         """

#         # Add the positional embedding to the input tensor.
#         x += self.positional_embedding

#         # Apply the rotation matrix to the input tensor.
#         x = torch.matmul(x, self.rotation_matrix)

#         return x
