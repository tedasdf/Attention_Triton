import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        # 1. Generate the matrix once
        theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        m = torch.arange(max_seq_len).float()
        idx_theta = torch.einsum("n,d->nd", m, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # 2. Store them as buffers (Fixed forever)
        self.register_buffer("cos_cached", idx_theta2.cos()[None, None, :, :])
        self.register_buffer("sin_cached", idx_theta2.sin()[None, None, :, :])

    def forward(self, x):
        # Just slice what you need from the fixed cache
        seq_len = x.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return (x * cos) + (self._neg_half(x) * sin)
