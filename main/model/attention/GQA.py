import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .standard import CausalSelfAttention


@dataclass
class GQAConfig:
    n_kv_head: int


class GroupQueryAttn(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        if hasattr(self, "kv"):
            del self.kv
        self.n_kv_heads = cfg.n_kv_heads
        assert self.n_head % self.n_kv_heads == 0
        self.num_queries_per_kv = self.n_head // self.n_kv_heads

        self.kv_proj = nn.Linear(self.d_model, 2 * self.n_kv_heads * self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        # 1. Project Q (using parent's projection)
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Project KV (GQA style)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        k = kv[:, :, 0, :, :].transpose(1, 2)  # (B, nkv, T, hs)
        v = kv[:, :, 1, :, :].transpose(1, 2)  # (B, nkv, T, hs)

        # 3. Repeat KV heads to match Q heads
        k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
        v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)

        # 4. Attention (Scaled Dot Product)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)


if __name__ == "__main__":
    raise ValueError
