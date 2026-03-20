import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from model.config import AttentionConfig
from model.RoPE import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, is_RoPE: bool):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0

        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.is_RoPE = is_RoPE
        if self.is_RoPE:
            assert self.head_dim % 2 == 0, "head_dim must be even for RoPE."
            self.rope = RotaryEmbedding(self.head_dim)

        self.register_buffer("mask", self.build_mask(cfg.block_size), persistent=False)

    def build_mask(self, block_size: int) -> torch.Tensor:
        return torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        qkv = (
            self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(0, 3, 2, 1, 4)
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, H, T, Dh]

        if self.is_RoPE:
            cos, sin = self.rope.get_cos_sin(T, x.device, q.dtype)
            q = self.rope.apply_rotary(q, cos, sin)
            k = self.rope.apply_rotary(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
