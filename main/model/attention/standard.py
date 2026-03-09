import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from model.config import AttentionConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer("mask", self.build_mask(cfg.block_size))

    def build_mask(self, block_size: int) -> torch.Tensor:
        return torch.tril(torch.ones(block_size, block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        qkv = (
            self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(0, 3, 2, 1, 4)
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
