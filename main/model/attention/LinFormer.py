import torch.nn as nn
import torch.nn.functional as F
import math


class LinFormerAttn(nn.Module):  # Inherit or ensure CausalSelfAttention matches
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.k_proj = cfg.linformer_k  # The reduced dimension, e.g., 128 or 256

        # Projection matrices E and F (mapping sequence length T -> k)
        # Note: This usually requires a fixed maximum sequence length
        self.E = nn.Linear(cfg.block_size, self.k_proj)
        self.F = nn.Linear(cfg.block_size, self.k_proj)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, T, C = x.size()

        # 1. Get Q, K, V
        # Output shape: (B, n_head, T, head_dim)
        qkv = (
            self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Project K and V from (T) to (k_proj)
        # We transpose to project the sequence dimension T
        k = self.E(k.transpose(-2, -1))  # Shape: (B, n_head, head_dim, k_proj)
        v = self.F(v.transpose(-2, -1))  # Shape: (B, n_head, head_dim, k_proj)

        # Transpose back: (B, n_head, k_proj, head_dim)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # 3. Linear Attention
        # Q @ K^T -> (T, head_dim) @ (head_dim, k_proj) = (T, k_proj)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # NOTE: Standard tril masking won't work here because
        # the last dimension is size 'k_proj', not 'T'.

        att = F.softmax(att, dim=-1)

        # 4. Multiply by V
        # (T, k_proj) @ (k_proj, head_dim) = (T, head_dim)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
