import torch
import torch.nn as nn
import math


class MultiLatentHeadAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.d_head = cfg.d_model // cfg.n_head
        self.d_c = cfg.d_c

        # --- KV compression ---
        self.W_DKV = nn.Linear(cfg.d_model, cfg.d_c)
        self.W_UK = nn.Linear(cfg.d_c, cfg.d_model)
        self.W_UV = nn.Linear(cfg.d_c, cfg.d_model)

        # --- Q compression ---
        self.W_DQ = nn.Linear(cfg.d_model, cfg.d_c)
        self.W_UQ = nn.Linear(cfg.d_c, cfg.d_model)

        # --- RoPE projections ---
        self.W_QR = nn.Linear(cfg.d_model, cfg.d_c)
        self.W_KR = nn.Linear(cfg.d_model, cfg.d_c)

        # --- KV cache for autoregressive inference ---
        self.register_buffer(
            "kv_cache_key", torch.zeros(1, self.n_head, cfg.block_size, self.d_head)
        )
        self.register_buffer(
            "kv_cache_value", torch.zeros(1, self.n_head, cfg.block_size, self.d_head)
        )
        self.kv_index = 0  # current position in cache

    # --- Simple rotary embedding for demonstration ---
    def RoPE(self, x):
        # x: (B, H, T, d_head)
        seq_len = x.size(2)
        dim = x.size(-1)
        freqs = torch.arange(0, dim, 2, device=x.device).float() / dim
        freqs = 10000 ** (-freqs)
        pos = torch.arange(seq_len, device=x.device).float()
        theta = pos[:, None] * freqs[None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x

    # --- reshape for multi-head ---
    def to_multi_head(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        return x

    def forward(self, x, use_cache=False):
        B, T, C = x.size()

        # --- KV Compression ---
        c_kv = self.W_DKV(x)  # (B, T, d_c)
        k_content = self.W_UK(c_kv)  # (B, T, d_model)
        v_content = self.W_UV(c_kv)  # (B, T, d_model)

        # --- Query Compression ---
        c_q = self.W_DQ(x)  # (B, T, d_c)
        q_content = self.W_UQ(c_q)  # (B, T, d_model)

        # --- RoPE projections ---
        q_rope = self.W_QR(c_q)  # (B, T, d_c)
        k_rope = self.W_KR(x)  # (B, T, d_c)

        # --- Reshape to multi-head ---
        q_content = self.to_multi_head(q_content)
        k_content = self.to_multi_head(k_content)
        v_content = self.to_multi_head(v_content)

        q_rope = self.to_multi_head(q_rope)
        k_rope = self.to_multi_head(k_rope)

        # --- Apply RoPE ---
        q_rope = self.RoPE(q_rope)
        k_rope = self.RoPE(k_rope)

        # --- Combine content + RoPE ---
        Q = torch.cat([q_content, q_rope], dim=-1)  # (B, H, T, 2*d_head)
        K = torch.cat([k_content, k_rope], dim=-1)

        # --- KV caching ---
        if use_cache:
            # Only store the last token(s)
            t_idx = self.kv_index
            T_step = Q.size(2)
            self.kv_cache_key[:, :, t_idx : t_idx + T_step, :] = K
            self.kv_cache_value[:, :, t_idx : t_idx + T_step, :] = v_content
            self.kv_index += T_step

            K = self.kv_cache_key[:, :, : self.kv_index, :]
            V = self.kv_cache_value[:, :, : self.kv_index, :]
        else:
            V = v_content

        # --- Compute attention ---
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)  # (B, H, T, d_head)

        # --- Merge heads back ---
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return out
