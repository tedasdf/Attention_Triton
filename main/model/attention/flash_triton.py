import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import AttentionConfig
from model.kernels.flash2_kernel import (
    flash_attn2_triton,
)  # rename to your actual kernel file


class FlashTritonAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model

        self.causal = getattr(cfg, "causal", True)
        self.attn_dropout_p = getattr(cfg, "dropout", 0.0)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_drop = nn.Dropout(self.attn_dropout_p)
        self.resid_drop = nn.Dropout(self.attn_dropout_p)

        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
            persistent=False,
        )

    def _reshape_qkv(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = (
            self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        )
        q, k, v = qkv.unbind(dim=3)  # each: [B, H, T, D]
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        # q, k, v: [B, H, T, D]
        B, H, T, D = q.shape

        att = torch.matmul(q, k.transpose(-1, -2)) * self.sm_scale

        if causal:
            mask = self.tril[:T, :T]
            att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)  # [B, H, T, D]
        return y

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and x.dtype == torch.float16
            and self.head_dim in (16, 32, 64, 128)
        )

    def forward(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        B, T, C = x.size()
        use_causal = self.causal if causal is None else causal

        q, k, v = self._reshape_qkv(x)

        # Triton path
        # Use fallback when:
        # - not CUDA / not fp16 / unsupported head dim
        # - attention dropout is active during training
        if self._can_use_triton(x) and not (
            self.training and self.attn_dropout_p > 0.0
        ):
            y = flash_attn2_triton(
                q,
                k,
                v,
                causal=use_causal,
                sm_scale=self.sm_scale,
            )
        else:
            y = self._standard_attention(q, k, v, causal=use_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
