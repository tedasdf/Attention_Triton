import torch
from .standard import CausalSelfAttention


class SlidingWindowSelfAttention(CausalSelfAttention):
    def __init__(self, cfg):
        self.window_size = cfg.window_size
        super().__init__(cfg)

    def build_mask(self, block_size: int) -> torch.Tensor:
        full_causal = torch.tril(torch.ones(block_size, block_size))
        return torch.triu(full_causal, diagonal=-(self.window_size - 1))
