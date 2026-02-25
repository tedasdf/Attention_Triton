import torch
import torch.nn as nn


class GroupQueryAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.num_queries_kv = self.n_head // self.n_kv_head

        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, cfg.d_m)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x):
        raise ValueError


if __name__ == "__main__":
    raise ValueError
