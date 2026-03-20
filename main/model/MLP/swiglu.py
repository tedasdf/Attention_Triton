import torch
import torch.nn as nn

from model.config import MLPConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        hidden_dim = 256 * ((int(8 * cfg.d_model / 3) + 256 - 1) // 256)
        self.w1 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.w_out = nn.Linear(hidden_dim, cfg.d_model, bias=False)

    def forward(self, x):
        x = torch.nn.functional.silu(self.w1(x)) * self.w2(x)
        x = self.w_out(x)
        x = self.dropout(x)
        return x
