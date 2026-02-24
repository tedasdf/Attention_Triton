import torch.nn as nn

from model.config import MLPConfig


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)
