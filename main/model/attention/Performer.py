import torch
import math

from .standard import CausalSelfAttention
from model.config import PerformerAttentionConfig


class PerFormerAttention(CausalSelfAttention):
    def __init__(self, cfg: PerformerAttentionConfig):
        super().__init__(cfg)

        self.nb_features = cfg.nb_features
        # We need to generate the Orthogonal Random Features (W)
        # These are projected against our head_dim
        self.register_buffer(
            "W", self._generate_ortho_features(self.head_dim, cfg.nb_features)
        )

    def _generate_ortho_features(self, d, m):
        nb_full_blocks = m // d
        blocks = []
        for _ in range(nb_full_blocks):
            q, _ = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q.t())

        if m % d != 0:
            q, _ = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q.t()[: m % d])

        W = torch.cat(blocks, dim=0)
        # Gaussian length normalization
        multiplier = torch.randn(m, d).norm(dim=1, keepdim=True)
        return W * multiplier

    def feature_map(self, x):
        """The FAVOR+ Positive Statistical Feature Map"""
        # x: [B, H, T, D]
        # W: [M, D]
        # We project D-dim head to M-dim features
        projection = torch.einsum("bhtd, md -> bhtm", x, self.W)

        # h(x) part of the theorem: exp(-|x|^2 / 2)
        # We add a small constant for numerical stability
        x_norm = torch.sum(x**2, dim=-1, keepdim=True) / 2.0

        # The positive feature map: exp(W^T x - |x|^2 / 2) / sqrt(M)
        phi = torch.exp(projection - x_norm) / math.sqrt(self.nb_features)
        return phi

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        # 1. Get Q, K, V (Inherited from parent)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(2)  # Each is [B, H, T, D]

        # 2. Transform Q and K to kernels
        q_prime = self.feature_map(q)  # [B, H, T, M]
        k_prime = self.feature_map(k)  # [B, H, T, M]

        # 3. Causal Linear Attention (The FAVOR+ trick)
        # Instead of Q(K^T V), we use prefix sums (cumsum) to maintain O(N)
        # K'V has shape [B, H, T, M, D]
        kv = torch.einsum("bhtm, bhtd -> bhtmd", k_prime, v)
        kv_cumsum = torch.cumsum(kv, dim=2)  # [B, H, T, M, D]

        # Numerator: q_prime * cumsum(k_prime * v)
        num = torch.einsum("bhtm, bhtmd -> bhtd", q_prime, kv_cumsum)

        # Denominator: q_prime * cumsum(k_prime)
        den_cumsum = torch.cumsum(k_prime, dim=2)  # [B, H, T, M]
        den = torch.einsum("bhtm, bhtm -> bht", q_prime, den_cumsum)
        den = den.unsqueeze(-1) + 1e-6  # Avoid div by zero

        y = num / den  # [B, H, T, D]

        # 4. Final projection (Inherited logic)
        y = y.transpose(1, 2).contiguous().view(B, T, self.intermediate_dim)
        return self.resid_drop(self.proj(y))
