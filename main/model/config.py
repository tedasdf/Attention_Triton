from dataclasses import dataclass
from typing import Union


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 128
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-3
    weight_decay: float = 0.0
    evals_per_epoch: int = 3
    attn_type: str = "standard"
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


@dataclass
class AttentionConfig:
    attn_type: str = "standard"
    d_model: int = 512
    n_head: int = 8
    block_size: int = 128
    dropout: float = 0.1


@dataclass
class GQAConfig(AttentionConfig):
    attn_type: str = "gqa"
    n_kv_heads: int = 2  # Branching specific field


@dataclass
class MLPConfig:
    d_model: int = 512
    dropout: float = 0.1
    expansion_factor: int = 4


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    block_size: int
    dropout: float
    # attn can be either standard AttentionConfig or the branched GQAConfig
    attn: Union[AttentionConfig, GQAConfig]
    mlp: MLPConfig

    @classmethod
    def from_flat(cls, h: Hyperparameters):
        # The Branching Logic
        if h.attn_type == "gqa":
            attn_cfg = GQAConfig(
                d_model=h.d_model,
                n_head=h.n_head,
                n_kv_heads=h.n_kv_heads,
                block_size=h.block_size,
                dropout=h.dropout,
            )
        else:
            attn_cfg = AttentionConfig(
                attn_type="standard",
                d_model=h.d_model,
                n_head=h.n_head,
                block_size=h.block_size,
                dropout=h.dropout,
            )

        return cls(
            vocab_size=h.vocab_size,
            n_layer=h.n_layer,
            d_model=h.d_model,
            block_size=h.block_size,
            dropout=h.dropout,
            attn=attn_cfg,
            mlp=MLPConfig(h.d_model, h.dropout),
        )
