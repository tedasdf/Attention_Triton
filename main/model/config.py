from dataclasses import dataclass


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 128
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 16
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-3
    weight_decay: float = 0.0
    evals_per_epoch: int = 3
    attn_type: str = "performer"
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


@dataclass
class BaseAttentionConfig:
    attn_type: str
    d_model: int
    n_head: int
    block_size: int
    dropout: float


@dataclass
class StandardAttentionConfig(BaseAttentionConfig):
    pass


@dataclass
class PerformerAttentionConfig(BaseAttentionConfig):
    nb_features: int = 256


@dataclass
class AttentionConfig:
    attn_type: str
    d_model: int
    n_head: int
    block_size: int
    dropout: float


@dataclass
class MLPConfig:
    d_model: int
    dropout: float
    expansion_factor: int = 4


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    block_size: int
    dropout: int
    # Nested configs
    attn: AttentionConfig
    mlp: MLPConfig

    # Factory method to create from a flat dict (like your Hyperparameters)
    @classmethod
    def from_flat(cls, h: Hyperparameters):
        if h.attn_type == "standard":
            attn_cfg = StandardAttentionConfig(
                h.attn_type,
                h.d_model,
                h.n_head,
                h.block_size,
                h.dropout,
            )

        elif h.attn_type == "performer":
            attn_cfg = PerformerAttentionConfig(
                h.attn_type,
                h.d_model,
                h.n_head,
                h.block_size,
                h.dropout,
                nb_features=h.nb_features,
            )

        else:
            raise ValueError(f"Unknown attention type: {h.attn_type}")

        mlp_cfg = MLPConfig(h.d_model, h.dropout)
        return cls(
            h.vocab_size,
            h.n_layer,
            h.d_model,
            h.block_size,
            h.dropout,
            attn_cfg,
            mlp_cfg,
        )
