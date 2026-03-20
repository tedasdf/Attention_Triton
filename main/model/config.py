from dataclasses import dataclass


@dataclass
class Hyperparameters:
    log_file: str = "./logs/mainrun.log"
    data_dir: str = "./artifacts/datasets/hn_v1"
    output_dir: str = "./artifacts/runs/train"

    vocab_size: int = 50432
    lr: float = 6e-3
    betas: tuple[float, float] = (0.99, 0.95)

    weight_decay: float = 0.0
    warmup_step: int = 3000
    evals_per_epoch: int = 3
    accumulation_steps: int = 2

    attn_type: str = "standard"
    block_size: int = 128
    batch_size: int = 128
    n_layer: int = 6
    n_head: int = 16
    d_model: int = 512
    dropout: float = 0.1

    mlp_type: str = "standard"

    is_RoPE: bool = False

    # nb_features: int = 256
    # window_size: int = 32
    # d_compression: int = 128
    # n_kv_heads: int = 16
    # linformer_k: int = 32
    # causal: bool = True


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
class AttentionConfig:
    attn_type: str
    d_model: int
    n_head: int
    block_size: int
    dropout: float


# @dataclass
# class SlidingWindowAttentionConfig(AttentionConfig):
#     window_size: int

# @dataclass
# class PerformerAttentionConfig(BaseAttentionConfig):
#     nb_features: int = 256


# @dataclass
# class MultiLatentHeadAttnConfig(BaseAttentionConfig):
#     d_compression: int = 128


# @dataclass
# class GQAConfig(BaseAttentionConfig):
#     n_kv_heads: int = 16


# @dataclass
# class LinFormerConfig(BaseAttentionConfig):
#     linformer_k: int = 32


@dataclass
class MLPConfig:
    mlp_type: str
    d_model: int
    dropout: float


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    block_size: int
    dropout: int
    is_RoPE: bool
    # Nested configs
    attn: AttentionConfig
    mlp: MLPConfig

    # Factory method to create from a flat dict (like your Hyperparameters)
    @classmethod
    def from_flat(cls, h: Hyperparameters):
        if h.attn_type == "standard" or h.attn_type == "flash":
            attn_cfg = StandardAttentionConfig(
                h.attn_type,
                h.d_model,
                h.n_head,
                h.block_size,
                h.dropout,
            )
        # elif h.attn_type == "sliding":
        #     attn_cfg = SlidingWindowAttentionConfig(
        #         h.attn_type,
        #         h.d_model,
        #         h.n_head,
        #         h.block_size,
        #         h.dropout,
        #         window_size=h.block_size // 4,
        #     )
        # elif h.attn_type == "performer":
        #     attn_cfg = PerformerAttentionConfig(
        #         h.attn_type,
        #         h.d_model,
        #         h.n_head,
        #         h.block_size,
        #         h.dropout,
        #         nb_features=h.nb_features,
        #     )

        # elif h.attn_type == "gpa":
        #     attn_cfg = GQAConfig(
        #         h.attn_type, h.d_model, h.n_head, h.block_size, h.dropout, h.n_kv_heads
        #     )

        # elif h.attn_type == "linformer":
        #     attn_cfg = LinFormerConfig(
        #         h.attn_type, h.d_model, h.n_head, h.block_size, h.dropout, h.linformer_k
        #     )

        # elif h.attn_type == "mla":
        #     attn_cfg = MultiLatentHeadAttnConfig(
        #         h.attn_type,
        #         h.d_model,
        #         h.n_head,
        #         h.block_size,
        #         h.dropout,
        #         h.d_compression,
        #     )
        else:
            raise ValueError(f"Unknown attention type: {h.attn_type}")

        if h.mlp_type == "standard" or h.mlp_type == "swiglu":
            mlp_cfg = MLPConfig(h.mlp_type, h.d_model, h.dropout)
        else:
            raise ValueError(f"Unknown mlp type: {h.mlp_type}")

        return cls(
            h.vocab_size,
            h.n_layer,
            h.d_model,
            h.block_size,
            h.dropout,
            h.is_RoPE,
            attn_cfg,
            mlp_cfg,
        )
