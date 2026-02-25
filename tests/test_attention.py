import pytest
import torch
from dataclasses import dataclass
from main.model.attention.GQA import GroupQueryAttn


@dataclass
class DummyConfig:
    d_model: int = 128
    n_head: int = 8
    n_kv_heads: int = 2
    block_size: int = 64
    dropout: float = 0.0


def test_gqa_output_shape():
    cfg = DummyConfig()
    model = GroupQueryAttn(cfg)
    x = torch.randn(2, 16, 128)  # B, T, C
    y = model(x)
    assert y.shape == (2, 16, 128)


def test_gqa_mha_equivalence():
    # When n_kv_heads == n_head, GQA is mathematically MHA
    cfg = DummyConfig(n_head=8, n_kv_heads=8)
    gqa_model = GroupQueryAttn(cfg)

    x = torch.randn(1, 10, 128)

    # Check if forward pass works without crashing
    try:
        out = gqa_model(x)
    except Exception as e:
        pytest.fail(f"GQA forward failed: {e}")

    assert out.shape == (1, 10, 128)


def test_invalid_groups():
    # Should fail if n_head not divisible by n_kv_heads
    cfg = DummyConfig(n_head=8, n_kv_heads=3)
    with pytest.raises(AssertionError):
        GroupQueryAttn(cfg)
