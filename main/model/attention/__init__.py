# model/attention/__init__.py
from .standard import CausalSelfAttention

# from .GQA import GroupQueryAttn
from .Performer import PerFormerAttention

ATTENTION_REGISTRY = {
    "standard": CausalSelfAttention,
    "performer": PerFormerAttention,
    # "gpa": GroupQueryAttn,
    # "flash": FlashAttention,
    # "sliding": SlidingAttention
}


def get_attention(name, **kwargs):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {name}")
    return ATTENTION_REGISTRY[name](**kwargs)
