import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def _kernel_fused_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    batch,
    num_heads,
    seq_len,
    dim,
    qk_scale,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # the program id basically whatever the grid is
    pid = tl.program_id(0)  # seq_len // BLOCK_SIZE_M
    pid_bh = tl.program_id(1)  #

    # all operating component
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    # logical index
    b = pid_bh // num_heads
    h = pid_bh % num_heads
    start_m = pid * BLOCK_SIZE_M

    # all index across D model , SIZE_M and SIZE_N
    d_idx = tl.arange(0, BLOCK_DMODEL)
    m_idx = tl.arange(0, BLOCK_SIZE_M)
    n_idx = tl.arange(0, BLOCK_SIZE_N)

    q_pos = start_m + m_idx

    # Fixated on one Q
    offset_qd = d_idx * stride_qd
    offset_qs = q_pos * stride_qs
    offset_qh = h * stride_qh
    offset_qb = b * stride_qb

    offsets_q = offset_qb + offset_qh + offset_qs[:, None] + offset_qd[None, :]
    mask_q = (q_pos[:, None] < seq_len) & (d_idx[None, :] < dim)
    q_val = tl.load(q_ptr + offsets_q, mask=mask_q, other=0.0)

    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        k_pos = start_n + n_idx
        # find the K
        offset_kd = d_idx * stride_kd
        offset_ks = k_pos * stride_ks
        offset_kh = h * stride_kh
        offset_kb = b * stride_kb

        offsets_k = offset_kb + offset_kh + offset_ks[None, :] + offset_kd[:, None]
        mask_k = (k_pos[None, :] < seq_len) & (d_idx[:, None] < dim)

        k_val = tl.load(k_ptr + offsets_k, mask=mask_k, other=0.0)

        # find the V
        offset_vd = d_idx * stride_vd
        offset_vs = k_pos * stride_vs
        offset_vh = h * stride_vh
        offset_vb = b * stride_vb

        offsets_v = offset_vb + offset_vh + offset_vs[:, None] + offset_vd[None, :]
        mask_v = (k_pos[:, None] < seq_len) & (d_idx[None, :] < dim)

        v_val = tl.load(v_ptr + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        # find attention score
        S_ij = tl.dot(q_val, k_val) * qk_scale

        key_mask = k_pos[None, :] < seq_len

        if IS_CAUSAL:
            key_mask = key_mask & (q_pos[:, None] >= k_pos[None, :])

        S_ij = tl.where(key_mask, S_ij, -float("inf"))

        # skibidi max across row
        m_hat_ij = tl.max(S_ij, axis=1)

        # find the exponential temr
        P_ij = tl.exp(S_ij - m_hat_ij[:, None])

        # what the sigm row sum
        l_hat_i = tl.sum(P_ij, axis=1)

        # find maximum
        m_new = tl.maximum(m_i, m_hat_ij)

        scale_old = tl.exp(m_i - m_new).to(tl.float32)
        scale_new = tl.exp(m_hat_ij - m_new).to(tl.float32)

        l_new_i = scale_old * l_i + scale_new * l_hat_i
        acc = scale_old[:, None] * acc
        acc = acc + scale_new[:, None] * tl.dot(P_ij, v_val, out_dtype=tl.float32)
        # update
        m_i = m_new
        l_i = l_new_i

    acc /= l_i[:, None]

    # find the V
    offset_od = d_idx * stride_od
    offset_os = (start_m + m_idx) * stride_os
    offset_oh = h * stride_oh
    offset_ob = b * stride_ob

    offsets_o = offset_ob + offset_oh + offset_os[:, None] + offset_od[None, :]
    mask_o = ((start_m + m_idx)[:, None] < seq_len) & (d_idx[None, :] < dim)

    tl.store(o_ptr + offsets_o, acc, mask=mask_o)


def triton_fused_attention(Q, K, V, is_causal):
    assert Q.ndim == 4, f"Q must be 4D [B, H, S, D], got shape {Q.shape}"
    assert K.ndim == 4, f"K must be 4D [B, H, S, D], got shape {K.shape}"
    assert V.ndim == 4, f"V must be 4D [B, H, S, D], got shape {V.shape}"

    assert Q.shape == K.shape == V.shape, (
        f"Q, K, V must have the same shape, got "
        f"Q={Q.shape}, K={K.shape}, V={V.shape}"
    )

    assert Q.is_cuda, "Q must be on CUDA"
    assert K.is_cuda, "K must be on CUDA"
    assert V.is_cuda, "V must be on CUDA"

    assert Q.device == K.device == V.device, (
        f"Q, K, V must be on the same device, got "
        f"Q={Q.device}, K={K.device}, V={V.device}"
    )

    assert Q.dtype == K.dtype == V.dtype, (
        f"Q, K, V must have the same dtype, got "
        f"Q={Q.dtype}, K={K.dtype}, V={V.dtype}"
    )

    assert Q.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"Unsupported dtype {Q.dtype}. "
        "Expected one of: torch.float16, torch.bfloat16, torch.float32"
    )

    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"
    assert V.is_contiguous(), "V must be contiguous"

    batch, num_heads, seq_len, dim = Q.shape

    assert dim > 0, f"dim must be > 0, got {dim}"
    assert seq_len > 0, f"seq_len must be > 0, got {seq_len}"
    assert num_heads > 0, f"num_heads must be > 0, got {num_heads}"
    assert batch > 0, f"batch must be > 0, got {batch}"

    qk_scale = 1.0 / math.sqrt(dim)
    Output = torch.empty_like(Q)

    BLOCK_SIZE_M = 64
    grid = (
        triton.cdiv(seq_len, BLOCK_SIZE_M),
        num_heads * batch,
    )

    _kernel_fused_attention[grid](
        Q,
        K,
        V,
        Output,
        batch,
        num_heads,
        seq_len,
        dim,
        qk_scale,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        Output.stride(0),
        Output.stride(1),
        Output.stride(2),
        Output.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=64,
        BLOCK_DMODEL=max(dim, 16),
        IS_CAUSAL=int(is_causal),
        num_warps=4,
        num_stages=2,
    )

    return Output


def test_dense_attention(
    batch,
    seq_len,
    dim,
    num_heads=1,
    is_causal: bool = True,
    dtype=torch.float16,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    q = torch.randn((batch, num_heads, seq_len, dim), device=DEVICE, dtype=dtype)
    k = torch.randn((batch, num_heads, seq_len, dim), device=DEVICE, dtype=dtype)
    v = torch.randn((batch, num_heads, seq_len, dim), device=DEVICE, dtype=dtype)

    ref_out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    triton_out = triton_fused_attention(q, k, v, is_causal=is_causal)

    torch.testing.assert_close(
        triton_out,
        ref_out,
        rtol=rtol,
        atol=atol,
    )
    print("DONE YIPPY")


def benchmark():
    raise ValueError


if __name__ == "__main__":
    for dim in [4, 8, 16, 32, 64, 128]:
        try:
            test_dense_attention(2, 8, dim, 8, dtype=torch.float32)
            print(dim, "fp32 pass")
        except Exception as e:
            print(dim, "fp32 fail", e)

    for dim in [4, 8, 16, 32, 64, 128]:
        try:
            test_dense_attention(2, 8, dim, 8, dtype=torch.float16)
            print(dim, "fp16 pass")
        except Exception as e:
            print(dim, "fp16 fail", e)
