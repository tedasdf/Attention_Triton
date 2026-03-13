import math
from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


def _require_cuda_fp16_bhtd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    # take in q,k,v tensor to do valiation check
    if not (q.is_cuda and k.is_cuda and v.is_cuda):  # if they are cuda tensor
        raise ValueError("q, k, v must be CUDA tensors")
    if not (
        q.dtype == k.dtype == v.dtype == torch.float16
    ):  # the torch type is float16
        raise ValueError("This minimal implementation expects float16 q/k/v")
    if not (q.ndim == k.ndim == v.ndim == 4):  # tensor are 4 dimension
        raise ValueError("Expected q, k, v to have shape [B, H, T, D]")
    if not (q.shape == k.shape == v.shape):  # tensors have same shape
        raise ValueError("q, k, v must have identical shapes")
    if q.shape[-1] not in (16, 32, 64, 128):  # d shape is reaonsable value
        raise ValueError("Head dim D must be one of {16, 32, 64, 128}")


@triton.jit
def _flash2_fwd_kernel(
    Q,
    K,
    V,
    O,  # noqa : E741
    LSE,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program_id is the index of the current Triton program instance
    pid_m = tl.program_id(0)  # which query-row tile this program handles
    pid_bh = tl.program_id(1)  # batch * head

    b = pid_bh // H
    h = pid_bh % H
    # indicate this gpu will work on the (pid_m)th block, (b)th batch and (h)th head

    # indices for block m, n and d
    offs_m = (
        pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    )  # shape (BLOCK_M) pid_m * BLOCK_M is the starting idx of the mth block and (0 -> BLOCK_M )
    offs_n = tl.arange(0, BLOCK_N)  # shape (BLOCK_N)
    offs_d = tl.arange(0, D)  # shape (D)

    row_mask = offs_m < T  # constraining the row

    q_base = Q + b * stride_qb + h * stride_qh  # Q[b, h, :, :]
    k_base = K + b * stride_kb + h * stride_kh  # K[b, h, :, :]
    v_base = V + b * stride_vb + h * stride_vh  # V[b, h, :, :]
    o_base = O + b * stride_ob + h * stride_oh  # O[b, h, :, :]
    lse_base = LSE + b * stride_lb + h * stride_lh  # LSE[b, h, :]

    q_ptrs = (
        q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    )  # shape (BLOCK_M, D)
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)  # load Q_block

    m_i = tl.where(
        row_mask, -float("inf"), 0.0
    )  # shape: (BLOCK_M) make a list for maximum value of each row
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # shape: (BLOCK_M) softmax denominator
    acc = tl.zeros(
        (BLOCK_M, D), dtype=tl.float32
    )  # shape: (BLOCK_M, D) accumulation variable

    for start_n in range(0, T, BLOCK_N):  # across the column of K'
        cols = start_n + offs_n  # shape: (BLOCK_N) each start idx of block_N
        col_mask = cols < T  # constrain the number taking

        k_ptrs = (
            k_base + cols[:, None] * stride_kt + offs_d[None, :] * stride_kd
        )  # shape: (BLOCK_N, D)
        v_ptrs = (
            v_base + cols[:, None] * stride_vt + offs_d[None, :] * stride_vd
        )  # shape: (BLOCK_N, D)

        # grab the value
        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        # compute scaled QK^T logits and convert them to base-2 exponent form
        logits2 = tl.dot(q, tl.trans(k)) * SCALE_LOG2

        # check the valid idx for the value
        valid = row_mask[:, None] & col_mask[None, :]
        if CAUSAL:
            # the valid range change if it is causal
            valid = valid & (offs_m[:, None] >= cols[None, :])

        # mask invalid positions with a large negative value so their softmax contribution becomes ~0
        logits2 = tl.where(valid, logits2, -1.0e6)

        # find the max of the block
        block_max = tl.max(logits2, axis=1)
        # find the new max
        new_m = tl.where(row_mask, tl.maximum(m_i, block_max), 0.0)

        # use the new max to find the attention score
        p = tl.where(valid, tl.math.exp2(logits2 - new_m[:, None]), 0.0)
        # rescale previous running sums from old max m_i to new max new_m
        alpha = tl.math.exp2(m_i - new_m)

        # correction factor update
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # rescale previous accumulator to the new row max
        acc = acc * alpha[:, None]
        # add current tile's contribution to the unnormalized output
        acc = tl.dot(p.to(tl.float16), v, acc)

        # assign new max
        m_i = new_m

    l_safe = tl.where(row_mask, l_i, 1.0)
    out = acc / l_safe[:, None]
    # store row-wise log-sum-exp in base-2 form for backward
    lse = tl.where(row_mask, m_i + tl.math.log2(l_i), 0.0)

    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.float16), mask=row_mask[:, None])

    lse_ptrs = lse_base + offs_m * stride_lt
    tl.store(lse_ptrs, lse, mask=row_mask)


@triton.jit
def _flash2_bwd_delta_kernel(
    O,  # noqa : E741
    DO,
    DELTA,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)  # yea looking at the block m of output
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_mask = offs_m < T

    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh

    # delat is the output of o * do for the term calulating dsoftmax
    delta_base = DELTA + b * stride_db + h * stride_dh

    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod

    o = tl.load(o_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(delta_base + offs_m * stride_dt, delta, mask=row_mask)


@triton.jit
def _flash2_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)  # n block cause we are finding dV dK
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    col_mask = offs_n < T

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    do_base = DO + b * stride_dob + h * stride_doh
    lse_base = LSE + b * stride_lb + h * stride_lh
    delta_base = DELTA + b * stride_db + h * stride_dh
    dk_base = DK + b * stride_dkb + h * stride_dkh
    dv_base = DV + b * stride_dvb + h * stride_dvh

    k_ptrs = (
        k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    )  # shape: (BLOCK_N, D)
    v_ptrs = (
        v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    )  # shape: (BLOCK_N, D)

    k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for start_m in range(0, T, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        row_mask = offs_m < T

        q_ptrs = (
            q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        )  # shape: (BLOCK_M,d)
        do_ptrs = (
            do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
        )  # shape: (BLOCK_M, d)

        q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

        lse = tl.load(lse_base + offs_m * stride_lt, mask=row_mask, other=0.0)
        delta = tl.load(delta_base + offs_m * stride_dt, mask=row_mask, other=0.0)

        logits2_t = (
            tl.dot(k, tl.trans(q)) * SCALE_LOG2
        )  # recompute the logit to memroy : 2 x n x d instead of n^2

        valid = col_mask[:, None] & row_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])

        # reconstruct the probability tile P^T using the stored row-wise log-sum-exp
        p_t = tl.where(valid, tl.math.exp2(logits2_t - lse[None, :]), 0.0)

        # find dv as p v = O , # dV += P^T @ dO and # dP^T = V @ dO^T
        dv_acc += tl.dot(p_t.to(tl.float16), do)
        dp_t = tl.dot(v, tl.trans(do)).to(tl.float32)

        # dsoftmax dS^T = P^T * (dP^T - delta)
        ds_t = tl.where(valid, p_t * (dp_t - delta[None, :]), 0.0)
        # dK += dS^T @ Q
        dk_acc += tl.dot(ds_t.to(tl.float16), q)

    # normalise
    dk_acc *= SM_SCALE

    dk_ptrs = dk_base + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk_acc.to(tl.float16), mask=col_mask[:, None])
    tl.store(dv_ptrs, dv_acc.to(tl.float16), mask=col_mask[:, None])


@triton.jit
def _flash2_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_dqb,
    stride_dqh,
    stride_dqt,
    stride_dqd,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_mask = offs_m < T

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    do_base = DO + b * stride_dob + h * stride_doh
    lse_base = LSE + b * stride_lb + h * stride_lh
    delta_base = DELTA + b * stride_db + h * stride_dh
    dq_base = DQ + b * stride_dqb + h * stride_dqh

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

    lse = tl.load(lse_base + offs_m * stride_lt, mask=row_mask, other=0.0)
    delta = tl.load(delta_base + offs_m * stride_dt, mask=row_mask, other=0.0)

    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < T

        k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        # compute the logits again like i said memory thing
        logits2 = tl.dot(q, tl.trans(k)) * SCALE_LOG2

        valid = row_mask[:, None] & col_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.where(valid, tl.math.exp2(logits2 - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        # using k to find dq
        dq_acc += tl.dot(ds.to(tl.float16), k)

    dq_acc *= SM_SCALE

    dq_ptrs = dq_base + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(tl.float16), mask=row_mask[:, None])


class _FlashAttn2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        sm_scale: float,
    ) -> torch.Tensor:
        _require_cuda_fp16_bhtd(q, k, v)

        # turn tensors into contiguous
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        # take the tensor shape
        B, H, T, D = q.shape
        # make an empty tensor for output
        o = torch.empty_like(q)

        # log sum exp each row has one
        lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        # BLOCK_N: number of key/value rows per tile
        # BLOCK_M: number of query rows per tile
        block_m = 64 if D <= 64 else 128
        block_n = 64
        num_warps = 4 if D <= 64 else 8

        # launch one program per (query tile, batch-head pair)

        # grid[0] spans query-row tiles

        # grid[1] spans batch-head combinations
        grid = (triton.cdiv(T, block_m), B * H)

        _flash2_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale

        if not do.is_contiguous():
            do = do.contiguous()

        B, H, T, D = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        pre_block = 128
        # seq_len // pre_block, B*H
        pre_grid = (triton.cdiv(T, pre_block), B * H)
        _flash2_bwd_delta_kernel[pre_grid](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            D=D,
            BLOCK_M=pre_block,
            num_warps=4,
            num_stages=1,
        )

        block_m = 64 if D <= 64 else 128
        block_n = 64
        num_warps = 4 if D <= 64 else 8

        grid_kv = (triton.cdiv(T, block_n), B * H)
        _flash2_bwd_dkdv_kernel[grid_kv](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        grid_q = (triton.cdiv(T, block_m), B * H)
        _flash2_bwd_dq_kernel[grid_q](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        return dq, dk, dv, None, None


def flash_attn2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    _require_cuda_fp16_bhtd(q, k, v)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttn2Triton.apply(q, k, v, causal, sm_scale)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * sm_scale

    if causal:
        T = q.shape[-2]
        mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, -float("inf"))

    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    B, H, T, D = 2, 4, 512, 64

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = flash_attn2_triton(q, k, v, causal=True)
    out_ref = reference_attention(q_ref, k_ref, v_ref, causal=True)

    loss = out.float().square().mean()
    loss_ref = out_ref.float().square().mean()

    loss.backward()
    loss_ref.backward()

    print("forward max abs err:", (out - out_ref).abs().max().item())
    print("dq max abs err     :", (q.grad - q_ref.grad).abs().max().item())
    print("dk max abs err     :", (k.grad - k_ref.grad).abs().max().item())
    print("dv max abs err     :", (v.grad - v_ref.grad).abs().max().item())
