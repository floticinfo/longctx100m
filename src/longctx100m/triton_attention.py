"""Triton tiled causal attention kernel for single-device fallback/validation.

Install the `kernel` extra on a supported Linux/CUDA stack. Distributed
multi-device communication is implemented in ring_attention.py.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # optional dependency; import module remains safe
    triton = None
    tl = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _causal_attention(
        Q,
        K,
        V,
        O,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ot,
        stride_oh,
        stride_od,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        POS_OFFSET: tl.constexpr,
        CAUSAL_LIMIT: tl.constexpr,
        BQ: tl.constexpr,
        BK: tl.constexpr,
        BD: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_h = tl.program_id(1)
        qidx = pid_q * BQ + tl.arange(0, BQ)
        d = tl.arange(0, BD)
        kk = tl.arange(0, BK)
        qptr = (
            Q + qidx[:, None] * stride_qt + pid_h * stride_qh + d[None, :] * stride_qd
        )
        q = tl.load(qptr, (qidx[:, None] < T) & (d[None, :] < D), other=0)
        m = tl.full((BQ,), float("-inf"), tl.float32)
        l = tl.zeros((BQ,), tl.float32)
        acc = tl.zeros((BQ, BD), tl.float32)
        q_abs = qidx + POS_OFFSET
        for kstart in range(0, tl.minimum((pid_q + 1) * BQ, T), BK):
            kidx = kstart + kk
            k_abs = kidx + POS_OFFSET
            kp = (
                K
                + kidx[None, :] * stride_kt
                + pid_h * stride_kh
                + d[:, None] * stride_kd
            )
            vp = (
                V
                + kidx[:, None] * stride_vt
                + pid_h * stride_vh
                + d[None, :] * stride_vd
            )
            kval = tl.load(kp, (kidx[None, :] < T) & (d[:, None] < D), other=0)
            vval = tl.load(vp, (kidx[:, None] < T) & (d[None, :] < D), other=0)
            score = tl.dot(q, kval) * SCALE
            mask = (qidx[:, None] < T) & (k_abs[None, :] <= q_abs[:, None])
            if CAUSAL_LIMIT >= 0:
                mask = mask & (kidx[None, :] < CAUSAL_LIMIT)
            score = tl.where(mask, score, float("-inf"))
            block_m = tl.max(score, 1)
            new_m = tl.maximum(m, block_m)
            alpha = tl.exp(m - new_m)
            alpha = tl.where(m == float("-inf"), 0.0, alpha)
            prob = tl.exp(score - new_m[:, None])
            prob = tl.where(mask, prob, 0.0)
            l = alpha * l + tl.sum(prob, 1)
            acc = alpha[:, None] * acc + tl.dot(prob, vval)
            m = new_m
        out = acc / l[:, None]
        op = O + qidx[:, None] * stride_ot + pid_h * stride_oh + d[None, :] * stride_od
        tl.store(op, out, (qidx[:, None] < T) & (d[None, :] < D))


def tiled_causal_attention(
    q, k, v, *, block_q=64, block_k=64, position_offset=0, causal_limit=None
):
    """Triton kernel for contiguous [T,H,D] CUDA tensors. Returns same shape."""
    if triton is None:
        raise RuntimeError(
            "Triton kernel unavailable; install the 'kernel' extra on Linux/CUDA"
        ) from _IMPORT_ERROR
    import torch

    if (
        q.device.type != "cuda"
        or q.ndim != 3
        or q.shape != k.shape
        or q.shape != v.shape
    ):
        raise ValueError("q, k, v must be matching CUDA tensors [tokens, heads, dim]")
    t, h, d = q.shape
    out = torch.empty_like(q)
    bd = triton.next_power_of_2(d)
    _causal_attention[(triton.cdiv(t, block_q), h)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        t,
        h,
        d,
        d**-0.5,
        position_offset,
        -1 if causal_limit is None else causal_limit,
        block_q,
        block_k,
        bd,
        num_warps=4,
    )
    return out
