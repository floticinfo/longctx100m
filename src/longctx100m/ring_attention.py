"""Exact causal context-parallel ring attention using PyTorch distributed.

Each rank owns a disjoint Q/K/V token shard. K/V shards circulate once around
a rank ring; each query shard attends every causally prior token. Online softmax
combines block contributions without gathering the global KV cache.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _online_update(m, l, acc, q, k, v, *, causal_mask=None):
    """Online log-sum-exp update for one KV block (fp32 accumulation)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask, float("-inf"))
    block_m = scores.amax(dim=-1, keepdim=True)
    new_m = torch.maximum(m, block_m)
    alpha = torch.exp(m - new_m)
    # (-inf) - (-inf) occurs for a query with no valid key in a block.
    alpha = torch.nan_to_num(alpha, nan=0.0)
    p = torch.exp(scores - new_m)
    p = torch.nan_to_num(p, nan=0.0)
    new_l = alpha * l + p.sum(dim=-1, keepdim=True)
    new_acc = alpha * acc + torch.matmul(p, v.float())
    return new_m, new_l, new_acc


def _tiled_update(
    m, l, acc, q, k, v, *, q_start, k_start, causal, valid_length, block_k
):
    """Apply one complete rank KV shard with O(block_k) score workspace."""
    total = k.shape[-2]
    for offset in range(0, total, block_k):
        end = min(offset + block_k, total)
        q_pos = q_start + torch.arange(q.shape[-2], device=q.device)
        k_pos = k_start + offset + torch.arange(end - offset, device=q.device)
        mask = None
        if causal or valid_length is not None:
            invalid = torch.zeros(
                (q.shape[-2], end - offset), dtype=torch.bool, device=q.device
            )
            if causal:
                invalid |= k_pos[None, :] > q_pos[:, None]
            if valid_length is not None:
                invalid |= k_pos[None, :] >= valid_length
            mask = invalid[None, None, :, :]
        m, l, acc = _online_update(
            m, l, acc, q, k[..., offset:end, :], v[..., offset:end, :], causal_mask=mask
        )
    return m, l, acc


def _ring_causal_attention_forward(
    q,
    k,
    v,
    *,
    group=None,
    causal=True,
    block_q=64,
    block_k=128,
    valid_length=None,
    q_position_start=None,
    return_stats=False,
):
    """Exact dense attention with rank-sharded Q/K/V and bounded KV workspace.

    Inputs are padded equally sized sequence shards [B, local_tokens, H, D].
    K/V shards circulate around the process-group ring and each rank processes
    score tiles of at most [block_q, block_k]. Arithmetic remains exact dense
    O(N^2); the ring shards cache/communication but does not reduce FLOPs.
    `valid_length` masks the padding tail on the final global token shard.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [batch, tokens, heads, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v shard shapes must match")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")
    if q.device.type not in {"cuda", "cpu"}:
        raise ValueError("ring attention requires CPU or CUDA tensors")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if q.device.type == "cuda" and not str(dist.get_backend(group)).lower().endswith(
        "nccl"
    ):
        raise RuntimeError("CUDA ring attention requires an NCCL process group")
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world < 1:
        raise RuntimeError("invalid distributed group")

    batch, local_n, heads, dim = q.shape
    if local_n == 0:
        raise ValueError("local_tokens must be positive")
    if valid_length is not None and not 0 < valid_length <= world * local_n:
        raise ValueError("valid_length must be within the padded distributed sequence")
    qh = q.permute(0, 2, 1, 3).contiguous()
    kh = k.permute(0, 2, 1, 3).contiguous()
    vh = v.permute(0, 2, 1, 3).contiguous()
    m = torch.full((batch, heads, local_n, 1), float("-inf"), device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(
        (batch, heads, local_n, dim), dtype=torch.float32, device=q.device
    )

    q_positions = (
        rank * local_n if q_position_start is None else q_position_start
    ) + torch.arange(local_n, device=q.device)
    # Circulate each rank's KV shard exactly once per ring hop; update all local
    # query tiles while that shard is resident (communication independent of Nq blocks).
    current_k, current_v, owner = kh, vh, rank
    for hop in range(world):
        for qs in range(0, local_n, block_q):
            qe = min(qs + block_q, local_n)
            qm, ql = m[:, :, qs:qe], l[:, :, qs:qe]
            qa, qq = acc[:, :, qs:qe], qh[:, :, qs:qe]
            qm, ql, qa = _tiled_update(
                qm,
                ql,
                qa,
                qq,
                current_k,
                current_v,
                q_start=q_positions[qs],
                k_start=owner * local_n,
                causal=causal,
                valid_length=valid_length,
                block_k=block_k,
            )
            m[:, :, qs:qe], l[:, :, qs:qe], acc[:, :, qs:qe] = qm, ql, qa
        if hop + 1 < world:
            src, dst = (rank - 1) % world, (rank + 1) % world
            next_k, next_v = torch.empty_like(current_k), torch.empty_like(current_v)
            ops = [
                dist.P2POp(dist.irecv, next_k, src, group=group),
                dist.P2POp(dist.irecv, next_v, src, group=group),
                dist.P2POp(dist.isend, current_k.contiguous(), dst, group=group),
                dist.P2POp(dist.isend, current_v.contiguous(), dst, group=group),
            ]
            for work in dist.batch_isend_irecv(ops):
                work.wait()
            current_k, current_v, owner = next_k, next_v, (owner - 1) % world

    out = acc / l.clamp_min(torch.finfo(torch.float32).tiny)
    if return_stats:
        return out.permute(0, 2, 1, 3).contiguous().to(dtype=q.dtype), m, l
    out = out.permute(0, 2, 1, 3).contiguous().to(dtype=q.dtype)
    return out


def reference_causal_attention(q, k, v, *, causal=True, valid_length=None):
    """Single-process correctness oracle for [B, T, H, D] tensors."""
    qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    t = q.shape[1]
    mask = None
    if causal or valid_length is not None:
        mask = torch.zeros((t, t), device=q.device, dtype=torch.bool)
        if causal:
            mask |= torch.triu(
                torch.ones((t, t), device=q.device, dtype=torch.bool), diagonal=1
            )
        if valid_length is not None:
            mask[:, valid_length:] = True
        scores = scores.masked_fill(mask, float("-inf"))
    return (
        torch.matmul(torch.softmax(scores, dim=-1), vh).permute(0, 2, 1, 3).to(q.dtype)
    )


def ring_causal_attention(
    q,
    k,
    v,
    *,
    group=None,
    causal=True,
    block_q=64,
    block_k=128,
    valid_length=None,
    q_position_start=None,
):
    """Exact ring attention with custom activation-recomputing backward."""
    return _RingCausalAttention.apply(
        q, k, v, group, causal, block_q, block_k, valid_length, q_position_start
    )


class _RingCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, group, causal, block_q, block_k, valid_length, q_position_start
    ):
        ctx.group, ctx.causal, ctx.block_q, ctx.block_k, ctx.valid_length = (
            group,
            causal,
            block_q,
            block_k,
            valid_length,
        )
        ctx.q_position_start = q_position_start
        with torch.no_grad():
            out, m, l = _ring_causal_attention_forward(
                q,
                k,
                v,
                group=group,
                causal=causal,
                block_q=block_q,
                block_k=block_k,
                valid_length=valid_length,
                q_position_start=q_position_start,
                return_stats=True,
            )
        ctx.save_for_backward(q, k, v, out, m, l)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, m, l = ctx.saved_tensors
        if ctx.q_position_start is not None and dist.get_world_size(ctx.group) == 1:
            dq, dk, dv = _local_decode_backward(
                q,
                k,
                v,
                out,
                m,
                l,
                grad_out,
                causal=ctx.causal,
                q_position_start=ctx.q_position_start,
                valid_length=ctx.valid_length,
            )
            return dq, dk, dv, None, None, None, None, None, None
        dq, dk, dv = _ring_attention_backward(
            q,
            k,
            v,
            out,
            m,
            l,
            grad_out,
            group=ctx.group,
            causal=ctx.causal,
            block_q=ctx.block_q,
            block_k=ctx.block_k,
            valid_length=ctx.valid_length,
            q_position_start=ctx.q_position_start,
        )
        return dq, dk, dv, None, None, None, None, None, None


def _local_decode_backward(
    q, k, v, out, m, l, grad_out, *, causal, q_position_start, valid_length
):
    """Single-rank dynamic-position decode tail gradient (not used for context parallel shards)."""
    _, t, _, d = q.shape
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    go = grad_out.permute(0, 2, 1, 3).float()
    oh = out.permute(0, 2, 1, 3).float()
    scores = qh @ kh.transpose(-2, -1) * (d**-0.5)
    # K/V context begins at absolute position zero and query rows can be a decode-tail shard.
    qpos = q_position_start + torch.arange(t, device=q.device)
    kpos = torch.arange(t, device=q.device)
    invalid = torch.zeros((t, t), dtype=torch.bool, device=q.device)
    if causal:
        invalid |= kpos[None, :] > qpos[:, None]
    if valid_length is not None:
        invalid |= kpos[None, :] >= valid_length
    prob = torch.softmax(scores.masked_fill(invalid[None, None], float("-inf")), dim=-1)
    delta = (go * oh).sum(-1, keepdim=True)
    ds = prob * (go @ vh.transpose(-2, -1) - delta)
    dq = (ds @ kh) * (d**-0.5)
    dk = ds.transpose(-2, -1) @ qh * (d**-0.5)
    dv = prob.transpose(-2, -1) @ go
    return (
        dq.permute(0, 2, 1, 3).to(q.dtype),
        dk.permute(0, 2, 1, 3).to(k.dtype),
        dv.permute(0, 2, 1, 3).to(v.dtype),
    )


def _ring_attention_backward(
    q,
    k,
    v,
    out,
    m,
    l,
    grad_out,
    *,
    group,
    causal,
    block_q,
    block_k,
    valid_length,
    q_position_start,
):
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    _, n, _, d = q.shape
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    go = grad_out.permute(0, 2, 1, 3).float()
    oh = out.permute(0, 2, 1, 3).float()
    dq = torch.zeros_like(qh)
    dk = torch.zeros_like(kh)
    dv = torch.zeros_like(vh)
    scale = d**-0.5
    qpos = (rank * n if q_position_start is None else q_position_start) + torch.arange(
        n, device=q.device
    )
    m = m.float()
    l = l.float()
    # Pass 1: keep local KV fixed; circulate Q, dO, and O so each KV owner
    # accumulates gradient contributions from every query shard.
    rq, rgo, ro, rm, rl = qh, go, oh, m, l
    query_owner = rank
    for hop in range(world):
        qpos = query_owner * n + torch.arange(n, device=q.device)
        # Caller-supplied offset describes a decode tail that is not part of the shard owner index.
        if q_position_start is not None and world == 1:
            qpos = q_position_start + torch.arange(n, device=q.device)
        for qs in range(0, n, block_q):
            qe = min(qs + block_q, n)
            qq = rq[:, :, qs:qe]
            goq = rgo[:, :, qs:qe]
            delta = (goq * ro[:, :, qs:qe]).sum(-1, keepdim=True)
            for ks in range(0, n, block_k):
                ke = min(ks + block_k, n)
                kb = kh[:, :, ks:ke]
                vb = vh[:, :, ks:ke]
                kp = rank * n + ks + torch.arange(ke - ks, device=q.device)
                scores = qq @ kb.transpose(-2, -1) * scale
                invalid = torch.zeros(
                    (qe - qs, ke - ks), dtype=torch.bool, device=q.device
                )
                if causal:
                    invalid |= kp[None, :] > qpos[qs:qe, None]
                if valid_length is not None:
                    invalid |= kp[None, :] >= valid_length
                scores = scores.masked_fill(invalid[None, None], float("-inf"))
                prob = torch.exp(scores - rm[:, :, qs:qe])
                prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0)
                prob = prob / rl[:, :, qs:qe].clamp_min(torch.finfo(torch.float32).tiny)
                ds = prob * (goq @ vb.transpose(-2, -1) - delta)
                dk[:, :, ks:ke] += ds.transpose(-2, -1) @ qq * scale
                dv[:, :, ks:ke] += prob.transpose(-2, -1) @ goq
        if hop + 1 < world:
            rq, rgo, ro, rm, rl = _rotate_quintuple(
                rq, rgo, ro, rm, rl, rank, world, group
            )
            query_owner = (query_owner - 1) % world
    # Pass 2: circulate K/V so each query owner forms its local dQ.
    qpos = rank * n + torch.arange(n, device=q.device)
    current_k, current_v, owner = kh, vh, rank
    for hop in range(world):
        kpos = owner * n + torch.arange(n, device=q.device)
        for qs in range(0, n, block_q):
            qe = min(qs + block_q, n)
            qq = qh[:, :, qs:qe]
            goq = go[:, :, qs:qe]
            delta = (goq * oh[:, :, qs:qe]).sum(-1, keepdim=True)
            for ks in range(0, n, block_k):
                ke = min(ks + block_k, n)
                kb = current_k[:, :, ks:ke]
                vb = current_v[:, :, ks:ke]
                kp = kpos[ks:ke]
                scores = qq @ kb.transpose(-2, -1) * scale
                invalid = torch.zeros(
                    (qe - qs, ke - ks), dtype=torch.bool, device=q.device
                )
                if causal:
                    invalid |= kp[None, :] > qpos[qs:qe, None]
                if valid_length is not None:
                    invalid |= kp[None, :] >= valid_length
                scores = scores.masked_fill(invalid[None, None], float("-inf"))
                prob = torch.exp(scores - m[:, :, qs:qe])
                prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0)
                prob = prob / l[:, :, qs:qe].clamp_min(torch.finfo(torch.float32).tiny)
                ds = prob * (goq @ vb.transpose(-2, -1) - delta)
                dq[:, :, qs:qe] += ds @ kb * scale
        if hop + 1 < world:
            current_k, current_v = _rotate_pair(
                current_k, current_v, rank, world, group
            )
            owner = (owner - 1) % world
    return (
        dq.permute(0, 2, 1, 3).to(q.dtype),
        dk.permute(0, 2, 1, 3).to(k.dtype),
        dv.permute(0, 2, 1, 3).to(v.dtype),
    )


def _rotate_quintuple(q, go, out, m, l, rank, world, group):
    src, dst = (rank - 1) % world, (rank + 1) % world
    tensors = (q, go, out, m, l)
    buffers = [torch.empty_like(x).contiguous() for x in tensors]
    ops = [dist.P2POp(dist.irecv, buffers[i], src, group=group) for i in range(5)] + [
        dist.P2POp(dist.isend, x.contiguous(), dst, group=group) for x in tensors
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return tuple(buffers)


def _rotate_pair(k, v, rank, world, group):
    src, dst = (rank - 1) % world, (rank + 1) % world
    nk, nv = torch.empty_like(k).contiguous(), torch.empty_like(v).contiguous()
    ops = [
        dist.P2POp(dist.irecv, nk, src, group=group),
        dist.P2POp(dist.irecv, nv, src, group=group),
        dist.P2POp(dist.isend, k.contiguous(), dst, group=group),
        dist.P2POp(dist.isend, v.contiguous(), dst, group=group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return nk, nv
