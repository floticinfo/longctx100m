//! Exact causal context-parallel ring attention, mirroring
//! `longctx100m.ring_attention`.
//!
//! Each rank owns a disjoint Q/K/V token shard. K/V shards circulate once
//! around a rank ring; each query shard attends every causally prior token.
//! Online softmax combines block contributions without gathering the global KV
//! cache. This slice implements the exact same arithmetic on CPU with an
//! in-process ring (world_size simulated by sharding); the P2P/NCCL transport
//! is isolated so a CUDA/NCCL backend can be dropped in later.

use candle_core::{DType, Result, Tensor};

/// Online log-sum-exp update for one KV block (fp32 accumulation), mirroring
/// `ring_attention._online_update`.
fn online_update(
    m: &Tensor,
    l: &Tensor,
    acc: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal_mask: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    // q: [B, H, Q, D], k/v: [B, H, K, D], m/l: [B, H, Q, 1], acc: [B, H, Q, D]
    let scale = (q.dim(3)? as f64).powf(-0.5);
    let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let scores = match causal_mask {
        Some(mask) => {
            // Python masked_fill(mask, -inf): true -> -inf, false -> unchanged.
            let neg_inf = f32::NEG_INFINITY;
            let inf_t = Tensor::full(neg_inf, scores.shape(), scores.device())?;
            mask.where_cond(&inf_t, &scores)?
        }
        None => scores,
    };
    let block_m = scores.max_keepdim(3)?;
    let new_m = m.broadcast_maximum(&block_m)?;
    // alpha = exp(m - new_m); when new_m == -inf (all scores masked), the
    // difference is nan in Python and replaced by 0. candle has no nan_to_num,
    // so we explicitly zero the -inf rows.
    let neg_inf = f32::NEG_INFINITY;
    let all_masked = new_m.eq(neg_inf)?;
    let alpha_raw = m.broadcast_sub(&new_m)?.exp()?;
    let alpha = all_masked.where_cond(&Tensor::zeros_like(&alpha_raw)?, &alpha_raw)?;
    // p = exp(scores - new_m); same masking for fully-masked rows.
    let p_raw = scores.broadcast_sub(&new_m)?.exp()?;
    let p = all_masked
        .broadcast_as(p_raw.shape())?
        .where_cond(&Tensor::zeros_like(&p_raw)?, &p_raw)?;
    let new_l = alpha.mul(&l)?.add(&p.sum_keepdim(3)?)?;
    let new_acc = alpha
        .broadcast_mul(&acc)?
        .add(&p.matmul(&v.to_dtype(DType::F32)?)?)?;
    Ok((new_m, new_l, new_acc))
}
#[allow(clippy::too_many_arguments)]
fn tiled_update(
    m: &Tensor,
    l: &Tensor,
    acc: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_start: usize,
    k_start: usize,
    causal: bool,
    valid_length: Option<usize>,
    block_k: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let total = k.dim(2)?;
    let q_n = q.dim(2)?;
    let device = q.device();
    let (mut m, mut l, mut acc) = (m.clone(), l.clone(), acc.clone());
    for offset in (0..total).step_by(block_k) {
        let end = (offset + block_k).min(total);
        let k_len = end - offset;
        let k_pos = Tensor::arange((k_start + offset) as u32, (k_start + end) as u32, device)?;
        let q_pos = Tensor::arange(q_start as u32, (q_start + q_n) as u32, device)?;
        // invalid[q, k] = (causal && k_pos > q_pos) || (valid && k_pos >= valid_length)
        let mut invalid = Tensor::zeros((q_n, k_len), DType::U8, device)?;
        if causal {
            let k_row = k_pos.unsqueeze(0)?.broadcast_as((q_n, k_len))?;
            let q_col = q_pos.unsqueeze(1)?.broadcast_as((q_n, k_len))?;
            let gt = k_row.gt(&q_col)?;
            invalid = invalid.broadcast_maximum(&gt.to_dtype(DType::U8)?)?;
        }
        if let Some(valid) = valid_length {
            let k_row = k_pos.unsqueeze(0)?.broadcast_as((q_n, k_len))?;
            let ge = k_row.ge(valid as u32)?;
            invalid = invalid.broadcast_maximum(&ge.to_dtype(DType::U8)?)?;
        }
        let mask = invalid.eq(1u8)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((
            q.dim(0)?,
            q.dim(1)?,
            q_n,
            k_len,
        ))?;
        let kb = k.narrow(2, offset, k_len)?;
        let vb = v.narrow(2, offset, k_len)?;
        let (nm, nl, na) = online_update(&m, &l, &acc, q, &kb, &vb, Some(&mask))?;
        m = nm;
        l = nl;
        acc = na;
    }
    Ok((m, l, acc))
}
/// Exact dense attention with rank-sharded Q/K/V and bounded KV workspace.
///
/// `shards` holds each rank's `[B, local_n, H, D]` Q/K/V triple in rank order.
/// `rank` selects this process's shard; the function simulates the ring
/// circulation in-process (hop h attends the shard owned by `(rank - h) mod
/// world`), so the arithmetic is identical to the distributed Python
/// `ring_attention._ring_causal_attention_forward`. `valid_length` masks the
/// padding tail on the final global token shard.
#[allow(clippy::too_many_arguments)]
pub fn ring_causal_attention_forward(
    shards: &[(Tensor, Tensor, Tensor)],
    rank: usize,
    causal: bool,
    block_q: usize,
    block_k: usize,
    valid_length: Option<usize>,
    q_position_start: Option<usize>,
) -> Result<Tensor> {
    let (out, _, _) = ring_causal_attention_forward_with_stats(
        shards,
        rank,
        causal,
        block_q,
        block_k,
        valid_length,
        q_position_start,
    )?;
    Ok(out)
}

/// Forward pass that also returns the online-softmax statistics `m` and `l`
/// needed by the backward pass, mirroring Python's `return_stats=True`.
pub fn ring_causal_attention_forward_with_stats(
    shards: &[(Tensor, Tensor, Tensor)],
    rank: usize,
    causal: bool,
    block_q: usize,
    block_k: usize,
    valid_length: Option<usize>,
    q_position_start: Option<usize>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let world_size = shards.len();
    if rank >= world_size {
        candle_core::bail!("rank {rank} out of range for world_size {world_size}");
    }
    let (q, k, v) = &shards[rank];
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        candle_core::bail!("q, k, v must have shape [batch, tokens, heads, head_dim]");
    }
    if q.shape() != k.shape() || q.shape() != v.shape() {
        candle_core::bail!("q, k, v shard shapes must match");
    }
    if block_q == 0 || block_k == 0 {
        candle_core::bail!("block_q and block_k must be positive");
    }
    let dims = q.dims();
    let (batch, local_n, heads, dim) = (dims[0], dims[1], dims[2], dims[3]);
    if local_n == 0 {
        candle_core::bail!("local_tokens must be positive");
    }
    if let Some(valid) = valid_length {
        if !(0 < valid && valid <= world_size * local_n) {
            candle_core::bail!("valid_length must be within the padded distributed sequence");
        }
    }
    let device = q.device();
    let qh = q.permute((0, 2, 1, 3))?.contiguous()?;
    // Pre-rotate all KV shards into [B, H, local_n, D] form.
    let kv_shards: Vec<(Tensor, Tensor)> = shards
        .iter()
        .map(|(_, k, v)| {
            Ok((
                k.permute((0, 2, 1, 3))?.contiguous()?,
                v.permute((0, 2, 1, 3))?.contiguous()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let neg_inf = f32::NEG_INFINITY;
    let mut m = Tensor::full(neg_inf, (batch, heads, local_n, 1), device)?;
    let mut l = Tensor::zeros((batch, heads, local_n, 1), DType::F32, device)?;
    let mut acc = Tensor::zeros((batch, heads, local_n, dim), DType::F32, device)?;

    let q_positions_start = q_position_start.unwrap_or(rank * local_n);

    // Circulate each rank's KV shard exactly once per ring hop. Hop h attends
    // the shard owned by (rank - h) mod world (Python rotates KV toward the
    // previous rank each hop).
    let mut owner = rank;
    for hop in 0..world_size {
        let (cur_k, cur_v) = &kv_shards[owner];
        for qs in (0..local_n).step_by(block_q) {
            let qe = (qs + block_q).min(local_n);
            let qm = m.narrow(2, qs, qe - qs)?;
            let ql = l.narrow(2, qs, qe - qs)?;
            let qa = acc.narrow(2, qs, qe - qs)?;
            let qq = qh.narrow(2, qs, qe - qs)?;
            let (nm, nl, na) = tiled_update(
                &qm,
                &ql,
                &qa,
                &qq,
                cur_k,
                cur_v,
                q_positions_start + qs,
                owner * local_n,
                causal,
                valid_length,
                block_k,
            )?;
            m = m.slice_scatter(&nm, 2, qs)?;
            l = l.slice_scatter(&nl, 2, qs)?;
            acc = acc.slice_scatter(&na, 2, qs)?;
        }
        if hop + 1 < world_size {
            // Ring rotation: the next hop processes the previous rank's shard
            // (a real distributed backend P2P-exchanges K/V here).
            owner = (owner + world_size - 1) % world_size;
        }
    }

    let out = acc.broadcast_div(&l.clamp(f32::MIN_POSITIVE, f32::MAX)?)?;
    let out = out
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .to_dtype(q.dtype())?;
    Ok((out, m, l))
}

/// Single-process correctness oracle for `[B, T, H, D]` tensors, mirroring
/// `ring_attention.reference_causal_attention`.
pub fn reference_causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    valid_length: Option<usize>,
) -> Result<Tensor> {
    let qh = q.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    let kh = k.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    let vh = v.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    let scale = (q.dim(3)? as f64).powf(-0.5);
    let mut scores = qh.matmul(&kh.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let t = q.dim(1)?;
    let device = q.device();
    if causal || valid_length.is_some() {
        let arange = Tensor::arange(0u32, t as u32, device)?;
        let row = arange.reshape((t, 1))?.broadcast_as((t, t))?;
        let col = arange.reshape((1, t))?.broadcast_as((t, t))?;
        let mut mask = col.gt(&row)?; // strictly above diagonal
        if let Some(valid) = valid_length {
            let ge = col.ge(valid as u32)?;
            mask = mask.broadcast_maximum(&ge)?;
        }
        let neg_inf = f32::NEG_INFINITY;
        let mask_t = mask.where_cond(
            &Tensor::full(neg_inf, (t, t), device)?,
            &Tensor::zeros((t, t), DType::F32, device)?,
        )?;
        scores = scores.broadcast_add(&mask_t.unsqueeze(0)?.unsqueeze(0)?)?;
    }
    let attn = candle_nn::ops::softmax(&scores, 3)?;
    let out = attn.matmul(&vh)?;
    out.permute((0, 2, 1, 3))?.contiguous()?.to_dtype(q.dtype())
}

/// Exact ring-attention backward, mirroring `ring_attention._ring_attention_backward`.
///
/// `shards` holds each rank's `[B, local_n, H, D]` Q/K/V triple in rank order;
/// `outs`, `ms`, `ls` are every rank's forward outputs and online-softmax stats
/// (from `ring_causal_attention_forward_with_stats` per rank); `grad_outs` holds
/// every rank's upstream gradient. The ring circulation is simulated in-process
/// (hop h attends the shard owned by `(rank - h) mod world`), matching the
/// distributed Python arithmetic. Returns `(dq, dk, dv)` for this rank's shard.
#[allow(clippy::too_many_arguments)]
pub fn ring_causal_attention_backward(
    shards: &[(Tensor, Tensor, Tensor)],
    rank: usize,
    outs: &[Tensor],
    ms: &[Tensor],
    ls: &[Tensor],
    grad_outs: &[Tensor],
    causal: bool,
    block_q: usize,
    block_k: usize,
    valid_length: Option<usize>,
    q_position_start: Option<usize>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let world_size = shards.len();
    if rank >= world_size {
        candle_core::bail!("rank {rank} out of range for world_size {world_size}");
    }
    let (q, k, v) = &shards[rank];
    let dims = q.dims();
    let (batch, local_n, heads, dim) = (dims[0], dims[1], dims[2], dims[3]);
    let device = q.device();
    let scale = (dim as f64).powf(-0.5);
    let neg_inf = f32::NEG_INFINITY;

    // Rotate all shards to [B, H, local_n, D].
    let qh = q.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    let kh = k.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    let vh = v.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
    // Per-rank forward outputs, stats, and upstream gradients (rotated to [B,H,n,D]).
    let all_q: Vec<Tensor> = shards
        .iter()
        .map(|(q, _, _)| q.permute((0, 2, 1, 3))?.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let all_go: Vec<Tensor> = grad_outs
        .iter()
        .map(|g| g.permute((0, 2, 1, 3))?.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let all_ro: Vec<Tensor> = outs
        .iter()
        .map(|o| o.permute((0, 2, 1, 3))?.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let all_rm: Vec<Tensor> = ms
        .iter()
        .map(|x| x.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let all_rl: Vec<Tensor> = ls
        .iter()
        .map(|x| x.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let all_kv: Vec<(Tensor, Tensor)> = shards
        .iter()
        .map(|(_, k, v)| {
            Ok((
                k.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?,
                v.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut dq = Tensor::zeros_like(&qh)?;
    let mut dk = Tensor::zeros_like(&kh)?;
    let mut dv = Tensor::zeros_like(&vh)?;
    let q_positions_start = q_position_start.unwrap_or(rank * local_n);

    // Helper closure to build the causal/valid mask for a [Q, K] score block.
    let build_mask = |q_pos: &Tensor, k_pos: &Tensor, q_n: usize, k_len: usize| -> Result<Tensor> {
        let mut invalid = Tensor::zeros((q_n, k_len), DType::U8, device)?;
        if causal {
            let k_row = k_pos.unsqueeze(0)?.broadcast_as((q_n, k_len))?;
            let q_col = q_pos.unsqueeze(1)?.broadcast_as((q_n, k_len))?;
            let gt = k_row.gt(&q_col)?;
            invalid = invalid.broadcast_maximum(&gt.to_dtype(DType::U8)?)?;
        }
        if let Some(valid) = valid_length {
            let k_row = k_pos.unsqueeze(0)?.broadcast_as((q_n, k_len))?;
            let ge = k_row.ge(valid as u32)?;
            invalid = invalid.broadcast_maximum(&ge.to_dtype(DType::U8)?)?;
        }
        let mask = invalid
            .eq(1u8)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch, heads, q_n, k_len))?;
        Ok(mask)
    };

    // Pass 1: keep local KV fixed; circulate Q, dO, and O so each KV owner
    // accumulates gradient contributions from every query shard. In-process:
    // hop h uses the tensors owned by (rank - h) mod world.
    let mut query_owner = rank;
    for _hop in 0..world_size {
        let rq = &all_q[query_owner];
        let rgo = &all_go[query_owner];
        let ro = &all_ro[query_owner];
        let rm = &all_rm[query_owner];
        let rl = &all_rl[query_owner];
        let qpos = arange_from(query_owner * local_n, local_n, device)?;
        for qs in (0..local_n).step_by(block_q) {
            let qe = (qs + block_q).min(local_n);
            let qq = rq.narrow(2, qs, qe - qs)?;
            let goq = rgo.narrow(2, qs, qe - qs)?;
            let delta = goq.mul(&ro.narrow(2, qs, qe - qs)?)?.sum_keepdim(3)?;
            for ks in (0..local_n).step_by(block_k) {
                let ke = (ks + block_k).min(local_n);
                let kb = kh.narrow(2, ks, ke - ks)?;
                let vb = vh.narrow(2, ks, ke - ks)?;
                let kp = arange_from(rank * local_n + ks, ke - ks, device)?;
                let qpos_block = qpos.narrow(0, qs, qe - qs)?;
                let scores = qq.matmul(&kb.transpose(2, 3)?)?.affine(scale, 0.0)?;
                let mask = build_mask(&qpos_block, &kp, qe - qs, ke - ks)?;
                let scores =
                    mask.where_cond(&Tensor::full(neg_inf, scores.shape(), device)?, &scores)?;
                // prob = exp(scores - rm) / rl, with fully-masked rows zeroed.
                let rm_block = rm.narrow(2, qs, qe - qs)?;
                let rl_block = rl.narrow(2, qs, qe - qs)?;
                let all_masked = rm_block.eq(neg_inf)?;
                let prob_raw = scores.broadcast_sub(&rm_block)?.exp()?;
                let prob_zero = all_masked
                    .broadcast_as(prob_raw.shape())?
                    .where_cond(&Tensor::zeros_like(&prob_raw)?, &prob_raw)?;
                let prob =
                    prob_zero.broadcast_div(&rl_block.clamp(f32::MIN_POSITIVE, f32::MAX)?)?;
                let ds = prob.mul(&goq.matmul(&vb.transpose(2, 3)?)?.broadcast_sub(&delta)?)?;
                let dk_block = ds.transpose(2, 3)?.matmul(&qq)?.affine(scale, 0.0)?;
                let dv_block = prob.transpose(2, 3)?.matmul(&goq)?;
                let cur_dk = dk.narrow(2, ks, ke - ks)?.add(&dk_block)?;
                dk = dk.slice_scatter(&cur_dk, 2, ks)?;
                let cur_dv = dv.narrow(2, ks, ke - ks)?.add(&dv_block)?;
                dv = dv.slice_scatter(&cur_dv, 2, ks)?;
            }
        }
        if _hop + 1 < world_size {
            // Ring rotation: next hop processes the previous query owner's shard.
            query_owner = (query_owner + world_size - 1) % world_size;
        }
    }

    // Pass 2: circulate K/V so each query owner forms its local dQ.
    let qpos = arange_from(q_positions_start, local_n, device)?;
    let go = &all_go[rank];
    let oh = &all_ro[rank];
    let m = &all_rm[rank];
    let l = &all_rl[rank];
    let mut owner = rank;
    for _hop in 0..world_size {
        let (cur_k, cur_v) = &all_kv[owner];
        let kpos = arange_from(owner * local_n, local_n, device)?;
        for qs in (0..local_n).step_by(block_q) {
            let qe = (qs + block_q).min(local_n);
            let qq = qh.narrow(2, qs, qe - qs)?;
            let goq = go.narrow(2, qs, qe - qs)?;
            let delta = goq.mul(&oh.narrow(2, qs, qe - qs)?)?.sum_keepdim(3)?;
            for ks in (0..local_n).step_by(block_k) {
                let ke = (ks + block_k).min(local_n);
                let kb = cur_k.narrow(2, ks, ke - ks)?;
                let vb = cur_v.narrow(2, ks, ke - ks)?;
                let kp = kpos.narrow(0, ks, ke - ks)?;
                let qpos_block = qpos.narrow(0, qs, qe - qs)?;
                let scores = qq.matmul(&kb.transpose(2, 3)?)?.affine(scale, 0.0)?;
                let mask = build_mask(&qpos_block, &kp, qe - qs, ke - ks)?;
                let scores =
                    mask.where_cond(&Tensor::full(neg_inf, scores.shape(), device)?, &scores)?;
                let m_block = m.narrow(2, qs, qe - qs)?;
                let l_block = l.narrow(2, qs, qe - qs)?;
                let all_masked = m_block.eq(neg_inf)?;
                let prob_raw = scores.broadcast_sub(&m_block)?.exp()?;
                let prob_zero = all_masked
                    .broadcast_as(prob_raw.shape())?
                    .where_cond(&Tensor::zeros_like(&prob_raw)?, &prob_raw)?;
                let prob = prob_zero.broadcast_div(&l_block.clamp(f32::MIN_POSITIVE, f32::MAX)?)?;
                let ds = prob.mul(&goq.matmul(&vb.transpose(2, 3)?)?.broadcast_sub(&delta)?)?;
                let dq_block = ds.matmul(&kb)?.affine(scale, 0.0)?;
                let cur_dq = dq.narrow(2, qs, qe - qs)?.add(&dq_block)?;
                dq = dq.slice_scatter(&cur_dq, 2, qs)?;
            }
        }
        if _hop + 1 < world_size {
            owner = (owner + world_size - 1) % world_size;
        }
    }

    let dq = dq
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .to_dtype(q.dtype())?;
    let dk = dk
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .to_dtype(k.dtype())?;
    let dv = dv
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .to_dtype(v.dtype())?;
    Ok((dq, dk, dv))
}

fn arange_u32(len: usize, device: &candle_core::Device) -> Result<Tensor> {
    Tensor::arange(0u32, len as u32, device)
}

fn arange_from(start: usize, len: usize, device: &candle_core::Device) -> Result<Tensor> {
    Tensor::arange(start as u32, (start + len) as u32, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    fn shard_tensors(
        device: &Device,
        tokens: usize,
        heads: usize,
        dim: usize,
        world: usize,
    ) -> Vec<Tensor> {
        // Deterministic pseudo-random shards so multi-rank comparison is stable.
        let mut out = Vec::new();
        let mut seed = 42u64;
        for _ in 0..world {
            let vals: Vec<f32> = (0..tokens * heads * dim)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
                })
                .collect();
            out.push(
                Tensor::new(vals.as_slice(), device)
                    .unwrap()
                    .reshape((1, tokens, heads, dim))
                    .unwrap(),
            );
        }
        out
    }

    #[test]
    fn ring_matches_reference_single_rank() {
        let device = Device::Cpu;
        let (b, t, h, d) = (1usize, 8usize, 2usize, 4usize);
        let q = Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device).unwrap();
        let shards = vec![(q.clone(), k.clone(), v.clone())];
        let ring = ring_causal_attention_forward(&shards, 0, true, 2, 3, None, None).unwrap();
        let ref_ = reference_causal_attention(&q, &k, &v, true, None).unwrap();
        let rv: Vec<f32> = ring.flatten_all().unwrap().to_vec1().unwrap();
        let fv: Vec<f32> = ref_.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in rv.iter().zip(fv.iter()) {
            assert!((a - b).abs() < 1e-5, "single-rank mismatch {a} vs {b}");
        }
    }

    #[test]
    fn ring_matches_reference_two_rank() {
        let device = Device::Cpu;
        let (t, h, d) = (4usize, 2usize, 4usize);
        let world = 2usize;
        let qs = shard_tensors(&device, t, h, d, world);
        let ks = shard_tensors(&device, t, h, d, world);
        let vs = shard_tensors(&device, t, h, d, world);
        // Concatenate shards into the full sequence for the dense oracle.
        let q_full = Tensor::cat(&qs.iter().collect::<Vec<_>>(), 1).unwrap();
        let k_full = Tensor::cat(&ks.iter().collect::<Vec<_>>(), 1).unwrap();
        let v_full = Tensor::cat(&vs.iter().collect::<Vec<_>>(), 1).unwrap();
        let ref_ = reference_causal_attention(&q_full, &k_full, &v_full, true, None).unwrap();
        let shards: Vec<(Tensor, Tensor, Tensor)> = (0..world)
            .map(|i| (qs[i].clone(), ks[i].clone(), vs[i].clone()))
            .collect();
        // Each rank's ring output must equal the dense oracle's shard slice.
        for rank in 0..world {
            let ring =
                ring_causal_attention_forward(&shards, rank, true, 2, 3, None, None).unwrap();
            let slice = ref_.narrow(1, rank * t, t).unwrap();
            let rv: Vec<f32> = ring.flatten_all().unwrap().to_vec1().unwrap();
            let fv: Vec<f32> = slice.flatten_all().unwrap().to_vec1().unwrap();
            for (a, b) in rv.iter().zip(fv.iter()) {
                assert!((a - b).abs() < 1e-5, "rank {rank} mismatch {a} vs {b}");
            }
        }
    }

    #[test]
    fn ring_masks_valid_length_padding() {
        let device = Device::Cpu;
        let (t, h, d) = (4usize, 2usize, 4usize);
        let world = 2usize;
        let qs = shard_tensors(&device, t, h, d, world);
        let ks = shard_tensors(&device, t, h, d, world);
        let vs = shard_tensors(&device, t, h, d, world);
        let q_full = Tensor::cat(&qs.iter().collect::<Vec<_>>(), 1).unwrap();
        let k_full = Tensor::cat(&ks.iter().collect::<Vec<_>>(), 1).unwrap();
        let v_full = Tensor::cat(&vs.iter().collect::<Vec<_>>(), 1).unwrap();
        let shards: Vec<(Tensor, Tensor, Tensor)> = (0..world)
            .map(|i| (qs[i].clone(), ks[i].clone(), vs[i].clone()))
            .collect();
        // valid_length = 6 of 8 padded tokens: the last two are masked.
        let ref_ = reference_causal_attention(&q_full, &k_full, &v_full, true, Some(6)).unwrap();
        for rank in 0..world {
            let ring =
                ring_causal_attention_forward(&shards, rank, true, 2, 3, Some(6), None).unwrap();
            let slice = ref_.narrow(1, rank * t, t).unwrap();
            let rv: Vec<f32> = ring.flatten_all().unwrap().to_vec1().unwrap();
            let fv: Vec<f32> = slice.flatten_all().unwrap().to_vec1().unwrap();
            for (a, b) in rv.iter().zip(fv.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "rank {rank} valid-length mismatch {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn ring_backward_matches_dense_autograd() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let device = Device::Cpu;
        let (b, t, h, d) = (1usize, 8usize, 2usize, 4usize);
        let q = Var::from_tensor(&Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device)?)?;
        let k = Var::from_tensor(&Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device)?)?;
        let v = Var::from_tensor(&Tensor::randn(0.0f32, 1.0, (b, t, h, d), &device)?)?;
        let shards = vec![(
            q.as_tensor().clone(),
            k.as_tensor().clone(),
            v.as_tensor().clone(),
        )];

        // Ring forward + stats for this (single) rank.
        let (out, m, l) =
            ring_causal_attention_forward_with_stats(&shards, 0, true, 2, 3, None, None)?;
        let grad_out = Tensor::ones((b, t, h, d), DType::F32, &device)?;
        let (bdq, bdk, bdv) = ring_causal_attention_backward(
            &shards,
            0,
            &[out.clone()],
            &[m.clone()],
            &[l.clone()],
            &[grad_out.clone()],
            true,
            2,
            3,
            None,
            None,
        )?;

        // Dense autograd ground truth: rebuild the same forward through candle's
        // autodiff (dense attention is the world_size=1 path in SelfAttention).
        // loss = sum(dense_attention(q,k,v)).
        let qh = q.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
        let kh = k.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
        let vh = v.permute((0, 2, 1, 3))?.to_dtype(DType::F32)?;
        let scale = (d as f64).powf(-0.5);
        let scores = qh.matmul(&kh.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let arange = Tensor::arange(0u32, t as u32, &device)?;
        let row = arange.reshape((t, 1))?.broadcast_as((t, t))?;
        let col = arange.reshape((1, t))?.broadcast_as((t, t))?;
        let above = col.gt(&row)?;
        let neg_inf = f32::NEG_INFINITY;
        let mask = above.where_cond(
            &Tensor::full(neg_inf, (t, t), &device)?,
            &Tensor::zeros((t, t), DType::F32, &device)?,
        )?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out_dense = attn.matmul(&vh)?;
        let loss = out_dense.sum_all()?;
        let grads = loss.backward()?;
        let dq_dense = grads.get(q.as_tensor()).unwrap().clone();
        let dk_dense = grads.get(k.as_tensor()).unwrap().clone();
        let dv_dense = grads.get(v.as_tensor()).unwrap().clone();

        let cmp = |a: &Tensor, b: &Tensor, name: &str| {
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
            let max_abs = av
                .iter()
                .zip(bv.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 1e-4, "{name} max abs diff {max_abs}");
        };
        cmp(&bdq, &dq_dense, "dq");
        cmp(&bdk, &dk_dense, "dk");
        cmp(&bdv, &dv_dense, "dv");
        Ok(())
    }
}
