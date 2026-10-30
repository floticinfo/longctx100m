use crate::CoreError;
use serde::Serialize;

const ASSUMPTIONS: [&str; 4] = [
    "KV cache only; excludes weights, MLP activations, optimizer, comm buffers and allocator slack",
    "cluster_gpu_count_for_context_ring requires the rounded per-rank shard to fit on every sequence-parallel rank; aggregate KV count ignores topology/placement and is not a deployment size",
    "model/tensor parallel devices per context rank are not modeled by this estimator",
    "dense full attention; sequence ring shards memory and FLOPs but does not reduce O(N^2) arithmetic",
];

#[derive(Debug, Clone, Copy)]
pub struct CapacityInput {
    pub tokens: u64,
    pub layers: u64,
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub bytes_per_element: u64,
    pub world: u64,
    pub local_shard_tokens: u64,
    pub block_q: u64,
    pub block_k: u64,
    pub output_tokens: u64,
    pub report_gpu_memory_gib: Option<f64>,
    pub usable_memory_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CapacityReport {
    pub tokens_input: u64,
    pub tokens_with_output: u64,
    pub world_size: u64,
    pub local_tokens_per_rank: u64,
    pub kv_cache_bytes_per_rank_lower_bound: u128,
    pub attention_score_tile_bytes_per_rank: u128,
    pub dense_causal_attention_flops_lower_bound: u128,
    pub context_ring_has_one_cuda_device_per_rank: bool,
    pub assumptions: [&'static str; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_only_bytes_fit_one_gpu_per_rank_at_requested_world: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_only_gpus_per_rank_if_model_parallel_existed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_context_ranks_for_kv_only_one_gpu_per_rank: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_world_size_has_kv_only_context_headroom: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_supports_multiple_gpus_per_context_rank: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_gib_each: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usable_memory_fraction: Option<f64>,
}

fn product(values: &[u64]) -> Result<u128, CoreError> {
    values.iter().try_fold(1u128, |a, v| {
        a.checked_mul(u128::from(*v))
            .ok_or(CoreError::ArithmeticOverflow)
    })
}
fn ceil_div(a: u128, b: u128) -> Result<u128, CoreError> {
    a.checked_add(b - 1)
        .map(|x| x / b)
        .ok_or(CoreError::ArithmeticOverflow)
}
fn ceil_positive(value: f64) -> u64 {
    value.ceil().clamp(1.0, u64::MAX as f64) as u64
}

/// Capacity lower bound, field names match `longctx100m.capacity.estimate`.
pub fn estimate_capacity(i: CapacityInput) -> Result<CapacityReport, CoreError> {
    let dims = [
        i.tokens,
        i.layers,
        i.query_heads,
        i.kv_heads,
        i.head_dim,
        i.bytes_per_element,
        i.world,
        i.local_shard_tokens,
        i.block_q,
        i.block_k,
        i.output_tokens,
    ];
    if dims.contains(&0) {
        return Err(CoreError::InvalidDimensions);
    }
    if i.query_heads % i.kv_heads != 0 {
        return Err(CoreError::InvalidHeadRatio);
    }
    let total = i
        .tokens
        .checked_add(i.output_tokens)
        .ok_or(CoreError::ArithmeticOverflow)?;
    if u128::from(total) > u128::from(i.world) * u128::from(i.local_shard_tokens) {
        return Err(CoreError::InsufficientShardCapacity);
    }
    let shard = ceil_div(u128::from(total), u128::from(i.world))?;
    let kv_total = product(&[
        2,
        i.layers,
        i.kv_heads,
        i.head_dim,
        i.bytes_per_element,
        total,
    ])?;
    let kv_rank = ceil_div(kv_total, u128::from(i.world))?;
    let tile = product(&[
        4,
        i.block_q.min(i.local_shard_tokens),
        i.block_k.min(i.local_shard_tokens),
        i.query_heads.max(1),
    ])?;
    let flops = product(&[2, i.layers, i.tokens, i.tokens, i.query_heads, i.head_dim])?;
    let mut r = CapacityReport {
        tokens_input: i.tokens,
        tokens_with_output: total,
        world_size: i.world,
        local_tokens_per_rank: u64::try_from(shard).map_err(|_| CoreError::ArithmeticOverflow)?,
        kv_cache_bytes_per_rank_lower_bound: kv_rank,
        attention_score_tile_bytes_per_rank: tile,
        dense_causal_attention_flops_lower_bound: flops,
        context_ring_has_one_cuda_device_per_rank: true,
        assumptions: ASSUMPTIONS,
        kv_only_bytes_fit_one_gpu_per_rank_at_requested_world: None,
        kv_only_gpus_per_rank_if_model_parallel_existed: None,
        minimum_context_ranks_for_kv_only_one_gpu_per_rank: None,
        requested_world_size_has_kv_only_context_headroom: None,
        runtime_supports_multiple_gpus_per_context_rank: None,
        gpu_memory_gib_each: None,
        usable_memory_fraction: None,
    };
    if let Some(mem) = i.report_gpu_memory_gib {
        if !mem.is_finite()
            || mem <= 0.0
            || !i.usable_memory_fraction.is_finite()
            || !(0.0 < i.usable_memory_fraction && i.usable_memory_fraction <= 1.0)
        {
            return Err(CoreError::InvalidMemoryParameters);
        }
        let usable = mem * 1024f64.powi(3) * i.usable_memory_fraction;
        let rank_bytes = kv_rank as f64;
        let capacity_ranks = ceil_div(u128::from(total), u128::from(i.local_shard_tokens))? as u64;
        r.kv_only_bytes_fit_one_gpu_per_rank_at_requested_world = Some(rank_bytes <= usable);
        r.kv_only_gpus_per_rank_if_model_parallel_existed =
            Some(ceil_positive(rank_bytes / usable));
        r.minimum_context_ranks_for_kv_only_one_gpu_per_rank = Some(ceil_positive(
            (kv_total as f64 / usable).max(capacity_ranks as f64),
        ));
        r.requested_world_size_has_kv_only_context_headroom =
            Some(shard <= u128::from(i.local_shard_tokens) && rank_bytes <= usable);
        r.runtime_supports_multiple_gpus_per_context_rank = Some(false);
        r.gpu_memory_gib_each = Some(mem);
        r.usable_memory_fraction = Some(i.usable_memory_fraction);
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(world: u64, gpu: Option<f64>) -> CapacityInput {
        CapacityInput {
            tokens: 100_000_000,
            layers: 24,
            query_heads: 16,
            kv_heads: 16,
            head_dim: 128,
            bytes_per_element: 2,
            world,
            local_shard_tokens: 1_000_000,
            block_q: 64,
            block_k: 128,
            output_tokens: 2048,
            report_gpu_memory_gib: gpu,
            usable_memory_fraction: 0.8,
        }
    }
    #[test]
    fn matches_python_capacity_golden_values() {
        let r = estimate_capacity(fixture(128, Some(80.0))).unwrap();
        assert_eq!(r.tokens_with_output, 100_002_048);
        assert_eq!(r.local_tokens_per_rank, 781_266);
        assert_eq!(r.kv_cache_bytes_per_rank_lower_bound, 153_603_145_728);
        assert_eq!(r.attention_score_tile_bytes_per_rank, 524_288);
        assert_eq!(
            r.dense_causal_attention_flops_lower_bound,
            983_040_000_000_000_000_000
        );
        assert_eq!(r.kv_only_gpus_per_rank_if_model_parallel_existed, Some(3));
        assert_eq!(
            r.minimum_context_ranks_for_kv_only_one_gpu_per_rank,
            Some(287)
        );
        assert_eq!(
            r.requested_world_size_has_kv_only_context_headroom,
            Some(false)
        );
    }
    #[test]
    fn rejects_invalid_head_ratio_and_capacity() {
        let mut i = fixture(128, None);
        i.kv_heads = 3;
        assert_eq!(estimate_capacity(i), Err(CoreError::InvalidHeadRatio));
        let mut i = fixture(128, None);
        i.local_shard_tokens = 700_000;
        assert_eq!(
            estimate_capacity(i),
            Err(CoreError::InsufficientShardCapacity)
        );
    }
    #[test]
    fn omits_unrequested_optional_report_fields() {
        let r = estimate_capacity(fixture(128, None)).unwrap();
        let v = serde_json::to_value(r).unwrap();
        assert!(v.get("gpu_memory_gib_each").is_none());
        assert!(
            v.get("runtime_supports_multiple_gpus_per_context_rank")
                .is_none()
        );
    }
}
