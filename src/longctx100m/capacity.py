"""Lower-bound memory/work calculator for choosing distributed engine shape."""

from __future__ import annotations

import argparse
import json
import math


def estimate(
    tokens,
    layers,
    query_heads,
    kv_heads,
    head_dim,
    bytes_per_element,
    world,
    local_shard_tokens,
    block_q,
    block_k,
    output_tokens,
    report_gpu_memory=None,
    usable_memory_fraction=0.8,
):
    if (
        min(
            tokens,
            layers,
            query_heads,
            kv_heads,
            head_dim,
            bytes_per_element,
            world,
            local_shard_tokens,
            block_q,
            block_k,
            output_tokens,
        )
        <= 0
    ):
        raise ValueError("all estimator dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if tokens + output_tokens > world * local_shard_tokens:
        raise ValueError("per-rank shard capacity is smaller than input plus output")
    shard = math.ceil((tokens + output_tokens) / world)
    block_q = min(block_q, local_shard_tokens)
    block_k = min(block_k, local_shard_tokens)
    # K,V for the current layer/rank shard; weights/activations/optimizer excluded.
    kv_total_bytes = (
        2 * layers * kv_heads * head_dim * bytes_per_element * (tokens + output_tokens)
    )
    kv_bytes = math.ceil(kv_total_bytes / world)
    score_tile_bytes = 4 * block_q * block_k * max(1, query_heads)
    # Exact causal QK+AV multiply-add count is approximately 2*N^2*QH*D FLOPs/layer.
    attention_flops = 2 * layers * tokens * tokens * query_heads * head_dim
    report = {
        "tokens_input": tokens,
        "tokens_with_output": tokens + output_tokens,
        "world_size": world,
        "local_tokens_per_rank": shard,
        "kv_cache_bytes_per_rank_lower_bound": kv_bytes,
        "attention_score_tile_bytes_per_rank": score_tile_bytes,
        "dense_causal_attention_flops_lower_bound": attention_flops,
        "context_ring_has_one_cuda_device_per_rank": True,
        "assumptions": [
            "KV cache only; excludes weights, MLP activations, optimizer, comm buffers and allocator slack",
            "cluster_gpu_count_for_context_ring requires the rounded per-rank shard to fit on every sequence-parallel rank; aggregate KV count ignores topology/placement and is not a deployment size",
            "model/tensor parallel devices per context rank are not modeled by this estimator",
            "dense full attention; sequence ring shards memory and FLOPs but does not reduce O(N^2) arithmetic",
        ],
    }
    if report_gpu_memory is not None:
        if report_gpu_memory <= 0 or not 0 < usable_memory_fraction <= 1:
            raise ValueError("invalid device-memory estimate parameters")
        usable = report_gpu_memory * 1024**3 * usable_memory_fraction
        per_rank_gpu_count = math.ceil(kv_bytes / usable)
        aggregate_kv_only_count = math.ceil(kv_total_bytes / usable)
        ranks_for_shard_capacity = math.ceil(
            (tokens + output_tokens) / local_shard_tokens
        )
        min_context_ranks = max(aggregate_kv_only_count, ranks_for_shard_capacity)
        report["kv_only_bytes_fit_one_gpu_per_rank_at_requested_world"] = (
            kv_bytes <= usable
        )
        report["kv_only_gpus_per_rank_if_model_parallel_existed"] = per_rank_gpu_count
        report["minimum_context_ranks_for_kv_only_one_gpu_per_rank"] = min_context_ranks
        report["requested_world_size_has_kv_only_context_headroom"] = (
            tokens + output_tokens
        ) / world <= local_shard_tokens and kv_bytes <= usable
        report["runtime_supports_multiple_gpus_per_context_rank"] = False
        report["gpu_memory_gib_each"] = report_gpu_memory
        report["usable_memory_fraction"] = usable_memory_fraction
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Estimate required context-parallel topology (not a benchmark)"
    )
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--query-heads", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--bytes-per-element", type=int, default=2)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--local-shard-tokens", type=int, required=True)
    p.add_argument("--block-q", type=int, default=64)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--gpu-memory-gib", type=float, default=None)
    p.add_argument("--usable-memory-fraction", type=float, default=0.8)

    a = p.parse_args(argv)
    vals = [
        a.tokens,
        a.output_tokens,
        a.layers,
        a.query_heads,
        a.kv_heads,
        a.head_dim,
        a.bytes_per_element,
        a.world_size,
        a.local_shard_tokens,
        a.block_q,
        a.block_k,
    ]
    if min(vals) <= 0:
        p.error("all integer dimensions must be positive")
    if a.tokens + a.output_tokens > a.world_size * a.local_shard_tokens:
        p.error("world-size * local-shard-tokens cannot hold input plus output")
    print(
        json.dumps(
            estimate(
                a.tokens,
                a.layers,
                a.query_heads,
                a.kv_heads,
                a.head_dim,
                a.bytes_per_element,
                a.world_size,
                a.local_shard_tokens,
                a.block_q,
                a.block_k,
                a.output_tokens,
                a.gpu_memory_gib,
                a.usable_memory_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
