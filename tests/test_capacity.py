from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from longctx100m.capacity import estimate


class CapacityTests(unittest.TestCase):
    def test_separates_ring_placement_from_aggregate_kv_sum(self):
        r = estimate(
            tokens=100_000_000,
            output_tokens=2048,
            layers=24,
            query_heads=16,
            kv_heads=16,
            head_dim=128,
            bytes_per_element=2,
            world=128,
            local_shard_tokens=1_000_000,
            block_q=64,
            block_k=128,
            report_gpu_memory=80,
            usable_memory_fraction=0.8,
        )
        self.assertFalse(r["kv_only_bytes_fit_one_gpu_per_rank_at_requested_world"])
        self.assertEqual(r["kv_only_gpus_per_rank_if_model_parallel_existed"], 3)
        self.assertEqual(r["minimum_context_ranks_for_kv_only_one_gpu_per_rank"], 287)
        self.assertEqual(r["kv_cache_bytes_per_rank_lower_bound"], 153603145728)
        self.assertFalse(r["requested_world_size_has_kv_only_context_headroom"])
        self.assertFalse(r["runtime_supports_multiple_gpus_per_context_rank"])
        self.assertEqual(r["local_tokens_per_rank"], 781266)

    def test_accepts_minimum_capacity_on_large_context_world(self):
        r = estimate(
            tokens=100_000_000,
            output_tokens=2048,
            layers=2,
            query_heads=2,
            kv_heads=2,
            head_dim=8,
            bytes_per_element=2,
            world=102,
            local_shard_tokens=1_000_000,
            block_q=32,
            block_k=32,
            report_gpu_memory=80,
        )
        self.assertTrue(r["requested_world_size_has_kv_only_context_headroom"])
        self.assertEqual(r["local_tokens_per_rank"], 980413)

    def test_rejects_shard_bytes_that_do_not_fit_device(self):
        r = estimate(
            tokens=1000,
            output_tokens=8,
            layers=24,
            query_heads=16,
            kv_heads=16,
            head_dim=128,
            bytes_per_element=2,
            world=2,
            local_shard_tokens=600,
            block_q=8,
            block_k=8,
            report_gpu_memory=0.01,
            usable_memory_fraction=0.5,
        )
        self.assertFalse(r["requested_world_size_has_kv_only_context_headroom"])
        self.assertTrue(r["context_ring_has_one_cuda_device_per_rank"])

    def test_rejects_impossible_shard_capacity(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            estimate(
                tokens=100,
                output_tokens=1,
                layers=1,
                query_heads=1,
                kv_heads=1,
                head_dim=8,
                bytes_per_element=2,
                world=2,
                local_shard_tokens=40,
                block_q=8,
                block_k=8,
            )


if __name__ == "__main__":
    unittest.main()
