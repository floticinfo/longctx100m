"""Single-request token-file inference for the context-parallel decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .model import LongContextDecoder, ModelConfig


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run one complete token-ID sequence using ring context parallelism"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", required=True, help="uint32 token id file")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--block-q", type=int, default=64)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument(
        "--seed", type=int, default=0, help="seed for identical weights/token sampling"
    )
    p.add_argument(
        "--local-shard-tokens",
        type=int,
        default=128,
        help="capacity tokens per context rank including generated tokens",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="total sequence-parallel ranks",
    )
    p.add_argument(
        "--token-output",
        default="generated.rank0.txt",
        help="raw uint32 generated token ids written by rank zero",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--gpu-memory-gib",
        type=float,
        default=None,
        help="optional per-device memory for explicit KV-only lower-bound estimate",
    )
    a = p.parse_args(argv)
    if a.local_shard_tokens <= 0 or a.world_size <= 0 or a.max_new_tokens < 0:
        p.error(
            "world-size/shard tokens must be positive; max-new-tokens cannot be negative"
        )
    if a.temperature != 0:
        p.error("only deterministic greedy decoding is currently supported")
    if a.gpu_memory_gib is not None:
        from .capacity import estimate

        estimate(
            tokens=100_000_000,
            output_tokens=a.max_new_tokens,
            layers=24,
            query_heads=16,
            kv_heads=16,
            head_dim=128,
            bytes_per_element=2,
            world=a.world_size,
            local_shard_tokens=a.local_shard_tokens,
            block_q=a.block_q,
            block_k=a.block_k,
            report_gpu_memory=a.gpu_memory_gib,
        )
    token_path = Path(a.tokens)
    file_bytes = token_path.stat().st_size
    if file_bytes % 4:
        raise ValueError("input token file size must be a multiple of uint32")
    valid = file_bytes // 4
    if valid < 100_000_000:
        raise ValueError(
            "input token file is shorter than the required 100M-token request"
        )
    if valid + a.max_new_tokens > a.world_size * a.local_shard_tokens:
        raise ValueError(
            "requested context ranks do not have input plus output shard capacity"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "distributed inference requires CUDA/NCCL; CPU is supported for tests only"
        )
    if a.world_size < 2:
        raise ValueError(
            "100M inference requires at least two context-parallel CUDA ranks"
        )
    env_world = int(os.getenv("WORLD_SIZE", "1"))
    if env_world != a.world_size:
        raise ValueError(
            f"set torchrun WORLD_SIZE={a.world_size}; current launcher environment is {env_world}"
        )
    local = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != a.world_size:
        raise ValueError(
            f"launcher world size {world} does not match requested world-size {a.world_size}"
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    from .model import safe_load_file

    checkpoint_dir = Path(a.checkpoint)
    if (
        checkpoint_dir.is_dir()
        and (checkpoint_dir / "config.json").is_file()
        and (checkpoint_dir / "manifest.json").is_file()
        and (checkpoint_dir / "tokenizer").is_dir()
    ):
        manifest = json.loads(
            (checkpoint_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("attention") != "exact_dense_causal_ring"
        ):
            raise ValueError("unsupported imported checkpoint manifest")
        config = ModelConfig(
            **json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        )
        weights = safe_load_file(checkpoint_dir / "model.safetensors", device="cpu")
        if config.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    else:
        if checkpoint_dir.is_dir():
            raise ValueError(
                "converted checkpoint directory is incomplete (config, manifest, model.safetensors and tokenizer are required)"
            )
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        config = ModelConfig(**checkpoint["config"])
        weights = checkpoint["model"]
    config.validate()
    model = LongContextDecoder(config).to(
        device=f"cuda:{local}", dtype=next(iter(weights.values())).dtype
    )
    model.load_state_dict(weights, strict=True)
    model.eval()
    if a.max_new_tokens == 0:
        raise ValueError("max-new-tokens must be positive")
    local_n = a.local_shard_tokens
    if local_n <= 0:
        raise ValueError("local-shard-tokens must be positive")
    padded = local_n * world
    if valid > padded:
        raise ValueError(
            "input exceeds current context-parallel capacity; increase local-shard-token capacity"
        )
    if valid + a.max_new_tokens > padded:
        raise ValueError(
            "reserve output capacity by increasing --local-shard-tokens or world size"
        )
    if valid + a.max_new_tokens > config.max_position_embeddings:
        raise ValueError("input plus requested generation exceeds model context")
    local_start = rank * local_n
    local_count = max(0, min(local_n, valid - local_start))
    with token_path.open("rb") as f:
        f.seek(local_start * 4)
        shard_bytes = f.read(local_count * 4)
    shard = (
        torch.frombuffer(bytearray(shard_bytes), dtype=torch.uint32).long()
        if local_count
        else torch.empty(0, dtype=torch.long)
    )
    if local_count and int(shard.max()) >= config.vocab_size:
        raise ValueError("token file includes ids outside model vocabulary")
    if shard.numel() < local_n:
        shard = torch.cat(
            (shard, torch.zeros(local_n - shard.numel(), dtype=torch.long))
        )
    local_tokens = shard.to(f"cuda:{local}").unsqueeze(0)
    torch.manual_seed(a.seed)
    model.load_state_dict(checkpoint["model"])
    # Keep PAD distinct from valid input; attention masks the suffix on every layer.
    generated_ids = []
    with torch.inference_mode():
        for step in range(a.max_new_tokens):
            local_valid = valid + step
            if local_valid >= padded or local_valid >= config.max_position_embeddings:
                raise ValueError(
                    "generation reached the allocated shard or model context limit"
                )
            logits = model(
                local_tokens,
                group=dist.group.WORLD if dist.is_initialized() else None,
                block_q=a.block_q,
                block_k=a.block_k,
                valid_length=local_valid,
                position_offset=rank * local_n,
            )
            # Only the rank owning the final actual sequence position proposes a token.
            owner = (local_valid - 1) // local_n
            proposed = torch.zeros((1,), dtype=torch.long, device=f"cuda:{local}")
            if rank == owner:
                last_local = (local_valid - 1) - rank * local_n
                scores = logits[:, last_local].float()
                proposed = scores.argmax(-1).reshape(-1)
            if dist.is_initialized():
                dist.broadcast(proposed, src=owner)
            token = proposed.reshape(1, 1)
            # All ranks append the exact same generated token at the global tail owner.
            append_rank = local_valid // local_n
            if append_rank >= world:
                raise ValueError(
                    "generation reached the context-parallel padding capacity; increase --local-shard-tokens"
                )
            if rank == append_rank:
                offset = local_valid - append_rank * local_n
                local_tokens[:, offset : offset + 1] = token
            generated_ids.append(int(proposed.item()))
    if rank == 0:
        import struct

        Path(a.token_output).write_bytes(
            struct.pack(f"<{len(generated_ids)}I", *generated_ids)
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
