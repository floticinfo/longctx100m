"""Reference distributed pretraining loop for the ring-attention decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .model import LongContextDecoder, ModelConfig


def init_distributed():
    world = int(os.getenv("WORLD_SIZE", "1"))
    if world > 1 and not torch.cuda.is_available():
        raise RuntimeError(
            "multi-rank training requires CUDA/NCCL; Gloo is reserved for correctness tests"
        )
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return (
        device,
        dist.get_world_size() if dist.is_initialized() else 1,
        dist.get_rank() if dist.is_initialized() else 0,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Train the ring-attention decoder from token-id shards"
    )
    p.add_argument(
        "--tokens", required=True, help="flat uint32 token-id file shared across nodes"
    )
    p.add_argument("--config", required=True, help="model config JSON")
    p.add_argument(
        "--sequence-tokens",
        type=int,
        default=32768,
        help="input tokens per packed sequence; corpus record contains sequence_tokens+1 IDs",
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block-q", type=int, default=64)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--resume",
        default=None,
        help="resume model and optimizer state from a saved checkpoint",
    )
    p.add_argument(
        "--initial-checkpoint",
        default=None,
        help="initialize model weights/tokenizer from imported HF checkpoint directory",
    )
    p.add_argument("--out", default="runs/longctx")
    a = p.parse_args(argv)
    device, world, rank = init_distributed()
    config = ModelConfig(**json.loads(Path(a.config).read_text(encoding="utf-8")))
    config.validate()
    tokenizer_source = None
    if a.initial_checkpoint:
        initial_paths = [a.initial_checkpoint if rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(initial_paths, src=0)
        a.initial_checkpoint = initial_paths[0]
        init_dir = Path(a.initial_checkpoint)
        if a.resume:
            raise ValueError("--initial-checkpoint and --resume are mutually exclusive")
        if (
            not (init_dir / "config.json").is_file()
            or not (init_dir / "manifest.json").is_file()
            or not (init_dir / "model.safetensors").is_file()
            or not (init_dir / "tokenizer").is_dir()
        ):
            raise ValueError("initial checkpoint directory is incomplete")
        init_config = ModelConfig(
            **json.loads((init_dir / "config.json").read_text(encoding="utf-8"))
        )
        if init_config != config:
            raise ValueError(
                "initial checkpoint model config must exactly match --config"
            )
        init_manifest = json.loads(
            (init_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if init_manifest.get("tokenizer_length") != config.vocab_size:
            raise ValueError("initial tokenizer vocabulary differs from model config")
        tokenizer_source = str(init_dir / "tokenizer")
    if a.sequence_tokens <= 0 or a.steps <= 0 or a.save_every <= 0 or a.lr <= 0:
        raise ValueError("sequence-tokens, steps, save-every and lr must be positive")
    # Parse a raw uint32 token file without NumPy dependency and reject corrupt bytes.
    corpus_path = Path(a.tokens)
    corpus_bytes = corpus_path.stat().st_size
    if corpus_bytes % 4:
        raise ValueError("token corpus byte count must be divisible by uint32 size")
    # Map the shared corpus file; only the selected packed row is copied per step.
    tokens = torch.from_file(
        str(corpus_path), shared=False, size=corpus_bytes // 4, dtype=torch.uint32
    )
    record_tokens = a.sequence_tokens + 1
    if tokens.numel() % record_tokens:
        raise ValueError(
            "corpus token count must be divisible by sequence_tokens+1; no partial record truncation"
        )
    if tokens.numel() < record_tokens:
        raise ValueError("token corpus smaller than one packed record")
    # Validate IDs lazily per selected row; scanning all 100M+ token IDs during
    # startup would fault the entire memory map into every rank's page cache.
    tokens = tokens.view(-1, record_tokens)
    padded_n = ((a.sequence_tokens + world - 1) // world) * world
    pad_right = padded_n - a.sequence_tokens
    local_n = padded_n // world
    model = LongContextDecoder(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    if a.initial_checkpoint:
        initial_paths = [a.initial_checkpoint if rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(initial_paths, src=0)
        a.initial_checkpoint = initial_paths[0]
    if a.resume and a.initial_checkpoint:
        raise ValueError("--resume and --initial-checkpoint are mutually exclusive")
    tokenizer_source = None
    if a.resume:
        resume_paths = [a.resume if rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(resume_paths, src=0)
        a.resume = resume_paths[0]
    if a.resume and Path(a.resume).is_dir() and rank == 0:
        resume_dir = Path(a.resume)
        if (
            not (resume_dir / "config.json").is_file()
            or not (resume_dir / "manifest.json").is_file()
            or not (resume_dir / "model.safetensors").is_file()
            or not (resume_dir / "tokenizer").is_dir()
        ):
            raise ValueError("imported checkpoint directory is incomplete")
        manifest = json.loads(
            (resume_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            ModelConfig(
                **json.loads((resume_dir / "config.json").read_text(encoding="utf-8"))
            )
            != config
        ):
            raise ValueError("resume checkpoint model config mismatch")
        tokenizer_source = str(resume_dir / "tokenizer")
        if manifest.get("tokenizer_length") != config.vocab_size:
            raise ValueError("resume tokenizer vocabulary differs from model config")
    start_step = 0
    if a.initial_checkpoint and not a.resume and rank == 0:
        from .model import safe_load_file

        init_state = safe_load_file(
            Path(a.initial_checkpoint) / "model.safetensors", device="cpu"
        )
        if config.tie_word_embeddings and "lm_head.weight" not in init_state:
            init_state["lm_head.weight"] = init_state["model.embed_tokens.weight"]
        model.load_state_dict(init_state, strict=True)
    if a.initial_checkpoint:
        tokenizer_source = str(Path(a.initial_checkpoint) / "tokenizer")
    if dist.is_initialized():
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
    if a.resume:
        # Rank zero loads checkpoint tensors into host RAM only; no multi-GB pickling.
        metadata = [None]
        state = None
        if rank == 0:
            resume_path = Path(a.resume)
            if resume_path.is_dir():
                from .model import safe_load_file

                config_data = json.loads(
                    (resume_path / "config.json").read_text(encoding="utf-8")
                )
                weights = safe_load_file(
                    resume_path / "model.safetensors", device="cpu"
                )
                if (
                    config_data.get("tie_word_embeddings")
                    and "lm_head.weight" not in weights
                ):
                    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
                state = {
                    "config": config_data,
                    "model": weights,
                    "step": -1,
                    "world_size": world,
                    "tokenizer_source": str(resume_path / "tokenizer"),
                }
            else:
                state = torch.load(a.resume, map_location="cpu", weights_only=False)
                portable_tokenizer = resume_path.parent / "tokenizer"
                if portable_tokenizer.is_dir():
                    state["tokenizer_source"] = str(portable_tokenizer)
            metadata[0] = {
                "config": state.get("config"),
                "world_size": state.get("world_size", 1),
                "step": state.get("step", -1),
                "has_optimizer": "optimizer" in state,
                "tokenizer_source": state.get("tokenizer_source"),
            }
        if dist.is_initialized():
            dist.broadcast_object_list(metadata, src=0)
        meta = metadata[0]
        if world > 1 and int(meta["world_size"]) != world:
            raise ValueError("resume checkpoint context world size differs")
        if meta["config"] != model.config_dict():
            raise ValueError("resume checkpoint model config mismatch")
        if meta.get("tokenizer_source"):
            tokenizer_source = meta["tokenizer_source"]
        if rank == 0 and state:
            legacy = state["model"]
            if "embed.weight" in legacy and "model.embed_tokens.weight" in legacy:
                legacy.pop("embed.weight")
        for name, param in model.state_dict().items():
            tensor = (
                state["model"][name].to(device)
                if rank == 0
                else torch.empty_like(param, device=device)
            )
            if dist.is_initialized():
                dist.broadcast(tensor, src=0)
            param.copy_(tensor)
        if meta["has_optimizer"]:
            if rank == 0:
                opt.load_state_dict(state["optimizer"])
            for param in model.parameters():
                slots = opt.state[param] if rank == 0 else {}
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = slots.get(key)
                    flag = (
                        torch.tensor(
                            [value is not None], device=device, dtype=torch.int32
                        )
                        if rank == 0
                        else torch.zeros(1, device=device, dtype=torch.int32)
                    )
                    if dist.is_initialized():
                        dist.broadcast(flag, src=0)
                    if not flag.item():
                        continue
                    if rank == 0:
                        value = value.to(device)
                    elif key == "step":
                        value = torch.zeros((), device=device, dtype=torch.float32)
                    else:
                        value = torch.empty_like(param, device=device)
                    if dist.is_initialized():
                        dist.broadcast(value, src=0)
                    if rank != 0:
                        opt.state[param][key] = value
        start_step = int(meta["step"]) + 1
    model.train()
    # Pad the global sequence once to equal local shards; mask padding in attention
    # and ignore its labels so valid training tokens are never dropped.
    # Each rank evaluates a disjoint target shard while attention spans all ranks.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, start_step + a.steps):
        row = tokens[(step % len(tokens))].to(device).long()
        if int(row.min()) < 0 or int(row.max()) >= config.vocab_size:
            raise ValueError("corpus token id is outside model vocabulary")
        x = row[:-1]
        y = row[1:]
        if pad_right:
            x = torch.cat((x, torch.zeros(pad_right, dtype=x.dtype, device=device)))
            y = torch.cat((y, torch.zeros(pad_right, dtype=y.dtype, device=device)))
        start = rank * local_n
        x_local = x[start : start + local_n].unsqueeze(0).contiguous()
        y_local = y[start : start + local_n].contiguous()
        opt.zero_grad(set_to_none=True)
        logits = model(
            x_local,
            group=dist.group.WORLD if dist.is_initialized() else None,
            block_q=a.block_q,
            block_k=a.block_k,
            valid_length=a.sequence_tokens,
            position_offset=start,
        )
        # Ignore padded target positions; all valid next-token labels remain in the loss.
        targets = y_local.clone()
        if pad_right and start + local_n > a.sequence_tokens:
            targets[max(0, a.sequence_tokens - start) :] = -100
        loss = (
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / a.sequence_tokens
        )
        loss.backward()
        if dist.is_initialized() and world > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            reduced = loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(
                    json.dumps({"step": step, "loss": reduced.item() / world}),
                    flush=True,
                )
        if step % a.save_every == 0 or step == start_step + a.steps - 1:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                core = model.module if hasattr(model, "module") else model
                checkpoint = {
                    "model": core.export_pretrained_state_dict(),
                    "tokenizer_source": tokenizer_source,
                    "optimizer": opt.state_dict(),
                    "config": core.config_dict(),
                    "step": step,
                    "world_size": world,
                    "rng_state": torch.get_rng_state(),
                }
                tmp = out / f"step-{step}.tmp"
                torch.save(checkpoint, tmp)
                tmp.replace(out / f"step-{step}.pt")
                if tokenizer_source and Path(tokenizer_source).is_dir():
                    import shutil

                    tokenizer_dir = out / "tokenizer"
                    if not tokenizer_dir.exists():
                        shutil.copytree(tokenizer_source, tokenizer_dir)
                if (
                    tokenizer_source
                    and Path(tokenizer_source).is_dir()
                    and not (out / "manifest.json").exists()
                ):
                    source_dir = Path(tokenizer_source).parent
                    source_manifest = json.loads(
                        (source_dir / "manifest.json").read_text(encoding="utf-8")
                    )
                    source_manifest.update(
                        {
                            "pretrained_for_100m_context": False,
                            "continued_pretraining_step": step,
                            "source_checkpoint": source_dir.name,
                        }
                    )
                    (out / "manifest.json").write_text(
                        json.dumps(source_manifest, indent=2), encoding="utf-8"
                    )
            if dist.is_initialized():
                dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
