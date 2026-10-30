"""Rust/Python model runtime parity harness.

Verifies that the Rust `longctx100m-model` decoder produces identical logits
and training losses to the Python `LongContextDecoder` when both load the same
checkpoint. Requires the Rust binaries to be built:
    cargo build -p longctx100m-model --offline
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from longctx100m.model import LongContextDecoder, ModelConfig


def build_parity_checkpoint(dir: Path, config: ModelConfig, seed: int = 0) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "config.json").write_text(
        json.dumps(config.__dict__, indent=2), encoding="utf-8"
    )
    torch.manual_seed(seed)
    model = LongContextDecoder(config)
    state = model.export_pretrained_state_dict()
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    from safetensors.torch import save_file

    save_file(state, str(dir / "model.safetensors"))
    tokenizer_dir = dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = {
        "model_id": "parity",
        "revision": "local",
        "tokenizer_length": config.vocab_size,
        "loaded_tensors": len(state),
        "parameters": sum(v.numel() for v in state.values()),
        "source_max_position_embeddings": config.max_position_embeddings,
        "target_max_position_embeddings": config.max_position_embeddings,
        "rope_scaling": None,
        "attention_bias": config.attention_bias,
        "mlp_bias": config.mlp_bias,
        "tie_word_embeddings": config.tie_word_embeddings,
        "attention": "exact_dense_causal_ring",
        "pretrained_for_100m_context": False,
        "long_context_training_required": True,
    }
    (dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dir


def rust_logits(checkpoint: Path, prompt: list[int]) -> list[float]:
    cmd = [
        "cargo",
        "run",
        "--offline",
        "-p",
        "longctx100m-model",
        "--bin",
        "longctx100m-infer",
        "--",
        "--checkpoint",
        str(checkpoint),
        "--prompt-tokens",
        ",".join(str(t) for t in prompt),
        "--dump-logits",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if "last_position_logits" in line:
            payload = json.loads(line)
            return payload["last_position_logits"]
    raise RuntimeError(f"no logits in rust output: {out}")


def python_logits(checkpoint: Path, prompt: list[int]) -> list[float]:
    from safetensors.torch import load_file

    config = ModelConfig(
        **json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    if config.tie_word_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model = LongContextDecoder(config).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        logits = model(torch.tensor([prompt]))
    return logits[0, -1, :].tolist()


def rust_train_losses(
    checkpoint: Path,
    corpus: Path,
    steps: int,
    sequence_tokens: int,
    resume_dir: Path | None = None,
) -> list[float]:
    out_dir = Path(tempfile.mkdtemp(prefix="rust-train-"))
    cmd = [
        "cargo",
        "run",
        "--offline",
        "-p",
        "longctx100m-model",
        "--bin",
        "longctx100m-train",
        "--",
        "--tokens",
        str(corpus),
        "--config",
        str(checkpoint / "config.json"),
        "--sequence-tokens",
        str(sequence_tokens),
        "--steps",
        str(steps),
        "--save-every",
        str(steps),
        "--out",
        str(out_dir),
    ]
    if resume_dir is not None:
        cmd += ["--resume", str(resume_dir)]
    else:
        cmd += ["--initial-checkpoint", str(checkpoint)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    losses = []
    for line in out.splitlines():
        if '"loss"' in line:
            losses.append(json.loads(line)["loss"])
    return losses


def python_train_losses(checkpoint: Path, corpus: Path, steps: int, sequence_tokens: int) -> list[float]:
    out_dir = Path(tempfile.mkdtemp(prefix="py-train-"))
    cmd = [
        sys.executable,
        "-m",
        "longctx100m.train",
        "--tokens",
        str(corpus),
        "--config",
        str(checkpoint / "config.json"),
        "--sequence-tokens",
        str(sequence_tokens),
        "--steps",
        str(steps),
        "--save-every",
        str(steps),
        "--out",
        str(out_dir),
        "--initial-checkpoint",
        str(checkpoint),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    losses = []
    for line in out.splitlines():
        if '"loss"' in line:
            losses.append(json.loads(line)["loss"])
    return losses


def rust_ring_output(
    shards: list[tuple],
    rank: int,
    block_q: int = 2,
    block_k: int = 3,
    valid_length: int | None = None,
    q_position_start: int | None = None,
) -> list[float]:
    """Run the Rust ringcheck binary and parse the rank's output."""
    spec = {
        "world_size": len(shards),
        "rank": rank,
        "block_q": block_q,
        "block_k": block_k,
        "valid_length": valid_length,
        "q_position_start": q_position_start,
        "q": [{"data": q.flatten().tolist(), "shape": list(q.shape)} for q, _, _ in shards],
        "k": [{"data": k.flatten().tolist(), "shape": list(k.shape)} for _, k, _ in shards],
        "v": [{"data": v.flatten().tolist(), "shape": list(v.shape)} for _, _, v in shards],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name
    try:
        cmd = [
            "cargo",
            "run",
            "--offline",
            "-p",
            "longctx100m-model",
            "--bin",
            "longctx100m-ringcheck",
            "--",
            spec_path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if '"output"' in line:
                return json.loads(line)["output"]
        raise RuntimeError(f"no output in ringcheck: {out}")
    finally:
        Path(spec_path).unlink(missing_ok=True)
def rust_ring_backward(
    shards: list[tuple],
    rank: int,
    grad_out,
    block_q: int = 2,
    block_k: int = 3,
) -> tuple[list[float], list[float], list[float]]:
    """Run the Rust ringcheck binary in backward mode."""
    spec = {
        "world_size": len(shards),
        "rank": rank,
        "block_q": block_q,
        "block_k": block_k,
        "q": [{"data": q.flatten().tolist(), "shape": list(q.shape)} for q, _, _ in shards],
        "k": [{"data": k.flatten().tolist(), "shape": list(k.shape)} for _, k, _ in shards],
        "v": [{"data": v.flatten().tolist(), "shape": list(v.shape)} for _, _, v in shards],
        "grad_out": {"data": grad_out.flatten().tolist(), "shape": list(grad_out.shape)},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name
    try:
        cmd = [
            "cargo",
            "run",
            "--offline",
            "-p",
            "longctx100m-model",
            "--bin",
            "longctx100m-ringcheck",
            "--",
            spec_path,
            "--backward",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if '"dq"' in line:
                payload = json.loads(line)
                return payload["dq"], payload["dk"], payload["dv"]
        raise RuntimeError(f"no backward output in ringcheck: {out}")
    finally:
        Path(spec_path).unlink(missing_ok=True)


def main() -> None:
    config = ModelConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        intermediate_size=12,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rope_scaling=None,
        num_key_value_heads=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checkpoint = build_parity_checkpoint(tmp / "parity-model", config)
        prompt = [1, 2, 3, 4]
        rust = rust_logits(checkpoint, prompt)
        py = python_logits(checkpoint, prompt)
        max_abs = max(abs(a - b) for a, b in zip(rust, py))
        print(f"forward parity: max abs logit diff = {max_abs:.3e}")
        assert max_abs < 1e-4, f"forward mismatch: {rust} vs {py}"

        corpus = tmp / "corpus.bin"
        corpus.write_bytes(struct.pack("<10I", 1, 2, 3, 4, 5, 3, 1, 4, 1, 5))
        rust_losses = rust_train_losses(checkpoint, corpus, 20, 4)
        py_losses = python_train_losses(checkpoint, corpus, 20, 4)
        print(f"rust train losses: {rust_losses}")
        print(f"python train losses: {py_losses}")
        for rs, ps in zip(rust_losses, py_losses):
            assert abs(rs - ps) < 1e-3, f"train loss mismatch: {rs} vs {ps}"
        print("train parity: OK")

        # Resume: train 10 steps, then resume for 10 more; the resumed losses
        # must match the continuous 20-step run's second half.
        out_dir = tmp / "resume-runs"
        cmd = [
            "cargo",
            "run",
            "--offline",
            "-p",
            "longctx100m-model",
            "--bin",
            "longctx100m-train",
            "--",
            "--tokens",
            str(corpus),
            "--config",
            str(checkpoint / "config.json"),
            "--sequence-tokens",
            "4",
            "--steps",
            "10",
            "--save-every",
            "5",
            "--out",
            str(out_dir),
            "--initial-checkpoint",
            str(checkpoint),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        resumed = rust_train_losses(checkpoint, corpus, 10, 4, resume_dir=out_dir)
        print(f"resumed losses: {resumed}")
        # The resumed run reports step 10; it must match the continuous run's
        # step-10 loss exactly (optimizer moments restored).
        assert len(resumed) >= 1, "resumed run produced no losses"
        assert (
            abs(resumed[0] - rust_losses[1]) < 1e-4
        ), f"resume mismatch: {resumed[0]} vs {rust_losses[1]}"
        print("resume parity: OK")

        # Ring attention: Rust ring output must match the Python dense oracle
        # for both ranks of a 2-shard split (single-rank case is covered by the
        # Rust unit test; Python ring_causal_attention is verified against the
        # dense oracle in tests/test_ring_attention.py).
        torch.manual_seed(7)
        world = 2
        t, h, d = 4, 2, 4
        shards = []
        for _ in range(world):
            shards.append(
                (
                    torch.randn(1, t, h, d),
                    torch.randn(1, t, h, d),
                    torch.randn(1, t, h, d),
                )
            )
        q_full = torch.cat([s[0] for s in shards], 1)
        k_full = torch.cat([s[1] for s in shards], 1)
        v_full = torch.cat([s[2] for s in shards], 1)
        from longctx100m.ring_attention import reference_causal_attention

        ref = reference_causal_attention(q_full, k_full, v_full, causal=True, valid_length=None)
        for rank in range(world):
            rust_out = rust_ring_output(shards, rank)
            py_slice = ref[0, rank * t : (rank + 1) * t].flatten().tolist()
            max_abs = max(abs(a - b) for a, b in zip(rust_out, py_slice))
            print(f"ring parity rank {rank}: max abs diff = {max_abs:.3e}")
            assert max_abs < 1e-4, f"ring mismatch rank {rank}: {rust_out} vs {py_slice}"
        print("ring parity: OK")

        # Ring backward: Rust gradients must match Python dense autograd.
        q_full.requires_grad_(True)
        k_full.requires_grad_(True)
        v_full.requires_grad_(True)
        out = reference_causal_attention(q_full, k_full, v_full, causal=True, valid_length=None)
        loss = out.sum()
        loss.backward()
        grad_out = torch.ones_like(out)
        for rank in range(world):
            rq, rk, rv = rust_ring_backward(shards, rank, grad_out)
            pq = q_full.grad[0, rank * t : (rank + 1) * t].flatten().tolist()
            pk = k_full.grad[0, rank * t : (rank + 1) * t].flatten().tolist()
            pv = v_full.grad[0, rank * t : (rank + 1) * t].flatten().tolist()
            for name, r, p in (("dq", rq, pq), ("dk", rk, pk), ("dv", rv, pv)):
                max_abs = max(abs(a - b) for a, b in zip(r, p))
                print(f"ring backward rank {rank} {name}: max abs diff = {max_abs:.3e}")
                assert max_abs < 1e-4, f"ring backward mismatch rank {rank} {name}"
        print("ring backward parity: OK")

        # Inference token output: Rust greedy continuation must match Python
        # greedy argmax, and --token-output must write the generated ids as
        # little-endian uint32.
        gen_out = tmp / "generated.bin"
        cmd = [
            "cargo",
            "run",
            "--offline",
            "-p",
            "longctx100m-model",
            "--bin",
            "longctx100m-infer",
            "--",
            "--checkpoint",
            str(checkpoint),
            "--prompt-tokens",
            "1,2,3,4",
            "--generate",
            "3",
            "--token-output",
            str(gen_out),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rust_ids = json.loads(out.splitlines()[-1])["output"]
        # Python greedy continuation from the same checkpoint.
        py_model = LongContextDecoder(config).eval()
        from safetensors.torch import load_file

        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        if config.tie_word_embeddings:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        py_model.load_state_dict(state, strict=True)
        py_ids = [1, 2, 3, 4]
        with torch.no_grad():
            for _ in range(3):
                logits = py_model(torch.tensor([py_ids]))
                py_ids.append(int(logits[0, -1].argmax()))
        assert rust_ids == py_ids, f"infer mismatch: {rust_ids} vs {py_ids}"
        assert gen_out.read_bytes() == struct.pack("<3I", *py_ids[-3:])
        print("infer parity: OK")


if __name__ == "__main__":
    main()
