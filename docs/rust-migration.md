# Rust-first migration plan and compatibility boundary

## State of this slice

The Rust workspace now includes an opt-in `longctx100m-gateway`, a Rust `longctx100m-import-llama` checkpoint importer, and a Rust `longctx100m-model` model/training crate (candle-core/candle-nn), alongside `longctx100m-rs` and the pure core. The gateway retains the compatible browser/API surface and backend protocol-v1 preflight/inference/attestation. The importer fetches a 40-character pinned HF revision, validates supported dense Llama config, tensor inventory/shape/dtype and fast-tokenizer vocabulary, and emits the Python-readable `model.safetensors`, `config.json`, `manifest.json`, and tokenizer directory using staged/atomic directory creation. The model crate implements the exact dense Llama decoder (embedding, RMSNorm, RoPE, GQA attention, SwiGLU, tied LM head), checkpoint loading from importer output, masked cross-entropy loss, gradient clipping, AdamW training with resume, and the exact causal ring-attention forward **and backward** (`ring_attention` module) whose arithmetic matches the Python distributed implementation; plus `longctx100m-train`, `longctx100m-infer`, and `longctx100m-ringcheck` binaries. Windows/MSVC workspace tests cover 6 core, 3 gateway, 5 importer, and 6 model unit tests (Python: 40 passed, 1 skipped). Rust and Python importers were additionally run against the cached pinned `HuggingFaceTB/SmolLM2-135M` revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`: all 272 exported tensor keys, shapes, dtypes, and bytes matched exactly; `manifest.json` was identical; `config.json` was semantically identical; tokenizer vocab/special/added tokens and sample encodings matched. The only differences are tokenizers-library serialization conventions (Rust copies the HF `tokenizer.json` verbatim with string merges; Python `save_pretrained` re-serializes with list merges and an `extra_special_tokens` key) — behaviorally equivalent. `scripts/verify_model_parity.py` verifies Rust/Python forward logits (< 1.5e-6), training losses (< 1e-3), resume continuity (< 1e-4), and ring-attention outputs for a 2-rank split (< 1.7e-7). Both Rust commands remain opt-in; Python remains rollback until a real CUDA/NCCL transport, Triton, and Linux CI are validated.

## Mapping

| Current path | Rust owner | First-slice status | Keep/replace rule |
|---|---|---|---|
| `longctx100m-capacity` → `capacity.estimate` | `longctx100m-core::estimate_capacity`; `longctx100m-rs capacity` | Golden fixture and CLI parity | Keep Python command until downstream users migrate |
| `BackendCapabilities.from_payload/require_context` | `longctx100m-core::BackendCapabilities` | Contract fixture and capability preflight in Rust gateway | Keep Python behavior available until downstream callers opt in |
| `InferenceResult.verify` + digest/upload response handling | `longctx100m-gateway` | Rust gateway implemented; fixture-based protocol/API parity; live backend not yet exercised | Use Rust gateway only as an opt-in compatible launcher; preserve Python rollback |
| HF tokenizer + safetensors + Llama import | `longctx100m-import-llama` | Synthetic fixture tests + exact local SmolLM2-135M pinned-revision parity (272/272 tensors, Python strict loader and short forward pass) | Opt-in Rust import; keep Python importer as rollback until broader model coverage |
| Decoder/GQA/RoPE/model checkpoint ops | `longctx100m-model` (candle) | Rust forward/backward parity with Python on shared checkpoint; `longctx100m-train`/`longctx100m-infer` | Keep Python implementation and state-dict format until CUDA/NCCL parity |
| Ring P2P backward, CUDA/NCCL, Triton, distributed KV decode | CUDA/Rust bindings or native kernels | Ring attention forward + backward ported and parity-verified (in-process ring); P2P/NCCL transport and Triton not ported; no CUDA in current environment | Do not stub/replace with CPU paths |

## Compatibility and rollout

1. **Expand (in progress):** Rust workspace, pure core/CLI, opt-in HTTP gateway, opt-in HF Llama importer, and Rust model/training runtime (candle) are present. Python wheel and entrypoints remain available. The importer writes the existing Python checkpoint directory format; the Rust model loads the same directory and matches Python forward/train losses; Python train/infer readers have not been retired.
2. **Verify (Windows/MSVC local):** workspace checks/tests include 6 core, 3 gateway, 5 importer and 6 model tests; Rust capacity output matches the Python golden fixture. The pinned SmolLM2 comparison verifies one real cached checkpoint conversion path. `scripts/verify_model_parity.py` verifies Rust/Python forward logits (< 1.5e-6), training losses (< 1e-3), resume continuity, and 2-rank ring-attention forward (< 1.7e-7) and backward gradients (< 3.5e-7) on a shared checkpoint. Broader HF architectures, Linux CI, live protocol backend, and a real CUDA/NCCL transport remain pending.
3. **Migrate callers:** change scripts/operator docs to use Rust only after explicit parity, error contract, and distribution review.
4. **Contract/remove Python:** not authorized in this migration slice. Require real HF importer parity, equivalent model/trainer/inference runtime, CUDA/NCCL and target-GPU evidence, rollback/observability gates, and explicit approval before removal.

**Rollback now:** use the existing Python command (`longctx100m-capacity`, gateway/train/infer scripts); no files or model artifacts are converted or deleted. Rust commands are opt-in. Cargo dependency failures must not break Python installation.

## Opt-in Rust HTTP gateway

The Rust gateway serves the same browser page and client paths as Python and sends the same protocol-v1 requests to an existing backend/adapter. Python remains unchanged and is the rollback command. To run the Rust version on Windows PowerShell:

```powershell
$env:LONGCTX_BACKEND_URL = "http://127.0.0.1:9000"
$env:LONGCTX_API_KEY = "<service-key>" # required if LONGCTX_HOST is not localhost
cargo run --offline -p longctx100m-gateway
```

`LONGCTX_HOST`, `LONGCTX_PORT`, `LONGCTX_BACKEND_API_KEY`, `LONGCTX_REQUIRED_INPUT_TOKENS`, `LONGCTX_MAX_UPLOAD_BYTES`, `LONGCTX_REQUEST_TIMEOUT_SECONDS`, and `LONGCTX_TEMP_DIR` are supported. The gateway intentionally does not implement a tokenizer or a model: `/tokenize` and `/infer` must be supplied by a protocol-v1 backend. Keep using `longctx100m-serve` to roll back.

## Rust checkpoint importer

The Rust importer currently supports public/authorized Hub models that provide a fast `tokenizer.json` and root-level dense Llama safetensors shards. It rejects mutable/non-40-hex revisions, unsupported model families/features, incomplete/extra/duplicate tensors, shape/dtype mismatch, tokenizer vocabulary mismatch, and an existing output path. It stages the full directory beside the destination and atomically renames it on success. Use the same flags as the Python importer:

```powershell
cargo run --offline -p longctx100m-import-llama -- --model org/model --revision <40-character-commit-sha> --max-position-embeddings 100002048 --out C:\models\llama-base
```

The resulting directory has the Python-consumed filenames and config schema, and Rust emits/reads standard safetensors. The existing Python importer remains supported. Local Rust tests use synthetic checkpoint tensors; the pinned SmolLM2 comparison above validated one real model import (tensor bytes/keys, tokenizer behavior, manifest fields). Broader HF architectures and Linux CI still need validation. This importer does **not** train/extend a model or implement forward/backward.

## Rust model and training runtime

`longctx100m-model` is a candle-core/candle-nn crate that mirrors the Python decoder exactly: embedding, RMSNorm, RoPE (linear/llama3 scaling), GQA causal attention, SwiGLU, tied or untied LM head, and Transformers-compatible state-dict naming. It loads importer checkpoint directories directly (`config.json` + `model.safetensors`) and reconstructs the tied LM head from the embedding table exactly like the Python loader.

Two binaries are included:

```powershell
# Train from a flat uint32 token corpus, initializing from an imported checkpoint
cargo run --offline -p longctx100m-model --bin longctx100m-train -- --tokens corpus.bin --config C:\models\llama-base\config.json --sequence-tokens 32768 --steps 1000 --lr 3e-4 --save-every 100 --out runs\longctx --initial-checkpoint C:\models\llama-base

# Resume interrupted training from the same --out directory (weights + optimizer moments)
cargo run --offline -p longctx100m-model --bin longctx100m-train -- --tokens corpus.bin --config C:\models\llama-base\config.json --steps 1000 --out runs\longctx --resume runs\longctx

# Greedy continuation or logits dump from an imported checkpoint
cargo run --offline -p longctx100m-model --bin longctx100m-infer -- --checkpoint C:\models\llama-base --prompt-tokens 1,2,3,4 --generate 8
cargo run --offline -p longctx100m-model --bin longctx100m-infer -- --checkpoint C:\models\llama-base --prompt-tokens 1,2,3,4 --dump-logits

# Token-file inference mirroring distributed_infer's single-rank path: validate
# the uint32 file (optional 100M-token requirement), greedy-decode, and write
# the generated ids as little-endian uint32.
cargo run --offline -p longctx100m-model --bin longctx100m-infer -- --checkpoint C:\models\llama-base --input-tokens request.bin --generate 128 --token-output generated.bin --require-100m
```

The training loop mirrors `longctx100m.train` for world_size=1: packed `sequence_tokens+1` records, lazy per-row token-ID validation, cross-entropy with `ignore_index=-100` padding semantics divided by the sequence length, global L2 gradient clipping at 1.0, and AdamW with PyTorch defaults (beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01). Checkpoints write `model.safetensors`, `optimizer.safetensors` (AdamW first/second moments), `resume.json` (step, lr, hyperparameters), and `config.json`; `--resume` restores all of them so continued training matches an uninterrupted run exactly.

The `ring_attention` module ports the exact online-softmax ring-attention forward and backward (bounded KV workspace, causal + valid-length masking, fp32 accumulation, two-pass gradient circulation) and matches the Python distributed implementation's arithmetic: unit tests verify single-rank and 2-rank shard splits against the dense reference/autograd, and `scripts/verify_model_parity.py` checks a 2-rank split against the Python oracle for forward (< 1.7e-7) and backward gradients (< 3.5e-7). The P2P/NCCL transport is isolated behind the shard-index interface so a CUDA/NCCL backend can be dropped in later; Triton and Python `.pt` resume-format compatibility are not yet implemented in Rust.

`longctx100m-infer` also mirrors the single-rank path of `longctx100m.distributed_infer`: uint32 token-file validation (byte alignment, optional 100M-token requirement, vocab range), greedy argmax continuation, and `--token-output` writing generated ids as little-endian uint32. `scripts/verify_model_parity.py` asserts the Rust greedy continuation equals Python argmax decoding and the token file bytes match. The distributed (CUDA/NCCL) multi-rank path of `distributed_infer` still requires a GPU transport.

## Verification commands

```powershell
cargo fmt --all -- --check
cargo test --workspace --offline
cargo build --workspace --release --offline
cargo run --offline -p longctx100m-cli -- migration-status
cargo run --offline -p longctx100m-cli -- capacity --tokens 100000000 --output-tokens 2048 --layers 24 --query-heads 16 --kv-heads 16 --head-dim 128 --world-size 128 --local-shard-tokens 1000000 --gpu-memory-gib 80
cargo run --offline -p longctx100m-cli -- validate-capabilities --input .\capabilities.json --required-input-tokens 100000000 --output-tokens 2048
python scripts/verify_model_parity.py
```

The estimator remains a KV-only lower bound, not a benchmark or deployment recommendation. Rust now has an opt-in HF Llama importer plus a CPU model/training runtime with verified forward/backward parity against Python; distributed GPU (CUDA/NCCL) execution, Triton kernels, and Linux CI are the remaining gaps.
