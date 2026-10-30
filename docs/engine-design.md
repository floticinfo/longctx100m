# Long-context Engine: model, kernel and cluster

## Architecture

- Decoder-only Transformer with tied/untied token embeddings, RMSNorm, rotary positions, QKV projections and SwiGLU MLP (`src/longctx100m/model.py`). Default config is a *shape template*, not pretrained weights.
- Every rank owns an equal contiguous token interval. A layer computes local Q/K/V; the ring rotates K/V shards in peer-to-peer transfers. Online log-sum-exp accumulates exact causal dense-attention output over each KV tile; only local input/output/gradients and bounded score tiles are resident (`ring_attention.py`).
- Custom backward recomputes attention probabilities from saved output/logsumexp statistics. Q/K/V derivatives are compared to a full dense reference under a two-process context-parallel Gloo run. Parameters are updated identically after each rank computes its disjoint local target loss; training launcher synchronizes summed gradients.
- Triton single-device tiled causal attention is an optional kernel path (`triton_attention.py`) intended for Linux/CUDA. The context-parallel reference ring currently uses PyTorch ATen matmuls over bounded tiles; a fused distributed Triton kernel (overlap communication with compute, CUDA graphs, topology-aware transport) remains a performance optimization.

## Complexity / capacity gates

The ring layer preserves **exact dense causal attention semantics** for all unpadded positions. It does not reduce O(N²) multiply-add work. For N=100M, attention arithmetic remains roughly 10,000x the 1M-token case. Ring spreads KV state and arithmetic over ranks; it does not make work cheap. `longctx100m-capacity` estimates KV cache lower bound and score-tile workspace; it excludes weights, activations, optimizer state, transport buffers and temporary framework memory. With 80-GiB devices at 80% usable memory, 24 layers, 16 KV heads, head dim 128, BF16 and 128 context ranks, the KV-only cache bound is about 153.6 GB/rank, so the report correctly flags that this candidate does not fit one GPU/rank. A simple aggregate-KV-only lower bound needs 287 ranks to fit shards on 80-GiB GPUs, but it omits all model weights, non-KV memory and topology constraints; it is not a deployment recommendation or tested topology. The current runtime binds one CUDA device to each context rank and does not have model/tensor parallel inside a rank. Target cluster memory/throughput must be profiled. Gloo is correctness-test only.

This implementation does not include a pretrained 100M model, tokenizer, production multi-node CUDA/Triton validation, quality/recall benchmark at 100M, or proven feasible throughput/cost. A random decoder initialized from this code is not a capable language model. Build a training tokenizer/vocabulary and train/extend weights over long sequences before user-facing inference.

## Install and test

Python package optional extras:

```bash
python -m pip install -e '.[engine]'              # PyTorch model and reference path
python -m pip install -e '.[kernel]'              # Linux/CUDA Triton kernel development
python -m pytest tests/test_ring_attention.py -q
longctx100m-capacity --tokens 100000000 --output-tokens 2048 --layers 24 --query-heads 16 --kv-heads 16 --head-dim 128 --world-size 128 --local-shard-tokens 1000000 --gpu-memory-gib 80
```

The tests use tiny inputs to compare forward and Q/K/V gradients against dense attention, run two-process context-parallel Gloo backward and greedy-token parity, exercise model/training checkpoint save-resume, verify GQA projections against a dense reference, and test exact state-dict import validation. They prove basic math/checkpoint integration, not 100M memory fit or performance. The Triton test is skipped unless Triton and CUDA are available.

## Train token-id corpus

Prepare one flat raw `uint32` token-ID corpus and a JSON config. The corpus must use the exact model tokenizer/vocabulary and remain on a shared filesystem visible to all ranks. Training memory-maps the flat uint32 file and copies only a selected packed row to device per step. Example config:

```json
{"vocab_size": 32000, "hidden_size": 2048, "num_layers": 24, "num_heads": 16, "max_position_embeddings": 100002048, "rope_theta": 1000000.0, "intermediate_size": 5632}
```

Launch from a Linux multi-GPU/node environment with `torchrun`, NCCL, PyTorch CUDA and the CUDA attention/runtime stack installed:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$RANK \
  --master-addr=$MASTER_ADDR --master-port=29500 \
  -m longctx100m.train --tokens /shared/train.uint32 --config /shared/model.json \
  --sequence-tokens 32768 --steps 100000 --out /shared/checkpoints
```

The example is a training entry point, not a recipe that finishes training a 100M-context model. Each raw `uint32` record is one packed training row with `sequence_tokens + 1` IDs: model inputs are all but the final ID and next-token labels are all but the first. Sequence rows are padded to equal context-rank shards; padded token positions are masked and their labels ignored. Choose sequence/model size and batch after profiling; position limit alone is not evidence that weights learned long dependencies.

Training checkpointing writes model, optimizer, configuration, step and CPU RNG state to a temporary file and atomically renames it. Continue a run with `--resume /shared/checkpoints/step-<n>.pt`; the training script verifies config equality before loading. Because token-row selection is deterministic by step, the saved step resumes the next record; exact CUDA RNG and dataloader state are not yet checkpointed.

`longctx100m-capacity` is a KV-only memory/work lower-bound estimator, not profiling. With 100M+2,048 tokens, 24 layers, 16 KV heads, D=128, BF16, and 80-GiB GPUs at 80% usable memory, a 128-rank candidate's KV-only lower bound is about 153.6 GB/rank and does not fit one GPU/rank. The current runtime uses one CUDA device per context rank and has no multi-GPU-per-rank model parallelism. `longctx100m-capacity` estimates a minimum of 287 ranks for KV-only one-GPU-per-rank placement, but that is not a benchmarked topology and excludes model weights, MLP, optimizer and communication buffers. The inference preflight rejects topology with fewer than 100M tokens plus output capacity; training operates on the configured short sequence length and does not train a 100M-length sequence in its default shape. To pursue the 100M request, first target a measured cluster layout that fits each rank's KV shard, then build/model-parallel support as needed, and run actual 100M training/prefill benchmarks.


## Distributed generation launcher

`longctx100m-infer` loads a distributed checkpoint, reads each rank's input token range directly from a raw uint32 file, performs a full-context prefill, and repeatedly chooses the next greedy token on the rank that owns the global tail. That token is broadcast and appended at the correct shard-local position on every rank; a Gloo parity test compares the first generated token with single-process dense attention, including a rank-boundary tail.

The launcher reserves `--local-shard-tokens` tokens per rank (input plus output headroom), rejects any input shorter than 100M tokens, and rejects an input/output budget larger than shard capacity or the model limit. Only deterministic greedy decoding is supported. Each decode step recomputes the full prompt and re-circulates K/V instead of incrementally caching layer KV. This is a correctness reference, not a latency-efficient/production long-context service. An incremental distributed KV cache remains a performance milestone.



## Pretrained Llama weight import

A HF `LlamaForCausalLM` checkpoint can be imported with the optional `pretrained` extra:

```bash
python -m pip install -e '.[engine,pretrained]'
longctx100m-import-llama --model HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --max-position-embeddings 100002048 --out /shared/smollm-base
```

Only dense, non-sliding-window Llama attention with supported projection shapes, SiLU gated MLP and default/linear/Llama-3 RoPE profiles is accepted. The importer reads `safetensors`, validates every model tensor key/shape and tokenizer vocabulary, then writes `model.safetensors`, `config.json`, `manifest.json` and the tokenizer. Pin the immutable repo revision. Use `--initial-checkpoint /shared/smollm-base` for continual pretraining (the training `--config` must match exactly); its tokenizer directory is copied to training output and checkpoint resume keeps the source association.

**Import does not extend training context.** The tested public SmolLM2-135M source has 8,192 trained positions; setting 100,002,048 target positions only configures RoPE/ring address range. Its imported manifest records `pretrained_for_100m_context: false`. Long-context training and quality tests are still needed. A smoke run successfully imported the 135M model, loaded 272 tensors (162,826,560 parameters, BF16, 49,152-token vocabulary), ran the imported model with finite output logits, and matched the HF eager reference's greedy token on a short prompt. This is a loader/short-forward correctness test, not 100M-context evidence.

## Prototype checkpoint

This is a working research engine skeleton and exact ring attention reference, not a released 100M model product. The CPU Gloo tests verify collectives and custom backward on short contexts; a Triton test is marked to run only when a Linux/CUDA Triton runtime is present. A pretrained tokenizer/base checkpoint import workflow is available for compatible Llama checkpoints, but there is no 100M-trained weight file, quality result or profile for 100M throughput. Distributed generation still retains/recomputes the full input on every decode step instead of incrementally caching KV. These remain explicit next engineering gates.
