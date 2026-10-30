"""Decoder-only Transformer built around context-parallel ring attention."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn

from .ring_attention import ring_causal_attention

try:
    from .triton_attention import tiled_causal_attention
except (ImportError, RuntimeError):
    tiled_causal_attention = None


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    max_position_embeddings: int = 100_002_048
    rope_theta: float = 10_000.0
    intermediate_size: int = 5632
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: dict | None = None
    num_key_value_heads: int | None = None

    @classmethod
    def from_hf_llama_config(cls, config, *, max_position_embeddings=None):
        """Map a plain Transformers LlamaConfig to this exact Llama-family profile."""
        rope = getattr(config, "rope_parameters", None)
        if rope is None:
            rope = getattr(config, "rope_scaling", None)
        if rope:
            rope = dict(rope)
            if rope.get("rope_type", rope.get("type", "default")) == "default":
                rope = None
        result = cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_hidden_layers),
            num_heads=int(config.num_attention_heads),
            max_position_embeddings=int(
                max_position_embeddings or config.max_position_embeddings
            ),
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            intermediate_size=int(config.intermediate_size),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            hidden_act=str(getattr(config, "hidden_act", "silu")),
            attention_bias=bool(getattr(config, "attention_bias", False)),
            mlp_bias=bool(getattr(config, "mlp_bias", False)),
            rope_scaling=rope,
            num_key_value_heads=int(
                getattr(config, "num_key_value_heads", None)
                or config.num_attention_heads
            ),
        )
        result.validate()
        return result

    def validate(self):
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type", "default")
            )
            if rope_type not in {"default", "linear", "llama3"}:
                raise ValueError(f"unsupported RoPE scaling profile {rope_type!r}")
            if rope_type == "llama3" and not {
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            }.issubset(self.rope_scaling):
                raise ValueError("incomplete llama3 rope scaling profile")
            if rope_type == "linear" and float(self.rope_scaling.get("factor", 0)) <= 0:
                raise ValueError("linear RoPE factor must be positive")
        if (
            min(
                self.vocab_size,
                self.hidden_size,
                self.num_layers,
                self.num_heads,
                self.max_position_embeddings,
                self.intermediate_size,
            )
            <= 0
        ):
            raise ValueError("model dimensions must be positive")
        kv_heads = (
            self.num_heads
            if self.num_key_value_heads is None
            else self.num_key_value_heads
        )
        if kv_heads <= 0 or self.num_heads % kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by positive num_key_value_heads"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must divide evenly into attention heads")
        if self.hidden_size // self.num_heads % 2:
            raise ValueError("head dimension must be even for rotary embeddings")


def _rms_norm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


def _apply_rope(x, positions, theta, *, rope_scaling=None):
    """x: [B,T,H,D], positions: [T]."""
    dim = x.shape[-1]
    if rope_scaling is not None:
        partial = float(rope_scaling.get("partial_rotary_factor", 1.0))
        if partial != 1.0:
            raise ValueError(
                "partial rotary factors are not supported by the current Llama profile"
            )
    inv = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    )
    attention_factor = 1.0
    if rope_scaling is not None:
        rope_theta = float(rope_scaling.get("rope_theta", theta))
        if rope_theta != theta:
            raise ValueError("RoPE theta differs between model and scaling config")
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        factor = float(rope_scaling.get("factor", 1.0))
        if factor <= 0:
            raise ValueError("rope scaling factor must be positive")
        if rope_type == "linear":
            positions = positions / factor
        elif rope_type == "llama3":
            required = (
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            )
            if any(key not in rope_scaling for key in required):
                raise ValueError(
                    "llama3 RoPE needs low/high frequency factors and original context length"
                )
            low = float(rope_scaling["low_freq_factor"])
            high = float(rope_scaling["high_freq_factor"])
            original = int(rope_scaling["original_max_position_embeddings"])
            if low <= 0 or high <= low or original <= 0:
                raise ValueError("invalid llama3 RoPE scaling values")
            wavelength = 2 * torch.pi / inv
            low_wavelen = original / low
            high_wavelen = original / high
            scaled = inv / factor
            smooth = (original / wavelength - low) / (high - low)
            low_scaled = inv / factor
            mid = (1 - smooth) * low_scaled + smooth * inv
            inv = torch.where(wavelength > low_wavelen, scaled, inv)
            in_mid = (wavelength >= high_wavelen) & (wavelength <= low_wavelen)
            inv = torch.where(in_mid, mid, inv)
            attention_factor = float(rope_scaling.get("attention_factor", 1.0))
            if attention_factor <= 0:
                raise ValueError("llama3 attention_factor must be positive")
        elif rope_type not in {"default", None}:
            raise ValueError(f"unsupported RoPE type {rope_type!r}")
    freq = torch.outer(positions.float(), inv)
    # Llama-family RoPE duplicates half-dimensional frequencies and rotates
    # split halves (rotate_half), not adjacent even/odd pairs.
    freq = torch.cat((freq, freq), dim=-1)
    cos = freq.cos()[None, :, None, :] * attention_factor
    sin = freq.sin()[None, :, None, :] * attention_factor
    first, second = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat(
        (
            first * cos[..., : dim // 2] - second * sin[..., : dim // 2],
            second * cos[..., : dim // 2] + first * sin[..., : dim // 2],
        ),
        dim=-1,
    )


class RingSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.kv_heads = (
            config.num_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        x,
        *,
        group=None,
        valid_length=None,
        position_offset=None,
        block_q=64,
        block_k=128,
    ):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.config.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim)
        rank = (
            dist.get_rank(group) if dist.is_initialized() and group is not None else 0
        )
        base = position_offset if position_offset is not None else rank * t
        pos = base + torch.arange(t, device=x.device)
        if position_offset is not None:
            q_start = position_offset
        else:
            q_start = rank * t
        q = _apply_rope(
            q, pos, self.config.rope_theta, rope_scaling=self.config.rope_scaling
        )
        k = _apply_rope(
            k, pos, self.config.rope_theta, rope_scaling=self.config.rope_scaling
        )
        if self.kv_heads != self.config.num_heads:
            repeat = self.config.num_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        if dist.is_initialized():
            active_group = group if group is not None else dist.group.WORLD
            out = ring_causal_attention(
                q,
                k,
                v,
                group=active_group,
                block_q=block_q,
                block_k=block_k,
                valid_length=valid_length,
                q_position_start=q_start,
            )
        else:
            if valid_length is not None and valid_length != t:
                raise ValueError(
                    "padded full-sequence execution requires distributed ring attention"
                )
            if position_offset not in (None, 0):
                raise ValueError(
                    "nonzero global position shards require context-parallel attention"
                )
            use_tiled_kernel = (
                tiled_causal_attention is not None
                and x.is_cuda
                and b == 1
                and not self.training
            )
            if t > 2048 and not use_tiled_kernel:
                raise ValueError(
                    "large sequences require tiled CUDA attention in eval mode or context-parallel ring attention"
                )
            if use_tiled_kernel:
                out = tiled_causal_attention(
                    q.squeeze(0),
                    k.squeeze(0),
                    v.squeeze(0),
                    block_q=block_q,
                    block_k=block_k,
                    position_offset=position_offset or 0,
                    causal_limit=valid_length,
                ).unsqueeze(0)
            else:
                scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * (
                    self.head_dim**-0.5
                )
                mask = torch.triu(
                    torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(mask[None, None], float("-inf"))
                out = torch.einsum("bhts,bshd->bthd", scores.softmax(-1).to(v.dtype), v)
        return self.o_proj(out.reshape(b, t, c))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        if config.hidden_act != "silu":
            raise ValueError(
                "this engine currently supports only Llama-family SiLU gated MLP"
            )

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = RingSelfAttention(config)
        self.mlp = SwiGLU(config)

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x), **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class LongContextDecoder(nn.Module):
    """Decoder with Transformers LlamaForCausalLM state-dict naming."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        object.__setattr__(self, "embed", self.model.embed_tokens)
        self.model.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        *,
        group=None,
        valid_length=None,
        position_offset=None,
        block_q=64,
        block_k=128,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, local_tokens]")
        global_start = position_offset if position_offset is not None else 0
        live_tokens = (
            input_ids.shape[1]
            if valid_length is None
            else max(0, min(input_ids.shape[1], valid_length - global_start))
        )
        if global_start + live_tokens > self.config.max_position_embeddings:
            raise ValueError("input shard exceeds configured model context")
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(
                x,
                group=group,
                valid_length=valid_length,
                position_offset=position_offset,
                block_q=block_q,
                block_k=block_k,
            )
        x = self.model.norm(x)
        return self.lm_head(x)

    def config_dict(self):
        return asdict(self.config)

    def export_pretrained_state_dict(self):
        """CPU-safe HF-style checkpoint tensors, without module-alias duplicate keys."""
        state = {name: t.detach().cpu() for name, t in self.state_dict().items()}
        # Backward-compatible module alias is intentionally absent from HF state dict.
        state.pop("embed.weight", None)
        return state


def safe_load_file(filename, device="cpu"):
    """Load safetensors lazily; dependency is required only for this import path."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Install importer dependencies with pip install -e '.[pretrained]'"
        ) from exc
    return load_file(str(filename), device=str(device))


def import_hf_llama_state_dict(model, state_dict):
    """Validate and copy a HF LlamaForCausalLM state dict into this model.

    HF may omit lm_head for tied embeddings. All other keys, tensor types and
    shapes are validated before mutating any parameter.
    """
    state_dict = dict(state_dict)
    tied = model.config.tie_word_embeddings
    if (
        tied
        and "lm_head.weight" not in state_dict
        and "model.embed_tokens.weight" in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    expected = model.state_dict()
    # Ignore the legacy Python compatibility alias when ingesting old .pt saves.
    if "embed.weight" in state_dict and "embed.weight" not in expected:
        if (
            state_dict["embed.weight"].shape
            != expected["model.embed_tokens.weight"].shape
        ):
            raise ValueError(
                "legacy embed.weight shape does not match model embeddings"
            )
        state_dict.pop("embed.weight")
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatches = []
    for key in sorted(set(expected) & set(state_dict)):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            mismatches.append(f"{key}: not a tensor")
        elif value.shape != expected[key].shape:
            mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(expected[key].shape)}"
            )
    if missing or unexpected or mismatches:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        if mismatches:
            details.append("shape_mismatch=" + "; ".join(mismatches[:8]))
        raise ValueError(
            "Llama checkpoint is not exactly compatible; " + " | ".join(details)
        )
    converted = {
        key: value.detach().to(device=expected[key].device, dtype=expected[key].dtype)
        for key, value in state_dict.items()
    }
    model.load_state_dict(converted, strict=True)
    return {
        "loaded_tensors": len(converted),
        "parameters": sum(v.numel() for v in converted.values()),
    }


def load_hf_llama_checkpoint(
    model_id,
    *,
    revision=None,
    cache_dir=None,
    max_position_embeddings=None,
    trust_remote_code=False,
):
    """Load a public/authorized HF LlamaForCausalLM; never adapts non-Llama architectures."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install checkpoint importer dependencies with pip install -e '.[pretrained]'"
        ) from exc
    config_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "token": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }
    config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(
            f"checkpoint model_type={getattr(config, 'model_type', None)!r}; only plain Llama is supported"
        )
    target_config = ModelConfig.from_hf_llama_config(
        config, max_position_embeddings=max_position_embeddings
    )
    target_config.validate()
    if (
        bool(getattr(config, "attention_bias", False)) != target_config.attention_bias
        or bool(getattr(config, "mlp_bias", False)) != target_config.mlp_bias
    ):
        raise ValueError("Llama projection bias configuration mismatch")
    if getattr(config, "num_experts", None) or getattr(
        config, "n_routed_experts", None
    ):
        raise ValueError(
            "MoE Llama-family checkpoint is not supported by this dense decoder"
        )
    if getattr(config, "head_dim", None) and int(config.head_dim) != int(
        config.hidden_size // config.num_attention_heads
    ):
        raise ValueError("nonstandard Llama head_dim is not supported by this loader")
    sliding = getattr(config, "sliding_window", None)
    if sliding not in (None, 0):
        raise ValueError(
            "sliding-window Llama checkpoint is not exact full causal attention; do not import as dense full attention"
        )
    custom = getattr(config, "attention_chunk_size", None)
    if custom:
        raise ValueError(
            "chunked/special attention Llama checkpoint requires its own validated profile"
        )
    files = list_repo_files(model_id, revision=revision, token=None)
    safe_files = sorted(name for name in files if name.endswith(".safetensors"))
    if not safe_files:
        raise ValueError(
            "checkpoint has no safetensors weights; pickle .bin checkpoints are not accepted"
        )
    source_dtype = getattr(config, "torch_dtype", None)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if isinstance(source_dtype, str):
        source_dtype = dtype_map.get(source_dtype.replace("torch.", ""))
    if source_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        source_dtype = torch.float32
    target = LongContextDecoder(target_config).to(dtype=source_dtype)
    if target_config.tie_word_embeddings and "lm_head.weight" in target.state_dict():
        # HF tied-weight checkpoints commonly omit the redundant LM head tensor.
        target.lm_head.weight = target.model.embed_tokens.weight
    expected = target.state_dict()
    mapped = {}
    duplicate = []
    mismatched = []
    unexpected = []
    # Index safetensors headers first. Load a tensor only after its key/shape is known,
    # then immediately copy it to the target device and release the source shard.
    for name in safe_files:
        local = hf_hub_download(
            model_id, filename=name, revision=revision, cache_dir=cache_dir, token=None
        )
        with safe_open(local, framework="pt", device="cpu") as sf:
            for key in sf.keys():  # noqa: SIM118
                if key in mapped:
                    duplicate.append(key)
                    continue
                if key not in expected:
                    unexpected.append(key)
                    continue
                if tuple(sf.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    mismatched.append(key)
                    continue
                expected[key].copy_(
                    sf.get_tensor(key).to(
                        dtype=expected[key].dtype, device=expected[key].device
                    )
                )
                mapped[key] = True
    missing = sorted(set(expected) - set(mapped))
    if "model.embed_tokens.weight" in mapped and "embed.weight" in missing:
        missing.remove("embed.weight")
    if (
        target_config.tie_word_embeddings
        and "lm_head.weight" in missing
        and "model.embed_tokens.weight" in mapped
    ):
        target.lm_head.weight = target.model.embed_tokens.weight
        missing.remove("lm_head.weight")
    if missing or mismatched or duplicate or unexpected:
        # Do not silently accept architecture-specific tensors that this engine ignores.
        raise ValueError(
            f"incompatible HF checkpoint: missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatch={mismatched[:8]}, duplicate={duplicate[:8]}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=bool(config_kwargs.get("token")),
        trust_remote_code=trust_remote_code,
    )
    if len(tokenizer) != target_config.vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} differs from embedding vocab {target_config.vocab_size}"
        )
    report = {
        "model_id": model_id,
        "revision": revision,
        "tokenizer_length": len(tokenizer),
        "loaded_tensors": len(mapped),
        "parameters": sum(v.numel() for v in expected.values()),
        "source_max_position_embeddings": int(config.max_position_embeddings),
        "target_max_position_embeddings": target_config.max_position_embeddings,
        "rope_scaling": target_config.rope_scaling,
        "attention_bias": target_config.attention_bias,
        "mlp_bias": target_config.mlp_bias,
        "tie_word_embeddings": target_config.tie_word_embeddings,
    }
    return target, tokenizer, target_config, report
