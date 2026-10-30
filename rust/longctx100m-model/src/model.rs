//! Dense Llama-family decoder model mirroring `longctx100m.model.LongContextDecoder`.

use crate::config::ModelConfig;
use crate::rope::apply_rope;
use candle_core::{DType, Result, Tensor, Var};
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("config error: {0}")]
    Config(#[from] crate::config::ModelConfigError),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("input_ids must have shape [batch, local_tokens]")]
    BadInputShape,
    #[error("input shard exceeds configured model context")]
    ContextOverflow,
}

impl From<ModelError> for candle_core::Error {
    fn from(e: ModelError) -> Self {
        candle_core::Error::Msg(e.to_string())
    }
}

impl From<crate::config::ModelConfigError> for candle_core::Error {
    fn from(e: crate::config::ModelConfigError) -> Self {
        candle_core::Error::Msg(e.to_string())
    }
}

/// RMSNorm with a single weight parameter (no bias), mirroring Python `RMSNorm`.
#[derive(Clone)]
pub struct RmsNorm {
    weight: Var,
    eps: f64,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        let weight = Var::from_tensor(&weight)?;
        Ok(Self { weight, eps })
    }

    pub fn weight(&self) -> &Var {
        &self.weight
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x * rsqrt(mean(x^2, -1) + eps) * weight, computed in fp32.
        let xf = x.to_dtype(DType::F32)?;
        let mean_sq = xf.sqr()?.mean(x.rank() - 1)?;
        let inv = mean_sq.affine(1.0, self.eps)?.sqrt()?.recip()?;
        let normed = xf.broadcast_mul(&inv.unsqueeze(x.rank() - 1)?)?;
        normed
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.weight.as_tensor())
    }
}

fn linear_bias(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<candle_nn::Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

/// Convert a tensor returned by a VarBuilder-backed layer into a trainable Var.
fn var_from_tensor(t: &Tensor) -> Result<Var> {
    Var::from_tensor(t)
}

/// Repeat each KV head `repeat` times along the head axis (dim 2), matching
/// PyTorch `repeat_interleave(repeat, dim=2)`.
fn repeat_interleave_heads(x: &Tensor, repeat: usize) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() != 4 {
        candle_core::bail!("expected rank-4 tensor for head repeat");
    }
    let (b, t, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let x = x.reshape((b, t, h, 1, d))?;
    let x = x.repeat((1, 1, 1, repeat, 1))?;
    x.reshape((b, t, h * repeat, d))
}

/// GQA causal self-attention with RoPE, mirroring `RingSelfAttention` in single-rank mode.
#[derive(Clone)]
pub struct SelfAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    config: ModelConfig,
}

impl SelfAttention {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let kv_heads = config.kv_heads();
        let vb_q = vb.push_prefix("self_attn");
        let q_proj = linear_bias(
            config.hidden_size,
            config.num_heads * head_dim,
            config.attention_bias,
            vb_q.push_prefix("q_proj"),
        )?;
        let k_proj = linear_bias(
            config.hidden_size,
            kv_heads * head_dim,
            config.attention_bias,
            vb_q.push_prefix("k_proj"),
        )?;
        let v_proj = linear_bias(
            config.hidden_size,
            kv_heads * head_dim,
            config.attention_bias,
            vb_q.push_prefix("v_proj"),
        )?;
        let o_proj = linear_bias(
            config.num_heads * head_dim,
            config.hidden_size,
            config.attention_bias,
            vb_q.push_prefix("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config: config.clone(),
        })
    }

    /// Single-rank exact dense causal attention (world_size = 1 path in Python).
    pub fn forward(&self, x: &Tensor, position_offset: Option<usize>) -> Result<Tensor> {
        self.forward_impl(x, position_offset, None)
    }

    /// Ring attention path: `shards` holds every rank's `[B, local_n, H, D]`
    /// Q/K/V triple in rank order; `rank` selects this process's shard. This
    /// mirrors the Python `dist.is_initialized()` branch of `RingSelfAttention`.
    /// Kept for the future distributed backend; the ring attention module itself
    /// is unit-tested against the dense reference.
    #[allow(dead_code)]
    pub fn forward_ring(
        &self,
        x: &Tensor,
        shards: &[(Tensor, Tensor, Tensor)],
        rank: usize,
        position_offset: Option<usize>,
        valid_length: Option<usize>,
    ) -> Result<Tensor> {
        self.forward_impl(x, position_offset, Some((shards, rank, valid_length)))
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        position_offset: Option<usize>,
        ring: Option<(&[(Tensor, Tensor, Tensor)], usize, Option<usize>)>,
    ) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(ModelError::BadInputShape.into());
        }
        let (b, t, c) = (dims[0], dims[1], dims[2]);
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_heads;
        let kv_heads = self.config.kv_heads();

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, num_heads, head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, kv_heads, head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, kv_heads, head_dim))?;

        let base = position_offset.unwrap_or(0);
        let pos = Tensor::arange(base as u32, (base + t) as u32, x.device())?;
        let q = apply_rope(&q, &pos, &self.config)?;
        let k = apply_rope(&k, &pos, &self.config)?;

        let repeat = num_heads / kv_heads;
        let (k, v) = if repeat > 1 {
            (
                repeat_interleave_heads(&k, repeat)?,
                repeat_interleave_heads(&v, repeat)?,
            )
        } else {
            (k, v)
        };

        let out = if let Some((shards, rank, valid_length)) = ring {
            // Ring path: rebuild the shard list with this rank's projected Q/K/V
            // and delegate to the ring attention module.
            let mut shards_owned: Vec<(Tensor, Tensor, Tensor)> = shards.to_vec();
            shards_owned[rank] = (q, k, v);
            crate::ring_attention::ring_causal_attention_forward(
                &shards_owned,
                rank,
                true,
                64,
                128,
                valid_length,
                position_offset,
            )?
        } else {
            // scores[b,h,t,s] = sum_d(q[b,t,h,d] * k[b,s,h,d]) * scale
            let scale = (head_dim as f64).powf(-0.5);
            let qf = q.to_dtype(DType::F32)?;
            let kf = k.to_dtype(DType::F32)?;
            let q_bht = qf.permute((0, 2, 1, 3))?.contiguous()?;
            let k_bhs = kf.permute((0, 2, 1, 3))?.contiguous()?;
            let scores = q_bht.matmul(&k_bhs.transpose(2, 3)?)?.affine(scale, 0.0)?;

            // Causal mask: -inf strictly above the diagonal, 0 on/below.
            // Build strictly-above-diagonal bool mask (Python triu(ones, diagonal=1)).
            let arange = Tensor::arange(0u32, t as u32, x.device())?;
            let row = arange.reshape((t, 1))?.broadcast_as((t, t))?;
            let col = arange.reshape((1, t))?.broadcast_as((t, t))?;
            let above = col.gt(&row)?; // true when column > row (strictly above diagonal)
            let neg_inf = f32::NEG_INFINITY;
            let mask = above.where_cond(
                &Tensor::full(neg_inf, (t, t), x.device())?,
                &Tensor::zeros((t, t), DType::F32, x.device())?,
            )?;
            let scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;

            let attn = candle_nn::ops::softmax(&scores, 3)?;
            let vf = v.to_dtype(DType::F32)?;
            let v_bhs = vf.permute((0, 2, 1, 3))?.contiguous()?;
            let out = attn.matmul(&v_bhs)?;
            out.permute((0, 2, 1, 3))?
                .contiguous()?
                .to_dtype(x.dtype())?
        };
        let out = out.reshape((b, t, c))?;
        self.o_proj.forward(&out)
    }

    fn all_vars(&self) -> Vec<Var> {
        vec![
            var_from_tensor(self.q_proj.weight()).unwrap(),
            var_from_tensor(self.k_proj.weight()).unwrap(),
            var_from_tensor(self.v_proj.weight()).unwrap(),
            var_from_tensor(self.o_proj.weight()).unwrap(),
        ]
    }

    fn named_vars(&self, prefix: &str) -> Vec<(String, Var)> {
        vec![
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                var_from_tensor(self.q_proj.weight()).unwrap(),
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                var_from_tensor(self.k_proj.weight()).unwrap(),
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                var_from_tensor(self.v_proj.weight()).unwrap(),
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                var_from_tensor(self.o_proj.weight()).unwrap(),
            ),
        ]
    }
}
/// SwiGLU MLP mirroring the Python `SwiGLU`.
#[derive(Clone)]
pub struct SwiGlu {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl SwiGlu {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_mlp = vb.push_prefix("mlp");
        let gate_proj = linear_bias(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
            vb_mlp.push_prefix("gate_proj"),
        )?;
        let up_proj = linear_bias(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
            vb_mlp.push_prefix("up_proj"),
        )?;
        let down_proj = linear_bias(
            config.intermediate_size,
            config.hidden_size,
            config.mlp_bias,
            vb_mlp.push_prefix("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }

    fn all_vars(&self) -> Vec<Var> {
        vec![
            var_from_tensor(self.gate_proj.weight()).unwrap(),
            var_from_tensor(self.up_proj.weight()).unwrap(),
            var_from_tensor(self.down_proj.weight()).unwrap(),
        ]
    }

    fn named_vars(&self, prefix: &str) -> Vec<(String, Var)> {
        vec![
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                var_from_tensor(self.gate_proj.weight()).unwrap(),
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                var_from_tensor(self.up_proj.weight()).unwrap(),
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                var_from_tensor(self.down_proj.weight()).unwrap(),
            ),
        ]
    }
}

/// One decoder layer: norm -> attention (residual) -> norm -> MLP (residual).
#[derive(Clone)]
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn: SelfAttention,
    mlp: SwiGlu,
}

impl DecoderLayer {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_layer = vb.push_prefix("layers");
        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb_layer.push_prefix("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb_layer.push_prefix("post_attention_layernorm"),
        )?;
        let self_attn = SelfAttention::new(config, vb_layer.clone())?;
        let mlp = SwiGlu::new(config, vb_layer)?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, position_offset: Option<usize>) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, position_offset)?;
        let x = x.add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }

    #[allow(dead_code)]
    fn forward_ring(
        &self,
        x: &Tensor,
        shards: &[(Tensor, Tensor, Tensor)],
        rank: usize,
        position_offset: Option<usize>,
        valid_length: Option<usize>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out =
            self.self_attn
                .forward_ring(&normed, shards, rank, position_offset, valid_length)?;
        let x = x.add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }

    fn all_vars(&self) -> Vec<Var> {
        let mut vars = vec![
            self.input_layernorm.weight.clone(),
            self.post_attention_layernorm.weight.clone(),
        ];
        vars.extend(self.self_attn.all_vars());
        vars.extend(self.mlp.all_vars());
        vars
    }

    fn named_vars(&self, layer_index: usize) -> Vec<(String, Var)> {
        let prefix = format!("model.layers.{layer_index}");
        let mut out = vec![
            (
                format!("{prefix}.input_layernorm.weight"),
                self.input_layernorm.weight.clone(),
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                self.post_attention_layernorm.weight.clone(),
            ),
        ];
        out.extend(self.self_attn.named_vars(&prefix));
        out.extend(self.mlp.named_vars(&prefix));
        out
    }
}
/// Decoder with Transformers LlamaForCausalLM state-dict naming.
#[derive(Clone)]
pub struct LongContextDecoder {
    pub config: ModelConfig,
    embed_tokens: Var,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Var,
}

impl LongContextDecoder {
    /// Build the model from a `VarBuilder` (VarMap for training, safetensors for inference).
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let vb_model = vb.push_prefix("model");
        let embed_tokens = vb_model.get_with_hints(
            (config.vocab_size, config.hidden_size),
            "embed_tokens.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let embed_tokens = Var::from_tensor(&embed_tokens)?;
        let layers = (0..config.num_layers)
            .map(|i| DecoderLayer::new(config, vb_model.push_prefix(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb_model.push_prefix("norm"),
        )?;
        let lm_head = vb.get_with_hints(
            (config.vocab_size, config.hidden_size),
            "lm_head.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let lm_head = Var::from_tensor(&lm_head)?;
        if config.tie_word_embeddings {
            lm_head.set(&embed_tokens)?;
        }
        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn device(&self) -> candle_core::Device {
        self.embed_tokens.device().clone()
    }

    pub fn all_vars(&self) -> Vec<Var> {
        let mut vars = vec![self.embed_tokens.clone()];
        for layer in &self.layers {
            vars.extend(layer.all_vars());
        }
        vars.push(self.norm.weight.clone());
        vars.push(self.lm_head.clone());
        vars
    }

    /// Debug helper: return the raw embedding output for the given input.
    pub fn debug_embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let flat = input_ids.flatten_all()?;
        self.embed_tokens.as_tensor().embedding(&flat)
    }

    /// Debug helper: compute layer-0 input RMSNorm output.
    pub fn debug_layer0_norm(&self, input_ids: &Tensor) -> Result<Tensor> {
        let flat = input_ids.flatten_all()?;
        let mut final_dims = input_ids.dims().to_vec();
        final_dims.push(self.config.hidden_size);
        let x = self
            .embed_tokens
            .as_tensor()
            .embedding(&flat)?
            .reshape(final_dims)?;
        self.layers[0].input_layernorm.forward(&x)
    }

    /// Debug helper: compute layer-0 q/k/v projections (post-RoPE, pre-attention).
    pub fn debug_layer0_qkv(&self, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let flat = input_ids.flatten_all()?;
        let mut final_dims = input_ids.dims().to_vec();
        final_dims.push(self.config.hidden_size);
        let x = self
            .embed_tokens
            .as_tensor()
            .embedding(&flat)?
            .reshape(final_dims)?;
        let normed = self.layers[0].input_layernorm.forward(&x)?;
        let attn = &self.layers[0].self_attn;
        let (b, t, _) = (normed.dims()[0], normed.dims()[1], normed.dims()[2]);
        let head_dim = self.config.head_dim();
        let q = attn
            .q_proj
            .forward(&normed)?
            .reshape((b, t, self.config.num_heads, head_dim))?;
        let k = attn
            .k_proj
            .forward(&normed)?
            .reshape((b, t, self.config.kv_heads(), head_dim))?;
        let pos = Tensor::arange(0u32, t as u32, x.device())?;
        let q = apply_rope(&q, &pos, &self.config)?;
        let k = apply_rope(&k, &pos, &self.config)?;
        Ok((q, k))
    }

    /// Debug helper: compute layer-0 attention output (post o_proj).
    pub fn debug_layer0_attn_out(&self, input_ids: &Tensor) -> Result<Tensor> {
        let flat = input_ids.flatten_all()?;
        let mut final_dims = input_ids.dims().to_vec();
        final_dims.push(self.config.hidden_size);
        let x = self
            .embed_tokens
            .as_tensor()
            .embedding(&flat)?
            .reshape(final_dims)?;
        let normed = self.layers[0].input_layernorm.forward(&x)?;
        let attn = &self.layers[0].self_attn;
        let (b, t, _) = (normed.dims()[0], normed.dims()[1], normed.dims()[2]);
        let head_dim = self.config.head_dim();
        let q = attn
            .q_proj
            .forward(&normed)?
            .reshape((b, t, self.config.num_heads, head_dim))?;
        let k = attn
            .k_proj
            .forward(&normed)?
            .reshape((b, t, self.config.kv_heads(), head_dim))?;
        let v = attn
            .v_proj
            .forward(&normed)?
            .reshape((b, t, self.config.kv_heads(), head_dim))?;
        let pos = Tensor::arange(0u32, t as u32, x.device())?;
        let q = apply_rope(&q, &pos, &self.config)?;
        let k = apply_rope(&k, &pos, &self.config)?;
        let repeat = self.config.num_heads / self.config.kv_heads();
        let k = repeat_interleave_heads(&k, repeat)?;
        let v = repeat_interleave_heads(&v, repeat)?;
        let scale = (head_dim as f64).powf(-0.5);
        let qf = q.to_dtype(DType::F32)?;
        let kf = k.to_dtype(DType::F32)?;
        let q_bht = qf.permute((0, 2, 1, 3))?.contiguous()?;
        let k_bhs = kf.permute((0, 2, 1, 3))?.contiguous()?;
        let scores = q_bht.matmul(&k_bhs.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let arange = Tensor::arange(0u32, t as u32, x.device())?;
        let row = arange.reshape((t, 1))?.broadcast_as((t, t))?;
        let col = arange.reshape((1, t))?.broadcast_as((t, t))?;
        let above = col.gt(&row)?;
        let neg_inf = f32::NEG_INFINITY;
        let mask = above.where_cond(
            &Tensor::full(neg_inf, (t, t), x.device())?,
            &Tensor::zeros((t, t), DType::F32, x.device())?,
        )?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn_soft = candle_nn::ops::softmax(&scores, 3)?;
        let vf = v.to_dtype(DType::F32)?;
        let v_bhs = vf.permute((0, 2, 1, 3))?.contiguous()?;
        let out = attn_soft.matmul(&v_bhs)?;
        let out = out
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .to_dtype(x.dtype())?
            .reshape((b, t, self.config.hidden_size))?;
        attn.o_proj.forward(&out)
    }

    /// Debug helper: compute layer-0 MLP output (post down_proj).
    pub fn debug_layer0_mlp_out(&self, input_ids: &Tensor) -> Result<Tensor> {
        let flat = input_ids.flatten_all()?;
        let mut final_dims = input_ids.dims().to_vec();
        final_dims.push(self.config.hidden_size);
        let x = self
            .embed_tokens
            .as_tensor()
            .embedding(&flat)?
            .reshape(final_dims)?;
        let normed = self.layers[0].post_attention_layernorm.forward(&x)?;
        self.layers[0].mlp.forward(&normed)
    }

    /// Debug helper: dump all parameter tensors as flattened f32 vectors.
    pub fn debug_params(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let mut out = Vec::new();
        for (name, var) in self.export_state_dict() {
            let v: Vec<f32> = var.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            out.push((name, v));
        }
        Ok(out)
    }

    /// Forward pass over `input_ids` of shape `[batch, tokens]` -> logits `[batch, tokens, vocab]`.
    pub fn forward(&self, input_ids: &Tensor, position_offset: Option<usize>) -> Result<Tensor> {
        let dims = input_ids.dims();
        if dims.len() != 2 {
            return Err(ModelError::BadInputShape.into());
        }
        let t = dims[1];
        let global_start = position_offset.unwrap_or(0);
        if global_start + t > self.config.max_position_embeddings {
            return Err(ModelError::ContextOverflow.into());
        }
        let x = {
            // Python nn.Embedding: [B, T] -> [B, T, H].
            let flat = input_ids.flatten_all()?;
            let values = self.embed_tokens.as_tensor().embedding(&flat)?;
            let mut final_dims = input_ids.dims().to_vec();
            final_dims.push(self.config.hidden_size);
            values.reshape(final_dims)?
        };
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, position_offset)?;
        }
        x = self.norm.forward(&x)?;
        let w = if self.config.tie_word_embeddings {
            // Tied head aliases the embedding table (Python `lm_head.weight = embed.weight`).
            self.embed_tokens.as_tensor()
        } else {
            self.lm_head.as_tensor()
        };
        // Linear without bias: [..., hidden] @ [hidden, vocab].
        let dims = x.dims();
        let (b, t) = (dims[0], dims[1]);
        let flat = x.reshape((b * t, self.config.hidden_size))?;
        let out = flat.matmul(&w.t()?)?;
        out.reshape((b, t, self.config.vocab_size))
    }

    /// Load a checkpoint directory written by the importer (`model.safetensors`).
    pub fn load_checkpoint(&mut self, dir: &std::path::Path) -> Result<()> {
        let path = dir.join("model.safetensors");
        if !path.is_file() {
            return Err(ModelError::Checkpoint(format!(
                "missing model.safetensors in {}",
                dir.display()
            ))
            .into());
        }
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path)? };
        let mut expected: HashMap<String, Var> = HashMap::new();
        expected.insert(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.clone(),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, var) in layer.named_vars(i) {
                expected.insert(name, var);
            }
        }
        expected.insert("model.norm.weight".to_string(), self.norm.weight.clone());
        // The importer omits lm_head.weight for tied embeddings (safetensors forbids aliases);
        // reconstruct it from the embedding table exactly like the Python loader.
        if self.config.tie_word_embeddings {
            let head = self.lm_head.clone();
            head.set(&self.embed_tokens.as_tensor())?;
        } else {
            expected.insert("lm_head.weight".to_string(), self.lm_head.clone());
        }
        for (name, var) in expected.iter() {
            let mut data = tensors.load(name, var.device())?;
            if data.dtype() != var.dtype() {
                data = data.to_dtype(var.dtype())?;
            }
            if data.shape() != var.shape() {
                return Err(ModelError::Checkpoint(format!(
                    "shape mismatch on {name}: checkpoint {:?} != model {:?}",
                    data.shape(),
                    var.shape()
                ))
                .into());
            }
            if data.dtype() != var.dtype() {
                candle_core::bail!(
                    "dtype mismatch on {name}: checkpoint {:?} != model {:?}",
                    data.dtype(),
                    var.dtype()
                );
            }
            var.set(&data)?;
        }
        Ok(())
    }

    /// Export the current weights as a safetensors-compatible state dict (Python-compatible keys).
    pub fn export_state_dict(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.as_tensor().clone(),
        )];
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, var) in layer.named_vars(i) {
                out.push((name, var.as_tensor().clone()));
            }
        }
        out.push((
            "model.norm.weight".to_string(),
            self.norm.weight.as_tensor().clone(),
        ));
        out.push((
            "lm_head.weight".to_string(),
            self.lm_head.as_tensor().clone(),
        ));
        out
    }

    /// Return the named trainable `Var`s in the same order as `export_state_dict`.
    pub fn state_vars(&self) -> Vec<(String, Var)> {
        let mut out = vec![(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.clone(),
        )];
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, var) in layer.named_vars(i) {
                out.push((name, var));
            }
        }
        out.push(("model.norm.weight".to_string(), self.norm.weight.clone()));
        out.push(("lm_head.weight".to_string(), self.lm_head.clone()));
        out
    }
}
