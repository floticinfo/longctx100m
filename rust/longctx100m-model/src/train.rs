//! Training runtime mirroring `longctx100m.train`: flat uint32 token corpus,
//! packed records, cross-entropy with padding masking, grad clipping, AdamW.

use crate::config::ModelConfig;
use crate::model::LongContextDecoder;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

pub struct TrainConfig {
    pub sequence_tokens: usize,
    pub steps: usize,
    pub lr: f64,
    pub save_every: usize,
    pub block_q: usize,
    pub block_k: usize,
    pub out: std::path::PathBuf,
    pub initial_checkpoint: Option<std::path::PathBuf>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            sequence_tokens: 32768,
            steps: 1000,
            lr: 3e-4,
            save_every: 100,
            block_q: 64,
            block_k: 128,
            out: "runs/longctx".into(),
            initial_checkpoint: None,
        }
    }
}

/// Read a flat uint32 token corpus and validate its packing.
pub fn read_corpus(path: &Path, sequence_tokens: usize) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).map_err(|e| {
        candle_core::Error::Msg(format!("cannot read corpus {}: {e}", path.display()))
    })?;
    if bytes.len() % 4 != 0 {
        candle_core::bail!("token corpus byte count must be divisible by uint32 size");
    }
    let mut tokens = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let record_tokens = sequence_tokens + 1;
    if tokens.len() % record_tokens != 0 {
        candle_core::bail!(
            "corpus token count must be divisible by sequence_tokens+1; no partial record truncation"
        );
    }
    if tokens.len() < record_tokens {
        candle_core::bail!("token corpus smaller than one packed record");
    }
    Ok(tokens)
}

/// Cross-entropy loss with `ignore_index = -100` semantics for padded targets,
/// divided by the total sequence token count (mirrors Python train loop).
pub fn masked_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    sequence_tokens: usize,
) -> Result<Tensor> {
    // logits: [B, T, V], targets: [B, T] (i64; -100 for ignored)
    let dims = logits.dims();
    let (b, t, v) = (dims[0], dims[1], dims[2]);
    let logits = logits.reshape((b * t, v))?;
    let targets = targets.reshape((b * t,))?;
    // candle_nn::loss::cross_entropy computes mean over all targets; we need
    // sum over valid targets divided by sequence_tokens. Compute manually.
    let logp = candle_nn::ops::log_softmax(&logits, 1)?;
    let target = targets.to_dtype(DType::U32)?;
    // Build a mask of valid targets (not -100).
    let valid = targets
        .to_dtype(DType::F32)?
        .affine(1.0, sequence_tokens as f64)?;
    // valid[i] = target + sequence_tokens; for target>=0 this is positive.
    let valid_mask = valid.gt(0.0)?;
    let gathered = logp.gather(&target.unsqueeze(1)?, 1)?.squeeze(1)?;
    let masked = gathered.mul(&valid_mask.to_dtype(DType::F32)?)?;
    // Cross-entropy is the negative log-likelihood (Python F.cross_entropy).
    let sum = masked.neg()?.sum_all()?;
    sum.affine(1.0 / sequence_tokens as f64, 0.0)
}

/// Clip gradients to a global L2 norm of 1.0 (mirrors `clip_grad_norm_`).
pub fn clip_grad_norm(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[Var],
    max_norm: f64,
) -> Result<f64> {
    let mut total_sq = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var) {
            let g = g.to_dtype(DType::F32)?;
            total_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let total = total_sq.sqrt();
    if total > max_norm {
        let scale = max_norm / total;
        for var in vars {
            if let Some(g) = grads.get(var) {
                let scaled = g.affine(scale, 0.0)?;
                grads.insert(var, scaled);
            }
        }
    }
    Ok(total)
}

/// One training step over a single packed record.
///
/// `x_local` is `[1, local_n]`, `y_local` is `[local_n]` with -100 for padded
/// positions. `position_offset` is the global start of this shard.
pub fn train_step(
    model: &LongContextDecoder,
    opt: &mut AdamW,
    varmap: &VarMap,
    x_local: &Tensor,
    y_local: &Tensor,
    sequence_tokens: usize,
    position_offset: usize,
) -> Result<f64> {
    let logits = model.forward(x_local, Some(position_offset))?;
    let loss = masked_cross_entropy(&logits, y_local, sequence_tokens)?;
    let mut grads = loss.backward()?;
    let vars = varmap.all_vars();
    clip_grad_norm(&mut grads, &vars, 1.0)?;
    opt.step(&grads)?;
    let loss_f32 = loss.to_scalar::<f32>()?;
    Ok(loss_f32 as f64)
}

/// AdamW optimizer with PyTorch defaults and serializable moments, mirroring
/// `torch.optim.AdamW` and `candle_nn::AdamW` exactly. The state (step count,
/// first/second moments) can be saved and restored for `--resume`.
#[derive(Clone, Debug)]
pub struct AdamWState {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub step_t: usize,
}

impl Default for AdamWState {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_t: 0,
        }
    }
}

pub struct AdamW {
    vars: Vec<(Var, Var, Var)>, // (param, first_moment, second_moment)
    state: AdamWState,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, state: AdamWState) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first = Var::zeros(shape, dtype, device)?;
                let second = Var::zeros(shape, dtype, device)?;
                Ok((var, first, second))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { vars, state })
    }

    pub fn state(&self) -> &AdamWState {
        &self.state
    }

    pub fn set_state(&mut self, state: AdamWState) {
        self.state = state;
    }

    /// Restore moments from a saved optimizer checkpoint. `names` must line up
    /// with the model's `export_state_dict` key order.
    pub fn load_moments(&mut self, first: &[Tensor], second: &[Tensor]) -> Result<()> {
        if first.len() != self.vars.len() || second.len() != self.vars.len() {
            candle_core::bail!("optimizer moment count mismatch");
        }
        for (i, (_, m, v)) in self.vars.iter().enumerate() {
            m.set(&first[i])?;
            v.set(&second[i])?;
        }
        Ok(())
    }

    pub fn moments(&self) -> Vec<(Tensor, Tensor)> {
        self.vars
            .iter()
            .map(|(_, m, v)| (m.as_tensor().clone(), v.as_tensor().clone()))
            .collect()
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.state.step_t += 1;
        let lr = self.state.lr;
        let lambda = self.state.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.state.beta1;
        let beta2 = self.state.beta2;
        let scale_m = 1.0 / (1.0 - beta1.powi(self.state.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.state.step_t as i32));
        for (var, m, v) in self.vars.iter() {
            if let Some(g) = grads.get(var) {
                let next_m = (m.as_tensor() * beta1)?.add(&(g * (1.0 - beta1))?)?;
                let next_v = (v.as_tensor() * beta2)?.add(&(g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = next_m.affine(scale_m, 0.0)?;
                let v_hat = next_v.affine(scale_v, 0.0)?;
                let next_theta = var.as_tensor().affine(1.0 - lr_lambda, 0.0)?;
                let adjusted_grad = m_hat.div(&v_hat.sqrt()?.affine(1.0, self.state.eps)?)?;
                let next_theta = next_theta.sub(&adjusted_grad.affine(lr, 0.0)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                var.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

/// Build the default AdamW state (PyTorch `torch.optim.AdamW(lr=...)`).
pub fn adamw_state(lr: f64) -> AdamWState {
    AdamWState {
        lr,
        ..AdamWState::default()
    }
}

/// Build a model on a fresh VarMap (random init), optionally loading an
/// imported checkpoint directory.
pub fn build_model(
    config: &ModelConfig,
    device: &Device,
    initial_checkpoint: Option<&Path>,
) -> Result<(LongContextDecoder, VarMap)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut model = LongContextDecoder::new(config, vb)?;
    if let Some(dir) = initial_checkpoint {
        model.load_checkpoint(dir)?;
    }
    Ok((model, varmap))
}

/// AdamW with the same defaults as `torch.optim.AdamW(lr=...)`.
pub fn adamw(vars: Vec<Var>, lr: f64) -> Result<AdamW> {
    AdamW::new(vars, adamw_state(lr))
}
