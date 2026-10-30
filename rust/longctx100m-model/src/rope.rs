//! RoPE (rotary position embeddings) mirroring `longctx100m.model._apply_rope`.

use crate::config::ModelConfig;
use candle_core::{DType, Result, Tensor};

/// Apply Llama-family RoPE to `x` of shape `[B, T, H, D]` with positions `[T]`.
///
/// Mirrors the Python `_apply_rope`: half-dimensional frequencies duplicated,
/// rotate_half split-halves rotation, optional linear/llama3 scaling.
pub fn apply_rope(x: &Tensor, positions: &Tensor, config: &ModelConfig) -> Result<Tensor> {
    let dim = x.dim(x.rank() - 1)?;
    let device = x.device();
    let theta = config.rope_theta;

    // inv = theta ** (-arange(0, dim, 2) / dim)
    let half = dim / 2;
    let arange = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
    // Python uses arange(0, dim, 2): indices 0, 2, 4, ..., dim-2.
    let mut inv = arange
        .affine(-2.0 * (theta.ln()) / dim as f64, 0.0)?
        .exp()?;

    let mut attention_factor = 1.0f64;
    let mut positions = positions.clone();
    if let Some(rope) = &config.rope_scaling {
        let rope_type = rope
            .get("rope_type")
            .or_else(|| rope.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        match rope_type {
            "default" => {}
            "linear" => {
                let factor = rope.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                if factor <= 0.0 {
                    candle_core::bail!("rope scaling factor must be positive");
                }
                let f32_pos = positions.to_dtype(DType::F32)?;
                positions = f32_pos.affine(1.0 / factor, 0.0)?;
            }
            "llama3" => {
                let required = [
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max_position_embeddings",
                ];
                if !required.iter().all(|k| rope.contains_key(*k)) {
                    candle_core::bail!(
                        "llama3 RoPE needs low/high frequency factors and original context length"
                    );
                }
                let low = rope
                    .get("low_freq_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap();
                let high = rope
                    .get("high_freq_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap();
                let original = rope
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_f64())
                    .unwrap();
                let factor = rope.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                if low <= 0.0 || high <= low || original <= 0.0 {
                    candle_core::bail!("invalid llama3 RoPE scaling values");
                }
                // wavelength = 2*pi / inv
                let two_pi = 2.0 * std::f64::consts::PI;
                let wavelength = inv.affine(two_pi, 0.0)?.div(&inv)?;
                let low_wavelen = original / low;
                let high_wavelen = original / high;
                let scaled = inv.affine(1.0 / factor, 0.0)?;
                // smooth = (original / wavelength - low) / (high - low)
                let smooth = wavelength
                    .recip()?
                    .affine(original / (high - low), -(low / (high - low)))?;
                // mid = (1 - smooth) * low_scaled + smooth * inv
                let low_scaled = inv.affine(1.0 / factor, 0.0)?;
                let mid = smooth
                    .affine(-1.0, 1.0)?
                    .mul(&low_scaled)?
                    .add(&smooth.mul(&inv)?)?;
                let gt_low = wavelength.gt(low_wavelen)?;
                let in_mid = wavelength
                    .ge(high_wavelen)?
                    .mul(&wavelength.le(low_wavelen)?)?;
                inv = gt_low.where_cond(&scaled, &inv)?;
                inv = in_mid.where_cond(&mid, &inv)?;
                attention_factor = rope
                    .get("attention_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                if attention_factor <= 0.0 {
                    candle_core::bail!("llama3 attention_factor must be positive");
                }
            }
            other => candle_core::bail!("unsupported RoPE type {other:?}"),
        }
    }

    // freq = outer(positions, inv); duplicate along the last dim.
    let freq = positions
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .matmul(&inv.unsqueeze(0)?)?;
    let freq = Tensor::cat(&[&freq, &freq], 1)?;
    // Python: cos = freq.cos()[None, :, None, :] * attention_factor
    let cos = freq
        .cos()?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .affine(attention_factor, 0.0)?;
    let sin = freq.sin()?.unsqueeze(0)?.unsqueeze(2)?;

    // rotate_half: split halves. candle needs broadcast_mul for broadcasting.
    let first = x.narrow(x.rank() - 1, 0, half)?;
    let second = x.narrow(x.rank() - 1, half, half)?;
    let cos_first = cos.narrow(cos.rank() - 1, 0, half)?;
    let sin_first = sin.narrow(sin.rank() - 1, 0, half)?;
    let out_first = first
        .broadcast_mul(&cos_first)?
        .sub(&second.broadcast_mul(&sin_first)?)?;
    let out_second = second
        .broadcast_mul(&cos_first)?
        .add(&first.broadcast_mul(&sin_first)?)?;
    Tensor::cat(&[&out_first, &out_second], x.rank() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::model::LongContextDecoder;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            intermediate_size: 12,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            hidden_act: "silu".into(),
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            num_key_value_heads: Some(1),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let config = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = LongContextDecoder::new(&config, vb).unwrap();
        let ids = Tensor::new(&[1u32, 2, 3, 4], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = model.forward(&ids, None).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, 16]);
        let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn export_import_roundtrip() {
        let config = test_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = LongContextDecoder::new(&config, vb).unwrap();
        let ids = Tensor::new(&[1u32, 2, 3, 4], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let before = model.forward(&ids, None).unwrap();

        // Export to safetensors, rebuild, and confirm identical forward.
        let dir = tempfile::tempdir().unwrap();
        let state = model.export_state_dict();
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor);
        }
        candle_core::safetensors::save(&map, dir.path().join("model.safetensors")).unwrap();
        let varmap2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
        let mut model2 = LongContextDecoder::new(&config, vb2).unwrap();
        model2.load_checkpoint(dir.path()).unwrap();
        let after = model2.forward(&ids, None).unwrap();
        let bv: Vec<f32> = before.flatten_all().unwrap().to_vec1().unwrap();
        let av: Vec<f32> = after.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(bv.len(), av.len());
        for (b, a) in bv.iter().zip(av.iter()) {
            assert!((b - a).abs() < 1e-5, "roundtrip mismatch {b} vs {a}");
        }
    }
}
