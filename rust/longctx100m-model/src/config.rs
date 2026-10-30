//! Model configuration mirroring `longctx100m.model.ModelConfig`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum ModelConfigError {
    #[error("model dimensions must be positive")]
    NonPositiveDimensions,
    #[error("num_attention_heads must be divisible by positive num_key_value_heads")]
    InvalidKvHeads,
    #[error("hidden_size must divide evenly into attention heads")]
    HiddenNotDivisible,
    #[error("head dimension must be even for rotary embeddings")]
    OddHeadDim,
    #[error("unsupported RoPE scaling profile {0:?}")]
    UnsupportedRopeProfile(String),
    #[error("incomplete llama3 rope scaling profile")]
    IncompleteLlama3Profile,
    #[error("linear RoPE factor must be positive")]
    NonPositiveLinearFactor,
    #[error("this engine currently supports only Llama-family SiLU gated MLP")]
    UnsupportedActivation,
}

fn default_kv_heads() -> Option<usize> {
    None
}

/// Decoder configuration with Transformers LlamaConfig-compatible fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: Option<usize>,
}

impl ModelConfig {
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Validate the config with the same invariants as the Python model.
    pub fn validate(&self) -> Result<(), ModelConfigError> {
        if self.rope_scaling.is_some() {
            let rope = self.rope_scaling.as_ref().unwrap();
            let rope_type = rope
                .get("rope_type")
                .or_else(|| rope.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();
            match rope_type.as_str() {
                "default" => {}
                "linear" => {
                    let factor = rope.get("factor").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if factor <= 0.0 {
                        return Err(ModelConfigError::NonPositiveLinearFactor);
                    }
                }
                "llama3" => {
                    let required = [
                        "factor",
                        "low_freq_factor",
                        "high_freq_factor",
                        "original_max_position_embeddings",
                    ];
                    if !required.iter().all(|k| rope.contains_key(*k)) {
                        return Err(ModelConfigError::IncompleteLlama3Profile);
                    }
                }
                other => return Err(ModelConfigError::UnsupportedRopeProfile(other.into())),
            }
        }
        if [
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.max_position_embeddings,
            self.intermediate_size,
        ]
        .into_iter()
        .any(|v| v == 0)
        {
            return Err(ModelConfigError::NonPositiveDimensions);
        }
        let kv = self.kv_heads();
        if kv == 0 || self.num_heads % kv != 0 {
            return Err(ModelConfigError::InvalidKvHeads);
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(ModelConfigError::HiddenNotDivisible);
        }
        if self.head_dim() % 2 != 0 {
            return Err(ModelConfigError::OddHeadDim);
        }
        if self.hidden_act != "silu" {
            return Err(ModelConfigError::UnsupportedActivation);
        }
        Ok(())
    }

    /// Load and validate a config from the checkpoint `config.json` schema.
    pub fn from_json_str(json: &str) -> Result<Self, ModelConfigError> {
        let config: ModelConfig = serde_json::from_str(json).map_err(|e| {
            ModelConfigError::UnsupportedRopeProfile(format!("invalid config json: {e}"))
        })?;
        config.validate()?;
        Ok(config)
    }
}
