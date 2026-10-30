use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const PROTOCOL_VERSION: u64 = 1;
pub const SUPPORTED_ATTENTION_MODES: [&str; 3] = [
    "dense_full_attention",
    "sparse_attention",
    "recurrent_compressed",
];

#[derive(Debug, Error, PartialEq)]
pub enum CoreError {
    #[error("all estimator dimensions must be positive")]
    InvalidDimensions,
    #[error("query heads must be divisible by KV heads")]
    InvalidHeadRatio,
    #[error("per-rank shard capacity is smaller than input plus output")]
    InsufficientShardCapacity,
    #[error("invalid device-memory estimate parameters")]
    InvalidMemoryParameters,
    #[error("capacity arithmetic overflow")]
    ArithmeticOverflow,
    #[error("Backend capabilities must be a JSON object")]
    CapabilitiesNotObject,
    #[error("Invalid backend capability manifest: {0}")]
    InvalidCapabilities(String),
    #[error("Unsupported backend protocol version {0}; expected 1")]
    UnsupportedProtocol(u64),
    #[error("Backend must declare non-empty model_id and tokenizer_id")]
    EmptyModelOrTokenizer,
    #[error("context_limit_tokens must be positive")]
    InvalidContextLimit,
    #[error("Backend must support one complete logical input request")]
    SingleRequestRequired,
    #[error("Backend truncation_policy must be 'reject'; silent truncation is unsafe")]
    RejectTruncationRequired,
    #[error(
        "attention_mode must be one of: dense_full_attention, sparse_attention, recurrent_compressed"
    )]
    UnsupportedAttentionMode,
    #[error(
        "Backend advertises {limit} tokens, below the required input plus output budget of {required}"
    )]
    ContextTooSmall { limit: u64, required: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub model_id: String,
    pub tokenizer_id: String,
    pub context_limit_tokens: u64,
    pub supports_single_request: bool,
    pub truncation_policy: String,
    pub attention_mode: String,
    pub protocol_version: u64,
}
fn string_field(v: &Value, k: &str) -> Result<String, CoreError> {
    v.get(k)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| CoreError::InvalidCapabilities(format!("missing or invalid {k}")))
}
fn integer_field(v: &Value, k: &str) -> Result<u64, CoreError> {
    v.get(k)
        .and_then(Value::as_u64)
        .ok_or_else(|| CoreError::InvalidCapabilities(format!("missing or invalid {k}")))
}
impl BackendCapabilities {
    pub fn from_json(s: &str) -> Result<Self, CoreError> {
        let v: Value =
            serde_json::from_str(s).map_err(|e| CoreError::InvalidCapabilities(e.to_string()))?;
        Self::from_value(&v)
    }
    pub fn from_value(v: &Value) -> Result<Self, CoreError> {
        if !v.is_object() {
            return Err(CoreError::CapabilitiesNotObject);
        }
        let c = Self {
            model_id: string_field(v, "model_id")?,
            tokenizer_id: string_field(v, "tokenizer_id")?,
            context_limit_tokens: integer_field(v, "context_limit_tokens")?,
            supports_single_request: v
                .get("supports_single_request")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    CoreError::InvalidCapabilities(
                        "missing or invalid supports_single_request".into(),
                    )
                })?,
            truncation_policy: string_field(v, "truncation_policy")?,
            attention_mode: string_field(v, "attention_mode")?,
            protocol_version: integer_field(v, "protocol_version")?,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(CoreError::UnsupportedProtocol(self.protocol_version));
        }
        if self.model_id.trim().is_empty() || self.tokenizer_id.trim().is_empty() {
            return Err(CoreError::EmptyModelOrTokenizer);
        }
        if self.context_limit_tokens == 0 {
            return Err(CoreError::InvalidContextLimit);
        }
        if !self.supports_single_request {
            return Err(CoreError::SingleRequestRequired);
        }
        if self.truncation_policy != "reject" {
            return Err(CoreError::RejectTruncationRequired);
        }
        if !SUPPORTED_ATTENTION_MODES.contains(&self.attention_mode.as_str()) {
            return Err(CoreError::UnsupportedAttentionMode);
        }
        Ok(())
    }
    pub fn require_context(&self, input: u64, output: u64) -> Result<(), CoreError> {
        let required = input
            .checked_add(output)
            .ok_or(CoreError::ArithmeticOverflow)?;
        if self.context_limit_tokens < required {
            return Err(CoreError::ContextTooSmall {
                limit: self.context_limit_tokens,
                required,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"protocol_version":1,"model_id":"research-100m","tokenizer_id":"research-tokenizer-v1","context_limit_tokens":100002048,"supports_single_request":true,"truncation_policy":"reject","attention_mode":"sparse_attention"}"#;

    #[test]
    fn accepts_python_protocol_v1_golden_and_budget() {
        let caps = BackendCapabilities::from_json(GOOD).unwrap();
        assert!(caps.require_context(100_000_000, 2_048).is_ok());
        let value = serde_json::to_value(&caps).unwrap();
        assert_eq!(value["model_id"], "research-100m");
        assert_eq!(value["context_limit_tokens"], 100_002_048u64);
    }

    #[test]
    fn rejects_unsafe_claims_and_context_overflow() {
        let caps = BackendCapabilities::from_json(GOOD).unwrap();
        assert_eq!(
            caps.require_context(100_000_001, 2_048),
            Err(CoreError::ContextTooSmall {
                limit: 100_002_048,
                required: 100_002_049
            })
        );
        let silent = GOOD.replace("\"reject\"", "\"silent\"");
        assert_eq!(
            BackendCapabilities::from_json(&silent),
            Err(CoreError::RejectTruncationRequired)
        );
    }

    #[test]
    fn rejects_bad_json_mode_and_empty_model_id() {
        assert_eq!(
            BackendCapabilities::from_json("[]"),
            Err(CoreError::CapabilitiesNotObject)
        );
        let unknown = GOOD.replace("sparse_attention", "made_up");
        assert_eq!(
            BackendCapabilities::from_json(&unknown),
            Err(CoreError::UnsupportedAttentionMode)
        );
        let empty = GOOD.replace("research-100m", " ");
        assert_eq!(
            BackendCapabilities::from_json(&empty),
            Err(CoreError::EmptyModelOrTokenizer)
        );
    }
}
