//! Rust ring-attention decoder model and training runtime.
//!
//! This crate mirrors the Python `longctx100m.model` decoder exactly: dense
//! Llama-family GQA attention with RoPE, RMSNorm, SwiGLU MLP, tied or untied
//! embeddings, and Transformers-compatible state-dict naming. It uses candle
//! for CPU (and later CUDA) execution with `Var`-based autodiff for training.

pub mod config;
pub mod model;
pub mod ring_attention;
pub mod rope;
pub mod train;

pub use config::{ModelConfig, ModelConfigError};
pub use model::{LongContextDecoder, ModelError};
pub use ring_attention::{
    reference_causal_attention, ring_causal_attention_backward, ring_causal_attention_forward,
    ring_causal_attention_forward_with_stats,
};
pub use train::{AdamW, AdamWState};
