//! Rust-first, dependency-light core for longctx100m.
//!
//! The initial migration covers exact integer capacity estimates and the backend
//! capability contract. Existing Python HTTP/PyTorch/NCCL paths remain available
//! until parity and target-hardware validation justify retiring them.

mod capacity;
mod contract;

pub use capacity::{CapacityInput, CapacityReport, estimate_capacity};
pub use contract::{BackendCapabilities, CoreError, PROTOCOL_VERSION, SUPPORTED_ATTENTION_MODES};
