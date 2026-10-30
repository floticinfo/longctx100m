use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use longctx100m_core::{BackendCapabilities, CapacityInput, CoreError, estimate_capacity};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "longctx100m-rs",
    version,
    about = "Rust-first core tools for longctx100m; Python/CUDA paths remain available during migration"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the Rust core migration phase.
    MigrationStatus,
    /// Match the Python longctx100m-capacity report (KV-only lower bound; not a benchmark).
    Capacity {
        #[arg(long)]
        tokens: u64,
        #[arg(long, default_value_t = 2048)]
        output_tokens: u64,
        #[arg(long)]
        layers: u64,
        #[arg(long)]
        query_heads: u64,
        #[arg(long)]
        kv_heads: u64,
        #[arg(long)]
        head_dim: u64,
        #[arg(long, default_value_t = 2)]
        bytes_per_element: u64,
        #[arg(long)]
        world_size: u64,
        #[arg(long)]
        local_shard_tokens: u64,
        #[arg(long, default_value_t = 64)]
        block_q: u64,
        #[arg(long, default_value_t = 128)]
        block_k: u64,
        #[arg(long)]
        gpu_memory_gib: Option<f64>,
        #[arg(long, default_value_t = 0.8)]
        usable_memory_fraction: f64,
    },
    /// Validate a backend protocol v1 capability JSON without starting Python.
    ValidateCapabilities {
        /// Path to a JSON object containing the protocol-v1 capability fields.
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 0)]
        required_input_tokens: u64,
        #[arg(long, default_value_t = 0)]
        output_tokens: u64,
    },
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::MigrationStatus => println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "migration_phase": "expand_rust_model_runtime",
                "rust_core_compiled": true,
                "rust_http_gateway_compiled": true,
                "rust_checkpoint_importer_compiled": true,
                "rust_model_runtime_compiled": true,
                "python_runtime_retained": true,
                "http_gateway_rust": true,
                "tokenizer_import_rust": true,
                "rust_importer_parity_verified": true,
                "rust_model_forward_backward_parity": true,
                "rust_training_runtime": true,
                "rust_ring_attention_forward": true,
                "rust_ring_attention_backward": true,
                "rust_inference_runtime": true,
                "pytorch_cuda_nccl_rust": false,
                "triton_rust": false,
                "rollback": "continue using existing Python entrypoints",
                "next_migration": "CUDA/NCCL transport for ring attention and Linux CI; keep Python until GPU parity",
            }))?
        ),
        Command::Capacity {
            tokens,
            output_tokens,
            layers,
            query_heads,
            kv_heads,
            head_dim,
            bytes_per_element,
            world_size,
            local_shard_tokens,
            block_q,
            block_k,
            gpu_memory_gib,
            usable_memory_fraction,
        } => {
            let report = estimate_capacity(CapacityInput {
                tokens,
                layers,
                query_heads,
                kv_heads,
                head_dim,
                bytes_per_element,
                world: world_size,
                local_shard_tokens,
                block_q,
                block_k,
                output_tokens,
                report_gpu_memory_gib: gpu_memory_gib,
                usable_memory_fraction,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::ValidateCapabilities {
            input,
            required_input_tokens,
            output_tokens,
        } => {
            let source = fs::read_to_string(input)?;
            let caps = BackendCapabilities::from_json(&source)?;
            caps.require_context(required_input_tokens, output_tokens)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "valid": true,
                    "protocol_version": caps.protocol_version,
                    "model_id": caps.model_id,
                    "tokenizer_id": caps.tokenizer_id,
                    "context_limit_tokens": caps.context_limit_tokens,
                    "attention_mode": caps.attention_mode,
                    "required_input_tokens": required_input_tokens,
                    "output_tokens": output_tokens,
                }))?
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let status = match error.downcast_ref::<CoreError>() {
                Some(
                    CoreError::InsufficientShardCapacity
                    | CoreError::InvalidDimensions
                    | CoreError::InvalidHeadRatio
                    | CoreError::InvalidMemoryParameters,
                ) => 2,
                Some(
                    CoreError::ContextTooSmall { .. }
                    | CoreError::UnsupportedProtocol(_)
                    | CoreError::RejectTruncationRequired
                    | CoreError::SingleRequestRequired
                    | CoreError::UnsupportedAttentionMode
                    | CoreError::EmptyModelOrTokenizer
                    | CoreError::InvalidContextLimit
                    | CoreError::CapabilitiesNotObject
                    | CoreError::InvalidCapabilities(_),
                ) => 2,
                _ => 1,
            };
            eprintln!("error: {error}");
            if status == 2 {
                eprintln!("error_code: longctx100m_validation_error");
            }
            ExitCode::from(status)
        }
    }
}
