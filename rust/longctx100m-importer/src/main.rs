use clap::Parser;
use std::{path::PathBuf, process::ExitCode};
#[derive(Debug, Parser)]
#[command(
    name = "longctx100m-import-llama",
    version,
    about = "Import a pinned Hugging Face Llama safetensors checkpoint"
)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, required = true)]
    revision: String,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 100_002_048)]
    max_position_embeddings: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = false)]
    trust_remote_code: bool,
}
fn main() -> ExitCode {
    let a = Args::parse();
    if a.trust_remote_code {
        eprintln!("error: trust-remote-code is unsupported by Rust importer");
        return ExitCode::from(2);
    }
    if let Err(e) = longctx100m_importer::importer::validate_revision(&a.revision) {
        eprintln!("error: {e}");
        return ExitCode::from(2);
    }
    match longctx100m_importer::run_import(
        &a.model,
        &a.revision,
        a.cache_dir.as_deref(),
        a.max_position_embeddings,
        &a.out,
    ) {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
