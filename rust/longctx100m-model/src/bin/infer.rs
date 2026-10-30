//! Rust inference runtime for the ring-attention decoder.
//!
//! Loads an imported checkpoint directory (`config.json` + `model.safetensors`)
//! and produces next-token logits or greedy continuation over a token-id file.
//! Mirrors the single-rank path of `longctx100m.distributed_infer`: token-file
//! validation, greedy decoding, and uint32 token output.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use longctx100m_model::config::ModelConfig;
use longctx100m_model::model::LongContextDecoder;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut checkpoint = None;
    let mut input_tokens: Option<PathBuf> = None;
    let mut generate = 0usize;
    let mut prompt = String::new();
    let mut dump_logits = false;
    let mut token_output: Option<PathBuf> = None;
    let mut require_100m = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    args.next().ok_or("--checkpoint needs a value")?,
                ))
            }
            "--input-tokens" => input_tokens = Some(PathBuf::from(args.next().ok_or("value")?)),
            "--generate" => generate = args.next().ok_or("value")?.parse()?,
            "--prompt-tokens" => prompt = args.next().ok_or("value")?,
            "--dump-logits" => dump_logits = true,
            "--token-output" => token_output = Some(PathBuf::from(args.next().ok_or("value")?)),
            "--require-100m" => require_100m = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let checkpoint = checkpoint.ok_or("--checkpoint is required")?;
    if !checkpoint.join("config.json").is_file() || !checkpoint.join("model.safetensors").is_file()
    {
        return Err("checkpoint directory must contain config.json and model.safetensors".into());
    }

    let config =
        ModelConfig::from_json_str(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut model = LongContextDecoder::new(&config, vb)?;
    model.load_checkpoint(&checkpoint)?;

    let mut ids: Vec<u32> = if let Some(path) = &input_tokens {
        let bytes = std::fs::read(path)?;
        if bytes.len() % 4 != 0 {
            return Err("token file byte count must be divisible by uint32 size".into());
        }
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if require_100m && tokens.len() < 100_000_000 {
            return Err("input token file is shorter than the required 100M-token request".into());
        }
        if let Some(&max_id) = tokens.iter().max() {
            if max_id as usize >= config.vocab_size {
                return Err("token file includes ids outside model vocabulary".into());
            }
        }
        tokens
    } else if !prompt.is_empty() {
        prompt
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err("provide --input-tokens or --prompt-tokens".into());
    };

    if ids.is_empty() {
        return Err("empty prompt".into());
    }
    if dump_logits {
        let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, None)?;
        let last = logits.narrow(1, ids.len() - 1, 1)?;
        let v: Vec<f32> = last.flatten_all()?.to_vec1()?;
        println!(
            "{{\"prompt_tokens\": {}, \"last_position_logits\": {:?}}}",
            ids.len(),
            v
        );
        return Ok(());
    }
    let generated: Vec<u32> = (0..generate)
        .map(|_| {
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, None)?;
            // Last position argmax over vocab.
            let last = logits.narrow(1, ids.len() - 1, 1)?;
            let next = last.argmax(2)?.flatten_all()?.get(0)?.to_scalar::<u32>()?;
            ids.push(next);
            Ok(next)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    if let Some(out_path) = &token_output {
        // Write generated token ids as little-endian uint32 (rank-0 output in
        // distributed_infer).
        let mut bytes = Vec::with_capacity(generated.len() * 4);
        for t in &generated {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(out_path, bytes)?;
    }

    println!(
        "{{\"prompt_tokens\": {}, \"generated_tokens\": {}, \"output\": {:?}}}",
        ids.len() - generate,
        generate,
        ids
    );
    Ok(())
}
