//! Rust training loop for the ring-attention decoder, mirroring
//! `python -m longctx100m.train`. Single-rank CPU execution in this slice.

use candle_core::{DType, Device, Tensor, Var};
use longctx100m_model::config::ModelConfig;
use longctx100m_model::model::LongContextDecoder;
use longctx100m_model::train::{adamw, build_model, read_corpus, train_step};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: longctx100m-train --tokens <corpus.bin> --config <config.json> \
[--sequence-tokens N] [--steps N] [--lr F] [--save-every N] [--out DIR] \
[--initial-checkpoint DIR] [--resume DIR]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut tokens_path: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut sequence_tokens = 32768usize;
    let mut steps = 1000usize;
    let mut lr = 3e-4f64;
    let mut save_every = 100usize;
    let mut out = PathBuf::from("runs/longctx");
    let mut initial_checkpoint: Option<PathBuf> = None;
    let mut resume: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(args.next().ok_or("--tokens needs a value")?))
            }
            "--config" => {
                config_path = Some(PathBuf::from(args.next().ok_or("--config needs a value")?))
            }
            "--sequence-tokens" => sequence_tokens = args.next().ok_or("value")?.parse()?,
            "--steps" => steps = args.next().ok_or("value")?.parse()?,
            "--lr" => lr = args.next().ok_or("value")?.parse()?,
            "--save-every" => save_every = args.next().ok_or("value")?.parse()?,
            "--out" => out = PathBuf::from(args.next().ok_or("value")?),
            "--initial-checkpoint" => {
                initial_checkpoint = Some(PathBuf::from(args.next().ok_or("value")?))
            }
            "--resume" => resume = Some(PathBuf::from(args.next().ok_or("value")?)),
            other => return Err(format!("unknown argument {other}; {USAGE}").into()),
        }
    }
    let tokens_path = tokens_path.ok_or("--tokens is required")?;
    let config_path = config_path.ok_or("--config is required")?;
    if sequence_tokens == 0 || steps == 0 || save_every == 0 || lr <= 0.0 {
        return Err("sequence-tokens, steps, save-every and lr must be positive".into());
    }
    if initial_checkpoint.is_some() && resume.is_some() {
        return Err("--initial-checkpoint and --resume are mutually exclusive".into());
    }

    let config_json = std::fs::read_to_string(&config_path)?;
    let config = ModelConfig::from_json_str(&config_json)?;

    let tokens = read_corpus(&tokens_path, sequence_tokens)?;
    let record_tokens = sequence_tokens + 1;
    let records = tokens.len() / record_tokens;

    let device = Device::Cpu;
    let (model, varmap) = build_model(&config, &device, initial_checkpoint.as_deref())?;
    // Order optimizer vars to match `export_state_dict` so resume moments line up.
    let vars: Vec<Var> = model.state_vars().into_iter().map(|(_, v)| v).collect();
    let mut opt = adamw(vars, lr)?;
    let mut start_step = 0usize;
    if let Some(resume_dir) = &resume {
        start_step = load_resume(resume_dir, &config, &model, &mut opt, &device)?;
    }

    std::fs::create_dir_all(&out)?;
    for step in start_step..start_step + steps {
        let row = &tokens
            [(step % records) * record_tokens..(step % records) * record_tokens + record_tokens];
        // Validate token IDs lazily per selected row.
        let x = &row[..row.len() - 1];
        let y = &row[1..];
        if x.iter().any(|&t| t as usize >= config.vocab_size) {
            return Err("corpus token id is outside model vocabulary".into());
        }
        // world_size = 1 in this slice: no padding needed.
        let x_local = Tensor::new(x, &device)?.unsqueeze(0)?;
        let y_local = Tensor::new(y, &device)?;
        let loss = train_step(
            &model,
            &mut opt,
            &varmap,
            &x_local,
            &y_local,
            sequence_tokens,
            0,
        )?;
        if step % 10 == 0 {
            println!(
                "{{\"step\": {step}, \"loss\": {loss:.6}, \"tokens_per_step\": {}}}",
                sequence_tokens
            );
        }
        if step % save_every == 0 || step == start_step + steps - 1 {
            save_checkpoint(&out, &model, &config, &opt, step)?;
        }
    }
    Ok(())
}

/// Restore model weights, optimizer moments, and the step counter from a
/// checkpoint directory written by `save_checkpoint`.
fn load_resume(
    resume_dir: &Path,
    config: &ModelConfig,
    model: &LongContextDecoder,
    opt: &mut longctx100m_model::train::AdamW,
    device: &Device,
) -> Result<usize, Box<dyn std::error::Error>> {
    let meta_path = resume_dir.join("resume.json");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let step = meta["step"].as_u64().ok_or("resume.json missing step")? as usize;
    let saved_lr = meta["lr"].as_f64().ok_or("resume.json missing lr")?;
    let weights_path = resume_dir.join("model.safetensors");
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights_path)? };
    for (name, var) in model.state_vars() {
        if config.tie_word_embeddings && name == "lm_head.weight" {
            // Tied head aliases the embedding; reconstruct after embed_tokens loads.
            continue;
        }
        let data = tensors.load(&name, device)?;
        let data = if data.dtype() != var.dtype() {
            data.to_dtype(var.dtype())?
        } else {
            data
        };
        if data.shape() != var.shape() {
            return Err(format!(
                "resume shape mismatch on {name}: {:?} != {:?}",
                data.shape(),
                var.shape()
            )
            .into());
        }
        var.set(&data)?;
    }
    if config.tie_word_embeddings {
        let state = model.state_vars();
        let head = state
            .iter()
            .find(|(n, _)| n == "lm_head.weight")
            .map(|(_, v)| v)
            .ok_or("missing lm_head var")?;
        let embed = state
            .iter()
            .find(|(n, _)| n == "model.embed_tokens.weight")
            .map(|(_, v)| v)
            .ok_or("missing embed var")?;
        head.set(&embed.as_tensor())?;
    }
    let moments_path = resume_dir.join("optimizer.safetensors");
    let moments = unsafe { candle_core::safetensors::MmapedSafetensors::new(&moments_path)? };
    let mut first = Vec::new();
    let mut second = Vec::new();
    for (name, _) in model.export_state_dict() {
        let f = moments.load(&format!("{name}.exp_avg"), device)?;
        let s = moments.load(&format!("{name}.exp_avg_sq"), device)?;
        let f = if f.dtype() != DType::F32 {
            f.to_dtype(DType::F32)?
        } else {
            f
        };
        let s = if s.dtype() != DType::F32 {
            s.to_dtype(DType::F32)?
        } else {
            s
        };
        first.push(f);
        second.push(s);
    }
    opt.load_moments(&first, &second)?;
    let mut new_state = opt.state().clone();
    new_state.lr = saved_lr;
    new_state.step_t = step + 1;
    opt.set_state(new_state);
    Ok(step + 1)
}

/// Save model weights, optimizer moments, and resume metadata atomically.
fn save_checkpoint(
    out: &Path,
    model: &LongContextDecoder,
    config: &ModelConfig,
    opt: &longctx100m_model::train::AdamW,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = model.export_state_dict();
    let weights_path = out.join("model.safetensors");
    let mut tensors = HashMap::new();
    for (name, tensor) in state.iter() {
        if config.tie_word_embeddings && name == "lm_head.weight" {
            // safetensors forbids aliasing; the tied head is reconstructed at load.
            continue;
        }
        tensors.insert(name.clone(), tensor.clone());
    }
    candle_core::safetensors::save(&tensors, &weights_path)?;

    let moments_path = out.join("optimizer.safetensors");
    let mut moments = HashMap::new();
    let state_moments = opt.moments();
    for ((name, _), (first, second)) in state.iter().zip(state_moments) {
        moments.insert(format!("{name}.exp_avg"), first);
        moments.insert(format!("{name}.exp_avg_sq"), second);
    }
    candle_core::safetensors::save(&moments, &moments_path)?;

    let meta = serde_json::json!({
        "step": step,
        "lr": opt.state().lr,
        "beta1": opt.state().beta1,
        "beta2": opt.state().beta2,
        "eps": opt.state().eps,
        "weight_decay": opt.state().weight_decay,
        "optimizer_step": opt.state().step_t,
    });
    let meta_path = out.join("resume.json");
    let tmp_meta = out.join("resume.json.tmp");
    std::fs::write(&tmp_meta, serde_json::to_string_pretty(&meta)?)?;
    std::fs::rename(&tmp_meta, &meta_path)?;
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    println!(
        "{{\"step\": {step}, \"checkpoint\": \"{}\"}}",
        out.display()
    );
    Ok(())
}
