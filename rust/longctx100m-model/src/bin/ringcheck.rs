//! Ring attention parity checker: reads a JSON spec of shard tensors and
//! prints the ring attention forward output or backward gradients for each
//! rank, so Python can compare against the dense oracle / autograd.

use candle_core::{Device, Tensor};
use longctx100m_model::ring_attention::{
    ring_causal_attention_backward, ring_causal_attention_forward,
    ring_causal_attention_forward_with_stats,
};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: longctx100m-ringcheck <spec.json> [--backward]")?;
    let mode = args.next().unwrap_or_default();
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let world = spec["world_size"].as_u64().unwrap() as usize;
    let rank = spec["rank"].as_u64().unwrap() as usize;
    let block_q = spec["block_q"].as_u64().unwrap_or(2) as usize;
    let block_k = spec["block_k"].as_u64().unwrap_or(3) as usize;
    let valid_length = spec["valid_length"].as_u64().map(|v| v as usize);
    let q_position_start = spec["q_position_start"].as_u64().map(|v| v as usize);
    let device = Device::Cpu;
    let shards: Vec<(Tensor, Tensor, Tensor)> = (0..world)
        .map(|i| {
            let q = tensor_from_json(&spec["q"][i], &device)?;
            let k = tensor_from_json(&spec["k"][i], &device)?;
            let v = tensor_from_json(&spec["v"][i], &device)?;
            Ok((q, k, v))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    if mode == "--backward" {
        // Compute forward stats for every rank so the backward can circulate
        // Q/dO/O/m/l across query owners.
        let mut outs = Vec::new();
        let mut ms = Vec::new();
        let mut ls = Vec::new();
        for r in 0..world {
            let (o, m, l) = ring_causal_attention_forward_with_stats(
                &shards,
                r,
                true,
                block_q,
                block_k,
                valid_length,
                q_position_start,
            )?;
            outs.push(o);
            ms.push(m);
            ls.push(l);
        }
        let grad_out = tensor_from_json(&spec["grad_out"], &device)?;
        let grad_outs: Vec<Tensor> = (0..world).map(|_| grad_out.clone()).collect();
        let (dq, dk, dv) = ring_causal_attention_backward(
            &shards,
            rank,
            &outs,
            &ms,
            &ls,
            &grad_outs,
            true,
            block_q,
            block_k,
            valid_length,
            q_position_start,
        )?;
        let dqv: Vec<f32> = dq.flatten_all()?.to_vec1()?;
        let dkv: Vec<f32> = dk.flatten_all()?.to_vec1()?;
        let dvv: Vec<f32> = dv.flatten_all()?.to_vec1()?;
        println!(
            "{{\"rank\": {rank}, \"dq\": {:?}, \"dk\": {:?}, \"dv\": {:?}}}",
            dqv, dkv, dvv
        );
        return Ok(());
    }

    let out = ring_causal_attention_forward(
        &shards,
        rank,
        true,
        block_q,
        block_k,
        valid_length,
        q_position_start,
    )?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    println!("{{\"rank\": {rank}, \"output\": {:?}}}", vals);
    Ok(())
}

fn tensor_from_json(v: &Value, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let dims: Vec<usize> = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    Ok(Tensor::new(flat.as_slice(), device)?.reshape(dims)?)
}
