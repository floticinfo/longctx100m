use hf_hub::{
    Repo,
    api::sync::{ApiBuilder, ApiRepo},
};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use tempfile::Builder as TempDirBuilder;
use tokenizers::Tokenizer;
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ModelConfig {
    vocab_size: u64,
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    max_position_embeddings: u64,
    rope_theta: f64,
    intermediate_size: u64,
    rms_norm_eps: f64,
    tie_word_embeddings: bool,
    hidden_act: String,
    attention_bias: bool,
    mlp_bias: bool,
    rope_scaling: Option<Value>,
    num_key_value_heads: u64,
}
#[derive(Debug, Serialize)]
struct Manifest {
    model_id: String,
    revision: String,
    tokenizer_length: u64,
    loaded_tensors: usize,
    parameters: u64,
    source_max_position_embeddings: u64,
    target_max_position_embeddings: u64,
    rope_scaling: Option<Value>,
    attention_bias: bool,
    mlp_bias: bool,
    tie_word_embeddings: bool,
    attention: String,
    pretrained_for_100m_context: bool,
    long_context_training_required: bool,
}
#[derive(Debug, Clone)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
    source: PathBuf,
    offset: u64,
    len: u64,
}
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn get_u(v: &Value, k: &str) -> Result<u64, String> {
    v.get(k)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid {k}"))
}
fn get_f(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(Value::as_f64).unwrap_or(d)
}
fn get_b(v: &Value, k: &str, d: bool) -> bool {
    v.get(k).and_then(Value::as_bool).unwrap_or(d)
}
pub fn validate_revision(s: &str) -> Result<(), String> {
    if s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err("pin --revision to a 40-character immutable model commit SHA".into())
    }
}
fn parse_config(raw: &Value, target: u64) -> Result<ModelConfig, String> {
    if raw.get("model_type").and_then(Value::as_str) != Some("llama") {
        return Err("only plain Llama checkpoints are supported".into());
    }
    let heads = get_u(raw, "num_attention_heads")?;
    let rr = raw
        .get("rope_parameters")
        .filter(|v| !v.is_null())
        .or_else(|| raw.get("rope_scaling").filter(|v| !v.is_null()));
    let rope = rr.cloned().and_then(|v| {
        if v.get("rope_type")
            .or_else(|| v.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default")
            == "default"
        {
            None
        } else {
            Some(v)
        }
    });
    let c = ModelConfig {
        vocab_size: get_u(raw, "vocab_size")?,
        hidden_size: get_u(raw, "hidden_size")?,
        num_layers: get_u(raw, "num_hidden_layers")?,
        num_heads: heads,
        max_position_embeddings: target,
        rope_theta: get_f(raw, "rope_theta", 10000.),
        intermediate_size: get_u(raw, "intermediate_size")?,
        rms_norm_eps: get_f(raw, "rms_norm_eps", 1e-6),
        tie_word_embeddings: get_b(raw, "tie_word_embeddings", false),
        hidden_act: raw
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu")
            .into(),
        attention_bias: get_b(raw, "attention_bias", false),
        mlp_bias: get_b(raw, "mlp_bias", false),
        rope_scaling: rope,
        num_key_value_heads: raw
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .filter(|v| *v > 0)
            .unwrap_or(heads),
    };
    if [
        c.vocab_size,
        c.hidden_size,
        c.num_layers,
        c.num_heads,
        c.num_key_value_heads,
        c.intermediate_size,
        target,
    ]
    .contains(&0)
    {
        return Err("model dimensions must be positive".into());
    }
    if c.num_heads % c.num_key_value_heads != 0
        || c.hidden_size % c.num_heads != 0
        || (c.hidden_size / c.num_heads) % 2 != 0
    {
        return Err("invalid attention head dimensions".into());
    }
    if c.hidden_act != "silu" {
        return Err("only SiLU gated MLP is supported".into());
    }
    if let Some(r) = &c.rope_scaling {
        let k = r
            .get("rope_type")
            .or_else(|| r.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        if !["default", "linear", "llama3"].contains(&k) {
            return Err(format!("unsupported RoPE profile {k}"));
        }
        if r.get("rope_theta")
            .and_then(Value::as_f64)
            .is_some_and(|theta| (theta - c.rope_theta).abs() > f64::EPSILON)
        {
            return Err("RoPE theta differs between model and scaling config".into());
        }
        if r.get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .is_some_and(|factor| factor != 1.0)
        {
            return Err("partial rotary factors are not supported".into());
        }
        if k == "linear" && r.get("factor").and_then(Value::as_f64).unwrap_or(0.) <= 0. {
            return Err("linear RoPE factor must be positive".into());
        }
        if k == "llama3"
            && [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            ]
            .iter()
            .any(|x| r.get(x).is_none())
        {
            return Err("incomplete llama3 RoPE profile".into());
        }
    }
    if raw.get("num_experts").and_then(Value::as_u64).unwrap_or(0) > 0
        || raw
            .get("n_routed_experts")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    {
        return Err("MoE checkpoints are unsupported".into());
    }
    if raw
        .get("head_dim")
        .and_then(Value::as_u64)
        .is_some_and(|d| d != c.hidden_size / c.num_heads)
    {
        return Err("nonstandard head_dim is unsupported".into());
    }
    if raw
        .get("sliding_window")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        > 0
        || raw
            .get("attention_chunk_size")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    {
        return Err("sliding/chunked attention is unsupported".into());
    }
    Ok(c)
}
fn expected_shapes(c: &ModelConfig) -> BTreeMap<String, Vec<usize>> {
    let mut m = BTreeMap::new();
    let h = c.hidden_size as usize;
    let v = c.vocab_size as usize;
    let inter = c.intermediate_size as usize;
    let heads = c.num_heads as usize;
    let kv = c.num_key_value_heads as usize;
    let d = h / heads;
    m.insert("model.embed_tokens.weight".into(), vec![v, h]);
    for i in 0..c.num_layers {
        let p = format!("model.layers.{i}");
        m.insert(format!("{p}.input_layernorm.weight"), vec![h]);
        m.insert(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        for (n, s) in [
            ("q_proj", vec![heads * d, h]),
            ("k_proj", vec![kv * d, h]),
            ("v_proj", vec![kv * d, h]),
            ("o_proj", vec![h, h]),
        ] {
            m.insert(format!("{p}.self_attn.{n}.weight"), s.clone());
            if c.attention_bias {
                m.insert(format!("{p}.self_attn.{n}.bias"), vec![s[0]]);
            }
        }
        for (n, s) in [
            ("gate_proj", vec![inter, h]),
            ("up_proj", vec![inter, h]),
            ("down_proj", vec![h, inter]),
        ] {
            m.insert(format!("{p}.mlp.{n}.weight"), s.clone());
            if c.mlp_bias {
                m.insert(format!("{p}.mlp.{n}.bias"), vec![s[0]]);
            }
        }
    }
    m.insert("model.norm.weight".into(), vec![h]);
    m.insert("lm_head.weight".into(), vec![v, h]);
    m
}
fn tensor_bytes(d: Dtype, s: &[usize]) -> Option<u64> {
    s.iter()
        .try_fold(d.size() as u64, |n, x| n.checked_mul(*x as u64))
}
fn validate_state(c: &ModelConfig, specs: &BTreeMap<String, TensorSpec>) -> Result<(), String> {
    let mut shapes = expected_shapes(c);
    if c.tie_word_embeddings {
        shapes.remove("lm_head.weight");
    }
    let missing: Vec<_> = shapes
        .keys()
        .filter(|k| !specs.contains_key(*k) && !(c.tie_word_embeddings && *k == "lm_head.weight"))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "checkpoint missing required tensors: {:?}",
            missing.iter().take(8).collect::<Vec<_>>()
        ));
    }
    for (k, s) in shapes {
        if specs.get(&k).is_some_and(|v| v.shape != s) {
            return Err(format!("shape mismatch: {k}"));
        }
    }
    Ok(())
}
fn write_output(records: &BTreeMap<String, TensorSpec>, path: &Path) -> Result<(), String> {
    let mut hdata = BTreeMap::<String, Value>::new();
    let mut off = 0u64;
    for (k, s) in records {
        let end = off.checked_add(s.len).ok_or("checkpoint size overflow")?;
        hdata.insert(
            k.clone(),
            json!({"dtype":s.dtype,"shape":s.shape,"data_offsets":[off,end]}),
        );
        off = end
    }
    let mut header = serde_json::to_vec(&hdata).map_err(error)?;
    while header.len() % 8 != 0 {
        header.push(b' ')
    }
    let mut out = File::create(path).map_err(error)?;
    out.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(error)?;
    out.write_all(&header).map_err(error)?;
    let mut buf = vec![0u8; 1024 * 1024];
    for s in records.values() {
        let mut f = File::open(&s.source).map_err(error)?;
        f.seek(SeekFrom::Start(s.offset)).map_err(error)?;
        let mut left = s.len;
        while left > 0 {
            let n = std::cmp::min(left, buf.len() as u64) as usize;
            f.read_exact(&mut buf[..n]).map_err(error)?;
            out.write_all(&buf[..n]).map_err(error)?;
            left -= n as u64
        }
    }
    out.sync_all().map_err(error)?;
    Ok(())
}
fn fetch_optional(repo: &ApiRepo, name: &str) -> Result<Option<PathBuf>, String> {
    match repo.get(name) {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            let x = e.to_string();
            if x.contains("404")
                || x.contains("Not Found")
                || x.contains("no such file")
                || x.contains("does not exist")
            {
                Ok(None)
            } else {
                Err(x)
            }
        }
    }
}
pub fn run_import(
    model: &str,
    revision: &str,
    cache_dir: Option<&Path>,
    max_positions: u64,
    out: &Path,
) -> Result<Value, String> {
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("pin --revision to a 40-character immutable model commit SHA".into());
    }
    if max_positions == 0 {
        return Err("max-position-embeddings must be positive".into());
    }
    if out.exists() {
        return Err("output checkpoint directory already exists".into());
    }
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Some(c) = cache_dir {
        builder = builder.with_cache_dir(c.to_path_buf())
    }
    if let Ok(t) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        builder = builder.with_token(Some(t))
    }
    let api = builder.build().map_err(error)?;
    let repo = api.repo(Repo::with_revision(
        model.into(),
        hf_hub::RepoType::Model,
        revision.into(),
    ));
    let raw: Value =
        serde_json::from_slice(&fs::read(repo.get("config.json").map_err(error)?).map_err(error)?)
            .map_err(error)?;
    let config = parse_config(&raw, max_positions)?;
    let token_path = fetch_optional(&repo, "tokenizer.json")?;
    let tokenizer = if let Some(path) = &token_path {
        Some(Tokenizer::from_file(path).map_err(error)?)
    } else {
        None
    };
    let tokenizer_length = if let Some(t) = &tokenizer {
        t.get_vocab_size(true) as u64
    } else {
        get_u(&raw, "vocab_size")?
    };
    if tokenizer_length != config.vocab_size {
        return Err(format!(
            "tokenizer length {tokenizer_length} differs from embedding vocab {}",
            config.vocab_size
        ));
    }
    let info = repo.info().map_err(error)?;
    if info.sha != revision {
        return Err(format!(
            "resolved revision {} differs from requested {revision}",
            info.sha
        ));
    }
    let shards: Vec<_> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|s| s.ends_with(".safetensors"))
        .collect();
    if info
        .siblings
        .iter()
        .any(|s| s.rfilename == "model.safetensors.index.json")
    {
        return Err("sharded safetensors index is not supported in this import slice; single-file root checkpoints only".into());
    }
    if shards.is_empty() {
        return Err("root safetensors weights required; .bin unsupported".into());
    }
    let expected = expected_shapes(&config);
    let mut specs = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut dtype = None;
    let mut params = 0u64;
    for shard in shards {
        let path = repo.get(&shard).map_err(error)?;
        let file = File::open(&path).map_err(error)?;
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(error)?;
        let tensors = SafeTensors::deserialize(&map).map_err(error)?;
        for key in tensors.names() {
            if !seen.insert(key.clone()) {
                return Err(format!("duplicate tensor {key}"));
            }
            if config.tie_word_embeddings && key == "lm_head.weight" {
                let head = tensors.tensor(key).map_err(error)?;
                if head.shape() != expected["model.embed_tokens.weight"] {
                    return Err("tied lm_head shape differs from embedding weight".into());
                }
                continue;
            }
            let t = tensors.tensor(key).map_err(error)?;
            if !matches!(t.dtype(), Dtype::F16 | Dtype::BF16 | Dtype::F32) {
                return Err(format!("unsupported tensor dtype: {key}"));
            }
            if dtype.is_some_and(|d| d != t.dtype()) {
                return Err("mixed floating dtypes unsupported".into());
            }
            dtype = Some(t.dtype());
            let Some(shape) = expected.get(key) else {
                return Err(format!("unexpected tensor key {key}"));
            };
            if t.shape() != shape {
                return Err(format!("shape mismatch for {key}"));
            }
            if !(config.tie_word_embeddings && key == "lm_head.weight") {
                params = params
                    .checked_add(t.shape().iter().try_fold(1u64, |n, x| {
                        n.checked_mul(*x as u64).ok_or("parameter overflow")
                    })?)
                    .ok_or("parameter overflow")?;
            }
            let offset = t.data().as_ptr() as usize - map.as_ptr() as usize;
            specs.insert(
                key.clone(),
                TensorSpec {
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    source: path.clone(),
                    offset: offset as u64,
                    len: tensor_bytes(t.dtype(), t.shape()).ok_or("tensor size overflow")?,
                },
            );
        }
    }
    if config.tie_word_embeddings && !seen.contains("model.embed_tokens.weight") {
        return Err("tied embedding weights missing".into());
    }
    if config.tie_word_embeddings {
        // Python manifest counts the embedding and tied lm_head state-dict keys separately.
        let tied_head_parameters = config
            .vocab_size
            .checked_mul(config.hidden_size)
            .ok_or("parameter count overflow")?;
        params = params
            .checked_add(tied_head_parameters)
            .ok_or("parameter count overflow")?;
    }
    validate_state(&config, &specs)?;
    let source_max = raw
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(error)?;
    let stage_guard = TempDirBuilder::new()
        .prefix("longctx100m-import-")
        .tempdir_in(parent)
        .map_err(error)?;
    let stage = stage_guard.path().to_path_buf();
    let result = (|| -> Result<(), String> {
        fs::create_dir(stage.join("tokenizer")).map_err(error)?;
        write_output(&specs, &stage.join("model.safetensors.tmp"))?;
        fs::rename(
            stage.join("model.safetensors.tmp"),
            stage.join("model.safetensors"),
        )
        .map_err(error)?;
        fs::write(
            stage.join("config.json"),
            serde_json::to_vec_pretty(&config).map_err(error)?,
        )
        .map_err(error)?;
        let manifest = Manifest {
            model_id: model.into(),
            revision: revision.into(),
            tokenizer_length,
            loaded_tensors: seen.len(),
            parameters: params,
            source_max_position_embeddings: source_max,
            target_max_position_embeddings: config.max_position_embeddings,
            rope_scaling: config.rope_scaling.clone(),
            attention_bias: config.attention_bias,
            mlp_bias: config.mlp_bias,
            tie_word_embeddings: config.tie_word_embeddings,
            attention: "exact_dense_causal_ring".into(),
            pretrained_for_100m_context: false,
            long_context_training_required: true,
        };
        fs::write(
            stage.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).map_err(error)?,
        )
        .map_err(error)?;
        if let Some(path) = &token_path {
            fs::copy(path, stage.join("tokenizer/tokenizer.json")).map_err(error)?;
        }
        for n in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        ] {
            if let Some(p) = fetch_optional(&repo, n)? {
                fs::copy(p, stage.join("tokenizer").join(n)).map_err(error)?;
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        return Err(e);
    }
    let committed_path = stage_guard.keep();
    if let Err(e) = fs::rename(&committed_path, out) {
        let _ = fs::remove_dir_all(&committed_path);
        return Err(format!("checkpoint commit failed: {e}"));
    }
    Ok(
        json!({"model_id":model,"revision":revision,"tokenizer_length":tokenizer_length,"loaded_tensors":specs.len(),"parameters":params,"source_max_position_embeddings":source_max,"target_max_position_embeddings":config.max_position_embeddings,"rope_scaling":config.rope_scaling,"attention_bias":config.attention_bias,"mlp_bias":config.mlp_bias,"tie_word_embeddings":config.tie_word_embeddings,"attention":"exact_dense_causal_ring","pretrained_for_100m_context":false,"long_context_training_required":true,"output":out}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    fn config(tied: bool) -> Value {
        json!({"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":64,"rope_theta":10000.0,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":tied})
    }
    fn tensor_fixture(path: &Path, tied: bool) -> BTreeMap<String, TensorSpec> {
        let model = parse_config(&config(tied), 128).unwrap();
        let shapes = expected_shapes(&model);
        let mut values = Vec::new();
        for (name, shape) in shapes {
            let data = vec![0f32; shape.iter().product()];
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
            values.push((
                name,
                safetensors::tensor::TensorView::new(Dtype::F32, shape, bytes).unwrap(),
            ));
        }
        let bytes = safetensors::tensor::serialize(values, &None).unwrap();
        fs::write(path, bytes).unwrap();
        let file = File::open(path).unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();
        let tensors = SafeTensors::deserialize(&map).unwrap();
        let mut specs = BTreeMap::new();
        for name in tensors.names() {
            let t = tensors.tensor(name).unwrap();
            let offset = t.data().as_ptr() as usize - map.as_ptr() as usize;
            specs.insert(
                name.clone(),
                TensorSpec {
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    source: path.to_path_buf(),
                    offset: offset as u64,
                    len: t.data().len() as u64,
                },
            );
        }
        specs
    }
    #[test]
    fn importer_maps_supported_llama_and_linear_rope() {
        let mut r = config(false);
        r["rope_scaling"] = json!({"rope_type":"linear","factor":2.});
        let c = parse_config(&r, 1000).unwrap();
        assert_eq!(c.num_key_value_heads, 1);
        assert_eq!(c.max_position_embeddings, 1000);
        assert_eq!(c.rope_scaling.unwrap()["factor"], 2.);
    }
    #[test]
    fn importer_rejects_unsupported_model_features() {
        let mut r = config(false);
        r["model_type"] = json!("mistral");
        assert!(parse_config(&r, 100).unwrap_err().contains("plain Llama"));
        let mut r = config(false);
        r["num_experts"] = json!(4);
        assert!(parse_config(&r, 100).unwrap_err().contains("MoE"));
        let mut r = config(false);
        r["sliding_window"] = json!(16);
        assert!(parse_config(&r, 100).unwrap_err().contains("sliding"));
    }
    #[test]
    fn importer_requires_pinned_revision() {
        assert!(validate_revision(&"a".repeat(40)).is_ok());
        assert!(validate_revision("main").is_err());
    }
    #[test]
    fn importer_checks_exact_required_tensor_inventory() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("model.safetensors");
        let mut specs = tensor_fixture(&path, false);
        let c = parse_config(&config(false), 128).unwrap();
        assert!(validate_state(&c, &specs).is_ok());
        specs.remove("model.layers.0.self_attn.q_proj.weight");
        assert!(validate_state(&c, &specs).unwrap_err().contains("missing"));
    }
    #[test]
    fn importer_streams_standard_safetensors_and_supports_tied_head() {
        let td = TempDir::new().unwrap();
        let source = td.path().join("source.safetensors");
        let specs = tensor_fixture(&source, true);
        let c = parse_config(&config(true), 128).unwrap();
        assert!(validate_state(&c, &specs).is_ok());
        let out = td.path().join("model.safetensors");
        write_output(&specs, &out).unwrap();
        let bytes = fs::read(out).unwrap();
        let saved = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(saved.names().len(), specs.len());
        for (name, spec) in specs {
            let t = saved.tensor(&name).unwrap();
            assert_eq!(t.dtype(), spec.dtype);
            assert_eq!(t.shape(), spec.shape);
            assert_eq!(t.data().len() as u64, spec.len);
        }
    }
}
