use base64::{
    Engine as _,
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode, Uri,
    body::Incoming,
    header::{AUTHORIZATION, CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use longctx100m_core::BackendCapabilities;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{convert::Infallible, env, net::ToSocketAddrs, path::PathBuf, sync::Arc, time::Duration};
use subtle::ConstantTimeEq;
use tokio::{fs::File, io::AsyncWriteExt, net::TcpListener, sync::Semaphore};
const CHUNK: usize = 1024 * 1024;
const QUESTION_MAX_BYTES: usize = 128 * 1024;
const QUESTION_MAX_B64_BYTES: usize = 180_000;
const RESPONSE_MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
struct Settings {
    host: String,
    port: u16,
    backend_url: String,
    backend_api_key: String,
    api_key: String,
    required_input_tokens: u64,
    max_upload_bytes: u64,
    timeout_seconds: u64,
    temp_dir: PathBuf,
    backend_prefix: String,
}
impl Settings {
    fn from_env() -> Result<Self, String> {
        let host = env::var("LONGCTX_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = env_num("LONGCTX_PORT", 8080)?;
        if !(1..=65535).contains(&port) {
            return Err("LONGCTX_PORT must be between 1 and 65535".into());
        }
        let api_key = env::var("LONGCTX_API_KEY").unwrap_or_default();
        if !["127.0.0.1", "localhost", "::1"].contains(&host.as_str()) && api_key.is_empty() {
            return Err("LONGCTX_API_KEY is required when binding beyond localhost".into());
        }
        let backend_url = env::var("LONGCTX_BACKEND_URL")
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string();
        let mut backend_prefix = String::new();
        if !backend_url.is_empty() {
            let u: Uri = backend_url
                .parse()
                .map_err(|_| "LONGCTX_BACKEND_URL must be an absolute http(s) URL")?;
            let scheme = u.scheme_str().unwrap_or("");
            let authority = u
                .authority()
                .ok_or("LONGCTX_BACKEND_URL must be an absolute http(s) URL")?;
            if !["http", "https"].contains(&scheme) || u.query().is_some() {
                return Err(
                    "LONGCTX_BACKEND_URL must be an absolute http(s) URL without query or fragment"
                        .into(),
                );
            }
            backend_prefix = u.path().trim_end_matches('/').to_string();
            if scheme != "https" && !["127.0.0.1", "localhost", "::1"].contains(&authority.host()) {
                return Err("Remote LONGCTX_BACKEND_URL must use HTTPS".into());
            }
        }
        Ok(Self {
            host,
            port: port as u16,
            backend_url,
            backend_api_key: env::var("LONGCTX_BACKEND_API_KEY").unwrap_or_default(),
            api_key,
            required_input_tokens: env_num("LONGCTX_REQUIRED_INPUT_TOKENS", 100_000_000)?,
            max_upload_bytes: env_num("LONGCTX_MAX_UPLOAD_BYTES", 64 * 1024 * 1024 * 1024)?,
            timeout_seconds: env_num("LONGCTX_REQUEST_TIMEOUT_SECONDS", 7200)?,
            temp_dir: env::var_os("LONGCTX_TEMP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(env::temp_dir),
            backend_prefix,
        })
    }
}
fn env_num(key: &str, default: u64) -> Result<u64, String> {
    let n = match env::var(key) {
        Ok(v) => v.parse().map_err(|_| format!("{key} must be an integer"))?,
        Err(_) => default,
    };
    if n == 0 {
        return Err(format!("{key} must be positive"));
    }
    Ok(n)
}
#[derive(Clone)]
struct App {
    cfg: Arc<Settings>,
    client: Client,
    slots: Arc<Semaphore>,
}
#[derive(Deserialize)]
struct TokenCount {
    input_tokens: u64,
    input_sha256: String,
}
#[derive(Deserialize)]
struct BackendResult {
    answer: String,
    model_id: String,
    tokenizer_id: String,
    input_tokens: u64,
    processed_input_tokens: u64,
    input_sha256: String,
    context_limit_tokens: u64,
    attention_mode: String,
    truncated: bool,
}
fn json_response(status: StatusCode, value: Value) -> Response<BoxBody<Bytes, Infallible>> {
    let bytes = serde_json::to_vec(&value)
        .unwrap_or_else(|_| b"{\"error\":\"serialization error\"}".to_vec());
    let mut r = Response::new(
        Full::new(Bytes::from(bytes))
            .map_err(|never| match never {})
            .boxed(),
    );
    *r.status_mut() = status;
    r.headers_mut().insert(
        CONTENT_TYPE,
        "application/json; charset=utf-8".parse().unwrap(),
    );
    r.headers_mut()
        .insert(CACHE_CONTROL, "no-store".parse().unwrap());
    r.headers_mut()
        .insert("x-content-type-options", "nosniff".parse().unwrap());
    r
}
fn err(status: StatusCode, message: impl Into<String>) -> Response<BoxBody<Bytes, Infallible>> {
    json_response(status, json!({"error":message.into()}))
}
fn authorized(req: &Request<Incoming>, key: &str) -> bool {
    if key.is_empty() {
        return true;
    }
    let Some(value) = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };
    let Some(given) = value.strip_prefix("Bearer ") else {
        return false;
    };
    bool::from(given.as_bytes().ct_eq(key.as_bytes()))
}

fn backend_endpoint(app: &App, path: &str) -> String {
    let uri: Uri = app.cfg.backend_url.parse().expect("validated backend URL");
    format!(
        "{}://{}{}{}",
        uri.scheme_str().unwrap(),
        uri.authority().unwrap(),
        app.cfg.backend_prefix,
        path
    )
}
async fn backend_get(app: &App, path: &str) -> Result<Value, (StatusCode, String)> {
    let mut r = app.client.get(backend_endpoint(app, path));
    if !app.cfg.backend_api_key.is_empty() {
        r = r.bearer_auth(&app.cfg.backend_api_key)
    }
    let mut res = r.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("backend request failed: {e}"),
        )
    })?;
    if !res.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("backend returned HTTP {}", res.status()),
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = res.chunk().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("backend response read failed: {e}"),
        )
    })? {
        if bytes.len().saturating_add(chunk.len()) > 1024 * 1024 {
            return Err((
                StatusCode::BAD_GATEWAY,
                "backend capabilities response exceeded 1 MiB".into(),
            ));
        }
        bytes.extend_from_slice(&chunk)
    }
    serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("invalid backend JSON: {e}"),
        )
    })
}
async fn post_file(
    app: &App,
    path: &str,
    file: File,
    len: u64,
    question: &str,
    digest: &str,
) -> Result<reqwest::Response, (StatusCode, String)> {
    let body_stream =
        stream::try_unfold((file, vec![0u8; CHUNK]), |(mut file, mut buf)| async move {
            let n = tokio::io::AsyncReadExt::read(&mut file, &mut buf).await?;
            if n == 0 {
                Ok::<_, std::io::Error>(None)
            } else {
                let chunk = Bytes::copy_from_slice(&buf[..n]);
                Ok::<_, std::io::Error>(Some((chunk, (file, buf))))
            }
        });
    let mut req = app
        .client
        .post(backend_endpoint(app, path))
        .header(CONTENT_TYPE, "application/octet-stream")
        .header(CONTENT_LENGTH, len)
        .header("x-question-b64", URL_SAFE_NO_PAD.encode(question))
        .header("x-input-sha256", digest)
        .body(reqwest::Body::wrap_stream(body_stream));
    if !app.cfg.backend_api_key.is_empty() {
        req = req.bearer_auth(&app.cfg.backend_api_key)
    }
    req.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("backend request failed: {e}"),
        )
    })
}

async fn handle(
    app: App,
    req: Request<Incoming>,
) -> Result<Response<BoxBody<Bytes, Infallible>>, Infallible> {
    let response = route(app, req).await;
    Ok(response)
}
async fn route(app: App, mut req: Request<Incoming>) -> Response<BoxBody<Bytes, Infallible>> {
    let route_path = req
        .uri()
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or(req.uri().path());
    if !["/", "/healthz", "/api/capabilities", "/api/infer"].contains(&req.uri().path()) {
        return err(StatusCode::NOT_FOUND, "not found");
    }
    match (req.method(), route_path) {
        (&Method::GET, "/healthz") => json_response(StatusCode::OK, json!({"status":"ok"})),
        (&Method::GET, "/") => {
            let html = include_bytes!("../../../src/longctx100m/web/index.html");
            let mut r = Response::new(
                Full::new(Bytes::from_static(html))
                    .map_err(|never| match never {})
                    .boxed(),
            );
            r.headers_mut()
                .insert(CONTENT_TYPE, "text/html; charset=utf-8".parse().unwrap());
            r.headers_mut()
                .insert(CACHE_CONTROL, "no-store".parse().unwrap());
            r.headers_mut()
                .insert("x-content-type-options", "nosniff".parse().unwrap());
            r.headers_mut().insert("content-security-policy","default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'".parse().unwrap());
            r
        }
        (&Method::GET, "/api/capabilities") => {
            if !authorized(&req, &app.cfg.api_key) {
                return err(StatusCode::UNAUTHORIZED, "authentication required");
            }
            if app.cfg.backend_url.is_empty() {
                return err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "inference backend is not configured",
                );
            }
            let value = match backend_get(&app, "/v1/long-context/capabilities").await {
                Ok(v) => v,
                Err((s, e)) => return err(s, e),
            };
            let caps = match BackendCapabilities::from_value(&value) {
                Ok(c) => c,
                Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
            };
            if let Err(e) = caps.require_context(app.cfg.required_input_tokens, 0) {
                return err(StatusCode::SERVICE_UNAVAILABLE, e.to_string());
            }
            json_response(
                StatusCode::OK,
                json!({"required_input_tokens":app.cfg.required_input_tokens,"backend":{"model_id":caps.model_id,"tokenizer_id":caps.tokenizer_id,"context_limit_tokens":caps.context_limit_tokens,"attention_mode":caps.attention_mode,"supports_single_request":caps.supports_single_request,"truncation_policy":caps.truncation_policy}}),
            )
        }
        (&Method::POST, "/api/infer") => infer(&app, &mut req).await,
        _ => err(StatusCode::NOT_FOUND, "not found"),
    }
}
fn header_str<'a>(req: &'a Request<Incoming>, name: &'static str) -> Option<&'a str> {
    req.headers().get(name).and_then(|v| v.to_str().ok())
}
async fn infer(app: &App, req: &mut Request<Incoming>) -> Response<BoxBody<Bytes, Infallible>> {
    if !authorized(req, &app.cfg.api_key) {
        return err(StatusCode::UNAUTHORIZED, "authentication required");
    }
    if app.cfg.backend_url.is_empty() {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference backend is not configured",
        );
    }
    if header_str(req, "transfer-encoding").is_some() {
        return err(StatusCode::LENGTH_REQUIRED, "Content-Length is required");
    }
    let Some(length_s) = header_str(req, CONTENT_LENGTH.as_str()) else {
        return err(
            StatusCode::LENGTH_REQUIRED,
            "valid Content-Length is required",
        );
    };
    let Ok(length) = length_s.parse::<u64>() else {
        return err(
            StatusCode::LENGTH_REQUIRED,
            "valid Content-Length is required",
        );
    };
    if length == 0 {
        return err(StatusCode::BAD_REQUEST, "input file is empty");
    }
    if length > app.cfg.max_upload_bytes {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "upload exceeds configured byte limit",
        );
    }
    if header_str(req, CONTENT_ENCODING.as_str())
        .unwrap_or("identity")
        .to_ascii_lowercase()
        != "identity"
    {
        return err(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "compressed upload unsupported",
        );
    }
    if !header_str(req, CONTENT_TYPE.as_str())
        .unwrap_or("")
        .starts_with("text/")
    {
        return err(StatusCode::UNSUPPORTED_MEDIA_TYPE, "upload must be text");
    }
    let Some(encoded) = header_str(req, "x-question-b64") else {
        return err(StatusCode::BAD_REQUEST, "question missing or too large");
    };
    if encoded.is_empty() || encoded.len() > QUESTION_MAX_B64_BYTES {
        return err(StatusCode::BAD_REQUEST, "question missing or too large");
    }
    let raw = match URL_SAFE_NO_PAD
        .decode(encoded)
        .or_else(|_| URL_SAFE.decode(encoded))
    {
        Ok(v) => v,
        Err(_) => return err(StatusCode::BAD_REQUEST, "invalid question base64"),
    };
    if raw.len() > QUESTION_MAX_BYTES {
        return err(StatusCode::BAD_REQUEST, "question too large");
    };
    let question = match String::from_utf8(raw) {
        Ok(v) => v.trim().to_owned(),
        Err(_) => return err(StatusCode::BAD_REQUEST, "question must be UTF-8"),
    };
    if question.is_empty() {
        return err(StatusCode::BAD_REQUEST, "invalid question or output limit");
    }
    let output_tokens = match header_str(req, "x-max-output-tokens")
        .unwrap_or("2048")
        .parse::<u64>()
    {
        Ok(n) if (1..=32768).contains(&n) => n,
        _ => return err(StatusCode::BAD_REQUEST, "invalid question or output limit"),
    };
    let _permit = match app.slots.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return err(
                StatusCode::TOO_MANY_REQUESTS,
                "long-context inference capacity is busy; retry later",
            );
        }
    };

    async fn json_bytes(res: reqwest::Response) -> Result<Value, (StatusCode, String)> {
        if !res.status().is_success() {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!("backend returned HTTP {}", res.status()),
            ));
        }
        let mut response = res;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("backend response read failed: {e}"),
            )
        })? {
            if bytes.len().saturating_add(chunk.len()) > RESPONSE_MAX_BYTES {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    "backend response exceeded 16 MiB".into(),
                ));
            }
            bytes.extend_from_slice(&chunk)
        }
        serde_json::from_slice(&bytes).map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("invalid backend JSON: {e}"),
            )
        })
    }

    let tmp = match tempfile::Builder::new()
        .prefix("longctx-input-")
        .suffix(".txt")
        .tempfile_in(&app.cfg.temp_dir)
    {
        Ok(f) => f,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("upload failed: {e}")),
    };
    let (std_file, temp_path) = tmp.into_parts();
    let keep_path = temp_path.to_path_buf();
    let path = keep_path.clone();
    let mut file = File::from_std(std_file);
    let mut remaining = length;
    let mut hasher = Sha256::new();
    let mut pending = Vec::<u8>::new();
    while remaining > 0 {
        let next = match req.body_mut().frame().await {
            Some(Ok(frame)) => match frame.into_data() {
                Ok(d) => d,
                Err(_) => continue,
            },
            Some(Err(e)) => {
                return err(StatusCode::BAD_REQUEST, format!("upload failed: {e}"));
            }
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "upload ended before Content-Length",
                );
            }
        };
        let take = std::cmp::min(next.len() as u64, remaining) as usize;
        let data = &next[..take];
        hasher.update(data);
        pending.extend_from_slice(data);
        match std::str::from_utf8(&pending) {
            Ok(_) => pending.clear(),
            Err(e) if e.error_len().is_none() => {
                let valid = e.valid_up_to();
                pending.drain(..valid);
            }
            Err(_) => {
                return err(StatusCode::BAD_REQUEST, "upload failed: invalid UTF-8");
            }
        }
        if file.write_all(data).await.is_err() {
            return err(
                StatusCode::BAD_REQUEST,
                "upload failed: could not stage input",
            );
        }
        remaining -= take as u64;
    }
    if !pending.is_empty() {
        return err(StatusCode::BAD_REQUEST, "upload failed: invalid UTF-8");
    };
    if file.flush().await.is_err() {
        return err(
            StatusCode::BAD_REQUEST,
            "upload failed: could not stage input",
        );
    }
    let digest = format!("{:x}", hasher.finalize());
    drop(file);
    let _temp_path = temp_path;
    let caps_value = match backend_get(app, "/v1/long-context/capabilities").await {
        Ok(v) => v,
        Err((s, e)) => {
            return err(s, e);
        }
    };
    let caps = match BackendCapabilities::from_value(&caps_value) {
        Ok(c) => c,
        Err(e) => {
            return err(StatusCode::SERVICE_UNAVAILABLE, e.to_string());
        }
    };
    if let Err(e) = caps.require_context(app.cfg.required_input_tokens, output_tokens) {
        return err(StatusCode::SERVICE_UNAVAILABLE, e.to_string());
    }
    let tok_file = match File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                format!("cannot reopen staged input: {e}"),
            );
        }
    };
    let tok_res = match post_file(
        app,
        "/v1/long-context/tokenize",
        tok_file,
        length,
        &question,
        &digest,
    )
    .await
    {
        Ok(r) => r,
        Err((s, e)) => {
            return err(s, e);
        }
    };
    let tok_value = match json_bytes(tok_res).await {
        Ok(v) => v,
        Err((s, e)) => {
            return err(s, e);
        }
    };
    let count = match serde_json::from_value::<TokenCount>(tok_value) {
        Ok(c) if c.input_tokens > 0 && c.input_sha256.eq_ignore_ascii_case(&digest) => {
            c.input_tokens
        }
        Ok(_) => {
            return err(
                StatusCode::BAD_GATEWAY,
                "tokenizer digest mismatch or invalid token count",
            );
        }
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("invalid tokenizer response: {e}"),
            );
        }
    };
    if count < app.cfg.required_input_tokens {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "input contains {count} tokens; at least {} are required",
                app.cfg.required_input_tokens
            ),
        );
    }
    let total = match count.checked_add(output_tokens) {
        Some(n) => n,
        None => {
            return err(
                StatusCode::PAYLOAD_TOO_LARGE,
                "input plus output budget overflows",
            );
        }
    };
    if total > caps.context_limit_tokens {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "input plus output budget is {total} tokens; backend limit is {}",
                caps.context_limit_tokens
            ),
        );
    }
    let infer_file = match File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                format!("cannot reopen staged input: {e}"),
            );
        }
    };
    let infer_res = match post_inference(
        app,
        infer_file,
        length,
        &question,
        &digest,
        count,
        output_tokens,
        &caps,
    )
    .await
    {
        Ok(r) => r,
        Err((s, e)) => {
            return err(s, e);
        }
    };
    let value = match json_bytes(infer_res).await {
        Ok(v) => v,
        Err((s, e)) => {
            return err(s, e);
        }
    };
    let result = match serde_json::from_value::<BackendResult>(value) {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("invalid inference result: {e}"),
            );
        }
    };
    let check = verify(
        &result,
        &digest,
        app.cfg.required_input_tokens,
        total,
        output_tokens,
        &caps,
    );
    if let Err(e) = check {
        return err(
            StatusCode::BAD_GATEWAY,
            format!("Backend response failed verification: {e}"),
        );
    };
    drop(_temp_path);
    json_response(
        StatusCode::OK,
        json!({"answer":result.answer,"model_id":result.model_id,"tokenizer_id":result.tokenizer_id,"input_tokens":result.input_tokens,"processed_input_tokens":result.processed_input_tokens,"input_sha256":result.input_sha256,"context_limit_tokens":result.context_limit_tokens,"attention_mode":result.attention_mode,"truncated":result.truncated}),
    )
}

async fn post_inference(
    app: &App,
    file: File,
    length: u64,
    question: &str,
    digest: &str,
    count: u64,
    out: u64,
    caps: &BackendCapabilities,
) -> Result<reqwest::Response, (StatusCode, String)> {
    let body_stream = stream::try_unfold((file, vec![0u8; CHUNK]), |(mut f, mut b)| async move {
        let n = tokio::io::AsyncReadExt::read(&mut f, &mut b).await?;
        if n == 0 {
            Ok::<_, std::io::Error>(None)
        } else {
            let chunk = Bytes::copy_from_slice(&b[..n]);
            Ok::<_, std::io::Error>(Some((chunk, (f, b))))
        }
    });
    let ctx = count
        .checked_add(out)
        .ok_or((StatusCode::BAD_REQUEST, "context budget overflow".into()))?;
    let mut req = app
        .client
        .post(backend_endpoint(app, "/v1/long-context/infer"))
        .header(CONTENT_TYPE, "application/octet-stream")
        .header(CONTENT_LENGTH, length)
        .header("x-question-b64", URL_SAFE_NO_PAD.encode(question))
        .header("x-input-sha256", digest)
        .header("x-required-input-tokens", count)
        .header("x-minimum-input-tokens", app.cfg.required_input_tokens)
        .header("x-max-output-tokens", out)
        .header("x-required-context-tokens", ctx)
        .header(
            "x-capability-model-id",
            URL_SAFE_NO_PAD.encode(caps.model_id.as_bytes()),
        )
        .body(reqwest::Body::wrap_stream(body_stream));
    if !app.cfg.backend_api_key.is_empty() {
        req = req.bearer_auth(&app.cfg.backend_api_key)
    }
    req.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("inference backend request failed: {e}"),
        )
    })
}
fn verify(
    r: &BackendResult,
    digest: &str,
    min: u64,
    required_context: u64,
    out: u64,
    c: &BackendCapabilities,
) -> Result<(), String> {
    if r.truncated {
        return Err("Backend reports that the input was truncated".into());
    }
    if !r.input_sha256.eq_ignore_ascii_case(digest) {
        return Err("Backend input digest does not match the complete uploaded input".into());
    }
    if r.input_tokens < min {
        return Err(format!(
            "Backend counted {} input tokens, below the required {}",
            r.input_tokens, min
        ));
    }
    if r.input_tokens != r.processed_input_tokens {
        return Err("Backend did not process every input token it counted".into());
    }
    if r.input_tokens
        .checked_add(out)
        .map_or(true, |n| n > r.context_limit_tokens)
    {
        return Err("Backend input plus requested output exceeds its own context limit".into());
    }
    if r.context_limit_tokens < required_context {
        return Err("Backend result context limit is below input plus output budget".into());
    }
    if r.model_id != c.model_id || r.tokenizer_id != c.tokenizer_id {
        return Err("Backend model/tokenizer changed between preflight and inference".into());
    }
    if r.attention_mode != c.attention_mode {
        return Err("Backend attention mode changed between preflight and inference".into());
    }
    if !longctx100m_core::SUPPORTED_ATTENTION_MODES.contains(&r.attention_mode.as_str()) {
        return Err("Backend returned an unknown attention_mode".into());
    }
    Ok(())
}
fn print_help() {
    println!(
        "longctx100m-gateway — Rust HTTP gateway for the protocol-v1 long-context backend\n\nEnvironment variables:\n  LONGCTX_HOST                         Bind address (default 127.0.0.1)\n  LONGCTX_PORT                         HTTP port (default 8080)\n  LONGCTX_BACKEND_URL                  Existing protocol-v1 backend base URL\n  LONGCTX_API_KEY                      Optional client bearer key (required off localhost)\n  LONGCTX_BACKEND_API_KEY              Optional backend bearer key\n  LONGCTX_REQUIRED_INPUT_TOKENS        Minimum tokenizer count (default 100000000)\n  LONGCTX_MAX_UPLOAD_BYTES             Upload byte limit (default 64 GiB)\n  LONGCTX_REQUEST_TIMEOUT_SECONDS      Backend timeout (default 7200)\n  LONGCTX_TEMP_DIR                     Private upload staging directory\n"
    );
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_help();
        return Ok(());
    }
    let cfg = Arc::new(Settings::from_env().map_err(std::io::Error::other)?);
    let timeout = Duration::from_secs(cfg.timeout_seconds);
    let client = Client::builder()
        .connect_timeout(timeout.min(Duration::from_secs(30)))
        .timeout(timeout)
        .build()?;
    let app = App {
        cfg: cfg.clone(),
        client,
        slots: Arc::new(Semaphore::new(2)),
    };
    let bind_addr = (cfg.host.as_str(), cfg.port)
        .to_socket_addrs()?
        .next()
        .ok_or("LONGCTX_HOST did not resolve to an address")?;
    let listener = TcpListener::bind(bind_addr).await?;
    eprintln!(
        "Rust long-context gateway listening on {}:{}",
        cfg.host, cfg.port
    );
    loop {
        let (stream, _) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req| handle(app.clone(), req));
            if let Err(e) = http1::Builder::new().serve_connection(io, service).await {
                eprintln!("gateway connection failed: {e}")
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::{Response as HResponse, body::Bytes as HBytes};
    use serde_json::json;
    use std::convert::Infallible;
    async fn serve(
        listener: TcpListener,
        handler: impl Fn(
            Request<Incoming>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<HResponse<Full<HBytes>>, Infallible>>
                    + Send,
            >,
        > + Send
        + Sync
        + 'static,
    ) -> tokio::task::JoinHandle<()> {
        let h = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let h = h.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |req| h(req));
                    let _ = http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        })
    }
    fn response(value: Value) -> HResponse<Full<HBytes>> {
        HResponse::new(Full::new(HBytes::from(serde_json::to_vec(&value).unwrap())))
    }
    async fn fixture() -> (
        String,
        App,
        tokio::task::JoinHandle<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let backend_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend_addr = backend_listener.local_addr().unwrap();
        let _backend_task = serve(backend_listener, |req| Box::pin(async move {
            let request_path = req
                .uri()
                .path()
                .strip_prefix("/prefix")
                .unwrap_or(req.uri().path())
                .to_owned();
            let digest_header = req.headers().get("x-input-sha256").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
            let question_header = req.headers().get("x-question-b64").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
            let required_header = req.headers().get("x-required-input-tokens").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
            let context_header = req.headers().get("x-required-context-tokens").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
            let body = req.into_body().collect().await.unwrap().to_bytes();
            let is_short = body.as_ref() == b"short";
            let digest = format!("{:x}", Sha256::digest(&body));
            let value = match request_path.as_str() {
                "/v1/long-context/capabilities" => json!({"protocol_version":1,"model_id":"test-model","tokenizer_id":"test-tokenizer","context_limit_tokens":4000,"supports_single_request":true,"truncation_policy":"reject","attention_mode":"sparse_attention"}),
                "/v1/long-context/tokenize" => json!({"input_tokens":if is_short {3} else {10},"input_sha256":digest}),
                "/v1/long-context/infer" => { let headers_ok = question_header == "V2hhdCBpcyBpbiB0aGUgZmlsZT8" && required_header == "10" && context_header == "2058"; json!({"answer":if headers_ok {"verified"} else {"header mismatch"},"model_id":"test-model","tokenizer_id":"test-tokenizer","input_tokens":10,"processed_input_tokens":10,"input_sha256":digest,"context_limit_tokens":4000,"attention_mode":"sparse_attention","truncated":false,"received_header_digest":digest_header}) },
                _ => json!({"error":"not found"}),
            };
            Ok(response(value))
        })).await;
        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_addr = gateway_listener.local_addr().unwrap();
        let cfg = Arc::new(Settings {
            host: "127.0.0.1".into(),
            port: gateway_addr.port(),
            backend_url: format!("http://{backend_addr}/prefix"),
            backend_api_key: String::new(),
            api_key: "test-key".into(),
            required_input_tokens: 10,
            max_upload_bytes: 1024,
            timeout_seconds: 10,
            temp_dir: env::temp_dir(),
            backend_prefix: "/prefix".into(),
        });
        let app = App {
            cfg,
            client: Client::new(),
            slots: Arc::new(Semaphore::new(2)),
        };
        let app2 = app.clone();
        let gt = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = gateway_listener.accept().await else {
                    break;
                };
                let app = app2.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |req| handle(app.clone(), req));
                    let _ = http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        });
        (
            format!("http://{gateway_addr}"),
            app,
            gt,
            tokio::spawn(async {}),
        )
    }
    fn reqwest_test_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }
    #[tokio::test]
    async fn health_ui_auth_and_full_protocol_round_trip() {
        let (base, _app, server, _) = fixture().await;
        let c = reqwest_test_client();
        assert_eq!(
            c.get(format!("{base}/healthz"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let ui = c
            .get(format!("{base}/"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(ui.contains("Single-request long-context inference"));
        assert_eq!(
            c.get(format!("{base}/api/capabilities"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let caps = c
            .get(format!("{base}/api/capabilities"))
            .bearer_auth("test-key")
            .send()
            .await
            .unwrap();
        assert_eq!(caps.status(), StatusCode::OK);
        let body = "whole-file test ☃".as_bytes().to_vec();
        let digest = format!("{:x}", Sha256::digest(&body));
        let result = c
            .post(format!("{base}/api/infer"))
            .bearer_auth("test-key")
            .header(CONTENT_TYPE, "text/plain")
            .header("x-question-b64", "V2hhdCBpcyBpbiB0aGUgZmlsZT8")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        let result_status = result.status();
        let result_body = result.text().await.unwrap();
        assert_eq!(result_status, StatusCode::OK, "{result_body}");
        let value: Value = serde_json::from_str(&result_body).unwrap();
        assert_eq!(value["answer"], "verified");
        assert_eq!(value["input_sha256"], digest);
        server.abort();
    }
    #[tokio::test]
    async fn rejects_invalid_utf8_and_short_tokenized_inputs() {
        let (base, _app, server, _) = fixture().await;
        let c = reqwest_test_client();
        let invalid = c
            .post(format!("{base}/api/infer"))
            .bearer_auth("test-key")
            .header(CONTENT_TYPE, "text/plain")
            .header("x-question-b64", "cQ")
            .body(vec![0xff, 0xfe])
            .send()
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        let short = c
            .post(format!("{base}/api/infer"))
            .bearer_auth("test-key")
            .header(CONTENT_TYPE, "text/plain")
            .header("x-question-b64", "cQ")
            .body("short")
            .send()
            .await
            .unwrap();
        assert_eq!(short.status(), StatusCode::PAYLOAD_TOO_LARGE);
        server.abort();
    }
    #[test]
    fn response_attestation_requires_digest_complete_tokens_and_no_truncation() {
        let caps = BackendCapabilities::from_value(&json!({"protocol_version":1,"model_id":"m","tokenizer_id":"t","context_limit_tokens":100,"supports_single_request":true,"truncation_policy":"reject","attention_mode":"dense_full_attention"})).unwrap();
        let digest = "a".repeat(64);
        let mut result = BackendResult {
            answer: "ok".into(),
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            input_tokens: 10,
            processed_input_tokens: 10,
            input_sha256: digest.clone(),
            context_limit_tokens: 100,
            attention_mode: "dense_full_attention".into(),
            truncated: false,
        };
        assert!(verify(&result, &digest, 10, 20, 10, &caps).is_ok());
        result.truncated = true;
        assert!(
            verify(&result, &digest, 10, 20, 10, &caps)
                .unwrap_err()
                .contains("truncated")
        );
    }
}
