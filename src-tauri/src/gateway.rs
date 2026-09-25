//! Unified OpenAI-compatible reverse proxy gateway.
//!
//! Listens on a single local port (default 11434) and routes chat
//! completion requests to whichever running vLLM instance actually serves
//! the requested model. External tools (Cursor, Continue.dev, LibreChat,
//! any OpenAI-compatible client) can point at one stay-put endpoint
//! instead of juggling per-server ports.
//!
//! The HTTP layer is intentionally small (Tokio `TcpListener` + hand-rolled
//! HTTP/1.1 framing) to honor the project's "no unnecessary external
//! dependencies" constraint. Only read framing, two routes, and a streaming
//! reverse proxy are needed.

use crate::server::apply_vllm_auth;
use crate::state::{AppState, ServerDef, ServerStatus};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tauri::Manager;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Default gateway port (the one Ollama-style tools expect).
pub const DEFAULT_GATEWAY_PORT: u16 = 11434;
const MAX_HEAD_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, PartialEq)]
pub struct RequestHead {
    pub method: String,
    pub target: String,
    pub headers: HashMap<String, String>,
    pub content_length: usize,
}

/// All live servers of `task` status that are currently `Running`.
fn running_servers<'a>(
    servers: &'a BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<&'a crate::state::LiveServer> {
    servers
        .values()
        .filter(|ls| ls.status == ServerStatus::Running && ls.def.task == task)
        .collect()
}

/// Route a requested model name to the port of the running server that
/// serves it with the given task. Exact match on the served model name or
/// HF id wins first, then a fuzzy suffix match against both (so
/// `Qwen2.5-7B-Instruct` finds `Qwen/Qwen2.5-7B-Instruct`).
pub fn find_server_port_for(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
    task: &str,
) -> Option<u16> {
    let m = model.trim();
    if m.is_empty() {
        return None;
    }
    let running = running_servers(servers, task);
    for ls in &running {
        if ls.def.effective_model_name().eq_ignore_ascii_case(m)
            || ls.def.model_id.eq_ignore_ascii_case(m)
        {
            return Some(ls.def.port);
        }
    }
    let lower = m.to_lowercase();
    for ls in &running {
        let hf = ls.def.model_id.to_lowercase();
        let name = ls.def.effective_model_name().to_lowercase();
        if hf.ends_with(&lower) || name.ends_with(&lower) || lower.ends_with(&hf) {
            return Some(ls.def.port);
        }
    }
    None
}

fn configured_context_tokens(def: &ServerDef) -> Option<usize> {
    match def.backend.as_str() {
        "vllm" => def.max_model_len.filter(|value| *value > 0),
        "llamacpp" => def.ctx_size.filter(|value| *value > 0),
        _ => None,
    }
}

fn model_row(def: &ServerDef, context: Option<usize>, context_source: &str) -> serde_json::Value {
    let context = context.filter(|value| *value > 0);
    let is_instruct = def.task == "instruct";
    let mut row = serde_json::json!({
        "id": def.effective_model_name(),
        "object": "model",
        "created": 0,
        "owned_by": "local-llm-panel",
        "root": def.model_id.clone(),
        "parent": serde_json::Value::Null,
        // vLLM uses max_model_len and llama.cpp calls the same value n_ctx;
        // the other names make the effective context discoverable to clients
        // that use OpenAI/LM Studio-style model metadata instead.
        "max_model_len": context,
        "context_length": context,
        "max_context_length": context,
        "context_window": context,
        "n_ctx": context,
        "context_source": context_source,
        "backend": def.backend.clone(),
        "quantization": def.quant.clone(),
        "task": def.task.clone(),
        "capabilities": {
            "chat_completions": is_instruct,
            "completions": is_instruct,
            "embeddings": def.task == "embed",
            "streaming": is_instruct,
        },
        "port": def.port,
    });
    if let Some(params) = def.params_b.filter(|value| *value > 0.0) {
        row["parameters_billions"] = serde_json::json!(params);
    }
    row
}

/// OpenAI-style `data` rows for `GET /v1/models`. Prefer the served model
/// name when present (that is the name a client would pass in `"model"`).
///
/// The context fields are extensions to the OpenAI schema. They are
/// deliberately derived from the same definition used to launch the backend,
/// so a harness never sees a different context than the process is using.
pub fn model_rows(servers: &BTreeMap<String, crate::state::LiveServer>) -> Vec<serde_json::Value> {
    running_servers(servers, "instruct")
        .into_iter()
        .map(|ls| {
            let context = configured_context_tokens(&ls.def);
            model_row(
                &ls.def,
                context,
                if context.is_some() {
                    "configured"
                } else {
                    "unknown"
                },
            )
        })
        .collect()
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    let parsed = value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| {
            value
                .as_str()
                .and_then(|value| value.trim().parse::<usize>().ok())
        })?;
    (parsed > 0).then_some(parsed)
}

/// Read the context token limit from vLLM's model list or llama.cpp's
/// `/props` response. Different llama.cpp releases put `n_ctx` in slightly
/// different places, hence the small recursive lookup.
fn extract_runtime_context(value: &serde_json::Value) -> Option<usize> {
    const CONTEXT_KEYS: &[&str] = &[
        "max_model_len",
        "max_context_length",
        "context_length",
        "context_window",
        "n_ctx",
    ];
    if let Some(object) = value.as_object() {
        for key in CONTEXT_KEYS {
            if let Some(value) = object.get(*key).and_then(json_usize) {
                return Some(value);
            }
        }
        for key in [
            "data",
            "models",
            "default_generation_settings",
            "model_info",
        ] {
            if let Some(child) = object.get(key) {
                if let Some(context) = extract_runtime_context(child) {
                    return Some(context);
                }
            }
        }
    } else if let Some(items) = value.as_array() {
        for item in items {
            if let Some(context) = extract_runtime_context(item) {
                return Some(context);
            }
        }
    }
    None
}

async fn fetch_runtime_context(state: &Arc<AppState>, def: &ServerDef) -> Option<usize> {
    let api_key = crate::server::server_api_key(state, def);
    let paths: &[&str] = if def.backend == "llamacpp" {
        &["/props", "/v1/models"]
    } else {
        &["/v1/models"]
    };
    for path in paths {
        let url = format!("http://127.0.0.1:{}{path}", def.port);
        let request = apply_vllm_auth(state.http.get(&url), api_key.as_deref());
        let response =
            match tokio::time::timeout(std::time::Duration::from_millis(350), request.send()).await
            {
                Ok(Ok(response)) if response.status().is_success() => response,
                _ => continue,
            };
        let value = match tokio::time::timeout(
            std::time::Duration::from_millis(350),
            response.json::<serde_json::Value>(),
        )
        .await
        {
            Ok(Ok(value)) => value,
            _ => continue,
        };
        if let Some(context) = extract_runtime_context(&value) {
            return Some(context);
        }
    }
    None
}

async fn model_rows_for_state(state: &Arc<AppState>) -> Vec<serde_json::Value> {
    let defs: Vec<ServerDef> = {
        let servers = state.servers.lock().unwrap();
        running_servers(&servers, "instruct")
            .into_iter()
            .map(|server| server.def.clone())
            .collect()
    };
    let mut rows = Vec::with_capacity(defs.len());
    for def in defs {
        let configured = configured_context_tokens(&def);
        let (context, source) = if let Some(context) = configured {
            (Some(context), "configured")
        } else if let Some(context) = fetch_runtime_context(state, &def).await {
            (Some(context), "runtime")
        } else {
            (None, "unknown")
        };
        rows.push(model_row(&def, context, source));
    }
    rows
}

fn decode_path_component(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = *bytes.get(index + 1)?;
            let low = *bytes.get(index + 2)?;
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                b'A'..=b'F' => Some(byte - b'A' + 10),
                _ => None,
            };
            decoded.push((digit(high)? << 4) | digit(low)?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn model_id_from_path(path: &str) -> Option<String> {
    let raw = path.split('?').next()?.strip_prefix("/v1/models/")?;
    if raw.is_empty() {
        None
    } else {
        decode_path_component(raw)
    }
}

/// Extract the requested model id from a chat/completions JSON body.
pub fn model_from_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::to_string)
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

/// Validate the Authorization header against the configured gateway API key.
///
/// A gateway without a key is a misconfiguration, not an open server.  Return
/// a configuration error instead of silently exposing every routed model to
/// any local process (or browser page) that can reach localhost.
pub fn check_gateway_auth(
    headers: &HashMap<String, String>,
    global_key: Option<&str>,
) -> Result<(), (u16, String)> {
    let Some(expected_key) = global_key.map(str::trim).filter(|key| !key.is_empty()) else {
        return Err((
            503,
            "Gateway API key is not configured; set one in Settings before using /v1 routes"
                .to_string(),
        ));
    };
    let auth_header = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"));
    match auth_header {
        Some(h) if h.starts_with("Bearer ") => {
            let provided = h["Bearer ".len()..].trim();
            if !provided.is_empty() && constant_time_eq(provided, expected_key) {
                Ok(())
            } else {
                Err((401, "Invalid API key".to_string()))
            }
        }
        Some(_) => Err((
            401,
            "Authorization header must use Bearer scheme".to_string(),
        )),
        None => Err((401, "Missing Authorization header".to_string())),
    }
}

/// Is the chunk of bytes between the request line and the blank line a
/// valid `Content-Length: N` header we must read before routing?
pub fn parse_head(bytes: &[u8]) -> Option<RequestHead> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    if method.is_empty() || target.is_empty() {
        return None;
    }
    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let mut it = line.splitn(2, ':');
        let name = it.next()?.trim().to_ascii_lowercase();
        let value = it.next().unwrap_or("").trim().to_string();
        if name == "content-length" {
            content_length = value.parse().unwrap_or(0);
        }
        headers.insert(name, value);
    }
    Some(RequestHead {
        method,
        target,
        headers,
        content_length,
    })
}

/// Non-origin CORS headers shared by all allowed responses.  Kept public for
/// callers that need to inspect the policy without trusting a request origin.
pub const CORS_HEADERS: &str = "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n";

/// Return CORS headers only for explicitly trusted local origins.
///
/// The gateway is bound to loopback, but a browser page on any origin can
/// still attempt a localhost request.  Echoing an arbitrary `Origin` (or `*`)
/// would turn the gateway into a cross-site credential/data exfiltration point.
/// The Tauri origins and loopback development origins are the only origins
/// accepted by the local panel.
pub fn cors_headers(origin: Option<&str>) -> String {
    let Some(origin) = origin.filter(|value| is_allowed_origin(value)) else {
        return String::new();
    };
    format!(
        "Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n{CORS_HEADERS}"
    )
}

fn is_allowed_origin(origin: &str) -> bool {
    if origin == "tauri://localhost" || origin == "https://tauri.localhost" {
        return true;
    }
    ["http://127.0.0.1:", "http://localhost:"]
        .iter()
        .any(|prefix| {
            origin.strip_prefix(prefix).is_some_and(|port| {
                !port.is_empty()
                    && port.bytes().all(|byte| byte.is_ascii_digit())
                    && port.parse::<u16>().is_ok_and(|number| number > 0)
            })
        })
}

pub fn is_head_terminated(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn running_model_names(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<String> {
    running_servers(servers, task)
        .into_iter()
        .map(|ls| ls.def.effective_model_name())
        .collect()
}

fn not_found_payload_with_hint(model: &str, kind: &str, running: &[String]) -> String {
    serde_json::json!({
        "error": {
            "message": format!(
                "no running {kind} server for model '{model}'. Running: [{}]",
                running.join(", ")
            ),
            "type": "invalid_request_error",
        },
        "running": running,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Connection serving
// ---------------------------------------------------------------------------

async fn serve_connection(stream: &mut TcpStream, state: &Arc<AppState>) -> Result<(), String> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 8192];
    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|e| format!("read request: {e}"))?;
        if n == 0 {
            return Err("connection closed before request completed".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(sep) = is_head_terminated(&buf) {
            let head =
                parse_head(&buf[..sep]).ok_or_else(|| "malformed request head".to_string())?;
            if head.content_length > MAX_BODY_BYTES {
                return Err("request body too large".into());
            }
            let total = sep + 4;
            while buf.len() < total + head.content_length {
                let m = stream
                    .read(&mut tmp)
                    .await
                    .map_err(|e| format!("read body: {e}"))?;
                if m == 0 {
                    return Err("connection closed during body".into());
                }
                buf.extend_from_slice(&tmp[..m]);
            }
            let body = buf[total..total + head.content_length].to_vec();
            return route(stream, state, &head, &body).await;
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err("request head too large".into());
        }
    }
}

async fn route(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    head: &RequestHead,
    body: &[u8],
) -> Result<(), String> {
    if head.method.as_str() == "OPTIONS" {
        if let Some(origin) = head.headers.get("origin") {
            if !is_allowed_origin(origin) {
                return write_request_response(stream, head, 403, "text/plain", b"origin not allowed").await;
            }
        }
        return write_request_response(stream, head, 204, "text/plain", b"").await;
    }

    // Auth check for all /v1/* routes except /health
    let is_protected = head.target.starts_with("/v1/") && head.target != "/health";
    if is_protected {
        let cfg = state.config();
        let global_key = cfg.advanced_settings.api_key.as_deref();
        if let Err((status, msg)) = check_gateway_auth(&head.headers, global_key) {
            return write_request_response(
                stream,
                head,
                status,
                "application/json",
                serde_json::json!({ "error": { "message": msg, "type": "authentication_error" } })
                    .to_string()
                    .as_bytes(),
            )
            .await;
        }
    }

    let path = head.target.split('?').next().unwrap_or("");
    match (head.method.as_str(), path) {
        ("GET", "/v1/models") => {
            let text = serde_json::json!({
                "object": "list",
                "data": model_rows_for_state(state).await,
            })
            .to_string();
            write_request_response(stream, head, 200, "application/json", text.as_bytes()).await
        }
        (candidate_path, _)
            if head.method == "GET" && candidate_path.starts_with("/v1/models/") =>
        {
            let Some(model_id) = model_id_from_path(candidate_path) else {
                return write_request_response(
                    stream,
                    head,
                    400,
                    "application/json",
                    serde_json::json!({
                        "error": {
                            "message": "Model id is required",
                            "type": "invalid_request_error",
                        }
                    })
                    .to_string()
                    .as_bytes(),
                )
                .await;
            };
            let rows = model_rows_for_state(state).await;
            let Some(row) = rows.into_iter().find(|row| {
                row["id"].as_str() == Some(model_id.as_str())
                    || row["root"].as_str() == Some(model_id.as_str())
            }) else {
                let payload = serde_json::json!({
                    "error": {
                        "code": "model_not_found",
                        "message": format!("Model '{model_id}' was not found"),
                        "type": "invalid_request_error",
                    }
                })
                .to_string();
                return write_request_response(stream, head, 404, "application/json", payload.as_bytes()).await;
            };
            write_request_response(stream, head, 200, "application/json", row.to_string().as_bytes()).await
        }
        ("GET", "/health") => {
            write_request_response(stream, head, 200, "application/json", b"{\"status\":\"ok\"}").await
        }
        ("POST", "/v1/chat/completions") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "instruct")
            };
            match port {
                Some(port) => proxy_chat(stream, state, port, body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "instruct");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "instruction", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        ("POST", "/v1/completions") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "instruct")
            };
            match port {
                Some(port) => proxy_post(stream, state, port, "/v1/completions", body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "instruct");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "instruction", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        ("POST", "/v1/embeddings") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "embed")
            };
            match port {
                Some(port) => proxy_post(stream, state, port, "/v1/embeddings", body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "embed");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "embedding", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        (method, "/v1/models") if method != "GET" => {
            write_request_response(stream, head, 405, "text/plain", b"method not allowed").await
        }
        _ => write_request_response(stream, head, 404, "text/plain", b"not found").await,
    }
}

async fn write_request_response(
    stream: &mut TcpStream,
    head: &RequestHead,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    write_response_with_origin(
        stream,
        status,
        content_type,
        body,
        head.headers.get("origin").map(String::as_str),
    )
    .await
}

async fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    write_response_with_origin(stream, status, content_type, body, None).await
}

async fn write_response_with_origin(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    origin: Option<&str>,
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let cors = cors_headers(origin);
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{cors}Connection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(body).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())
}

async fn proxy_chat(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    body: &[u8],
    origin: Option<&str>,
) -> Result<(), String> {
    proxy_post(stream, state, port, "/v1/chat/completions", body, origin).await
}

/// Forward a chat/completions request to the target vLLM port, preserving
/// streaming when the caller asked for `"stream": true`.
async fn proxy_post(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    upstream_path: &str,
    body: &[u8],
    origin: Option<&str>,
) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}{upstream_path}");
    let streaming = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    let api_key = state
        .config()
        .servers
        .iter()
        .find(|server| server.port == port)
        .and_then(|server| crate::server::server_api_key(state, server));
    let req = state
        .http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec());
    let resp = apply_vllm_auth(req, api_key.as_deref())
        .send()
        .await
        .map_err(|e| format!("upstream POST {url}: {e}"))?;

    let status = resp.status().as_u16();
    let reason = resp.status().canonical_reason().unwrap_or("OK");

    if streaming {
        let head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{}Connection: close\r\n\r\n",
            cors_headers(origin)
        );
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        let mut upstream = resp;
        loop {
            match upstream.chunk().await {
                Ok(Some(chunk)) => {
                    // Chunked framing: <hex size>\r\n<data>\r\n
                    let size = format!("{:x}\r\n", chunk.len());
                    stream
                        .write_all(size.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                    stream.write_all(&chunk).await.map_err(|e| e.to_string())?;
                    stream.write_all(b"\r\n").await.map_err(|e| e.to_string())?;
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(format!("stream read error: {e}"));
                }
            }
        }
        stream
            .write_all(b"0\r\n\r\n")
            .await
            .map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())
    } else {
        let body_bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("upstream body read: {e}"))?;
        write_response_with_origin(stream, status, "application/json", &body_bytes, origin).await
    }
}

async fn handle_connection(mut stream: TcpStream, state: Arc<AppState>) {
    let result = serve_connection(&mut stream, &state).await;
    if let Err(e) = result {
        let _ = write_response(
            &mut stream,
            500,
            "text/plain",
            format!("gateway error: {e}").as_bytes(),
        )
        .await;
    }
    let _ = stream.shutdown().await;
}

/// Bind the gateway on `port` and serve until aborted. Returns a join handle
/// so the supervisor can stop it by aborting.
pub fn start_gateway(state: Arc<AppState>, port: u16) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = match TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[gateway] failed to bind {port}: {e}");
                return;
            }
        };
        eprintln!("[gateway] OpenAI-compatible gateway listening on 127.0.0.1:{port}");
        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let st = Arc::clone(&state);
                    tokio::spawn(handle_connection(stream, st));
                }
                Err(e) => {
                    eprintln!("[gateway] accept error: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                }
            }
        }
    })
}

/// Poll the persisted config once a second and keep the gateway listening on
/// the requested port & enable state, restarting it whenever those change.
pub fn spawn_supervisor(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut running: Option<(u16, tokio::task::JoinHandle<()>)> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let state: tauri::State<Arc<AppState>> = app.state();
            let cfg = state.config();
            let wanted = (cfg.advanced_settings.gateway_enabled
                && cfg
                    .advanced_settings
                    .api_key
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|key| !key.is_empty()))
            .then_some(cfg.advanced_settings.gateway_port);
            if let Some(port) = wanted {
                let stale = matches!(&running, Some((p, h)) if *p != port || h.is_finished());
                if stale {
                    if let Some((_, h)) = running.take() {
                        h.abort();
                    }
                }
                if running.is_none() {
                    running = Some((port, start_gateway(Arc::clone(&state), port)));
                }
            } else if let Some((_, h)) = running.take() {
                h.abort();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{LiveServer, ServerDef, ServerStatus, VecDequeLog};
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    fn live_server(
        id: &str,
        model_id: &str,
        served: Option<&str>,
        port: u16,
        task: &str,
        status: ServerStatus,
    ) -> LiveServer {
        LiveServer {
            def: ServerDef {
                backend: "vllm".into(),
                id: id.to_string(),
                name: id.to_string(),
                model_id: model_id.to_string(),
                task: task.to_string(),
                port,
                gpu_mem_util: 0.92,
                max_model_len: None,
                quant: "fp16".to_string(),
                served_model_name: served.map(|s| s.to_string()),
                kv_cache_dtype: None,
                llamacpp_channel: crate::state::LlamaCppChannel::Upstream,
                enforce_eager: true,
                params_b: None,
                swap_space_gb: None,
                cpu_offload_gb: None,
                was_running: false,
                model_path: None,
                mmproj_path: None,
                ctx_size: None,
                n_gpu_layers: Some(99),
                n_cpu_moe: None,
                fit: true,
                fit_target: None,
                device: None,
                api_key: None,
                log_verbosity: None,
                flash_attn: true,
                cache_type_k: "q8_0".into(),
                cache_type_v: "q8_0".into(),
                threads: None,
                batch_size: None,
                ubatch_size: None,
                parallel: 1,
                jinja: true,
                no_kv_offload: false,
                metrics: true,
                extra_args: Vec::new(),
                env: BTreeMap::new(),
            },
            status,
            error: None,
            wsl_child: None,
            native_child: None,
            wsl_pid: None,
            log_ring: Mutex::new(VecDequeLog::new()),
            last_metrics: None,
            stopping: false,
            crash_retry_count: 0,
        }
    }

    fn running_map() -> BTreeMap<String, LiveServer> {
        let mut servers = BTreeMap::new();
        servers.insert(
            "srv-a".to_string(),
            live_server(
                "srv-a",
                "Qwen/Qwen2.5-7B-Instruct",
                Some("qwen-7b"),
                8001,
                "instruct",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-b".to_string(),
            live_server(
                "srv-b",
                "meta-llama/Llama-3.1-8B-Instruct",
                None,
                8002,
                "instruct",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-embed".to_string(),
            live_server(
                "srv-embed",
                "BAAI/bge-small-en",
                None,
                8003,
                "embed",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-stopped".to_string(),
            live_server(
                "srv-stopped",
                "mistralai/Mistral-7B-Instruct-v0.3",
                None,
                8004,
                "instruct",
                ServerStatus::Stopped,
            ),
        );
        servers
    }

    #[test]
    fn test_route_exact_served_model_name() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "qwen-7b", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "Qwen-7B", "instruct"),
            Some(8001)
        );
    }

    #[test]
    fn test_route_exact_hf_id() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "Qwen/Qwen2.5-7B-Instruct", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "meta-llama/Llama-3.1-8B-Instruct", "instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_fuzzy_suffix_match() {
        let servers = running_map();
        // Client asks for just the tail of the HF id
        assert_eq!(
            find_server_port_for(&servers, "Qwen2.5-7B-Instruct", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "Llama-3.1-8B-Instruct", "instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_ignores_embed_and_stopped_servers() {
        let servers = running_map();
        // Embed server must not be routed for chat
        assert_eq!(
            find_server_port_for(&servers, "BAAI/bge-small-en", "instruct"),
            None
        );
        // Stopped instruct server must not be routed
        assert_eq!(
            find_server_port_for(&servers, "mistralai/Mistral-7B-Instruct-v0.3", "instruct"),
            None
        );
    }

    #[test]
    fn test_route_unknown_or_empty_model() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "nope/model", "instruct"),
            None
        );
        assert_eq!(find_server_port_for(&servers, "", "instruct"), None);
        assert_eq!(find_server_port_for(&servers, "   ", "instruct"), None);
    }

    #[test]
    fn test_route_embed_server_exact_and_fuzzy() {
        let servers = running_map();
        // BAAI/bge-small-en is task=embed, port 8003 in running_map()
        assert_eq!(
            find_server_port_for(&servers, "BAAI/bge-small-en", "embed"),
            Some(8003)
        );
        assert_eq!(
            find_server_port_for(&servers, "bge-small-en", "embed"),
            Some(8003)
        );
    }

    #[test]
    fn test_route_embed_ignores_instruct_servers() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "Qwen/Qwen2.5-7B-Instruct", "embed"),
            None
        );
        assert_eq!(find_server_port_for(&servers, "", "embed"), None);
    }

    #[test]
    fn test_not_found_payload_includes_running_hint() {
        let payload = not_found_payload_with_hint(
            "nope/model",
            "instruction",
            &["qwen-7b".to_string(), "llama-8b".to_string()],
        );
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nope/model"));
        assert_eq!(v["running"][0], "qwen-7b");
    }

    #[test]
    fn test_model_rows_only_lists_running_instruct_servers() {
        let servers = running_map();
        let rows = model_rows(&servers);
        // 2 running instruct servers, prefer the served_model_name for display
        let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
        assert_eq!(ids, vec!["qwen-7b", "meta-llama/Llama-3.1-8B-Instruct"]);
        assert!(rows
            .iter()
            .all(|r| r["object"] == serde_json::json!("model")));
    }

    #[test]
    fn test_model_rows_include_effective_context_metadata() {
        let mut servers = running_map();
        servers.get_mut("srv-a").unwrap().def.max_model_len = Some(32_768);
        let rows = model_rows(&servers);
        let row = rows.iter().find(|row| row["id"] == "qwen-7b").unwrap();
        for key in [
            "max_model_len",
            "context_length",
            "max_context_length",
            "context_window",
            "n_ctx",
        ] {
            assert_eq!(row[key], serde_json::json!(32_768));
        }
        assert_eq!(row["context_source"], "configured");
        assert_eq!(row["backend"], "vllm");
        assert_eq!(row["capabilities"]["chat_completions"], true);
    }

    #[test]
    fn test_llama_model_rows_use_configured_context_size() {
        let mut servers = running_map();
        let def = &mut servers.get_mut("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.max_model_len = None;
        def.ctx_size = Some(262_144);
        let rows = model_rows(&servers);
        let row = rows.iter().find(|row| row["id"] == "qwen-7b").unwrap();
        assert_eq!(row["max_model_len"], serde_json::json!(262_144));
        assert_eq!(row["context_length"], serde_json::json!(262_144));
        assert_eq!(row["context_source"], "configured");
    }

    #[test]
    fn test_runtime_context_parser_reads_llama_props() {
        let value = serde_json::json!({
            "default_generation_settings": { "n_ctx": "262144" }
        });
        assert_eq!(extract_runtime_context(&value), Some(262_144));
    }

    #[test]
    fn test_model_path_decodes_url_encoded_ids() {
        assert_eq!(
            model_id_from_path("/v1/models/Qwen%2FQwen2.5-7B-Instruct"),
            Some("Qwen/Qwen2.5-7B-Instruct".to_string())
        );
    }

    #[test]
    fn test_model_extracted_from_body() {
        let body = br#"
            {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}
        "#;
        assert_eq!(model_from_body(body).as_deref(), Some("qwen-7b"));
        assert_eq!(model_from_body(br#"{"msg":"no model"}"#), None);
        assert_eq!(model_from_body(b"not json at all"), None);
    }

    #[test]
    fn test_model_from_body_preserves_fim_suffix() {
        let body = br#"{"model":"qwen-7b","prompt":"def f(","suffix":"):\n pass","stream":false}"#;
        assert_eq!(model_from_body(body).as_deref(), Some("qwen-7b"));
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(v["suffix"], "):\n pass");
    }

    #[test]
    fn test_parse_request_head_extracts_content_length() {
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 1024\r\n\r\n";
        let parsed = parse_head(head).expect("valid head parses");
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.target, "/v1/chat/completions");
        assert_eq!(parsed.content_length, 1024);
        assert_eq!(
            parsed.headers.get("content-type").map(|s| s.as_str()),
            Some("application/json")
        );
    }

    #[test]
    fn test_parse_request_head_garbage() {
        assert!(parse_head(b"").is_none());
        assert!(parse_head(b"\r\n\r\n").is_none());
        assert!(parse_head(b"noSpacesAtAll\r\n\r\n").is_none());
    }

    #[test]
    fn test_head_terminator_found() {
        let buf = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\nbody-here";
        assert_eq!(is_head_terminated(buf), Some(32));
        assert_eq!(is_head_terminated(b"GET /v1/models HTTP/1.1\r\n"), None);
    }

    #[test]
    fn test_cors_headers_only_allow_trusted_local_origins() {
        let trusted = cors_headers(Some("http://127.0.0.1:1420"));
        assert!(trusted.contains("Access-Control-Allow-Origin: http://127.0.0.1:1420"));
        assert!(trusted.contains("Vary: Origin"));
        assert!(trusted.contains("Access-Control-Allow-Methods"));
        assert!(trusted.contains("Authorization"));
        assert_eq!(cors_headers(Some("https://evil.example")), "");
        assert_eq!(cors_headers(None), "");
    }

    #[test]
    fn test_check_gateway_auth_no_key_configured() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer anything".to_string());
        let err = check_gateway_auth(&headers, None).unwrap_err();
        assert_eq!(err.0, 503);
        assert!(err.1.contains("not configured"));
        assert_eq!(check_gateway_auth(&headers, Some("")).unwrap_err().0, 503);
    }

    #[test]
    fn test_check_gateway_auth_valid_key() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer secret123".to_string());
        assert!(check_gateway_auth(&headers, Some("secret123")).is_ok());
    }

    #[test]
    fn test_check_gateway_auth_invalid_key() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer wrong".to_string());
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Invalid API key"));
    }

    #[test]
    fn test_check_gateway_auth_missing_header() {
        let headers = HashMap::new();
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Missing Authorization header"));
    }

    #[test]
    fn test_check_gateway_auth_wrong_scheme() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Basic secret123".to_string());
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Bearer scheme"));
    }

    #[test]
    fn test_check_gateway_auth_case_insensitive_header_name() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer secret123".to_string());
        assert!(check_gateway_auth(&headers, Some("secret123")).is_ok());
    }
}
