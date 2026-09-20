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
use crate::state::{AppState, ServerStatus};
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

/// All live instruct servers that are currently `Running`.
fn running_instruct_servers(
    servers: &BTreeMap<String, crate::state::LiveServer>,
) -> Vec<&crate::state::LiveServer> {
    servers
        .values()
        .filter(|ls| ls.status == ServerStatus::Running && ls.def.task == "instruct")
        .collect()
}

/// Route a requested model name to the port of the running server that
/// serves it. Exact match on the served model name or HF id wins first,
/// then a fuzzy suffix match against both (so `Qwen2.5-7B-Instruct` finds
/// `Qwen/Qwen2.5-7B-Instruct`).
pub fn find_server_port_for_model(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
) -> Option<u16> {
    let m = model.trim();
    if m.is_empty() {
        return None;
    }
    let running = running_instruct_servers(servers);
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

/// Route a requested embedding model name to the port of the running
/// server that serves it. Mirrors find_server_port_for_model but for
/// task == "embed".
pub fn find_server_port_for_embed(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
) -> Option<u16> {
    let m = model.trim();
    if m.is_empty() {
        return None;
    }
    let running: Vec<&crate::state::LiveServer> = servers
        .values()
        .filter(|ls| ls.status == crate::state::ServerStatus::Running && ls.def.task == "embed")
        .collect();
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

/// OpenAI-style `data` rows for `GET /v1/models`. Prefer the served model
/// name when present (that is the name a client would pass in `"model"`).
pub fn model_rows(servers: &BTreeMap<String, crate::state::LiveServer>) -> Vec<serde_json::Value> {
    running_instruct_servers(servers)
        .into_iter()
        .map(|ls| {
            let id = if ls.def.served_model_name.is_some() {
                ls.def.effective_model_name()
            } else {
                ls.def.model_id.clone()
            };
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "local-llm-panel",
                "port": ls.def.port,
            })
        })
        .collect()
}

/// Extract the requested model id from a chat/completions JSON body.
pub fn model_from_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::to_string)
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

pub fn is_head_terminated(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn not_found_payload(model: &str) -> String {
    serde_json::json!({
        "error": {
            "message": format!("no running instruction server for model '{model}'"),
            "type": "invalid_request_error",
        }
    })
    .to_string()
}

fn running_model_names(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<String> {
    servers
        .values()
        .filter(|ls| {
            ls.status == crate::state::ServerStatus::Running && ls.def.task == task
        })
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
    match (head.method.as_str(), head.target.as_str()) {
        ("GET", "/v1/models") => {
            let text = {
                let servers = state.servers.lock().unwrap();
                let payload = serde_json::json!({ "object": "list", "data": model_rows(&servers) });
                payload.to_string()
            };
            write_response(stream, 200, "application/json", text.as_bytes()).await
        }
        ("POST", "/v1/chat/completions") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for_model(&servers, &model)
            };
            match port {
                Some(port) => proxy_chat(stream, state, port, body).await,
                None => {
                    write_response(
                        stream,
                        404,
                        "application/json",
                        not_found_payload(&model).as_bytes(),
                    )
                    .await
                }
            }
        }
        (method, "/v1/models") if method != "GET" => {
            write_response(stream, 405, "text/plain", b"method not allowed").await
        }
        _ => write_response(stream, 404, "text/plain", b"not found").await,
    }
}

async fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(body).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())
}

/// Forward a chat/completions request to the target vLLM port, preserving
/// streaming when the caller asked for `"stream": true`.
async fn proxy_chat(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    body: &[u8],
) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let streaming = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    let api_key = state.config().advanced_settings.api_key.clone();
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
            "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
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
        write_response(stream, status, "application/json", &body_bytes).await
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
            let wanted = cfg
                .advanced_settings
                .gateway_enabled
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
                id: id.to_string(),
                name: id.to_string(),
                model_id: model_id.to_string(),
                task: task.to_string(),
                port,
                gpu_mem_util: 0.92,
                max_model_len: None,
                quant: "fp16".to_string(),
                served_model_name: served.map(|s| s.to_string()),
                enforce_eager: true,
                params_b: None,
                swap_space_gb: None,
                cpu_offload_gb: None,
                was_running: false,
            },
            status,
            error: None,
            wsl_child: None,
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
        assert_eq!(find_server_port_for_model(&servers, "qwen-7b"), Some(8001));
        assert_eq!(find_server_port_for_model(&servers, "Qwen-7B"), Some(8001));
    }

    #[test]
    fn test_route_exact_hf_id() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for_model(&servers, "Qwen/Qwen2.5-7B-Instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for_model(&servers, "meta-llama/Llama-3.1-8B-Instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_fuzzy_suffix_match() {
        let servers = running_map();
        // Client asks for just the tail of the HF id
        assert_eq!(
            find_server_port_for_model(&servers, "Qwen2.5-7B-Instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for_model(&servers, "Llama-3.1-8B-Instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_ignores_embed_and_stopped_servers() {
        let servers = running_map();
        // Embed server must not be routed for chat
        assert_eq!(
            find_server_port_for_model(&servers, "BAAI/bge-small-en"),
            None
        );
        // Stopped instruct server must not be routed
        assert_eq!(
            find_server_port_for_model(&servers, "mistralai/Mistral-7B-Instruct-v0.3"),
            None
        );
    }

    #[test]
    fn test_route_unknown_or_empty_model() {
        let servers = running_map();
        assert_eq!(find_server_port_for_model(&servers, "nope/model"), None);
        assert_eq!(find_server_port_for_model(&servers, ""), None);
        assert_eq!(find_server_port_for_model(&servers, "   "), None);
    }

    #[test]
    fn test_route_embed_server_exact_and_fuzzy() {
        let servers = running_map();
        // BAAI/bge-small-en is task=embed, port 8003 in running_map()
        assert_eq!(
            find_server_port_for_embed(&servers, "BAAI/bge-small-en"),
            Some(8003)
        );
        assert_eq!(
            find_server_port_for_embed(&servers, "bge-small-en"),
            Some(8003)
        );
    }

    #[test]
    fn test_route_embed_ignores_instruct_servers() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for_embed(&servers, "Qwen/Qwen2.5-7B-Instruct"),
            None
        );
        assert_eq!(find_server_port_for_embed(&servers, ""), None);
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
        assert!(v["error"]["message"].as_str().unwrap().contains("nope/model"));
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
    fn test_model_extracted_from_body() {
        let body = br#"
            {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}
        "#;
        assert_eq!(model_from_body(body).as_deref(), Some("qwen-7b"));
        assert_eq!(model_from_body(br#"{"msg":"no model"}"#), None);
        assert_eq!(model_from_body(b"not json at all"), None);
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
}
