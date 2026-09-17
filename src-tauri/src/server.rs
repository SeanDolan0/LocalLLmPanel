//! Multi-instance vLLM server lifecycle: launch, monitor, stop, metrics, chat.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::net::TcpListener;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::estimate;
use crate::state::{AppState, LiveServer, ServerDef, ServerStatus, VecDequeLog};
pub use crate::state::MetricsSnapshot;
use crate::wsl;
use tauri::Emitter;

const DEFAULT_PORT_START: u16 = 8000;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const HEALTH_POLL: Duration = Duration::from_secs(2);
const METRICS_POLL: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Port allocation
// ---------------------------------------------------------------------------

/// Allocate a free localhost port, starting at 8000 (or after `min`),
/// skipping any ports already bound by existing server definitions.
pub fn alloc_port(existing: &[u16]) -> Result<u16> {
    let mut port = DEFAULT_PORT_START;
    loop {
        if existing.contains(&port) {
            port += 1;
            continue;
        }
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
        port = port.wrapping_add(1);
        if port == DEFAULT_PORT_START {
            bail!("no free port found");
        }
    }
}

// ---------------------------------------------------------------------------
// Launch script
// ---------------------------------------------------------------------------

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Build the `bash -lc` launcher. The script activates the venv, records the
/// process PID (bash exec → vLLM keeps the same PID), then `exec`s vLLM so it
/// runs in the foreground of the wsl.exe console (logs stream to the panel).
fn launch_script(venv_dir: &str, def: &ServerDef, hf_token: &str) -> String {
    let mut parts: Vec<String> = vec![
        format!("cd {}/..", venv_dir),
        format!(". {}/bin/activate", venv_dir),
        format!("echo $$ > {}/../run/{}.pid", venv_dir, def.id),
    ];
    let mut args: Vec<String> = Vec::new();
    args.push("--model".into());
    args.push(shell_quote(&def.model_id));
    args.push("--host".into());
    args.push("127.0.0.1".into());
    args.push("--port".into());
    args.push(def.port.to_string());
    args.push("--gpu-memory-utilization".into());
    args.push(format!("{:.2}", def.gpu_mem_util));
    if def.task == "embed" {
        args.push("--runner".into());
        args.push("pooling".into());
    }
    match def.quant.to_ascii_lowercase().as_str() {
        "fp8" => {
            args.push("--quantization".into());
            args.push("fp8".into());
        }
        "awq" => {
            args.push("--quantization".into());
            args.push("awq".into());
        }
        "gptq" => {
            args.push("--quantization".into());
            args.push("gptq".into());
        }
        _ => {}
    }
    let max_len = def.max_model_len.unwrap_or(4096);
    args.push("--max-model-len".into());
    args.push(max_len.to_string());
    if def.enforce_eager {
        args.push("--enforce-eager".into());
    }
    if let Some(served) = &def.served_model_name {
        args.push("--served-model-name".into());
        args.push(shell_quote(served));
    }
    if let Some(swap) = def.swap_space_gb {
        if swap > 0 {
            args.push("--swap-space".into());
            args.push(swap.to_string());
        }
    }
    if let Some(offload) = def.cpu_offload_gb {
        if offload > 0 {
            args.push("--cpu-offload-gb".into());
            args.push(offload.to_string());
        }
    }

    // vLLM disables pinned-memory/UVA on WSL2 by default (see
    // vllm/platforms/cuda.py), which crashes the V1 engine with
    // "UVA is not available". Kernels >= 4.19.121 support it once enabled.
    let preamble = "export VLLM_WSL2_ENABLE_PIN_MEMORY=1";
    parts.insert(0, preamble.into());
    let token_ok = !hf_token.is_empty()
        && hf_token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || (c.is_ascii_punctuation() && c != '\''));
    if token_ok {
        parts.insert(0, format!("export HF_TOKEN='{}'", hf_token));
    }
    parts.push(format!("exec python -m vllm.entrypoints.openai.api_server {}", args.join(" ")));
    parts.join(" && ")
}

pub fn build_start_command(def: &ServerDef, hf_token: &str) -> String {
    launch_script("~/llm-lp/.venv", def, hf_token)
}


// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub struct ServerStatusEvent {
    pub id: String,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Clone, Serialize)]
struct ServerLogEvent {
    id: String,
    line: String,
}

fn emit_status(app: Option<&tauri::AppHandle>, id: &str, status: ServerStatus, error: Option<String>) {
    let payload = ServerStatusEvent {
        id: id.to_string(),
        status: status.label().to_string(),
        error,
    };
    if let Some(app) = app {
        let _ = app.emit("server-status", payload);
    }
}

// ---------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------

/// Start one server. Health is polled asynchronously; a monitor task follows
/// lifecycle and emits events. `state` is an Arc so the monitor task can hold
/// it beyond the command call.
pub fn start_server(
    state: &Arc<AppState>,
    app: Option<&tauri::AppHandle>,
    id: &str,
) -> Result<()> {
    let cfg = state.config();
    let def = cfg
        .find_server(id)
        .cloned()
        .ok_or_else(|| anyhow!("no server with id {id}"))?;
    {
        let servers = state.servers.lock().unwrap();
        if let Some(ls) = servers.get(id) {
            if ls.status == ServerStatus::Running || ls.status == ServerStatus::Starting {
                bail!("server {} already {}", def.name, ls.status.label());
            }
        }
    }

    let script = launch_script(&cfg.venv_dir, &def, &cfg.hf_token);
    let id_log = id.to_string();
    let state_log = Arc::clone(state);
    let app_ev = app.map(|a| (*a).clone());
    let log_cb = move |line: String| {
        if let Some(ls) = state_log.servers.lock().unwrap().get_mut(&id_log) {
            ls.log_ring.lock().unwrap().push(line.clone());
        }
        if let Some(app) = &app_ev {
            let _ = app.emit("server-log", ServerLogEvent { id: id_log.clone(), line });
        }
    };
    let child = wsl::WslChild::spawn(&cfg.distro, &script, log_cb)
        .map_err(|e| anyhow!("failed to launch wsl: {e}"))?;
    let wsl_pid = child.pid();

    {
        let mut servers = state.servers.lock().unwrap();
        servers.insert(
            id.to_string(),
            LiveServer {
                def: def.clone(),
                status: ServerStatus::Starting,
                error: None,
                wsl_child: Some(child),
                wsl_pid: Some(wsl_pid),
                log_ring: std::sync::Mutex::new(VecDequeLog::new()),
                last_metrics: None,
                stopping: false,
            },
        );
    }
    emit_status(app, id, ServerStatus::Starting, None);

    // Async monitor: health → running, then metrics + liveness loop.
    let app = app.map(|a| (*a).clone());
    let http = state.http.clone();
    let state_task = Arc::clone(state);
    let id_task = id.to_string();
    let model_for_metrics = def.model_id.clone();
    tauri::async_runtime::spawn(async move {
        let url = format!("http://127.0.0.1:{}/health", def.port);
        let mut ok = false;
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            if let Ok(resp) = http.get(&url).send().await {
                if resp.status().is_success() {
                    ok = true;
                    break;
                }
            }
            tokio::time::sleep(HEALTH_POLL).await;
        }
        if !ok {
            emit_status(app.as_ref(), &id_task, ServerStatus::Error, Some("health check timed out".into()));
            update_status(&state_task, &id_task, ServerStatus::Error);
            return;
        }
        emit_status(app.as_ref(), &id_task, ServerStatus::Running, None);
        update_status(&state_task, &id_task, ServerStatus::Running);

        // Metrics + liveness loop
        let metrics_url = format!("http://127.0.0.1:{}/metrics", def.port);
        let mut last_gen = 0u64;
        let mut last_prompt = 0u64;
        let mut last_measured_at = now_ms();
        let mut first_sample = true;
        loop {
            // Liveness: did the wsl process die on its own (and we're not stopping)?
            let (exited, stopping) = {
                let mut servers = state_task.servers.lock().unwrap();
                let mut ls = servers.get_mut(&id_task);
                let exit = ls
                    .as_mut()
                    .and_then(|ls| ls.wsl_child.as_mut())
                    .and_then(|c| c.try_wait().ok().flatten());
                let stopping = ls.map(|ls| ls.stopping).unwrap_or(false);
                (exit, stopping)
            };
            if let Some(exit) = exited {
                if !stopping {
                    emit_status(
                        app.as_ref(),
                        &id_task,
                        ServerStatus::Error,
                        Some(format!("vLLM process exited ({exit})")),
                    );
                    update_status(&state_task, &id_task, ServerStatus::Error);
                } else {
                    update_status(&state_task, &id_task, ServerStatus::Stopped);
                }
                return;
            }

            if let Ok(resp) = http.get(&metrics_url).send().await {
                if let Ok(text) = resp.text().await {
                    if let Some(m) = parse_metrics(&text) {
                        let measured = if first_sample {
                            None
                        } else {
                            let dt = (now_ms() - last_measured_at).max(1000) as f64 / 1000.0;
                            Some(crate::state::MeasuredStats {
                                tokens_per_sec: Some(((m.total_generation_tokens - last_gen) as f64) / dt),
                                prompt_tokens_per_sec: Some(((m.total_prompt_tokens - last_prompt) as f64) / dt),
                                total_prompt_tokens: m.total_prompt_tokens,
                                total_generation_tokens: m.total_generation_tokens,
                                requests: m.requests,
                                measured_at_ms: Some(now_ms()),
                            })
                        };
                        last_gen = m.total_generation_tokens;
                        last_prompt = m.total_prompt_tokens;
                        last_measured_at = now_ms();
                        first_sample = false;
                        let snapshot = MetricsSnapshot {
                            running: m.running,
                            waiting: m.waiting,
                            total_prompt_tokens: m.total_prompt_tokens,
                            total_generation_tokens: m.total_generation_tokens,
                            requests: m.requests,
                            measured: measured.clone(),
                        };
                        {
                            let mut servers = state_task.servers.lock().unwrap();
                            if let Some(ls) = servers.get_mut(&id_task) {
                                ls.last_metrics = Some(snapshot.clone());
                            }
                            if let Some(app) = &app {
                                let _ = app.emit("server-metrics", snapshot);
                            }
                        }
                        if let Some(ms) = measured {
                            let mut cfg = state_task.config.lock().unwrap();
                            cfg.measured.insert(model_for_metrics.clone(), ms);
                            if let Err(e) = cfg.save() {
                                eprintln!("[server] save measured: {e}");
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(METRICS_POLL).await;
        }
    });

    Ok(())
}

fn update_status(state: &Arc<AppState>, id: &str, status: ServerStatus) {
    let mut servers = state.servers.lock().unwrap();
    if let Some(ls) = servers.get_mut(id) {
        ls.status = status;
    }
}

/// Stop a server: SIGTERM to the WSL-side PID via pidfile, then wait; on
/// timeout kill the wsl.exe process tree. Idempotent.
pub fn stop_server(state: &Arc<AppState>, app: Option<&tauri::AppHandle>, id: &str) -> Result<()> {
    let cfg = state.config();
    let mut child = {
        let mut servers = state.servers.lock().unwrap();
        match servers.get_mut(id) {
            Some(ls) if ls.status != ServerStatus::Stopped => {
                ls.stopping = true;
                ls.wsl_child.take()
            }
            // Already stopped — idempotent no-op.
            _ => None,
        }
    };

    // 1) SIGTERM on the WSL side (graceful: vLLM drains in-flight requests).
    let pid_from_file = wsl::run_script(
        &cfg.distro,
        &format!("cat {}/../run/{id}.pid 2>/dev/null || true", cfg.venv_dir),
    );
    let mut term_ok = false;
    if let Some(pid) = pid_from_file.stdout.trim().parse::<u32>().ok() {
        let kill = wsl::run_script(&cfg.distro, &format!("kill -TERM {pid} 2>/dev/null && echo killed || echo nograb"));
        term_ok = kill.stdout.contains("killed");
    }

    // 2) Wait up to 30s for the child to exit on its own.
    if let Some(c) = child.as_mut() {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = c.try_wait() {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        let still_alive = c.try_wait().map(|r| r.is_none()).unwrap_or(false);
        if still_alive {
            // 3) Fallback: kill the wsl.exe tree from Windows.
            let mut cmd = std::process::Command::new("taskkill");
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x08000000);
            }
            let _ = cmd
                .args(["/T", "/F", "/PID", &c.pid().to_string()])
                .status();
        }
        if let Some(c) = child.take() {
            let mut c = c;
            c.join();
        }
    }

    // Mark stopped.
    {
        let mut servers = state.servers.lock().unwrap();
        if let Some(ls) = servers.get_mut(id) {
            ls.status = ServerStatus::Stopped;
            ls.wsl_child = None;
            ls.wsl_pid = None;
            ls.error = None;
            ls.last_metrics = None;
        }
    }
    let _ = term_ok;
    emit_status(app, id, ServerStatus::Stopped, None);
    Ok(())
}

/// Restart = tolerant stop then start.
pub fn restart_server(state: &Arc<AppState>, app: Option<&tauri::AppHandle>, id: &str) -> Result<()> {
    let _ = stop_server(state, app, id);
    start_server(state, app, id)
}

// ---------------------------------------------------------------------------
// Metrics parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub running: u64,
    pub waiting: u64,
    pub total_prompt_tokens: u64,
    pub total_generation_tokens: u64,
    pub requests: u64,
}

/// Parse the Prometheus text from vLLM `/metrics`.
pub fn parse_metrics(text: &str) -> Option<Metrics> {
    let mut m = Metrics::default();
    let mut saw = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once(' ') else {
            continue;
        };
        let Ok(val) = value.trim().parse::<f64>() else {
            continue;
        };
        match name {
            "vllm:num_requests_running" | "vllm:num_requests_running_gauge" => {
                m.running = val as u64;
                saw = true;
            }
            "vllm:num_requests_waiting" | "vllm:num_requests_waiting_gauge" => {
                m.waiting = val as u64;
                saw = true;
            }
            "vllm:prompt_tokens_total" | "vllm:prompt_tokens_succeeded_total" => {
                m.total_prompt_tokens = val as u64;
                saw = true;
            }
            "vllm:generation_tokens_total" | "vllm:generation_tokens_succeeded_total" => {
                m.total_generation_tokens = val as u64;
                saw = true;
            }
            "vllm:generation_tokens_failed_total" => {
                m.total_generation_tokens += val as u64;
                saw = true;
            }
            "vllm:request_success_total" | "vllm:requests_succeeded_total" => {
                m.requests = val as u64;
                saw = true;
            }
            _ => {}
        }
    }
    if saw {
        Some(m)
    } else {
        None
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

pub fn list_servers(
    state: &Arc<AppState>,
) -> Vec<(ServerDef, ServerStatus, Option<String>, Option<MetricsSnapshot>)> {
    let defs = state.config().servers.clone();
    let servers = state.servers.lock().unwrap();
    defs.into_iter()
        .map(|def| {
            if let Some(ls) = servers.get(&def.id) {
                (def, ls.status, ls.error.clone(), ls.last_metrics.clone())
            } else {
                (def, ServerStatus::Stopped, None, None)
            }
        })
        .collect()
}

/// Estimated weight footprint (GB) of all RUNNING servers — for the VRAM
/// guidance bar vs. current free VRAM.
pub fn running_weight_gb(state: &Arc<AppState>) -> f64 {
    let servers = state.servers.lock().unwrap();
    servers
        .values()
        .filter(|ls| ls.status == ServerStatus::Running && ls.def.params_b.is_some())
        .map(|ls| estimate::weight_gb(ls.def.params_b.unwrap_or(0.0), &ls.def.quant))
        .sum()
}

/// Issue a chat completion against an instruct server (server-side, no CORS).
pub async fn chat(
    state: &Arc<AppState>,
    server_id: &str,
    messages: Vec<serde_json::Value>,
) -> Result<serde_json::Value> {
    let def = state
        .config()
        .find_server(server_id)
        .cloned()
        .ok_or_else(|| anyhow!("no server {server_id}"))?;
    if def.task != "instruct" {
        bail!("server {server_id} is not an instruct server");
    }
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", def.port);
    let body = serde_json::json!({
        "model": def.effective_model_name(),
        "messages": messages,
        "stream": false,
    });
    let resp = state
        .http
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("chat failed ({}): {}", status, text);
    }
    Ok(resp.json().await.context("chat response JSON")?)
}

pub fn server_logs(state: &Arc<AppState>, server_id: &str, since: usize) -> String {
    let servers = state.servers.lock().unwrap();
    match servers.get(server_id) {
        Some(ls) => ls.log_ring.lock().unwrap().since_line(since),
        None => String::new(),
    }
}

pub fn server_metrics(state: &Arc<AppState>, server_id: &str) -> Option<MetricsSnapshot> {
    let servers = state.servers.lock().unwrap();
    servers.get(server_id).and_then(|ls| ls.last_metrics.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_allocator_skips_bound_ports() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let taken = listener.local_addr().unwrap().port();
        let p = alloc_port(&[taken]).unwrap();
        assert_ne!(p, taken);
        assert!(p >= DEFAULT_PORT_START);
    }

    #[test]
    fn port_allocator_respects_existing_defs() {
        let p = alloc_port(&[8000, 8001, 8002]).unwrap();
        assert!(p >= 8003);
    }

    fn def(model: &str, task: &str, port: u16, quant: &str, served: Option<&str>) -> ServerDef {
        ServerDef {
            id: "s1".into(),
            name: "test".into(),
            model_id: model.into(),
            task: task.into(),
            port,
            gpu_mem_util: 0.92,
            max_model_len: Some(2048),
            quant: quant.into(),
            served_model_name: served.map(|s| s.into()),
            enforce_eager: true,
            params_b: None,
            swap_space_gb: None,
            cpu_offload_gb: None,
        }
    }

    #[test]
    fn launch_script_instruct() {
        let script = launch_script("~/llm-lp/.venv", &def("Qwen/Qwen2.5-0.5B-Instruct", "instruct", 8010, "fp16", None), "");
        assert!(script.contains("exec python -m vllm.entrypoints.openai.api_server"));
        assert!(script.contains("--model 'Qwen/Qwen2.5-0.5B-Instruct'"));
        assert!(script.contains("--port 8010"));
        assert!(script.contains("--gpu-memory-utilization 0.92"));
        assert!(script.contains("--max-model-len 2048"));
        assert!(script.contains("$$ > ~/llm-lp/.venv/../run/s1.pid"));
        assert!(!script.contains("--runner pooling"));
        assert!(!script.contains("--quantization"));
        assert!(!script.contains("export HF_TOKEN="));
    }

    #[test]
    fn launch_script_embed_with_quant_and_served() {
        let script = launch_script(
            "~/llm-lp/.venv",
            &def("BAAI/bge-small-en-v1.5", "embed", 8020, "fp8", Some("embedder")),
            "hf_secret_123",
        );
        assert!(script.contains("--runner pooling"));
        assert!(script.contains("--quantization fp8"));
        assert!(script.contains("--served-model-name 'embedder'"));
        assert!(script.contains("--max-model-len 2048"));
        assert!(script.contains("export HF_TOKEN='hf_secret_123'"));
    }

    #[test]
    fn test_build_start_command_swap_and_cpu_offload() {
        let mut d = def("Qwen/Qwen2.5-7B-Instruct", "instruct", 8010, "fp16", None);
        d.swap_space_gb = Some(8);
        d.cpu_offload_gb = Some(4);
        let cmd = build_start_command(&d, "");
        assert!(cmd.contains("--swap-space 8"), "command must include --swap-space 8: {cmd}");
        assert!(cmd.contains("--cpu-offload-gb 4"), "command must include --cpu-offload-gb 4: {cmd}");
        assert!(cmd.contains("export VLLM_WSL2_ENABLE_PIN_MEMORY=1"));
    }

    #[test]
    fn test_build_start_command_swap_and_cpu_offload_omitted_when_zero_or_none() {
        let mut d = def("Qwen/Qwen2.5-7B-Instruct", "instruct", 8010, "fp16", None);
        d.swap_space_gb = Some(0);
        d.cpu_offload_gb = Some(0);
        let cmd_zero = build_start_command(&d, "");
        assert!(!cmd_zero.contains("--swap-space"), "command must not include --swap-space: {cmd_zero}");
        assert!(!cmd_zero.contains("--cpu-offload-gb"), "command must not include --cpu-offload-gb: {cmd_zero}");

        d.swap_space_gb = None;
        d.cpu_offload_gb = None;
        let cmd_none = build_start_command(&d, "");
        assert!(!cmd_none.contains("--swap-space"), "command must not include --swap-space: {cmd_none}");
        assert!(!cmd_none.contains("--cpu-offload-gb"), "command must not include --cpu-offload-gb: {cmd_none}");
    }


    #[test]
    fn metrics_parse_both_generations() {
        let text = "# HELP vllm:generation_tokens_total Total generation tokens\nvllm:generation_tokens_total 1234\nvllm:prompt_tokens_total 567\nvllm:num_requests_running 2\nvllm:num_requests_waiting 3\n";
        let m = parse_metrics(text).unwrap();
        assert_eq!(m.total_generation_tokens, 1234);
        assert_eq!(m.total_prompt_tokens, 567);
        assert_eq!(m.running, 2);
        assert_eq!(m.waiting, 3);

        let v1 = "vllm:generation_tokens_succeeded_total 10\nvllm:generation_tokens_failed_total 4\nvllm:num_requests_running 1\n";
        let m1 = parse_metrics(v1).unwrap();
        assert_eq!(m1.total_generation_tokens, 14);
    }

    #[test]
    fn metrics_parse_ignores_garbage() {
        assert!(parse_metrics("hello world\nnot a metric").is_none());
    }
}