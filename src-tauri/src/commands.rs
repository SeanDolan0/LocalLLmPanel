//! Tauri command layer: the app's public surface.

use anyhow::Result;
use serde::Serialize;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};

use crate::estimate;
use crate::hf;
use crate::provision::{self, ProvisionReport};
use crate::server;
use crate::state::{AppState, MeasuredStats, PersistedConfig, ServerDef, GpuSnapshot};

// ---------------------------------------------------------------------------
// Env
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct EnvStatus {
    pub wsl_ok: bool,
    pub distro: String,
    pub apt_based: bool,
    pub provisioned: bool,
    pub report: Option<ProvisionReport>,
    pub gpu: Option<GpuSnapshot>,
    pub servers_running: usize,
    pub running_weight_gb: f64,
    pub gpu_bandwidth_gbs: f64,
    pub gpu_bw_known: bool,
}

fn gpu_snapshot(distro: &str) -> Option<GpuSnapshot> {
    let out = crate::wsl::run_script(
        distro,
        "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1",
    );
    let line = out.stdout.trim();
    if line.is_empty() {
        return None;
    }
    // "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227, 8456, 12"
    let mut it = line.split(',');
    let name = it.next()?.trim().to_string();
    let total = it.next()?.trim().parse::<u64>().ok()?;
    let free = it.next()?.trim().parse::<u64>().ok()?;
    let util = it.next()?.trim().parse::<u32>().ok().unwrap_or(0);
    Some(GpuSnapshot { name, vram_total_mb: total, vram_free_mb: free, util_percent: util })
}

#[tauri::command]
pub fn env_status(state: State<'_, Arc<AppState>>) -> EnvStatus {
    let st = (*state).clone();
    let cfg = st.config();
    let distro_detected = crate::wsl::detect_default_distro().unwrap_or_else(|| cfg.distro.clone());
    let wsl_ok = crate::wsl::run_script(&distro_detected, "echo ok").ok;
    let report = crate::wsl::run_script(&distro_detected, "cat ~/llm-lp/.provisioned 2>/dev/null || true").stdout.contains("\"provisioned\": true");
    let gpu = gpu_snapshot(&distro_detected);
    let (bandwidth, known) = gpu
        .as_ref()
        .map(|g| estimate::gpu_bandwidth(&g.name))
        .unwrap_or((700.0, false));
    let running = {
        let servers = st.servers.lock().unwrap();
        servers.values().filter(|ls| ls.status == crate::state::ServerStatus::Running).count()
    };
    let running_weight_gb = server::running_weight_gb(&st);
    let env_report = provision::env_probe(&distro_detected, &cfg.venv_dir);
    let apt_based = crate::wsl::is_apt_distro(&distro_detected);
    EnvStatus {
        wsl_ok,
        distro: distro_detected,
        apt_based,
        provisioned: report,
        report: Some(env_report),
        gpu,
        servers_running: running,
        running_weight_gb,
        gpu_bandwidth_gbs: bandwidth,
        gpu_bw_known: known,
    }
}

#[tauri::command]
pub async fn provision(app: AppHandle, state: State<'_, Arc<AppState>>, target: Option<String>) -> Result<ProvisionReport, String> {
    let st = (*state).clone();
    let app = app.clone();
    let cfg = st.config();
    let distro = cfg.distro.clone();
    let venv = cfg.venv_dir.clone();
    let _ = target;
    tauri::async_runtime::spawn_blocking(move || {
        let on_log = |phase: &str, line: &str| {
            let _ = app.emit("wsl-log", serde_json::json!({ "phase": phase, "line": line }));
        };
        provision::provision_all(&distro, &venv, on_log)
    })
    .await
    .map_err(|e| format!("provision task error: {e}"))?
    .map_err(|e| format!("provision failed: {e}"))
}

// ---------------------------------------------------------------------------
// HF search & stats
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ModelWithStats {
    pub id: String,
    pub downloads: i64,
    pub likes: i64,
    pub trending_score: f64,
    pub pipeline_tag: Option<String>,
    pub params_b: Option<f64>,
    pub context: Option<usize>,
    pub context_source: Option<&'static str>,
    pub context_estimated: bool,
    pub head_dim: Option<usize>,
    pub max_tok_s: Option<f64>,
    pub quant_assumed: String,
}

#[tauri::command]
pub async fn search_models(
    state: State<'_, Arc<AppState>>,
    query: String,
    quant: Option<String>,
) -> Result<Vec<ModelWithStats>, String> {
    let st = (*state).clone();
    let quant = quant.unwrap_or_else(|| st.config().default_quant);
    let results = hf::search(&st.http, &query, 12)
        .await
        .map_err(|e| e.to_string())?;
    // Parallel enrichment, bounded at 12.
    let mut enriched: Vec<ModelWithStats> = Vec::with_capacity(results.len());
    for m in results {
        let stats = hf::enrich(&st.http, &m.id).await;
        let (params_b, context, context_source, context_estimated, head_dim) = match &stats {
            Some(s) => (s.params_b, Some(s.context), Some(s.context_source), s.context_estimated, s.head_dim),
            None => (None, None, None, false, None),
        };
        let bandwidth = st
            .gpu
            .lock()
            .unwrap()
            .as_ref()
            .map(|g| estimate::gpu_bandwidth(&g.name).0)
            .unwrap_or(700.0);
        let max_tok_s = match (params_b, bandwidth) {
            (Some(pb), bw) if pb > 0.0 => Some(estimate::tokens_per_sec(bw, pb, &quant)),
            _ => None,
        };
        enriched.push(ModelWithStats {
            id: m.id,
            downloads: m.downloads,
            likes: m.likes,
            trending_score: m.trending_score,
            pipeline_tag: m.pipeline_tag,
            params_b,
            context,
            context_source,
            context_estimated,
            head_dim,
            max_tok_s,
            quant_assumed: quant.clone(),
        });
    }
    Ok(enriched)
}

#[derive(Serialize)]
pub struct ModelStats {
    pub model_id: String,
    pub params_b: Option<f64>,
    pub context: Option<usize>,
    pub context_source: Option<&'static str>,
    pub context_estimated: bool,
    pub head_dim: Option<usize>,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub kv_bytes_per_token: Option<f64>,
    pub weight_gb: Option<f64>,
    pub context_fit: Option<usize>,
    pub max_tok_s_fp16: Option<f64>,
    pub max_tok_s_fp8: Option<f64>,
    pub max_tok_s_int4: Option<f64>,
    pub measured: Option<MeasuredStats>,
    pub vram_total_mb: Option<u64>,
    pub gpu_name: Option<String>,
}

#[tauri::command]
pub async fn model_stats(
    state: State<'_, Arc<AppState>>,
    model_id: String,
    quant: Option<String>,
) -> Result<ModelStats, String> {
    let st = (*state).clone();
    let quant = quant.unwrap_or_else(|| st.config().default_quant);
    let stats = hf::enrich(&st.http, &model_id)
        .await
        .ok_or_else(|| format!("could not enrich {model_id}"))?;
    let (gpu_name, vram_mb) = {
        let gpu = st.gpu.lock().unwrap().clone();
        gpu.map(|g| (g.name, g.vram_total_mb)).unwrap_or_default()
    };
    let kv_bpt = match (stats.n_layers, stats.n_kv_heads, stats.head_dim) {
        (Some(l), Some(k), Some(h)) => Some(estimate::kv_bytes_per_token(l, k, h)),
        _ => None,
    };
    let gpu_util = st
        .config()
        .servers
        .iter()
        .find(|s| s.model_id == model_id)
        .map(|s| s.gpu_mem_util)
        .unwrap_or(0.92);
    let (weight_gb, context_fit) = match (stats.params_b, kv_bpt, vram_mb) {
        (Some(pb), Some(kb), vram) if vram > 0 => {
            let wb = estimate::weight_gb(pb, &quant);
            (
                Some(wb),
                Some(estimate::context_fit(vram as f64, gpu_util, pb, &quant, kb, 2500.0)),
            )
        }
        _ => (stats.params_b.map(|pb| estimate::weight_gb(pb, &quant)), None),
    };
    let (bw, _) = if gpu_name.is_empty() {
        (700.0, false)
    } else {
        estimate::gpu_bandwidth(&gpu_name)
    };
    let pb = stats.params_b.unwrap_or(0.0);
    let measured = st.config().measured.get(&model_id).cloned();
    Ok(ModelStats {
        model_id: model_id.clone(),
        params_b: stats.params_b,
        context: Some(stats.context),
        context_source: Some(stats.context_source),
        context_estimated: stats.context_estimated,
        head_dim: stats.head_dim,
        n_layers: stats.n_layers,
        n_kv_heads: stats.n_kv_heads,
        kv_bytes_per_token: kv_bpt,
        weight_gb,
        context_fit,
        max_tok_s_fp16: (pb > 0.0).then(|| estimate::tokens_per_sec(bw, pb, "fp16")),
        max_tok_s_fp8: (pb > 0.0).then(|| estimate::tokens_per_sec(bw, pb, "fp8")),
        max_tok_s_int4: (pb > 0.0).then(|| estimate::tokens_per_sec(bw, pb, "awq")),
        measured,
        vram_total_mb: Some(vram_mb).filter(|v| *v > 0),
        gpu_name: Some(gpu_name).filter(|g| !g.is_empty()),
    })
}

#[tauri::command]
pub fn pull_model(state: State<'_, Arc<AppState>>, app: AppHandle, model_id: String) -> Result<(), String> {
    let st = (*state).clone();
    hf::pull_model(&st, app, &model_id).map_err(|e| e.to_string())
}

#[derive(Serialize)]
pub struct PullState {
    pub pulling: Vec<String>,
}

#[tauri::command]
pub fn pull_status(state: State<'_, Arc<AppState>>) -> PullState {
    let st = (*state).clone();
    let pulling = st.pulling.lock().unwrap();
    PullState { pulling: pulling.keys().cloned().collect() }
}

// ---------------------------------------------------------------------------
// Servers
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ServerListRow {
    pub def: ServerDef,
    pub status: String,
    pub error: Option<String>,
    pub metrics: Option<server::MetricsSnapshot>,
}

#[tauri::command]
pub fn servers_list(state: State<'_, Arc<AppState>>) -> Vec<ServerListRow> {
    let st = (*state).clone();
    server::list_servers(&st)
        .into_iter()
        .map(|(def, status, error, metrics)| ServerListRow {
            def,
            status: status.label().to_string(),
            error,
            metrics,
        })
        .collect()
}

#[derive(serde::Deserialize)]
pub struct CreateServerInput {
    pub name: String,
    pub model_id: String,
    pub task: Option<String>,
    pub port: Option<u16>,
    pub gpu_mem_util: Option<f64>,
    pub max_model_len: Option<usize>,
    pub quant: Option<String>,
    pub served_model_name: Option<String>,
}

#[tauri::command]
pub async fn servers_create(
    state: State<'_, Arc<AppState>>,
    input: CreateServerInput,
) -> Result<ServerDef, String> {
    let st = (*state).clone();
    // Pick a free port if not given.
    let existing: Vec<u16> = st.config().servers.iter().map(|s| s.port).collect();
    let port = match input.port {
        Some(p) => {
            if existing.contains(&p) {
                return Err(format!("port {p} already used by another server"));
            }
            // verify it's actually free
            if std::net::TcpListener::bind(("127.0.0.1", p)).is_err() {
                return Err(format!("port {p} is in use outside the app"));
            }
            p
        }
        None => server::alloc_port(&existing).map_err(|e| e.to_string())?,
    };
    let quant = input.quant.unwrap_or_else(|| st.config().default_quant.clone());
    let task = input.task.unwrap_or_else(|| "instruct".into());
    // Default max_model_len := min(declared context, VRAM context-fit) at quant.
    let max_model_len = match input.max_model_len {
        Some(l) => Some(l),
        None => {
            let stats = hf::enrich(&st.http, &input.model_id).await;
            let gpu = st.gpu.lock().unwrap().clone();
            let fit = match (&stats, gpu.as_ref()) {
                (Some(s), Some(g))
                    if s.n_layers.is_some()
                        && s.n_kv_heads.is_some()
                        && s.head_dim.is_some()
                        && s.params_b.is_some() &&
                        g.vram_total_mb > 0 =>
                {
                    let kvb = estimate::kv_bytes_per_token(s.n_layers.unwrap(), s.n_kv_heads.unwrap(), s.head_dim.unwrap());
                    Some(estimate::context_fit(
                        g.vram_total_mb as f64,
                        input.gpu_mem_util.unwrap_or(0.92),
                        s.params_b.unwrap(),
                        &quant,
                        kvb,
                        2500.0,
                    ))
                }
                _ => None,
            };
            let max_ctx = stats.as_ref().map(|s| s.context).unwrap_or(4096);
            Some(fit.map(|f| f.min(max_ctx)).unwrap_or(max_ctx))
        }
    };
    let params_b = hf::enrich(&st.http, &input.model_id).await.and_then(|s| s.params_b);
    let def = ServerDef {
        id: format!("srv-{}", uuid::Uuid::new_v4().simple()),
        name: input.name.trim().to_string(),
        model_id: input.model_id,
        task,
        port,
        gpu_mem_util: input.gpu_mem_util.unwrap_or(0.92),
        max_model_len,
        quant,
        served_model_name: input.served_model_name.filter(|s| !s.trim().is_empty()),
        params_b,
    };
    let mut cfg = st.config.lock().unwrap();
    cfg.servers.push(def.clone());
    cfg.save().map_err(|e| e.to_string())?;
    Ok(def)
}

#[tauri::command]
pub fn servers_delete(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::stop_server(&st, &app, &id).ok();
    let mut cfg = st.config.lock().unwrap();
    cfg.servers.retain(|s| s.id != id);
    cfg.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn servers_start(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::start_server(&st, &app, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn servers_stop(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::stop_server(&st, &app, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn servers_restart(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::restart_server(&st, &app, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn servers_logs(state: State<'_, Arc<AppState>>, id: String, since: usize) -> String {
    let st = (*state).clone();
    server::server_logs(&st, &id, since)
}

#[tauri::command]
pub fn servers_metrics(state: State<'_, Arc<AppState>>, id: String) -> Option<server::MetricsSnapshot> {
    let st = (*state).clone();
    server::server_metrics(&st, &id)
}

#[tauri::command]
pub async fn servers_chat(
    state: State<'_, Arc<AppState>>,
    id: String,
    messages: Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let st = (*state).clone();
    server::chat(&st, &id, messages).await.map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn settings_get(state: State<'_, Arc<AppState>>) -> PersistedConfig {
    let st = (*state).clone();
    st.config()
}

#[derive(serde::Deserialize)]
pub struct SettingsPatch {
    pub distro: Option<String>,
    pub llm_dir: Option<String>,
    pub venv_dir: Option<String>,
    pub hf_token: Option<String>,
    pub default_quant: Option<String>,
}

#[tauri::command]
pub fn settings_set(state: State<'_, Arc<AppState>>, patch: SettingsPatch) -> Result<PersistedConfig, String> {
    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    if let Some(d) = patch.distro {
        if !d.trim().is_empty() {
            cfg.distro = d.trim().to_string();
        }
    }
    if let Some(d) = patch.llm_dir {
        if !d.trim().is_empty() {
            cfg.llm_dir = d.trim().to_string();
        }
    }
    if let Some(v) = patch.venv_dir {
        if !v.trim().is_empty() {
            cfg.venv_dir = v.trim().to_string();
        }
    }
    if let Some(t) = patch.hf_token {
        cfg.hf_token = t.trim().to_string();
    }
    if let Some(q) = patch.default_quant {
        cfg.default_quant = q;
    }
    cfg.save().map_err(|e| e.to_string())?;
    Ok(cfg.clone())
}

#[derive(Serialize)]
pub struct WslConfigInfo {
    pub path: Option<String>,
    pub content: Option<String>,
}

#[tauri::command]
pub fn wslconfig_get() -> WslConfigInfo {
    let path = dirs::home_dir().map(|h| h.join(".wslconfig"));
    let Some(p) = path else {
        return WslConfigInfo { path: None, content: None };
    };
    let content = std::fs::read_to_string(&p).ok();
    WslConfigInfo { path: Some(p.display().to_string()), content }
}

// ---------------------------------------------------------------------------
// GPU status (dashboard polling)
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn gpu_status(state: State<'_, Arc<AppState>>) -> Option<GpuSnapshot> {
    let st = (*state).clone();
    let distro = st.config().distro;
    let snap = gpu_snapshot(&distro);
    if let Some(s) = snap.clone() {
        let mut gpu = st.gpu.lock().unwrap();
        *gpu = Some(s);
    }
    snap
}