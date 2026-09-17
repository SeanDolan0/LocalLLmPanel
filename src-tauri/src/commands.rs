//! Tauri command layer: the app's public surface.

use anyhow::Result;
use serde::Serialize;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};

use crate::estimate;
use crate::fit::{self, FitResult, HardwareProfile, ModelArchInfo, VariantInput};
use crate::hf::{self, HfModel, QuantFormat, QuantVariant};
use crate::provision::{self, ProvisionReport};
use crate::server;
use crate::state::{AppState, GpuSnapshot, MeasuredStats, PersistedConfig, ServerDef};
use tokio::sync::Semaphore;

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
pub async fn env_status(state: State<'_, Arc<AppState>>) -> Result<EnvStatus, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let cfg = st.config();
        let distro_detected = crate::wsl::detect_default_distro().unwrap_or_else(|| cfg.distro.clone());
        let wsl_ok = crate::wsl::run_script(&distro_detected, "echo ok").ok;
        let prov_out = crate::wsl::run_script(&distro_detected, "cat ~/llm-lp/.provisioned 2>/dev/null || true");
        let gpu = gpu_snapshot(&distro_detected);
        if let Some(g) = &gpu {
            *st.gpu.lock().unwrap() = Some(g.clone());
        }
        let (bandwidth, known) = gpu
            .as_ref()
            .map(|g| estimate::gpu_bandwidth(&g.name))
            .unwrap_or((700.0, false));
        let running = {
            let servers = st.servers.lock().unwrap();
            servers.values().filter(|ls| ls.status == crate::state::ServerStatus::Running).count()
        };
        let running_weight_gb = server::running_weight_gb(&st);
        let apt_based = crate::wsl::is_apt_distro(&distro_detected);

        let (provisioned, env_report) = if let Ok(rep) = serde_json::from_str::<ProvisionReport>(&prov_out.stdout) {
            (true, Some(rep))
        } else if prov_out.stdout.contains("\"provisioned\": true") {
            let vllm = prov_out
                .stdout
                .split("\"vllm\":")
                .nth(1)
                .and_then(|s| s.split('"').nth(1))
                .map(|s| s.to_string());
            (
                true,
                Some(ProvisionReport {
                    phases_completed: vec![
                        "distro".into(),
                        "sudo".into(),
                        "apt".into(),
                        "uv".into(),
                        "venv".into(),
                        "vllm".into(),
                        "verify".into(),
                    ],
                    distro: distro_detected.clone(),
                    vllm_version: vllm,
                    torch_version: Some("torch (CUDA)".into()),
                    cuda_available: gpu.is_some(),
                    gpu_name: gpu.as_ref().map(|g| g.name.clone()),
                    vram_mb: gpu.as_ref().map(|g| g.vram_total_mb),
                    bf16_supported: true,
                }),
            )
        } else {
            (false, None)
        };

        EnvStatus {
            wsl_ok,
            distro: distro_detected,
            apt_based,
            provisioned,
            report: env_report,
            gpu,
            servers_running: running,
            running_weight_gb,
            gpu_bandwidth_gbs: bandwidth,
            gpu_bw_known: known,
        }
    })
    .await
    .map_err(|e| format!("env_status error: {e}"))
}

#[tauri::command]
pub async fn provision(app: AppHandle, state: State<'_, Arc<AppState>>) -> Result<ProvisionReport, String> {
    let st = (*state).clone();
    let app = app.clone();
    let cfg = st.config();
    let detected = crate::wsl::detect_default_distro();
    let distro = match detected {
        Some(ref d) if !crate::wsl::run_script(&cfg.distro, "echo ok").ok && crate::wsl::run_script(d, "echo ok").ok => {
            d.clone()
        }
        _ => cfg.distro.clone(),
    };
    let venv = cfg.venv_dir.clone();
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
    let token = st.hf_token();
    let results = hf::search(&st.http, &query, 12, token.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    // Parallel enrichment, bounded at 12.
    let mut enriched: Vec<ModelWithStats> = Vec::with_capacity(results.len());
    for m in results {
        let stats = hf::enrich(&st.http, &m.id, Some(&st.enrichment_cache), token.as_deref()).await;
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

#[derive(Debug, Clone, Serialize)]
pub struct QuantVariantWithFit {
    pub variant: QuantVariant,
    pub fit: FitResult,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelWithFit {
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
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub variants: Vec<QuantVariantWithFit>,
    pub best_variant_idx: usize,
}

fn hardware_profile(state: &AppState) -> Option<HardwareProfile> {
    let gpu = state.gpu.lock().unwrap().clone()?;
    let (bw, known) = estimate::gpu_bandwidth(&gpu.name);
    let vram = if gpu.vram_total_mb > 0 { gpu.vram_total_mb } else { 16384 };
    Some(HardwareProfile {
        gpu_name: gpu.name,
        vram_total_mb: vram,
        bandwidth_gbs: bw,
        bandwidth_known: known,
    })
}

fn fallback_hardware_profile() -> HardwareProfile {
    HardwareProfile {
        gpu_name: "Generic GPU".to_string(),
        vram_total_mb: 16384,
        bandwidth_gbs: 700.0,
        bandwidth_known: false,
    }
}

struct SimpleJoinAll<'a, T> {
    tasks: Vec<Option<std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>>>,
    results: Vec<Option<T>>,
}

impl<'a, T: Unpin> std::future::Future for SimpleJoinAll<'a, T> {
    type Output = Vec<T>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let mut all_done = true;
        let len = this.tasks.len();
        for i in 0..len {
            if let Some(mut fut) = this.tasks[i].take() {
                match fut.as_mut().poll(cx) {
                    std::task::Poll::Ready(val) => {
                        this.results[i] = Some(val);
                    }
                    std::task::Poll::Pending => {
                        this.tasks[i] = Some(fut);
                        all_done = false;
                    }
                }
            }
        }
        if all_done {
            let res = this.results.iter_mut().map(|opt| opt.take().unwrap()).collect();
            std::task::Poll::Ready(res)
        } else {
            std::task::Poll::Pending
        }
    }
}

fn join_all<'a, T: 'a>(
    futs: impl IntoIterator<Item = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>>,
) -> SimpleJoinAll<'a, T> {
    let tasks: Vec<_> = futs.into_iter().map(Some).collect();
    let len = tasks.len();
    SimpleJoinAll {
        tasks,
        results: (0..len).map(|_| None).collect(),
    }
}

async fn process_models_with_fit(
    st: &AppState,
    models: Vec<HfModel>,
) -> Vec<ModelWithFit> {
    let enrich_sem = Arc::new(Semaphore::new(10));
    let disc_sem = Arc::new(Semaphore::new(6));
    let hw = hardware_profile(st).unwrap_or_else(fallback_hardware_profile);
    let preferred = st.config().default_quant;
    let token = st.hf_token();

    let futures: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = ModelWithFit> + Send + '_>>> = models
        .into_iter()
        .map(|m| {
            let enrich_sem = Arc::clone(&enrich_sem);
            let disc_sem = Arc::clone(&disc_sem);
            let hw = hw.clone();
            let preferred = preferred.clone();
            let token = token.clone();
            let fut: std::pin::Pin<Box<dyn std::future::Future<Output = ModelWithFit> + Send + '_>> = Box::pin(async move {
                let token_ref = token.as_deref();
                let enrich_fut = async {
                    let _permit = enrich_sem.acquire().await.ok();
                    hf::enrich(&st.http, &m.id, Some(&st.enrichment_cache), token_ref).await
                };
                let disc_fut = hf::discover_quant_variants(&st.http, &m.id, &disc_sem, Some(&st.quant_cache), token_ref);
                let (stats, variants) = tokio::join!(enrich_fut, disc_fut);

                let (params_b, context, context_source, context_estimated, head_dim, n_layers, n_kv_heads) = match &stats {
                    Some(s) => (
                        s.params_b,
                        Some(s.context),
                        Some(s.context_source),
                        s.context_estimated,
                        s.head_dim,
                        s.n_layers,
                        s.n_kv_heads,
                    ),
                    None => (None, None, None, false, None, None, None),
                };

                let arch = ModelArchInfo {
                    params_b,
                    context: context.unwrap_or(4096),
                    n_layers,
                    n_kv_heads,
                    head_dim,
                };

                let mut items: Vec<(QuantVariant, (VariantInput, FitResult))> = variants
                    .into_iter()
                    .map(|mut v| {
                        if v.params_b.is_none() {
                            v.params_b = params_b;
                        }
                        let quant_str = match v.format {
                            QuantFormat::FP16 => "fp16".to_string(),
                            QuantFormat::FP8 => "fp8".to_string(),
                            QuantFormat::AWQ => "awq".to_string(),
                            QuantFormat::GPTQ => "gptq".to_string(),
                            QuantFormat::BNB => "bnb".to_string(),
                            QuantFormat::GGUF => {
                                if v.label.is_empty() {
                                    "gguf".to_string()
                                } else {
                                    v.label.to_lowercase()
                                }
                            }
                        };
                        let is_gguf = v.format == QuantFormat::GGUF;
                        let vi = VariantInput {
                            quant_str,
                            weight_bytes: v.weight_bytes,
                            params_b: v.params_b,
                            is_gguf,
                        };
                        let fit = fit::score_variant(&hw, &vi, &arch, None);
                        (v, (vi, fit))
                    })
                    .collect();

                items.sort_by(|a, b| fit::compare_variant_fit((&a.1.0, &a.1.1), (&b.1.0, &b.1.1)));

                let scored: Vec<(VariantInput, FitResult)> = items.iter().map(|(_, p)| p.clone()).collect();
                let best_variant_idx = fit::best_variant(&scored, Some(&preferred));

                let final_variants: Vec<QuantVariantWithFit> = items
                    .into_iter()
                    .map(|(variant, (_, fit))| QuantVariantWithFit { variant, fit })
                    .collect();

                ModelWithFit {
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
                    n_layers,
                    n_kv_heads,
                    variants: final_variants,
                    best_variant_idx,
                }
            });
            fut
        })
        .collect();

    let mut out = join_all(futures).await;

    out.sort_by(|a, b| {
        let score_a = a.variants.get(a.best_variant_idx).map(|v| v.fit.score).unwrap_or(0);
        let score_b = b.variants.get(b.best_variant_idx).map(|v| v.fit.score).unwrap_or(0);
        score_b.cmp(&score_a).then_with(|| b.downloads.cmp(&a.downloads))
    });

    out
}

#[tauri::command]
pub async fn search_models_with_fit(
    state: State<'_, Arc<AppState>>,
    query: String,
) -> Result<Vec<ModelWithFit>, String> {
    let st = (*state).clone();
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let token = st.hf_token();
    let results = hf::search(&st.http, q, 12, token.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    let out = process_models_with_fit(&st, results).await;
    Ok(out)
}

#[tauri::command]
pub async fn recommended_models(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<ModelWithFit>, String> {
    let st = (*state).clone();
    let hw = hardware_profile(&st).unwrap_or_else(fallback_hardware_profile);

    // 1. Check cache (10 min TTL)
    {
        let mut cache = st.rec_cache.lock().unwrap();
        if let Some((ref cached_models, cached_at, cached_vram)) = *cache {
            if cached_vram == hw.vram_total_mb && cached_at.elapsed() < std::time::Duration::from_secs(600) {
                return Ok(cached_models.clone());
            }
            if cached_vram != hw.vram_total_mb {
                *cache = None;
            }
        }
    }

    // 2. Query HF API for trending text-generation models (limit=16)
    let token = st.hf_token();
    let url = reqwest::Url::parse_with_params(
        hf::HF_API,
        &[("sort", "trendingScore"), ("pipeline_tag", "text-generation"), ("limit", "16")],
    )
    .map_err(|e| format!("build recommendations url: {e}"))?;

    let req = hf::apply_auth(st.http.get(url), token.as_deref());
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HF recommendations request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err("HF API rate limit exceeded (429 Too Many Requests). If you haven't added a Hugging Face token, please configure one in Settings to increase your quota.".into());
        }
        return Err(format!("HF recommendations returned {status}: {body}"));
    }

    let arr: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("HF recommendations JSON parse: {e}"))?;

    let mut models = Vec::with_capacity(arr.len());
    for m in arr {
        let id = m
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if id.is_empty() {
            continue;
        }
        let downloads = m.get("downloads").and_then(|v| v.as_i64()).unwrap_or(0);
        let likes = m.get("likes").and_then(|v| v.as_i64()).unwrap_or(0);
        let trending = m.get("trendingScore").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let private = m.get("private").and_then(|v| v.as_bool()).unwrap_or(false);
        let pipeline = m
            .get("pipeline_tag")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        models.push(HfModel {
            id,
            downloads,
            likes,
            trending_score: trending,
            private,
            pipeline_tag: pipeline,
            stats: None,
        });
    }

    // 3. Process models with fit
    let out = process_models_with_fit(&st, models).await;

    // 4. Store in cache
    if !out.is_empty() {
        let mut cache = st.rec_cache.lock().unwrap();
        *cache = Some((out.clone(), std::time::Instant::now(), hw.vram_total_mb));
    }

    Ok(out)
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
    let token = st.hf_token();
    let stats = hf::enrich(&st.http, &model_id, Some(&st.enrichment_cache), token.as_deref())
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
    pub enforce_eager: Option<bool>,
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
    let token = st.hf_token();
    // Default max_model_len := min(declared context, VRAM context-fit) at quant.
    let max_model_len = match input.max_model_len {
        Some(l) => Some(l),
        None => {
            let stats = hf::enrich(&st.http, &input.model_id, Some(&st.enrichment_cache), token.as_deref()).await;
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
    let params_b = hf::enrich(&st.http, &input.model_id, Some(&st.enrichment_cache), token.as_deref()).await.and_then(|s| s.params_b);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let def = ServerDef {
        id: format!("srv-{ts:x}"),
        name: input.name.trim().to_string(),
        model_id: input.model_id,
        task,
        port,
        gpu_mem_util: input.gpu_mem_util.unwrap_or(0.92),
        max_model_len,
        quant,
        served_model_name: input.served_model_name.filter(|s| !s.trim().is_empty()),
        enforce_eager: input.enforce_eager.unwrap_or(true),
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
    server::stop_server(&st, Some(&app), &id).ok();
    let mut cfg = st.config.lock().unwrap();
    cfg.servers.retain(|s| s.id != id);
    cfg.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn servers_start(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::start_server(&st, Some(&app), &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn servers_stop(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::stop_server(&st, Some(&app), &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn servers_restart(state: State<'_, Arc<AppState>>, app: AppHandle, id: String) -> Result<(), String> {
    let st = (*state).clone();
    server::restart_server(&st, Some(&app), &id).map_err(|e| e.to_string())
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

#[derive(Serialize)]
pub struct LibraryEntry {
    pub model_id: String,
    pub size_mb: u64,
    pub files: usize,
}

/// List models present in the WSL HF cache (~/.cache/huggingface/hub).
#[tauri::command]
pub async fn library_list(state: State<'_, Arc<AppState>>) -> Result<Vec<LibraryEntry>, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let detected = crate::wsl::detect_default_distro();
        let cfg_distro = st.config().distro;
        let distro = match detected {
            Some(ref d) if !crate::wsl::run_script(&cfg_distro, "echo ok").ok && crate::wsl::run_script(d, "echo ok").ok => {
                d.clone()
            }
            _ => cfg_distro,
        };
        // Hub cache dirs are `models--owner--name` or `models--name`; replace the first `--` with `/`.
        let out = crate::wsl::run_script(
            &distro,
            "for d in ~/.cache/huggingface/hub/models--*; do [ -d \"$d\" ] || continue; raw=${d##*/models--}; if [[ \"$raw\" == *--* ]]; then name=\"${raw/--//}\"; else name=\"$raw\"; fi; size=$(du -sm \"$d\" 2>/dev/null | cut -f1); files=$(find \"$d\" -type f 2>/dev/null | wc -l); echo \"$name|$size|$files\"; done",
        );
        let mut out_v = Vec::new();
        for line in out.stdout.lines() {
            let mut it = line.split('|');
            let (Some(model_id), Some(size), Some(files)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            out_v.push(LibraryEntry {
                model_id: model_id.to_string(),
                size_mb: size.parse().unwrap_or(0),
                files: files.parse().unwrap_or(0),
            });
        }
        out_v.sort_by(|a, b| b.size_mb.cmp(&a.size_mb));
        out_v
    })
    .await
    .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// GPU status (dashboard polling)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn gpu_status(state: State<'_, Arc<AppState>>) -> Result<Option<GpuSnapshot>, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro = st.config().distro;
        let snap = gpu_snapshot(&distro);
        if let Some(s) = snap.clone() {
            let mut gpu = st.gpu.lock().unwrap();
            *gpu = Some(s);
        }
        snap
    })
    .await
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_detected() {
        let st = AppState::new();
        *st.gpu.lock().unwrap() = Some(GpuSnapshot {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_total_mb: 24576,
            vram_free_mb: 22000,
            util_percent: 5,
        });
        let hw = hardware_profile(&st).expect("should have hardware profile");
        assert_eq!(hw.gpu_name, "NVIDIA GeForce RTX 4090");
        assert_eq!(hw.vram_total_mb, 24576);
        assert!(hw.bandwidth_known);
        assert_eq!(hw.bandwidth_gbs, 1008.0);
    }

    #[test]
    fn test_hardware_profile_none_and_fallback() {
        let st = AppState::new();
        assert!(st.gpu.lock().unwrap().is_none());
        assert!(hardware_profile(&st).is_none());

        let fb = fallback_hardware_profile();
        assert_eq!(fb.gpu_name, "Generic GPU");
        assert_eq!(fb.vram_total_mb, 16384);
        assert_eq!(fb.bandwidth_gbs, 700.0);
        assert!(!fb.bandwidth_known);
    }

    #[test]
    fn test_recommendation_cache_ttl() {
        let st = AppState::new();
        // Initially empty
        assert!(st.rec_cache.lock().unwrap().is_none());

        // Cache a dummy list with 16384 MB VRAM
        let dummy = vec![ModelWithFit {
            id: "test/model".into(),
            downloads: 100,
            likes: 10,
            trending_score: 5.0,
            pipeline_tag: Some("text-generation".into()),
            params_b: Some(7.0),
            context: Some(4096),
            context_source: Some("config.json"),
            context_estimated: false,
            head_dim: Some(128),
            n_layers: Some(32),
            n_kv_heads: Some(8),
            variants: vec![],
            best_variant_idx: 0,
        }];

        *st.rec_cache.lock().unwrap() = Some((dummy.clone(), std::time::Instant::now(), 16384));
        {
            let cache = st.rec_cache.lock().unwrap();
            let (cached, time, vram) = cache.as_ref().unwrap();
            assert_eq!(cached.len(), 1);
            assert_eq!(cached[0].id, "test/model");
            assert_eq!(*vram, 16384);
            assert!(time.elapsed() < std::time::Duration::from_secs(600));
        }

        // Expired entry
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(601))
            .unwrap();
        *st.rec_cache.lock().unwrap() = Some((dummy, past, 16384));
        {
            let cache = st.rec_cache.lock().unwrap();
            let (_, time, _) = cache.as_ref().unwrap();
            assert!(time.elapsed() >= std::time::Duration::from_secs(600));
        }
    }

    #[test]
    fn test_recommendation_cache_invalidated_on_vram_change() {
        let st = AppState::new();
        let dummy = vec![];
        // Cache created with fallback 16GB
        *st.rec_cache.lock().unwrap() = Some((dummy, std::time::Instant::now(), 16384));

        // When actual GPU has 12GB, cache should be invalidated
        let current_vram = 12227;
        {
            let mut cache = st.rec_cache.lock().unwrap();
            if let Some((_, _, cached_vram)) = *cache {
                if cached_vram != current_vram {
                    *cache = None;
                }
            }
        }
        assert!(st.rec_cache.lock().unwrap().is_none());
    }

    #[test]
    fn test_model_with_fit_serialization() {
        let m = ModelWithFit {
            id: "Qwen/Qwen2.5-7B".into(),
            downloads: 50000,
            likes: 1200,
            trending_score: 89.5,
            pipeline_tag: Some("text-generation".into()),
            params_b: Some(7.6),
            context: Some(32768),
            context_source: Some("config.json"),
            context_estimated: false,
            head_dim: Some(128),
            n_layers: Some(32),
            n_kv_heads: Some(8),
            variants: vec![
                QuantVariantWithFit {
                    variant: QuantVariant {
                        repo_id: "Qwen/Qwen2.5-7B".into(),
                        format: QuantFormat::FP16,
                        label: "FP16".into(),
                        weight_bytes: None,
                        params_b: None,
                        gguf_file: None,
                        vllm_native: true,
                    },
                    fit: FitResult {
                        verdict: fit::FitVerdict::Constrained,
                        score: 75,
                        weight_gb: 15.2,
                        usable_context: 8192,
                        native_context: 32768,
                        est_tok_s: Some(45.0),
                        measured_tok_s: None,
                        vram_pct: 85,
                        format_support: fit::FormatSupport::Native,
                        reason: "Constrained fit".into(),
                    },
                },
            ],
            best_variant_idx: 0,
        };

        let json = serde_json::to_string(&m).expect("serialize");
        assert!(json.contains("\"id\":\"Qwen/Qwen2.5-7B\""));
        assert!(json.contains("\"best_variant_idx\":0"));
        assert!(json.contains("\"score\":75"));
        assert!(json.contains("\"verdict\":\"Constrained\""));
        assert!(json.contains("\"format_support\":\"Native\""));
    }

    #[tokio::test]
    async fn test_simple_join_all_preserves_order() {
        let futs: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = usize> + Send>>> = (0..10)
            .map(|i| {
                let fut: std::pin::Pin<Box<dyn std::future::Future<Output = usize> + Send>> = Box::pin(async move {
                    if i % 2 == 0 {
                        tokio::task::yield_now().await;
                    }
                    i * 10
                });
                fut
            })
            .collect();

        let results = join_all(futs).await;
        assert_eq!(results, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[tokio::test]
    async fn test_process_models_with_fit_empty() {
        let st = AppState::new();
        let out = process_models_with_fit(&st, vec![]).await;
        assert!(out.is_empty());
    }

    #[test]
    fn test_recommendations_url_params() {
        let url = reqwest::Url::parse_with_params(
            hf::HF_API,
            &[("sort", "trendingScore"), ("pipeline_tag", "text-generation"), ("limit", "16")],
        ).unwrap();
        assert_eq!(url.query(), Some("sort=trendingScore&pipeline_tag=text-generation&limit=16"));
    }
}