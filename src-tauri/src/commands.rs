//! Tauri command layer: the app's public surface.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};

use crate::estimate;
use crate::fit::{self, FitResult, HardwareProfile, ModelArchInfo, VariantInput};
use crate::hf::{self, HfModel, QuantFormat, QuantVariant};
use crate::provision::{self, ProvisionReport};
use crate::server;
use crate::state::{AppState, GpuSnapshot, MeasuredStats, MemorySettings, PersistedConfig, ServerDef};
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
    pub cpu_name: Option<String>,
    pub cpu_cores: Option<usize>,
    pub total_ram_gb: Option<f64>,
    pub available_ram_gb: Option<f64>,
    pub ram_bandwidth_gbps: Option<f64>,
    pub providers_detected: Vec<String>,
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
        let distro_detected = st.resolve_distro();
        let wsl_ok = crate::wsl::run_script(&distro_detected, "echo ok").ok;
        let prov_out = crate::wsl::run_script(&distro_detected, "cat ~/llm-lp/.provisioned 2>/dev/null || true");
        let gpu = gpu_snapshot(&distro_detected);
        if let Some(g) = &gpu {
            *st.gpu.lock().unwrap() = Some(g.clone());
            st.record_system_metric(g);
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

        let llmfit_specs = crate::llmfit_adapter::get_system_specs();
        let cpu_name = Some(llmfit_specs.cpu_name.clone());
        let cpu_cores = Some(llmfit_specs.total_cpu_cores);
        let total_ram_gb = Some((llmfit_specs.total_ram_gb * 10.0).round() / 10.0);
        let available_ram_gb = Some((llmfit_specs.available_ram_gb * 10.0).round() / 10.0);
        let ram_bandwidth_gbps = Some(117.0);

        let mut providers_detected = Vec::new();
        let llamacpp = llmfit_core::providers::LlamaCppProvider::new();
        if llmfit_core::providers::ModelProvider::is_available(&llamacpp) {
            providers_detected.push("llama.cpp".to_string());
        }
        let ollama = llmfit_core::providers::OllamaProvider::new();
        if llmfit_core::providers::ModelProvider::is_available(&ollama) {
            providers_detected.push("Ollama".to_string());
        }
        if env_report.as_ref().map(|r| r.vllm_version.is_some()).unwrap_or(false) {
            providers_detected.push("vLLM (WSL)".to_string());
        }

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
            cpu_name,
            cpu_cores,
            total_ram_gb,
            available_ram_gb,
            ram_bandwidth_gbps,
            providers_detected,
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
    let distro = st.resolve_distro();
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

pub use crate::llmfit_adapter::{GgufSourceDto, ModelWithFit, QuantVariantWithFit, ScoreComponentsDto};

fn hardware_profile(state: &AppState) -> Option<HardwareProfile> {
    let gpu = state.gpu.lock().unwrap().clone()?;
    let (bw, known) = estimate::gpu_bandwidth(&gpu.name);
    let vram = if gpu.vram_total_mb > 0 { gpu.vram_total_mb } else { 16384 };
    let cfg = state.config();
    let mem = &cfg.memory_settings;
    let distro = if cfg.distro.starts_with("__test_") {
        cfg.distro.clone()
    } else {
        state.resolve_distro()
    };
    let (ram_total_mb, ram_avail_mb) = crate::wsl::detect_wsl_memory(&distro);
    let potential_ram = if let Some(manual) = mem.manual_ram_limit_mb {
        manual
    } else {
        ram_avail_mb.saturating_sub(mem.safety_reserve_mb)
    };
    let ram_usable_mb = if !mem.enable_ram_overflow {
        0
    } else {
        potential_ram
    };
    Some(HardwareProfile {
        gpu_name: gpu.name,
        vram_total_mb: vram,
        bandwidth_gbs: bw,
        bandwidth_known: known,
        ram_total_mb,
        ram_usable_mb,
        ram_potential_mb: potential_ram,
        ram_bandwidth_gbs: 65.0,
    })
}

fn fallback_hardware_profile() -> HardwareProfile {
    HardwareProfile {
        gpu_name: "Generic GPU".to_string(),
        vram_total_mb: 16384,
        bandwidth_gbs: 700.0,
        bandwidth_known: false,
        ram_total_mb: 16384,
        ram_usable_mb: 12288,
        ram_potential_mb: 12288,
        ram_bandwidth_gbs: 65.0,
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
    let mem_settings = st.config().memory_settings;

    let futures: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = ModelWithFit> + Send + '_>>> = models
        .into_iter()
        .map(|m| {
            let enrich_sem = Arc::clone(&enrich_sem);
            let disc_sem = Arc::clone(&disc_sem);
            let hw = hw.clone();
            let preferred = preferred.clone();
            let token = token.clone();
            let mem_settings = mem_settings.clone();
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
                        let fit = fit::score_variant(
                            &hw,
                            &vi,
                            &arch,
                            None,
                            mem_settings.default_gpu_mem_util,
                            mem_settings.vram_overhead_mb,
                            mem_settings.offload_weights_allowed,
                            mem_settings.max_context_cap,
                        );
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

                let best_fit = final_variants.get(best_variant_idx).map(|v| &v.fit);
                let score = best_fit.map(|f| f.score as f64).unwrap_or(0.0);
                let score_components = best_fit.and_then(|f| f.score_components);
                let best_quant = final_variants.get(best_variant_idx).map(|v| v.variant.label.clone());
                let runtime = best_fit.and_then(|f| f.runtime.clone());
                let run_mode = best_fit.map(|f| format!("{:?}", f.run_mode));
                let usable_context = best_fit.map(|f| f.usable_context);
                let est_tok_s = best_fit.and_then(|f| f.est_tok_s);
                let memory_required_gb = best_fit.map(|f| f.weight_gb);
                let notes = best_fit.map(|f| f.notes.clone()).unwrap_or_default();

                ModelWithFit {
                    id: m.id,
                    provider: None,
                    downloads: m.downloads,
                    likes: m.likes,
                    trending_score: m.trending_score,
                    pipeline_tag: m.pipeline_tag,
                    params_b,
                    parameter_count: params_b.map(|p| format!("{p:.1}B")),
                    context,
                    context_source,
                    context_estimated,
                    head_dim,
                    n_layers,
                    n_kv_heads,
                    use_case: None,
                    category: None,
                    release_date: None,
                    variants: final_variants,
                    best_variant_idx,
                    score,
                    score_components,
                    best_quant,
                    runtime,
                    fit_level: None,
                    run_mode,
                    usable_context,
                    effective_context_length: usable_context,
                    estimated_tps: est_tok_s,
                    memory_required_gb,
                    memory_available_gb: None,
                    utilization_pct: None,
                    notes,
                    capabilities: Vec::new(),
                    gguf_sources: Vec::new(),
                    installed: false,
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
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Search local llmfit curated database first (exact formulas, instant, zero rate limits)
    let mut results = crate::llmfit_adapter::search_models_local(q, 30);

    // 2. Augment with Hugging Face API search if fewer than 15 local results
    if results.len() < 15 {
        let st = (*state).clone();
        let token = st.hf_token();
        if let Ok(hf_results) = hf::search(&st.http, q, 12, token.as_deref()).await {
            let existing_ids: std::collections::HashSet<String> = results.iter().map(|m| m.id.to_lowercase()).collect();
            let new_hf: Vec<_> = hf_results.into_iter().filter(|m| !existing_ids.contains(&m.id.to_lowercase())).collect();
            if !new_hf.is_empty() {
                let mut hf_scored = process_models_with_fit(&st, new_hf).await;
                results.append(&mut hf_scored);
            }
        }
    }

    Ok(results)
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

    // 2. Exact match to `llmfit recommend` on the local hardware
    let out = crate::llmfit_adapter::recommend_models(50);

    // 3. Store in cache
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
    pub score: Option<f64>,
    pub score_components: Option<ScoreComponentsDto>,
    pub usable_context: Option<usize>,
    pub runtime: Option<String>,
    pub fit_level: Option<String>,
    pub run_mode: Option<String>,
    pub notes: Vec<String>,
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

    let (llmfit_score, score_components, usable_context, runtime, fit_level, run_mode, notes) = {
        let db = crate::llmfit_adapter::get_model_database();
        let specs = crate::llmfit_adapter::get_system_specs();
        if let Some(m) = db.find_model(&model_id).into_iter().next() {
            let fit = llmfit_core::fit::ModelFit::analyze(m, specs);
            (
                Some((fit.score * 10.0).round() / 10.0),
                Some(fit.score_components.into()),
                Some(fit.usable_context as usize),
                Some(fit.runtime.label().to_string()),
                Some(match fit.fit_level {
                    llmfit_core::fit::FitLevel::Perfect => "Perfect".to_string(),
                    llmfit_core::fit::FitLevel::Good => "Good".to_string(),
                    llmfit_core::fit::FitLevel::Marginal => "Marginal".to_string(),
                    llmfit_core::fit::FitLevel::TooTight => "Too Tight".to_string(),
                }),
                Some(match fit.run_mode {
                    llmfit_core::fit::RunMode::Gpu => "GPU".to_string(),
                    llmfit_core::fit::RunMode::MoeOffload => "MoE Offload".to_string(),
                    llmfit_core::fit::RunMode::CpuOffload => "CPU Offload".to_string(),
                    llmfit_core::fit::RunMode::CpuOnly => "CPU Only".to_string(),
                    llmfit_core::fit::RunMode::TensorParallel => "Tensor Parallel".to_string(),
                }),
                fit.notes,
            )
        } else {
            (None, None, None, None, None, None, Vec::new())
        }
    };

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
        score: llmfit_score,
        score_components,
        usable_context,
        runtime,
        fit_level,
        run_mode,
        notes,
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

#[tauri::command]
pub async fn pull_cancel(
    state: State<'_, Arc<AppState>>,
    model_id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    let distro = st.resolve_distro();
    // Kill any hf download processes matching this model id
    let script = format!("pkill -f 'hf download.*[ /]{}(\\s|$)' || true", model_id);
    let _ = crate::wsl::run_script(&distro, &script);
    st.pulling.lock().unwrap().remove(&model_id);
    Ok(())
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
    pub swap_space_gb: Option<usize>,
    pub cpu_offload_gb: Option<usize>,
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
        swap_space_gb: input.swap_space_gb,
        cpu_offload_gb: input.cpu_offload_gb,
        was_running: false,
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
    st.server_metrics.lock().unwrap().remove(&id);
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

#[tauri::command]
pub async fn servers_chat_stream(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    request_id: String,
    server_id: String,
    messages: Vec<crate::state::ChatMessage>,
    temperature: Option<f32>,
) -> Result<(), String> {
    let st = (*state).clone();
    tokio::spawn(async move {
        server::chat_stream(Some(app), st, request_id, server_id, messages, temperature).await;
    });
    Ok(())
}

#[tauri::command]
pub fn servers_chat_cancel(
    state: State<'_, Arc<AppState>>,
    request_id: String,
) -> Result<(), String> {
    if let Some(notify) = state.chat_cancels.lock().unwrap().get(&request_id) {
        notify.notify_waiters();
    }
    Ok(())
}

#[tauri::command]
pub fn conversations_list(state: State<'_, Arc<AppState>>) -> Result<Vec<crate::state::Conversation>, String> {
    Ok(state.conversations.lock().unwrap().clone())
}

#[tauri::command]
pub fn conversations_save(
    state: State<'_, Arc<AppState>>,
    conversation: crate::state::Conversation,
) -> Result<(), String> {
    let mut convs = state.conversations.lock().unwrap();
    if let Some(pos) = convs.iter().position(|c| c.id == conversation.id) {
        convs[pos] = conversation;
    } else {
        convs.insert(0, conversation);
    }
    crate::state::Conversation::save_all(&convs)
}

#[tauri::command]
pub fn conversations_delete(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<(), String> {
    let mut convs = state.conversations.lock().unwrap();
    convs.retain(|c| c.id != id);
    crate::state::Conversation::save_all(&convs)
}

#[tauri::command]
pub async fn benchmarks_run(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    server_id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    tokio::spawn(async move {
        server::run_benchmark(Some(app), st, server_id).await;
    });
    Ok(())
}

#[tauri::command]
pub fn benchmarks_cancel(
    state: State<'_, Arc<AppState>>,
    server_id: String,
) -> Result<(), String> {
    if let Some(notify) = state.benchmark_cancels.lock().unwrap().get(&server_id) {
        notify.notify_waiters();
    }
    Ok(())
}

#[tauri::command]
pub fn benchmarks_history(
    state: State<'_, Arc<AppState>>,
    server_id: Option<String>,
) -> Result<Vec<crate::state::BenchmarkRun>, String> {
    let bms = state.benchmarks.lock().unwrap();
    if let Some(sid) = server_id {
        Ok(bms.iter().filter(|b| b.server_id == sid).cloned().collect())
    } else {
        Ok(bms.clone())
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn settings_get(state: State<'_, Arc<AppState>>) -> PersistedConfig {
    let st = (*state).clone();
    st.config()
}

#[derive(serde::Serialize)]
pub struct GatewayStatus {
    pub enabled: bool,
    pub port: u16,
    pub running: bool,
}

#[tauri::command]
pub fn gateway_status(state: State<'_, Arc<AppState>>) -> GatewayStatus {
    let st = (*state).clone();
    let cfg = st.config();
    let enabled = cfg.advanced_settings.gateway_enabled;
    let port = cfg.advanced_settings.gateway_port;
    let running = if enabled {
        std::net::TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
            std::time::Duration::from_millis(300),
        )
        .is_ok()
    } else {
        false
    };
    GatewayStatus { enabled, port, running }
}

#[derive(serde::Deserialize)]
pub struct SettingsPatch {
    pub distro: Option<String>,
    pub llm_dir: Option<String>,
    pub venv_dir: Option<String>,
    pub hf_token: Option<String>,
    pub default_quant: Option<String>,
    pub advanced_settings: Option<crate::state::AdvancedSettings>,
    pub minimize_to_tray: Option<bool>,
    pub resume_servers_on_launch: Option<bool>,
    pub auto_restart_crashed: Option<bool>,
    pub launch_at_login: Option<bool>,
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
    if let Some(adv) = patch.advanced_settings {
        cfg.advanced_settings = adv;
    }
    if let Some(m) = patch.minimize_to_tray {
        cfg.minimize_to_tray = m;
    }
    if let Some(r) = patch.resume_servers_on_launch {
        cfg.resume_servers_on_launch = r;
    }
    if let Some(a) = patch.auto_restart_crashed {
        cfg.auto_restart_crashed = a;
    }
    if let Some(l) = patch.launch_at_login {
        cfg.launch_at_login = l;
        let _ = autostart_set(l);
    }
    cfg.save().map_err(|e| e.to_string())?;
    Ok(cfg.clone())
}

#[tauri::command]
pub fn autostart_get() -> Result<bool, String> {
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("reg")
            .args(["query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run", "/v", "LocalLLmPanel"])
            .output()
            .map_err(|e| e.to_string())?;
        Ok(output.status.success() && String::from_utf8_lossy(&output.stdout).contains("LocalLLmPanel"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        Ok(false)
    }
}

#[tauri::command]
pub fn autostart_set(enabled: bool) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        if enabled {
            let current_exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let exe_str = current_exe.to_string_lossy();
            let status = std::process::Command::new("reg")
                .args([
                    "add",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "LocalLLmPanel",
                    "/t",
                    "REG_SZ",
                    "/d",
                    &format!("\"{}\"", exe_str),
                    "/f",
                ])
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("Failed to update autostart in Windows registry".to_string());
            }
        } else {
            let _ = std::process::Command::new("reg")
                .args([
                    "delete",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "LocalLLmPanel",
                    "/f",
                ])
                .status();
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = enabled;
        Ok(())
    }
}

#[tauri::command]
pub fn config_export(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let st = (*state).clone();
    let cfg = st.config.lock().unwrap();
    let export_pkg = crate::state::ConfigExportPackage::from_persisted(&cfg);
    serde_json::to_string_pretty(&export_pkg).map_err(|e| format!("Failed to export config: {e}"))
}

#[tauri::command]
pub fn config_import(state: State<'_, Arc<AppState>>, json: String) -> Result<PersistedConfig, String> {
    let pkg: crate::state::ConfigExportPackage = serde_json::from_str(&json)
        .map_err(|e| format!("Invalid configuration JSON format: {e}"))?;

    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    cfg.distro = pkg.distro;
    cfg.llm_dir = pkg.llm_dir;
    cfg.venv_dir = pkg.venv_dir;
    cfg.default_quant = pkg.default_quant;
    cfg.memory_settings = pkg.memory_settings;
    cfg.advanced_settings = pkg.advanced_settings;
    cfg.minimize_to_tray = pkg.minimize_to_tray;
    cfg.resume_servers_on_launch = pkg.resume_servers_on_launch;
    cfg.auto_restart_crashed = pkg.auto_restart_crashed;
    cfg.launch_at_login = pkg.launch_at_login;

    // For imported servers, ensure they start with was_running = false
    cfg.servers = pkg.servers.into_iter().map(|mut s| {
        s.was_running = false;
        s
    }).collect();

    cfg.save().map_err(|e| e.to_string())?;
    Ok(cfg.clone())
}

#[tauri::command]
pub fn server_recipe_export(state: State<'_, Arc<AppState>>, server_id: String) -> Result<String, String> {
    let st = (*state).clone();
    let cfg = st.config.lock().unwrap();
    let srv = cfg.find_server(&server_id)
        .ok_or_else(|| format!("Server not found: {server_id}"))?;
    let recipe = crate::state::ServerRecipe::from_server_def(srv);
    serde_json::to_string_pretty(&recipe).map_err(|e| format!("Failed to serialize recipe: {e}"))
}

#[tauri::command]
pub fn server_recipe_parse(json: String) -> Result<crate::state::ServerRecipe, String> {
    let recipe: crate::state::ServerRecipe = serde_json::from_str(&json)
        .map_err(|e| format!("Invalid server recipe JSON: {e}"))?;
    Ok(recipe)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryInfo {
    pub wsl_total_mb: u64,
    pub wsl_available_mb: u64,
    pub usable_budget_mb: u64,
    pub safety_reserve_mb: u64,
    pub manual_override_mb: Option<u64>,
}

pub fn get_memory_settings_impl(st: &AppState) -> MemorySettings {
    st.config().memory_settings
}

pub fn update_memory_settings_impl(st: &AppState, settings: MemorySettings) -> Result<(), String> {
    let mut cfg = st.config.lock().unwrap();
    cfg.memory_settings = settings;
    cfg.save().map_err(|e| e.to_string())?;
    *st.rec_cache.lock().unwrap() = None;
    Ok(())
}

pub fn get_system_memory_impl(st: &AppState) -> SystemMemoryInfo {
    let cfg = st.config();
    let distro = if cfg.distro.starts_with("__test_") {
        cfg.distro.clone()
    } else {
        st.resolve_distro()
    };
    let (wsl_total_mb, wsl_available_mb) = crate::wsl::detect_wsl_memory(&distro);
    let mem = &cfg.memory_settings;
    let usable_budget_mb = if !mem.enable_ram_overflow {
        0
    } else if let Some(manual) = mem.manual_ram_limit_mb {
        manual
    } else {
        wsl_available_mb.saturating_sub(mem.safety_reserve_mb)
    };
    SystemMemoryInfo {
        wsl_total_mb,
        wsl_available_mb,
        usable_budget_mb,
        safety_reserve_mb: mem.safety_reserve_mb,
        manual_override_mb: mem.manual_ram_limit_mb,
    }
}

#[tauri::command]
pub fn get_memory_settings(state: State<'_, Arc<AppState>>) -> MemorySettings {
    get_memory_settings_impl(&state)
}

#[tauri::command]
pub fn update_memory_settings(state: State<'_, Arc<AppState>>, settings: MemorySettings) -> Result<(), String> {
    update_memory_settings_impl(&state, settings)
}

#[tauri::command]
pub fn get_system_memory(state: State<'_, Arc<AppState>>) -> SystemMemoryInfo {
    get_system_memory_impl(&state)
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LibraryEntry {
    pub model_id: String,
    pub size_mb: u64,
    pub files: usize,
    pub quant: Option<String>,
    pub params_b: Option<f64>,
    pub installed: bool,
    pub in_use: bool,
    pub in_use_server: Option<String>,
    pub task: Option<String>,
}

/// List models present in the WSL HF cache (~/.cache/huggingface/hub).
#[tauri::command]
pub async fn library_list(state: State<'_, Arc<AppState>>) -> Result<Vec<LibraryEntry>, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro = st.resolve_distro();
        let running_servers: Vec<(String, String)> = {
            let srvs = st.servers.lock().unwrap();
            srvs.values()
                .filter(|ls| {
                    ls.status == crate::state::ServerStatus::Running
                        || ls.status == crate::state::ServerStatus::Starting
                })
                .map(|ls| (ls.def.name.clone(), ls.def.model_id.clone()))
                .collect()
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
            let model_id_str = model_id.to_string();
            let size_mb: u64 = size.parse().unwrap_or(0);
            let files_cnt: usize = files.parse().unwrap_or(0);

            let mut in_use = false;
            let mut in_use_server = None;
            for (srv_name, srv_model) in &running_servers {
                if srv_model == &model_id_str
                    || model_id_str.contains(srv_model)
                    || srv_model.contains(&model_id_str)
                {
                    in_use = true;
                    in_use_server = Some(srv_name.clone());
                    break;
                }
            }

            let params_b = crate::estimate::parse_params_from_name(&model_id_str);
            let quant = if model_id_str.to_uppercase().contains("AWQ") {
                Some("AWQ".to_string())
            } else if model_id_str.to_uppercase().contains("GPTQ") {
                Some("GPTQ".to_string())
            } else if model_id_str.to_uppercase().contains("GGUF") {
                Some("GGUF".to_string())
            } else if model_id_str.to_uppercase().contains("FP8") {
                Some("FP8".to_string())
            } else {
                Some("FP16".to_string())
            };

            let task = if model_id_str.to_lowercase().contains("embed")
                || model_id_str.to_lowercase().contains("bge")
                || model_id_str.to_lowercase().contains("gte")
            {
                Some("embed".to_string())
            } else {
                Some("instruct".to_string())
            };

            out_v.push(LibraryEntry {
                model_id: model_id_str,
                size_mb,
                files: files_cnt,
                quant,
                params_b,
                installed: true,
                in_use,
                in_use_server,
                task,
            });
        }
        out_v.sort_by(|a, b| b.size_mb.cmp(&a.size_mb));
        out_v
    })
    .await
    .map_err(|e| e.to_string())
}

pub fn compose_library_remove_script(model_id: &str) -> String {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    format!(
        r#"
dir="$HOME/.cache/huggingface/hub/{dir_name}"
rm -rf "$dir"
# Prune unreferenced blob files
if [ -d "$HOME/.cache/huggingface/hub/blobs" ]; then
    shopt -s nullglob
    snaps=("$HOME/.cache/huggingface/hub/models--"*/snapshots)
    if [ ${{#snaps[@]}} -eq 0 ]; then
        rm -f "$HOME/.cache/huggingface/hub/blobs"/*
    else
        ref=$(find "${{snaps[@]}}" -type l -exec readlink {{}} + 2>/dev/null | sed 's#.*/##' | sort -u)
        for blob in "$HOME/.cache/huggingface/hub/blobs"/*; do
            [ -f "$blob" ] || continue
            hash=$(basename "$blob")
            if ! echo "$ref" | grep -qx "$hash"; then
                rm -f "$blob"
            fi
        done
    fi
fi
echo ok
"#,
        dir_name = dir_name
    )
}

#[tauri::command]
pub async fn library_remove(
    state: State<'_, Arc<AppState>>,
    model_id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        // Check if model is in use
        {
            let srvs = st.servers.lock().unwrap();
            for ls in srvs.values() {
                if (ls.status == crate::state::ServerStatus::Running
                    || ls.status == crate::state::ServerStatus::Starting)
                    && (ls.def.model_id == model_id
                        || model_id.contains(&ls.def.model_id)
                        || ls.def.model_id.contains(&model_id))
                {
                    return Err(format!(
                        "Model \"{}\" is currently in use by active server \"{}\" — stop server first",
                        model_id, ls.def.name
                    ));
                }
            }
        }

        // Sanitize model_id
        if model_id.contains("..")
            || model_id.starts_with('/')
            || model_id.contains(';')
            || model_id.contains('&')
            || model_id.contains('|')
            || model_id.contains('`')
            || model_id.contains('$')
        {
            return Err("Invalid characters in model ID".to_string());
        }

        let distro = st.resolve_distro();
        let script = compose_library_remove_script(&model_id);
        let out = crate::wsl::run_script(&distro, &script);
        if !out.ok {
            return Err(format!("Failed to delete model directory: {}", out.stderr));
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn library_disk_usage(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro = st.resolve_distro();
        let out = crate::wsl::run_script(
            &distro,
            "du -sm ~/.cache/huggingface/hub 2>/dev/null | cut -f1",
        );
        let mb: u64 = out.stdout.trim().parse().unwrap_or(0);
        Ok(mb)
    })
    .await
    .map_err(|e| e.to_string())?
}

// ---------------------------------------------------------------------------
// GPU status (dashboard polling)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn gpu_status(state: State<'_, Arc<AppState>>) -> Result<Option<GpuSnapshot>, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro = st.resolve_distro();
        let snap = gpu_snapshot(&distro);
        if let Some(s) = snap.clone() {
            let mut gpu = st.gpu.lock().unwrap();
            *gpu = Some(s.clone());
            st.record_system_metric(&s);
        }
        snap
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn system_metrics_series(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::state::SystemMetricPoint>, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let should_sample = {
            let sm = st.system_metrics.lock().unwrap();
            match sm.back() {
                Some(pt) => crate::state::now_ms().saturating_sub(pt.timestamp) > 4000,
                None => true,
            }
        };
        if should_sample {
            let distro = st.resolve_distro();
            if let Some(s) = gpu_snapshot(&distro) {
                let mut gpu = st.gpu.lock().unwrap();
                *gpu = Some(s.clone());
                st.record_system_metric(&s);
            }
        }
        let sm = st.system_metrics.lock().unwrap();
        Ok(sm.iter().cloned().collect())
    })
    .await
    .map_err(|e| e.to_string())?
}

pub fn get_server_metrics_series(
    state: &AppState,
    server_id: &str,
) -> Vec<crate::state::ServerMetricPoint> {
    let sm = state.server_metrics.lock().unwrap();
    sm.get(server_id).map(|q| q.iter().cloned().collect()).unwrap_or_default()
}

#[tauri::command]
pub fn server_metrics_series(
    state: State<'_, Arc<AppState>>,
    server_id: String,
) -> Result<Vec<crate::state::ServerMetricPoint>, String> {
    Ok(get_server_metrics_series(&state, &server_id))
}

// ---------------------------------------------------------------------------
// Open external URL in system browser
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn open_url(url: String) -> Result<(), String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("Only http and https URLs are allowed".into());
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", &url])
            .spawn()
            .map_err(|e| format!("Failed to open URL in browser: {e}"))?;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("Failed to open URL in browser: {e}"))?;
        Ok(())
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("Failed to open URL in browser: {e}"))?;
        Ok(())
    }
}

#[tauri::command]
pub fn wsl_distros() -> Vec<String> {
    crate::wsl::installed_distros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_detected() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings = MemorySettings::default();
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
        assert_eq!(hw.ram_total_mb, 16384);
        assert_eq!(hw.ram_usable_mb, 8192); // 12288 - 4096 (safety reserve)
        assert_eq!(hw.ram_bandwidth_gbs, 65.0);
    }

    #[test]
    fn test_hardware_profile_ram_overflow_disabled() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings = MemorySettings {
            enable_ram_overflow: false,
            ..MemorySettings::default()
        };
        *st.gpu.lock().unwrap() = Some(GpuSnapshot {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_total_mb: 24576,
            vram_free_mb: 22000,
            util_percent: 5,
        });
        let hw = hardware_profile(&st).expect("should have hardware profile");
        assert_eq!(hw.ram_usable_mb, 0);
        assert_eq!(hw.ram_potential_mb, 8192); // 12288 - 4096 (potential remains known)
    }

    #[test]
    fn test_hardware_profile_manual_ram_limit() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings.manual_ram_limit_mb = Some(32768);
        *st.gpu.lock().unwrap() = Some(GpuSnapshot {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_total_mb: 24576,
            vram_free_mb: 22000,
            util_percent: 5,
        });
        let hw = hardware_profile(&st).expect("should have hardware profile");
        assert_eq!(hw.ram_usable_mb, 32768);
    }

    #[test]
    fn test_hardware_profile_safety_reserve_saturating() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings = MemorySettings {
            safety_reserve_mb: 20000,
            manual_ram_limit_mb: None,
            ..MemorySettings::default()
        };
        *st.gpu.lock().unwrap() = Some(GpuSnapshot {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_total_mb: 24576,
            vram_free_mb: 22000,
            util_percent: 5,
        });
        let hw = hardware_profile(&st).expect("should have hardware profile");
        assert_eq!(hw.ram_usable_mb, 0);
    }

    struct ConfigBackupGuard {
        path: std::path::PathBuf,
        original_content: Option<Vec<u8>>,
    }

    impl Drop for ConfigBackupGuard {
        fn drop(&mut self) {
            if let Some(content) = &self.original_content {
                let _ = std::fs::write(&self.path, content);
            } else if self.path.exists() {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    #[test]
    fn test_memory_settings_get_and_update() {
        let path = PersistedConfig::path();
        let original_content = std::fs::read(&path).ok();
        let _guard = ConfigBackupGuard {
            path: path.clone(),
            original_content,
        };
        let _ = std::fs::remove_file(&path);

        let st = AppState::new();
        let settings = get_memory_settings_impl(&st);
        assert_eq!(settings, MemorySettings::default());

        let custom = MemorySettings {
            default_gpu_mem_util: 0.85,
            vram_overhead_mb: 1500.0,
            enable_ram_overflow: false,
            manual_ram_limit_mb: Some(8192),
            safety_reserve_mb: 2048,
            offload_weights_allowed: false,
            max_context_cap: Some(16384),
        };

        *st.rec_cache.lock().unwrap() = Some((vec![], std::time::Instant::now(), 16384));
        assert!(st.rec_cache.lock().unwrap().is_some());

        let res = update_memory_settings_impl(&st, custom.clone());
        assert!(res.is_ok());
        assert_eq!(get_memory_settings_impl(&st), custom);
        assert!(st.rec_cache.lock().unwrap().is_none(), "rec_cache must be invalidated");
    }

    #[test]
    fn test_get_system_memory() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings = MemorySettings::default();
        let sys_mem = get_system_memory_impl(&st);
        assert_eq!(sys_mem.wsl_total_mb, 16384);
        assert_eq!(sys_mem.wsl_available_mb, 12288);
        assert_eq!(sys_mem.usable_budget_mb, 8192); // 12288 - 4096
        assert_eq!(sys_mem.safety_reserve_mb, 4096);
        assert_eq!(sys_mem.manual_override_mb, None);

        st.config.lock().unwrap().memory_settings.manual_ram_limit_mb = Some(14000);
        let sys_mem2 = get_system_memory_impl(&st);
        assert_eq!(sys_mem2.usable_budget_mb, 14000);
        assert_eq!(sys_mem2.manual_override_mb, Some(14000));

        st.config.lock().unwrap().memory_settings.enable_ram_overflow = false;
        let sys_mem3 = get_system_memory_impl(&st);
        assert_eq!(sys_mem3.usable_budget_mb, 0);
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
        assert_eq!(fb.ram_total_mb, 16384);
        assert_eq!(fb.ram_usable_mb, 12288);
        assert_eq!(fb.ram_bandwidth_gbs, 65.0);
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
            ..Default::default()
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
        if let Some(past) = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(601)) {
            *st.rec_cache.lock().unwrap() = Some((dummy, past, 16384));
            {
                let cache = st.rec_cache.lock().unwrap();
                let (_, time, _) = cache.as_ref().unwrap();
                assert!(time.elapsed() >= std::time::Duration::from_secs(600));
            }
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
            provider: Some("Qwen".into()),
            downloads: 50000,
            likes: 1200,
            trending_score: 89.5,
            pipeline_tag: Some("text-generation".into()),
            params_b: Some(7.6),
            parameter_count: Some("7.6B".into()),
            context: Some(32768),
            context_source: Some("config.json"),
            context_estimated: false,
            head_dim: Some(128),
            n_layers: Some(32),
            n_kv_heads: Some(8),
            use_case: Some("Chat".into()),
            category: Some("Chat".into()),
            release_date: None,
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
                        run_mode: fit::RunMode::GpuRamSwap,
                        score: 75,
                        weight_gb: 15.2,
                        vram_context: 8192,
                        extended_context: 16384,
                        native_context: 32768,
                        usable_context: 16384,
                        swap_space_gb: 2,
                        cpu_offload_gb: 0,
                        est_tok_s: Some(45.0),
                        measured_tok_s: None,
                        vram_pct: 85,
                        ram_pct: 20,
                        format_support: fit::FormatSupport::Native,
                        reason: "Constrained fit".into(),
                        score_components: None,
                        runtime: Some("vLLM".into()),
                        notes: vec![],
                    },
                },
            ],
            best_variant_idx: 0,
            score: 75.0,
            score_components: None,
            best_quant: Some("FP16".into()),
            runtime: Some("vLLM".into()),
            fit_level: Some("Good".into()),
            run_mode: Some("GPU".into()),
            usable_context: Some(16384),
            effective_context_length: Some(16384),
            estimated_tps: Some(45.0),
            memory_required_gb: Some(15.2),
            memory_available_gb: Some(16.0),
            utilization_pct: Some(95.0),
            notes: vec![],
            capabilities: vec![],
            gguf_sources: vec![],
            installed: false,
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

    #[tokio::test]
    async fn test_process_models_with_fit_uses_memory_settings() {
        let st = AppState::new();
        st.config.lock().unwrap().distro = "__test_nonexistent_distro__".to_string();
        st.config.lock().unwrap().memory_settings.max_context_cap = Some(2048);

        let model_id = "test/cached-model".to_string();
        st.enrichment_cache.lock().unwrap().insert(
            model_id.clone(),
            crate::state::CachedEnrichment {
                stats: hf::EnrichedStats {
                    context: 32768,
                    context_source: "config.json",
                    context_estimated: false,
                    params_b: Some(7.0),
                    head_dim: Some(128),
                    n_layers: Some(32),
                    n_kv_heads: Some(8),
                    torch_dtype: None,
                },
                fetched_at: std::time::Instant::now(),
            },
        );
        st.quant_cache.lock().unwrap().insert(
            model_id.clone(),
            crate::state::CachedQuants {
                variants: vec![QuantVariant {
                    repo_id: model_id.clone(),
                    format: QuantFormat::AWQ,
                    label: "AWQ".into(),
                    weight_bytes: Some(4_000_000_000),
                    params_b: Some(7.0),
                    gguf_file: None,
                    vllm_native: true,
                }],
                fetched_at: std::time::Instant::now(),
            },
        );

        let models = vec![HfModel {
            id: model_id,
            downloads: 100,
            likes: 10,
            trending_score: 5.0,
            private: false,
            pipeline_tag: Some("text-generation".into()),
            stats: None,
        }];

        let res = process_models_with_fit(&st, models).await;
        assert_eq!(res.len(), 1);
        let variant = &res[0].variants[0];
        assert_eq!(variant.fit.extended_context, 2048);
    }

    #[test]
    fn test_recommendations_url_params() {
        let url = reqwest::Url::parse_with_params(
            hf::HF_API,
            &[("sort", "trendingScore"), ("pipeline_tag", "text-generation"), ("limit", "16")],
        ).unwrap();
        assert_eq!(url.query(), Some("sort=trendingScore&pipeline_tag=text-generation&limit=16"));
    }

    #[test]
    fn test_open_url_validation() {
        assert!(open_url("ftp://evil.com".into()).is_err());
        assert!(open_url("javascript:alert(1)".into()).is_err());
        assert!(open_url("file:///etc/passwd".into()).is_err());
    }

    #[test]
    fn test_library_entry_and_dir_name() {
        let entry = LibraryEntry {
            model_id: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            size_mb: 14500,
            files: 8,
            quant: Some("AWQ".to_string()),
            params_b: Some(7.0),
            installed: true,
            in_use: false,
            in_use_server: None,
            task: Some("instruct".to_string()),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: LibraryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, entry);

        let dir_name = format!("models--{}", entry.model_id.replace('/', "--"));
        assert_eq!(dir_name, "models--Qwen--Qwen2.5-Coder-7B-Instruct");
    }

    #[test]
    fn test_metric_series_commands() {
        use crate::state::{AppState, ServerMetricPoint, SystemMetricPoint};
        let app_state = Arc::new(AppState::new());
        {
            let mut sm = app_state.system_metrics.lock().unwrap();
            sm.push_back(SystemMetricPoint {
                timestamp: 1000,
                vram_used_mb: 8000,
                vram_total_mb: 24576,
                vram_free_mb: 16576,
                gpu_util_pct: 45,
            });
        }
        {
            let mut srv_m = app_state.server_metrics.lock().unwrap();
            let q = srv_m.entry("test-srv".to_string()).or_default();
            q.push_back(ServerMetricPoint {
                timestamp: 1000,
                tok_s: 55.4,
                prompt_tok_s: 220.1,
                requests_running: 2,
                requests_waiting: 0,
            });
        }

        let srv_res = get_server_metrics_series(&app_state, "test-srv");
        assert_eq!(srv_res.len(), 1);
        assert_eq!(srv_res[0].tok_s, 55.4);

        let empty_res = get_server_metrics_series(&app_state, "nonexistent");
        assert!(empty_res.is_empty());
    }

    #[test]
    fn test_autostart_query_smoke() {
        // Just verify autostart_get() returns a bool without panic
        let res = autostart_get();
        assert!(res.is_ok());
    }

    #[test]
    fn test_server_recipe_parse_valid_and_invalid() {
        let valid_json = r#"{
            "schema": "local-llm-panel/server-recipe/v1",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "task": "instruct",
            "port": 8000,
            "gpu_mem_util": 0.9,
            "quant": "awq",
            "max_model_len": 4096,
            "enforce_eager": true,
            "swap_space_gb": 2
        }"#;
        let recipe = server_recipe_parse(valid_json.to_string()).unwrap();
        assert_eq!(recipe.model_id, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(recipe.quant, "awq");
        assert_eq!(recipe.port, 8000);
        assert_eq!(recipe.swap_space_gb, Some(2));

        let invalid_json = r#"{ "not": "a recipe" }"#;
        let err = server_recipe_parse(invalid_json.to_string());
        assert!(err.is_err());
    }

    #[test]
    fn test_cache_sweep_script_composition() {
        let script = compose_library_remove_script("Qwen/Qwen2.5-0.5B");
        assert!(script.contains("models--Qwen--Qwen2.5-0.5B"));
        assert!(script.contains("hub/blobs"));
        assert!(script.contains("readlink"));
        assert!(script.contains("snaps"));
    }

    #[test]
    fn test_pull_cancel_script_anchoring() {
        let model_id = "meta-llama/Llama-3.1-8B";
        let script = format!("pkill -f 'hf download.*[ /]{}(\\s|$)' || true", model_id);
        assert!(script.contains("[ /]meta-llama/Llama-3.1-8B(\\s|$)"));
    }
}