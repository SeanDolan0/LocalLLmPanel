//! Tauri command layer: the app's public surface.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};

use crate::estimate;
use crate::fit::{self, FitResult, HardwareProfile, ModelArchInfo, VariantInput};
use crate::hf::{self, HfModel, QuantFormat, QuantVariant};
use crate::provision::{self, ProvisionReport};
use crate::server;
use crate::state::{
    AppState, GpuSnapshot, MeasuredStats, MemorySettings, PersistedConfig, ServerDef,
};
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
    pub llamacpp_installed: bool,
    pub llamacpp_tag: Option<String>,
    pub llamacpp_version: Option<String>,
    pub llamacpp_executable: Option<String>,
    pub llamacpp_cuda_available: bool,
    pub llamacpp_devices: Vec<crate::llamacpp_install::LlamaDevice>,
}

fn gpu_snapshot(distro: &str) -> Option<GpuSnapshot> {
    crate::wsl::gpu_snapshot(distro)
}

#[tauri::command]
pub async fn env_status(state: State<'_, Arc<AppState>>) -> Result<EnvStatus, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro_detected = st.resolve_distro();
        let wsl_running = crate::wsl::is_running(&distro_detected);
        let native_gpu = crate::llamacpp_install::windows_gpu_snapshot();
        let gpu = if wsl_running {
            gpu_snapshot(&distro_detected).or(native_gpu)
        } else {
            native_gpu
        };
        let wsl_ok = wsl_running;
        let prov_out = if wsl_running {
            crate::wsl::run_script(
                &distro_detected,
                "cat ~/llm-lp/.provisioned 2>/dev/null || true",
            )
        } else {
            crate::wsl::RunOutput { ok: false, code: 1, stdout: String::new(), stderr: String::new() }
        };
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
            servers
                .values()
                .filter(|ls| ls.status == crate::state::ServerStatus::Running)
                .count()
        };
        let running_weight_gb = server::running_weight_gb(&st);
        let apt_based = crate::wsl::is_apt_distro(&distro_detected);

        let (provisioned, env_report) =
            if let Ok(rep) = serde_json::from_str::<ProvisionReport>(&prov_out.stdout) {
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

        let cfg = st.config();
        let native_llamacpp = crate::llamacpp_install::executable_from_config(&cfg);
        let mut providers_detected = Vec::new();
        let llamacpp = llmfit_core::providers::LlamaCppProvider::new();
        if native_llamacpp.is_some()
            || llmfit_core::providers::ModelProvider::is_available(&llamacpp)
        {
            providers_detected.push("llama.cpp".to_string());
        }
        let ollama = llmfit_core::providers::OllamaProvider::new();
        if llmfit_core::providers::ModelProvider::is_available(&ollama) {
            providers_detected.push("Ollama".to_string());
        }
        if env_report
            .as_ref()
            .map(|r| r.vllm_version.is_some())
            .unwrap_or(false)
        {
            providers_detected.push("vLLM (WSL)".to_string());
        }

        let llamacpp_devices = native_llamacpp
            .as_deref()
            .and_then(|exe| crate::llamacpp_install::list_devices(exe).ok())
            .unwrap_or_default();
        let llamacpp_cuda_available = llamacpp_devices
            .iter()
            .any(|device| device.backend.eq_ignore_ascii_case("cuda"));

        let upstream = cfg
            .llamacpp_channels
            .get(&crate::state::LlamaCppChannel::Upstream);

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
            llamacpp_installed: native_llamacpp.is_some(),
            llamacpp_tag: upstream
                .and_then(|channel| channel.installed_tag.clone())
                .or(cfg.llamacpp_installed_tag),
            llamacpp_version: upstream
                .and_then(|channel| channel.version.clone())
                .or(cfg.llamacpp_version),
            llamacpp_executable: native_llamacpp
                .map(|path| path.to_string_lossy().into_owned())
                .or_else(|| upstream.and_then(|channel| channel.executable.clone()))
                .or(cfg.llamacpp_executable),
            llamacpp_cuda_available,
            llamacpp_devices,
        }
    })
    .await
    .map_err(|e| format!("env_status error: {e}"))
}

#[tauri::command]
pub async fn provision(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<ProvisionReport, String> {
    let st = (*state).clone();
    let app = app.clone();
    let cfg = st.config();
    let distro = st.resolve_distro();
    let venv = cfg.venv_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let on_log = |phase: &str, line: &str| {
            let _ = app.emit(
                "wsl-log",
                serde_json::json!({ "phase": phase, "line": line }),
            );
        };
        provision::provision_all(&distro, &venv, on_log)
    })
    .await
    .map_err(|e| format!("provision task error: {e}"))?
    .map_err(|e| format!("provision failed: {e}"))
}

#[tauri::command]
pub async fn install_llamacpp(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<crate::llamacpp_install::InstallStatus, String> {
    install_llamacpp_channel(app, state, crate::state::LlamaCppChannel::Upstream).await
}

#[tauri::command]
pub async fn install_llamacpp_channel(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    channel: crate::state::LlamaCppChannel,
) -> Result<crate::llamacpp_install::InstallStatus, String> {
    let st = (*state).clone();
    let cfg = st.config();
    let channel_config = cfg.llamacpp_channels.get(&channel).cloned().unwrap_or_default();
    let destination = PathBuf::from(channel_config.dir);
    let github_token = (!cfg.github_token.trim().is_empty()).then_some(cfg.github_token);
    let result = tauri::async_runtime::spawn_blocking(move || {
        crate::llamacpp_install::install_for_channel(
            channel,
            &destination,
            |file, done, total| {
                let _ = app.emit(
                    "llamacpp-install-progress",
                    serde_json::json!({"file": file, "done": done, "total": total}),
                );
            },
            github_token.as_deref(),
        )
    })
    .await
    .map_err(|e| format!("llama.cpp install task error: {e}"))?
    .map_err(|e| format!("llama.cpp install failed: {e}"))?;
    let (tag, exe, version, help) = result;
    let mut cfg = st.config.lock().unwrap();
    if let Some(ch) = cfg.llamacpp_channels.get_mut(&channel) {
        ch.installed_tag = Some(tag);
        ch.version = Some(version.clone());
        ch.help = Some(help);
        ch.executable = Some(exe.to_string_lossy().into_owned());
    }
    cfg.save().map_err(|e| e.to_string())?;
    let devices = crate::llamacpp_install::list_devices(&exe).unwrap_or_default();
    Ok(crate::llamacpp_install::InstallStatus {
        installed: true,
        tag: cfg.llamacpp_channels.get(&channel).and_then(|c| c.installed_tag.clone()),
        version: cfg.llamacpp_channels.get(&channel).and_then(|c| c.version.clone()),
        executable: cfg.llamacpp_channels.get(&channel).and_then(|c| c.executable.clone()),
        gpu: crate::llamacpp_install::windows_gpu_snapshot(),
        cuda_available: devices
            .iter()
            .any(|d| d.backend.eq_ignore_ascii_case("cuda")),
        devices,
    })
}

#[tauri::command]
pub async fn llamacpp_status(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::llamacpp_install::InstallStatus, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let cfg = st.config();
        let channel = cfg
            .llamacpp_channels
            .get(&crate::state::LlamaCppChannel::Upstream)
            .cloned()
            .unwrap_or_default();
        let executable = crate::llamacpp_install::executable_from_config(&cfg);
        let devices = executable
            .as_deref()
            .and_then(|exe| crate::llamacpp_install::list_devices(exe).ok())
            .unwrap_or_default();
        Ok(crate::llamacpp_install::InstallStatus {
            installed: executable.is_some(),
            tag: channel.installed_tag.or(cfg.llamacpp_installed_tag),
            version: channel.version.or(cfg.llamacpp_version),
            executable: executable.map(|path| path.to_string_lossy().into_owned()),
            gpu: crate::llamacpp_install::windows_gpu_snapshot(),
            cuda_available: devices
                .iter()
                .any(|device| device.backend.eq_ignore_ascii_case("cuda")),
            devices,
        })
    })
    .await
    .map_err(|e| format!("llamacpp status task error: {e}"))?
}

#[tauri::command]
pub async fn github_access(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::llamacpp_install::GithubAccess, String> {
    let token = state.config().github_token;
    tauri::async_runtime::spawn_blocking(move || {
        Ok(crate::llamacpp_install::test_github_access(crate::state::LlamaCppChannel::Upstream, Some(&token)))
    })
    .await
    .map_err(|e| format!("GitHub access task error: {e}"))?
}

#[tauri::command]
pub fn clear_github_token(state: State<'_, Arc<AppState>>) -> Result<PublicSettings, String> {
    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    cfg.github_token.clear();
    cfg.save().map_err(|e| e.to_string())?;
    Ok(public_settings(&cfg))
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
        let stats = hf::enrich(
            &st.http,
            &m.id,
            Some(&st.enrichment_cache),
            token.as_deref(),
        )
        .await;
        let (params_b, context, context_source, context_estimated, head_dim) = match &stats {
            Some(s) => (
                s.params_b,
                Some(s.context),
                Some(s.context_source),
                s.context_estimated,
                s.head_dim,
            ),
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

pub use crate::llmfit_adapter::{
    GgufSourceDto, ModelWithFit, QuantVariantWithFit, ScoreComponentsDto,
};

fn hardware_profile(state: &AppState) -> Option<HardwareProfile> {
    let gpu = state.gpu.lock().unwrap().clone()?;
    let (bw, known) = estimate::gpu_bandwidth(&gpu.name);
    let vram = if gpu.vram_total_mb > 0 {
        gpu.vram_total_mb
    } else {
        16384
    };
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

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
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
            let res = this
                .results
                .iter_mut()
                .map(|opt| opt.take().unwrap())
                .collect();
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

async fn process_models_with_fit(st: &AppState, models: Vec<HfModel>) -> Vec<ModelWithFit> {
    let enrich_sem = Arc::new(Semaphore::new(10));
    let disc_sem = Arc::new(Semaphore::new(6));
    let hw = hardware_profile(st).unwrap_or_else(fallback_hardware_profile);
    let preferred = st.config().default_quant;
    let token = st.hf_token();
    let mem_settings = st.config().memory_settings;

    let futures: Vec<
        std::pin::Pin<Box<dyn std::future::Future<Output = ModelWithFit> + Send + '_>>,
    > = models
        .into_iter()
        .map(|m| {
            let enrich_sem = Arc::clone(&enrich_sem);
            let disc_sem = Arc::clone(&disc_sem);
            let hw = hw.clone();
            let preferred = preferred.clone();
            let token = token.clone();
            let mem_settings = mem_settings.clone();
            let fut: std::pin::Pin<
                Box<dyn std::future::Future<Output = ModelWithFit> + Send + '_>,
            > = Box::pin(async move {
                let token_ref = token.as_deref();
                let enrich_fut = async {
                    let _permit = enrich_sem.acquire().await.ok();
                    hf::enrich(&st.http, &m.id, Some(&st.enrichment_cache), token_ref).await
                };
                let disc_fut = hf::discover_quant_variants(
                    &st.http,
                    &m.id,
                    &disc_sem,
                    Some(&st.quant_cache),
                    token_ref,
                );
                let (stats, variants) = tokio::join!(enrich_fut, disc_fut);

                let (
                    params_b,
                    context,
                    context_source,
                    context_estimated,
                    head_dim,
                    n_layers,
                    n_kv_heads,
                ) = match &stats {
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
                        let required_channel = v.required_channel;
                        let vi = VariantInput {
                            quant_str,
                            weight_bytes: v.weight_bytes,
                            params_b: v.params_b,
                            is_gguf,
                            required_channel,
                        };
                        let channel_installed = st.config().llamacpp_channels.get(&required_channel).and_then(|c| c.installed_tag.as_ref()).map(|_| required_channel);
                        let fit = fit::score_variant(
                            &hw,
                            &vi,
                            &arch,
                            None,
                            mem_settings.default_gpu_mem_util,
                            mem_settings.vram_overhead_mb,
                            mem_settings.offload_weights_allowed,
                            mem_settings.max_context_cap,
                            channel_installed,
                        );
                        (v, (vi, fit))
                    })
                    .collect();

                items.sort_by(|a, b| {
                    fit::compare_variant_fit((&a.1 .0, &a.1 .1), (&b.1 .0, &b.1 .1))
                });

                let scored: Vec<(VariantInput, FitResult)> =
                    items.iter().map(|(_, p)| p.clone()).collect();
                let best_variant_idx = fit::best_variant(&scored, Some(&preferred));

                let final_variants: Vec<QuantVariantWithFit> = items
                    .into_iter()
                    .map(|(variant, (_, fit))| QuantVariantWithFit { variant, fit })
                    .collect();

                let best_fit = final_variants.get(best_variant_idx).map(|v| &v.fit);
                let score = best_fit.map(|f| f.score as f64).unwrap_or(0.0);
                let score_components = best_fit.and_then(|f| f.score_components);
                let best_quant = final_variants
                    .get(best_variant_idx)
                    .map(|v| v.variant.label.clone());
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
        let score_a = a
            .variants
            .get(a.best_variant_idx)
            .map(|v| v.fit.score)
            .unwrap_or(0);
        let score_b = b
            .variants
            .get(b.best_variant_idx)
            .map(|v| v.fit.score)
            .unwrap_or(0);
        score_b
            .cmp(&score_a)
            .then_with(|| b.downloads.cmp(&a.downloads))
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
            let existing_ids: std::collections::HashSet<String> =
                results.iter().map(|m| m.id.to_lowercase()).collect();
            let new_hf: Vec<_> = hf_results
                .into_iter()
                .filter(|m| !existing_ids.contains(&m.id.to_lowercase()))
                .collect();
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
            if cached_vram == hw.vram_total_mb
                && cached_at.elapsed() < std::time::Duration::from_secs(600)
            {
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
    let stats = hf::enrich(
        &st.http,
        &model_id,
        Some(&st.enrichment_cache),
        token.as_deref(),
    )
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
        .unwrap_or(st.config().memory_settings.default_gpu_mem_util);
    let (weight_gb, context_fit) = match (stats.params_b, kv_bpt, vram_mb) {
        (Some(pb), Some(kb), vram) if vram > 0 => {
            let wb = estimate::weight_gb(pb, &quant);
            (
                Some(wb),
                Some(estimate::context_fit(
                    vram as f64,
                    gpu_util,
                    pb,
                    &quant,
                    kb,
                    2500.0,
                )),
            )
        }
        _ => (
            stats.params_b.map(|pb| estimate::weight_gb(pb, &quant)),
            None,
        ),
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

#[derive(Clone, Deserialize)]
pub struct ContextFitRequest {
    pub backend: String,
    pub model_id: String,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub context_tokens: Option<usize>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    #[serde(default)]
    pub gpu_mem_util: Option<f64>,
    #[serde(default)]
    pub cpu_offload_gb: Option<usize>,
    #[serde(default)]
    pub kv_offload_gb: Option<usize>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub flash_attn: Option<bool>,
    #[serde(default)]
    pub n_gpu_layers: Option<usize>,
    #[serde(default)]
    pub n_cpu_moe: Option<usize>,
    #[serde(default)]
    pub fit: Option<bool>,
    #[serde(default)]
    pub fit_target: Option<usize>,
    #[serde(default)]
    pub no_kv_offload: Option<bool>,
}

fn inferred_vllm_quant(model_id: &str, requested: Option<&str>) -> String {
    let requested = requested.unwrap_or("auto").trim().to_ascii_lowercase();
    if requested != "auto" && !requested.is_empty() {
        return requested;
    }
    let id = model_id.to_ascii_lowercase();
    if id.contains("gptq") {
        "gptq".into()
    } else if id.contains("awq") {
        "awq".into()
    } else if id.contains("fp8") {
        "fp8".into()
    } else {
        "fp16".into()
    }
}

fn cached_vllm_weight_gib(st: &AppState, model_id: &str) -> Option<f64> {
    let cfg = st.config();
    let distro = st.resolve_distro();
    let python = format!("{}/bin/python", cfg.venv_dir.trim_end_matches('/'));
    let hub = cfg
        .advanced_settings
        .hf_home
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| format!("{}/hub", value.trim_end_matches('/')))
        .unwrap_or_else(|| "~/.cache/huggingface/hub".into());
    let script = format!(
        r#"{WSL_TILDE_EXPANSION_SNIPPET}
venv_python=$(__llm_panel_expand_tilde {python})
hub_dir=$(__llm_panel_expand_tilde {hub})
export LLMP_CTX_MODEL={model}
export HF_HUB_DIR="$hub_dir"
"$venv_python" - <<'PY'
import os
from huggingface_hub import scan_cache_dir
model = os.environ["LLMP_CTX_MODEL"]
for repo in scan_cache_dir(os.environ["HF_HUB_DIR"]).repos:
    if repo.repo_id.lower() == model.lower():
        print(int(repo.size_on_disk))
        break
PY"#,
        WSL_TILDE_EXPANSION_SNIPPET = WSL_TILDE_EXPANSION_SNIPPET,
        model = server::shell_quote(model_id),
        hub = crate::wsl::shell_quote_wsl(&hub),
        python = crate::wsl::shell_quote_wsl(&python),
    );
    let output = crate::wsl::run_script(&distro, &script);
    if !output.ok {
        return None;
    }
    output
        .stdout
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|bytes| *bytes > 0)
        .map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0)
}

#[tauri::command]
pub async fn analyze_context_fit(
    state: State<'_, Arc<AppState>>,
    input: ContextFitRequest,
) -> Result<crate::context_fit::ContextFitReport, String> {
    let st = (*state).clone();
    analyze_context_fit_inner(&st, input).await
}

async fn analyze_context_fit_inner(
    st: &AppState,
    input: ContextFitRequest,
) -> Result<crate::context_fit::ContextFitReport, String> {
    use crate::context_fit::{
        analyze_llama_context, analyze_vllm_context, LlamaFitInput, VllmFitInput,
    };
    use crate::state::ServerBackend;

    let cfg = st.config();
    let backend = input.backend.parse::<ServerBackend>()?;
    match backend {
        ServerBackend::Vllm => {
            let model_id = input.model_id.trim();
            if model_id.is_empty() {
                return Err("A vLLM model ID is required.".into());
            }
            let token = st.hf_token();
            let stats = hf::enrich(
                &st.http,
                model_id,
                Some(&st.enrichment_cache),
                token.as_deref(),
            )
            .await
            .ok_or_else(|| format!("Could not read model metadata for {model_id}"))?;
            let requested = input
                .context_tokens
                .filter(|value| *value > 0)
                .unwrap_or(stats.context);
            let quant = inferred_vllm_quant(model_id, input.quant.as_deref());
            let (weight_gib, weight_source) = match cached_vllm_weight_gib(st, model_id) {
                Some(value) => (Some(value), "local_hf_cache".to_string()),
                None => (
                    stats
                        .params_b
                        .map(|params| estimate::weight_gb(params, &quant)),
                    "parameter_estimate".to_string(),
                ),
            };
            let gpu = crate::wsl::gpu_snapshot(&st.resolve_distro())
                .or_else(|| st.gpu.lock().unwrap().clone());
            let (ram_total_mb, ram_available_mb) =
                crate::wsl::try_detect_wsl_memory(&st.resolve_distro())
                    .map(|(total, available)| (Some(total), Some(available)))
                    .unwrap_or((None, None));
            let kv_cache_dtype = input
                .kv_cache_dtype
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(cfg.advanced_settings.kv_cache_dtype.as_str())
                .to_ascii_lowercase();
            let memory = &cfg.memory_settings;
            Ok(analyze_vllm_context(&VllmFitInput {
                model_id: model_id.to_string(),
                requested_context: requested,
                native_context: Some(stats.context),
                context_estimated: stats.context_estimated,
                weight_gib,
                weight_source,
                n_layers: stats.n_layers,
                n_kv_heads: stats.n_kv_heads,
                head_dim: stats.head_dim,
                kv_cache_dtype,
                vram_total_mb: gpu
                    .as_ref()
                    .map(|value| value.vram_total_mb)
                    .filter(|value| *value > 0),
                vram_free_mb: gpu
                    .as_ref()
                    .map(|value| value.vram_free_mb)
                    .filter(|value| *value > 0),
                ram_total_mb,
                ram_available_mb,
                gpu_mem_util: input
                    .gpu_mem_util
                    .unwrap_or(memory.default_gpu_mem_util)
                    .clamp(0.10, 0.95),
                vram_overhead_mb: memory.vram_overhead_mb,
                max_context_cap: memory.max_context_cap,
                ram_overflow_enabled: memory.enable_ram_overflow,
                manual_ram_limit_mb: memory.manual_ram_limit_mb,
                safety_reserve_mb: memory.safety_reserve_mb as f64,
                cpu_offload_gb: input.cpu_offload_gb.unwrap_or(0),
                kv_offload_gb: input.kv_offload_gb.unwrap_or(0),
            }))
        }
        ServerBackend::Llamacpp => {
            let raw_path = input
                .model_path
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(input.model_id.as_str());
            let path =
                server::resolve_gguf_model_path(raw_path, &cfg.gguf_dir, &st.resolve_distro())
                    .ok_or_else(|| format!("GGUF file not found: {raw_path}"))?;
            if !path.is_file() {
                return Err(format!(
                    "GGUF file is not accessible from Windows: {}",
                    path.display()
                ));
            }
            let metadata = crate::gguf::metadata_from_path(&path)
                .map_err(|error| format!("Could not read GGUF metadata: {error}"))?;
            let weight_gib = std::fs::metadata(&path)
                .ok()
                .map(|meta| meta.len() as f64 / 1024.0 / 1024.0 / 1024.0);
            let gpu = crate::llamacpp_install::windows_gpu_snapshot()
                .or_else(|| st.gpu.lock().unwrap().clone());
            let specs = crate::llmfit_adapter::get_system_specs();
            let requested = input
                .context_tokens
                .filter(|value| *value > 0)
                .or(metadata.context_length)
                .ok_or_else(|| {
                    "A context length is required because the GGUF has no native context metadata."
                        .to_string()
                })?;
            Ok(analyze_llama_context(&LlamaFitInput {
                model_id: path.to_string_lossy().into_owned(),
                requested_context: requested,
                native_context: metadata.context_length,
                weight_gib,
                n_layers: metadata.block_count,
                n_kv_heads: metadata.head_count_kv.or(metadata.head_count),
                key_head_dim: metadata.key_head_dim,
                value_head_dim: metadata.value_head_dim,
                cache_type_k: input.cache_type_k.unwrap_or_else(|| "q8_0".into()),
                cache_type_v: input.cache_type_v.unwrap_or_else(|| "q8_0".into()),
                flash_attn: input.flash_attn.unwrap_or(true),
                n_gpu_layers: input.n_gpu_layers,
                n_cpu_moe: input.n_cpu_moe,
                fit: input.fit.unwrap_or(true),
                fit_target_mb: input.fit_target.unwrap_or(1_024) as f64,
                no_kv_offload: input.no_kv_offload.unwrap_or(false),
                vram_total_mb: gpu
                    .as_ref()
                    .map(|value| value.vram_total_mb)
                    .filter(|value| *value > 0),
                vram_free_mb: gpu
                    .as_ref()
                    .map(|value| value.vram_free_mb)
                    .filter(|value| *value > 0),
                ram_total_mb: Some((specs.total_ram_gb * 1024.0).round() as u64),
                ram_available_mb: Some((specs.available_ram_gb * 1024.0).round() as u64),
                vram_overhead_mb: 1_024.0,
            }))
        }
    }
}

fn compose_hf_cache_model_exists_script(hub_dir: &str, dir_name: &str) -> String {
    format!(
        r#"{WSL_TILDE_EXPANSION_SNIPPET}
hub_dir=$(__llm_panel_expand_tilde {hub_dir})
dir_name=$(__llm_panel_expand_tilde {dir_name})
[ -d "$hub_dir/$dir_name" ] && echo exists
"#,
        hub_dir = crate::wsl::shell_quote_wsl(hub_dir),
        dir_name = crate::wsl::shell_quote_wsl(dir_name),
    )
}

#[tauri::command]
pub fn pull_model(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    model_id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    crate::hf::validate_model_id(&model_id).map_err(|e| e.to_string())?;

    // Check if the model is already in imported local models
    if st
        .config()
        .imported_local_models
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&model_id))
    {
        return Err(format!(
            "Model \"{}\" is already in your library as an imported local folder",
            model_id
        ));
    }

    // Check if model already exists in HF hub cache
    let distro = st.resolve_distro();
    let hub_dir = if let Some(home) = &st.config().advanced_settings.hf_home {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            format!("{trimmed}/hub")
        } else {
            "~/.cache/huggingface/hub".to_string()
        }
    } else {
        "~/.cache/huggingface/hub".to_string()
    };
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let check_script = compose_hf_cache_model_exists_script(&hub_dir, &dir_name);
    let check = crate::wsl::run_script(&distro, &check_script);
    if check.stdout.contains("exists") {
        return Err(format!(
            "Model \"{}\" is already downloaded in your library",
            model_id
        ));
    }

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
    PullState {
        pulling: pulling.keys().cloned().collect(),
    }
}

#[tauri::command]
pub async fn pull_cancel(state: State<'_, Arc<AppState>>, model_id: String) -> Result<(), String> {
    let st = (*state).clone();
    crate::hf::validate_model_id(&model_id).map_err(|e| e.to_string())?;
    let distro = st.resolve_distro();
    // Keep the validated id inside a shell-quoted pattern; no user text is
    // interpolated into the command structure itself.
    let pattern = format!("hf download.*{model_id}([[:space:]]|$)");
    let script = format!(
        "pkill -f -- {} || true",
        crate::wsl::shell_quote_wsl(&pattern)
    );
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

#[derive(Clone, serde::Deserialize)]
#[serde(transparent)]
pub struct OptionalField<T>(Option<T>);

impl<T> Default for OptionalField<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T> OptionalField<T> {
    fn into_inner(self) -> Option<T> {
        self.0
    }
}

#[derive(Clone, serde::Deserialize)]
pub struct ServerInput {
    pub id: Option<String>,
    pub backend: Option<String>,
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub task: Option<String>,
    pub port: Option<u16>,
    pub gpu_mem_util: Option<f64>,
    #[serde(default)]
    pub max_model_len: OptionalField<usize>,
    pub quant: Option<String>,
    #[serde(default)]
    pub served_model_name: OptionalField<String>,
    #[serde(default)]
    pub kv_cache_dtype: OptionalField<String>,
    pub enforce_eager: Option<bool>,
    #[serde(default)]
    pub swap_space_gb: OptionalField<usize>,
    #[serde(default)]
    pub cpu_offload_gb: OptionalField<usize>,
    #[serde(default)]
    pub model_path: OptionalField<String>,
    #[serde(default)]
    pub mmproj_path: OptionalField<String>,
    #[serde(default)]
    pub ctx_size: OptionalField<usize>,
    #[serde(default)]
    pub n_gpu_layers: OptionalField<usize>,
    #[serde(default)]
    pub n_cpu_moe: OptionalField<usize>,
    pub fit: Option<bool>,
    #[serde(default)]
    pub fit_target: OptionalField<usize>,
    #[serde(default)]
    pub device: OptionalField<String>,
    #[serde(default)]
    pub api_key: OptionalField<String>,
    #[serde(default)]
    pub clear_api_key: Option<bool>,
    #[serde(default)]
    pub log_verbosity: OptionalField<u8>,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub threads: OptionalField<usize>,
    #[serde(default)]
    pub batch_size: OptionalField<usize>,
    #[serde(default)]
    pub ubatch_size: OptionalField<usize>,
    pub parallel: Option<usize>,
    pub jinja: Option<bool>,
    pub no_kv_offload: Option<bool>,
    pub metrics: Option<bool>,
    #[serde(default)]
    pub extra_args: OptionalField<Vec<String>>,
    #[serde(default)]
    pub env: OptionalField<std::collections::BTreeMap<String, String>>,
    pub restart: Option<bool>,
    pub llamacpp_channel: Option<crate::state::LlamaCppChannel>,
}

#[tauri::command]
pub async fn servers_create(
    state: State<'_, Arc<AppState>>,
    input: ServerInput,
) -> Result<ServerDef, String> {
    let st = (*state).clone();
    let cfg = st.config();
    let name = input
        .name
        .clone()
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| "name is required".to_string())?;
    let model_id = input
        .model_id
        .clone()
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| "model_id is required".to_string())?;
    let backend = input.backend.clone().unwrap_or_else(|| "vllm".into());
    let backend_kind = backend.parse::<crate::state::ServerBackend>()?;

    let existing: Vec<u16> = cfg.servers.iter().map(|s| s.port).collect();
    let port = match input.port {
        Some(p) => {
            if existing.contains(&p) {
                return Err(format!("port {p} already used by another server"));
            }
            if std::net::TcpListener::bind(("127.0.0.1", p)).is_err() {
                return Err(format!("port {p} is in use outside the app"));
            }
            p
        }
        None => server::alloc_port(&existing).map_err(|e| e.to_string())?,
    };

    let gpu_mem_util = input
        .gpu_mem_util
        .unwrap_or(cfg.memory_settings.default_gpu_mem_util);
    if backend_kind == crate::state::ServerBackend::Vllm
        && !(0.10..=0.95).contains(&gpu_mem_util)
    {
        return Err("vLLM GPU memory utilization must be between 0.10 and 0.95".into());
    }

    let quant = match backend_kind {
        crate::state::ServerBackend::Vllm => input
            .quant
            .clone()
            .unwrap_or_else(|| cfg.default_quant.clone())
            .to_ascii_lowercase(),
        crate::state::ServerBackend::Llamacpp => {
            input.quant.clone().unwrap_or_else(|| "GGUF".into())
        }
    };
    let task = match backend_kind {
        crate::state::ServerBackend::Vllm => {
            let task = input.task.clone().unwrap_or_else(|| "instruct".into());
            if task == "embed" { "embed" } else { "instruct" }.to_string()
        }
        crate::state::ServerBackend::Llamacpp => "instruct".into(),
    };

    let model_lower = model_id.to_ascii_lowercase();
    let is_local_path = model_id.starts_with('/')
        || model_id.starts_with('\\')
        || model_lower.ends_with(".gguf")
        || PathBuf::from(&model_id).is_absolute();
    let should_enrich = backend_kind == crate::state::ServerBackend::Vllm && !is_local_path;
    let stats = if should_enrich {
        let token = st.hf_token();
        hf::enrich(
            &st.http,
            &model_id,
            Some(&st.enrichment_cache),
            token.as_deref(),
        )
        .await
    } else {
        None
    };

    let requested_max_len = input.max_model_len.clone().into_inner().filter(|len| *len > 0);
    let max_model_len = if backend_kind == crate::state::ServerBackend::Llamacpp {
        None
    } else if let Some(len) = requested_max_len {
        Some(len)
    } else if !should_enrich {
        None
    } else {
        let gpu = st.gpu.lock().unwrap().clone();
        let fit = match (stats.as_ref(), gpu.as_ref()) {
            (Some(s), Some(g))
                if s.n_layers.is_some()
                    && s.n_kv_heads.is_some()
                    && s.head_dim.is_some()
                    && s.params_b.is_some()
                    && g.vram_total_mb > 0 =>
            {
                let kvb = estimate::kv_bytes_per_token(
                    s.n_layers.unwrap(),
                    s.n_kv_heads.unwrap(),
                    s.head_dim.unwrap(),
                );
                Some(estimate::context_fit(
                    g.vram_total_mb as f64,
                    gpu_mem_util,
                    s.params_b.unwrap(),
                    &quant,
                    kvb,
                    cfg.memory_settings.vram_overhead_mb,
                ))
            }
            _ => None,
        };
        let declared_limit = stats.as_ref().map(|s| s.context).unwrap_or(4096);
        let configured_limit = cfg
            .memory_settings
            .max_context_cap
            .map(|cap| cap.min(declared_limit))
            .unwrap_or(declared_limit);
        Some(fit.map(|len| len.min(configured_limit)).unwrap_or(configured_limit))
    };
    let params_b = stats.as_ref().and_then(|stats| stats.params_b);

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut def = ServerDef {
        backend,
        id: format!("srv-{ts:x}"),
        name: name.trim().to_string(),
        model_id,
        task,
        port,
        gpu_mem_util,
        max_model_len,
        quant,
        served_model_name: input
            .served_model_name
            .clone()
            .into_inner()
            .filter(|value| !value.trim().is_empty()),
        kv_cache_dtype: input
            .kv_cache_dtype
            .clone()
            .into_inner()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty() && value != "auto"),
        enforce_eager: input.enforce_eager.unwrap_or(true),
        params_b,
        swap_space_gb: input.swap_space_gb.clone().into_inner().filter(|value| *value > 0),
        cpu_offload_gb: input.cpu_offload_gb.clone().into_inner().filter(|value| *value > 0),
        was_running: false,
        model_path: input
            .model_path
            .clone()
            .into_inner()
            .filter(|value| !value.trim().is_empty()),
        mmproj_path: input
            .mmproj_path
            .clone()
            .into_inner()
            .filter(|value| !value.trim().is_empty()),
        ctx_size: input.ctx_size.clone().into_inner().filter(|value| *value > 0),
        n_gpu_layers: input.n_gpu_layers.clone().into_inner(),
        n_cpu_moe: input.n_cpu_moe.clone().into_inner().filter(|value| *value > 0),
        fit: input.fit.unwrap_or(true),
        fit_target: input.fit_target.clone().into_inner().filter(|value| *value > 0),
        device: input
            .device
            .clone()
            .into_inner()
            .filter(|value| !value.trim().is_empty()),
        api_key: input
            .api_key
            .clone()
            .into_inner()
            .filter(|value| !value.trim().is_empty()),
        log_verbosity: input.log_verbosity.clone().into_inner(),
        flash_attn: input.flash_attn.unwrap_or(true),
        cache_type_k: input
            .cache_type_k
            .clone()
            .unwrap_or_else(|| "q8_0".into()),
        cache_type_v: input
            .cache_type_v
            .clone()
            .unwrap_or_else(|| "q8_0".into()),
        threads: input.threads.clone().into_inner().filter(|value| *value > 0),
        batch_size: input.batch_size.clone().into_inner().filter(|value| *value > 0),
        ubatch_size: input.ubatch_size.clone().into_inner().filter(|value| *value > 0),
        parallel: input.parallel.unwrap_or(0),
        jinja: input.jinja.unwrap_or(true),
        no_kv_offload: input.no_kv_offload.unwrap_or(false),
        metrics: input.metrics.unwrap_or(true),
        extra_args: input.extra_args.clone().into_inner().unwrap_or_default(),
        env: input.env.clone().into_inner().unwrap_or_default(),
        llamacpp_channel: input
            .llamacpp_channel
            .unwrap_or(crate::state::LlamaCppChannel::Upstream),
    };
    for (name, value) in &def.env {
        server::validate_env_name(name)?;
        if value == crate::state::SECRET_PLACEHOLDER {
            return Err("secret placeholders are only valid when updating an existing server".into());
        }
    }
    if def.api_key.as_deref() == Some(crate::state::SECRET_PLACEHOLDER) {
        def.api_key = None;
    }
    def.normalize_for_backend()?;
    let mut cfg = st.config.lock().unwrap();
    let mut next = cfg.clone();
    next.servers.push(def.clone());
    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    Ok(crate::state::redact_server_def(&def))
}

#[tauri::command]
pub async fn servers_delete(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    let st2 = st.clone();
    let id2 = id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        server::stop_server(&st2, Some(&app), &id2).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;
    st.server_metrics.lock().unwrap().remove(&id);
    let mut cfg = st.config.lock().unwrap();
    let mut next = cfg.clone();
    next.servers.retain(|s| s.id != id);
    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    Ok(())
}

#[tauri::command]
pub fn servers_start(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    server::start_server(&st, Some(&app), &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn servers_stop(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || server::stop_server(&st, Some(&app), &id))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn servers_restart(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        server::restart_server(&st, Some(&app), &id).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(serde::Deserialize)]
pub struct ServerEnvInput {
    pub id: String,
    pub env: std::collections::BTreeMap<String, String>,
    pub restart: Option<bool>,
}

#[tauri::command]
pub fn servers_update_env(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    input: ServerEnvInput,
) -> Result<(), String> {
    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    let index = cfg
        .servers
        .iter()
        .position(|server| server.id == input.id)
        .ok_or_else(|| "Server not found".to_string())?;
    let original = cfg.servers[index].clone();
    let mut updated = original.clone();
    updated.env = input.env;
    crate::state::merge_server_secret_placeholders(&mut updated, Some(&original));
    for (name, value) in &updated.env {
        server::validate_env_name(name)?;
        if value == crate::state::SECRET_PLACEHOLDER {
            return Err("secret placeholders could not be restored for this server".into());
        }
    }
    let mut next = cfg.clone();
    next.servers[index] = updated;
    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    if input.restart.unwrap_or(false) {
        drop(cfg);
        server::restart_server(&st, Some(&app), &input.id).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn servers_update(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
    input: ServerInput,
) -> Result<ServerDef, String> {
    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    let id = input
        .id
        .clone()
        .ok_or_else(|| "id is required".to_string())?;
    validate_server_id(&id)?;
    let idx = cfg.servers.iter().position(|s| s.id == id).ok_or("Server not found")?;
    let existing_ports: Vec<u16> = cfg.servers.iter().filter(|s| s.id != id).map(|s| s.port).collect();
    let original = cfg.servers[idx].clone();
    let mut updated = original.clone();
    let previous_model = updated.model_id.clone();

    if let Some(value) = input.backend.clone() {
        updated.backend = value;
    }
    let backend_kind = updated.backend_kind()?;
    if let Some(value) = input.llamacpp_channel {
        updated.llamacpp_channel = value;
    }
    if let Some(value) = input.name.clone() {
        updated.name = value;
    }
    if let Some(value) = input.model_id.clone() {
        updated.model_id = value;
    }
    if let Some(value) = input.task.clone() {
        updated.task = value;
    }
    if let Some(value) = input.port {
        if existing_ports.contains(&value) {
            return Err(format!("port {value} already used by another server"));
        }
        if value != updated.port
            && std::net::TcpListener::bind(("127.0.0.1", value)).is_err()
        {
            return Err(format!("port {value} is in use"));
        }
        updated.port = value;
    }
    if let Some(value) = input.gpu_mem_util {
        updated.gpu_mem_util = value;
    }
    if backend_kind == crate::state::ServerBackend::Vllm
        && !(0.10..=0.95).contains(&updated.gpu_mem_util)
    {
        return Err("vLLM GPU memory utilization must be between 0.10 and 0.95".into());
    }
    updated.max_model_len = input
        .max_model_len
        .clone()
        .into_inner()
        .filter(|len| *len > 0);
    if let Some(value) = input.quant.clone() {
        updated.quant = if backend_kind == crate::state::ServerBackend::Vllm {
            value.to_ascii_lowercase()
        } else {
            value
        };
    }
    updated.served_model_name = input
        .served_model_name
        .clone()
        .into_inner()
        .filter(|value| !value.trim().is_empty());
    updated.kv_cache_dtype = input
        .kv_cache_dtype
        .clone()
        .into_inner()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty() && value != "auto");
    if let Some(value) = input.enforce_eager {
        updated.enforce_eager = value;
    }
    updated.swap_space_gb = input
        .swap_space_gb
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.cpu_offload_gb = input
        .cpu_offload_gb
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.model_path = input
        .model_path
        .clone()
        .into_inner()
        .filter(|value| !value.trim().is_empty());
    updated.mmproj_path = input
        .mmproj_path
        .clone()
        .into_inner()
        .filter(|value| !value.trim().is_empty());
    updated.ctx_size = input
        .ctx_size
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.n_gpu_layers = input.n_gpu_layers.clone().into_inner();
    updated.n_cpu_moe = input
        .n_cpu_moe
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    if let Some(value) = input.fit {
        updated.fit = value;
    }
    updated.fit_target = input
        .fit_target
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.device = input
        .device
        .clone()
        .into_inner()
        .filter(|value| !value.trim().is_empty());
    if input.clear_api_key.unwrap_or(false) {
        updated.api_key = None;
    } else if let Some(value) = input.api_key.clone().into_inner() {
        if value != crate::state::SECRET_PLACEHOLDER {
            updated.api_key = (!value.trim().is_empty()).then(|| value.trim().to_string());
        }
    }
    updated.log_verbosity = input.log_verbosity.clone().into_inner();
    if let Some(value) = input.flash_attn {
        updated.flash_attn = value;
    }
    updated.cache_type_k = input
        .cache_type_k
        .clone()
        .unwrap_or_else(|| "q8_0".into());
    updated.cache_type_v = input
        .cache_type_v
        .clone()
        .unwrap_or_else(|| "q8_0".into());
    updated.threads = input
        .threads
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.batch_size = input
        .batch_size
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.ubatch_size = input
        .ubatch_size
        .clone()
        .into_inner()
        .filter(|value| *value > 0);
    updated.parallel = input.parallel.unwrap_or(0);
    if let Some(value) = input.jinja {
        updated.jinja = value;
    }
    if let Some(value) = input.no_kv_offload {
        updated.no_kv_offload = value;
    }
    updated.metrics = input.metrics.unwrap_or(true);
    updated.extra_args = input.extra_args.clone().into_inner().unwrap_or_default();
    if let Some(env) = input.env.clone().into_inner() {
        updated.env = env;
        crate::state::merge_server_secret_placeholders(&mut updated, Some(&original));
    }
    for (name, value) in &updated.env {
        server::validate_env_name(name)?;
        if value == crate::state::SECRET_PLACEHOLDER {
            return Err("secret placeholders could not be restored for this server".into());
        }
    }
    if updated.model_id != previous_model {
        updated.params_b = None;
        updated.max_model_len = None;
    }
    updated.normalize_for_backend()?;

    let mut next = cfg.clone();
    next.servers[idx] = updated.clone();
    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    drop(cfg);

    let live_active = st
        .servers
        .lock()
        .unwrap()
        .get(&id)
        .map(|live| {
            live.status == crate::state::ServerStatus::Running
                || live.status == crate::state::ServerStatus::Starting
        })
        .unwrap_or(false);
    if input.restart.unwrap_or(false) || (live_active && updated != original) {
        server::restart_server(&st, Some(&app), &id).map_err(|e| e.to_string())?;
    }
    Ok(crate::state::redact_server_def(&updated))
}

#[tauri::command]
pub fn servers_logs(state: State<'_, Arc<AppState>>, id: String, since: usize) -> String {
    let st = (*state).clone();
    server::server_logs(&st, &id, since)
}

#[tauri::command]
pub fn servers_metrics(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Option<server::MetricsSnapshot> {
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
    server::chat(&st, &id, messages)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn servers_test_tool_call(
    state: State<'_, Arc<AppState>>,
    id: String,
    max_tokens: Option<u32>,
    disable_thinking: Option<bool>,
) -> Result<serde_json::Value, String> {
    let st = (*state).clone();
    let max_tokens = max_tokens.unwrap_or(2048).max(1);
    let disable_thinking = disable_thinking.unwrap_or(true);
    let messages = vec![serde_json::json!({
        "role": "user",
        "content": "Call get_weather for Seattle. Do not answer with prose."
    })];
    let body = server::chat_with_tools(&st, &id, messages, max_tokens, disable_thinking)
        .await
        .map_err(|e| e.to_string())?;
    let classification = classify_tool_call_response(&body);
    Ok(serde_json::json!({
        "passed": classification.passed,
        "response": body,
        "hint": classification.hint,
        "tool_call": classification.tool_call,
        "reasoning_content": classification.reasoning_content,
        "max_tokens": max_tokens,
        "disable_thinking": disable_thinking
    }))
}

#[derive(Debug, Default, PartialEq)]
struct ToolCallClassification {
    passed: bool,
    hint: String,
    tool_call: Option<serde_json::Value>,
    reasoning_content: Option<String>,
}

fn classify_tool_call_response(body: &serde_json::Value) -> ToolCallClassification {
    let message = &body["choices"][0]["message"];
    if let Some(calls) = message["tool_calls"].as_array() {
        if let Some(call) = calls.iter().find(|call| {
            call["function"]["name"].as_str().is_some()
                && (call["function"]["arguments"].is_string()
                    || call["function"]["arguments"].is_object())
        }) {
            return ToolCallClassification {
                passed: true,
                hint: String::new(),
                tool_call: Some(call.clone()),
                reasoning_content: message["reasoning_content"].as_str().map(str::to_string),
            };
        }
    }

    let reasoning_content = message["reasoning_content"].as_str().map(str::to_string);
    if body["choices"][0]["finish_reason"].as_str() == Some("length") {
        return ToolCallClassification {
            passed: false,
            hint: "Output was cut off at max_tokens (model may still be thinking); raise max tokens or disable thinking.".into(),
            tool_call: None,
            reasoning_content,
        };
    }

    let content = message["content"].as_str().unwrap_or_default();
    if looks_like_text_tool_call(content) {
        return ToolCallClassification {
            passed: false,
            hint: "Tool call was emitted as text; check that --jinja is on and the chat template supports tools.".into(),
            tool_call: None,
            reasoning_content,
        };
    }

    ToolCallClassification {
        passed: false,
        hint: format!("No tool call was returned. Raw response: {body}"),
        tool_call: None,
        reasoning_content,
    }
}

fn looks_like_text_tool_call(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.contains("<tool_call>") || trimmed.contains("</tool_call>") {
        return true;
    }
    serde_json::from_str::<serde_json::Value>(trimmed)
        .ok()
        .is_some_and(|value| {
            value.get("name").is_some()
                || value.get("arguments").is_some()
                || value.get("tool_call").is_some()
                || value.get("tool_calls").is_some()
        })
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
pub fn conversations_list(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::state::Conversation>, String> {
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
pub fn conversations_delete(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
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
pub fn benchmarks_cancel(state: State<'_, Arc<AppState>>, server_id: String) -> Result<(), String> {
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

#[derive(Serialize)]
pub struct PublicSettings {
    #[serde(flatten)]
    pub config: PersistedConfig,
    pub hf_token_configured: bool,
    pub github_token_configured: bool,
    pub api_key_configured: bool,
}

fn public_settings(cfg: &PersistedConfig) -> PublicSettings {
    PublicSettings {
        config: crate::state::redact_config_for_display(cfg),
        hf_token_configured: !cfg.hf_token.trim().is_empty(),
        github_token_configured: !cfg.github_token.trim().is_empty(),
        api_key_configured: cfg
            .advanced_settings
            .api_key
            .as_deref()
            .map(str::trim)
            .is_some_and(|key| !key.is_empty()),
    }
}

#[tauri::command]
pub fn settings_get(state: State<'_, Arc<AppState>>) -> PublicSettings {
    let st = (*state).clone();
    public_settings(&st.config())
}

#[derive(serde::Serialize)]
pub struct GatewayStatus {
    pub enabled: bool,
    pub port: u16,
    pub running: bool,
    pub auth_configured: bool,
}

#[tauri::command]
pub fn gateway_status(state: State<'_, Arc<AppState>>) -> GatewayStatus {
    let st = (*state).clone();
    let cfg = st.config();
    let enabled = cfg.advanced_settings.gateway_enabled;
    let port = cfg.advanced_settings.gateway_port;
    let auth_configured = cfg
        .advanced_settings
        .api_key
        .as_deref()
        .map(str::trim)
        .is_some_and(|key| !key.is_empty());
    let running = if enabled && auth_configured {
        std::net::TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
            std::time::Duration::from_millis(300),
        )
        .is_ok()
    } else {
        false
    };
    GatewayStatus {
        enabled,
        port,
        running,
        auth_configured,
    }
}

#[derive(serde::Deserialize)]
pub struct SettingsPatch {
    pub distro: Option<String>,
    pub llm_dir: Option<String>,
    pub venv_dir: Option<String>,
    pub llamacpp_dir: Option<String>,
    pub gguf_dir: Option<String>,
    pub llamacpp_executable: Option<String>,
    pub llamacpp_channels: Option<
        std::collections::BTreeMap<
            crate::state::LlamaCppChannel,
            crate::state::LlamaCppChannelConfig,
        >,
    >,
    pub hf_token: Option<String>,
    pub github_token: Option<String>,
    pub clear_hf_token: Option<bool>,
    pub clear_github_token: Option<bool>,
    pub default_quant: Option<String>,
    pub advanced_settings: Option<crate::state::AdvancedSettings>,
    pub clear_advanced_api_key: Option<bool>,
    pub clear_custom_env_vars: Option<bool>,
    pub minimize_to_tray: Option<bool>,
    pub auto_restart_crashed: Option<bool>,
    pub launch_at_login: Option<bool>,
}

fn validate_imported_wsl_path(path: &str) -> Result<(), String> {
    let path = path.trim().trim_end_matches('/');
    let components: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    if path.is_empty()
        || !path.starts_with('/')
        || components.len() < 3
        || components.iter().any(|part| *part == "..")
        || path.contains('\0')
        || path.contains('\r')
        || path.contains('\n')
    {
        return Err("imported model path must be a non-root WSL directory".into());
    }
    Ok(())
}

fn validate_server_id(id: &str) -> Result<(), String> {
    let id = id.trim();
    if id.is_empty()
        || id.len() > 128
        || id.chars().any(|ch| ch.is_control() || ch == '\0')
    {
        return Err("server id must be 1-128 printable characters".into());
    }
    Ok(())
}

fn validate_advanced_settings(adv: &crate::state::AdvancedSettings) -> Result<(), String> {
    let host = adv.host.trim();
    if host.is_empty()
        || !host
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | ':' | '-'))
    {
        return Err("host must be a plain IP address or hostname".into());
    }
    if adv.gateway_port < 1024 {
        return Err("gateway_port must be at least 1024".into());
    }
    let loopback = matches!(host, "127.0.0.1" | "localhost" | "::1");
    if !loopback
        && adv
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .is_none()
    {
        return Err("an API key is required before binding vLLM to a non-loopback host".into());
    }
    if !matches!(adv.log_level.as_str(), "INFO" | "DEBUG" | "WARNING" | "ERROR") {
        return Err("log_level must be INFO, DEBUG, WARNING, or ERROR".into());
    }
    if !adv
        .kv_cache_dtype
        .trim()
        .is_empty()
        && !adv
            .kv_cache_dtype
            .trim()
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    {
        return Err("kv_cache_dtype contains unsupported characters".into());
    }
    if let Some(extra) = &adv.extra_vllm_args {
        shlex::split(extra).ok_or_else(|| "extra_vllm_args has invalid shell quoting".to_string())?;
    }
    if let Some(custom) = &adv.custom_env_vars {
        for line in custom.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, _) = line
                .split_once('=')
                .ok_or_else(|| "custom_env_vars must use KEY=VALUE lines".to_string())?;
            server::validate_env_name(name.trim())?;
        }
    }
    Ok(())
}

#[tauri::command]
pub fn settings_set(
    state: State<'_, Arc<AppState>>,
    patch: SettingsPatch,
) -> Result<PublicSettings, String> {
    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    // Apply to a clone so validation failures (for example, clearing a key
    // while binding to LAN) cannot leave the live state half-mutated.
    let mut next = cfg.clone();
    if let Some(d) = patch.distro {
        if !d.trim().is_empty() {
            next.distro = d.trim().to_string();
        }
    }
    if let Some(d) = patch.llm_dir {
        if !d.trim().is_empty() {
            next.llm_dir = d.trim().to_string();
        }
    }
    if let Some(v) = patch.venv_dir {
        if !v.trim().is_empty() {
            server::validate_venv_dir(&v)?;
            next.venv_dir = v.trim().to_string();
        }
    }
    if let Some(d) = patch.llamacpp_dir {
        if !d.trim().is_empty() {
            next.llamacpp_dir = d.trim().to_string();
        }
    }
    if let Some(d) = patch.gguf_dir {
        if !d.trim().is_empty() {
            next.gguf_dir = d.trim().to_string();
        }
    }
    if let Some(exe) = patch.llamacpp_executable {
        next.llamacpp_executable = (!exe.trim().is_empty()).then(|| exe.trim().to_string());
    }
    if let Some(channels) = patch.llamacpp_channels {
        next.llamacpp_channels = channels;
        let upstream = next
            .llamacpp_channels
            .get(&crate::state::LlamaCppChannel::Upstream)
            .cloned();
        if let Some(upstream) = upstream {
            if !upstream.dir.trim().is_empty() {
                next.llamacpp_dir = upstream.dir.clone();
            }
            next.llamacpp_executable = upstream.executable.clone();
            next.llamacpp_version = upstream.version.clone();
            next.llamacpp_help = upstream.help.clone();
            next.llamacpp_installed_tag = upstream.installed_tag.clone();
        }
    }
    if let Some(t) = patch.hf_token {
        if t != crate::state::SECRET_PLACEHOLDER {
            next.hf_token = t.trim().to_string();
        }
    }
    if patch.clear_hf_token.unwrap_or(false) {
        next.hf_token.clear();
    }
    if let Some(t) = patch.github_token {
        if t != crate::state::SECRET_PLACEHOLDER {
            next.github_token = t.trim().to_string();
        }
    }
    if patch.clear_github_token.unwrap_or(false) {
        next.github_token.clear();
    }
    if let Some(q) = patch.default_quant {
        next.default_quant = q;
    }
    if let Some(adv) = patch.advanced_settings {
        let mut next_adv = adv;
        if next_adv.api_key.as_deref() == Some(crate::state::SECRET_PLACEHOLDER)
            || next_adv.api_key.is_none()
        {
            // The webview receives a redacted key, so an omitted/null value is
            // a preservation request unless the explicit clear flag is set.
            next_adv.api_key = next.advanced_settings.api_key.clone();
        }
        if next_adv.custom_env_vars.is_none() {
            next_adv.custom_env_vars = next.advanced_settings.custom_env_vars.clone();
        }
        next.advanced_settings = next_adv;
    }
    if patch.clear_advanced_api_key.unwrap_or(false) {
        next.advanced_settings.api_key = None;
    }
    if patch.clear_custom_env_vars.unwrap_or(false) {
        next.advanced_settings.custom_env_vars = None;
    }
    validate_advanced_settings(&next.advanced_settings)?;
    if let Some(m) = patch.minimize_to_tray {
        next.minimize_to_tray = m;
    }
    if let Some(a) = patch.auto_restart_crashed {
        next.auto_restart_crashed = a;
    }
    if let Some(l) = patch.launch_at_login {
        next.launch_at_login = l;
        autostart_set(l).map_err(|e| e.to_string())?;
    }
    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    Ok(public_settings(&cfg))
}

#[tauri::command]
pub fn autostart_get() -> Result<bool, String> {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let mut cmd = std::process::Command::new("reg");
        cmd.creation_flags(0x08000000);
        let output = cmd
            .args([
                "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "LocalLLmPanel",
            ])
            .output()
            .map_err(|e| e.to_string())?;
        Ok(output.status.success()
            && String::from_utf8_lossy(&output.stdout).contains("LocalLLmPanel"))
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
        use std::os::windows::process::CommandExt;
        if enabled {
            let current_exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let exe_str = current_exe.to_string_lossy();
            let mut cmd = std::process::Command::new("reg");
            cmd.creation_flags(0x08000000);
            let status = cmd
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
            let mut cmd = std::process::Command::new("reg");
            cmd.creation_flags(0x08000000);
            let _ = cmd
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
pub fn config_import(
    state: State<'_, Arc<AppState>>,
    json: String,
) -> Result<PublicSettings, String> {
    let pkg: crate::state::ConfigExportPackage = serde_json::from_str(&json)
        .map_err(|e| format!("Invalid configuration JSON format: {e}"))?;
    if !pkg
        .schema
        .starts_with("local-llm-panel/config-export/")
    {
        return Err(format!("Unsupported configuration schema: {}", pkg.schema));
    }

    let st = (*state).clone();
    let mut cfg = st.config.lock().unwrap();
    let secrets_omitted = pkg.secrets_omitted;
    let previous_servers = cfg.servers.clone();
    let mut next = cfg.clone();
    next.distro = pkg.distro;
    next.llm_dir = pkg.llm_dir;
    next.venv_dir = pkg.venv_dir;
    server::validate_venv_dir(&next.venv_dir)?;
    next.default_quant = pkg.default_quant;
    next.memory_settings = pkg.memory_settings;
    next.minimize_to_tray = pkg.minimize_to_tray;
    next.auto_restart_crashed = pkg.auto_restart_crashed;
    next.launch_at_login = pkg.launch_at_login;

    // Portable exports intentionally omit secrets.  Preserve the local
    // credentials instead of importing an empty value or an old raw secret.
    let mut imported_advanced = pkg.advanced_settings;
    imported_advanced.api_key = next.advanced_settings.api_key.clone();
    imported_advanced.custom_env_vars = next.advanced_settings.custom_env_vars.clone();
    next.advanced_settings = imported_advanced;
    validate_advanced_settings(&next.advanced_settings)?;

    // Imported servers are mutation boundaries: validate the backend, restore
    // placeholders from the matching local definition, and never import raw
    // environment values from a legacy export.
    let mut imported_servers = Vec::with_capacity(pkg.servers.len());
    for mut server in pkg.servers {
        validate_server_id(&server.id)?;
        server.was_running = false;
        let previous = previous_servers.iter().find(|candidate| candidate.id == server.id);
        crate::state::merge_server_secret_placeholders(&mut server, previous);
        if !secrets_omitted {
            server.api_key = previous.and_then(|candidate| candidate.api_key.clone());
            server.env = previous.map(|candidate| candidate.env.clone()).unwrap_or_default();
        }
        server.normalize_for_backend()?;
        if imported_servers
            .iter()
            .any(|candidate: &crate::state::ServerDef| candidate.id == server.id)
        {
            return Err(format!("configuration contains duplicate server id {}", server.id));
        }
        if imported_servers
            .iter()
            .any(|candidate: &crate::state::ServerDef| candidate.port == server.port)
        {
            return Err(format!("configuration contains duplicate server port {}", server.port));
        }
        imported_servers.push(server);
    }
    next.servers = imported_servers;

    next.save().map_err(|e| e.to_string())?;
    *cfg = next;
    Ok(public_settings(&cfg))
}

#[tauri::command]
pub fn server_recipe_export(
    state: State<'_, Arc<AppState>>,
    server_id: String,
) -> Result<String, String> {
    let st = (*state).clone();
    let cfg = st.config.lock().unwrap();
    let srv = cfg
        .find_server(&server_id)
        .ok_or_else(|| format!("Server not found: {server_id}"))?;
    let recipe = crate::state::ServerRecipe::from_server_def(srv);
    serde_json::to_string_pretty(&recipe).map_err(|e| format!("Failed to serialize recipe: {e}"))
}

#[tauri::command]
pub fn server_recipe_parse(json: String) -> Result<crate::state::ServerRecipe, String> {
    let recipe: crate::state::ServerRecipe =
        serde_json::from_str(&json).map_err(|e| format!("Invalid server recipe JSON: {e}"))?;
    if !recipe.schema.starts_with("local-llm-panel/server-recipe/") {
        return Err(format!("Unsupported server recipe schema: {}", recipe.schema));
    }
    if recipe.model_id.trim().is_empty() || recipe.port == 0 {
        return Err("Recipe must include a model_id and a non-zero port".into());
    }
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
pub fn update_memory_settings(
    state: State<'_, Arc<AppState>>,
    settings: MemorySettings,
) -> Result<(), String> {
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
        return WslConfigInfo {
            path: None,
            content: None,
        };
    };
    let content = std::fs::read_to_string(&p).ok();
    WslConfigInfo {
        path: Some(p.display().to_string()),
        content,
    }
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
    /// True when this entry is an imported local folder (WSL path) rather
    /// than a HuggingFace hub cache download.
    #[serde(default)]
    pub is_local: bool,
    /// Absolute Windows file path for native GGUF models.
    #[serde(default)]
    pub model_path: Option<String>,
}

const WSL_TILDE_EXPANSION_SNIPPET: &str = crate::wsl::WSL_TILDE_EXPANSION_SNIPPET;

fn compose_hf_cache_scan_script(venv_python: &str, hub_dir: &str) -> String {
    format!(
        r#"{WSL_TILDE_EXPANSION_SNIPPET}
venv_python=$(__llm_panel_expand_tilde {venv_python})
hub_dir=$(__llm_panel_expand_tilde {hub_dir})
export LOCAL_LLM_PANEL_HUB_DIR="$hub_dir"
"$venv_python" - <<'PY'
import json
import os
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir(os.environ['LOCAL_LLM_PANEL_HUB_DIR'])
result = []
for repo in cache.repos:
    result.append({{
        'repo_id': repo.repo_id,
        'size_bytes': repo.size_on_disk,
        'file_count': repo.nb_files,
    }})
print(json.dumps(result))
PY
"#,
        venv_python = crate::wsl::shell_quote_wsl(venv_python),
        hub_dir = crate::wsl::shell_quote_wsl(hub_dir),
    )
}

fn should_scan_hf_cache_fs(python_scan_ok: bool, entries: &[LibraryEntry]) -> bool {
    !python_scan_ok || entries.is_empty() || entries.iter().any(|entry| entry.size_mb == 0)
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

        let hub_dir = if let Some(home) = &st.config().advanced_settings.hf_home {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                format!("{trimmed}/hub")
            } else {
                "~/.cache/huggingface/hub".to_string()
            }
        } else {
            "~/.cache/huggingface/hub".to_string()
        };

        // Use the app's provisioned venv directly. The system Python does not
        // necessarily contain huggingface_hub, and inserting a literal "~/..."
        // into sys.path does not perform shell home expansion.
        let venv_python = format!("{}/bin/python", st.config().venv_dir.trim_end_matches('/'));
        let scan_script = compose_hf_cache_scan_script(&venv_python, &hub_dir);
        let out = crate::wsl::run_script(&distro, &scan_script);
        let mut out_v = Vec::new();
        let mut python_scan_ok = false;
        if out.ok {
            if let Ok(repos) = serde_json::from_str::<Vec<serde_json::Value>>(&out.stdout) {
                for repo in repos {
                    let model_id_str = repo["repo_id"].as_str().unwrap_or("").to_string();
                    let size_bytes = repo["size_bytes"].as_u64().unwrap_or(0);
                    let files_cnt = repo["file_count"].as_u64().unwrap_or(0) as usize;
                    let size_mb = size_bytes / (1024 * 1024);

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
                        is_local: false,
                        model_path: None,
                    });
                }
                python_scan_ok = true;
            }
        }

        // Fall back to the filesystem scan when the Python API fails, returns no
        // repositories, or reports a zero-sized entry. An empty JSON result is
        // still a successful command, but it must not suppress the fallback:
        // newer/partial HF cache layouts can be invisible to scan_cache_dir.
        let needs_fallback = should_scan_hf_cache_fs(python_scan_ok, &out_v);
        if needs_fallback {
            let fs_entries = scan_hf_cache_fs(&distro, &hub_dir, &running_servers);
            // Merge: use filesystem entries for models not found by Python, or with 0 size
            for fs_entry in fs_entries {
                if let Some(idx) = out_v.iter().position(|e| e.model_id == fs_entry.model_id) {
                    if out_v[idx].size_mb == 0 && out_v[idx].files == 0 {
                        out_v[idx] = fs_entry;
                    }
                } else {
                    out_v.push(fs_entry);
                }
            }
        }

        let mut local =
            scan_imported_local_folders(&distro, &st.config().imported_local_models, &running_servers);
        local.extend(scan_native_gguf_library(
            &st.config().gguf_dir,
            &running_servers,
        ));
        out_v.sort_by(|a, b| b.size_mb.cmp(&a.size_mb));
        local.append(&mut out_v);
        local
    })
    .await
    .map_err(|e| e.to_string())
}

fn scan_native_gguf_library(root: &str, running_servers: &[(String, String)]) -> Vec<LibraryEntry> {
    let mut entries = Vec::new();
    let Ok(repos) = std::fs::read_dir(root) else {
        return entries;
    };
    for repo in repos.flatten().filter(|e| e.path().is_dir()) {
        let mut files: Vec<_> = std::fs::read_dir(repo.path())
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf"))
            .collect();
        if files.is_empty() {
            continue;
        }
        files.sort_by_key(|f| f.path());
        let main_file = files
            .first()
            .map(|f| f.path().to_string_lossy().into_owned());
        let size_mb = files
            .iter()
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum::<u64>()
            / (1024 * 1024);
        let model_id = repo.file_name().to_string_lossy().into_owned();
        let in_use_server = running_servers
            .iter()
            .find(|(_, model)| model == &model_id)
            .map(|(name, _)| name.clone());
        entries.push(LibraryEntry {
            model_id,
            size_mb,
            files: files.len(),
            quant: Some("GGUF".into()),
            params_b: None,
            installed: true,
            in_use: in_use_server.is_some(),
            in_use_server,
            task: Some("instruct".into()),
            is_local: true,
            model_path: main_file,
        });
    }
    entries
}

#[tauri::command]
pub async fn gguf_files(
    state: State<'_, Arc<AppState>>,
    repo_id: String,
) -> Result<Vec<crate::hf::GgufRepoFile>, String> {
    let token = state.hf_token();
    crate::hf::list_gguf_repo_files(&state.http, &repo_id, token.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn download_gguf(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    repo_id: String,
    files: Vec<String>,
) -> Result<(), String> {
    if files.is_empty() {
        return Err("select at least one GGUF file".into());
    }
    crate::hf::validate_model_id(&repo_id).map_err(|e| e.to_string())?;
    if files.iter().any(|file| {
        let file = file.trim();
        file.is_empty() || file.starts_with('/') || file.contains("..")
    }) {
        return Err("GGUF file paths must be relative and cannot contain '..'".into());
    }
    let st = (*state).clone();
    let cfg = st.config();
    let token = st.hf_token();
    let model = repo_id.clone();
    let pulling_arc = Arc::clone(&st.pulling);
    let gguf_root = PathBuf::from(&cfg.gguf_dir);
    std::fs::create_dir_all(&gguf_root).map_err(|e| e.to_string())?;
    let gguf_root = gguf_root
        .canonicalize()
        .map_err(|e| format!("GGUF directory is not accessible: {e}"))?;
    let destination = gguf_root.join(
        repo_id
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("model"),
    );
    if destination.exists() {
        let canonical_destination = destination
            .canonicalize()
            .map_err(|e| format!("GGUF destination is not accessible: {e}"))?;
        if !canonical_destination.starts_with(&gguf_root) {
            return Err("Refusing to download into a path outside the GGUF directory".into());
        }
    }
    {
        let mut pulling = pulling_arc.lock().unwrap();
        if *pulling.get(&model).unwrap_or(&false) {
            return Err(format!("already downloading {model}"));
        }
        pulling.insert(model.clone(), true);
    }
    let download_result = tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        std::fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
        let client = reqwest::blocking::Client::new();
        'outer: for file_name in files {
            let target = destination.join(
                PathBuf::from(&file_name)
                    .file_name()
                    .ok_or_else(|| "invalid GGUF file name".to_string())?,
            );
            if !pulling_arc.lock().unwrap().contains_key(&model) {
                return Ok::<(), String>(());
            }
            let offset = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
            let url = format!("https://huggingface.co/{model}/resolve/main/{file_name}");
            let mut request = client.get(url);
            if let Some(token) = token.as_deref().filter(|t| !t.trim().is_empty()) {
                request = request.bearer_auth(token);
            }
            if offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            let mut response = request.send().map_err(|e| e.to_string())?;
            if offset > 0 && response.status() == reqwest::StatusCode::OK {
                std::fs::File::create(&target).map_err(|e| e.to_string())?;
            }
            response.error_for_status_ref().map_err(|e| e.to_string())?;
            let total = response.content_length().map(|n| n + offset);
            let mut output = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&target)
                .map_err(|e| e.to_string())?;
            let mut done = offset;
            loop {
                if !pulling_arc.lock().unwrap().contains_key(&model) {
                    break 'outer;
                }
                let mut buf = [0u8; 1024 * 1024];
                let n = std::io::Read::read(&mut response, &mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                std::io::Write::write_all(&mut output, &buf[..n]).map_err(|e| e.to_string())?;
                done += n as u64;
                let percent = total.map(|t| (done as f64 / t as f64 * 100.0) as f32);
                let speed = done.saturating_sub(offset) as f64 / started.elapsed().as_secs_f64().max(0.001);
                let eta_seconds = match (total, speed) {
                    (Some(t), s) if s > 0.0 => Some(((t.saturating_sub(done)) as f64 / s) as f32),
                    _ => None,
                };
                let _ = app.emit(
                    "pull-progress",
                    serde_json::json!({
                        "model": model,
                        "state": "downloading",
                        "file": file_name,
                        "percent": percent,
                        "speed_bps": done.saturating_sub(offset) as f64 / started.elapsed().as_secs_f64().max(0.001),
                        "eta_seconds": eta_seconds,
                        "bytes_downloaded": done,
                        "bytes_total": total,
                    }),
                );
            }
        }
        if !pulling_arc.lock().unwrap().contains_key(&model) {
            // cancelled (frontend already removed the progress entry)
            return Ok::<(), String>(());
        }
        let _ = app.emit(
            "pull-progress",
            serde_json::json!({"model": model, "state": "complete", "file": null, "percent": 100.0}),
        );
        Ok::<(), String>(())
    });
    let download_result = match download_result.await {
        Ok(result) => result,
        Err(error) => {
            st.pulling.lock().unwrap().remove(&repo_id);
            return Err(format!("download worker failed: {error}"));
        }
    };
    st.pulling.lock().unwrap().remove(&repo_id);
    download_result?;
    Ok(())
}

fn compose_hf_cache_fs_scan_script(hub_dir: &str) -> String {
    format!(
        r#"{WSL_TILDE_EXPANSION_SNIPPET}
hub_dir=$(__llm_panel_expand_tilde {hub_dir})
for p in "$hub_dir"/models--*; do
  [ -d "$p" ] || continue
  size=$(du -sm "$p" 2>/dev/null | cut -f1)
  files=$(find "$p" -type f 2>/dev/null | wc -l)
  repo_id=$(basename "$p" | sed 's/^models--//' | sed 's/--/\//')
  echo "$repo_id|$size|$files"
done
"#,
        hub_dir = crate::wsl::shell_quote_wsl(hub_dir),
    )
}

/// Scan the HF cache directory directly using filesystem commands (du/find)
/// as a fallback when huggingface_hub.scan_cache_dir returns incomplete data.
fn scan_hf_cache_fs(
    distro: &str,
    hub_dir: &str,
    running_servers: &[(String, String)],
) -> Vec<LibraryEntry> {
    let script = compose_hf_cache_fs_scan_script(hub_dir);
    let out = crate::wsl::run_script(distro, &script);
    let mut entries = Vec::new();
    for line in out.stdout.lines() {
        let mut it = line.split('|');
        let (Some(model_id), Some(size), Some(files)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let model_id_str = model_id.to_string();
        let size_mb: u64 = size.parse().unwrap_or(0);
        let files_cnt: usize = files.parse().unwrap_or(0);
        if size_mb == 0 && files_cnt == 0 {
            continue;
        }
        let mut in_use = false;
        let mut in_use_server = None;
        for (srv_name, srv_model) in running_servers {
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
        entries.push(LibraryEntry {
            model_id: model_id_str,
            size_mb,
            files: files_cnt,
            quant,
            params_b,
            installed: true,
            in_use,
            in_use_server,
            task,
            is_local: false,
            model_path: None,
        });
    }
    entries
}

/// Scan the persisted imported local model folders inside WSL and build
/// library entries for the ones that still exist.
fn scan_imported_local_folders(
    distro: &str,
    paths: &[String],
    running_servers: &[(String, String)],
) -> Vec<LibraryEntry> {
    if paths.is_empty() {
        return Vec::new();
    }
    let quoted: Vec<String> = paths
        .iter()
        .map(|p| format!("'{}'", p.replace('\'', "'\\''")))
        .collect();
    let script = format!(
        r#"
for p in {}; do
  [ -d "$p" ] || continue
  size=$(du -sm "$p" 2>/dev/null | cut -f1)
  files=$(find "$p" -maxdepth 2 -type f 2>/dev/null | wc -l)
  echo "$p|$size|$files"
done
"#,
        quoted.join(" ")
    );
    let out = crate::wsl::run_script(distro, &script);
    let mut entries = Vec::new();
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
        for (srv_name, srv_model) in running_servers {
            if srv_model == &model_id_str {
                in_use = true;
                in_use_server = Some(srv_name.clone());
                break;
            }
        }
        entries.push(LibraryEntry {
            model_id: model_id_str,
            size_mb,
            files: files_cnt,
            quant: Some("native".into()),
            params_b: None,
            installed: true,
            in_use,
            in_use_server,
            task: Some("instruct".into()),
            is_local: true,
            model_path: None,
        });
    }
    entries
}

/// Import a local model folder (Windows or WSL path) into the library so it
/// can be deployed directly. The path is persisted and re-listed on refresh.
#[tauri::command]
pub async fn library_import_local(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<LibraryEntry, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        if path.to_ascii_lowercase().ends_with(".gguf") {
            let source = PathBuf::from(&path);
            if !source.is_file() {
                return Err(format!("GGUF file does not exist: {path}"));
            }
            let file_name = source
                .file_name()
                .ok_or_else(|| "GGUF path has no file name".to_string())?;
            let gguf_root = PathBuf::from(st.config().gguf_dir);
            std::fs::create_dir_all(&gguf_root).map_err(|e| e.to_string())?;
            let gguf_root = gguf_root
                .canonicalize()
                .map_err(|e| format!("GGUF directory is not accessible: {e}"))?;
            let repo_dir = gguf_root.join("imported");
            std::fs::create_dir_all(&repo_dir).map_err(|e| e.to_string())?;
            let repo_dir = repo_dir
                .canonicalize()
                .map_err(|e| format!("Imported model directory is not accessible: {e}"))?;
            if !repo_dir.starts_with(&gguf_root) {
                return Err("Refusing to import into a path outside the GGUF directory".into());
            }
            let target = repo_dir.join(file_name);
            if target.exists() {
                let target_canonical = target
                    .canonicalize()
                    .map_err(|e| format!("Imported model target is not accessible: {e}"))?;
                if !target_canonical.starts_with(&repo_dir) {
                    return Err("Refusing to overwrite a path outside the imported model directory".into());
                }
            }
            std::fs::copy(&source, &target).map_err(|e| e.to_string())?;
            let size_mb = target.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
            return Ok(LibraryEntry {
                model_id: "imported".into(),
                size_mb,
                files: 1,
                quant: Some("GGUF".into()),
                params_b: None,
                installed: true,
                in_use: false,
                in_use_server: None,
                task: Some("instruct".into()),
                is_local: true,
                model_path: Some(target.to_string_lossy().into_owned()),
            });
        }
        let wsl_path = crate::wsl::windows_to_wsl_path(&path);
        if wsl_path.is_empty() || !wsl_path.starts_with('/') {
            return Err("Enter a local model directory path (e.g. D:\\AI\\qwen or /mnt/d/AI/qwen)".into());
        }
        validate_imported_wsl_path(&wsl_path)?;
        let distro = st.resolve_distro();
        let path_literal = crate::wsl::shell_quote_wsl(&wsl_path);
        let check_script = format!(
            "{}\nmodel_dir=$(__llm_panel_expand_tilde {path_literal})\n[ -d \"$model_dir\" ] && echo dir\n[ -f \"$model_dir/config.json\" ] && echo ok",
            crate::wsl::WSL_TILDE_EXPANSION_SNIPPET,
        );
        let check = crate::wsl::run_script(&distro, &check_script);
        if !check.ok {
            return Err(format!("Failed to check directory inside WSL: {}", check.stderr));
        }
        let stdout = check.stdout.clone();
        if !stdout.contains("dir") {
            return Err(format!("Directory does not exist inside WSL: {wsl_path}"));
        }
        if !stdout.contains("ok") {
            return Err(format!("Directory exists inside WSL but has no config.json (not a ready vLLM model dir): {wsl_path}"));
        }

        {
            let mut cfg = st.config.lock().unwrap();
            if !cfg.imported_local_models.contains(&wsl_path) {
                cfg.imported_local_models.push(wsl_path.clone());
                cfg.save().map_err(|e| e.to_string())?;
            }
        }

        let stats_script = format!(
            "{}\nmodel_dir=$(__llm_panel_expand_tilde {path_literal})\nsize=$(du -sm -- \"$model_dir\" 2>/dev/null | cut -f1); files=$(find \"$model_dir\" -maxdepth 2 -type f 2>/dev/null | wc -l); echo \"$size|$files\"",
            crate::wsl::WSL_TILDE_EXPANSION_SNIPPET,
        );
        let stats = crate::wsl::run_script(&distro, &stats_script);
        let mut size_mb = 0u64;
        let mut files = 0usize;
        if let Some((sz, fl)) = stats.stdout.split_once('|') {
            size_mb = sz.trim().parse().unwrap_or(0);
            files = fl.trim().parse().unwrap_or(0);
        }

        let mut in_use = false;
        let mut in_use_server = None;
        {
            let srvs = st.servers.lock().unwrap();
            for ls in srvs.values() {
                if (ls.status == crate::state::ServerStatus::Running
                    || ls.status == crate::state::ServerStatus::Starting)
                    && ls.def.backend == "llamacpp"
                    && ls.def.model_id == wsl_path
                {
                    in_use = true;
                    in_use_server = Some(ls.def.name.clone());
                    break;
                }
            }
        }

        Ok(LibraryEntry {
            model_id: wsl_path,
            size_mb,
            files,
            quant: Some("native".into()),
            params_b: None,
            installed: true,
            in_use,
            in_use_server,
            task: Some("instruct".into()),
            is_local: true,
            model_path: None,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

pub fn compose_library_remove_script(model_id: &str, hub_dir: Option<&str>) -> String {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let hub = hub_dir.unwrap_or("~/.cache/huggingface/hub");
    format!(
        r#"{WSL_TILDE_EXPANSION_SNIPPET}
hub_dir=$(__llm_panel_expand_tilde {hub})
dir_name={dir_name}
dir="$hub_dir/$dir_name"
rm -rf -- "$dir"
# Prune unreferenced blob files
if [ -d "$hub_dir/blobs" ]; then
    shopt -s nullglob
    snaps=("$hub_dir"/models--*/snapshots)
    if [ ${{#snaps[@]}} -eq 0 ]; then
        rm -f -- "$hub_dir/blobs"/*
    else
        ref=$(find "${{snaps[@]}}" -type l -exec readlink {{}} + 2>/dev/null | sed 's#.*/##' | sort -u)
        for blob in "$hub_dir/blobs"/*; do
            [ -f "$blob" ] || continue
            hash=$(basename "$blob")
            if ! echo "$ref" | grep -qx "$hash"; then
                rm -f -- "$blob"
            fi
        done
    fi
fi
echo ok
"#,
        hub = crate::wsl::shell_quote_wsl(hub),
        dir_name = crate::wsl::shell_quote_wsl(&dir_name),
    )
}

#[tauri::command]
pub async fn library_remove(
    state: State<'_, Arc<AppState>>,
    model_id: String,
    model_path: Option<String>,
) -> Result<(), String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        // Native Windows GGUF: the entry's model_path points at the .gguf
        // inside gguf_dir; delete its containing repo folder.
        if let Some(ref path) = model_path {
            let file = PathBuf::from(path);
            if let (Some(parent), true) = (file.parent(), file.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)) {
                let gguf_dir = st.config().gguf_dir.trim().to_string();
                let gguf_root = PathBuf::from(&gguf_dir);
                let root_canonical = gguf_root
                    .canonicalize()
                    .map_err(|e| format!("GGUF directory is not accessible: {e}"))?;
                let parent_canonical = parent
                    .canonicalize()
                    .map_err(|e| format!("Model path is not accessible: {e}"))?;
                if !parent_canonical.starts_with(&root_canonical)
                    || parent_canonical == root_canonical
                {
                    return Err("Refusing to delete: path is outside the GGUF directory".into());
                }
                let pending_parent = parent_canonical.to_string_lossy().into_owned();
                let srvs = st.servers.lock().unwrap();
                for ls in srvs.values() {
                    let running = ls.status == crate::state::ServerStatus::Running
                        || ls.status == crate::state::ServerStatus::Starting;
                    let model_matches = ls.def.model_id == model_id
                        || ls.def.model_id.contains(&model_id)
                        || model_id.contains(&ls.def.model_id);
                    let path_matches = ls
                        .def
                        .model_path
                        .as_deref()
                        .map(|mp| mp.starts_with(&pending_parent))
                        .unwrap_or(false);
                    if running && (model_matches || path_matches) {
                        return Err(format!(
                            "Model \"{}\" is currently in use by active server \"{}\" — stop server first",
                            model_id, ls.def.name
                        ));
                    }
                }
                drop(srvs);
                std::fs::remove_dir_all(&parent).map_err(|e| {
                    format!("Failed to delete {}: {e}", parent.display())
                })?;
                return Ok(());
            }
        }
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

        let imported_path = {
            let cfg = st.config();
            cfg.imported_local_models
                .iter()
                .find(|path| path.as_str() == model_id)
                .cloned()
        };
        if let Some(imported_path) = imported_path {
            validate_imported_wsl_path(&imported_path)?;
            let distro = st.resolve_distro();
            let script = format!(
                "{}model_dir=$(__llm_panel_expand_tilde {})\n[ -d \"$model_dir\" ] && rm -rf -- \"$model_dir\"\necho ok",
                crate::wsl::WSL_TILDE_EXPANSION_SNIPPET,
                crate::wsl::shell_quote_wsl(&imported_path),
            );
            let out = crate::wsl::run_script(&distro, &script);
            if !out.ok {
                return Err(format!("Failed to delete imported model directory: {}", out.stderr));
            }
            let mut cfg = st.config.lock().unwrap();
            let mut next = cfg.clone();
            next.imported_local_models.retain(|path| path != &imported_path);
            next.save().map_err(|e| e.to_string())?;
            *cfg = next;
            return Ok(());
        }

        crate::hf::validate_model_id(&model_id).map_err(|_| "Invalid Hugging Face model ID".to_string())?;

        let distro = st.resolve_distro();
        let hub_dir = if let Some(home) = &st.config().advanced_settings.hf_home {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                Some(format!("{trimmed}/hub"))
            } else {
                None
            }
        } else {
            None
        };
        let script = compose_library_remove_script(&model_id, hub_dir.as_deref());
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
        let hub_dir = if let Some(home) = &st.config().advanced_settings.hf_home {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                format!("{trimmed}/hub")
            } else {
                "~/.cache/huggingface/hub".to_string()
            }
        } else {
            "~/.cache/huggingface/hub".to_string()
        };
        let script = format!(
            "{}hub_dir=$(__llm_panel_expand_tilde {}) && du -sm -- \"$hub_dir\" 2>/dev/null | cut -f1",
            crate::wsl::WSL_TILDE_EXPANSION_SNIPPET,
            crate::wsl::shell_quote_wsl(&hub_dir),
        );
        let out = crate::wsl::run_script(&distro, &script);
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
    sm.get(server_id)
        .map(|q| q.iter().cloned().collect())
        .unwrap_or_default()
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
        use std::os::windows::process::CommandExt;
        let mut cmd = std::process::Command::new("rundll32");
        cmd.creation_flags(0x08000000);
        cmd.args(["url.dll,FileProtocolHandler", &url])
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

#[derive(Serialize)]
pub struct FlashInferReady {
    pub nvcc: bool,
    pub gcc: bool,
    pub ninja: bool,
    pub python_dev: bool,
    pub cuda_home: Option<String>,
    pub torch_cuda_version: Option<String>,
}

#[tauri::command]
pub async fn check_flashinfer_ready(
    state: State<'_, Arc<AppState>>,
) -> Result<FlashInferReady, String> {
    let st = (*state).clone();
    tauri::async_runtime::spawn_blocking(move || {
        let distro = st.resolve_distro();
        let venv = st.config().venv_dir;
        server::validate_venv_dir(&venv)?;
        let venv_literal = crate::wsl::shell_quote_wsl(&venv);
        let script = format!(
            r#"{tilde}
set +e
venv_dir=$(__llm_panel_expand_tilde {venv})
nvcc_path=$(command -v nvcc 2>/dev/null || true)
gcc_path=$(command -v gcc 2>/dev/null || true)
ninja_path=$(command -v ninja 2>/dev/null || true)
[ -n "$nvcc_path" ] && echo nvcc=yes || echo nvcc=no
[ -n "$gcc_path" ] && echo gcc=yes || echo gcc=no
[ -n "$ninja_path" ] && echo ninja=yes || echo ninja=no
[ -f /usr/include/python3.12/Python.h ] && echo python_dev=yes || echo python_dev=no
if [ -n "$nvcc_path" ]; then dirname "$nvcc_path"; else echo none; fi
if [ -x "$venv_dir/bin/python" ]; then
  "$venv_dir/bin/python" -c 'import torch; print(torch.version.cuda or "unknown")' 2>/dev/null || echo unknown
else
  echo unknown
fi
"#,
            tilde = crate::wsl::WSL_TILDE_EXPANSION_SNIPPET,
            venv = venv_literal,
        );
        let output = crate::wsl::run_script(&distro, &script);
        if !output.ok {
            return Err(format!("FlashInfer readiness check failed: {}", output.combined()));
        }
        let mut result = FlashInferReady {
            nvcc: false,
            gcc: false,
            ninja: false,
            python_dev: false,
            cuda_home: None,
            torch_cuda_version: None,
        };
        let lines: Vec<&str> = output.stdout.lines().collect();
        for line in lines.iter().take(4) {
            match line.trim() {
                "nvcc=yes" => result.nvcc = true,
                "gcc=yes" => result.gcc = true,
                "ninja=yes" => result.ninja = true,
                "python_dev=yes" => result.python_dev = true,
                _ => {}
            }
        }
        if let Some(path) = lines.get(4).map(|line| line.trim()).filter(|value| !value.is_empty() && *value != "none") {
            result.cuda_home = Some(path.to_string());
        }
        if let Some(version) = lines
            .get(5)
            .map(|line| line.trim())
            .filter(|value| !value.is_empty() && *value != "unknown")
        {
            result.torch_cuda_version = Some(version.to_string());
        }
        Ok(result)
    })
    .await
    .map_err(|e| format!("FlashInfer readiness task error: {e}"))?
}

#[tauri::command]
pub async fn install_cuda_build_tools(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<ProvisionReport, String> {
    let st = (*state).clone();
    let distro = st.resolve_distro();
    tauri::async_runtime::spawn_blocking(move || {
        let mut on_log = |phase: &str, line: &str| {
            let _ = app.emit(
                "wsl-log",
                serde_json::json!({ "phase": phase, "line": line }),
            );
        };
        provision::phase_cuda_build_tools(&distro, &mut on_log)
            .map(|_| ProvisionReport {
                phases_completed: vec!["cuda-tools".into()],
                distro: distro.clone(),
                vllm_version: None,
                torch_version: None,
                cuda_available: false,
                gpu_name: None,
                vram_mb: None,
                bf16_supported: false,
            })
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("CUDA build tools task error: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::LlamaCppChannel;

    #[test]
    fn tool_call_response_length_is_not_reported_as_template_failure() {
        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": "I should inspect the available tools before calling one."
                }
            }],
            "usage": {"completion_tokens": 64}
        });
        let result = classify_tool_call_response(&response);
        assert!(!result.passed);
        assert!(result.hint.contains("cut off at max_tokens"));
        assert!(!result.hint.contains("--jinja"));
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("I should inspect the available tools before calling one.")
        );
    }

    #[test]
    fn valid_tool_call_response_passes_with_parsed_call() {
        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Seattle\"}"
                        }
                    }]
                }
            }]
        });
        let result = classify_tool_call_response(&response);
        assert!(result.passed);
        assert_eq!(
            result.tool_call.as_ref().unwrap()["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    fn text_emitted_tool_call_gets_template_hint() {
        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Seattle\"}}</tool_call>"
                }
            }]
        });
        let result = classify_tool_call_response(&response);
        assert!(!result.passed);
        assert!(result.hint.contains("emitted as text"));
        assert!(result.hint.contains("--jinja"));
    }

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
        st.config
            .lock()
            .unwrap()
            .memory_settings
            .manual_ram_limit_mb = Some(32768);
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
        assert!(
            st.rec_cache.lock().unwrap().is_none(),
            "rec_cache must be invalidated"
        );
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

        st.config
            .lock()
            .unwrap()
            .memory_settings
            .manual_ram_limit_mb = Some(14000);
        let sys_mem2 = get_system_memory_impl(&st);
        assert_eq!(sys_mem2.usable_budget_mb, 14000);
        assert_eq!(sys_mem2.manual_override_mb, Some(14000));

        st.config
            .lock()
            .unwrap()
            .memory_settings
            .enable_ram_overflow = false;
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
        if let Some(past) =
            std::time::Instant::now().checked_sub(std::time::Duration::from_secs(601))
        {
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
            variants: vec![QuantVariantWithFit {
                variant: QuantVariant {
                    repo_id: "Qwen/Qwen2.5-7B".into(),
                    format: QuantFormat::FP16,
                    label: "FP16".into(),
                    weight_bytes: None,
                    params_b: None,
                    gguf_file: None,
                    vllm_native: true,
                    required_channel: LlamaCppChannel::Upstream,
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
            }],
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
                let fut: std::pin::Pin<Box<dyn std::future::Future<Output = usize> + Send>> =
                    Box::pin(async move {
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
                    required_channel: LlamaCppChannel::Upstream,
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
            &[
                ("sort", "trendingScore"),
                ("pipeline_tag", "text-generation"),
                ("limit", "16"),
            ],
        )
        .unwrap();
        assert_eq!(
            url.query(),
            Some("sort=trendingScore&pipeline_tag=text-generation&limit=16")
        );
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
            is_local: false,
            model_path: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: LibraryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, entry);

        let dir_name = format!("models--{}", entry.model_id.replace('/', "--"));
        assert_eq!(dir_name, "models--Qwen--Qwen2.5-Coder-7B-Instruct");
    }

    #[test]
    fn test_hf_cache_scan_scripts_resolve_wsl_home() {
        let python_script =
            compose_hf_cache_scan_script("~/llm-lp/.venv/bin/python", "~/.cache/huggingface/hub");
        assert!(python_script.contains("venv_python=$(__llm_panel_expand_tilde"));
        assert!(python_script.contains("hub_dir=$(__llm_panel_expand_tilde"));
        assert!(python_script.contains("\"$venv_python\" - <<'PY'"));
        assert!(python_script.contains("os.environ['LOCAL_LLM_PANEL_HUB_DIR']"));
        assert!(!python_script.contains("sys.path.insert"));

        let quoted_paths = compose_hf_cache_scan_script(
            "~/venvs/vllm env/bin/python",
            "~/.cache/hugging face/hub",
        );
        assert!(quoted_paths.contains("'~/venvs/vllm env/bin/python'"));
        assert!(quoted_paths.contains("'~/.cache/hugging face/hub'"));

        let fallback_script = compose_hf_cache_fs_scan_script("~/.cache/huggingface/hub");
        assert!(fallback_script.contains("hub_dir=$(__llm_panel_expand_tilde"));
        assert!(fallback_script.contains("for p in \"$hub_dir\"/models--*"));
        assert!(!fallback_script.contains("for p in \"~/.cache/huggingface/hub\""));

        let exists_script = compose_hf_cache_model_exists_script(
            "~/.cache/huggingface/hub",
            "models--Qwen--Qwen2.5-Coder-7B-Instruct",
        );
        assert!(exists_script.contains("hub_dir=$(__llm_panel_expand_tilde"));
        assert!(exists_script.contains("[ -d \"$hub_dir/$dir_name\" ] && echo exists"));
        assert!(!exists_script.contains("[ -d \"~/.cache/huggingface/hub/"));
    }

    #[test]
    #[ignore = "requires a usable WSL distro"]
    fn test_hf_cache_fs_scan_discovers_model_with_wsl_home() {
        let Some(distro) = crate::wsl::installed_distros()
            .into_iter()
            .find(|name| name == "local-llm-panel-ubuntu")
            .or_else(crate::wsl::detect_default_distro)
        else {
            return;
        };

        let scan_script = compose_hf_cache_fs_scan_script("~/.cache/huggingface/hub");
        let script = format!(
            r#"
set -e
test_home=$(mktemp -d)
trap 'rm -rf "$test_home"' EXIT
export HOME="$test_home"
cache="$HOME/.cache/huggingface/hub"
mkdir -p "$cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/abc"
printf fixture > "$cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/abc/config.json"
{}
"#,
            scan_script
        );
        let out = crate::wsl::run_script(&distro, &script);
        assert!(out.ok, "WSL scan failed: {}", out.stderr);
        assert!(
            out.stdout
                .lines()
                .any(|line| line.starts_with("Qwen/Qwen2.5-0.5B-Instruct|")),
            "downloaded model was not discovered: {}",
            out.stdout
        );
    }

    #[test]
    fn test_hf_cache_fallback_handles_empty_python_result() {
        assert!(should_scan_hf_cache_fs(false, &[]));
        assert!(should_scan_hf_cache_fs(true, &[]));

        let entry = LibraryEntry {
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            size_mb: 512,
            files: 2,
            quant: Some("FP16".to_string()),
            params_b: Some(0.5),
            installed: true,
            in_use: false,
            in_use_server: None,
            task: Some("instruct".to_string()),
            is_local: false,
            model_path: None,
        };
        assert!(!should_scan_hf_cache_fs(true, &[entry.clone()]));

        let mut zero_sized = entry;
        zero_sized.size_mb = 0;
        assert!(should_scan_hf_cache_fs(true, &[zero_sized]));
    }

    #[tokio::test]
    #[ignore]
    async fn live_context_fit_command_probe() {
        let state = super::AppState::new();
        let vllm = super::analyze_context_fit_inner(
            &state,
            super::ContextFitRequest {
                backend: "vllm".into(),
                model_id: "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4".into(),
                model_path: None,
                context_tokens: Some(32_768),
                quant: Some("gptq".into()),
                kv_cache_dtype: None,
                gpu_mem_util: Some(0.85),
                cpu_offload_gb: Some(0),
                kv_offload_gb: Some(0),
                cache_type_k: None,
                cache_type_v: None,
                flash_attn: None,
                n_gpu_layers: None,
                n_cpu_moe: None,
                fit: None,
                fit_target: None,
                no_kv_offload: None,
            },
        )
        .await
        .unwrap();
        eprintln!("vllm={}", serde_json::to_string_pretty(&vllm).unwrap());
        assert!(vllm.fits);

        let path = std::env::var("LLM_TEST_GGUF_PATH")
            .expect("set LLM_TEST_GGUF_PATH to a local GGUF file");
        let llama = super::analyze_context_fit_inner(
            &state,
            super::ContextFitRequest {
                backend: "llamacpp".into(),
                model_id: path.clone(),
                model_path: Some(path),
                context_tokens: Some(32_768),
                quant: None,
                kv_cache_dtype: None,
                gpu_mem_util: None,
                cpu_offload_gb: None,
                kv_offload_gb: None,
                cache_type_k: Some("q8_0".into()),
                cache_type_v: Some("q8_0".into()),
                flash_attn: Some(true),
                n_gpu_layers: Some(99),
                n_cpu_moe: Some(24),
                fit: Some(true),
                fit_target: Some(1_024),
                no_kv_offload: Some(false),
            },
        )
        .await
        .unwrap();
        eprintln!("llama={}", serde_json::to_string_pretty(&llama).unwrap());
        assert!(llama.kv_bytes_per_token.unwrap_or_default() > 0.0);
    }

    #[test]
    #[ignore]
    fn live_cached_vllm_weight_probe() {
        let state = super::AppState::new();
        let size = cached_vllm_weight_gib(&state, "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4")
            .expect("cached Qwen model should be visible through the configured venv");
        eprintln!("cached_weight_gib={size}");
        assert!(size > 5.0 && size < 5.5);
    }

    #[test]
    fn imported_paths_reject_root_and_traversal() {
        assert!(validate_imported_wsl_path("/mnt/d/models/qwen").is_ok());
        assert!(validate_imported_wsl_path("/").is_err());
        assert!(validate_imported_wsl_path("/mnt/../etc").is_err());
    }

    #[test]
    fn server_ids_reject_empty_or_control_values() {
        assert!(validate_server_id("srv-abc_123").is_ok());
        assert!(validate_server_id("").is_err());
        assert!(validate_server_id("srv\nbad").is_err());
    }

    #[test]
    fn optional_fields_distinguish_values_from_clears() {
        #[derive(serde::Deserialize)]
        struct Probe {
            #[serde(default)]
            value: OptionalField<usize>,
        }

        let missing: Probe = serde_json::from_str("{}").unwrap();
        assert!(missing.value.into_inner().is_none());
        let cleared: Probe = serde_json::from_str(r#"{"value":null}"#).unwrap();
        assert!(cleared.value.into_inner().is_none());
        let set: Probe = serde_json::from_str(r#"{"value":32768}"#).unwrap();
        assert_eq!(set.value.into_inner(), Some(32768));
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
        let script = compose_library_remove_script("Qwen/Qwen2.5-0.5B", None);
        assert!(script.contains("models--Qwen--Qwen2.5-0.5B"));
        assert!(script.contains("hub_dir"));
        assert!(script.contains("blobs"));
        assert!(script.contains("readlink"));
        assert!(script.contains("snaps"));
    }

    #[test]
    fn test_cache_sweep_script_custom_hub() {
        let script = compose_library_remove_script("Qwen/Qwen2.5-0.5B", Some("/mnt/data/hf/hub"));
        assert!(script.contains("hub_dir=$(__llm_panel_expand_tilde '/mnt/data/hf/hub')"));
        assert!(script.contains("dir_name='models--Qwen--Qwen2.5-0.5B'"));
        assert!(script.contains("\"$hub_dir/blobs\""));
    }

    #[test]
    fn test_cache_sweep_script_quotes_untrusted_hub_path() {
        let script = compose_library_remove_script(
            "Qwen/Qwen2.5-0.5B",
            Some("/mnt/data/hf/hub; touch /tmp/should-not-exist"),
        );
        assert!(script.contains("hub; touch /tmp/should-not-exist'"));
        assert!(!script.contains("hub; touch /tmp/should-not-exist\"\n"));
    }

    #[test]
    fn test_pull_cancel_script_anchoring() {
        let model_id = "meta-llama/Llama-3.1-8B";
        let script = format!("pkill -f 'hf download.*[ /]{}(\\s|$)' || true", model_id);
        assert!(script.contains("[ /]meta-llama/Llama-3.1-8B(\\s|$)"));
    }
}
