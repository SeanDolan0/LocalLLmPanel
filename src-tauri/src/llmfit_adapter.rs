//! Adapter integrating `llmfit-core` into `LocalLLmPanel`.
//!
//! Provides hardware detection, embedded model database access,
//! 4-dimensional scoring (Quality, Speed, Fit, Context), and model recommendations
//! matching `llmfit` exactly.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use llmfit_core::analysis::{self, InstalledIndex};
use llmfit_core::fit::{
    rank_models_by_fit, FitLevel, InferenceRuntime, ModelFit, RunMode as LlmfitRunMode,
    ScoreComponents,
};
use llmfit_core::hardware::{GpuBackend, SystemSpecs};
use llmfit_core::models::{LlmModel, ModelDatabase, UseCase, QUANT_HIERARCHY};

use crate::fit::{FitResult, FitVerdict, FormatSupport, RunMode};
use crate::hf::{QuantFormat, QuantVariant};
use crate::state::LlamaCppChannel;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoreComponentsDto {
    pub quality: f64,
    pub speed: f64,
    pub fit: f64,
    pub context: f64,
}

impl From<ScoreComponents> for ScoreComponentsDto {
    fn from(sc: ScoreComponents) -> Self {
        Self {
            quality: (sc.quality * 10.0).round() / 10.0,
            speed: (sc.speed * 10.0).round() / 10.0,
            fit: (sc.fit * 10.0).round() / 10.0,
            context: (sc.context * 10.0).round() / 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufSourceDto {
    pub provider: String,
    pub repo: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantVariantWithFit {
    pub variant: QuantVariant,
    pub fit: FitResult,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelWithFit {
    pub id: String,
    pub provider: Option<String>,
    pub downloads: i64,
    pub likes: i64,
    pub trending_score: f64,
    pub pipeline_tag: Option<String>,
    pub params_b: Option<f64>,
    pub parameter_count: Option<String>,
    pub context: Option<usize>,
    pub context_source: Option<&'static str>,
    pub context_estimated: bool,
    pub head_dim: Option<usize>,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub use_case: Option<String>,
    pub category: Option<String>,
    pub release_date: Option<String>,
    pub variants: Vec<QuantVariantWithFit>,
    pub best_variant_idx: usize,
    pub score: f64,
    pub score_components: Option<ScoreComponentsDto>,
    pub best_quant: Option<String>,
    pub runtime: Option<String>,
    pub fit_level: Option<String>,
    pub run_mode: Option<String>,
    pub usable_context: Option<usize>,
    pub effective_context_length: Option<usize>,
    pub estimated_tps: Option<f64>,
    pub memory_required_gb: Option<f64>,
    pub memory_available_gb: Option<f64>,
    pub utilization_pct: Option<f64>,
    pub notes: Vec<String>,
    pub capabilities: Vec<String>,
    pub gguf_sources: Vec<GgufSourceDto>,
    pub installed: bool,
}

impl Default for ModelWithFit {
    fn default() -> Self {
        Self {
            id: String::new(),
            provider: None,
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            pipeline_tag: None,
            params_b: None,
            parameter_count: None,
            context: None,
            context_source: None,
            context_estimated: false,
            head_dim: None,
            n_layers: None,
            n_kv_heads: None,
            use_case: None,
            category: None,
            release_date: None,
            variants: Vec::new(),
            best_variant_idx: 0,
            score: 0.0,
            score_components: None,
            best_quant: None,
            runtime: None,
            fit_level: None,
            run_mode: None,
            usable_context: None,
            effective_context_length: None,
            estimated_tps: None,
            memory_required_gb: None,
            memory_available_gb: None,
            utilization_pct: None,
            notes: Vec::new(),
            capabilities: Vec::new(),
            gguf_sources: Vec::new(),
            installed: false,
        }
    }
}

static SPECS: OnceLock<SystemSpecs> = OnceLock::new();
static MODEL_DB: OnceLock<ModelDatabase> = OnceLock::new();

pub fn get_system_specs() -> &'static SystemSpecs {
    SPECS.get_or_init(SystemSpecs::detect)
}

pub fn get_model_database() -> &'static ModelDatabase {
    MODEL_DB.get_or_init(ModelDatabase::new)
}

/// Convert llmfit `FitLevel` and `LlmfitRunMode` to our legacy-compatible `FitVerdict`
pub fn fit_level_to_verdict(level: FitLevel) -> FitVerdict {
    match level {
        FitLevel::Perfect => FitVerdict::Comfortable,
        FitLevel::Good => FitVerdict::Comfortable,
        FitLevel::Marginal => FitVerdict::Constrained,
        FitLevel::TooTight => FitVerdict::DoesNotFit,
    }
}

pub fn fit_run_mode_to_ui(mode: LlmfitRunMode) -> RunMode {
    match mode {
        LlmfitRunMode::Gpu => RunMode::Gpu,
        LlmfitRunMode::MoeOffload => RunMode::GpuRamSwap,
        LlmfitRunMode::CpuOffload => RunMode::CpuOffload,
        LlmfitRunMode::CpuOnly => RunMode::CpuOffload,
        LlmfitRunMode::TensorParallel => RunMode::Gpu,
    }
}

pub fn use_case_to_category_name(uc: UseCase) -> (&'static str, &'static str) {
    match uc {
        UseCase::General => ("General", "general"),
        UseCase::Coding => ("Coding", "coding"),
        UseCase::Reasoning => ("Reasoning", "reasoning"),
        UseCase::Chat => ("Chat", "chat"),
        UseCase::Multimodal => ("Multimodal", "multimodal"),
        UseCase::Embedding => ("Embedding", "embedding"),
    }
}

/// Transform an `LlmModel` + its `ModelFit` into the full `ModelWithFit` struct.
pub fn fit_to_model_with_fit(fit: &ModelFit) -> ModelWithFit {
    let m = &fit.model;
    let (cat_title, pipeline) = use_case_to_category_name(fit.use_case);

    let params_b = m.parameters_raw.map(|p| p as f64 / 1e9).or_else(|| {
        fit.model
            .parameter_count
            .trim_end_matches('B')
            .trim_end_matches('b')
            .parse::<f64>()
            .ok()
    });

    let mut variants: Vec<QuantVariantWithFit> = Vec::new();

    // Determine variant quant candidates
    let mut quant_candidates: Vec<String> = Vec::new();
    if fit.runtime == InferenceRuntime::Vllm {
        if !m.quantization.is_empty() {
            quant_candidates.push(m.quantization.clone());
        }
        for q in ["AWQ-4bit", "GPTQ-Int4", "FP8", "FP16"] {
            if !quant_candidates.iter().any(|c| c.eq_ignore_ascii_case(q)) {
                quant_candidates.push(q.to_string());
            }
        }
    } else {
        // GGUF / llama.cpp
        if !fit.best_quant.is_empty() && !quant_candidates.contains(&fit.best_quant) {
            quant_candidates.push(fit.best_quant.clone());
        }
        for q in QUANT_HIERARCHY {
            let qs = q.to_string();
            if !quant_candidates.contains(&qs) {
                quant_candidates.push(qs);
            }
        }
    }

    let specs = get_system_specs();

    for q_str in &quant_candidates {
        let is_best = q_str.eq_ignore_ascii_case(&fit.best_quant);
        let q_lower = q_str.to_lowercase();
        let format = if q_lower.contains("awq") {
            QuantFormat::AWQ
        } else if q_lower.contains("gptq") {
            QuantFormat::GPTQ
        } else if q_lower.contains("fp8") {
            QuantFormat::FP8
        } else if q_lower.contains("fp16") || q_lower.contains("f16") || q_lower.contains("bf16") {
            QuantFormat::FP16
        } else {
            QuantFormat::GGUF
        };

        let variant_repo = if let Some(src) = m.gguf_sources.first() {
            src.repo.clone()
        } else {
            m.name.clone()
        };

        let quant_variant = QuantVariant {
            repo_id: variant_repo,
            format: format.clone(),
            label: q_str.clone(),
            weight_bytes: None,
            params_b,
            gguf_file: None,
            vllm_native: format != QuantFormat::GGUF,
            required_channel: LlamaCppChannel::Upstream,
        };

        // Score this specific quant variant
        let mut model_variant = m.clone();
        model_variant.quantization = q_str.clone();
        let variant_fit = ModelFit::analyze(&model_variant, specs);

        let verdict = fit_level_to_verdict(variant_fit.fit_level);
        let run_mode = fit_run_mode_to_ui(variant_fit.run_mode);
        let score_val = (variant_fit.score.round() as u8).min(100);

        let vram_pct = (variant_fit.utilization_pct.round().min(255.0)) as u8;
        let ram_pct = if variant_fit.run_mode == LlmfitRunMode::CpuOffload
            || variant_fit.run_mode == LlmfitRunMode::MoeOffload
        {
            vram_pct
        } else {
            0
        };

        let reason_text = if !variant_fit.notes.is_empty() {
            variant_fit.notes.join("; ")
        } else {
            format!("{cat_title} fit: {} tok/s", variant_fit.estimated_tps)
        };

        let v_result = FitResult {
            verdict,
            run_mode,
            score: score_val,
            weight_gb: variant_fit.memory_required_gb,
            vram_context: if variant_fit.run_mode == LlmfitRunMode::Gpu {
                variant_fit.usable_context as usize
            } else {
                0
            },
            extended_context: variant_fit.usable_context as usize,
            native_context: m.context_length as usize,
            usable_context: variant_fit.usable_context as usize,
            swap_space_gb: variant_fit.moe_offloaded_gb.unwrap_or(0.0).ceil() as usize,
            cpu_offload_gb: if variant_fit.run_mode == LlmfitRunMode::CpuOffload {
                variant_fit.memory_required_gb.ceil() as usize
            } else {
                0
            },
            est_tok_s: Some(variant_fit.estimated_tps),
            measured_tok_s: variant_fit.measured_tps.map(|mt| mt.tok_s),
            vram_pct,
            ram_pct,
            format_support: if format == QuantFormat::GGUF {
                FormatSupport::Experimental
            } else {
                FormatSupport::Native
            },
            reason: reason_text,
            score_components: Some(variant_fit.score_components.into()),
            runtime: Some(variant_fit.runtime.label().to_string()),
            notes: variant_fit.notes.clone(),
        };

        variants.push(QuantVariantWithFit {
            variant: quant_variant,
            fit: v_result,
        });

        if is_best && variants.len() > 1 {
            let last_idx = variants.len() - 1;
            variants.swap(0, last_idx);
        }
    }

    let best_variant_idx = 0;

    let gguf_sources: Vec<GgufSourceDto> = m
        .gguf_sources
        .iter()
        .map(|s| GgufSourceDto {
            provider: s.provider.clone(),
            repo: s.repo.clone(),
        })
        .collect();

    let capabilities: Vec<String> = m
        .capabilities
        .iter()
        .map(|c| c.label().to_string())
        .collect();

    ModelWithFit {
        id: m.name.clone(),
        provider: Some(m.provider.clone()),
        downloads: 0,
        likes: 0,
        trending_score: fit.score,
        pipeline_tag: Some(pipeline.to_string()),
        params_b,
        parameter_count: Some(m.parameter_count.clone()),
        context: Some(m.context_length as usize),
        context_source: Some("llmfit database"),
        context_estimated: false,
        head_dim: m.head_dim.map(|h| h as usize),
        n_layers: m.num_hidden_layers.map(|l| l as usize),
        n_kv_heads: m.num_key_value_heads.map(|k| k as usize),
        use_case: Some(m.use_case.clone()),
        category: Some(cat_title.to_string()),
        release_date: m.release_date.clone(),
        variants,
        best_variant_idx,
        score: (fit.score * 10.0).round() / 10.0,
        score_components: Some(fit.score_components.into()),
        best_quant: Some(fit.best_quant.clone()),
        runtime: Some(fit.runtime.label().to_string()),
        fit_level: Some(match fit.fit_level {
            FitLevel::Perfect => "Perfect".to_string(),
            FitLevel::Good => "Good".to_string(),
            FitLevel::Marginal => "Marginal".to_string(),
            FitLevel::TooTight => "Too Tight".to_string(),
        }),
        run_mode: Some(match fit.run_mode {
            LlmfitRunMode::Gpu => "GPU".to_string(),
            LlmfitRunMode::MoeOffload => "MoE Offload".to_string(),
            LlmfitRunMode::CpuOffload => "CPU Offload".to_string(),
            LlmfitRunMode::CpuOnly => "CPU Only".to_string(),
            LlmfitRunMode::TensorParallel => "Tensor Parallel".to_string(),
        }),
        usable_context: Some(fit.usable_context as usize),
        effective_context_length: Some(fit.effective_context_length as usize),
        estimated_tps: Some(fit.estimated_tps),
        memory_required_gb: Some(fit.memory_required_gb),
        memory_available_gb: Some(fit.memory_available_gb),
        utilization_pct: Some(fit.utilization_pct),
        notes: fit.notes.clone(),
        capabilities,
        gguf_sources,
        installed: fit.installed,
    }
}

/// Fetch top model recommendations for current hardware using exact `llmfit` ranking.
pub fn recommend_models(limit: usize) -> Vec<ModelWithFit> {
    let specs = get_system_specs();
    let db = get_model_database();
    let installed = InstalledIndex::detect_all();

    let mut fits = analysis::build_model_fits(db, specs, &installed, None, None);

    // Hide MLX-only models on non-Apple Silicon systems
    let is_apple_silicon = specs.backend == GpuBackend::Metal && specs.unified_memory;
    if !is_apple_silicon {
        fits.retain(|f| !f.model.is_mlx_only());
    }

    // Retain only models that fit
    fits.retain(|f| f.fit_level != FitLevel::TooTight);

    // Exact llmfit ranking
    fits = rank_models_by_fit(fits);

    fits.into_iter()
        .take(limit)
        .map(|f| fit_to_model_with_fit(&f))
        .collect()
}

/// Search models using llmfit's embedded database and exact scoring.
pub fn search_models_local(query: &str, limit: usize) -> Vec<ModelWithFit> {
    let specs = get_system_specs();
    let db = get_model_database();
    let q_clean = query.trim().to_lowercase();

    if q_clean.is_empty() {
        return Vec::new();
    }

    let is_apple_silicon = specs.backend == GpuBackend::Metal && specs.unified_memory;

    let mut matches: Vec<&LlmModel> = db
        .get_all_models()
        .iter()
        .filter(|m| {
            if !is_apple_silicon && m.is_mlx_only() {
                return false;
            }
            let name = m.name.to_lowercase();
            let provider = m.provider.to_lowercase();
            let params = m.parameter_count.to_lowercase();
            let use_case = m.use_case.to_lowercase();
            name.contains(&q_clean)
                || provider.contains(&q_clean)
                || params.contains(&q_clean)
                || use_case.contains(&q_clean)
        })
        .collect();

    // Prioritize exact substring matches in model name
    matches.sort_by_key(|m| {
        let name = m.name.to_lowercase();
        if name == q_clean {
            0
        } else if name.starts_with(&q_clean) {
            1
        } else {
            2
        }
    });

    matches
        .into_iter()
        .take(limit)
        .map(|m| {
            let fit = ModelFit::analyze(m, specs);
            fit_to_model_with_fit(&fit)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_specs_detection() {
        let specs = get_system_specs();
        assert!(specs.total_ram_gb > 0.0);
        assert!(!specs.cpu_name.is_empty());
        assert!(specs.total_cpu_cores > 0);
        assert_eq!(
            specs.has_gpu,
            !specs.gpus.is_empty(),
            "has_gpu must agree with the detected GPU list"
        );
        if specs.has_gpu {
            assert!(specs.gpu_name.is_some());
            assert!(specs.gpu_count > 0);
        } else {
            assert!(specs.gpu_name.is_none());
            assert_eq!(specs.gpu_count, 0);
        }
    }

    #[test]
    fn test_recommend_models() {
        let recs = recommend_models(10);
        assert!(!recs.is_empty(), "Recommendations should not be empty");
        assert!(recs.len() <= 10);

        for (i, m) in recs.iter().take(5).enumerate() {
            eprintln!(
                "#{i}: id={}, score={}, quant={:?}, runtime={:?}",
                m.id, m.score, m.best_quant, m.runtime
            );
        }
        // Scores depend on the detected hardware (especially whether a GPU is
        // present), so assert ordering and validity rather than a machine-specific
        // score value.
        assert!(recs[0].score.is_finite());
        assert!(recs
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score));

        for m in &recs {
            assert!(!m.id.is_empty());
            assert!(m.score > 0.0, "Score should be positive: {}", m.score);
            assert!(
                m.score_components.is_some(),
                "Score components should be present"
            );
            let sc = m.score_components.unwrap();
            assert!(sc.quality >= 0.0 && sc.quality <= 100.0);
            assert!(sc.speed >= 0.0 && sc.speed <= 100.0);
            assert!(sc.fit >= 0.0 && sc.fit <= 100.0);
            assert!(sc.context >= 0.0 && sc.context <= 100.0);
            assert!(m.runtime.is_some(), "Runtime should be present");
            assert!(m.best_quant.is_some(), "Best quant should be present");
        }
    }

    #[test]
    fn test_search_models_local() {
        let results = search_models_local("qwen", 5);
        assert!(
            !results.is_empty(),
            "Search for 'qwen' should return results"
        );
        for m in &results {
            assert!(
                m.id.to_lowercase().contains("qwen")
                    || m.provider
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains("qwen")
            );
            assert!(m.score >= 0.0);
        }
    }
}
