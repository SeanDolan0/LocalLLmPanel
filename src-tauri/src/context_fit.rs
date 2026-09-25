use serde::Serialize;

const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;
const MIB_PER_GIB: f64 = 1024.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextFitStatus {
    Fits,
    FitsWithCpu,
    Tight,
    DoesNotFit,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextFitReport {
    pub backend: String,
    pub status: ContextFitStatus,
    pub fits: bool,
    pub model_id: String,
    pub requested_context: usize,
    pub native_context: Option<usize>,
    pub context_estimated: bool,
    pub weight_gib: Option<f64>,
    pub weight_source: String,
    pub kv_cache_label: String,
    pub kv_bytes_per_token: Option<f64>,
    pub required_kv_mb: Option<f64>,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub vram_budget_mb: Option<f64>,
    pub projected_vram_mb: Option<f64>,
    pub vram_headroom_mb: Option<f64>,
    pub max_vram_context: Option<usize>,
    pub max_with_ram_context: Option<usize>,
    pub max_context_no_overflow: Option<usize>,
    pub max_context_with_overflow: Option<usize>,
    pub overflow_enabled: bool,
    pub overflow_required_gb: Option<usize>,
    pub overflow_label: String,
    pub current_free_max_context: Option<usize>,
    pub fits_with_current_free: Option<bool>,
    pub ram_total_mb: Option<u64>,
    pub ram_available_mb: Option<u64>,
    pub ram_required_mb: Option<f64>,
    pub recommended_context: Option<usize>,
    pub recommended_kv_cache_dtype: Option<String>,
    pub recommended_cache_type_k: Option<String>,
    pub recommended_cache_type_v: Option<String>,
    pub effective_gpu_mem_util: Option<f64>,
    pub required_cpu_offload_gb: Option<usize>,
    pub recommendation: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct VllmFitInput {
    pub model_id: String,
    pub requested_context: usize,
    pub native_context: Option<usize>,
    pub context_estimated: bool,
    pub weight_gib: Option<f64>,
    pub weight_source: String,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub kv_cache_dtype: String,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub ram_total_mb: Option<u64>,
    pub ram_available_mb: Option<u64>,
    pub gpu_mem_util: f64,
    pub vram_overhead_mb: f64,
    pub max_context_cap: Option<usize>,
    pub ram_overflow_enabled: bool,
    pub manual_ram_limit_mb: Option<u64>,
    pub safety_reserve_mb: f64,
    pub cpu_offload_gb: usize,
    pub kv_offload_gb: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LlamaFitInput {
    pub model_id: String,
    pub requested_context: usize,
    pub native_context: Option<usize>,
    pub weight_gib: Option<f64>,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub key_head_dim: Option<usize>,
    pub value_head_dim: Option<usize>,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub flash_attn: bool,
    pub n_gpu_layers: Option<usize>,
    pub n_cpu_moe: Option<usize>,
    pub fit: bool,
    pub fit_target_mb: f64,
    pub no_kv_offload: bool,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub ram_total_mb: Option<u64>,
    pub ram_available_mb: Option<u64>,
    pub vram_overhead_mb: f64,
}

fn base_report(
    backend: &str,
    model_id: String,
    requested_context: usize,
    native_context: Option<usize>,
    context_estimated: bool,
    weight_gib: Option<f64>,
    weight_source: impl Into<String>,
    kv_cache_label: String,
    vram_total_mb: Option<u64>,
    vram_free_mb: Option<u64>,
    ram_total_mb: Option<u64>,
    ram_available_mb: Option<u64>,
) -> ContextFitReport {
    ContextFitReport {
        backend: backend.into(),
        status: ContextFitStatus::Unknown,
        fits: false,
        model_id,
        requested_context,
        native_context,
        context_estimated,
        weight_gib,
        weight_source: weight_source.into(),
        kv_cache_label,
        kv_bytes_per_token: None,
        required_kv_mb: None,
        vram_total_mb,
        vram_free_mb,
        vram_budget_mb: None,
        projected_vram_mb: None,
        vram_headroom_mb: None,
        max_vram_context: None,
        max_with_ram_context: None,
        max_context_no_overflow: None,
        max_context_with_overflow: None,
        overflow_enabled: false,
        overflow_required_gb: None,
        overflow_label: "CPU KV cache".into(),
        current_free_max_context: None,
        fits_with_current_free: None,
        ram_total_mb,
        ram_available_mb,
        ram_required_mb: None,
        recommended_context: None,
        recommended_kv_cache_dtype: None,
        recommended_cache_type_k: None,
        recommended_cache_type_v: None,
        effective_gpu_mem_util: None,
        required_cpu_offload_gb: None,
        recommendation: "Insufficient metadata for a reliable fit result.".into(),
        warnings: Vec::new(),
    }
}

pub(crate) fn effective_gpu_mem_util(
    requested: f64,
    vram_total_mb: f64,
    vram_free_mb: Option<f64>,
) -> f64 {
    let cap = if vram_total_mb >= 20_000.0 {
        0.92
    } else if vram_total_mb >= 15_000.0 {
        0.90
    } else {
        0.86
    };
    let safe = vram_free_mb
        .filter(|free| *free > 0.0)
        .map(|free| ((free - 1_400.0) / vram_total_mb * 100.0).floor() / 100.0)
        .unwrap_or(cap)
        .clamp(0.10, cap);
    requested.clamp(0.10, cap).min(safe)
}

pub(crate) fn vram_context_for_budget(budget_mb: f64, kv_bytes_per_token: f64) -> usize {
    if budget_mb <= 0.0 || kv_bytes_per_token <= 0.0 {
        return 0;
    }
    (budget_mb * BYTES_PER_MIB / kv_bytes_per_token) as usize
}

fn vllm_dtype_bytes(dtype: &str, torch_dtype: Option<&str>) -> Option<f64> {
    match dtype.to_ascii_lowercase().as_str() {
        "float16" | "f16" | "bfloat16" | "bf16" => Some(2.0),
        "fp8" | "fp8_e4m3" | "fp8_e5m2" => Some(1.0),
        "auto" => match torch_dtype.map(str::to_ascii_lowercase).as_deref() {
            Some(value) if value.contains("fp8") || value.contains("float8") => Some(1.0),
            _ => Some(2.0),
        },
        _ => None,
    }
}

fn round_context_down(value: usize) -> usize {
    if value == 0 {
        0
    } else if value < 1_024 {
        value
    } else {
        (value / 1_024) * 1_024
    }
}

fn usable_ram_mb(
    total: Option<u64>,
    available: Option<u64>,
    enabled: bool,
    manual: Option<u64>,
    reserve: f64,
) -> Option<f64> {
    if !enabled {
        return Some(0.0);
    }
    if let Some(manual) = manual.filter(|value| *value > 0) {
        return Some(manual as f64);
    }
    available
        .map(|value| (value as f64 - reserve).max(0.0))
        .or(total.map(|value| (value as f64 - reserve).max(0.0)))
}

pub(crate) fn analyze_vllm_context(input: &VllmFitInput) -> ContextFitReport {
    let mut report = base_report(
        "vllm",
        input.model_id.clone(),
        input.requested_context,
        input.native_context,
        input.context_estimated,
        input.weight_gib,
        input.weight_source.clone(),
        input.kv_cache_dtype.clone(),
        input.vram_total_mb,
        input.vram_free_mb,
        input.ram_total_mb,
        input.ram_available_mb,
    );
    let (Some(layers), Some(kv_heads), Some(head_dim), Some(total_mb), Some(weight_gib)) = (
        input.n_layers,
        input.n_kv_heads,
        input.head_dim,
        input.vram_total_mb,
        input.weight_gib,
    ) else {
        report
            .warnings
            .push("Model dimensions, weight size, or VRAM are unavailable.".into());
        return report;
    };
    if input.requested_context == 0 || layers == 0 || kv_heads == 0 || head_dim == 0 {
        report
            .warnings
            .push("Context and model dimensions must be greater than zero.".into());
        return report;
    }
    let Some(bytes_per_element) = vllm_dtype_bytes(&input.kv_cache_dtype, None) else {
        report.warnings.push(format!(
            "Unsupported vLLM KV cache dtype '{}'.",
            input.kv_cache_dtype
        ));
        return report;
    };

    let base_bpt = 2.0 * layers as f64 * kv_heads as f64 * head_dim as f64;
    let kv_bpt = base_bpt * bytes_per_element;
    let required_kv_mb = input.requested_context as f64 * kv_bpt / BYTES_PER_MIB;
    let native_limit = input
        .native_context
        .filter(|value| *value > 0)
        .map(|native| {
            input
                .max_context_cap
                .filter(|value| *value > 0)
                .map(|cap| native.min(cap))
                .unwrap_or(native)
        });
    let cpu_offload_mb = input.cpu_offload_gb as f64 * MIB_PER_GIB;
    let gpu_weight_gib = (weight_gib - input.cpu_offload_gb as f64).max(0.0);
    let budget_mb = total_mb as f64 * input.gpu_mem_util.clamp(0.10, 0.95)
        - gpu_weight_gib * MIB_PER_GIB
        - input.vram_overhead_mb;
    let max_vram = native_limit
        .map(|limit| vram_context_for_budget(budget_mb, kv_bpt).min(limit))
        .unwrap_or_else(|| vram_context_for_budget(budget_mb, kv_bpt));
    let current_util = effective_gpu_mem_util(
        input.gpu_mem_util,
        total_mb as f64,
        input.vram_free_mb.map(|value| value as f64),
    );
    let current_budget_mb =
        total_mb as f64 * current_util - gpu_weight_gib * MIB_PER_GIB - input.vram_overhead_mb;
    let current_max = native_limit
        .map(|limit| vram_context_for_budget(current_budget_mb, kv_bpt).min(limit))
        .unwrap_or_else(|| vram_context_for_budget(current_budget_mb, kv_bpt));
    let ram_usable = usable_ram_mb(
        input.ram_total_mb,
        input.ram_available_mb,
        input.ram_overflow_enabled,
        input.manual_ram_limit_mb,
        input.safety_reserve_mb,
    )
    .unwrap_or(0.0);
    let ram_for_kv = (ram_usable - cpu_offload_mb).max(0.0);
    let ram_max_context = if input.ram_overflow_enabled {
        vram_context_for_budget(ram_for_kv, kv_bpt)
    } else {
        0
    };
    let max_with_ram = native_limit
        .map(|limit| current_max.saturating_add(ram_max_context).min(limit))
        .unwrap_or_else(|| current_max.saturating_add(ram_max_context));
    let configured_ram_context = if input.ram_overflow_enabled && input.kv_offload_gb > 0 {
        let cap_context = vram_context_for_budget(input.kv_offload_gb as f64 * MIB_PER_GIB, kv_bpt);
        ram_max_context.min(cap_context)
    } else {
        0
    };
    let max_with_configured_overflow = native_limit
        .map(|limit| {
            current_max
                .saturating_add(configured_ram_context)
                .min(limit)
        })
        .unwrap_or_else(|| current_max.saturating_add(configured_ram_context));
    let overflow_enabled = input.ram_overflow_enabled && input.kv_offload_gb > 0;
    let overflow_required_gb = if input.requested_context > current_max {
        let overflow_bytes = (input.requested_context - current_max) as f64 * kv_bpt;
        Some(
            (overflow_bytes / BYTES_PER_MIB / MIB_PER_GIB)
                .ceil()
                .max(1.0) as usize,
        )
    } else {
        Some(0)
    };
    let native_fits = native_limit
        .map(|limit| input.requested_context <= limit)
        .unwrap_or(true);
    let vram_fits = required_kv_mb <= budget_mb.max(0.0) && native_fits;
    let current_fits = required_kv_mb <= current_budget_mb.max(0.0) && native_fits;
    let projected = gpu_weight_gib * MIB_PER_GIB + required_kv_mb + input.vram_overhead_mb;
    let headroom = budget_mb - required_kv_mb;
    let current_headroom = current_budget_mb - required_kv_mb;

    report.kv_bytes_per_token = Some(kv_bpt);
    report.required_kv_mb = Some(required_kv_mb);
    report.vram_budget_mb = Some(budget_mb);
    report.projected_vram_mb = Some(projected);
    report.vram_headroom_mb = Some(headroom);
    report.max_vram_context = Some(max_vram);
    report.max_with_ram_context = Some(max_with_configured_overflow);
    report.max_context_no_overflow = Some(current_max);
    report.max_context_with_overflow = Some(if overflow_enabled {
        max_with_configured_overflow
    } else {
        max_with_ram
    });
    report.overflow_enabled = overflow_enabled;
    report.overflow_required_gb = overflow_required_gb;
    report.current_free_max_context = Some(current_max);
    report.fits_with_current_free = Some(current_fits);
    report.effective_gpu_mem_util = Some(current_util);
    report.ram_required_mb = Some(
        cpu_offload_mb
            + if current_fits {
                0.0
            } else {
                (input.requested_context.saturating_sub(current_max)) as f64 * kv_bpt
                    / BYTES_PER_MIB
            },
    );
    if !native_fits {
        report
            .warnings
            .push("Requested context exceeds the model's configured native context.".into());
    }
    if current_util + 0.005 < input.gpu_mem_util {
        report.warnings.push(format!(
            "Current free VRAM will clamp effective utilization from {:.0}% to {:.0}%.",
            input.gpu_mem_util * 100.0,
            current_util * 100.0
        ));
    }
    if budget_mb <= 0.0 {
        let shortfall_mb = gpu_weight_gib * MIB_PER_GIB + input.vram_overhead_mb
            - total_mb as f64 * input.gpu_mem_util;
        report.required_cpu_offload_gb =
            Some((shortfall_mb / MIB_PER_GIB).ceil().max(0.0) as usize);
    }

    if current_fits {
        report.fits = true;
        report.status = if current_headroom < current_budget_mb.max(1.0) * 0.10 {
            ContextFitStatus::Tight
        } else if input.cpu_offload_gb > 0 {
            ContextFitStatus::FitsWithCpu
        } else {
            ContextFitStatus::Fits
        };
        report.recommended_context = Some(input.requested_context);
        report.recommendation = if report.status == ContextFitStatus::Tight {
            format!(
                "The context fits without overflow, but current headroom is only {:.0} MiB.",
                current_headroom.max(0.0)
            )
        } else {
            format!(
                "The requested context fits without overflow with about {:.0} MiB of current VRAM headroom.",
                current_headroom.max(0.0)
            )
        };
    } else if overflow_enabled
        && max_with_configured_overflow >= input.requested_context
        && native_fits
    {
        report.fits = true;
        report.status = ContextFitStatus::FitsWithCpu;
        report.recommended_context = Some(input.requested_context);
        report.recommendation = "The context fits with the configured CPU KV-cache overflow budget, but throughput may be lower while the offloaded portion is used.".into();
    } else if vram_fits && !current_fits {
        report.status = ContextFitStatus::Tight;
        report.fits = false;
        report.recommended_context = Some(round_context_down(current_max));
        if let Some(required) = report.overflow_required_gb.filter(|value| *value > 0) {
            report.recommendation = format!(
                "The configured GPU budget fits, but current VRAM only supports about {} tokens without overflow. Close GPU-heavy apps or enable at least {required} GB of CPU KV offload for {} tokens.",
                round_context_down(current_max),
                input.requested_context
            );
        } else {
            report.recommendation = format!(
                "Close GPU-heavy applications or lower the context to about {} tokens.",
                round_context_down(current_max)
            );
        }
    } else {
        report.status = ContextFitStatus::DoesNotFit;
        let fp8_bpt = base_bpt;
        let fp8_max = native_limit
            .map(|limit| vram_context_for_budget(current_budget_mb, fp8_bpt).min(limit))
            .unwrap_or_else(|| vram_context_for_budget(current_budget_mb, fp8_bpt));
        if native_fits && input.kv_cache_dtype != "fp8_e4m3" && fp8_max >= input.requested_context {
            report.recommended_kv_cache_dtype = Some("fp8_e4m3".into());
            report.recommended_context = Some(input.requested_context);
            report.recommendation = "Use FP8 E4M3 KV cache to fit this context without overflow; verify output quality after the change.".into();
        } else if native_fits && max_with_ram >= input.requested_context {
            if let Some(required) = report.overflow_required_gb.filter(|value| *value > 0) {
                report.recommendation = format!(
                    "Enable at least {required} GB of CPU KV offload to run {} tokens with overflow, or lower the context to the no-overflow limit.",
                    input.requested_context
                );
            } else {
                report.recommendation = "Enable CPU KV overflow to reach the requested context without overflowing GPU memory.".into();
            }
        } else {
            let recommended = round_context_down(current_max);
            report.recommended_context = Some(recommended);
            if recommended > 0 {
                report.recommendation = format!(
                    "Lower the context to at most {} tokens for the current no-overflow limit.",
                    recommended
                );
            } else if let Some(offload) = report.required_cpu_offload_gb {
                report.recommendation = format!(
                    "Lower the context or enable at least {offload} GB of CPU weight offload."
                );
            } else {
                report.recommendation =
                    "Lower the context, reduce GPU utilization pressure, or use a smaller model."
                        .into();
            }
        }
    }
    report
}

pub(crate) fn cache_bytes_per_element(cache_type: &str) -> Option<f64> {
    let value = cache_type.to_ascii_lowercase();
    if matches!(value.as_str(), "f32" | "float32") {
        return Some(4.0);
    }
    if matches!(value.as_str(), "f16" | "fp16" | "bf16" | "bfloat16") {
        return Some(2.0);
    }
    let (block, bytes) = match value.as_str() {
        "q2_k" => (256.0, 84.0),
        "q3_k" | "iq3_xxs" => (256.0, 110.0),
        "q4_0" | "iq4_nl" => (32.0, 18.0),
        "q4_1" => (32.0, 20.0),
        "q4_k" | "iq4_xs" => (256.0, 144.0),
        "q5_0" => (32.0, 22.0),
        "q5_1" => (32.0, 24.0),
        "q5_k" => (256.0, 176.0),
        "q6_k" => (256.0, 210.0),
        "q8_0" => (32.0, 34.0),
        "q8_1" => (32.0, 36.0),
        "q8_k" => (256.0, 292.0),
        _ => return None,
    };
    Some(bytes / block)
}

fn llama_kv_bytes_per_token(input: &LlamaFitInput) -> Option<f64> {
    let layers = input.n_layers?;
    let kv_heads = input.n_kv_heads?;
    let key_dim = input.key_head_dim?;
    let value_dim = input.value_head_dim?;
    let key_bytes = cache_bytes_per_element(&input.cache_type_k)?;
    let value_bytes = cache_bytes_per_element(&input.cache_type_v)?;
    if layers == 0 || kv_heads == 0 || key_dim == 0 || value_dim == 0 {
        return None;
    }
    Some(
        layers as f64
            * (kv_heads as f64 * key_dim as f64 * key_bytes
                + kv_heads as f64 * value_dim as f64 * value_bytes),
    )
}

pub(crate) fn analyze_llama_context(input: &LlamaFitInput) -> ContextFitReport {
    let label = format!("K {} / V {}", input.cache_type_k, input.cache_type_v);
    let mut report = base_report(
        "llamacpp",
        input.model_id.clone(),
        input.requested_context,
        input.native_context,
        false,
        input.weight_gib,
        "gguf_file_size",
        label,
        input.vram_total_mb,
        input.vram_free_mb,
        input.ram_total_mb,
        input.ram_available_mb,
    );
    let (Some(weight_gib), Some(total_mb)) = (input.weight_gib, input.vram_total_mb) else {
        report
            .warnings
            .push("GGUF file size or VRAM is unavailable.".into());
        return report;
    };
    let Some(kv_bpt) = llama_kv_bytes_per_token(input) else {
        report
            .warnings
            .push("GGUF KV dimensions or selected K/V cache types are unavailable.".into());
        return report;
    };
    let free_mb = input.vram_free_mb.unwrap_or(total_mb) as f64;
    if input.vram_free_mb.is_none() {
        report
            .warnings
            .push("Current free VRAM is unavailable; total VRAM was used.".into());
    }
    let gpu_weight_gib = match (
        input.n_gpu_layers,
        input.n_layers.filter(|layers| *layers > 0),
    ) {
        (Some(requested), Some(layers)) if requested < layers => {
            weight_gib * requested as f64 / layers as f64
        }
        _ => weight_gib,
    };
    let cpu_weight_mb = (weight_gib - gpu_weight_gib).max(0.0) * MIB_PER_GIB;
    let fit_margin = if input.fit {
        input.fit_target_mb.max(0.0)
    } else {
        0.0
    };
    let overhead = input.vram_overhead_mb.max(0.0) + fit_margin;
    let budget_mb = free_mb - gpu_weight_gib * MIB_PER_GIB - overhead;
    let native_limit = input.native_context.filter(|value| *value > 0);
    let max_vram = native_limit
        .map(|limit| vram_context_for_budget(budget_mb, kv_bpt).min(limit))
        .unwrap_or_else(|| vram_context_for_budget(budget_mb, kv_bpt));
    let required_kv_mb = input.requested_context as f64 * kv_bpt / BYTES_PER_MIB;
    let projected = gpu_weight_gib * MIB_PER_GIB + overhead + required_kv_mb;
    let headroom = free_mb - projected;
    let native_fits = native_limit
        .map(|limit| input.requested_context <= limit)
        .unwrap_or(true);
    let vram_fits = required_kv_mb <= budget_mb.max(0.0) && native_fits;
    let ram_available = input
        .ram_available_mb
        .map(|value| value as f64)
        .unwrap_or(0.0);
    let ram_for_kv = (ram_available - cpu_weight_mb).max(0.0);
    let ram_required = cpu_weight_mb + required_kv_mb;
    let ram_context = vram_context_for_budget(ram_for_kv, kv_bpt);
    let max_with_ram = native_limit
        .map(|limit| max_vram.saturating_add(ram_context).min(limit))
        .unwrap_or_else(|| max_vram.saturating_add(ram_context));
    let max_with_configured_overflow = if input.no_kv_offload {
        max_with_ram
    } else {
        max_vram
    };
    let overflow_enabled = input.no_kv_offload;

    report.kv_bytes_per_token = Some(kv_bpt);
    report.required_kv_mb = Some(required_kv_mb);
    report.vram_budget_mb = Some(budget_mb);
    report.projected_vram_mb = Some(projected);
    report.vram_headroom_mb = Some(headroom);
    report.max_vram_context = Some(max_vram);
    report.max_with_ram_context = Some(max_with_configured_overflow);
    report.max_context_no_overflow = Some(max_vram);
    report.max_context_with_overflow = Some(max_with_ram);
    report.overflow_enabled = overflow_enabled;
    report.overflow_required_gb = None;
    report.current_free_max_context = Some(max_vram);
    report.fits_with_current_free = Some(vram_fits);
    report.ram_required_mb = Some(ram_required);
    if projected > free_mb {
        report.required_cpu_offload_gb =
            Some(((projected - free_mb) / MIB_PER_GIB).ceil().max(0.0) as usize);
    }
    if !native_fits {
        report
            .warnings
            .push("Requested context exceeds the GGUF model's native context.".into());
    }
    if input.n_cpu_moe.unwrap_or(0) > 0 {
        report.warnings.push("CPU MoE offload can reduce GPU weight use, but the exact layer split is not known; this estimate is conservative.".into());
    }
    if input.flash_attn
        && matches!(
            input.cache_type_v.to_ascii_lowercase().as_str(),
            "q4_0" | "q4_1" | "q5_0" | "q5_1" | "iq4_nl"
        )
    {
        report.warnings.push("Flash Attention support for quantized V cache varies by llama.cpp build; the runtime may promote or reject this cache type.".into());
    }
    if input.fit {
        report.warnings.push("With --fit enabled, llama.cpp may move layers to CPU to make the model start; the GPU-only projection above is conservative.".into());
    }

    if vram_fits {
        report.fits = true;
        report.status = if headroom < free_mb.max(1.0) * 0.10 {
            ContextFitStatus::Tight
        } else {
            ContextFitStatus::Fits
        };
        report.recommended_context = Some(input.requested_context);
        report.recommendation = if report.status == ContextFitStatus::Tight {
            format!(
                "The context fits, but only about {:.0} MiB of VRAM remains.",
                headroom.max(0.0)
            )
        } else {
            format!(
                "The requested context fits with about {:.0} MiB of VRAM headroom.",
                headroom.max(0.0)
            )
        };
    } else if overflow_enabled && ram_required <= ram_available && native_fits {
        report.fits = true;
        report.status = ContextFitStatus::FitsWithCpu;
        report.recommended_context = Some(input.requested_context);
        report.recommendation = "The model can run with CPU KV-cache overflow, but generation will usually be much slower.".into();
    } else if native_fits && max_with_ram >= input.requested_context {
        report.status = ContextFitStatus::DoesNotFit;
        report.recommended_context = Some(round_context_down(max_vram));
        report.recommendation = format!(
            "Enable CPU KV-cache overflow (Disable KV-cache offload) to run {} tokens, or lower the context to about {} tokens.",
            input.requested_context,
            round_context_down(max_vram)
        );
    } else {
        report.status = ContextFitStatus::DoesNotFit;
        let q8_bpt = llama_kv_bytes_per_token(&LlamaFitInput {
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            ..input.clone()
        });
        let q4_bpt = llama_kv_bytes_per_token(&LlamaFitInput {
            cache_type_k: "q4_0".into(),
            cache_type_v: "q4_0".into(),
            ..input.clone()
        });
        let q8_max = q8_bpt
            .map(|bpt| {
                native_limit
                    .map(|limit| vram_context_for_budget(budget_mb, bpt).min(limit))
                    .unwrap_or_else(|| vram_context_for_budget(budget_mb, bpt))
            })
            .unwrap_or(0);
        let q4_max = q4_bpt
            .map(|bpt| {
                native_limit
                    .map(|limit| vram_context_for_budget(budget_mb, bpt).min(limit))
                    .unwrap_or_else(|| vram_context_for_budget(budget_mb, bpt))
            })
            .unwrap_or(0);
        if q8_max >= input.requested_context
            && input.cache_type_k != "q8_0"
            && input.cache_type_v != "q8_0"
        {
            report.recommended_cache_type_k = Some("q8_0".into());
            report.recommended_cache_type_v = Some("q8_0".into());
            report.recommended_context = Some(input.requested_context);
            report.recommendation =
                "Use q8_0 for both K and V cache to reduce KV-cache memory.".into();
        } else if q4_max >= input.requested_context
            && input.cache_type_k != "q4_0"
            && input.cache_type_v != "q4_0"
        {
            report.recommended_cache_type_k = Some("q4_0".into());
            report.recommended_cache_type_v = Some("q4_0".into());
            report.recommended_context = Some(input.requested_context);
            report.recommendation = "Use q4_0 K/V cache to fit this context; verify quality because K-cache quantization is more aggressive.".into();
        } else {
            let recommended = round_context_down(max_vram);
            report.recommended_context = Some(recommended);
            if recommended > 0 {
                report.recommendation = format!("Lower the context to at most {} tokens, reduce GPU layers, or enable CPU MoE offload.", recommended);
            } else if let Some(offload) = report.required_cpu_offload_gb {
                report.recommendation = format!(
                    "Reduce GPU-resident weights by about {offload} GB using fewer GPU layers or CPU MoE offload, or use a smaller GGUF."
                );
            } else {
                report.recommendation = "Lower GPU layers, enable CPU MoE offload, use a smaller GGUF, or lower the context enough for a CPU KV cache.".into();
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_vllm(requested_context: usize) -> VllmFitInput {
        VllmFitInput {
            model_id: "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4".into(),
            requested_context,
            native_context: Some(32_768),
            context_estimated: false,
            weight_gib: Some(5.27),
            weight_source: "local_cache".into(),
            n_layers: Some(28),
            n_kv_heads: Some(4),
            head_dim: Some(128),
            kv_cache_dtype: "auto".into(),
            vram_total_mb: Some(12_227),
            vram_free_mb: Some(12_200),
            ram_total_mb: Some(23_000),
            ram_available_mb: Some(22_000),
            gpu_mem_util: 0.85,
            vram_overhead_mb: 2_500.0,
            max_context_cap: None,
            ram_overflow_enabled: true,
            manual_ram_limit_mb: None,
            safety_reserve_mb: 4_096.0,
            cpu_offload_gb: 0,
            kv_offload_gb: 0,
        }
    }

    #[test]
    fn vllm_qwen_32k_fits_fp16_kv() {
        let report = analyze_vllm_context(&qwen_vllm(32_768));
        assert_eq!(report.status, ContextFitStatus::Fits);
        assert!(report.fits);
        assert!(report.max_vram_context.unwrap_or_default() >= 32_768);
        assert!(report.vram_headroom_mb.unwrap_or_default() > 0.0);
    }

    #[test]
    fn vllm_over_native_context_does_not_fit_even_with_memory() {
        let mut input = qwen_vllm(65_536);
        input.native_context = Some(32_768);
        let report = analyze_vllm_context(&input);
        assert_eq!(report.status, ContextFitStatus::DoesNotFit);
        assert!(!report.fits);
        assert!(
            report.warnings.iter().any(|w| w.contains("native")),
            "warnings: {:?}",
            report.warnings
        );
    }

    #[test]
    fn vllm_recommends_fp8_for_oversized_context() {
        let mut input = qwen_vllm(65_536);
        input.native_context = Some(131_072);
        let report = analyze_vllm_context(&input);
        assert_eq!(report.status, ContextFitStatus::DoesNotFit);
        assert_eq!(
            report.recommended_kv_cache_dtype.as_deref(),
            Some("fp8_e4m3")
        );
        assert!(report.recommended_context.unwrap_or_default() >= 32_768);
    }

    #[test]
    fn vllm_reports_no_overflow_and_potential_overflow_limits() {
        let mut input = qwen_vllm(32_768);
        input.vram_free_mb = Some(9_700);
        input.kv_offload_gb = 0;
        let report = analyze_vllm_context(&input);
        assert!(report.max_context_no_overflow.unwrap_or_default() < 32_768);
        assert!(report.max_context_with_overflow.unwrap_or_default() >= 32_768);
        assert!(!report.overflow_enabled);
        assert!(report.overflow_required_gb.unwrap_or_default() > 0);
    }

    #[test]
    fn vllm_configured_kv_offload_enables_overflow_mode() {
        let mut input = qwen_vllm(32_768);
        input.vram_free_mb = Some(9_700);
        input.kv_offload_gb = 4;
        let report = analyze_vllm_context(&input);
        assert!(report.overflow_enabled);
        assert_eq!(report.status, ContextFitStatus::FitsWithCpu);
        assert!(report.fits);
    }

    #[test]
    fn vllm_current_vram_pressure_is_reported_as_tight() {
        let mut input = qwen_vllm(32_768);
        input.vram_free_mb = Some(9_700);
        let report = analyze_vllm_context(&input);
        assert_eq!(report.status, ContextFitStatus::Tight);
        assert!(!report.fits);
        assert!(report.recommended_context.unwrap_or_default() < 32_768);
    }

    #[test]
    fn vllm_missing_dimensions_is_unknown_not_a_false_pass() {
        let mut input = qwen_vllm(32_768);
        input.n_layers = None;
        let report = analyze_vllm_context(&input);
        assert_eq!(report.status, ContextFitStatus::Unknown);
        assert!(!report.fits);
    }

    #[test]
    fn gpu_utilization_uses_runtime_free_vram_clamp() {
        let effective = effective_gpu_mem_util(0.85, 12_227.0, Some(8_000.0));
        assert!((effective - 0.54).abs() < 0.011);
        assert!(effective < 0.85);
    }

    #[test]
    fn context_math_does_not_floor_a_tiny_budget_to_512() {
        let max = vram_context_for_budget(1.0, 57_344.0);
        assert_eq!(max, 18);
    }

    fn ternary_llama(cache_k: &str, cache_v: &str) -> LlamaFitInput {
        LlamaFitInput {
            model_id: "Ternary-Bonsai-80B-PQ2_0.gguf".into(),
            requested_context: 32_768,
            native_context: Some(131_072),
            weight_gib: Some(8.0),
            n_layers: Some(64),
            n_kv_heads: Some(8),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            cache_type_k: cache_k.into(),
            cache_type_v: cache_v.into(),
            flash_attn: true,
            n_gpu_layers: Some(99),
            n_cpu_moe: Some(24),
            fit: true,
            fit_target_mb: 1_024.0,
            no_kv_offload: false,
            vram_total_mb: Some(12_227),
            vram_free_mb: Some(12_200),
            ram_total_mb: Some(32_000),
            ram_available_mb: Some(24_000),
            vram_overhead_mb: 1_024.0,
        }
    }

    #[test]
    fn llama_q8_cache_recommends_lower_context_for_large_model() {
        let report = analyze_llama_context(&ternary_llama("q8_0", "q8_0"));
        assert_eq!(report.status, ContextFitStatus::DoesNotFit);
        assert!(report.recommended_context.unwrap_or_default() < 32_768);
        assert!(report.max_vram_context.unwrap_or_default() > 0);
    }

    #[test]
    fn llama_reports_no_overflow_and_cpu_kv_overflow_limits() {
        let report = analyze_llama_context(&ternary_llama("q8_0", "q8_0"));
        assert!(report.max_context_no_overflow.unwrap_or_default() < 32_768);
        assert!(report.max_context_with_overflow.unwrap_or_default() >= 32_768);
        assert!(!report.overflow_enabled);
        assert_eq!(report.overflow_label, "CPU KV cache");
    }

    #[test]
    fn llama_q4_cache_increases_context_headroom() {
        let q8 = analyze_llama_context(&ternary_llama("q8_0", "q8_0"));
        let q4 = analyze_llama_context(&ternary_llama("q4_0", "q4_0"));
        assert!(q4.max_vram_context.unwrap_or_default() > q8.max_vram_context.unwrap_or_default());
    }

    #[test]
    fn llama_unknown_cache_type_is_unknown() {
        let report = analyze_llama_context(&ternary_llama("mystery", "q8_0"));
        assert_eq!(report.status, ContextFitStatus::Unknown);
        assert!(report.warnings.iter().any(|w| w.contains("cache")));
    }

    #[test]
    #[ignore]
    fn live_vllm_context_probe() {
        let mut input = qwen_vllm(32_768);
        input.weight_gib = Some(5.203_266_596_421_599);
        input.weight_source = "local_hf_cache".into();
        input.vram_free_mb = Some(10_625);
        let report = analyze_vllm_context(&input);
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert!(report.fits);
    }

    #[test]
    #[ignore]
    fn live_llama_context_probe() {
        let path = std::env::var("LLM_TEST_GGUF_PATH")
            .expect("set LLM_TEST_GGUF_PATH to a local GGUF file");
        let path = std::path::PathBuf::from(path);
        let metadata = crate::gguf::metadata_from_path(&path).unwrap();
        let requested = std::env::var("LLM_TEST_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(32_768);
        let report = analyze_llama_context(&LlamaFitInput {
            model_id: path.to_string_lossy().into_owned(),
            requested_context: requested,
            native_context: metadata.context_length,
            weight_gib: std::fs::metadata(&path)
                .ok()
                .map(|meta| meta.len() as f64 / 1024.0 / 1024.0 / 1024.0),
            n_layers: metadata.block_count,
            n_kv_heads: metadata.head_count_kv.or(metadata.head_count),
            key_head_dim: metadata.key_head_dim,
            value_head_dim: metadata.value_head_dim,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            flash_attn: true,
            n_gpu_layers: Some(99),
            n_cpu_moe: Some(24),
            fit: true,
            fit_target_mb: 1_024.0,
            no_kv_offload: false,
            vram_total_mb: Some(12_227),
            vram_free_mb: Some(12_200),
            ram_total_mb: Some(32_000),
            ram_available_mb: Some(24_000),
            vram_overhead_mb: 1_024.0,
        });
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert!(report.kv_bytes_per_token.unwrap_or_default() > 0.0);
    }
}
