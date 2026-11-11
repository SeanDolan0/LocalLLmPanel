use crate::hardware_sizer::types::{
    AllocationReport, FitReport, FitVerdict, HardwareSnapshot, ModelGeometry, TargetBackend,
};
use crate::state::{MemorySettings, ServerDef};

/// Bytes per parameter for various quantization formats.
pub fn bytes_per_param(quant: &str) -> f64 {
    let q = quant.to_ascii_lowercase();
    if q.starts_with("pq") || q.starts_with("ptq") {
        0.28
    } else if q.starts_with("q4")
        || q.contains("q4_")
        || q.contains("iq4")
        || q == "awq"
        || q == "gptq"
        || q == "int4"
    {
        0.55
    } else if q.starts_with("q8") || q.contains("q8_") {
        1.05
    } else if q.starts_with("q5") || q.contains("q5_") {
        0.68
    } else if q.starts_with("q6") || q.contains("q6_") {
        0.80
    } else if q.starts_with("q3") || q.contains("q3_") || q.contains("iq3") {
        0.45
    } else if q.starts_with("q2") || q.contains("q2_") || q.contains("iq2") {
        0.35
    } else if q.starts_with("q1") || q.contains("q1_") {
        0.25
    } else if q == "fp8" || q == "int8" {
        1.0
    } else if q == "gguf" {
        0.55
    } else {
        2.0 // fp16 / bf16 / default
    }
}

/// Compute raw model weights in gigabytes.
pub fn weight_gb(params_b: f64, quant: &str) -> f64 {
    params_b * bytes_per_param(quant)
}

/// Extract parameter count in billions from a model name or file path.
pub fn parse_params_from_name(name: &str) -> Option<f64> {
    let base = name.split(['/', '\\']).last().unwrap_or(name);
    // Find pattern like "7b", "0.5b", "70B", "1.5B"
    let re = regex_lite::Regex::new(r"(?i)(?:^|[-_ /])(\d+(?:\.\d+)?)[bB](?:[-_ /.]|$)").ok()?;
    let caps = re.captures(base)?;
    caps.get(1)?.as_str().parse::<f64>().ok()
}

/// Compute KV cache bytes per token given architectural geometry.
pub fn kv_bytes_per_token(
    num_layers: Option<u32>,
    num_kv_heads: Option<u32>,
    head_dim: Option<u32>,
    params_b: f64,
) -> f64 {
    if let (Some(nl), Some(nkv), Some(hd)) = (num_layers, num_kv_heads, head_dim) {
        // fp16 = 2 bytes * 2 (key + value) * layers * kv_heads * head_dim
        2.0 * 2.0 * nl as f64 * nkv as f64 * hd as f64
    } else {
        // Empirical step-function fallback based on model parameter tier
        if params_b <= 3.5 {
            32_768.0
        } else if params_b <= 9.0 {
            65_536.0
        } else if params_b <= 16.0 {
            131_072.0
        } else if params_b <= 35.0 {
            262_144.0
        } else {
            327_680.0
        }
    }
}

/// Estimated weight footprint in GB of a server definition: (main_model_gb, speculative_gb).
pub fn server_weights_gb(def: &ServerDef) -> (f64, f64) {
    let mut main_weight = 0.0;
    if let Some(p) = def.params_b.filter(|p| *p > 0.0) {
        main_weight = weight_gb(p, &def.quant);
    } else if let Some(p) = parse_params_from_name(&def.model_id) {
        main_weight = weight_gb(p, &def.quant);
    } else if let Some(ref path) = def.model_path {
        if let Ok(meta) = std::fs::metadata(path) {
            main_weight = meta.len() as f64 / 1_073_741_824.0;
        } else if let Some(p) = parse_params_from_name(path) {
            let q = if def.backend == "llamacpp" { "gguf" } else { &def.quant };
            main_weight = weight_gb(p, q);
        }
    }

    let mut spec_weight = 0.0;
    if def.speculative_enabled {
        if let Some(ref path) = def.speculative_model_path {
            if let Ok(meta) = std::fs::metadata(path) {
                spec_weight = meta.len() as f64 / 1_073_741_824.0;
            } else if let Some(p) = parse_params_from_name(path) {
                spec_weight = weight_gb(p, "gguf");
            }
        }
        if spec_weight == 0.0 {
            if let Some(ref model) = def.speculative_model {
                if let Ok(meta) = std::fs::metadata(model) {
                    spec_weight = meta.len() as f64 / 1_073_741_824.0;
                } else if let Some(p) = parse_params_from_name(model) {
                    let quant = if def.backend == "llamacpp" { "gguf" } else { "fp16" };
                    spec_weight = weight_gb(p, quant);
                }
            }
        }
    }

    (main_weight, spec_weight)
}

/// Compute allocation report across running servers given current hardware and settings.
pub fn calculate_allocation(
    running_defs: &[&ServerDef],
    hw: &HardwareSnapshot,
    settings: &MemorySettings,
) -> AllocationReport {
    let mut total_main = 0.0;
    let mut total_spec = 0.0;
    let mut explicit_ram = 0.0;
    let mut total_gpu_demanded = 0.0;

    for def in running_defs {
        let (main_w, spec_w) = server_weights_gb(def);
        let server_total = main_w + spec_w;
        total_main += main_w;
        total_spec += spec_w;

        if let Some(offload) = def.cpu_offload_gb.filter(|&gb| gb > 0) {
            explicit_ram += offload as f64;
            total_gpu_demanded += (server_total - offload as f64).max(0.0);
        } else if def.backend == "llamacpp" && def.n_gpu_layers == Some(0) {
            explicit_ram += server_total;
        } else {
            total_gpu_demanded += server_total;
        }
    }

    let total_weight_gb = total_main + total_spec;

    // Capacity spill beyond physical GPU VRAM
    let vram_total_gb = hw.vram_total_mb as f64 / 1024.0;
    let vram_overhead_gb = settings.vram_overhead_mb as f64 / 1024.0;
    let usable_gpu_capacity = (vram_total_gb - vram_overhead_gb).max(0.0);

    let capacity_spill = if total_gpu_demanded > usable_gpu_capacity && usable_gpu_capacity > 0.0 {
        total_gpu_demanded - usable_gpu_capacity
    } else {
        0.0
    };

    let ram_spill_gb = explicit_ram + capacity_spill;

    let vram_pressure_pct = if hw.vram_total_mb > 0 {
        let used = (hw.vram_total_mb as f64) - (hw.vram_effective_free_mb as f64);
        (used / hw.vram_total_mb as f64 * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    let effective_free_gb = hw.vram_effective_free_mb as f64 / 1024.0;
    let is_low_vram = !hw.is_unified_memory && effective_free_gb < 1.0 && hw.vram_total_mb > 0;

    let warning_message = if is_low_vram {
        Some(format!(
            "Free VRAM is critically low ({:.1} GB remaining with {} MB reserved). Additional models may spill into system RAM.",
            effective_free_gb, hw.vram_reserved_mb
        ))
    } else if ram_spill_gb > 0.5 {
        Some(format!(
            "{:.1} GB of active model weights are currently offloaded or spilling into system RAM.",
            ram_spill_gb
        ))
    } else {
        None
    };

    AllocationReport {
        running_servers_count: running_defs.len(),
        total_weight_gb,
        main_weight_gb: total_main,
        spec_weight_gb: total_spec,
        ram_spill_gb,
        total_reserved_vram_mb: hw.vram_reserved_mb,
        vram_pressure_pct,
        is_low_vram,
        warning_message,
    }
}

/// Sizing and fit evaluation for a given model architecture against target backend.
pub fn calculate_fit(
    geom: &ModelGeometry,
    backend: TargetBackend,
    hw: &HardwareSnapshot,
    settings: &MemorySettings,
) -> FitReport {
    let weight_gb = weight_gb(geom.params_b, &geom.quant);
    let _vram_total_gb = hw.vram_total_mb as f64 / 1024.0;
    let vram_free_gb = hw.vram_effective_free_mb as f64 / 1024.0;
    let vram_overhead_gb = settings.vram_overhead_mb as f64 / 1024.0;

    // Available target RAM based on backend
    let (_ram_total_gb, ram_available_gb) = match backend {
        TargetBackend::Vllm => (
            hw.wsl_ram_total_mb.unwrap_or(hw.host_ram_total_mb) as f64 / 1024.0,
            hw.wsl_ram_available_mb.unwrap_or(hw.host_ram_available_mb) as f64 / 1024.0,
        ),
        TargetBackend::LlamaCpp => (
            hw.host_ram_total_mb as f64 / 1024.0,
            hw.host_ram_available_mb as f64 / 1024.0,
        ),
        TargetBackend::Generic => {
            let host_avail = hw.host_ram_available_mb as f64 / 1024.0;
            let wsl_avail = hw.wsl_ram_available_mb.unwrap_or(0) as f64 / 1024.0;
            if wsl_avail > host_avail {
                (hw.wsl_ram_total_mb.unwrap_or(hw.host_ram_total_mb) as f64 / 1024.0, wsl_avail)
            } else {
                (hw.host_ram_total_mb as f64 / 1024.0, host_avail)
            }
        }
    };

    let usable_vram_gb = (vram_free_gb - vram_overhead_gb).max(0.0);
    let kv_bytes_tok = kv_bytes_per_token(
        geom.num_layers,
        geom.num_kv_heads,
        geom.head_dim,
        geom.params_b,
    );

    let (verdict, required_vram_gb, ram_offload_gb, max_ctx, score) = if hw.is_unified_memory {
        // Unified memory pool (Apple Silicon, AMD Strix Halo, Grace Hopper)
        let total_pool = ram_available_gb + vram_free_gb;
        if weight_gb <= total_pool * 0.85 {
            let kv_budget_bytes = ((total_pool * 0.85) - weight_gb) * 1_073_741_824.0;
            let ctx = ((kv_budget_bytes / kv_bytes_tok) as u32).min(131_072).max(1024);
            (FitVerdict::FullGpu, weight_gb, 0.0, ctx, 95)
        } else {
            (FitVerdict::TooTight, weight_gb, 0.0, 512, 10)
        }
    } else if weight_gb <= usable_vram_gb {
        // Fits completely into discrete GPU VRAM
        let kv_budget_bytes = (usable_vram_gb - weight_gb) * 1_073_741_824.0;
        let ctx = ((kv_budget_bytes / kv_bytes_tok) as u32).min(65_536).max(2048);
        (FitVerdict::FullGpu, weight_gb, 0.0, ctx, 90)
    } else if settings.offload_weights_allowed && (weight_gb <= usable_vram_gb + ram_available_gb * 0.8) {
        // Hybrid CPU/GPU offload
        let gpu_alloc = usable_vram_gb;
        let offload_w = weight_gb - gpu_alloc;
        let ctx = 4096.min(geom.context_length.unwrap_or(4096));
        (FitVerdict::HybridOffload, gpu_alloc, offload_w, ctx, 60)
    } else if weight_gb <= ram_available_gb * 0.85 {
        // CPU only
        let ctx = 2048.min(geom.context_length.unwrap_or(2048));
        (FitVerdict::CpuOnly, 0.0, weight_gb, ctx, 35)
    } else {
        (FitVerdict::TooTight, 0.0, weight_gb, 512, 10)
    };

    let mut notes = Vec::new();
    if backend == TargetBackend::Vllm && hw.wsl_ram_total_mb.is_some() {
        notes.push("Sized against WSL2 virtual machine memory pool.".to_string());
    } else if backend == TargetBackend::LlamaCpp {
        notes.push("Sized against Windows host physical RAM pool.".to_string());
    }

    if hw.vram_reserved_mb > 0 {
        notes.push(format!(
            "Evaluated against effective free VRAM ({} MB reserved by in-flight starts).",
            hw.vram_reserved_mb
        ));
    }

    // Throughput estimation (GB/s estimate: 800 GB/s for modern GPU, 60 GB/s for DDR5 RAM)
    let bandwidth_gbs = match verdict {
        FitVerdict::FullGpu => 800.0,
        FitVerdict::HybridOffload => 150.0,
        FitVerdict::CpuOnly => 60.0,
        FitVerdict::TooTight => 10.0,
    };
    let est_tok = (bandwidth_gbs * 0.5 / (geom.params_b * bytes_per_param(&geom.quant))).clamp(1.0, 250.0) as f32;

    FitReport {
        verdict,
        score,
        max_context: max_ctx,
        weight_gb,
        required_vram_gb,
        ram_offload_gb,
        est_tokens_per_sec: est_tok,
        target_backend: backend,
        notes,
    }
}
