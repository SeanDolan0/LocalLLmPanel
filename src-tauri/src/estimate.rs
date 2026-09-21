//! Heuristic estimators: context window, VRAM context-fit, max decode tok/s.
//!
//! All functions here are pure and unit-tested. Outputs are explicitly
//! estimates — measured data (from vLLM `/metrics`) always wins in the UI.

use serde::Serialize;

/// Bytes of memory per parameter for a given quantization.
/// fp16/bf16 = 2, fp8 = 1, int4 (AWQ/GPTQ) ≈ 0.55 (weights + scale overhead).
pub fn bytes_per_param(quant: &str) -> f64 {
    let q = quant.to_ascii_lowercase();
    let q_str = q.as_str();
    if q_str.starts_with("q4")
        || q_str.contains("q4_")
        || q_str.contains("iq4")
        || q_str == "awq"
        || q_str == "gptq"
        || q_str == "int4"
    {
        0.55 // 4-bit: 0.50 bytes/weight + ~0.05 scaling overhead
    } else if q_str.starts_with("q8") || q_str.contains("q8_") {
        1.05 // 8-bit: 1.00 byte/weight + scaling overhead
    } else if q_str.starts_with("q5") || q_str.contains("q5_") {
        0.68 // 5-bit: ~0.625 + overhead
    } else if q_str.starts_with("q6") || q_str.contains("q6_") {
        0.80 // 6-bit: ~0.75 + overhead
    } else if q_str.starts_with("q3") || q_str.contains("q3_") || q_str.contains("iq3") {
        0.45
    } else if q_str.starts_with("q2") || q_str.contains("q2_") || q_str.contains("iq2") {
        0.35
    } else if q_str == "fp8" || q_str == "int8" {
        1.0
    } else if q_str == "gguf" {
        0.55 // default GGUF assumption is ~Q4_K_M
    } else {
        2.0 // fp16 / bf16 / unset
    }
}

/// KV-cache bytes per single token-position, for fp16 cache:
/// `2 (K+V) × n_layers × n_kv_heads × head_dim × 2 bytes`.
pub fn kv_bytes_per_token(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> f64 {
    2.0 * n_layers as f64 * n_kv_heads as f64 * head_dim as f64 * 2.0
}

/// Fallback estimate of KV-cache bytes per token when specific layer dims are missing.
pub fn estimate_kv_bytes_per_token(params_b: f64) -> f64 {
    if params_b <= 0.0 {
        return 0.0;
    }
    if params_b <= 3.5 {
        32_768.0
    } else if params_b <= 10.0 {
        131_072.0
    } else if params_b <= 40.0 {
        262_144.0
    } else {
        327_680.0
    }
}

/// Weight footprint in GB (params_b is in billions).
pub fn weight_gb(params_b: f64, quant: &str) -> f64 {
    params_b * bytes_per_param(quant)
}

/// Context length possible under a VRAM budget given explicit weight GB.
///
/// `usable_kv = vram_usable_mb - overhead_mb - weight_gb * 1024`
/// where `vram_usable_mb = vram_total_mb * gpu_util`. Clamps non-positive KV to 0,
/// and positive fits to at least 512 tokens.
pub fn context_fit_with_weight(
    vram_total_mb: f64,
    gpu_util: f64,
    weight_gb: f64,
    kv_bytes_per_token: f64,
    overhead_mb: f64,
) -> usize {
    let usable_vram_mb = vram_total_mb * gpu_util;
    let weights_mb = weight_gb * 1024.0;
    let kv_vram_mb = usable_vram_mb - weights_mb - overhead_mb;
    if kv_vram_mb <= 0.0 || kv_bytes_per_token <= 0.0 {
        return 0;
    }
    let ctx = (kv_vram_mb * 1024.0 * 1024.0 / kv_bytes_per_token) as usize;
    ctx.max(512)
}

/// Context length possible under a VRAM budget at a given quantization.
///
/// Delegates to `context_fit_with_weight` using estimated weight footprint.
pub fn context_fit(
    vram_mb: f64,
    gpu_util: f64,
    params_b: f64,
    quant: &str,
    kv_bpt: f64,
    overhead_mb: f64,
) -> usize {
    let w_gb = weight_gb(params_b, quant);
    context_fit_with_weight(vram_mb, gpu_util, w_gb, kv_bpt, overhead_mb)
}

/// Estimated max decode tok/s for a single GPU: memory-bandwidth bound.
/// utilization ~0.5 for single-GPU decode; bytes per token = params × bytes/param.
pub fn tokens_per_sec(bandwidth_gbs: f64, params_b: f64, quant: &str) -> f64 {
    let bytes_per_token = params_b * 1e9 * bytes_per_param(quant);
    if bytes_per_token <= 0.0 {
        return 0.0;
    }

    bandwidth_gbs * 1e9 * 0.5 / bytes_per_token
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TieredContextFit {
    pub vram_context: usize,
    pub extended_context: usize,
    pub swap_space_gb: usize,
    pub cpu_offload_gb: usize,
}

/// Multi-tier context fit estimating VRAM context, RAM overflow swap context, and weight offload.
pub fn context_fit_tiered(
    vram_total_mb: f64,
    gpu_util: f64,
    ram_usable_mb: f64,
    weight_gb: f64,
    kv_bpt: f64,
    overhead_mb: f64,
    max_context: usize,
    allow_weight_offload: bool,
) -> TieredContextFit {
    if kv_bpt <= 0.0 || max_context == 0 {
        return TieredContextFit {
            vram_context: 0,
            extended_context: 0,
            swap_space_gb: 0,
            cpu_offload_gb: 0,
        };
    }

    let usable_vram_mb = vram_total_mb * gpu_util;
    let weights_mb = weight_gb * 1024.0;
    let kv_vram_mb = usable_vram_mb - weights_mb - overhead_mb;

    if kv_vram_mb > 0.0 {
        let vram_ctx_raw = (kv_vram_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
        let vram_context = vram_ctx_raw.min(max_context);

        if vram_context >= max_context || ram_usable_mb <= 0.0 {
            return TieredContextFit {
                vram_context,
                extended_context: vram_context,
                swap_space_gb: 0,
                cpu_offload_gb: 0,
            };
        }

        let deficit_tokens = max_context - vram_context;
        let ram_tokens_possible = (ram_usable_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
        let extended_tokens = deficit_tokens.min(ram_tokens_possible);
        let extended_context = vram_context + extended_tokens;
        let swap_space_gb = if extended_tokens == 0 {
            0
        } else {
            ((extended_tokens as f64 * kv_bpt) / (1024.0 * 1024.0 * 1024.0))
                .ceil()
                .max(1.0) as usize
        };

        TieredContextFit {
            vram_context,
            extended_context,
            swap_space_gb,
            cpu_offload_gb: 0,
        }
    } else if allow_weight_offload && ram_usable_mb > 0.0 {
        let shortfall_mb = (weights_mb + overhead_mb) - usable_vram_mb;
        let cpu_offload_gb = (shortfall_mb / 1024.0).ceil() as usize;
        let offload_mb = (cpu_offload_gb * 1024) as f64;

        if ram_usable_mb > offload_mb {
            let remaining_ram_mb = ram_usable_mb - offload_mb;
            let ram_tokens = (remaining_ram_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
            if ram_tokens == 0 {
                return TieredContextFit {
                    vram_context: 0,
                    extended_context: 0,
                    swap_space_gb: 0,
                    cpu_offload_gb,
                };
            }
            let extended_context = ram_tokens.min(max_context).max(512.min(max_context));
            let swap_space_gb = ((extended_context as f64 * kv_bpt) / (1024.0 * 1024.0 * 1024.0))
                .ceil()
                .max(1.0) as usize;

            TieredContextFit {
                vram_context: 0,
                extended_context,
                swap_space_gb,
                cpu_offload_gb,
            }
        } else {
            TieredContextFit {
                vram_context: 0,
                extended_context: 0,
                swap_space_gb: 0,
                cpu_offload_gb,
            }
        }
    } else {
        TieredContextFit {
            vram_context: 0,
            extended_context: 0,
            swap_space_gb: 0,
            cpu_offload_gb: 0,
        }
    }
}

/// Multi-tier decode tokens/second calculation factoring in swap penalties and CPU weight offloading.
pub fn tokens_per_sec_tiered(
    gpu_bandwidth_gbs: f64,
    _ram_bandwidth_gbs: f64,
    params_b: f64,
    quant: &str,
    swap_space_gb: usize,
    cpu_offload_gb: usize,
    weight_gb: f64,
) -> f64 {
    if cpu_offload_gb > 0 {
        // Bandwidth bottlenecked by PCIe 4.0/5.0 bus (~20 GB/s)
        tokens_per_sec(20.0, params_b, quant)
    } else if swap_space_gb > 0 {
        let tok_s_gpu = tokens_per_sec(gpu_bandwidth_gbs, params_b, quant);
        let total_mem = weight_gb + swap_space_gb as f64;
        if total_mem > 0.0 {
            tok_s_gpu * (1.0 - 0.20 * (swap_space_gb as f64 / total_mem))
        } else {
            tok_s_gpu
        }
    } else {
        tokens_per_sec(gpu_bandwidth_gbs, params_b, quant)
    }
}

/// Map a GPU name → memory bandwidth in GB/s + whether it was a known entry.
/// Values are nominal GDDR/GDDR6X/GDDR7 peak and intentionally conservative.
pub fn gpu_bandwidth(name: &str) -> (f64, bool) {
    let n = name.to_lowercase();
    let find = |needle: &str| n.contains(needle);
    // Blackwell desktop (GDDR7)
    if find("5090") && find("laptop") {
        return (1418.0, true);
    }
    if find("5090") {
        return (1792.0, true);
    }
    if find("5080") && find("laptop") {
        return (960.0, true);
    }
    if find("5080") {
        return (960.0, true);
    }
    if find("5070 ti") && find("laptop") {
        return (672.0, true);
    }
    if find("5070 ti") {
        return (896.0, true);
    }
    if find("5070") {
        return (448.0, true);
    }
    if find("5060 ti") {
        return (448.0, true);
    }
    if find("5060") {
        return (288.0, true);
    }
    // Ada Lovelace
    if find("4090") {
        return (1008.0, true);
    }
    if find("4080") && find("super") {
        return (736.0, true);
    }
    if find("4080") {
        return (716.0, true);
    }
    if find("4070 ti") && find("super") {
        return (672.0, true);
    }
    if find("4070 ti") {
        return (504.0, true);
    }
    if find("4070 super") {
        return (504.0, true);
    }
    if find("4070") {
        return (504.0, true);
    }
    if find("4060 ti") {
        return (288.0, true);
    }
    if find("4060") {
        return (272.0, true);
    }
    // Ampere
    if find("3090 ti") {
        return (1008.0, true);
    }
    if find("3090") {
        return (936.0, true);
    }
    if find("3080 ti") {
        return (912.0, true);
    }
    if find("3080") {
        return (760.0, true);
    }
    if find("3070 ti") {
        return (608.0, true);
    }
    if find("3070") {
        return (448.0, true);
    }
    if find("3060 ti") {
        return (448.0, true);
    }
    if find("3060") {
        return (360.0, true);
    }
    if find("3050") {
        return (224.0, true);
    }
    // Mildest fallback: something NVIDIA-like but unknown
    if find("nvidia")
        || find("geforce")
        || find("rtx")
        || find("quadro")
        || find("tesla")
        || find("a100")
        || find("h100")
    {
        return (700.0, false);
    }
    (700.0, false)
}

/// Where the reported context length came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ContextSource {
    /// Read directly from config.json
    Config,
    /// Family-based default because config.json was missing the key
    Family,
    /// No information at all — generic default
    Default,
}

/// Walk a HF `config.json` for a declared context length.
/// Checks the common keys and recurses into nested model configs
/// (`text_config`, `config`, `model_config`).
pub fn parse_context(config: &serde_json::Value) -> (usize, ContextSource) {
    let keys = [
        "max_position_embeddings",
        "model_max_length",
        "n_positions",
        "max_seq_len",
        "max_seq_length",
        "max_sequence_length",
        "window",
    ];
    for key in keys {
        if let Some(v) = config.get(key) {
            if let Some(n) = usize_of(v) {
                // Apply RoPE scaling if present
                let scaled = apply_rope_scaling(config, n);
                return (scaled, ContextSource::Config);
            }
        }
    }
    // Check nested configs: text_config, config, model_config
    for sub_key in ["text_config", "config", "model_config"] {
        if let Some(sub) = config.get(sub_key) {
            if !sub.is_null() {
                let (n, src) = parse_context(sub);
                if src != ContextSource::Default && n > 0 {
                    return (n, src);
                }
            }
        }
    }
    (0, ContextSource::Default)
}

fn apply_rope_scaling(config: &serde_json::Value, base_ctx: usize) -> usize {
    if let Some(rope) = config.get("rope_scaling") {
        let rope_type = rope
            .get("type")
            .or_else(|| rope.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match rope_type {
            "yarn" | "linear" | "dynamic" => {
                if let Some(factor) = rope.get("factor").and_then(|v| v.as_f64()) {
                    if factor > 1.0 {
                        return (base_ctx as f64 * factor) as usize;
                    }
                }
            }
            _ => {}
        }
    }
    base_ctx
}

/// Parse a JSON value as usize, accepting u64 or i64.
pub fn usize_of(v: &serde_json::Value) -> Option<usize> {
    v.as_u64()
        .map(|n| n as usize)
        .or_else(|| v.as_i64().and_then(|n| usize::try_from(n).ok()))
}

/// Family-based fallback context when config.json doesn't declare one.
pub fn family_fallback(model_id: &str) -> Option<usize> {
    let m = model_id.to_lowercase();
    let hits: &[(&[&str], usize)] = &[
        (&["qwen3"], 131_072),
        (&["qwen2"], 131_072),
        (&["qwen"], 32_768),
        (&["llama3", "llama-3"], 8_192),
        (&["llama"], 4_096),
        (&["mistral"], 32_768),
        (&["mixtral"], 32_768),
        (&["gemma"], 8_192),
        (&["deepseek"], 131_072),
        (&["phi-2", "phi2"], 2_048),
        (&["phi"], 131_072),
        (&["gpt-2", "gpt2"], 1_024),
        (&["gpt-j"], 2_048),
        (&["gpt-neox"], 2_048),
        (&["falcon"], 2_048),
        (&["olmo"], 2_048),
        (&["yi"], 4_096),
        (&["baichuan"], 4_096),
        (&["internlm"], 32_768),
        (&["glm"], 32_768),
        (&["codegen"], 2_048),
        (&["codellama"], 16_384),
        (&["starcoder"], 8_192),
        (&["bloom"], 2_048),
    ];
    for (needles, ctx) in hits {
        if needles.iter().any(|nd| m.contains(nd)) {
            return Some(*ctx);
        }
    }
    None
}

/// Generic default when nothing is known (some instruct models with no
/// declared context: assume a safe 4096).
pub const DEFAULT_CONTEXT: usize = 4096;

/// Try to parse head dimension as reported by the config (some archs declare
/// `head_dim` explicitly, e.g. Qwen3, Llama 3.x).
pub fn head_dim_from_config(cfg: &serde_json::Value) -> Option<usize> {
    // Direct check
    if let Some(hd) = cfg.get("head_dim").and_then(usize_of) {
        return Some(hd);
    }
    if let Some(hd) = derive_head_dim(cfg) {
        return Some(hd);
    }
    // Nested fallback
    for sub_key in ["text_config", "config", "model_config"] {
        if let Some(sub) = cfg.get(sub_key) {
            if let Some(hd) = sub.get("head_dim").and_then(usize_of) {
                return Some(hd);
            }
            if let Some(hd) = derive_head_dim(sub) {
                return Some(hd);
            }
        }
    }
    None
}

fn derive_head_dim(cfg: &serde_json::Value) -> Option<usize> {
    let hidden = cfg.get("hidden_size").and_then(usize_of)?;
    let heads = cfg.get("num_attention_heads").and_then(usize_of)?;
    if heads > 0 {
        Some(hidden / heads)
    } else {
        None
    }
}

/// Estimate parameter count (`params_b` = billions) from HF config dims.
/// Standard transformer arithmetic: ~12·L·h²  +  L·h·inter  +  vocab·h.
/// Labels as an estimate; used when `safetensors.index.json` is unavailable.
pub fn estimate_params_from_config(cfg: &serde_json::Value) -> Option<f64> {
    let sub = cfg
        .get("text_config")
        .or_else(|| cfg.get("model_config"))
        .or_else(|| cfg.get("config"));
    let resolve = |key: &str| {
        sub.and_then(|s| s.get(key))
            .or_else(|| cfg.get(key))
            .and_then(usize_of)
    };
    let layers = resolve("num_hidden_layers")?;
    let hidden = resolve("hidden_size")?;
    let vocab = resolve("vocab_size")?;
    let inter = resolve("intermediate_size").unwrap_or(hidden * 4);
    if layers == 0 || hidden == 0 {
        return None;
    }
    let attn = 12.0 * layers as f64 * (hidden as f64).powi(2);
    let mlp = layers as f64 * hidden as f64 * inter as f64;
    let emb = vocab as f64 * hidden as f64;
    Some((attn + mlp + emb) / 1e9)
}

static PARAMS_NAME_RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
    regex_lite::Regex::new(r"(?i)(?:^|[-_ /])(\d+(?:\.\d+)?)[bB](?:[-_ /.]|$)")
        .expect("valid params regex")
});

/// Try to parse parameter count in billions from a model ID / repo name (e.g. "Qwen3.8-27B-GGUF" -> 27.0).
pub fn parse_params_from_name(name: &str) -> Option<f64> {
    let base = name.split('/').last().unwrap_or(name);
    PARAMS_NAME_RE
        .captures(base)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<f64>().ok())
}

/// Parse param count from the HF API `?expand[]=safetensors` response.
/// The `safetensors.parameters` object has per-dtype counts; sum them.
pub fn parse_params_from_safetensors_api(model_info: &serde_json::Value) -> Option<f64> {
    let params = model_info.get("safetensors")?.get("parameters")?;
    let obj = params.as_object()?;
    let total: u64 = obj.values().filter_map(|v| v.as_u64()).sum();
    if total == 0 {
        return None;
    }
    Some(total as f64 / 1e9)
}

/// Parse params from a safetensors index `metadata.total_size` (bytes),
/// dividing by the dtype's byte width. `torch_dtype` from config.json.
pub fn parse_params_from_index(
    index: &serde_json::Value,
    torch_dtype: Option<&str>,
) -> Option<f64> {
    let total = index.get("metadata")?.get("total_size")?.as_u64()? as f64;
    let divisor = match torch_dtype.map(|d| d.to_ascii_lowercase()) {
        Some(ref d) if d.contains("float32") || d.contains("float16") => 2.0_f64, // fp16 stored as 2B
        Some(ref d) if d.contains("float8") => 1.0,
        _ => 2.0, // bf16 is the overwhelming majority on HF today
    };
    Some(total / divisor / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bytes_per_param_values() {
        assert_eq!(bytes_per_param("fp16"), 2.0);
        assert_eq!(bytes_per_param("bf16"), 2.0);
        assert_eq!(bytes_per_param(""), 2.0);
        assert_eq!(bytes_per_param("fp8"), 1.0);
        assert_eq!(bytes_per_param("AWQ"), 0.55);
        assert!((bytes_per_param("gptq") - 0.55).abs() < 1e-9);
    }

    #[test]
    fn test_quant_bytes_per_param_accurate_hierarchy() {
        assert!((bytes_per_param("fp16") - 2.0).abs() < 1e-6);
        assert!((bytes_per_param("fp8") - 1.0).abs() < 1e-6);
        assert!((bytes_per_param("awq") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("gptq") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("q4_k_m") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("q8_0") - 1.05).abs() < 1e-6);
        // AWQ/Q4 (4-bit) must be strictly smaller than FP8 (8-bit) and Q5 (5-bit)
        assert!(bytes_per_param("awq") < bytes_per_param("q5_k_m"));
        assert!(bytes_per_param("awq") < bytes_per_param("fp8"));
    }

    #[test]
    fn kv_bytes_for_qwen2_5_0b5() {
        // Qwen2.5-0.5B: 24 layers, 8 kv heads (GQA 2:8? actually n_kv=8), head_dim=32
        // config: hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
        //         head_dim=64 (explicit in newer files), 24 layers
        // kv_bpt = 2 * 24 * 2 * 64 * 2 = 12288 bytes/token
        let hd = 64;
        let bpt = kv_bytes_per_token(24, 2, hd);
        assert_eq!(bpt, 12288.0);
    }

    #[test]
    fn context_parse_standard() {
        let cfg = json!({ "max_position_embeddings": 32768, "hidden_size": 896 });
        let (ctx, src) = parse_context(&cfg);
        assert_eq!(ctx, 32768);
        assert_eq!(src, ContextSource::Config);
    }

    #[test]
    fn context_parse_alternative_keys() {
        let cfg = json!({ "model_max_length": 8192 });
        let (ctx, _) = parse_context(&cfg);
        assert_eq!(ctx, 8192);
    }

    #[test]
    fn context_parse_nested_text_config() {
        let cfg = json!({ "text_config": { "max_position_embeddings": 131072 } });
        let (ctx, src) = parse_context(&cfg);
        assert_eq!(ctx, 131072);
        assert_eq!(src, ContextSource::Config);
    }

    #[test]
    fn context_missing_falls_to_default_and_family() {
        let cfg = json!({ "hidden_size": 4096 });
        let (ctx, src) = parse_context(&cfg);
        assert_eq!(src, ContextSource::Default);
        assert_eq!(ctx, 0);
        // Family fallback is separate:
        assert_eq!(family_fallback("Qwen/Qwen2.5-7B-Instruct"), Some(131_072));
        assert_eq!(family_fallback("meta-llama/Meta-Llama-3-8B"), Some(8_192));
        assert_eq!(family_fallback("benjamin/burgers"), None);
    }

    #[test]
    fn context_fit_budget() {
        // 12GB GPU, util 0.92 → 11.04 GB usable, minus 2.5GB overhead.
        // 7B params fp16 = 14GB weights → no room.
        assert_eq!(context_fit(12227.0, 0.92, 7.0, "fp16", 2048.0, 2500.0), 0);
        // 0.5B fp16 = 1GB weights; kv_bpt 12KB → (11.04-1-2.5)=7.54GB / 12KB ≈ 657k → clamp to usize
        let ctx = context_fit(12227.0, 0.92, 0.494, "fp16", 12288.0, 2500.0);
        assert!(ctx > 500_000 && ctx < 800_000);
        // int4 makes 7B fit: 7*0.55=3.85GB → 11.25-3.85-2.5=4.90GB KV (bpt for 7b: 2*32*8*128*2=131072)
        // ≈ 38451 tokens
        let bpt = kv_bytes_per_token(32, 8, 128);
        let ctx7 = context_fit(12227.0, 0.92, 7.0, "awq", bpt, 2500.0);
        assert!(ctx7 > 35000 && ctx7 < 42000);
    }

    #[test]
    fn test_context_fit_with_weight() {
        let bpt = kv_bytes_per_token(32, 8, 128);
        // 12GB total, util 0.92 = 11048 MB usable - 4500 MB weight - 2500 MB overhead = 4048 MB KV
        let ctx = context_fit_with_weight(12000.0, 0.92, 4.39, bpt, 2500.0);
        assert!(ctx > 30000);

        // When weight exceeds usable VRAM, should return 0
        let ctx_zero = context_fit_with_weight(12000.0, 0.92, 10.0, bpt, 2500.0);
        assert_eq!(ctx_zero, 0);

        // Clamps floor to 512 when KV is tiny but positive
        let ctx_floor = context_fit_with_weight(12000.0, 0.92, 8.33, bpt, 2500.0);
        assert!(ctx_floor >= 512);
    }

    #[test]
    fn tokens_per_sec_smoke() {
        // RTX 5070 Ti Laptop 672 GB/s; 0.5B fp16 → 672e9*0.5/(0.494e9*2)
        let tps = tokens_per_sec(672.0, 0.494, "fp16");
        assert!((tps - 340.0).abs() < 30.0, "tps = {tps}");
        // 7B fp8 → 672e9*0.5/7e9 ≈ 48
        let tps7 = tokens_per_sec(672.0, 7.0, "fp8");
        assert!((tps7 - 48.0).abs() < 5.0, "tps7 = {tps7}");
    }

    #[test]
    fn bandwidth_table() {
        assert_eq!(
            gpu_bandwidth("NVIDIA GeForce RTX 5070 Ti Laptop GPU"),
            (672.0, true)
        );
        assert_eq!(gpu_bandwidth("NVIDIA GeForce RTX 4090"), (1008.0, true));
        assert_eq!(gpu_bandwidth("NVIDIA RTX 5090"), (1792.0, true));
        assert_eq!(gpu_bandwidth("NVIDIA GeForce GTX 1080"), (700.0, false)); // no match → fallback
        let (bw, known) = gpu_bandwidth("AMD Radeon 7900 XTX");
        assert_eq!((bw, known), (700.0, false));
    }

    #[test]
    fn params_from_config_and_index() {
        let cfg = json!({
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "vocab_size": 151936,
            "intermediate_size": 4864
        });
        let p = estimate_params_from_config(&cfg).unwrap();
        // 24*(12*896² + 896*4864) + 151936*896 ≈ 24*(9.63e6+4.358e6)+1.361e8 ≈ 3.358e8+1.361e8 = 4.72e8 → 0.47B
        assert!((p - 0.47).abs() < 0.06, "params = {p}");
        let idx = json!({ "metadata": { "total_size": 987_656_192 } }); // 0.494B * 2B = ~0.988GB
        let pi = parse_params_from_index(&idx, Some("bf16")).unwrap();
        assert!((pi - 0.494).abs() < 0.01, "index params = {pi}");
    }

    #[test]
    fn head_dim_explicit_or_derived() {
        assert_eq!(head_dim_from_config(&json!({ "head_dim": 64 })), Some(64));
        assert_eq!(
            head_dim_from_config(&json!({ "hidden_size": 896, "num_attention_heads": 14 })),
            Some(64)
        );
        assert_eq!(head_dim_from_config(&json!({ "hidden_size": 896 })), None);
    }

    #[test]
    fn test_rope_scaling_yarn() {
        let cfg = json!({
            "max_position_embeddings": 4096,
            "rope_scaling": { "type": "yarn", "factor": 4.0 }
        });
        let (ctx, src) = parse_context(&cfg);
        assert_eq!(ctx, 16384); // 4096 * 4
        assert_eq!(src, ContextSource::Config);
    }

    #[test]
    fn test_rope_scaling_linear() {
        let cfg = json!({
            "max_position_embeddings": 8192,
            "rope_scaling": { "type": "linear", "factor": 8.0 }
        });
        let (ctx, _) = parse_context(&cfg);
        assert_eq!(ctx, 65536);
    }

    #[test]
    fn test_rope_scaling_no_factor_ignored() {
        let cfg = json!({
            "max_position_embeddings": 4096,
            "rope_scaling": { "type": "dynamic" }
        });
        let (ctx, _) = parse_context(&cfg);
        assert_eq!(ctx, 4096); // No factor → no scaling
    }

    #[test]
    fn test_nested_model_config() {
        let cfg = json!({
            "model_config": { "max_position_embeddings": 65536 }
        });
        let (ctx, src) = parse_context(&cfg);
        assert_eq!(ctx, 65536);
        assert_eq!(src, ContextSource::Config);
    }

    #[test]
    fn test_nested_dims_text_config() {
        let cfg = json!({
            "text_config": {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "head_dim": 128
            }
        });
        let hd = head_dim_from_config(&cfg);
        assert_eq!(hd, Some(128));
    }

    #[test]
    fn test_parse_params_safetensors_api() {
        let api_resp = json!({
            "safetensors": {
                "parameters": {
                    "BF16": 7615616000_u64
                },
                "total": 15231232000_u64
            }
        });
        let p = parse_params_from_safetensors_api(&api_resp);
        assert!((p.unwrap() - 7.616).abs() < 0.01);
    }

    #[test]
    fn test_parse_params_from_name() {
        assert_eq!(
            parse_params_from_name("unsloth/Qwen3.8-27B-GGUF"),
            Some(27.0)
        );
        assert_eq!(
            parse_params_from_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
            Some(8.0)
        );
        assert_eq!(
            parse_params_from_name("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
            Some(1.5)
        );
        assert_eq!(
            parse_params_from_name("Qwen/Qwen2.5-0.5B-Instruct"),
            Some(0.5)
        );
        assert_eq!(
            parse_params_from_name("TheBloke/Llama-2-70B-Chat-GGUF"),
            Some(70.0)
        );
        assert_eq!(parse_params_from_name("google/gemma-2-27b-it"), Some(27.0));
        assert_eq!(parse_params_from_name("google/gemma-2-9b"), Some(9.0));
        assert_eq!(parse_params_from_name("microsoft/phi-4"), None);
    }

    #[test]
    fn test_estimate_params_nested_text_config() {
        // Qwen3.8-27B config snippet
        let cfg = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 64,
                "vocab_size": 248320
            }
        });
        let p = estimate_params_from_config(&cfg).unwrap();
        assert!((p - 27.1).abs() < 0.5, "params = {p}");
    }

    #[test]
    fn test_gguf_bytes_per_param() {
        assert_eq!(bytes_per_param("q4_k_m"), 0.55);
        assert_eq!(bytes_per_param("Q4_0"), 0.55);
        assert_eq!(bytes_per_param("Q8_0"), 1.05);
        assert_eq!(bytes_per_param("gguf"), 0.55);
    }

    #[test]
    fn test_context_fit_tiered_vram_and_swap() {
        let bpt = kv_bytes_per_token(32, 8, 128); // 7B model KV cache rate (~131072 bytes/tok)
                                                  // 12GB GPU (util 0.92 = 11048MB), 4.4GB weights, 2500MB overhead -> ~4048MB VRAM for KV (~32k tokens)
                                                  // Model max context: 131,072. Usable RAM: 16,384 MB (16 GB)
        let res = context_fit_tiered(12000.0, 0.92, 16384.0, 4.4, bpt, 2500.0, 131072, true);
        assert!(res.vram_context > 30000 && res.vram_context < 35000);
        assert!(res.extended_context > res.vram_context);
        assert!(res.extended_context <= 131072);
        assert!(res.swap_space_gb > 0);
        assert_eq!(res.cpu_offload_gb, 0);
    }

    #[test]
    fn test_context_fit_tiered_weight_offload() {
        let bpt = kv_bytes_per_token(40, 8, 128);
        // 14B model: weight 15GB on a 12GB GPU (usable VRAM ~11GB - 2.5GB overhead = 8.5GB budget) -> weights do not fit in VRAM
        let res = context_fit_tiered(12000.0, 0.92, 32768.0, 15.0, bpt, 2500.0, 32768, true);
        assert_eq!(res.vram_context, 0);
        assert!(res.cpu_offload_gb >= 6);
        assert!(res.extended_context >= 512);
    }

    #[test]
    fn test_tokens_per_sec_tiered_pure_gpu() {
        // Pure GPU: swap_space_gb == 0 && cpu_offload_gb == 0
        let res = tokens_per_sec_tiered(504.0, 45.0, 7.0, "fp16", 0, 0, 14.0);
        let expected = tokens_per_sec(504.0, 7.0, "fp16");
        assert!((res - expected).abs() < 1e-6);
        assert!((res - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_tokens_per_sec_tiered_gpu_ram_swap() {
        // GpuRamSwap: swap_space_gb > 0 && cpu_offload_gb == 0
        // tok_s_gpu * (1.0 - 0.20 * (swap_space_gb / (weight_gb + swap_space_gb)))
        // 18.0 * (1.0 - 0.20 * (6.0 / (14.0 + 6.0))) = 18.0 * (1.0 - 0.06) = 18.0 * 0.94 = 16.92
        let res = tokens_per_sec_tiered(504.0, 45.0, 7.0, "fp16", 6, 0, 14.0);
        assert!((res - 16.92).abs() < 1e-6);
    }

    #[test]
    fn test_tokens_per_sec_tiered_cpu_offload() {
        // CpuOffload: cpu_offload_gb > 0
        // (20.0 * 1e9 * 0.5) / (params_b * 1e9 * bytes_per_param(quant))
        // (10.0) / (7.0 * 2.0) = 10.0 / 14.0 = 5.0 / 7.0
        let res = tokens_per_sec_tiered(504.0, 45.0, 7.0, "fp16", 6, 4, 14.0);
        let expected = (20.0 * 1e9 * 0.5) / (7.0 * 1e9 * 2.0);
        assert!((res - expected).abs() < 1e-6);
    }

    #[test]
    fn test_context_fit_tiered_fits_fully_in_vram() {
        let bpt = kv_bytes_per_token(16, 4, 64);
        // Small 0.5B model on 24GB GPU: easily fits max context 32768 in VRAM
        let res = context_fit_tiered(24000.0, 0.92, 16384.0, 1.0, bpt, 2500.0, 32768, true);
        assert_eq!(res.vram_context, 32768);
        assert_eq!(res.extended_context, 32768);
        assert_eq!(res.swap_space_gb, 0);
        assert_eq!(res.cpu_offload_gb, 0);
    }

    #[test]
    fn test_context_fit_tiered_zero_kv_or_max_ctx() {
        let res1 = context_fit_tiered(12000.0, 0.92, 16384.0, 4.4, 0.0, 2500.0, 32768, true);
        assert_eq!(
            res1,
            TieredContextFit {
                vram_context: 0,
                extended_context: 0,
                swap_space_gb: 0,
                cpu_offload_gb: 0
            }
        );

        let res2 = context_fit_tiered(12000.0, 0.92, 16384.0, 4.4, 131072.0, 2500.0, 0, true);
        assert_eq!(
            res2,
            TieredContextFit {
                vram_context: 0,
                extended_context: 0,
                swap_space_gb: 0,
                cpu_offload_gb: 0
            }
        );
    }

    #[test]
    fn test_context_fit_tiered_no_usable_ram() {
        let bpt = kv_bytes_per_token(32, 8, 128);
        // ram_usable_mb is 0.0 -> no swap extension possible
        let res = context_fit_tiered(12000.0, 0.92, 0.0, 4.4, bpt, 2500.0, 131072, true);
        assert!(res.vram_context > 30000);
        assert_eq!(res.extended_context, res.vram_context);
        assert_eq!(res.swap_space_gb, 0);
        assert_eq!(res.cpu_offload_gb, 0);
    }

    #[test]
    fn test_context_fit_tiered_weight_offload_disallowed() {
        let bpt = kv_bytes_per_token(40, 8, 128);
        // 15GB model on 12GB GPU with allow_weight_offload = false -> Does not fit
        let res = context_fit_tiered(12000.0, 0.92, 32768.0, 15.0, bpt, 2500.0, 32768, false);
        assert_eq!(
            res,
            TieredContextFit {
                vram_context: 0,
                extended_context: 0,
                swap_space_gb: 0,
                cpu_offload_gb: 0
            }
        );
    }

    #[test]
    fn test_context_fit_tiered_weight_offload_insufficient_ram() {
        let bpt = kv_bytes_per_token(40, 8, 128);
        // 15GB model requires ~7GB offload, but only 4GB RAM usable -> does not fit
        let res = context_fit_tiered(12000.0, 0.92, 4096.0, 15.0, bpt, 2500.0, 32768, true);
        assert_eq!(res.vram_context, 0);
        assert_eq!(res.extended_context, 0);
        assert_eq!(res.swap_space_gb, 0);
        assert!(res.cpu_offload_gb >= 6);
    }

    #[test]
    fn test_tokens_per_sec_tiered_zero_params() {
        let res = tokens_per_sec_tiered(504.0, 45.0, 0.0, "fp16", 0, 0, 0.0);
        assert_eq!(res, 0.0);
        let res_offload = tokens_per_sec_tiered(504.0, 45.0, 0.0, "fp16", 0, 2, 0.0);
        assert_eq!(res_offload, 0.0);
    }

    #[test]
    fn test_context_fit_tiered_extended_tokens_zero_swap_zero() {
        let bpt = kv_bytes_per_token(32, 8, 128); // 131072 bytes/tok
                                                  // 12GB GPU, 4.4GB weights, 2500MB overhead -> ~32k tokens VRAM context.
                                                  // ram_usable_mb is tiny (e.g. 0.01 MB), not enough for even 1 token.
        let res = context_fit_tiered(12000.0, 0.92, 0.01, 4.4, bpt, 2500.0, 131072, true);
        assert!(res.vram_context > 30000);
        assert_eq!(res.extended_context, res.vram_context);
        assert_eq!(res.swap_space_gb, 0);
        assert_eq!(res.cpu_offload_gb, 0);
    }

    #[test]
    fn test_context_fit_tiered_weight_offload_small_max_context_and_zero_ram_tokens() {
        let bpt = kv_bytes_per_token(40, 8, 128);
        // Case A: ram has only 0.0001 MB remaining above offload -> ram_tokens == 0
        // 15GB model requires ~7GB (7168MB) offload. ram_usable_mb = 7168.0001
        let res_zero_tokens =
            context_fit_tiered(12000.0, 0.92, 7168.0001, 15.0, bpt, 2500.0, 32768, true);
        assert_eq!(res_zero_tokens.extended_context, 0);
        assert_eq!(res_zero_tokens.swap_space_gb, 0);
        assert!(res_zero_tokens.cpu_offload_gb >= 6);

        // Case B: max_context is 256 (< 512)
        let res_small_ctx =
            context_fit_tiered(12000.0, 0.92, 32768.0, 15.0, bpt, 2500.0, 256, true);
        assert_eq!(res_small_ctx.extended_context, 256);
        assert!(res_small_ctx.extended_context <= 256);
    }
}
