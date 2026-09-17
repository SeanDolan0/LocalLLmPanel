//! Heuristic estimators: context window, VRAM context-fit, max decode tok/s.
//!
//! All functions here are pure and unit-tested. Outputs are explicitly
//! estimates — measured data (from vLLM `/metrics`) always wins in the UI.

use serde::Serialize;

/// Bytes of memory per parameter for a given quantization.
/// fp16/bf16 = 2, fp8 = 1, int4 (AWQ/GPTQ) ≈ 1.1 (weights + scale overhead).
pub fn bytes_per_param(quant: &str) -> f64 {
    let q = quant.to_ascii_lowercase();
    match q.as_str() {
        "fp8" | "int8" => 1.0,
        "awq" | "gptq" | "int4" => 1.1,
        _ => 2.0, // fp16 / bf16 / unset
    }
}

/// KV-cache bytes per single token-position, for fp16 cache:
/// `2 (K+V) × n_layers × n_kv_heads × head_dim × 2 bytes`.
pub fn kv_bytes_per_token(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> f64 {
    2.0 * n_layers as f64 * n_kv_heads as f64 * head_dim as f64 * 2.0
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

/// Map a GPU name → memory bandwidth in GB/s + whether it was a known entry.
/// Values are nominal GDDR/GDDR6X/GDDR7 peak and intentionally conservative.
pub fn gpu_bandwidth(name: &str) -> (f64, bool) {
    let n = name.to_lowercase();
    let find = |needle: &str| n.contains(needle);
    // Blackwell desktop (GDDR7)
    if find("5090") && find("laptop") { return (1418.0, true); }
    if find("5090") { return (1792.0, true); }
    if find("5080") && find("laptop") { return (960.0, true); }
    if find("5080") { return (960.0, true); }
    if find("5070 ti") && find("laptop") { return (672.0, true); }
    if find("5070 ti") { return (896.0, true); }
    if find("5070") { return (448.0, true); }
    if find("5060 ti") { return (448.0, true); }
    if find("5060") { return (288.0, true); }
    // Ada Lovelace
    if find("4090") { return (1008.0, true); }
    if find("4080") && find("super") { return (736.0, true); }
    if find("4080") { return (716.0, true); }
    if find("4070 ti") && find("super") { return (672.0, true); }
    if find("4070 ti") { return (504.0, true); }
    if find("4070 super") { return (504.0, true); }
    if find("4070") { return (504.0, true); }
    if find("4060 ti") { return (288.0, true); }
    if find("4060") { return (272.0, true); }
    // Ampere
    if find("3090 ti") { return (1008.0, true); }
    if find("3090") { return (936.0, true); }
    if find("3080 ti") { return (912.0, true); }
    if find("3080") { return (760.0, true); }
    if find("3070 ti") { return (608.0, true); }
    if find("3070") { return (448.0, true); }
    if find("3060 ti") { return (448.0, true); }
    if find("3060") { return (360.0, true); }
    if find("3050") { return (224.0, true); }
    // Mildest fallback: something NVIDIA-like but unknown
    if find("nvidia") || find("geforce") || find("rtx") || find("quadro") || find("tesla") || find("a100") || find("h100") {
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
            if let Some(n) = as_usize(v) {
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
        let rope_type = rope.get("type")
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

fn as_usize(v: &serde_json::Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize).or_else(|| v.as_i64().and_then(|n| usize::try_from(n).ok()))
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
    if let Some(hd) = cfg.get("head_dim").and_then(as_usize) {
        return Some(hd);
    }
    if let Some(hd) = derive_head_dim(cfg) {
        return Some(hd);
    }
    // Nested fallback
    for sub_key in ["text_config", "config", "model_config"] {
        if let Some(sub) = cfg.get(sub_key) {
            if let Some(hd) = sub.get("head_dim").and_then(as_usize) {
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
    let hidden = cfg.get("hidden_size").and_then(as_usize)?;
    let heads = cfg.get("num_attention_heads").and_then(as_usize)?;
    if heads > 0 { Some(hidden / heads) } else { None }
}

/// Estimate parameter count (`params_b` = billions) from HF config dims.
/// Standard transformer arithmetic: ~12·L·h²  +  L·h·inter  +  vocab·h.
/// Labels as an estimate; used when `safetensors.index.json` is unavailable.
pub fn estimate_params_from_config(cfg: &serde_json::Value) -> Option<f64> {
    let layers = cfg.get("num_hidden_layers").and_then(as_usize)?;
    let hidden = cfg.get("hidden_size").and_then(as_usize)?;
    let vocab = cfg.get("vocab_size").and_then(as_usize)?;
    let inter = cfg
        .get("intermediate_size")
        .and_then(as_usize)
        .unwrap_or(hidden * 4);
    if layers == 0 || hidden == 0 {
        return None;
    }
    let attn = 12.0 * layers as f64 * (hidden as f64).powi(2);
    let mlp = layers as f64 * hidden as f64 * inter as f64;
    let emb = vocab as f64 * hidden as f64;
    Some((attn + mlp + emb) / 1e9)
}

/// Parse param count from the HF API `?expand[]=safetensors` response.
/// The `safetensors.parameters` object has per-dtype counts; sum them.
pub fn parse_params_from_safetensors_api(model_info: &serde_json::Value) -> Option<f64> {
    let params = model_info.get("safetensors")?.get("parameters")?;
    let obj = params.as_object()?;
    let total: u64 = obj.values().filter_map(|v| v.as_u64()).sum();
    if total == 0 { return None; }
    Some(total as f64 / 1e9)
}

/// Parse params from a safetensors index `metadata.total_size` (bytes),
/// dividing by the dtype's byte width. `torch_dtype` from config.json.
pub fn parse_params_from_index(index: &serde_json::Value, torch_dtype: Option<&str>) -> Option<f64> {
    let total = index
        .get("metadata")?
        .get("total_size")?
        .as_u64()? as f64;
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
        assert_eq!(bytes_per_param("AWQ"), 1.1);
        assert!((bytes_per_param("gptq") - 1.1).abs() < 1e-9);
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
        // int4 makes 7B fit: 7*1.1=7.7GB → 11.04-7.7-2.5=0.84GB/2KB(bpt for 7b: 2*32*8*128*2=131072)
        // ≈ 6726 tokens
        let bpt = kv_bytes_per_token(32, 8, 128);
        let ctx7 = context_fit(12227.0, 0.92, 7.0, "awq", bpt, 2500.0);
        assert!(ctx7 > 6000 && ctx7 < 8000);
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
        assert_eq!(gpu_bandwidth("NVIDIA GeForce RTX 5070 Ti Laptop GPU"), (672.0, true));
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
}