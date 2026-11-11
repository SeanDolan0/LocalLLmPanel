use crate::model_catalog::types::{DraftModelSuggestion, PublisherTier};

/// Categorize publisher into trust tier based on organization reputation.
pub fn classify_publisher(author: &str) -> PublisherTier {
    let a = author.to_ascii_lowercase();
    match a.as_str() {
        "meta-llama" | "qwen" | "mistralai" | "deepseek-ai" | "google" | "microsoft"
        | "tiiuae" | "cohere" | "internlm" | "01-ai" | "allenai" | "anthropic" => {
            PublisherTier::OfficialOrg
        }
        "bartowski" | "unsloth" | "quantfactory" | "thebloke" | "neuralmagic"
        | "hugging-quants" | "ista-daslab" | "mradermacher" | "city96" => {
            PublisherTier::TrustedQuantizer
        }
        _ => PublisherTier::Community,
    }
}

/// Infer canonical model family from model identifier.
pub fn infer_model_family(model_id: &str) -> Option<String> {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("llama-3") || lower.contains("llama3") {
        Some("Llama-3".to_string())
    } else if lower.contains("qwen2.5") || lower.contains("qwen-2.5") {
        Some("Qwen-2.5".to_string())
    } else if lower.contains("qwen2") || lower.contains("qwen-2") {
        Some("Qwen-2".to_string())
    } else if lower.contains("deepseek-r1") || lower.contains("deepseek_r1") {
        Some("DeepSeek-R1".to_string())
    } else if lower.contains("deepseek-v3") {
        Some("DeepSeek-V3".to_string())
    } else if lower.contains("gemma-2") || lower.contains("gemma2") {
        Some("Gemma-2".to_string())
    } else if lower.contains("mistral") {
        Some("Mistral".to_string())
    } else if lower.contains("phi-3") || lower.contains("phi-4") {
        Some("Phi".to_string())
    } else {
        None
    }
}

/// Strip quant suffixes to infer the underlying base model.
pub fn infer_base_model_id(model_id: &str) -> Option<String> {
    let lower = model_id.to_ascii_lowercase();
    if !lower.ends_with("-gguf")
        && !lower.ends_with("-awq")
        && !lower.ends_with("-gptq")
        && !lower.ends_with("-fp8")
        && !lower.contains("gguf")
    {
        return None;
    }

    let repo_name = model_id.split(['/', '\\']).last().unwrap_or(model_id);
    let mut base = repo_name.to_string();
    for suffix in ["-GGUF", "-gguf", "-AWQ", "-awq", "-GPTQ", "-gptq", "-FP8", "-fp8"] {
        if let Some(stripped) = base.strip_suffix(suffix) {
            base = stripped.to_string();
        }
    }

    // Map common base model families
    if base.to_ascii_lowercase().contains("llama-3.1-8b-instruct") {
        Some("meta-llama/Llama-3.1-8B-Instruct".to_string())
    } else if base.to_ascii_lowercase().contains("qwen2.5-7b-instruct") {
        Some("Qwen/Qwen2.5-7B-Instruct".to_string())
    } else if base.to_ascii_lowercase().contains("qwen2.5-coder-7b-instruct") {
        Some("Qwen/Qwen2.5-Coder-7B-Instruct".to_string())
    } else {
        None
    }
}

/// Recommend speculative decoding draft models based on architecture and parameter scale.
pub fn recommend_draft_models(model_id: &str, params_b: Option<f64>) -> Vec<DraftModelSuggestion> {
    let family = infer_model_family(model_id);
    let mut suggestions = Vec::new();

    match family.as_deref() {
        Some("Llama-3") => {
            if params_b.map_or(false, |p| p >= 60.0) {
                suggestions.push(DraftModelSuggestion {
                    draft_model_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                    params_b: 8.0,
                    quant: "Q4_K_M".to_string(),
                    compatibility_reason: "High token acceptance: identical tokenizer and attention architecture."
                        .to_string(),
                });
            } else if params_b.map_or(false, |p| p >= 7.0 && p <= 10.0) {
                suggestions.push(DraftModelSuggestion {
                    draft_model_id: "meta-llama/Llama-3.2-1B-Instruct".to_string(),
                    params_b: 1.0,
                    quant: "Q4_K_M".to_string(),
                    compatibility_reason: "Ultra-fast low-latency draft model for 8B inference.".to_string(),
                });
            }
        }
        Some("Qwen-2.5") => {
            if params_b.map_or(false, |p| p >= 30.0) {
                suggestions.push(DraftModelSuggestion {
                    draft_model_id: "Qwen/Qwen2.5-7B-Instruct".to_string(),
                    params_b: 7.0,
                    quant: "Q4_K_M".to_string(),
                    compatibility_reason: "Shares vocabulary and RoPE base frequency.".to_string(),
                });
                suggestions.push(DraftModelSuggestion {
                    draft_model_id: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
                    params_b: 1.5,
                    quant: "Q4_K_M".to_string(),
                    compatibility_reason: "Lightweight draft model offering high throughput speedups.".to_string(),
                });
            } else if params_b.map_or(false, |p| p >= 6.0 && p <= 16.0) {
                suggestions.push(DraftModelSuggestion {
                    draft_model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
                    params_b: 0.5,
                    quant: "Q4_K_M".to_string(),
                    compatibility_reason: "Low memory draft model (< 0.5 GB VRAM).".to_string(),
                });
            }
        }
        Some("DeepSeek-R1") => {
            suggestions.push(DraftModelSuggestion {
                draft_model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B".to_string(),
                params_b: 1.5,
                quant: "Q4_K_M".to_string(),
                compatibility_reason: "Distilled reasoning draft model sharing Qwen tokenizer.".to_string(),
            });
        }
        _ => {}
    }

    suggestions
}

/// Detect whether chat template supports explicit reasoning effort (e.g. `<think>` tokens).
pub fn detect_reasoning_effort(template_or_id: &str) -> bool {
    let lower = template_or_id.to_ascii_lowercase();
    lower.contains("<think>")
        || lower.contains("deepseek-r1")
        || lower.contains("qwq")
        || lower.contains("reasoning_effort")
        || lower.contains("thought")
}
