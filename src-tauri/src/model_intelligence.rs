//! Model intelligence and artifact resolution:
//! - Multi-shard GGUF grouping (merges -00001-of-00003 shards into single logical artifact)
//! - Multimodal companion (mmproj) resolution
//! - Model family and base model association
//! - Robust quantization classification across GGUF, AWQ, GPTQ, FP8, BNB, and Safetensors

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use crate::hf::{self, GgufRepoFile, QuantFormat, QuantVariant};
use crate::state::LlamaCppChannel;

/// Logical GGUF artifact option presented to the user.
/// Shards (e.g. `00001-of-00003`) are grouped together so downloading downloads all parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogicalGgufArtifact {
    pub repo_id: String,
    pub quant_label: String,
    #[serde(default)]
    pub quant: String,
    pub total_size_bytes: u64,
    pub shard_files: Vec<String>,
    #[serde(default)]
    pub files: Vec<String>,
    pub is_split: bool,
    #[serde(default)]
    pub shard_count: usize,
    #[serde(default)]
    pub has_mmproj: bool,
    #[serde(default)]
    pub mmproj_files: Vec<String>,
    pub companion_mmproj: Option<String>,
    pub required_channel: LlamaCppChannel,
}

/// Group a list of raw GGUF repository files into logical, multi-shard-aware artifacts.
pub fn group_gguf_repo_files(
    repo_id: &str,
    files: &[GgufRepoFile],
) -> Vec<LogicalGgufArtifact> {
    // 1. Separate mmproj companions
    let mut mmproj_file = None;
    for file in files {
        if file.is_mmproj {
            mmproj_file = Some(file.path.clone());
            break;
        }
    }

    // 2. Group non-auxiliary GGUF files by quantization label
    let mut groups: BTreeMap<String, Vec<&GgufRepoFile>> = BTreeMap::new();

    for file in files {
        if file.is_mmproj || hf::is_auxiliary_gguf_file(&file.path) {
            continue;
        }
        if let Some(label) = &file.quant {
            groups.entry(label.clone()).or_default().push(file);
        }
    }

    let mut artifacts = Vec::new();

    for (quant_label, mut shard_list) in groups {
        // Sort shards alphabetically so 00001 precedes 00002
        shard_list.sort_by(|a, b| a.path.cmp(&b.path));

        let total_size_bytes = shard_list.iter().map(|f| f.size_bytes).sum();
        let shard_files: Vec<String> = shard_list.iter().map(|f| f.path.clone()).collect();
        let is_split = shard_files.len() > 1;
        let shard_count = shard_files.len();
        let has_mmproj = mmproj_file.is_some();
        let mmproj_files = mmproj_file.clone().into_iter().collect();

        let required_channel = if hf::quant_requires_prism_channel(&quant_label) {
            LlamaCppChannel::Prism
        } else {
            LlamaCppChannel::Upstream
        };

        artifacts.push(LogicalGgufArtifact {
            repo_id: repo_id.to_string(),
            quant_label: quant_label.clone(),
            quant: quant_label,
            total_size_bytes,
            shard_files: shard_files.clone(),
            files: shard_files,
            is_split,
            shard_count,
            has_mmproj,
            mmproj_files,
            companion_mmproj: mmproj_file.clone(),
            required_channel,
        });
    }

    // Stable ordering: larger/higher precision first
    artifacts.sort_by(|a, b| b.total_size_bytes.cmp(&a.total_size_bytes));
    artifacts
}

/// Convert logical GGUF artifacts into QuantVariants for scoring and UI display.
pub fn artifacts_to_quant_variants(artifacts: &[LogicalGgufArtifact]) -> Vec<QuantVariant> {
    artifacts
        .iter()
        .map(|art| {
            let primary_file = art.shard_files.first().cloned();
            QuantVariant {
                repo_id: art.repo_id.clone(),
                format: QuantFormat::GGUF,
                label: art.quant_label.clone(),
                weight_bytes: Some(art.total_size_bytes),
                params_b: None,
                gguf_file: primary_file,
                vllm_native: false,
                required_channel: art.required_channel,
            }
        })
        .collect()
}

/// Model family classification derived from model ID and architecture tags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFamilyInfo {
    pub family_name: String,
    pub base_model_id: Option<String>,
    pub parameter_class: Option<String>,
    pub is_multimodal: bool,
    pub is_reasoning: bool,
    pub is_coding: bool,
    pub is_embedding: bool,
}

pub fn infer_model_family(model_id: &str, pipeline_tag: Option<&str>, tags: &[String]) -> ModelFamilyInfo {
    let lower = model_id.to_ascii_lowercase();

    let is_embedding = pipeline_tag == Some("feature-extraction")
        || lower.contains("embed")
        || lower.contains("bge")
        || lower.contains("gte")
        || tags.iter().any(|t| t == "feature-extraction" || t == "sentence-similarity");

    let is_multimodal = pipeline_tag == Some("image-text-to-text")
        || lower.contains("vision")
        || lower.contains("-vl")
        || lower.contains("pixtral")
        || lower.contains("llava")
        || tags.iter().any(|t| t == "image-text-to-text" || t == "multimodal");

    let is_coding = lower.contains("coder")
        || lower.contains("code")
        || lower.contains("starcoder")
        || lower.contains("deepseek-coder");

    let is_reasoning = lower.contains("r1")
        || lower.contains("reason")
        || lower.contains("qwq")
        || lower.contains("thinking");

    let family_name = if lower.contains("qwen2.5") {
        "Qwen2.5".to_string()
    } else if lower.contains("qwen") {
        "Qwen".to_string()
    } else if lower.contains("llama-3.3") {
        "Llama-3.3".to_string()
    } else if lower.contains("llama-3.2") {
        "Llama-3.2".to_string()
    } else if lower.contains("llama-3.1") {
        "Llama-3.1".to_string()
    } else if lower.contains("llama-3") {
        "Llama-3".to_string()
    } else if lower.contains("deepseek") {
        "DeepSeek".to_string()
    } else if lower.contains("gemma-2") {
        "Gemma-2".to_string()
    } else if lower.contains("gemma") {
        "Gemma".to_string()
    } else if lower.contains("mistral") {
        "Mistral".to_string()
    } else if lower.contains("phi-3.5") || lower.contains("phi-3") {
        "Phi-3".to_string()
    } else if is_embedding {
        "Embeddings".to_string()
    } else {
        model_id.split('/').last().unwrap_or(model_id).to_string()
    };

    // Extract parameter class (e.g. "7B", "8B", "14B", "70B")
    let parameter_class = extract_param_token(&lower);

    // Try to infer canonical base model for quantized repos
    let base_model_id = infer_base_model_id(model_id);

    ModelFamilyInfo {
        family_name,
        base_model_id,
        parameter_class,
        is_multimodal,
        is_reasoning,
        is_coding,
        is_embedding,
    }
}

fn extract_param_token(lower: &str) -> Option<String> {
    for token in ["0.5b", "1.5b", "3b", "4b", "7b", "8b", "9b", "12b", "14b", "27b", "32b", "70b", "72b"] {
        if lower.contains(token) {
            return Some(token.to_ascii_uppercase());
        }
    }
    None
}

/// Infer canonical base repository name from a quant repo name.
pub fn infer_base_model_id(repo_id: &str) -> Option<String> {
    let lower = repo_id.to_ascii_lowercase();
    if !lower.contains("-gguf") && !lower.contains("-awq") && !lower.contains("-gptq") && !lower.contains("-fp8") {
        return None;
    }

    let repo_name = repo_id.split('/').last().unwrap_or(repo_id);
    let mut base_name = repo_name.to_string();

    for suffix in [
        "-GGUF", "-gguf", "-Gguf",
        "-AWQ", "-awq",
        "-GPTQ-Int4", "-GPTQ", "-gptq",
        "-FP8-dynamic", "-FP8", "-fp8",
        "-bnb-4bit",
    ] {
        if let Some(stripped) = base_name.strip_suffix(suffix) {
            base_name = stripped.to_string();
            break;
        }
    }

    let org = repo_id.split('/').next().unwrap_or("");
    // If the quantizer is community, map to likely official creator
    let official_org = if base_name.to_ascii_lowercase().starts_with("qwen") {
        "Qwen"
    } else if base_name.to_ascii_lowercase().starts_with("llama") {
        "meta-llama"
    } else if base_name.to_ascii_lowercase().starts_with("mistral") {
        "mistralai"
    } else if base_name.to_ascii_lowercase().starts_with("deepseek") {
        "deepseek-ai"
    } else if base_name.to_ascii_lowercase().starts_with("gemma") {
        "google"
    } else {
        org
    };

    Some(format!("{official_org}/{base_name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_gguf_shards() {
        let files = vec![
            GgufRepoFile {
                path: "model-Q4_K_M-00001-of-00003.gguf".to_string(),
                size_bytes: 5_000_000_000,
                is_mmproj: false,
                quant: Some("Q4_K_M".to_string()),
            },
            GgufRepoFile {
                path: "model-Q4_K_M-00002-of-00003.gguf".to_string(),
                size_bytes: 5_000_000_000,
                is_mmproj: false,
                quant: Some("Q4_K_M".to_string()),
            },
            GgufRepoFile {
                path: "model-Q4_K_M-00003-of-00003.gguf".to_string(),
                size_bytes: 4_500_000_000,
                is_mmproj: false,
                quant: Some("Q4_K_M".to_string()),
            },
            GgufRepoFile {
                path: "mmproj-f16.gguf".to_string(),
                size_bytes: 800_000_000,
                is_mmproj: true,
                quant: None,
            },
        ];

        let artifacts = group_gguf_repo_files("test/repo", &files);
        assert_eq!(artifacts.len(), 1);
        let q4 = &artifacts[0];
        assert_eq!(q4.quant_label, "Q4_K_M");
        assert_eq!(q4.quant, "Q4_K_M");
        assert_eq!(q4.shard_files.len(), 3);
        assert_eq!(q4.files.len(), 3);
        assert_eq!(q4.shard_count, 3);
        assert_eq!(q4.total_size_bytes, 14_500_000_000);
        assert!(q4.is_split);
        assert!(q4.has_mmproj);
        assert_eq!(q4.companion_mmproj.as_deref(), Some("mmproj-f16.gguf"));
    }

    #[test]
    fn test_infer_base_model_id() {
        let base = infer_base_model_id("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(base.as_deref(), Some("Qwen/Qwen2.5-Coder-7B-Instruct"));
    }
}
