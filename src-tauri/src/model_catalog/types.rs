use serde::{Deserialize, Serialize};

/// Publisher trust level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublisherTier {
    OfficialOrg,
    TrustedQuantizer,
    Community,
}

/// Category feeds supported by discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryCategory {
    ForYou,
    NewTrending,
    Coding,
    Reasoning,
    GeneralChat,
    SmallFast,
    Embeddings,
    Vision,
}

impl DiscoveryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ForYou => "for_you",
            Self::NewTrending => "new_trending",
            Self::Coding => "coding",
            Self::Reasoning => "reasoning",
            Self::GeneralChat => "general_chat",
            Self::SmallFast => "small_fast",
            Self::Embeddings => "embeddings",
            Self::Vision => "vision",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "new_trending" | "trending" => Self::NewTrending,
            "coding" | "code" => Self::Coding,
            "reasoning" => Self::Reasoning,
            "general_chat" | "chat" => Self::GeneralChat,
            "small_fast" | "small" => Self::SmallFast,
            "embeddings" | "embed" => Self::Embeddings,
            "vision" => Self::Vision,
            _ => Self::ForYou,
        }
    }
}

/// A normalized model summary candidate for search results and category feeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub model_id: String,
    pub name: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub trending_score: f64,
    pub params_b: Option<f64>,
    pub family: Option<String>,
    pub publisher_tier: PublisherTier,
    pub available_quants: Vec<String>,
    pub best_quant: Option<String>,
    pub fit_score: u32,
    pub is_installed: bool,
}

/// Compatible low-latency speculative decoding draft model recommendation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftModelSuggestion {
    pub draft_model_id: String,
    pub params_b: f64,
    pub quant: String,
    pub compatibility_reason: String,
}

/// Deep model metadata for model cards and configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModelDetails {
    pub model_id: String,
    pub base_model_id: Option<String>,
    pub author: String,
    pub description: Option<String>,
    /// Architectural maximum context limit supported by the model weights.
    pub native_max_context: u32,
    /// Hardware-recommended context capacity calculated for the local machine.
    pub recommended_context: u32,
    /// Currently active context length if running on a live server instance.
    pub runtime_active_context: Option<u32>,
    pub params_b: Option<f64>,
    pub num_layers: Option<u32>,
    pub num_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub publisher_tier: PublisherTier,
    pub verified_variants: Vec<String>,
    pub compatible_draft_models: Vec<DraftModelSuggestion>,
    pub reasoning_effort_supported: bool,
    pub chat_template_preview: Option<String>,
}

/// A runnable model verified on local disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledModel {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub quant: String,
    pub size_bytes: u64,
    pub local_path: String,
    pub in_use: bool,
    pub is_split: bool,
}
