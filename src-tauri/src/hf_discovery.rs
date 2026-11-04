//! Enhanced Hugging Face discovery service: live feeds, query normalization,
//! category queries, trusted publisher filtering, and multi-tier caching.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const HF_MODELS_API: &str = "https://huggingface.co/api/models";

// ---------------------------------------------------------------------------
// Query normalization and parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
            DiscoveryCategory::ForYou => "for_you",
            DiscoveryCategory::NewTrending => "new_trending",
            DiscoveryCategory::Coding => "coding",
            DiscoveryCategory::Reasoning => "reasoning",
            DiscoveryCategory::GeneralChat => "general_chat",
            DiscoveryCategory::SmallFast => "small_fast",
            DiscoveryCategory::Embeddings => "embeddings",
            DiscoveryCategory::Vision => "vision",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "new_trending" | "trending" | "new" => DiscoveryCategory::NewTrending,
            "coding" | "code" | "coder" => DiscoveryCategory::Coding,
            "reasoning" | "r1" | "reason" | "math" => DiscoveryCategory::Reasoning,
            "general_chat" | "chat" | "instruct" => DiscoveryCategory::GeneralChat,
            "small_fast" | "small" | "fast" => DiscoveryCategory::SmallFast,
            "embeddings" | "embedding" | "rag" => DiscoveryCategory::Embeddings,
            "vision" | "multimodal" | "vlm" => DiscoveryCategory::Vision,
            _ => DiscoveryCategory::ForYou,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            DiscoveryCategory::ForYou => "For You / Best Fit",
            DiscoveryCategory::NewTrending => "New & Trending",
            DiscoveryCategory::Coding => "Coding",
            DiscoveryCategory::Reasoning => "Reasoning",
            DiscoveryCategory::GeneralChat => "General Chat",
            DiscoveryCategory::SmallFast => "Small & Fast",
            DiscoveryCategory::Embeddings => "Embeddings & RAG",
            DiscoveryCategory::Vision => "Vision & Multimodal",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            DiscoveryCategory::ForYou => "Optimal balance of quality, speed, and context tailored for your hardware",
            DiscoveryCategory::NewTrending => "Recently released and trending models gaining traction across the community",
            DiscoveryCategory::Coding => "Models specialized for code generation, software engineering, and tool calling",
            DiscoveryCategory::Reasoning => "Deep reasoning, mathematical problem solving, and step-by-step logic models",
            DiscoveryCategory::GeneralChat => "Versatile conversational assistants for general instruction following",
            DiscoveryCategory::SmallFast => "High-efficiency models prioritizing low memory footprint and high tok/s",
            DiscoveryCategory::Embeddings => "Models for semantic search, vector indexing, and retrieval-augmented generation",
            DiscoveryCategory::Vision => "Vision-language models capable of analyzing and understanding images",
        }
    }

    pub fn pipeline_tag(&self) -> Option<&'static str> {
        match self {
            DiscoveryCategory::Embeddings => Some("feature-extraction"),
            DiscoveryCategory::Vision => Some("image-text-to-text"),
            _ => Some("text-generation"),
        }
    }
}

/// Parsed elements extracted from a search query.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParsedSearchQuery {
    pub raw_query: String,
    pub clean_text: String,
    pub exact_repo_id: Option<String>,
    pub is_url: bool,
    pub format_filter: Option<String>,
    pub quant_filter: Option<String>,
    pub target_task: Option<String>,
    pub max_params_b: Option<f64>,
}

/// Strip URLs, branches, anchors, and whitespace from a model ID query.
pub fn clean_repo_query(input: &str) -> String {
    let mut s = input.trim();
    if let Some(rest) = s.strip_prefix("https://") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("http://") {
        s = rest;
    }
    if let Some(rest) = s.strip_prefix("huggingface.co/") {
        s = rest;
    }
    if let Some(idx) = s.find('?') {
        s = &s[..idx];
    }
    if let Some(idx) = s.find('#') {
        s = &s[..idx];
    }
    if let Some(idx) = s.find("/tree/") {
        s = &s[..idx];
    }
    if let Some(idx) = s.find("/blob/") {
        s = &s[..idx];
    }
    s.trim_end_matches('/').to_string()
}

/// Normalize an arbitrary user search string, extracting explicit format/quant
/// and task tokens so they can act as filters rather than choking the keyword search.
pub fn parse_search_query(raw: &str) -> ParsedSearchQuery {
    let trimmed = raw.trim();
    let cleaned = clean_repo_query(trimmed);

    // 1. Check for exact repo ID (owner/model) without spaces
    let is_exact = cleaned.contains('/') && !cleaned.contains(' ') && cleaned.chars().all(|ch| {
        ch.is_ascii_alphanumeric() || matches!(ch, '/' | '-' | '_' | '.')
    });
    let is_url = trimmed.starts_with("http://") || trimmed.starts_with("https://") || trimmed.starts_with("huggingface.co/");

    if is_exact {
        return ParsedSearchQuery {
            raw_query: trimmed.to_string(),
            clean_text: cleaned.clone(),
            exact_repo_id: Some(cleaned),
            is_url,
            format_filter: None,
            quant_filter: None,
            target_task: None,
            max_params_b: None,
        };
    }

    // 2. Tokenize and extract embedded filters
    let mut format_filter = None;
    let mut quant_filter = None;
    let mut target_task = None;
    let mut max_params_b = None;
    let mut remaining_words = Vec::new();

    for word in trimmed.split_whitespace() {
        let upper = word.to_ascii_uppercase();
        let lower = word.to_ascii_lowercase();

        if upper == "GGUF" {
            format_filter = Some("GGUF".to_string());
        } else if upper == "AWQ" {
            format_filter = Some("AWQ".to_string());
        } else if upper == "GPTQ" {
            format_filter = Some("GPTQ".to_string());
        } else if upper == "FP8" {
            format_filter = Some("FP8".to_string());
        } else if upper == "BNB" || upper == "BITSANDBYTES" {
            format_filter = Some("BNB".to_string());
        } else if upper.starts_with("Q") && (upper.contains("_K_") || upper.contains("_0") || upper.contains("_1")) {
            quant_filter = Some(upper);
        } else if upper.starts_with("IQ") {
            quant_filter = Some(upper);
        } else if lower == "coding" || lower == "code" {
            target_task = Some("coding".to_string());
            remaining_words.push(word);
        } else if lower == "reasoning" || lower == "reason" {
            target_task = Some("reasoning".to_string());
            remaining_words.push(word);
        } else if lower == "vision" || lower == "multimodal" || lower == "vlm" {
            target_task = Some("vision".to_string());
            remaining_words.push(word);
        } else if lower == "embedding" || lower == "embeddings" || lower == "rag" {
            target_task = Some("embedding".to_string());
            remaining_words.push(word);
        } else if let Some(params_str) = upper.strip_suffix('B') {
            if let Ok(params) = params_str.parse::<f64>() {
                max_params_b = Some(params);
                remaining_words.push(word);
            } else {
                remaining_words.push(word);
            }
        } else {
            remaining_words.push(word);
        }
    }

    let clean_text = if remaining_words.is_empty() {
        trimmed.to_string()
    } else {
        remaining_words.join(" ")
    };

    ParsedSearchQuery {
        raw_query: trimmed.to_string(),
        clean_text,
        exact_repo_id: None,
        is_url,
        format_filter,
        quant_filter,
        target_task,
        max_params_b,
    }
}

// ---------------------------------------------------------------------------
// Hugging Face API Structures & Client
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelRecord {
    pub id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: i64,
    #[serde(default)]
    pub downloads_all_time: Option<i64>,
    #[serde(default)]
    pub likes: i64,
    #[serde(default, rename = "trendingScore")]
    pub trending_score: f64,
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub private: bool,
    #[serde(default)]
    pub gated: Option<Value>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub library_name: Option<String>,
    #[serde(default)]
    pub sha: Option<String>,
    #[serde(default)]
    pub siblings: Option<Vec<HfSibling>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSibling {
    #[serde(default)]
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Check if a model repository is restricted/gated.
pub fn is_model_gated(record: &HfModelRecord) -> bool {
    match &record.gated {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.eq_ignore_ascii_case("false"),
        _ => false,
    }
}

/// Apply authorization header if token is non-empty.
pub fn apply_auth(
    mut req: reqwest::RequestBuilder,
    token: Option<&str>,
) -> reqwest::RequestBuilder {
    if let Some(t) = token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    req
}

/// Trusted publishers and official organizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublisherTier {
    OfficialOrg,
    TrustedQuantizer,
    Community,
}

pub struct PublisherRegistry;

impl PublisherRegistry {
    pub fn get_tier(author_or_repo: &str) -> PublisherTier {
        let author = author_or_repo.split('/').next().unwrap_or(author_or_repo).to_ascii_lowercase();

        const OFFICIAL_ORGS: &[&str] = &[
            "meta-llama",
            "qwen",
            "mistralai",
            "google",
            "deepseek-ai",
            "microsoft",
            "cohereforai",
            "allenai",
            "baai",
            "tiiuae",
            "stabilityai",
            "openbmb",
            "internlm",
            "01-ai",
            "bigcode",
        ];

        const TRUSTED_QUANTIZERS: &[&str] = &[
            "bartowski",
            "unsloth",
            "quantfactory",
            "maziyarpanahi",
            "mradermacher",
            "thebloke",
            "neuralmagic",
            "hugging-quants",
            "ista-daslab",
            "city96",
            "casperhansen",
        ];

        if OFFICIAL_ORGS.contains(&author.as_str()) {
            PublisherTier::OfficialOrg
        } else if TRUSTED_QUANTIZERS.contains(&author.as_str()) {
            PublisherTier::TrustedQuantizer
        } else {
            PublisherTier::Community
        }
    }

    pub fn is_trusted(author_or_repo: &str) -> bool {
        Self::get_tier(author_or_repo) != PublisherTier::Community
    }
}

// ---------------------------------------------------------------------------
// Discovery Feed & Search Service
// ---------------------------------------------------------------------------

pub struct DiscoveryService;

impl DiscoveryService {
    /// Fetch exact repository by ID from Hugging Face Hub.
    pub async fn fetch_exact_model(
        client: &reqwest::Client,
        repo_id: &str,
        token: Option<&str>,
    ) -> Result<Option<HfModelRecord>> {
        let url = format!("{HF_MODELS_API}/{repo_id}");
        let req = apply_auth(client.get(&url), token);
        let resp = req.send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("HF API returned {status}: {body}");
        }
        let record: HfModelRecord = resp.json().await?;
        Ok(Some(record))
    }

    /// Query the Hugging Face models API with structured params.
    pub async fn query_models(
        client: &reqwest::Client,
        params: &[(&str, &str)],
        token: Option<&str>,
    ) -> Result<Vec<HfModelRecord>> {
        let mut url = reqwest::Url::parse(HF_MODELS_API)
            .map_err(|e| anyhow!("Failed to parse HF URL: {e}"))?;
        
        {
            let mut query_pairs = url.query_pairs_mut();
            for (k, v) in params {
                if !v.is_empty() {
                    query_pairs.append_pair(k, v);
                }
            }
        }

        let req = apply_auth(client.get(url), token);
        let resp = req.send().await.context("HF API request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                bail!("HF API rate limit exceeded (429). Configure a Hugging Face token in Settings to increase rate limits.");
            }
            let body = resp.text().await.unwrap_or_default();
            bail!("HF API returned {status}: {body}");
        }

        let records: Vec<HfModelRecord> = resp.json().await.context("Failed to parse HF JSON")?;
        Ok(records)
    }

    /// Discover models for a specific recommendation category.
    pub async fn fetch_category_feed(
        client: &reqwest::Client,
        category: DiscoveryCategory,
        limit: usize,
        token: Option<&str>,
    ) -> Result<Vec<HfModelRecord>> {
        let limit_str = limit.to_string();
        let mut records = Vec::new();

        match category {
            DiscoveryCategory::ForYou => {
                // Top trending + popular text generation models
                let p1 = [
                    ("pipeline_tag", "text-generation"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "text-generation"),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", "15"),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
            DiscoveryCategory::NewTrending => {
                // High trending score and recent releases
                let p_trending = [
                    ("pipeline_tag", "text-generation"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p_created = [
                    ("pipeline_tag", "text-generation"),
                    ("sort", "createdAt"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p_trending, token),
                    Self::query_models(client, &p_created, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 {
                    // Filter newly created models to ensure basic quality (has some likes or downloads)
                    let filtered = list.into_iter().filter(|m| m.likes >= 1 || m.downloads >= 5);
                    records.extend(filtered);
                }
            }
            DiscoveryCategory::Coding => {
                let p1 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "coder"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "code"),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", "15"),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
            DiscoveryCategory::Reasoning => {
                let p1 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "reasoning"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "R1"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
            DiscoveryCategory::GeneralChat => {
                let p1 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "instruct"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "text-generation"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
            DiscoveryCategory::SmallFast => {
                let p1 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "3B"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", "15"),
                ];
                let p2 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "1.5B"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", "15"),
                ];
                let p3 = [
                    ("pipeline_tag", "text-generation"),
                    ("search", "0.5B"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", "15"),
                ];
                let (r1, r2, r3) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token),
                    Self::query_models(client, &p3, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
                if let Ok(list) = r3 { records.extend(list); }
            }
            DiscoveryCategory::Embeddings => {
                let p1 = [
                    ("pipeline_tag", "feature-extraction"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "feature-extraction"),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
            DiscoveryCategory::Vision => {
                let p1 = [
                    ("pipeline_tag", "image-text-to-text"),
                    ("sort", "trendingScore"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let p2 = [
                    ("pipeline_tag", "image-text-to-text"),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", &limit_str),
                ];
                let (r1, r2) = tokio::join!(
                    Self::query_models(client, &p1, token),
                    Self::query_models(client, &p2, token)
                );
                if let Ok(list) = r1 { records.extend(list); }
                if let Ok(list) = r2 { records.extend(list); }
            }
        }

        // Deduplicate records by model ID preserving order
        let mut seen = HashSet::new();
        let mut deduplicated = Vec::with_capacity(records.len());
        for m in records {
            let id_lower = m.id.to_ascii_lowercase();
            if seen.insert(id_lower) {
                deduplicated.push(m);
            }
        }

        Ok(deduplicated)
    }

    /// Perform structured live search against Hugging Face.
    pub async fn search_models(
        client: &reqwest::Client,
        parsed: &ParsedSearchQuery,
        limit: usize,
        token: Option<&str>,
    ) -> Result<Vec<HfModelRecord>> {
        let mut results = Vec::new();
        let mut seen_ids = HashSet::new();

        // 1. If an exact repo ID was recognized, fetch it directly
        if let Some(exact_id) = &parsed.exact_repo_id {
            if let Ok(Some(record)) = Self::fetch_exact_model(client, exact_id, token).await {
                seen_ids.insert(record.id.to_ascii_lowercase());
                results.push(record);
            }
        }

        // 2. Build API search queries
        let limit_str = limit.to_string();
        let mut search_term = parsed.clean_text.clone();
        if search_term.is_empty() {
            if let Some(task) = &parsed.target_task {
                search_term = task.clone();
            }
        }

        if !search_term.is_empty() {
            let params = [
                ("search", search_term.as_str()),
                ("limit", &limit_str),
                ("sort", "trendingScore"),
                ("direction", "-1"),
            ];
            if let Ok(records) = Self::query_models(client, &params, token).await {
                for r in records {
                    if seen_ids.insert(r.id.to_ascii_lowercase()) {
                        results.push(r);
                    }
                }
            }

            // Also search sorted by downloads to capture established staple models
            let params_dl = [
                ("search", search_term.as_str()),
                ("limit", "15"),
                ("sort", "downloads"),
                ("direction", "-1"),
            ];
            if let Ok(records) = Self::query_models(client, &params_dl, token).await {
                for r in records {
                    if seen_ids.insert(r.id.to_ascii_lowercase()) {
                        results.push(r);
                    }
                }
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Multi-Tier Discovery Cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CachedFeed {
    pub records: Vec<HfModelRecord>,
    pub fetched_at: Instant,
}

pub struct DiscoveryCache {
    feeds: Mutex<HashMap<String, CachedFeed>>,
    searches: Mutex<HashMap<String, CachedFeed>>,
}

impl DiscoveryCache {
    pub fn new() -> Self {
        Self {
            feeds: Mutex::new(HashMap::new()),
            searches: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_feed(&self, category: &DiscoveryCategory, max_age: Duration) -> Option<Vec<HfModelRecord>> {
        let key = category.as_str();
        let guard = self.feeds.lock().ok()?;
        let entry = guard.get(key)?;
        if entry.fetched_at.elapsed() <= max_age {
            Some(entry.records.clone())
        } else {
            None
        }
    }

    pub fn set_feed(&self, category: &DiscoveryCategory, records: Vec<HfModelRecord>) {
        let key = category.as_str().to_string();
        if let Ok(mut guard) = self.feeds.lock() {
            guard.insert(
                key,
                CachedFeed {
                    records,
                    fetched_at: Instant::now(),
                },
            );
        }
    }

    pub fn get_search(&self, query: &str, max_age: Duration) -> Option<Vec<HfModelRecord>> {
        let guard = self.searches.lock().ok()?;
        let entry = guard.get(query)?;
        if entry.fetched_at.elapsed() <= max_age {
            Some(entry.records.clone())
        } else {
            None
        }
    }

    pub fn set_search(&self, query: &str, records: Vec<HfModelRecord>) {
        if let Ok(mut guard) = self.searches.lock() {
            guard.insert(
                query.to_string(),
                CachedFeed {
                    records,
                    fetched_at: Instant::now(),
                },
            );
        }
    }

    pub fn invalidate_all(&self) {
        if let Ok(mut guard) = self.feeds.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.searches.lock() {
            guard.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_search_query_exact_repo_id() {
        let q = parse_search_query("Qwen/Qwen2.5-Coder-7B-Instruct");
        assert_eq!(q.exact_repo_id.as_deref(), Some("Qwen/Qwen2.5-Coder-7B-Instruct"));
        assert!(!q.is_url);
    }

    #[test]
    fn test_parse_search_query_url() {
        let q = parse_search_query("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(q.exact_repo_id.as_deref(), Some("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(q.is_url);
    }

    #[test]
    fn test_parse_search_query_filters() {
        let q = parse_search_query("Qwen2.5 coding model GGUF Q4_K_M");
        assert_eq!(q.format_filter.as_deref(), Some("GGUF"));
        assert_eq!(q.quant_filter.as_deref(), Some("Q4_K_M"));
        assert_eq!(q.target_task.as_deref(), Some("coding"));
        assert!(q.clean_text.contains("Qwen2.5"));
    }

    #[test]
    fn test_publisher_tier_classification() {
        assert_eq!(PublisherRegistry::get_tier("Qwen/Qwen2.5-7B"), PublisherTier::OfficialOrg);
        assert_eq!(PublisherRegistry::get_tier("meta-llama/Llama-3.1-8B"), PublisherTier::OfficialOrg);
        assert_eq!(PublisherRegistry::get_tier("bartowski/Qwen2.5-7B-GGUF"), PublisherTier::TrustedQuantizer);
        assert_eq!(PublisherRegistry::get_tier("unsloth/Llama-3.1-8B-GGUF"), PublisherTier::TrustedQuantizer);
        assert_eq!(PublisherRegistry::get_tier("some-random-user/my-model"), PublisherTier::Community);
    }
}
