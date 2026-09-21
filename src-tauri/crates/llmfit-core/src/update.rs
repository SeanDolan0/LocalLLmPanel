//! Cache helpers for previously-fetched HuggingFace model lists.
//!
//! The cache lives in `~/.llmfit/hf_models_cache.json` (Linux/macOS) or
//! `%APPDATA%\llmfit\hf_models_cache.json` (Windows) and is merged with the
//! embedded model list each time `ModelDatabase::new()` is called.

use crate::models::LlmModel;
use std::path::PathBuf;

/// Bump this when the `LlmModel` schema changes in a breaking way.
/// A cache written by an older version will be discarded.
const CACHE_VERSION: u32 = 4;

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEnvelope {
    version: u32,
    models: Vec<LlmModel>,
}

/// Returns the llmfit data directory.
/// Uses the platform-appropriate data directory via the `dirs` crate.
pub fn cache_dir() -> Option<PathBuf> {
    Some(dirs::data_dir()?.join("llmfit"))
}

/// Full path to the cached model list JSON file.
pub fn cache_file() -> Option<PathBuf> {
    Some(cache_dir()?.join("hf_models_cache.json"))
}

/// Load any previously cached models.
///
/// Returns an empty vec if the cache is missing, corrupt, or was written by
/// a different schema version.
pub fn load_cache() -> Vec<LlmModel> {
    let path = match cache_file() {
        Some(p) if p.exists() => p,
        _ => return vec![],
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    match serde_json::from_str::<CacheEnvelope>(&content) {
        Ok(env) if env.version == CACHE_VERSION => env.models,
        _ => vec![],
    }
}