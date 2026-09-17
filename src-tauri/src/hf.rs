//! Hugging Face API: model search, stats enrichment, background model pulls.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

use crate::estimate;
use crate::state::AppState;
use tauri::Emitter;
use std::sync::Arc as StdArc;

pub const HF_API: &str = "https://huggingface.co/api/models";

#[derive(Debug, Clone, Serialize)]
pub struct HfModel {
    pub id: String,
    pub downloads: i64,
    pub likes: i64,
    pub trending_score: f64,
    pub private: bool,
    pub pipeline_tag: Option<String>,
    /// Enriched (None if HF enrichment failed or timed out)
    pub stats: Option<EnrichedStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichedStats {
    /// Parameter count in billions (estimate when index.json missing).
    pub params_b: Option<f64>,
    /// Effective context: config.json value, else family default, else generic.
    pub context: usize,
    pub context_source: &'static str,
    /// True when config.json had no context field (family/generic used).
    pub context_estimated: bool,
    pub head_dim: Option<usize>,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub torch_dtype: Option<String>,
}

/// Search HF for models matching `query`.
pub async fn search(client: &reqwest::Client, query: &str, limit: usize) -> Result<Vec<HfModel>> {
    let url = reqwest::Url::parse_with_params(HF_API, &[("search", query), ("limit", &limit.to_string())])
        .map_err(|e| anyhow!("build url: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .await
        .context("HF search request failed")?;
    if !resp.status().is_success() {
        bail!("HF search returned {}", resp.status());
    }
    let arr: Vec<Value> = resp.json().await.context("HF search JSON parse")?;
    let mut out = Vec::with_capacity(arr.len());
    for m in arr {
        let id = m
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if id.is_empty() {
            continue;
        }
        let downloads = m.get("downloads").and_then(|v| v.as_i64()).unwrap_or(0);
        let likes = m.get("likes").and_then(|v| v.as_i64()).unwrap_or(0);
        let trending = m.get("trendingScore").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let private = m.get("private").and_then(|v| v.as_bool()).unwrap_or(false);
        let pipeline = m
            .get("pipeline_tag")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.push(HfModel {
            id,
            downloads,
            likes,
            trending_score: trending,
            private,
            pipeline_tag: pipeline,
            stats: None,
        });
    }
    Ok(out)
}

/// Fetch `raw/<branch>/<file>` for a model id; None on 404 / network error.
async fn fetch_raw(client: &reqwest::Client, model_id: &str, file: &str) -> Option<Value> {
    for branch in ["main", "master"] {
        let url = format!("https://huggingface.co/{model_id}/raw/{branch}/{file}");
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return resp.json().await.ok();
            }
        }
    }
    None
}

/// Enrich one model: config.json → context/dims; index.json → params.
pub async fn enrich(
    client: &reqwest::Client,
    model_id: &str,
    cache: Option<&std::sync::Mutex<std::collections::HashMap<String, crate::state::CachedEnrichment>>>,
) -> Option<EnrichedStats> {
    // 1. Check cache first (return if TTL < 1 hour / 3600 seconds)
    if let Some(c) = cache {
        if let Ok(guard) = c.lock() {
            if let Some(entry) = guard.get(model_id) {
                if entry.fetched_at.elapsed() < std::time::Duration::from_secs(3600) {
                    return Some(entry.stats.clone());
                }
            }
        }
    }

    // 2. Fetch config.json as existing
    let cfg = fetch_raw(client, model_id, "config.json").await?;
    let (context_config, src) = estimate::parse_context(&cfg);
    let (context, context_source, context_estimated) = if context_config > 0 {
        (context_config, "config.json", false)
    } else if let Some(fam) = estimate::family_fallback(model_id) {
        (fam, "family default", true)
    } else {
        (estimate::DEFAULT_CONTEXT, "generic default", true)
    };

    let head_dim = estimate::head_dim_from_config(&cfg);
    let n_layers = cfg
        .get("num_hidden_layers")
        .and_then(usize_of)
        .or_else(|| cfg.get("text_config").and_then(|t| t.get("num_hidden_layers")).and_then(usize_of));
    let n_kv_heads = cfg
        .get("num_key_value_heads")
        .and_then(usize_of)
        .or_else(|| cfg.get("num_attention_heads").and_then(usize_of))
        .or_else(|| cfg.get("text_config").and_then(|t| t.get("num_kv_heads")).and_then(usize_of));
    let torch_dtype = cfg
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // 3. Try ?expand[]=safetensors API for params before falling back to index files
    let mut params_b = None;
    let expand_url = format!("{HF_API}/{model_id}?expand[]=safetensors");
    if let Ok(resp) = client.get(&expand_url).send().await {
        if resp.status().is_success() {
            if let Ok(info) = resp.json::<Value>().await {
                // 4. If safetensors expand succeeds, set params_b
                params_b = estimate::parse_params_from_safetensors_api(&info);
            }
        }
    }

    // 5. If still None, fall back to index files logic
    if params_b.is_none() {
        for index_file in ["safetensors.index.json", "pytorch_model.bin.index.json"] {
            if let Some(idx) = fetch_raw(client, model_id, index_file).await {
                if let Some(p) = estimate::parse_params_from_index(&idx, torch_dtype.as_deref()) {
                    params_b = Some(p);
                    break;
                }
            }
        }
    }
    if params_b.is_none() {
        params_b = estimate::estimate_params_from_config(&cfg);
    }

    let _ = src; // context_source already encodes the outcome

    let stats = EnrichedStats {
        params_b,
        context,
        context_source,
        context_estimated,
        head_dim,
        n_layers,
        n_kv_heads,
        torch_dtype,
    };

    // 6. Store in cache if cache is Some
    if let Some(c) = cache {
        if let Ok(mut guard) = c.lock() {
            guard.insert(
                model_id.to_string(),
                crate::state::CachedEnrichment {
                    stats: stats.clone(),
                    fetched_at: std::time::Instant::now(),
                },
            );
        }
    }

    Some(stats)
}

fn usize_of(v: &Value) -> Option<usize> {
    v.as_u64()
        .map(|n| n as usize)
        .or_else(|| v.as_i64().and_then(|n| usize::try_from(n).ok()))
}

// ---------------------------------------------------------------------------
// Pulling
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PullStatus {
    pub model: String,
    pub state: String, // downloading | complete | failed
    pub file: Option<String>,
    pub percent: Option<f64>,
}

/// Start a background `hf download <model_id>` in the venv, streaming progress
/// lines to the `pull-progress` event. Idempotent per model.
///
/// `app.state()` is NOT usable from the spawned thread, so we hand it a clone
/// of the pulling map Arc + the AppHandle (for emitting events).
pub fn pull_model(
    state: &StdArc<AppState>,
    app: tauri::AppHandle,
    model_id: &str,
) -> Result<()> {
    {
        let mut pulling = state.pulling.lock().unwrap();
        if *pulling.get(model_id).unwrap_or(&false) {
            bail!("already pulling {model_id}");
        }
        pulling.insert(model_id.to_string(), true);
    }
    let pulling_arc = Arc::clone(&state.pulling);

    let app = app.clone();
    let model_id = model_id.to_string();
    let distro = state.config().distro;
    let venv = state.config().venv_dir;
    let hf_token = state.config().hf_token;
    let token_ok = !hf_token.is_empty()
        && hf_token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || (c.is_ascii_punctuation() && c != '\''));
    let token_env = if token_ok {
        format!("HF_TOKEN='{hf_token}' ")
    } else {
        String::new()
    };
    let mid = model_id.clone();

    std::thread::spawn(move || {
        // Replace shell-quote hazards minimally; model ids are safe by construction.
        let maybe_cd = if venv.contains("~") || venv.starts_with('/') {
            format!("cd {}/.. && ", venv)
        } else {
            String::new()
        };
        let script = format!(
            "{} . {}/bin/activate && HF_HUB_DISABLE_TQDM=1 {}hf download {} 2>&1 || echo __HF_PULL_FAILED__",
            maybe_cd, venv, token_env, mid
        );
        let model_ev = model_id.clone();
        let app_ev = app.clone();
        let on_line = move |line: &str| {
            let _ = app_ev.emit(
                "pull-progress",
                PullStatus {
                    model: model_ev.clone(),
                    state: "downloading".into(),
                    file: Some(line.to_string()),
                    percent: None,
                },
            );
        };
        let out = crate::wsl::run_script_stream(&distro, &script, on_line);
        let state_label = if out.ok && !out.stdout.contains("__HF_PULL_FAILED__") && !out.stderr.contains("__HF_PULL_FAILED__")
        {
            "complete"
        } else {
            "failed"
        };
        let _ = app.emit(
            "pull-progress",
            PullStatus { model: model_id.clone(), state: state_label.into(), file: None, percent: None },
        );
        let mut pulling = pulling_arc.lock().unwrap();
        pulling.remove(&model_id);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CachedEnrichment;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn test_enrich_cache_hit_returns_cached_without_network() {
        let client = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let dummy_stats = EnrichedStats {
            params_b: Some(3.5),
            context: 8192,
            context_source: "config.json",
            context_estimated: false,
            head_dim: Some(64),
            n_layers: Some(32),
            n_kv_heads: Some(8),
            torch_dtype: Some("bfloat16".into()),
        };
        cache.lock().unwrap().insert(
            "dummy/model".to_string(),
            CachedEnrichment {
                stats: dummy_stats.clone(),
                fetched_at: Instant::now(),
            },
        );

        let res = enrich(&client, "dummy/model", Some(&cache)).await;
        assert!(res.is_some());
        let s = res.unwrap();
        assert_eq!(s.context, 8192);
        assert_eq!(s.params_b, Some(3.5));
    }

    #[tokio::test]
    async fn test_enrich_cache_miss_expired_ttl() {
        let client = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let dummy_stats = EnrichedStats {
            params_b: Some(3.5),
            context: 8192,
            context_source: "config.json",
            context_estimated: false,
            head_dim: Some(64),
            n_layers: Some(32),
            n_kv_heads: Some(8),
            torch_dtype: Some("bfloat16".into()),
        };
        let past = Instant::now().checked_sub(Duration::from_secs(3605)).unwrap();
        cache.lock().unwrap().insert(
            "nonexistent/model-expired-xyz".to_string(),
            CachedEnrichment {
                stats: dummy_stats,
                fetched_at: past,
            },
        );

        let res = enrich(&client, "nonexistent/model-expired-xyz", Some(&cache)).await;
        assert!(res.is_none());
    }
}