//! Hugging Face API: model search, stats enrichment, background model pulls.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

use crate::estimate;
use crate::state::AppState;
use std::sync::Arc as StdArc;
use tauri::Emitter;

pub const HF_API: &str = "https://huggingface.co/api/models";

#[derive(Debug, Clone, Serialize)]
pub struct GgufRepoFile {
    pub path: String,
    pub size_bytes: u64,
    pub is_mmproj: bool,
}

pub async fn list_gguf_repo_files(
    client: &reqwest::Client,
    repo_id: &str,
    token: Option<&str>,
) -> Result<Vec<GgufRepoFile>> {
    let url = format!("{HF_API}/{repo_id}/tree/main");
    let mut request = client.get(url);
    if let Some(token) = token.filter(|t| !t.trim().is_empty()) {
        request = request.bearer_auth(token);
    }
    let values: Vec<Value> = request
        .query(&[("recursive", "false")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(values
        .into_iter()
        .filter_map(|v| {
            let path = v.get("path")?.as_str()?.to_string();
            if !path.to_ascii_lowercase().ends_with(".gguf") {
                return None;
            }
            Some(GgufRepoFile {
                is_mmproj: path.to_ascii_lowercase().starts_with("mmproj"),
                size_bytes: v.get("size").and_then(Value::as_u64).unwrap_or(0),
                path,
            })
        })
        .collect())
}

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

pub fn apply_auth(
    mut req: reqwest::RequestBuilder,
    token: Option<&str>,
) -> reqwest::RequestBuilder {
    if let Some(t) = token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    req
}

pub fn clean_model_query(input: &str) -> String {
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

/// Search HF for models matching `query`.
pub async fn search(
    client: &reqwest::Client,
    raw_query: &str,
    limit: usize,
    token: Option<&str>,
) -> Result<Vec<HfModel>> {
    let query = clean_model_query(raw_query);
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();

    // If query looks like an exact repo (e.g. "org/model"), try fetching it directly first
    if query.contains('/') && !query.contains(' ') {
        let exact_url = format!("{HF_API}/{query}");
        if let Ok(resp) = apply_auth(client.get(&exact_url), token).send().await {
            if resp.status().is_success() {
                if let Ok(m) = resp.json::<Value>().await {
                    if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                        let downloads = m.get("downloads").and_then(|v| v.as_i64()).unwrap_or(0);
                        let likes = m.get("likes").and_then(|v| v.as_i64()).unwrap_or(0);
                        let trending = m
                            .get("trendingScore")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let private = m.get("private").and_then(|v| v.as_bool()).unwrap_or(false);
                        let pipeline = m
                            .get("pipeline_tag")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        out.push(HfModel {
                            id: id.to_string(),
                            downloads,
                            likes,
                            trending_score: trending,
                            private,
                            pipeline_tag: pipeline,
                            stats: None,
                        });
                    }
                }
            }
        }
    }

    let url = reqwest::Url::parse_with_params(
        HF_API,
        &[("search", query.as_str()), ("limit", &limit.to_string())],
    )
    .map_err(|e| anyhow!("build url: {e}"))?;
    let resp = apply_auth(client.get(url), token)
        .send()
        .await
        .context("HF search request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            bail!("HF API rate limit exceeded (429 Too Many Requests). If you haven't added a Hugging Face token, please configure one in Settings to increase your quota.");
        }
        bail!("HF search returned {status}: {body}");
    }
    let arr: Vec<Value> = resp.json().await.context("HF search JSON parse")?;
    for m in arr {
        let id = m
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if id.is_empty() || out.iter().any(|existing: &HfModel| existing.id == id) {
            continue;
        }
        let downloads = m.get("downloads").and_then(|v| v.as_i64()).unwrap_or(0);
        let likes = m.get("likes").and_then(|v| v.as_i64()).unwrap_or(0);
        let trending = m
            .get("trendingScore")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
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
async fn fetch_raw(
    client: &reqwest::Client,
    model_id: &str,
    file: &str,
    token: Option<&str>,
) -> Option<Value> {
    for branch in ["main", "master"] {
        let url = format!("https://huggingface.co/{model_id}/raw/{branch}/{file}");
        if let Ok(resp) = apply_auth(client.get(&url), token).send().await {
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
    cache: Option<
        &std::sync::Mutex<std::collections::HashMap<String, crate::state::CachedEnrichment>>,
    >,
    token: Option<&str>,
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

    // 2. Fetch config.json from model or base model
    let mut cfg = fetch_raw(client, model_id, "config.json", token).await;
    if cfg.is_none() {
        let base_name = extract_base_name(model_id);
        let clean_base = base_name
            .trim_end_matches("-GGUF")
            .trim_end_matches("-gguf");
        let org = model_id.split('/').next().unwrap_or("");
        let candidates = [format!("{org}/{clean_base}"), clean_base.to_string()];
        for cand in candidates {
            if cand != model_id {
                if let Some(c) = fetch_raw(client, &cand, "config.json", token).await {
                    cfg = Some(c);
                    break;
                }
            }
        }
    }

    let (
        params_b,
        context,
        context_source,
        context_estimated,
        head_dim,
        n_layers,
        n_kv_heads,
        torch_dtype,
    ) = if let Some(cfg) = &cfg {
        let (context_config, _src) = estimate::parse_context(cfg);
        let (context, context_source, context_estimated) = if context_config > 0 {
            (context_config, "config.json", false)
        } else if let Some(fam) = estimate::family_fallback(model_id) {
            (fam, "family default", true)
        } else {
            (estimate::DEFAULT_CONTEXT, "generic default", true)
        };

        let head_dim = estimate::head_dim_from_config(cfg);
        let n_layers = cfg
            .get("num_hidden_layers")
            .or_else(|| {
                cfg.get("text_config")
                    .and_then(|t| t.get("num_hidden_layers"))
            })
            .and_then(estimate::usize_of);
        let n_kv_heads = cfg
            .get("num_key_value_heads")
            .or_else(|| cfg.get("num_attention_heads"))
            .or_else(|| {
                cfg.get("text_config").and_then(|t| {
                    t.get("num_key_value_heads")
                        .or_else(|| t.get("num_attention_heads"))
                        .or_else(|| t.get("num_kv_heads"))
                })
            })
            .and_then(estimate::usize_of);
        let torch_dtype = cfg
            .get("torch_dtype")
            .or_else(|| cfg.get("text_config").and_then(|t| t.get("torch_dtype")))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Try ?expand[]=safetensors API for params before falling back to index files
        let mut pb = None;
        let expand_url = format!("{HF_API}/{model_id}?expand[]=safetensors");
        if let Ok(resp) = apply_auth(client.get(&expand_url), token).send().await {
            if resp.status().is_success() {
                if let Ok(info) = resp.json::<Value>().await {
                    pb = estimate::parse_params_from_safetensors_api(&info);
                }
            }
        }

        // Index files
        if pb.is_none() {
            for index_file in ["safetensors.index.json", "pytorch_model.bin.index.json"] {
                if let Some(idx) = fetch_raw(client, model_id, index_file, token).await {
                    if let Some(p) = estimate::parse_params_from_index(&idx, torch_dtype.as_deref())
                    {
                        pb = Some(p);
                        break;
                    }
                }
            }
        }

        // Estimate from config
        if pb.is_none() {
            pb = estimate::estimate_params_from_config(cfg);
        }

        // Name-based fallback
        if pb.is_none() {
            pb = estimate::parse_params_from_name(model_id);
        }

        (
            pb,
            context,
            context_source,
            context_estimated,
            head_dim,
            n_layers,
            n_kv_heads,
            torch_dtype,
        )
    } else {
        let (context, context_source, context_estimated) =
            if let Some(fam) = estimate::family_fallback(model_id) {
                (fam, "family default", true)
            } else {
                (estimate::DEFAULT_CONTEXT, "generic default", true)
            };
        let pb = estimate::parse_params_from_name(model_id);
        (
            pb,
            context,
            context_source,
            context_estimated,
            None,
            None,
            None,
            None,
        )
    };

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

    // Store in cache if cache is Some
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum QuantFormat {
    FP16,
    FP8,
    AWQ,
    GPTQ,
    BNB,
    GGUF,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantVariant {
    pub repo_id: String,
    pub format: QuantFormat,
    pub label: String,
    pub weight_bytes: Option<u64>,
    pub params_b: Option<f64>,
    pub gguf_file: Option<String>,
    pub vllm_native: bool,
}

static GGUF_SHARD_RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
    regex_lite::Regex::new(r"-\d{5}-of-\d{5}$").expect("valid GGUF shard regex")
});

static GGUF_QUANT_RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
    regex_lite::Regex::new(
        r"[-_]((?:UD-)?(?:I?Q\d+(?:_(?:K(?:_[SMLX]{1,2})?|0|1|XXS|XS|S|M|NL))?|BF16|F16|FP16))$",
    )
    .expect("valid GGUF quant regex")
});

/// Returns true if a file is an auxiliary / helper file (e.g. MTP helper, mmproj, imatrix, vocab)
/// that cannot run as a standalone language model.
pub fn is_auxiliary_gguf_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let fname = path.split('/').last().unwrap_or(path).to_ascii_lowercase();

    // Multi-Token Prediction (MTP) helper files
    if lower.starts_with("mtp/")
        || lower.contains("/mtp/")
        || lower.contains("/mtp-")
        || fname.starts_with("mtp-")
        || fname.starts_with("mtp_")
        || fname.contains("-mtp-")
        || fname.contains("_mtp_")
        || fname.contains("-mtp.")
        || fname.contains("_mtp.")
    {
        return true;
    }

    // Multimodal vision projection helpers
    if fname.starts_with("mmproj") || lower.contains("/mmproj") {
        return true;
    }

    // Importance matrix calibration files
    if fname.starts_with("imatrix") || lower.contains("/imatrix") {
        return true;
    }

    // Vocabulary or tokenizer auxiliary files
    if fname.starts_with("vocab") || fname.starts_with("tokenizer") {
        return true;
    }

    // Draft / speculative helper files
    if fname.starts_with("draft") || fname.starts_with("speculative") {
        return true;
    }

    false
}

/// Parse GGUF quant label from filename. Returns None for non-GGUF or auxiliary helper files.
pub fn parse_gguf_quant_label(filename: &str) -> Option<String> {
    if !filename.ends_with(".gguf") || is_auxiliary_gguf_file(filename) {
        return None;
    }
    let stem = filename.strip_suffix(".gguf").unwrap();
    // Strip shard suffix like -00001-of-00003
    let stem = GGUF_SHARD_RE.replace(stem, "");
    // Match quant label at end: Q*, IQ*, UD-Q*, UD-IQ*, BF16, F16, FP16
    GGUF_QUANT_RE
        .captures(&stem)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// Extract base model name (part after the org/user prefix).
pub fn extract_base_name(model_id: &str) -> &str {
    model_id.split('/').last().unwrap_or(model_id)
}

/// Detect quant format from repo name suffix.
pub fn format_from_repo_suffix(name: &str) -> Option<QuantFormat> {
    let n = name.to_lowercase();
    if n.ends_with("-awq") || n.contains("-awq-") || n.contains("-awq_") {
        return Some(QuantFormat::AWQ);
    }
    if n.contains("-gptq") {
        return Some(QuantFormat::GPTQ);
    }
    if n.ends_with("-fp8") || n.ends_with("-fp8-dynamic") {
        return Some(QuantFormat::FP8);
    }
    if n.ends_with("-bnb-4bit") {
        return Some(QuantFormat::BNB);
    }
    None
}

/// Discover quant variants for a base model. Best-effort: failures are silently skipped.
pub async fn discover_quant_variants(
    client: &reqwest::Client,
    base_model_id: &str,
    sem: &tokio::sync::Semaphore,
    cache: Option<&std::sync::Mutex<std::collections::HashMap<String, crate::state::CachedQuants>>>,
    token: Option<&str>,
) -> Vec<QuantVariant> {
    // 1. Check cache first (TTL 1 hour)
    if let Some(c) = cache {
        if let Ok(guard) = c.lock() {
            if let Some(entry) = guard.get(base_model_id) {
                if entry.fetched_at.elapsed() < std::time::Duration::from_secs(3600) {
                    return entry.variants.clone();
                }
            }
        }
    }

    // 2. If the model is already a known quant format (e.g. -AWQ, -GPTQ, -FP8, -bnb-4bit),
    // don't burn network requests searching for child quants of a quantized repo.
    if let Some(fmt) = format_from_repo_suffix(base_model_id) {
        let res = vec![QuantVariant {
            repo_id: base_model_id.to_string(),
            format: fmt.clone(),
            label: match fmt {
                QuantFormat::AWQ => "AWQ".into(),
                QuantFormat::GPTQ => "GPTQ".into(),
                QuantFormat::FP8 => "FP8".into(),
                QuantFormat::BNB => "BNB-4bit".into(),
                _ => "Unknown".into(),
            },
            weight_bytes: None,
            params_b: None,
            gguf_file: None,
            vllm_native: true,
        }];
        if let Some(c) = cache {
            if let Ok(mut guard) = c.lock() {
                guard.insert(
                    base_model_id.to_string(),
                    crate::state::CachedQuants {
                        variants: res.clone(),
                        fetched_at: std::time::Instant::now(),
                    },
                );
            }
        }
        return res;
    }

    let base_name = extract_base_name(base_model_id);
    let org = base_model_id.split('/').next().unwrap_or("");

    // 3. If it is a GGUF repo, inspect it directly without searching other publishers
    if base_model_id.to_lowercase().ends_with("-gguf")
        || base_model_id.to_lowercase().contains(".gguf")
    {
        let res = check_gguf_repo(client, base_model_id, sem, token).await;
        if let Some(c) = cache {
            if let Ok(mut guard) = c.lock() {
                guard.insert(
                    base_model_id.to_string(),
                    crate::state::CachedQuants {
                        variants: res.clone(),
                        fetched_at: std::time::Instant::now(),
                    },
                );
            }
        }
        return res;
    }

    let mut variants = Vec::new();

    // 4. Always include FP16 as the base variant
    variants.push(QuantVariant {
        repo_id: base_model_id.to_string(),
        format: QuantFormat::FP16,
        label: "FP16".into(),
        weight_bytes: None,
        params_b: None,
        gguf_file: None,
        vllm_native: true,
    });

    // 5. Run cross-repo candidate, publisher, and GGUF queries concurrently
    let suffixes = [
        "-AWQ",
        "-GPTQ-Int4",
        "-GPTQ",
        "-FP8",
        "-FP8-dynamic",
        "-bnb-4bit",
    ];
    let (c0, c1, c2, c3, c4, c5, p0, p1, p2, p3, g0, g1, g2) = tokio::join!(
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[0]),
            sem,
            token
        ),
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[1]),
            sem,
            token
        ),
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[2]),
            sem,
            token
        ),
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[3]),
            sem,
            token
        ),
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[4]),
            sem,
            token
        ),
        check_candidate(
            client,
            format!("{org}/{base_name}{}", suffixes[5]),
            sem,
            token
        ),
        check_publisher(client, "neuralmagic", base_name, sem, token),
        check_publisher(client, "hugging-quants", base_name, sem, token),
        check_publisher(client, "ISTA-DASLab", base_name, sem, token),
        check_publisher(client, "TheBloke", base_name, sem, token),
        check_gguf(client, "unsloth", base_name, sem, token),
        check_gguf(client, "bartowski", base_name, sem, token),
        check_gguf(client, "TheBloke", base_name, sem, token),
    );

    for repo_id in [c0, c1, c2, c3, c4, c5, p0, p1, p2, p3]
        .into_iter()
        .flatten()
    {
        if let Some(fmt) = format_from_repo_suffix(&repo_id) {
            variants.push(QuantVariant {
                repo_id: repo_id.clone(),
                format: fmt.clone(),
                label: match fmt {
                    QuantFormat::AWQ => "AWQ".into(),
                    QuantFormat::GPTQ => "GPTQ".into(),
                    QuantFormat::FP8 => "FP8".into(),
                    QuantFormat::BNB => "BNB-4bit".into(),
                    _ => "Unknown".into(),
                },
                weight_bytes: None,
                params_b: None,
                gguf_file: None,
                vllm_native: true,
            });
        }
    }

    variants.extend(g0);
    variants.extend(g1);
    variants.extend(g2);

    // Deduplicate by format+label
    let mut unique = Vec::new();
    for v in variants {
        if !unique
            .iter()
            .any(|u: &QuantVariant| u.format == v.format && u.label == v.label)
        {
            unique.push(v);
        }
    }

    if let Some(c) = cache {
        if let Ok(mut guard) = c.lock() {
            guard.insert(
                base_model_id.to_string(),
                crate::state::CachedQuants {
                    variants: unique.clone(),
                    fetched_at: std::time::Instant::now(),
                },
            );
        }
    }

    unique
}

async fn check_candidate(
    client: &reqwest::Client,
    candidate: String,
    sem: &tokio::sync::Semaphore,
    token: Option<&str>,
) -> Option<String> {
    let _permit = sem.acquire().await.ok()?;
    let url = format!("https://huggingface.co/api/models/{candidate}");
    let req = apply_auth(client.get(&url), token);
    match req.send().await {
        Ok(resp) if resp.status().is_success() => Some(candidate),
        _ => None,
    }
}

async fn check_publisher(
    client: &reqwest::Client,
    pub_org: &'static str,
    base_name: &str,
    sem: &tokio::sync::Semaphore,
    token: Option<&str>,
) -> Option<String> {
    let _permit = sem.acquire().await.ok()?;
    let search_q = format!("{pub_org}/{base_name}");
    let url = format!("https://huggingface.co/api/models?search={search_q}&limit=5");
    let req = apply_auth(client.get(&url), token);
    match req.send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(arr) = resp.json::<Vec<Value>>().await {
                for m in arr {
                    if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                        if format_from_repo_suffix(id).is_some() {
                            return Some(id.to_string());
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

pub async fn check_gguf_repo(
    client: &reqwest::Client,
    repo_id: &str,
    sem: &tokio::sync::Semaphore,
    token: Option<&str>,
) -> Vec<QuantVariant> {
    let mut out = Vec::new();
    let _permit = match sem.acquire().await {
        Ok(p) => p,
        Err(_) => return out,
    };
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let req = apply_auth(client.get(&url), token);
    if let Ok(resp) = req.send().await {
        if resp.status().is_success() {
            if let Ok(info) = resp.json::<Value>().await {
                if let Some(siblings) = info.get("siblings").and_then(|v| v.as_array()) {
                    let mut variant_map: std::collections::BTreeMap<String, QuantVariant> =
                        std::collections::BTreeMap::new();
                    for sib in siblings {
                        let fname = sib.get("rfilename").and_then(|v| v.as_str()).unwrap_or("");
                        if let Some(quant_label) = parse_gguf_quant_label(fname) {
                            let size = sib.get("size").and_then(|v| v.as_u64());
                            match variant_map.entry(quant_label.clone()) {
                                std::collections::btree_map::Entry::Vacant(e) => {
                                    e.insert(QuantVariant {
                                        repo_id: repo_id.to_string(),
                                        format: QuantFormat::GGUF,
                                        label: quant_label,
                                        weight_bytes: size,
                                        params_b: None,
                                        gguf_file: Some(fname.to_string()),
                                        vllm_native: false,
                                    });
                                }
                                std::collections::btree_map::Entry::Occupied(mut e) => {
                                    let v = e.get_mut();
                                    if let (Some(existing), Some(addition)) = (v.weight_bytes, size)
                                    {
                                        v.weight_bytes = Some(existing + addition);
                                    }
                                }
                            }
                        }
                    }
                    out.extend(variant_map.into_values());
                }
            }
        }
    }
    out
}

async fn check_gguf(
    client: &reqwest::Client,
    pub_org: &str,
    base_name: &str,
    sem: &tokio::sync::Semaphore,
    token: Option<&str>,
) -> Vec<QuantVariant> {
    let repo_id = if pub_org.is_empty() {
        base_name.to_string()
    } else if base_name.to_lowercase().ends_with("-gguf") {
        format!("{pub_org}/{base_name}")
    } else {
        format!("{pub_org}/{base_name}-GGUF")
    };
    check_gguf_repo(client, &repo_id, sem, token).await
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
    pub speed_bps: Option<f64>, // bytes per second
    pub eta_seconds: Option<f64>, // estimated time remaining in seconds
}

/// Start a background `hf download <model_id>` in the venv, streaming progress
/// lines to the `pull-progress` event. Idempotent per model.
///
/// `app.state()` is NOT usable from the spawned thread, so we hand it a clone
/// of the pulling map Arc + the AppHandle (for emitting events).
pub fn pull_model(state: &StdArc<AppState>, app: tauri::AppHandle, model_id: &str) -> Result<()> {
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
    let distro = state.resolve_distro();
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
    let adv = state.config().advanced_settings;
    let home_env = if let Some(home) = &adv.hf_home {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            format!("mkdir -p {trimmed} && export HF_HOME={trimmed} && ")
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    let offline_env = if adv.hf_offline {
        "export HF_HUB_OFFLINE=1 && "
    } else {
        ""
    };
    let mid = model_id.clone();

    /// Parse a progress line from huggingface_hub download output.
/// Expected formats (HF_HUB_DISABLE_TQDM=1):
///   "Downloading:  10%|██       | 500M/5.3G [00:12<01:48, 120MB/s]"
///   "Fetching 10 files:  50%|█████     | 2/4 [00:30<00:30,  1.5s/it]"
///   "Resolving dependencies..."
fn parse_progress_line(line: &str) -> Option<(f64, Option<f64>, Option<f64>)> {
    // Look for percentage pattern like " 10%" or "100%"
    let percent_re = regex_lite::Regex::new(r"(\d+(?:\.\d+)?)%").ok()?;
    let percent_match = percent_re.captures(line)?;
    let percent = percent_match.get(1)?.as_str().parse::<f64>().ok()?;

    // Look for speed pattern like "120MB/s", "1.5GB/s", "500KB/s", "1.5s/it"
    let speed_re = regex_lite::Regex::new(r"(\d+(?:\.\d+)?)\s*([KMGT]?B)/s|(\d+(?:\.\d+)?)\s*s/it").ok()?;
    let speed_bps = speed_re.captures(line).and_then(|cap| {
        if let Some(bytes_match) = cap.get(1) {
            let val = bytes_match.as_str().parse::<f64>().ok()?;
            let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let multiplier = match unit.to_uppercase().as_str() {
                "B" => 1.0,
                "KB" => 1024.0,
                "MB" => 1024.0 * 1024.0,
                "GB" => 1024.0 * 1024.0 * 1024.0,
                "TB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
                _ => 1.0,
            };
            Some(val * multiplier)
        } else if let Some(it_match) = cap.get(3) {
            // Items per second - can't convert to bytes without file sizes
            None
        } else {
            None
        }
    });

    // Look for ETA pattern like "[00:12<01:48" or "[00:30<00:30"
    let eta_re = regex_lite::Regex::new(r"\[(?:\d{2}:\d{2})<(\d{2}):(\d{2})").ok()?;
    let eta_seconds = eta_re.captures(line).and_then(|cap| {
        let min = cap.get(1)?.as_str().parse::<f64>().ok()?;
        let sec = cap.get(2)?.as_str().parse::<f64>().ok()?;
        Some(min * 60.0 + sec)
    });

    Some((percent, speed_bps, eta_seconds))
}

std::thread::spawn(move || {
        // Replace shell-quote hazards minimally; model ids are safe by construction.
        let maybe_cd = if venv.contains("~") || venv.starts_with('/') {
            format!("cd {}/.. && ", venv)
        } else {
            String::new()
        };
        let script = format!(
            "{}{}{} . {}/bin/activate && HF_HUB_DISABLE_TQDM=1 {}hf download {} 2>&1 || echo __HF_PULL_FAILED__",
            home_env, offline_env, maybe_cd, venv, token_env, mid
        );
        let model_ev = model_id.clone();
        let app_ev = app.clone();
        let on_line = move |line: &str| {
            let (percent, speed_bps, eta_seconds) = parse_progress_line(line).unwrap_or((0.0, None, None));
            let _ = app_ev.emit(
                "pull-progress",
                PullStatus {
                    model: model_ev.clone(),
                    state: "downloading".into(),
                    file: Some(line.to_string()),
                    percent: Some(percent),
                    speed_bps,
                    eta_seconds,
                },
            );
        };
        let out = crate::wsl::run_script_stream(&distro, &script, on_line);

        // If the download was cancelled via pull_cancel, model_id was already removed from pulling_arc.
        // Do not emit "failed" or overwrite the cancellation state.
        {
            let pulling = pulling_arc.lock().unwrap();
            if !pulling.contains_key(&model_id) {
                return;
            }
        }

        let state_label = if out.ok
            && !out.stdout.contains("__HF_PULL_FAILED__")
            && !out.stderr.contains("__HF_PULL_FAILED__")
        {
            "complete"
        } else {
            "failed"
        };
        let err_msg = if state_label == "failed" {
            let clean_err: String = out.stderr.chars().filter(|c| *c != '\u{0}').collect();
            let clean_err = clean_err.trim();
            if !clean_err.is_empty() {
                Some(clean_err.to_string())
            } else {
                let clean_out: String = out.stdout.chars().filter(|c| *c != '\u{0}').collect();
                let last_line = clean_out.lines().last().unwrap_or("").trim();
                if !last_line.is_empty() {
                    Some(last_line.to_string())
                } else {
                    Some("Download failed".to_string())
                }
            }
        } else {
            None
        };
        let _ = app.emit(
            "pull-progress",
            PullStatus {
                model: model_id.clone(),
                state: state_label.into(),
                file: err_msg,
                percent: if state_label == "complete" { Some(100.0) } else { None },
                speed_bps: None,
                eta_seconds: None,
            },
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

        let res = enrich(&client, "dummy/model", Some(&cache), None).await;
        assert!(res.is_some());
        let s = res.unwrap();
        assert_eq!(s.context, 8192);
        assert_eq!(s.params_b, Some(3.5));
    }

    #[tokio::test]
    async fn test_quant_cache_hit_returns_cached_without_network() {
        let client = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let sem = tokio::sync::Semaphore::new(1);
        let dummy_quants = vec![QuantVariant {
            repo_id: "test/model-AWQ".into(),
            format: QuantFormat::AWQ,
            label: "AWQ".into(),
            weight_bytes: Some(4_000_000_000),
            params_b: Some(7.0),
            gguf_file: None,
            vllm_native: true,
        }];
        cache.lock().unwrap().insert(
            "test/model".to_string(),
            crate::state::CachedQuants {
                variants: dummy_quants.clone(),
                fetched_at: Instant::now(),
            },
        );

        let res = discover_quant_variants(&client, "test/model", &sem, Some(&cache), None).await;
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].label, "AWQ");
    }

    #[test]
    fn test_apply_auth_header() {
        let client = reqwest::Client::new();
        let req = apply_auth(
            client.get("https://huggingface.co/api/models"),
            Some("hf_test123"),
        );
        let built = req.build().unwrap();
        assert_eq!(
            built
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer hf_test123")
        );

        let req_none = apply_auth(client.get("https://huggingface.co/api/models"), None);
        let built_none = req_none.build().unwrap();
        assert!(built_none.headers().get("Authorization").is_none());
    }

    #[test]
    fn test_enrich_cache_miss_expired_ttl() {
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
        if let Some(past) = Instant::now().checked_sub(Duration::from_secs(3605)) {
            cache.lock().unwrap().insert(
                "dummy/model-expired-xyz".to_string(),
                CachedEnrichment {
                    stats: dummy_stats,
                    fetched_at: past,
                },
            );

            // Verify cache expiry without live network call:
            // Cache lookup requires elapsed() < 3600 seconds, so this expired entry is rejected.
            let guard = cache.lock().unwrap();
            let is_hit = guard
                .get("dummy/model-expired-xyz")
                .map(|entry| entry.fetched_at.elapsed() < Duration::from_secs(3600))
                .unwrap_or(false);
            assert!(
                !is_hit,
                "Expired cache entry (>3600s) should not count as a cache hit"
            );
        }
    }

    #[test]
    fn test_parse_gguf_quant_label() {
        assert_eq!(
            parse_gguf_quant_label("model-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("Qwen2.5-7B-Instruct-Q8_0.gguf"),
            Some("Q8_0".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-Q4_1.gguf"),
            Some("Q4_1".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-Q5_1.gguf"),
            Some("Q5_1".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-IQ4_NL.gguf"),
            Some("IQ4_NL".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-UD-Q4_K_XL.gguf"),
            Some("UD-Q4_K_XL".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-Q3_K_S-00001-of-00003.gguf"),
            Some("Q3_K_S".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf"),
            Some("BF16".to_string())
        );
        assert_eq!(parse_gguf_quant_label("model.safetensors"), None);
        assert_eq!(parse_gguf_quant_label("README.md"), None);
        assert_eq!(parse_gguf_quant_label("model.gguf"), None);
        assert_eq!(
            parse_gguf_quant_label("model-Q5_K_M.gguf"),
            Some("Q5_K_M".to_string())
        );
        assert_eq!(
            parse_gguf_quant_label("model-IQ2_XXS.gguf"),
            Some("IQ2_XXS".to_string())
        );

        // MTP helper files should be excluded
        assert_eq!(
            parse_gguf_quant_label("MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"),
            None
        );
        assert_eq!(
            parse_gguf_quant_label("MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf"),
            None
        );
        assert_eq!(
            parse_gguf_quant_label("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
            None
        );
        assert_eq!(parse_gguf_quant_label("mtp-Qwen3.8-27B-Q4_0.gguf"), None);

        // Vision projectors and imatrix should be excluded
        assert_eq!(parse_gguf_quant_label("mmproj-BF16.gguf"), None);
        assert_eq!(parse_gguf_quant_label("mmproj-F16.gguf"), None);
        assert_eq!(parse_gguf_quant_label("imatrix_unsloth.gguf"), None);
    }

    #[test]
    fn test_extract_base_model_name() {
        assert_eq!(
            extract_base_name("Qwen/Qwen2.5-7B-Instruct"),
            "Qwen2.5-7B-Instruct"
        );
        assert_eq!(
            extract_base_name("meta-llama/Meta-Llama-3.1-8B-Instruct"),
            "Meta-Llama-3.1-8B-Instruct"
        );
        assert_eq!(extract_base_name("gpt2"), "gpt2");
    }

    #[test]
    fn test_quant_format_from_repo_id() {
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-AWQ"),
            Some(QuantFormat::AWQ)
        );
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-AWQ-INT4"),
            Some(QuantFormat::AWQ)
        );
        assert_eq!(
            format_from_repo_suffix("Meta-Llama-3-8B-awq_int4"),
            Some(QuantFormat::AWQ)
        );
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-GPTQ-Int4"),
            Some(QuantFormat::GPTQ)
        );
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-FP8"),
            Some(QuantFormat::FP8)
        );
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-FP8-dynamic"),
            Some(QuantFormat::FP8)
        );
        assert_eq!(
            format_from_repo_suffix("Qwen2.5-7B-Instruct-bnb-4bit"),
            Some(QuantFormat::BNB)
        );
        assert_eq!(format_from_repo_suffix("Qwen2.5-7B-Instruct-GGUF"), None); // GGUF handled separately via file siblings
        assert_eq!(format_from_repo_suffix("Qwen2.5-7B-Instruct"), None);
    }

    #[test]
    fn test_clean_model_query() {
        assert_eq!(
            clean_model_query("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"),
            "unsloth/Qwen3.8-27B-GGUF"
        );
        assert_eq!(
            clean_model_query("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main"),
            "unsloth/Qwen3.8-27B-GGUF"
        );
        assert_eq!(
            clean_model_query("huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/"),
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
        );
        assert_eq!(
            clean_model_query("  unsloth/Qwen3.8-27B-GGUF?not-real=1  "),
            "unsloth/Qwen3.8-27B-GGUF"
        );
        assert_eq!(clean_model_query("Qwen2.5"), "Qwen2.5");
    }

    #[test]
    fn test_pull_cancellation_map() {
        let state = Arc::new(AppState::new());
        state
            .pulling
            .lock()
            .unwrap()
            .insert("test/model".into(), true);
        assert!(state.pulling.lock().unwrap().contains_key("test/model"));
        state.pulling.lock().unwrap().remove("test/model");
        assert!(!state.pulling.lock().unwrap().contains_key("test/model"));
    }
}
