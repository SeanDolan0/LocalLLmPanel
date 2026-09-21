//! Runtime model providers (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio, vLLM).
//!
//! Each provider can list locally installed models and pull new ones.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// A runtime provider that can serve LLM models locally.
pub trait ModelProvider {

    /// Whether the provider service is reachable right now.
    fn is_available(&self) -> bool;

    /// Return the set of model name stems that are currently installed.
    /// Names are normalised lowercase, e.g. "llama3.1:8b".
    fn installed_models(&self) -> HashSet<String>;

}

// ---------------------------------------------------------------------------
// Ollama provider
// ---------------------------------------------------------------------------

pub struct OllamaProvider {
    base_url: String,
    /// Fallback URL to try when `base_url` is unreachable.
    /// Set when using the default `localhost` address so that systems where
    /// `localhost` resolves to `::1` (IPv6) can fall back to `127.0.0.1`.
    fallback_url: Option<String>,
}

fn normalize_ollama_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        // Unsupported scheme (e.g. ftp://)
        return None;
    }

    Some(format!("http://{host}"))
}

/// Returns true if the URL's host is a wildcard bind address — `0.0.0.0`
/// (IPv4) or `[::]` (IPv6). Servers listen on these to accept traffic on
/// every interface, but they are never valid as a connect target. When
/// Ollama is started with `OLLAMA_HOST=0.0.0.0`, that value leaks into the
/// environment and we must not pass it to a client.
fn is_wildcard_bind_address(url: &str) -> bool {
    let after_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host_port = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);

    if let Some(rest) = host_port.strip_prefix('[') {
        if let Some(end_idx) = rest.find(']') {
            let host = &rest[..end_idx];
            return host == "::" || host == "0:0:0:0:0:0:0:0";
        }
        return false;
    }

    let host = host_port.split(':').next().unwrap_or("");
    host == "0.0.0.0"
}

impl Default for OllamaProvider {
    fn default() -> Self {
        let explicit = std::env::var("OLLAMA_HOST").ok().and_then(|raw| {
            let Some(normalized) = normalize_ollama_host(&raw) else {
                eprintln!(
                    "Warning: could not parse OLLAMA_HOST='{}'. Expected host:port or http(s)://host:port",
                    raw
                );
                return None;
            };
            if is_wildcard_bind_address(&normalized) {
                eprintln!(
                    "Warning: OLLAMA_HOST='{}' is a wildcard bind address; falling back to localhost.",
                    raw
                );
                return None;
            }
            Some(normalized)
        });

        if let Some(base_url) = explicit {
            // User supplied an explicit host — use it as-is, no fallback.
            Self {
                base_url,
                fallback_url: None,
            }
        } else {
            // Default: try `localhost` first; fall back to `127.0.0.1` for
            // systems where `localhost` resolves to the IPv6 loopback `::1`
            // while Ollama is only listening on the IPv4 `127.0.0.1`.
            Self {
                base_url: "http://localhost:11434".to_string(),
                fallback_url: Some("http://127.0.0.1:11434".to_string()),
            }
        }
    }
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the full API URL for a given endpoint path.
    fn api_url(&self, path: &str) -> String {
        format!("{}/api/{}", self.base_url.trim_end_matches('/'), path)
    }

    /// Delete a model from Ollama via its API.
    pub fn delete_model(&self, model_tag: &str) -> Result<(), String> {
        // Ollama DELETE /api/delete requires a JSON body.
        // ureq v3's delete() doesn't support request bodies, so we build a
        // raw http::Request and pass it to the agent's `run()` method.
        let body = serde_json::json!({ "name": model_tag }).to_string();
        let url = self.api_url("delete");
        let request = http::Request::builder()
            .method("DELETE")
            .uri(&url)
            .header("content-type", "application/json")
            .body(body)
            .map_err(|e| format!("Failed to build request: {}", e))?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build()
            .into();
        let resp = agent
            .run(request)
            .map_err(|e| format!("Ollama delete request failed: {}", e))?;
        if resp.status() == 200 {
            Ok(())
        } else {
            Err(format!("Ollama returned status {}", resp.status()))
        }
    }

    /// Single-pass startup probe to avoid duplicate `/api/tags` calls.
    /// Returns `(available, installed_models)`.
    /// When the primary URL (`localhost`) fails and a fallback (`127.0.0.1`)
    /// is configured, the fallback is tried and—if successful—adopted as the
    /// provider's base URL for all subsequent requests (pull, show, …).
    pub fn detect_with_installed(&mut self) -> (bool, HashSet<String>, usize) {
        let set = HashSet::new();

        let primary_ok = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call();

        let resp = match primary_ok {
            Ok(r) => r,
            Err(_) => {
                // Primary URL failed — try the fallback if one is set.
                let Some(ref fallback) = self.fallback_url.clone() else {
                    return (false, set, 0);
                };
                let fallback_url = format!("{}/api/tags", fallback.trim_end_matches('/'));
                let Ok(r) = ureq::get(&fallback_url)
                    .config()
                    .timeout_global(Some(std::time::Duration::from_millis(800)))
                    .build()
                    .call()
                else {
                    return (false, set, 0);
                };
                // Fallback worked: adopt it so that pull/show use 127.0.0.1.
                self.base_url = fallback.clone();
                self.fallback_url = None;
                r
            }
        };

        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return (true, set, 0);
        };
        let (set, count) = build_installed_set(tags.models);
        (true, set, count)
    }

    /// Like `installed_models`, but also returns the true model count.
    /// The HashSet may have fewer entries than 2*count due to family-name deduplication,
    /// so `len() / 2` is unreliable for counting models.
    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let Ok(resp) = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .call()
        else {
            return (HashSet::new(), 0);
        };
        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return (HashSet::new(), 0);
        };
        build_installed_set(tags.models)
    }

    /// Best-effort check that a tag exists in Ollama's remote registry.
    /// Uses the local Ollama daemon's `/api/show` resolution path.
    pub fn has_remote_tag(&self, model_tag: &str) -> bool {
        let body = serde_json::json!({ "model": model_tag });
        ureq::post(&self.api_url("show"))
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(1200)))
            .build()
            .send_json(&body)
            .is_ok()
    }
}

// -- JSON response types for Ollama API --

#[derive(serde::Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(serde::Deserialize, Default)]
struct OllamaModel {
    /// e.g. "llama3.1:8b-instruct-q4_K_M"
    name: String,
    /// On-disk size in bytes. Cloud-hosted models are served remotely and
    /// report `0` because nothing is stored locally.
    #[serde(default)]
    size: u64,
    #[serde(default)]
    details: OllamaModelDetails,
}

#[derive(serde::Deserialize, Default)]
struct OllamaModelDetails {
    /// Parameter count of the resolved weights as Ollama reports it, e.g.
    /// "8.2B" for `qwen3:latest`. Empty when the daemon omits it.
    #[serde(default)]
    parameter_size: String,
}

impl OllamaModel {
    /// Whether this entry is a cloud-hosted model rather than a local install.
    /// Ollama surfaces cloud models with a `-cloud` tag suffix (e.g.
    /// `qwen3-coder:480b-cloud`) and a zero on-disk size.
    fn is_cloud(&self) -> bool {
        let tag = self.name.rsplit(':').next().unwrap_or("");
        tag.ends_with("-cloud") || self.size == 0
    }
}

/// The tag Ollama resolves when a model is pulled without one.
const OLLAMA_DEFAULT_TAG: &str = "latest";

/// Ollama-style size tokens implied by the parameter count Ollama reports,
/// e.g. "8.2B" → `["8b", "8.2b"]`.
///
/// Most tags carry the marketing size rather than the true count (`qwen2.5:14b`
/// reports "14.8B"), hence the truncated form. Families tagged with a decimal
/// (`qwen3:1.7b`, `solar:10.7b`) need the verbatim form as well. Counts below
/// 1B are reported in "M" — `qwen3:0.6b` reports "596.05M" — and have no
/// reliable tag form, so they yield nothing rather than a bogus `0b`.
fn size_tokens_from_parameter_size(parameter_size: &str) -> Vec<String> {
    let raw = parameter_size.trim().to_lowercase();
    let Some(value) = raw
        .strip_suffix('b')
        .and_then(|digits| digits.parse::<f64>().ok())
        .filter(|v| *v >= 1.0)
    else {
        return Vec::new();
    };

    let mut tokens = vec![format!("{}b", value.trunc() as u64)];
    if tokens[0] != raw {
        tokens.push(raw);
    }
    tokens
}

/// Build the set of installed model name stems from Ollama's tag list, plus the
/// count of locally-installed models. Cloud-hosted models are skipped entirely:
/// they are not installed locally, and inserting their family stem (e.g.
/// `qwen3-coder` from `qwen3-coder:480b-cloud`) would falsely mark unrelated
/// models as installed (#619).
///
/// A **sized** install (`qwen3:8b`) contributes its tag and nothing else: the
/// tag already says exactly which weights are on disk, and adding the bare
/// family stem made every catalog entry in that family look installed — one
/// `qwen3:8b` marked 238 of 9,250 models, `Qwen3-235B-A22B` among them (#861).
/// Only an untagged / `:latest` install, where the size genuinely is unknown,
/// contributes a family stem — plus the sized alias its parameter count implies,
/// so `qwen3:latest` still matches `Qwen/Qwen3-8B` specifically.
fn build_installed_set(models: Vec<OllamaModel>) -> (HashSet<String>, usize) {
    let mut set = HashSet::new();
    let mut count = 0;
    for m in models {
        if m.is_cloud() {
            continue;
        }
        count += 1;
        let lower = m.name.to_lowercase();
        set.insert(lower.clone());

        let (family, tag) = lower
            .split_once(':')
            .unwrap_or((lower.as_str(), OLLAMA_DEFAULT_TAG));
        if tag != OLLAMA_DEFAULT_TAG {
            continue;
        }
        set.insert(family.to_string());
        for size in size_tokens_from_parameter_size(&m.details.parameter_size) {
            set.insert(format!("{family}:{size}"));
        }
    }
    (set, count)
}

impl ModelProvider for OllamaProvider {

    fn is_available(&self) -> bool {
        ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// OpenAI-compatible provider helpers
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub(crate) struct OpenAiModelList {
    data: Vec<OpenAiModel>,
}

#[derive(serde::Deserialize)]
struct OpenAiModel {
    /// Model id, e.g. "meta-llama/Llama-3.1-8B-Instruct".
    id: String,
    /// OpenAI-compatible providers may include an owner string. oMLX uses
    /// `owned_by: "omlx"`, which lets us disambiguate it from vLLM.
    owned_by: Option<String>,
}

fn openai_models_url(base_url: &str) -> String {
    format!("{}/v1/models", base_url.trim_end_matches('/'))
}

/// Identity of an OpenAI-compatible endpoint, read from the `/v1/models`
/// response itself so a server is recognized before any of its model ids
/// are imported (#791, #790).
///
/// Measured against live servers (2026-08-23): llama-server stamps
/// `Server: llama.cpp` on every response and lists models with
/// `owned_by: "llamacpp"`; llama-swap lists models with
/// `owned_by: "llama-swap"`; mlx_lm.server (0.31.3) sends a Python
/// `BaseHTTP` Server header and no `owned_by` field at all. vLLM and Docker
/// Model Runner were measured 2026-09-01 (see the variants below); Ferrum was
/// captured 2026-09-02 in #992. LM Studio is identified out-of-band via its
/// native /api/v0 API (`endpoint_is_lmstudio`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenAiEndpointIdentity {
    /// llama.cpp serving directly.
    LlamaCpp,
    /// A llama-swap proxy fronting llama.cpp instances.
    LlamaSwap,
    /// vLLM's OpenAI server (measured 2026-09-01: owned_by "vllm").
    Vllm,
    /// Ferrum's OpenAI-compatible server (owned_by "ferrum").
    Ferrum,
    /// Docker Model Runner (measured 2026-09-01: owned_by "docker").
    DockerModelRunner,
    /// No foreign marker recognized.
    Unrecognized,
}

fn classify_openai_endpoint(
    server_header: Option<&str>,
    list: &OpenAiModelList,
) -> OpenAiEndpointIdentity {
    let owned_by = |name: &str| {
        list.data.iter().any(|model| {
            model
                .owned_by
                .as_deref()
                .is_some_and(|owner| owner.eq_ignore_ascii_case(name))
        })
    };
    // vLLM stamps owned_by "vllm". Its Server: uvicorn header is shared by any
    // FastAPI app, so the owner string is the discriminator. Measured 2026-09-01.
    if owned_by("vllm") {
        return OpenAiEndpointIdentity::Vllm;
    }
    if owned_by("ferrum") {
        return OpenAiEndpointIdentity::Ferrum;
    }
    // Docker Model Runner stamps owned_by "docker" (plus a per-model `dmr`
    // object) and sends no Server header. Measured 2026-09-01.
    if owned_by("docker") {
        return OpenAiEndpointIdentity::DockerModelRunner;
    }
    if owned_by("llama-swap") {
        return OpenAiEndpointIdentity::LlamaSwap;
    }
    if owned_by("llamacpp") || server_header.is_some_and(|s| s.eq_ignore_ascii_case("llama.cpp")) {
        return OpenAiEndpointIdentity::LlamaCpp;
    }
    OpenAiEndpointIdentity::Unrecognized
}

pub(crate) fn fetch_openai_model_list(
    base_url: &str,
    timeout: std::time::Duration,
) -> Option<(OpenAiModelList, OpenAiEndpointIdentity)> {
    let resp = ureq::get(&openai_models_url(base_url))
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
        .ok()?;
    let server_header = resp
        .headers()
        .get("server")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let list = resp.into_body().read_json::<OpenAiModelList>().ok()?;
    let identity = classify_openai_endpoint(server_header.as_deref(), &list);
    Some((list, identity))
}

fn openai_model_list_is_omlx(list: &OpenAiModelList) -> bool {
    list.data.iter().any(|model| {
        model
            .owned_by
            .as_deref()
            .is_some_and(|owner| owner.eq_ignore_ascii_case("omlx"))
    })
}

pub(crate) fn openai_model_ids(list: &OpenAiModelList) -> impl Iterator<Item = &str> {
    list.data.iter().map(|model| model.id.as_str())
}

fn is_omlx_status_payload(json: &serde_json::Value) -> bool {
    json.get("status").and_then(|v| v.as_str()) == Some("ok")
        && json.get("version").and_then(|v| v.as_str()).is_some()
        && (json.get("models_discovered").is_some()
            || json.get("model_memory_max").is_some()
            || json.get("cache_efficiency").is_some())
}

fn endpoint_has_omlx_status(base_url: &str, timeout: std::time::Duration) -> bool {
    let url = format!("{}/api/status", base_url.trim_end_matches('/'));
    let Ok(resp) = ureq::get(&url)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
    else {
        return false;
    };
    let Ok(json) = resp.into_body().read_json::<serde_json::Value>() else {
        return false;
    };
    is_omlx_status_payload(&json)
}

/// One entry of LM Studio's native `/api/v0/models` response. `compatibility_type`
/// and `state` are LM Studio-specific fields, absent from the OpenAI `/v1/models`
/// schema, which lets us positively identify the runtime.
#[derive(serde::Deserialize)]
struct LmStudioNativeModel {
    #[serde(default)]
    compatibility_type: Option<String>,
    #[serde(default)]
    state: Option<String>,
}

#[derive(serde::Deserialize)]
struct LmStudioNativeList {
    data: Vec<LmStudioNativeModel>,
}

/// True when the endpoint answers LM Studio's native `/api/v0/models` route
/// with LM Studio's native schema. A generic OpenAI-compatible server (even
/// one behind Express) does not serve this LM Studio-specific route, so this
/// is the evidence that identifies LM Studio rather than the framework-wide
/// `X-Powered-By` header (#790). Measured 2026-09-01 against LM Studio 0.4.23:
/// the route lists every model on disk (loaded or not) with the native
/// `compatibility_type`/`state` fields, an unauthenticated `Authorization`
/// header is tolerated when the API key requirement is off, and unknown paths
/// answer a JSON `error` object rather than a model list.
fn endpoint_is_lmstudio(
    base_url: &str,
    api_key: Option<&str>,
    timeout: std::time::Duration,
) -> bool {
    let url = format!("{}/api/v0/models", base_url.trim_end_matches('/'));
    let mut req = ureq::get(&url)
        .config()
        .timeout_global(Some(timeout))
        .build();
    // Honor a configured key, like the /v1/models fetch: with LM Studio's
    // "Require API Key" enabled the native route needs it too, and sending it
    // when the requirement is off is accepted (measured).
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let Ok(resp) = req.call() else {
        return false;
    };
    let Ok(list) = resp.into_body().read_json::<LmStudioNativeList>() else {
        return false;
    };
    // Identify only on an entry carrying LM Studio-native fields. An empty
    // `data` array carries no evidence, so it stays unidentified: since the
    // native route lists on-disk models, a server with no downloaded models
    // has nothing to import anyway.
    list.data
        .iter()
        .any(|m| m.compatibility_type.is_some() || m.state.is_some())
}

/// True when the endpoint's root answers with Docker Model Runner's banner.
/// A runner with no models yet returns an empty `/v1/models` list that carries
/// no `owned_by` marker, so identity falls back to this probe. Measured
/// 2026-09-01: `GET /` answers `Docker Model Runner is running` as plain text.
fn endpoint_is_docker_model_runner_root(base_url: &str, timeout: std::time::Duration) -> bool {
    let url = format!("{}/", base_url.trim_end_matches('/'));
    let Ok(resp) = ureq::get(&url)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
    else {
        return false;
    };
    let Ok(body) = resp.into_body().read_to_string() else {
        return false;
    };
    body.trim() == "Docker Model Runner is running"
}

// ---------------------------------------------------------------------------
// MLX provider (Apple MLX framework via HuggingFace cache)
// ---------------------------------------------------------------------------

const MLX_DEFAULT_SERVER_URL: &str = "http://localhost:8080";
const OMLX_DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8000";

struct MlxServerCandidate<'a> {
    base_url: &'a str,
    require_omlx_identity: bool,
}

pub struct MlxProvider {
    server_url: String,
    server_url_explicit: bool,
}

impl Default for MlxProvider {
    fn default() -> Self {
        let explicit = std::env::var("MLX_LM_HOST").ok().and_then(|url| {
            if url.starts_with("http://") || url.starts_with("https://") {
                Some(url)
            } else {
                eprintln!(
                    "Warning: MLX_LM_HOST must start with http:// or https://, ignoring: {}",
                    url
                );
                None
            }
        });
        let server_url_explicit = explicit.is_some();
        let server_url = explicit.unwrap_or_else(|| MLX_DEFAULT_SERVER_URL.to_string());
        Self {
            server_url,
            server_url_explicit,
        }
    }
}

impl MlxProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_server_url(url: &str) -> Self {
        Self {
            server_url: url.to_string(),
            server_url_explicit: true,
        }
    }

    fn server_candidates(&self) -> Vec<MlxServerCandidate<'_>> {
        let mut candidates = vec![MlxServerCandidate {
            base_url: self.server_url.as_str(),
            require_omlx_identity: false,
        }];
        if !self.server_url_explicit && self.server_url != OMLX_DEFAULT_SERVER_URL {
            candidates.push(MlxServerCandidate {
                base_url: OMLX_DEFAULT_SERVER_URL,
                require_omlx_identity: true,
            });
        }
        candidates
    }

    fn fetch_candidate_models(
        candidate: &MlxServerCandidate<'_>,
        timeout: std::time::Duration,
    ) -> Option<OpenAiModelList> {
        let has_omlx_status = candidate.require_omlx_identity
            && endpoint_has_omlx_status(candidate.base_url, timeout);
        let (list, identity) = fetch_openai_model_list(candidate.base_url, timeout)?;
        // Shared identity gate (#791): a llama.cpp server or a llama-swap
        // proxy answering on this port is not an MLX runtime, so its model
        // list must not be imported as MLX models.
        if identity != OpenAiEndpointIdentity::Unrecognized {
            return None;
        }
        if candidate.require_omlx_identity && !has_omlx_status && !openai_model_list_is_omlx(&list)
        {
            return None;
        }
        Some(list)
    }

    /// Single-pass startup probe for MLX.
    /// On non-macOS, skips network checks and reports `available=false`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>) {
        let mut set = scan_hf_cache_for_mlx();
        if !cfg!(target_os = "macos") {
            return (false, set);
        }

        for candidate in self.server_candidates() {
            if let Some(list) =
                Self::fetch_candidate_models(&candidate, std::time::Duration::from_millis(800))
            {
                for id in openai_model_ids(&list) {
                    set.insert(id.to_lowercase());
                }
                return (true, set);
            }
        }

        (check_mlx_python(), set)
    }
}

/// Cache whether mlx_lm Python package is importable.
static MLX_PYTHON_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn check_mlx_python() -> bool {
    *MLX_PYTHON_AVAILABLE.get_or_init(|| {
        crate::hardware::silent_cmd("python3")
            .args(["-c", "import mlx_lm"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn is_likely_mlx_repo(owner: &str, repo: &str) -> bool {
    let owner_lower = owner.to_lowercase();
    let repo_lower = repo.to_lowercase();
    // Exclude GGUF repos — they belong to llama.cpp, not MLX
    if is_likely_gguf_repo(&repo_lower) {
        return false;
    }
    owner_lower == "mlx-community"
        || repo_lower.contains("-mlx-")
        || repo_lower.ends_with("-mlx")
        || repo_lower.contains("mlx-")
        || repo_lower.ends_with("mlx")
}

fn is_likely_gguf_repo(repo_lower: &str) -> bool {
    repo_lower.contains("-gguf") || repo_lower.ends_with("gguf")
}

/// Scan HuggingFace cache directories for MLX model directories.
fn scan_hf_cache_for_mlx() -> HashSet<String> {
    let mut set = HashSet::new();
    for cache_dir in dirs_hf_cache_all() {
        let Ok(entries) = std::fs::read_dir(&cache_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let Some(rest) = name_str.strip_prefix("models--") else {
                continue;
            };
            let mut parts = rest.splitn(2, "--");
            let Some(owner) = parts.next() else {
                continue;
            };
            let Some(repo) = parts.next() else {
                continue;
            };

            if !is_likely_mlx_repo(owner, repo) {
                continue;
            }

            let owner_lower = owner.to_lowercase();
            let repo_lower = repo.to_lowercase();
            set.insert(format!("{}/{}", owner_lower, repo_lower));
            set.insert(repo_lower);
        }
    }
    set
}

/// Scan HuggingFace cache directories for GGUF model directories.
fn scan_hf_cache_for_gguf() -> (HashSet<String>, usize) {
    let mut set = HashSet::new();
    let mut count = 0usize;
    for cache_dir in dirs_hf_cache_all() {
        let Ok(entries) = std::fs::read_dir(&cache_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let Some(rest) = name_str.strip_prefix("models--") else {
                continue;
            };
            let mut parts = rest.splitn(2, "--");
            let Some(owner) = parts.next() else {
                continue;
            };
            let Some(repo) = parts.next() else {
                continue;
            };

            if !is_likely_gguf_repo(&repo.to_lowercase()) {
                continue;
            }

            count += 1;
            let owner_lower = owner.to_lowercase();
            let repo_lower = repo.to_lowercase();
            set.insert(format!("{}/{}", owner_lower, repo_lower));
            set.insert(repo_lower);
        }
    }
    (set, count)
}

/// Return all candidate HuggingFace cache directories.
///
/// The HF CLI always uses `~/.cache/huggingface/hub` (XDG-style) regardless
/// of platform, but `dirs::cache_dir()` returns `~/Library/Caches` on macOS.
/// We check both to handle either location.
fn dirs_hf_cache_all() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(cache) = std::env::var("HF_HOME") {
        dirs.push(std::path::PathBuf::from(cache).join("hub"));
        return dirs;
    }

    // Platform-native cache dir (e.g. ~/Library/Caches on macOS)
    if let Some(cache) = dirs::cache_dir() {
        dirs.push(cache.join("huggingface").join("hub"));
    }

    // XDG-style ~/.cache (what the HF CLI actually uses on all platforms)
    if let Some(home) = dirs::home_dir() {
        let xdg = home.join(".cache").join("huggingface").join("hub");
        if !dirs.iter().any(|d| d == &xdg) {
            dirs.push(xdg);
        }
    }

    if dirs.is_empty() {
        dirs.push(std::path::PathBuf::from("/tmp/.cache/huggingface/hub"));
    }
    dirs
}

impl ModelProvider for MlxProvider {

    fn is_available(&self) -> bool {
        if !cfg!(target_os = "macos") {
            return false;
        }
        // Try MLX-compatible servers first.
        for candidate in self.server_candidates() {
            if Self::fetch_candidate_models(&candidate, std::time::Duration::from_secs(2)).is_some()
            {
                return true;
            }
        }
        // Fall back to checking if mlx_lm is installed
        check_mlx_python()
    }

    fn installed_models(&self) -> HashSet<String> {
        let mut set = scan_hf_cache_for_mlx();
        if !cfg!(target_os = "macos") {
            return set;
        }
        // Also try querying MLX-compatible servers if running.
        for candidate in self.server_candidates() {
            if let Some(list) =
                Self::fetch_candidate_models(&candidate, std::time::Duration::from_secs(2))
            {
                for id in openai_model_ids(&list) {
                    set.insert(id.to_lowercase());
                }
                break;
            }
        }
        set
    }

}

// ---------------------------------------------------------------------------
// llama.cpp provider (direct GGUF download from HuggingFace)
// ---------------------------------------------------------------------------

/// A provider that downloads GGUF model files directly from HuggingFace
/// and uses llama.cpp binaries (`llama-cli`, `llama-server`) to run them.
///
/// Unlike Ollama, this doesn't require a running daemon — it downloads
/// GGUF files to a local cache directory and invokes llama.cpp directly.
pub struct LlamaCppProvider {
    /// Directory where GGUF models are stored.
    models_dir: PathBuf,
    /// Path to llama-cli binary, if found.
    llama_cli: Option<String>,
    /// Path to llama-server binary, if found.
    llama_server: Option<String>,
    /// Whether a running llama-server was detected via health probe.
    server_running: bool,
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        let models_dir = llamacpp_models_dir();
        let llama_cli = find_binary("llama-cli");
        let llama_server = find_binary("llama-server");

        // If no binaries found, check if a server is already running
        let server_running = if llama_cli.is_none() && llama_server.is_none() {
            let port = std::env::var("LLAMA_SERVER_PORT").unwrap_or_else(|_| "8080".to_string());
            probe_llama_server(&format!("http://localhost:{}", port))
        } else {
            false
        };

        Self {
            models_dir,
            llama_cli,
            llama_server,
            server_running,
        }
    }
}

impl LlamaCppProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Like `installed_models`, but also returns the true GGUF file count.
    /// The HashSet may have fewer entries than 2*count due to deduplication
    /// when stripping quantization suffixes, so `len() / 2` is unreliable.
    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let mut set = HashSet::new();
        let mut count = 0usize;
        for path in self.list_gguf_files() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                count += 1;
                let lower = stem.to_lowercase();
                set.insert(lower.clone());
                if let Some(base) = strip_gguf_quant_suffix(&lower) {
                    set.insert(base);
                }
            }
        }
        // Also scan the HuggingFace cache for GGUF repos downloaded via `hf download`
        let (hf_set, hf_count) = scan_hf_cache_for_gguf();
        count += hf_count;
        set.extend(hf_set);
        (set, count)
    }

    /// Return the directory where GGUF models are cached.
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Override the models directory at runtime.
    pub fn set_models_dir(&mut self, dir: PathBuf) {
        self.models_dir = dir;
    }

    /// Delete a GGUF model file by tag (file stem match).
    pub fn delete_model(&self, model_tag: &str) -> Result<(), String> {
        let tag_lower = model_tag.to_lowercase();
        for path in self.list_gguf_files() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && stem.to_lowercase() == tag_lower
            {
                return std::fs::remove_file(&path)
                    .map_err(|e| format!("Failed to delete {}: {}", path.display(), e));
            }
        }
        Err(format!("Model file not found for '{}'", model_tag))
    }

    /// Path to `llama-cli` if detected.
    pub fn llama_cli_path(&self) -> Option<&str> {
        self.llama_cli.as_deref()
    }

    /// Path to `llama-server` if detected.
    pub fn llama_server_path(&self) -> Option<&str> {
        self.llama_server.as_deref()
    }

    /// Whether a running llama-server was detected via health probe.
    pub fn server_running(&self) -> bool {
        self.server_running
    }

    /// Return a short status hint describing how llama.cpp was (or wasn't) detected.
    pub fn detection_hint(&self) -> &'static str {
        if self.llama_cli.is_some() || self.llama_server.is_some() {
            ""
        } else if self.server_running {
            "server detected"
        } else {
            "not in PATH, set LLAMA_CPP_PATH"
        }
    }

    /// List all `.gguf` files in the cache directory, descending into
    /// subdirectories up to [`GGUF_SCAN_MAX_DEPTH`].
    pub fn list_gguf_files(&self) -> Vec<PathBuf> {
        collect_gguf_files(&self.models_dir, GGUF_SCAN_MAX_DEPTH)
    }
}

/// Default directory for llama.cpp GGUF model cache.
/// How far below a models root to look for `.gguf` files.
///
/// A flat scan is not enough: LM Studio stores `publisher/repo/model.gguf`,
/// and anyone pointing `LLMFIT_MODELS_DIR` at a tree with that shape hits the
/// same wall. Three levels covers those layouts while keeping the walk
/// bounded, so a large library does not turn into a full crawl.
pub const GGUF_SCAN_MAX_DEPTH: usize = 3;

/// Collect `.gguf` files under `root`, descending at most `max_depth`
/// directory levels (the root itself is level one).
///
/// The name is tested for the extension before anything asks the filesystem
/// about the entry, so only directories cost a `file_type` call. Unreadable
/// directories are skipped rather than aborting the walk.
fn collect_gguf_files(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if max_depth == 0 {
        return files;
    }
    let mut pending = vec![(root.to_path_buf(), 1usize)];

    while let Some((dir, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let looks_like_model = name.to_string_lossy().to_lowercase().ends_with(".gguf");
            let path = entry.path();

            // Only names that already look like models are resolved, so the
            // extension test is still what does the filtering. `metadata`
            // rather than `DirEntry::file_type` because the latter describes
            // the symlink instead of its target, and a symlinked model in
            // `LLMFIT_MODELS_DIR` is still a model.
            let resolved = looks_like_model
                .then(|| path.metadata())
                .and_then(Result::ok);
            if let Some(meta) = &resolved
                && meta.is_file()
            {
                files.push(path);
                continue;
            }

            if depth >= max_depth {
                continue;
            }
            // Descend into real directories only, which includes one named
            // `foo.gguf`: that is not a model, but what is inside it may be.
            //
            // Directory symlinks are deliberately not followed. Doing so lets
            // the walk leave the models root entirely, or alias a subtree back
            // into itself and count the same model twice, and callers act on
            // the paths returned here. Symlinked model *files* are still
            // resolved above, which is the case that actually came up.
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                pending.push((path, depth + 1));
            }
        }
    }
    dedupe_by_target(files)
}

/// Collapse paths that resolve to the same file.
///
/// A symlink and the model it points at both look like models, so a tree
/// holding both would report the same weights twice and inflate the count the
/// TUI shows. Resolution is per collected model rather than per directory
/// entry, so this stays proportional to the number of models found.
///
/// Where a target and an alias both appear, the target wins, so the path
/// handed to callers is the stable one.
fn dedupe_by_target(files: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut chosen: Vec<(PathBuf, bool)> = Vec::with_capacity(files.len());
    let mut seen: HashMap<PathBuf, usize> = HashMap::new();

    for path in files {
        let target = path.canonicalize().unwrap_or_else(|_| path.clone());
        // Ask whether this entry is a link, rather than comparing it to its
        // canonical form. Those differ whenever any parent directory is a
        // symlink, in which case neither the model nor its alias matches the
        // target and the survivor would come down to directory order. Callers
        // look the model up by stem, so the wrong survivor means a model that
        // cannot be found by its real name.
        let is_alias = std::fs::symlink_metadata(&path)
            .map(|meta| meta.file_type().is_symlink())
            .unwrap_or(false);

        match seen.get(&target) {
            Some(&index) => {
                if chosen[index].1 && !is_alias {
                    chosen[index] = (path, false);
                }
            }
            None => {
                seen.insert(target, chosen.len());
                chosen.push((path, is_alias));
            }
        }
    }
    chosen.into_iter().map(|(path, _)| path).collect()
}

pub fn llamacpp_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLMFIT_MODELS_DIR") {
        PathBuf::from(dir)
    } else if let Some(cache) = dirs::cache_dir() {
        cache.join("llmfit").join("models")
    } else {
        PathBuf::from(".llmfit").join("models")
    }
}

/// Check whether a binary is available on the system PATH.
/// Cross-platform: uses the `which` crate rather than shelling out to a
/// Unix-only `which` command, so it works on Windows too.
pub fn command_exists(name: &str) -> bool {
    which::which(name).is_ok()
}

/// Directory holding the llama.cpp binaries, when the caller supplied one
/// explicitly (the `--llama-cpp-path` CLI flag). Set once, before any provider
/// detection runs, so lookups stay consistent for the life of the process.
static LLAMA_CPP_PATH_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Point llama.cpp binary discovery at `dir` for the rest of this process.
///
/// Takes precedence over `LLAMA_CPP_PATH`. Only the first call has an effect,
/// which keeps discovery from changing shape midway through a run. This exists
/// so a caller can override the location without mutating the process
/// environment, which is `unsafe` and racy once other threads are running.
pub fn set_llama_cpp_path_override(dir: PathBuf) -> bool {
    LLAMA_CPP_PATH_OVERRIDE.set(dir).is_ok()
}

/// Find a binary by checking the explicit override, the `LLAMA_CPP_PATH` env
/// var, common install locations, and finally the system PATH via `which`.
fn find_binary(name: &str) -> Option<String> {
    // 1. An explicit override from the caller wins over the environment.
    if let Some(dir) = LLAMA_CPP_PATH_OVERRIDE.get() {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 2. Check LLAMA_CPP_PATH env var next
    if let Ok(dir) = std::env::var("LLAMA_CPP_PATH") {
        let candidate = PathBuf::from(&dir).join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 3. Check common install locations
    let mut common_dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/opt/llama.cpp/build/bin"),
    ];
    if let Some(home) = dirs::home_dir() {
        common_dirs.push(home.join(".local").join("bin"));
    }
    for dir in common_dirs {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 4. Fall back to PATH lookup
    which::which(name)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Check if a llama-server is reachable at the given URL by probing its
/// health endpoint. Returns `true` if the server responds.
fn probe_llama_server(base_url: &str) -> bool {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    crate::hardware::silent_cmd("curl")
        .args(["-sf", "--max-time", "2", &url])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

impl ModelProvider for LlamaCppProvider {

    fn is_available(&self) -> bool {
        self.llama_cli.is_some() || self.llama_server.is_some() || self.server_running
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// Docker Model Runner provider
// ---------------------------------------------------------------------------

/// Docker Model Runner — Docker Desktop's built-in model serving feature.
///
/// Exposes an OpenAI-compatible API at `http://localhost:12434` by default.
/// Models are listed via `GET /engines` and pulled via `docker model pull`.
pub struct DockerModelRunnerProvider {
    base_url: String,
}

/// Check if Docker Desktop is running on Linux by looking for its socket or process.
/// Returns `true` if Docker Desktop appears to be active, `false` otherwise.
/// This avoids a slow HTTP timeout on Linux systems without Docker Desktop.
fn is_docker_desktop_running() -> bool {
    // Docker Desktop on Linux creates a specific socket path
    if std::path::Path::new("/run/docker-desktop/docker.sock").exists()
        || std::path::Path::new(
            &std::env::var("HOME")
                .map(|h| format!("{h}/.docker/desktop/docker.sock"))
                .unwrap_or_default(),
        )
        .exists()
    {
        return true;
    }
    // Fall back to checking if the DOCKER_MODEL_RUNNER_HOST env var is explicitly set
    // to a non-empty value (an empty string means the user hasn't configured it).
    std::env::var("DOCKER_MODEL_RUNNER_HOST")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// Check whether the Docker Desktop application is installed, regardless of
/// whether it is currently running (#731). The Model Runner API probe only
/// succeeds while Docker Desktop is up, so this is what lets the UI say
/// "installed (not running)" instead of "not detected".
pub fn docker_desktop_installed() -> bool {
    docker_desktop_install_candidates(
        std::env::var("ProgramFiles").ok().as_deref(),
        dirs::home_dir().as_deref(),
    )
    .iter()
    .any(|p| p.exists())
}

/// Filesystem locations that identify a Docker Desktop (or docker-model
/// plugin) install. Pure so tests can cover the per-OS layouts; `exists()`
/// checks happen in [`docker_desktop_installed`].
fn docker_desktop_install_candidates(
    program_files: Option<&str>,
    home: Option<&Path>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(pf) = program_files {
        let docker = Path::new(pf).join("Docker").join("Docker");
        // Classic layout, and the frontend/ layout used by newer releases
        // (e.g. C:\Program Files\Docker\Docker\frontend\Docker Desktop.exe).
        candidates.push(docker.join("Docker Desktop.exe"));
        candidates.push(docker.join("frontend").join("Docker Desktop.exe"));
        candidates.push(Path::new(pf).join("Docker").join("cli-plugins"));
    }
    candidates.push(PathBuf::from("/Applications/Docker.app"));
    candidates.push(PathBuf::from("/opt/docker-desktop"));
    if let Some(home) = home {
        candidates.push(home.join("Applications").join("Docker.app"));
        candidates.push(home.join(".docker").join("desktop"));
        // Standalone Model Runner plugin (docker-model), installable
        // without Docker Desktop on Docker CE.
        let plugins = home.join(".docker").join("cli-plugins");
        candidates.push(plugins.join("docker-model"));
        candidates.push(plugins.join("docker-model.exe"));
    }
    candidates
}

fn normalize_docker_mr_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for DockerModelRunnerProvider {
    fn default() -> Self {
        let base_url = std::env::var("DOCKER_MODEL_RUNNER_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_docker_mr_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse DOCKER_MODEL_RUNNER_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://localhost:12434".to_string());
        Self { base_url }
    }
}

impl DockerModelRunnerProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(all(test, not(target_os = "linux")))]
    fn with_base_url(url: &str) -> Self {
        Self {
            base_url: url.to_string(),
        }
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        // Docker Model Runner is a Docker Desktop feature. On Linux, Docker Desktop
        // is uncommon. Skip the HTTP probe if Docker Desktop is not running to avoid
        // a ~800ms timeout on every startup.
        if cfg!(target_os = "linux") && !is_docker_desktop_running() {
            return (false, HashSet::new(), 0);
        }

        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set, 0);
        };

        let server_header = resp
            .headers()
            .get("server")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            return (true, set, 0);
        };
        // Identity gate (#791, #790): import only when the endpoint positively
        // identifies as Docker Model Runner, so a foreign OpenAI server on the
        // port is not imported. Measured 2026-09-01: a live docker/model-runner
        // lists models with owned_by "docker" (plus a per-model `dmr` object)
        // and sends no Server header. A runner with no models yet returns an
        // empty list with no marker, so that case falls back to the root
        // banner probe.
        let identity = classify_openai_endpoint(server_header.as_deref(), &list);
        let identified = identity == OpenAiEndpointIdentity::DockerModelRunner
            || (list.data.is_empty()
                && identity == OpenAiEndpointIdentity::Unrecognized
                && endpoint_is_docker_model_runner_root(
                    &self.base_url,
                    std::time::Duration::from_millis(800),
                ));
        if !identified {
            return (false, HashSet::new(), 0);
        }
        let engines = list.data;
        let count = engines.len();
        for e in engines {
            let lower = e.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the namespace (e.g. "ai/llama3.1" → "llama3.1")
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
            // Strip quantization tag if present (e.g. "llama3.1:8B-Q4_K_M" → "llama3.1:8b")
            if let Some(base) = lower.split(':').next() {
                set.insert(base.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

impl ModelProvider for DockerModelRunnerProvider {

    fn is_available(&self) -> bool {
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
        else {
            return false;
        };
        let server_header = resp
            .headers()
            .get("server")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            return false;
        };
        // Report available only when the endpoint identifies as Docker Model
        // Runner, matching the identity gate used for model import. An empty
        // model list carries no marker, so it falls back to the root banner.
        let identity = classify_openai_endpoint(server_header.as_deref(), &list);
        identity == OpenAiEndpointIdentity::DockerModelRunner
            || (list.data.is_empty()
                && identity == OpenAiEndpointIdentity::Unrecognized
                && endpoint_is_docker_model_runner_root(
                    &self.base_url,
                    std::time::Duration::from_secs(2),
                ))
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// LM Studio provider
// ---------------------------------------------------------------------------

/// LM Studio — local model server with REST API for model management.
///
/// Exposes an OpenAI-compatible API plus management endpoints at
/// `http://127.0.0.1:1234` by default. Models are downloaded via
/// `POST /api/v1/models/download` and listed via `GET /v1/models`.
pub struct LmStudioProvider {
    base_url: String,
    api_key: Option<String>,
}

/// Check whether the LM Studio application is installed, regardless of
/// whether its local server is running (#731). LM Studio's REST API is off
/// until the user starts the server (or `lms server start`), so the HTTP
/// probe alone reports installed-but-idle copies as missing.
pub fn lmstudio_app_installed() -> bool {
    if command_exists("lms") {
        return true;
    }
    lmstudio_install_candidates(
        std::env::var("ProgramFiles").ok().as_deref(),
        std::env::var("LOCALAPPDATA").ok().as_deref(),
        dirs::home_dir().as_deref(),
    )
    .iter()
    .any(|p| p.exists())
}

/// Filesystem locations that identify an LM Studio install. Pure so tests
/// can cover the per-OS layouts; `exists()` checks happen in
/// [`lmstudio_app_installed`].
fn lmstudio_install_candidates(
    program_files: Option<&str>,
    local_app_data: Option<&str>,
    home: Option<&Path>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    // Windows per-machine install (e.g. C:\Program Files\LM Studio\LM Studio.exe).
    if let Some(pf) = program_files {
        candidates.push(Path::new(pf).join("LM Studio").join("LM Studio.exe"));
    }
    // Windows per-user install (the installer default).
    if let Some(lad) = local_app_data {
        candidates.push(
            Path::new(lad)
                .join("Programs")
                .join("LM Studio")
                .join("LM Studio.exe"),
        );
    }
    candidates.push(PathBuf::from("/Applications/LM Studio.app"));
    if let Some(home) = home {
        candidates.push(home.join("Applications").join("LM Studio.app"));
        // ~/.lmstudio is created on first run on every OS (models, lms CLI).
        candidates.push(home.join(".lmstudio"));
    }
    candidates
}

/// Where LM Studio keeps downloaded models. Pure so tests can cover it
/// without a home directory, mirroring [`lmstudio_install_candidates`].
fn lmstudio_models_dir_for(home: Option<&Path>) -> Option<PathBuf> {
    Some(home?.join(".lmstudio").join("models"))
}

fn lmstudio_models_dir() -> Option<PathBuf> {
    lmstudio_models_dir_for(dirs::home_dir().as_deref())
}

/// Identifiers for every model in LM Studio's models directory.
///
/// The HTTP API only advertises models that are currently *loaded*, so a
/// library of thirteen with one loaded is reported as one. The download is
/// what makes a model installed, not whether it happens to be resident, so
/// the directory is the only complete picture.
///
/// Layout is `<models>/<publisher>/<repo>/<file>.gguf`. These names are
/// matched by equality (see [`is_model_installed_lmstudio_disk`]) rather than
/// the substring rule used for API ids, so only canonical forms go in:
/// feeding loose stems to a substring matcher makes `llama` match
/// `meta-llama-3.1-8b-instruct`.
///
/// Returns `(names, model_count)`. `mmproj-*` files are vision projectors
/// shipped beside a model rather than models of their own, so they are not
/// counted.
pub fn scan_lmstudio_models_dir() -> (HashSet<String>, usize) {
    scan_lmstudio_models_dir_at(lmstudio_models_dir().as_deref())
}

fn scan_lmstudio_models_dir_at(root: Option<&Path>) -> (HashSet<String>, usize) {
    let mut set = HashSet::new();
    let mut repos = HashSet::new();
    let Some(root) = root else {
        return (set, 0);
    };

    for path in collect_gguf_files(root, GGUF_SCAN_MAX_DEPTH) {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let stem = stem.to_lowercase();
        if stem.starts_with("mmproj-") {
            continue;
        }

        set.insert(stem.clone());
        if let Some(base) = strip_gguf_quant_suffix(&stem) {
            set.insert(base);
        }

        let repo = path
            .parent()
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().to_lowercase());
        let publisher = path
            .parent()
            .and_then(|d| d.parent())
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().to_lowercase());

        if let Some(repo) = repo {
            repos.insert(repo.clone());
            // LM Studio appends `-GGUF` to the repo it downloaded from; the
            // catalog id does not carry it.
            let trimmed = repo.trim_end_matches("-gguf").to_string();
            if let Some(publisher) = publisher {
                set.insert(format!("{publisher}/{repo}"));
                set.insert(format!("{publisher}/{trimmed}"));
            }
            set.insert(trimmed);
            set.insert(repo);
        }
    }

    let count = repos.len();
    (set, count)
}

/// Candidate ids for the equality-matched disk path.
///
/// Deliberately not [`hf_name_to_lmstudio_candidates`], which also offers a
/// form with `-instruct`, `-chat`, `-hf` and `-it` removed. Widening the net
/// that way is reasonable for a substring search, but under equality it
/// collides: it reduces the distinct `gemma-4-12b-it` entry to `gemma-4-12b`,
/// so an installed base model would mark the IT variant installed too.
fn hf_name_to_lmstudio_disk_candidates(hf_name: &str) -> Vec<String> {
    let full = hf_name.to_lowercase();
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    if repo == full {
        vec![full]
    } else {
        vec![full, repo]
    }
}

/// Is this catalog model present in LM Studio's models directory?
///
/// Deliberately equality rather than the substring test used for API ids:
/// these names come from directory and file names, which are numerous enough
/// that a substring rule starts matching unrelated catalog entries.
pub fn is_model_installed_lmstudio_disk(hf_name: &str, on_disk: &HashSet<String>) -> bool {
    hf_name_to_lmstudio_disk_candidates(hf_name)
        .iter()
        .any(|candidate| on_disk.contains(candidate))
}

fn normalize_lmstudio_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for LmStudioProvider {
    fn default() -> Self {
        let base_url = std::env::var("LMSTUDIO_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_lmstudio_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse LMSTUDIO_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://127.0.0.1:1234".to_string());
        let api_key = std::env::var("LMSTUDIO_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        Self { base_url, api_key }
    }
}

impl LmStudioProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_base_url(url: &str) -> Self {
        Self {
            base_url: url.to_string(),
            api_key: None,
        }
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ({
            let mut req = ureq::get(&self.models_url())
                .config()
                .timeout_global(Some(std::time::Duration::from_millis(800)))
                .build();
            if let Some(ref key) = self.api_key {
                req = req.header("Authorization", &format!("Bearer {}", key));
            }
            req.call()
        }) else {
            return (false, set, 0);
        };

        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            return (true, set, 0);
        };
        // Identity gate (#791, #790): LM Studio is identified by its native
        // /api/v0 API, not by the OpenAI-compatible /v1 shape (a generic Express
        // server would share the framework header). Import only when the
        // endpoint answers /api/v0/models with LM Studio's native schema.
        // Measured 2026-09-01 against LM Studio 0.4.23.
        if !endpoint_is_lmstudio(
            &self.base_url,
            self.api_key.as_deref(),
            std::time::Duration::from_millis(800),
        ) {
            return (false, set, 0);
        }
        let models = list.data;
        let count = models.len();
        for m in models {
            let lower = m.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the publisher (e.g. "lmstudio-community/Qwen3-1.7B-MLX-4bit" → "qwen3-1.7b-mlx-4bit")
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

impl ModelProvider for LmStudioProvider {

    fn is_available(&self) -> bool {
        // Report available only when the native /api/v0 API confirms LM Studio,
        // matching the identity gate used for model import.
        endpoint_is_lmstudio(
            &self.base_url,
            self.api_key.as_deref(),
            std::time::Duration::from_secs(2),
        )
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// LM Studio name-matching helpers
// ---------------------------------------------------------------------------

/// LM Studio uses HuggingFace model names directly. We match against the
/// model's GGUF sources and common naming patterns.
pub fn hf_name_to_lmstudio_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    let mut candidates = vec![hf_name.to_lowercase()];
    if repo != hf_name.to_lowercase() {
        candidates.push(repo.clone());
    }
    // Strip common suffixes for matching
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    if stripped != repo {
        candidates.push(stripped);
    }
    candidates
}

/// Check if any LM Studio candidates for an HF model appear in the installed set.
pub fn is_model_installed_lmstudio(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_lmstudio_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| installed_name.contains(candidate))
    })
}

// ---------------------------------------------------------------------------
// vLLM provider
// ---------------------------------------------------------------------------

/// vLLM — high-throughput inference server with an OpenAI-compatible API.
///
/// Exposes `GET /v1/models` to list loaded models at
/// `http://localhost:8000` by default. Override with `VLLM_HOST`.
///
/// vLLM does not have a pull/download endpoint — models are loaded at
/// server start via HuggingFace.
pub struct VllmProvider {
    base_url: String,
}

fn normalize_vllm_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for VllmProvider {
    fn default() -> Self {
        let base_url = std::env::var("VLLM_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_vllm_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse VLLM_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://localhost:8000".to_string());
        Self { base_url }
    }
}

impl VllmProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_base_url(url: &str) -> Self {
        Self {
            base_url: url.to_string(),
        }
    }

    fn models_url(&self) -> String {
        openai_models_url(&self.base_url)
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set, 0);
        };

        let server_header = resp
            .headers()
            .get("server")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            if endpoint_has_omlx_status(&self.base_url, std::time::Duration::from_millis(800)) {
                return (false, set, 0);
            }
            return (true, set, 0);
        };
        if openai_model_list_is_omlx(&list)
            || (list.data.is_empty()
                && endpoint_has_omlx_status(&self.base_url, std::time::Duration::from_millis(800)))
        {
            return (false, set, 0);
        }
        // Identity gate (#791, #790): import only when the endpoint positively
        // identifies as vLLM. Measured 2026-09-01: vLLM 0.28.0 lists models with
        // owned_by "vllm" (Server: uvicorn is shared by any FastAPI app, so
        // owned_by is the discriminator). An empty list never comes from vLLM
        // itself: vllm serve refuses to start without a model to serve.
        if classify_openai_endpoint(server_header.as_deref(), &list) != OpenAiEndpointIdentity::Vllm
        {
            return (false, set, 0);
        }
        let models = list.data;
        let count = models.len();
        for m in models {
            let lower = m.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the publisher
            // e.g. "meta-llama/Llama-3.1-8B-Instruct" → "llama-3.1-8b-instruct"
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

impl ModelProvider for VllmProvider {

    fn is_available(&self) -> bool {
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
        else {
            return false;
        };
        let server_header = resp
            .headers()
            .get("server")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            return false;
        };
        // Report available only when the endpoint identifies as vLLM, matching
        // the identity gate used for model import (a foreign OpenAI server on
        // the port is neither available nor imported).
        classify_openai_endpoint(server_header.as_deref(), &list) == OpenAiEndpointIdentity::Vllm
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// vLLM name-matching helpers
// ---------------------------------------------------------------------------

/// vLLM uses HuggingFace model names directly. We match against the
/// model's full HF name and common naming patterns.
pub fn hf_name_to_vllm_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    let mut candidates = vec![hf_name.to_lowercase()];
    if repo != hf_name.to_lowercase() {
        candidates.push(repo.clone());
    }
    // Strip common suffixes for matching
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    if stripped != repo {
        candidates.push(stripped);
    }
    candidates
}

/// Check if any vLLM candidates for an HF model appear in the installed set.
pub fn is_model_installed_vllm(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_vllm_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| installed_name.contains(candidate))
    })
}

// ---------------------------------------------------------------------------
// RamaLama provider
// ---------------------------------------------------------------------------

/// RamaLama — container-based model runner with an OpenAI-compatible API.
///
/// Exposes `GET /v1/models` to list served models at
/// `http://localhost:8080` by default. Override with `RAMALAMA_HOST`.
///
/// Like vLLM, RamaLama has no runtime pull endpoint — models are served
/// via `ramalama serve <model>`.
pub struct RamaLamaProvider {
    base_url: String,
}

fn normalize_ramalama_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for RamaLamaProvider {
    fn default() -> Self {
        let base_url = std::env::var("RAMALAMA_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_ramalama_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse RAMALAMA_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://localhost:8080".to_string());
        Self { base_url }
    }
}

impl RamaLamaProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_base_url(url: &str) -> Self {
        Self {
            base_url: url.to_string(),
        }
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    /// Single-pass startup probe.
    ///
    /// Prefers the running server's `/v1/models`. When that is unreachable,
    /// falls back to the local store via `ramalama ls --json`, so installed
    /// models are still detected without a served endpoint (mirrors how Docker
    /// Model Runner is recognized while "installed but not running").
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            // Server not reachable — fall back to the on-disk store.
            return match Self::installed_from_store() {
                Some((set, count)) => (true, set, count),
                None => (false, HashSet::new(), 0),
            };
        };

        let server_header = resp
            .headers()
            .get("server")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let Ok(list) = resp.into_body().read_json::<OpenAiModelList>() else {
            return (true, HashSet::new(), 0);
        };
        // Shared identity gate (#791, #790): a llama-swap proxy answering on
        // this port is not RamaLama, so treat it like no server at all and
        // fall back to the on-disk store. A plain llama.cpp identity is not
        // rejected on this path: `ramalama serve` drives llama-server inside
        // its container, so a genuine endpoint may present it.
        if classify_openai_endpoint(server_header.as_deref(), &list)
            == OpenAiEndpointIdentity::LlamaSwap
        {
            return match Self::installed_from_store() {
                Some((set, count)) => (true, set, count),
                None => (false, HashSet::new(), 0),
            };
        }
        let count = list.data.len();
        let mut set = HashSet::new();
        for m in list.data {
            insert_ramalama_name(&mut set, &m.id);
        }
        (true, set, count)
    }

    /// Detect installed models from the local RamaLama store using the CLI,
    /// so detection works without a running server. Returns `None` when the
    /// `ramalama` binary is absent or the command fails.
    fn installed_from_store() -> Option<(HashSet<String>, usize)> {
        let mut child = crate::hardware::silent_cmd("ramalama")
            .args(["ls", "--json"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()?;
        let mut stdout = child.stdout.take()?;
        let reader = std::thread::spawn(move || {
            let mut output = Vec::new();
            std::io::Read::read_to_end(&mut stdout, &mut output).map(|_| output)
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        let _ = reader.join();
                        return None;
                    }
                    let output = reader.join().ok()?.ok()?;
                    return parse_ramalama_store(&output);
                }
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                    return None;
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                    return None;
                }
            }
        }
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

/// A row from `ramalama ls --json`. Extra fields (modified, size) are ignored.
#[derive(serde::Deserialize)]
struct RamaLamaStoreModel {
    /// Transport-qualified name, e.g. "huggingface://meta-llama/Llama-3.1-8B-Instruct".
    name: String,
    /// Optional friendly alias from shortnames.conf; empty when unset.
    #[serde(default)]
    shortname: String,
}

/// Insert a RamaLama model identifier into the installed set: the full
/// lowercased identifier plus its trailing path component, so both
/// "huggingface://meta-llama/llama-3.1-8b-instruct" and "llama-3.1-8b-instruct"
/// match. Matching against these is substring-based (see
/// `is_model_installed_ramalama`).
fn insert_ramalama_name(set: &mut HashSet<String>, raw: &str) {
    let lower = raw.to_lowercase();
    if let Some(name) = lower.split('/').next_back()
        && name != lower
    {
        set.insert(name.to_string());
    }
    set.insert(lower);
}

/// Parse `ramalama ls --json` output into `(installed_set, count)`.
fn parse_ramalama_store(json: &[u8]) -> Option<(HashSet<String>, usize)> {
    let models: Vec<RamaLamaStoreModel> = serde_json::from_slice(json).ok()?;
    let count = models.len();
    let mut set = HashSet::new();
    for m in &models {
        insert_ramalama_name(&mut set, &m.name);
        if !m.shortname.is_empty() {
            insert_ramalama_name(&mut set, &m.shortname);
        }
    }
    Some((set, count))
}

impl ModelProvider for RamaLamaProvider {

    fn is_available(&self) -> bool {
        ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

}

// ---------------------------------------------------------------------------
// RamaLama name-matching helpers
// ---------------------------------------------------------------------------

/// RamaLama serves HuggingFace/OCI model names directly. We match against the
/// model's full HF name and common naming patterns.
pub fn hf_name_to_ramalama_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    let mut candidates = vec![hf_name.to_lowercase()];
    if repo != hf_name.to_lowercase() {
        candidates.push(repo.clone());
    }
    // Strip common suffixes for matching
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    if stripped != repo {
        candidates.push(stripped);
    }
    candidates
}

/// Check if any RamaLama candidates for an HF model appear in the installed set.
pub fn is_model_installed_ramalama(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_ramalama_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| installed_name.contains(candidate))
    })
}

// ---------------------------------------------------------------------------
// Docker Model Runner name-matching helpers
// ---------------------------------------------------------------------------

/// Embedded catalog of HF models confirmed to exist in Docker Hub's ai/ namespace.
/// Generated by `scripts/scrape_docker_models.py` and refreshed alongside the model DB.
const DOCKER_MODELS_JSON: &str = include_str!("../data/docker_models.json");

#[derive(serde::Deserialize)]
struct DockerModelCatalog {
    models: Vec<DockerModelEntry>,
}

#[derive(serde::Deserialize)]
struct DockerModelEntry {
    hf_name: String,
    docker_tag: String,
}

/// Lazily parsed Docker Model Runner catalog.
fn docker_mr_catalog() -> &'static [(String, String)] {
    use std::sync::OnceLock;
    static CATALOG: OnceLock<Vec<(String, String)>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let Ok(catalog) = serde_json::from_str::<DockerModelCatalog>(DOCKER_MODELS_JSON) else {
            return Vec::new();
        };
        catalog
            .models
            .into_iter()
            .map(|e| (e.hf_name.to_lowercase(), e.docker_tag))
            .collect()
    })
}

/// Given an HF model name, return the Docker Model Runner tag to use for pulling.
/// Returns `None` if the model has no confirmed Docker image.
pub fn docker_mr_pull_tag(hf_name: &str) -> Option<String> {
    let lower = hf_name.to_lowercase();
    docker_mr_catalog()
        .iter()
        .find(|(name, _)| *name == lower)
        .map(|(_, tag)| tag.clone())
}

/// Docker Model Runner uses the Ollama naming convention (e.g. "ai/llama3.1:8b").
/// We generate candidates from the confirmed catalog, plus base-name variants for
/// matching against locally installed models.
pub fn hf_name_to_docker_mr_candidates(hf_name: &str) -> Vec<String> {
    let Some(tag) = docker_mr_pull_tag(hf_name) else {
        return Vec::new();
    };
    let mut candidates = vec![tag.clone()];
    // Also add without "ai/" prefix for matching installed models
    if let Some(stripped) = tag.strip_prefix("ai/") {
        candidates.push(stripped.to_string());
    }
    // Add base repo name (without size tag) e.g. "ai/llama3.1"
    if let Some(base) = tag.split(':').next() {
        candidates.push(base.to_string());
    }
    candidates
}

/// Check if any of the Docker Model Runner candidates for an HF model
/// appear in the installed set.
pub fn is_model_installed_docker_mr(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_docker_mr_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| docker_mr_installed_matches(installed_name, candidate))
    })
}

fn docker_mr_installed_matches(installed_name: &str, candidate: &str) -> bool {
    if installed_name == candidate {
        return true;
    }
    // Allow variant tags, e.g. candidate "ai/llama3.1:8b" matching
    // installed "ai/llama3.1:8b-q4_k_m"
    if candidate.contains(':') {
        return installed_name.starts_with(&format!("{candidate}-"));
    }
    false
}

/// Strip quantization suffix from a GGUF file stem.
/// "llama-3.1-8b-instruct-q4_k_m" → "llama-3.1-8b-instruct"
pub fn strip_gguf_quant_suffix(stem: &str) -> Option<String> {
    // Matched on the quant *family* stem rather than each published variant:
    // K-quants as `-qN_k` and I-quants as `-iqN_`, so `_S`/`_M`/`_L`/`_XL`
    // and `_NL`/`_XS`/`_XXS` all reduce to the same base. Enumerating the
    // variants instead left whole publishers unmatched — bartowski `_L` files
    // and Unsloth Dynamic `_XL`/`IQ4_NL` files never reached the catalog id,
    // so they read as neither installed nor served.
    let quant_patterns = [
        "-q8_0", "-q8_k", "-q6_k", "-q5_k", "-q4_k", "-q4_0", "-q3_k", "-q2_k", "-iq4_", "-iq3_",
        "-iq2_", "-iq1_", "-f16", "-f32", "-bf16", ".q8_0", ".q6_k", ".q5_k", ".q4_k", ".q4_0",
        ".q3_k", ".q2_k",
    ];
    for pat in &quant_patterns {
        if let Some(pos) = stem.rfind(pat) {
            let base = &stem[..pos];
            // Unsloth "Dynamic" GGUFs embed a `-ud` marker between the model
            // name and the quant (e.g. `qwen3.6-35b-a3b-ud-q4_k_m`). It is not
            // part of the canonical model name, so strip it too — otherwise the
            // stem never reduces to the catalog id and the file reads as neither
            // installed nor served.
            let base = base.strip_suffix("-ud").unwrap_or(base);
            return Some(base.to_string());
        }
    }
    None
}

/// Strip an MLX quantization suffix from a lowercased model stem, so
/// mlx-community basenames reduce to catalog slugs (#854, #869).
/// "llama-3.2-1b-instruct-4bit" → "llama-3.2-1b-instruct"
///
/// End-anchored, unlike the GGUF list above: mlx-community always places the
/// quant scheme last, and dtype-like fragments can occur inside genuine model
/// names. Strips, in order: the quant scheme (`-<N>bit` or `-mxfp4`, each
/// with optional trailing variant markers such as `-4bit-dwq`, date-stamped
/// `-4bit-dwq-05082025`, `-mxfp4-q8`, `-mxfp4-bf16`, or a plain `-fp16`),
/// then a trailing `-mlx` marker (`...-instruct-mlx-8bit` → `...-instruct`).
///
/// Returns `None` when nothing was stripped, so lookup callers can tell "no
/// MLX suffix present" apart from "already reduced".
pub fn strip_mlx_quant_suffix(stem: &str) -> Option<String> {
    static MLX_QUANT_SUFFIX: OnceLock<Regex> = OnceLock::new();
    let re = MLX_QUANT_SUFFIX.get_or_init(|| {
        Regex::new(r"-(?:\d+bit(?:-[a-z0-9]+)*|mxfp4(?:-[a-z0-9]+)*|fp16)$")
            .expect("valid MLX suffix regex")
    });
    let without_quant = match re.find(stem) {
        Some(m) if m.start() > 0 => &stem[..m.start()],
        _ => stem,
    };
    let base = match without_quant.strip_suffix("-mlx") {
        Some(b) if !b.is_empty() => b,
        _ => without_quant,
    };
    if base.len() == stem.len() {
        return None;
    }
    let base = base.trim_matches('-');
    if base.is_empty() {
        return None;
    }
    Some(base.to_string())
}

// ---------------------------------------------------------------------------
// llama.cpp name-matching helpers
// ---------------------------------------------------------------------------

/// Check if a model is installed in the llama.cpp cache.
pub fn is_model_installed_llamacpp(hf_name: &str, installed: &HashSet<String>) -> bool {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();

    // Direct match on model name stem. The installed set already contains
    // both raw file stems and quant-suffix-stripped bases (see
    // `installed_models_counted`), so exact lookups cover files like
    // "qwen2.5-7b-instruct-q4_k_m.gguf" matched against the plain repo name.
    if installed.contains(&repo) {
        return true;
    }

    // Also accept a match with common variant suffixes stripped.
    //
    // Deliberately no substring matching here: a single "gemma-3.gguf" on
    // disk must not mark every gemma-3-* model in the database as installed
    // (`repo.contains("gemma-3")` is true for all of them).
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    installed.contains(&stripped)
}

// ---------------------------------------------------------------------------
// MLX name-matching helpers
// ---------------------------------------------------------------------------

fn push_unique_candidate(candidates: &mut Vec<String>, candidate: String) {
    if !candidate.is_empty() && !candidates.iter().any(|c| c == &candidate) {
        candidates.push(candidate);
    }
}

/// Normalize an MLX repo basename to its catalog base: thin wrapper over
/// [`strip_mlx_quant_suffix`] for callers that want the input back (dashes
/// trimmed) when no MLX suffix is present (#869).
fn normalize_mlx_repo_base(repo_lower: &str) -> String {
    strip_mlx_quant_suffix(repo_lower).unwrap_or_else(|| repo_lower.trim_matches('-').to_string())
}

fn strip_trailing_common_model_suffixes(name: &str) -> String {
    let mut out = name.to_string();
    loop {
        let mut changed = false;
        for suffix in ["-instruct", "-chat", "-hf", "-it", "-base"] {
            if let Some(stripped) = out.strip_suffix(suffix) {
                out = stripped.trim_end_matches('-').to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn explicit_mlx_repo_id(hf_name: &str) -> Option<String> {
    if hf_name.matches('/').count() != 1 {
        return None;
    }
    let mut parts = hf_name.splitn(2, '/');
    let owner = parts.next()?.trim();
    let repo = parts.next()?.trim();
    if owner.is_empty() || repo.is_empty() || !is_likely_mlx_repo(owner, repo) {
        return None;
    }
    Some(format!("{}/{}", owner.to_lowercase(), repo.to_lowercase()))
}

/// Map a HuggingFace model name to mlx-community repo name candidates.
/// Pattern: mlx-community/{RepoName}-{quant}bit
pub fn hf_name_to_mlx_candidates(hf_name: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    if let Some(repo_id) = explicit_mlx_repo_id(hf_name) {
        push_unique_candidate(&mut candidates, repo_id.clone());
        if let Some(repo_name) = repo_id.split('/').next_back() {
            push_unique_candidate(&mut candidates, repo_name.to_string());
        }
    }

    let repo = hf_name.split('/').next_back().unwrap_or(hf_name);
    let repo_lower = repo.to_lowercase();
    push_unique_candidate(&mut candidates, repo_lower.clone());

    let normalized_repo = normalize_mlx_repo_base(&repo_lower);

    // Explicit mappings: HF repo suffix → mlx-community repo name (without quant suffix)
    let mappings: &[(&str, &str)] = &[
        // Meta Llama
        ("Llama-3.3-70B-Instruct", "Llama-3.3-70B-Instruct"),
        ("Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
        ("Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
        ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("Llama-3.1-70B-Instruct", "Llama-3.1-70B-Instruct"),
        // Qwen
        ("Qwen2.5-72B-Instruct", "Qwen2.5-72B-Instruct"),
        ("Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct"),
        ("Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
        ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
        ("Qwen2.5-Coder-32B-Instruct", "Qwen2.5-Coder-32B-Instruct"),
        ("Qwen2.5-Coder-14B-Instruct", "Qwen2.5-Coder-14B-Instruct"),
        ("Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct"),
        ("Qwen3-32B", "Qwen3-32B"),
        ("Qwen3-14B", "Qwen3-14B"),
        ("Qwen3-8B", "Qwen3-8B"),
        ("Qwen3-4B", "Qwen3-4B"),
        ("Qwen3-1.7B", "Qwen3-1.7B"),
        ("Qwen3-0.6B", "Qwen3-0.6B"),
        ("Qwen3-30B-A3B", "Qwen3-30B-A3B"),
        ("Qwen3-235B-A22B", "Qwen3-235B-A22B"),
        // Qwen3.5
        ("Qwen3.5-0.6B", "Qwen3.5-0.6B"),
        ("Qwen3.5-1.7B", "Qwen3.5-1.7B"),
        ("Qwen3.5-4B", "Qwen3.5-4B"),
        ("Qwen3.5-8B", "Qwen3.5-8B"),
        ("Qwen3.5-9B", "Qwen3.5-9B"),
        ("Qwen3.5-14B", "Qwen3.5-14B"),
        ("Qwen3.5-27B", "Qwen3.5-27B"),
        ("Qwen3.5-32B", "Qwen3.5-32B"),
        ("Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),
        ("Qwen3.5-72B", "Qwen3.5-72B"),
        ("Qwen3.5-122B-A10B", "Qwen3.5-122B-A10B"),
        ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
        // Mistral
        ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
        (
            "Mistral-Small-24B-Instruct-2501",
            "Mistral-Small-24B-Instruct-2501",
        ),
        ("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
        (
            "Mistral-Small-3.1-24B-Instruct-2503",
            "Mistral-Small-3.1-24B-Instruct-2503",
        ),
        ("Ministral-8B-Instruct-2410", "Ministral-8B-Instruct-2410"),
        ("Mistral-Nemo-Instruct-2407", "Mistral-Nemo-Instruct-2407"),
        // DeepSeek
        (
            "DeepSeek-R1-Distill-Qwen-32B",
            "DeepSeek-R1-Distill-Qwen-32B",
        ),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
        (
            "DeepSeek-R1-Distill-Qwen-14B",
            "DeepSeek-R1-Distill-Qwen-14B",
        ),
        (
            "DeepSeek-R1-Distill-Llama-8B",
            "DeepSeek-R1-Distill-Llama-8B",
        ),
        (
            "DeepSeek-R1-Distill-Llama-70B",
            "DeepSeek-R1-Distill-Llama-70B",
        ),
        // Gemma
        ("gemma-3-12b-it", "gemma-3-12b-it"),
        ("gemma-2-27b-it", "gemma-2-27b-it"),
        ("gemma-2-9b-it", "gemma-2-9b-it"),
        ("gemma-2-2b-it", "gemma-2-2b-it"),
        ("gemma-3-1b-it", "gemma-3-1b-it"),
        ("gemma-3-4b-it", "gemma-3-4b-it"),
        ("gemma-3-27b-it", "gemma-3-27b-it"),
        ("gemma-3n-E4B-it", "gemma-3n-E4B-it"),
        ("gemma-3n-E2B-it", "gemma-3n-E2B-it"),
        // Phi
        ("Phi-4", "Phi-4"),
        ("Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct"),
        ("Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct"),
        ("Phi-4-mini-instruct", "Phi-4-mini-instruct"),
        ("Phi-4-reasoning", "Phi-4-reasoning"),
        ("Phi-4-mini-reasoning", "Phi-4-mini-reasoning"),
        // Llama 4
        (
            "Llama-4-Scout-17B-16E-Instruct",
            "Llama-4-Scout-17B-16E-Instruct",
        ),
        (
            "Llama-4-Maverick-17B-128E-Instruct",
            "Llama-4-Maverick-17B-128E-Instruct",
        ),
    ];

    for &(hf_suffix, mlx_base) in mappings {
        let mapped_suffix = hf_suffix.to_lowercase();
        if repo_lower == mapped_suffix || normalized_repo == mapped_suffix {
            let base_lower = mlx_base.to_lowercase();
            push_unique_candidate(&mut candidates, format!("{}-4bit", base_lower));
            push_unique_candidate(&mut candidates, format!("{}-8bit", base_lower));
            push_unique_candidate(&mut candidates, base_lower);
            return candidates;
        }
    }

    // Fallback heuristic: normalize explicit MLX names and try common variants.
    if !normalized_repo.is_empty() {
        push_unique_candidate(&mut candidates, format!("{}-4bit", normalized_repo));
        push_unique_candidate(&mut candidates, format!("{}-8bit", normalized_repo));
        // Some mlx-community repos use a -MLX- infix (e.g. Model-MLX-4bit)
        push_unique_candidate(&mut candidates, format!("{}-mlx-4bit", normalized_repo));
        push_unique_candidate(&mut candidates, format!("{}-mlx-8bit", normalized_repo));
        push_unique_candidate(&mut candidates, normalized_repo.clone());
    }

    let stripped = strip_trailing_common_model_suffixes(&normalized_repo);
    if !stripped.is_empty() && stripped != normalized_repo {
        push_unique_candidate(&mut candidates, format!("{}-4bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-8bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-mlx-4bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-mlx-8bit", stripped));
        push_unique_candidate(&mut candidates, stripped);
    }

    candidates
}

/// Check if any MLX candidates for an HF model appear in the installed set.
pub fn is_model_installed_mlx(hf_name: &str, installed: &HashSet<String>) -> bool {
    // Quick check: installed set may contain the full HF name (lowercased)
    if installed.contains(&hf_name.to_lowercase()) {
        return true;
    }

    let candidates = hf_name_to_mlx_candidates(hf_name);
    candidates.iter().any(|c| installed.contains(c))
}

// ---------------------------------------------------------------------------
// Ollama name-matching helpers
// ---------------------------------------------------------------------------

/// Authoritative mapping from HF repo name (lowercased, after slash) to Ollama tag.
/// Only models with a known Ollama registry entry are listed here.
/// If a model is not in this table, it cannot be pulled from Ollama.
const OLLAMA_MAPPINGS: &[(&str, &str)] = &[
    // Meta Llama family
    ("llama-3.3-70b-instruct", "llama3.3:70b"),
    ("llama-3.2-11b-vision-instruct", "llama3.2-vision:11b"),
    ("llama-3.2-3b-instruct", "llama3.2:3b"),
    ("llama-3.2-3b", "llama3.2:3b"),
    ("llama-3.2-1b-instruct", "llama3.2:1b"),
    ("llama-3.2-1b", "llama3.2:1b"),
    ("llama-3.1-405b-instruct", "llama3.1:405b"),
    ("llama-3.1-405b", "llama3.1:405b"),
    ("llama-3.1-70b-instruct", "llama3.1:70b"),
    ("llama-3.1-8b-instruct", "llama3.1:8b"),
    ("llama-3.1-8b", "llama3.1:8b"),
    ("meta-llama-3-8b-instruct", "llama3:8b"),
    ("meta-llama-3-8b", "llama3:8b"),
    ("llama-2-7b-hf", "llama2:7b"),
    ("codellama-34b-instruct-hf", "codellama:34b"),
    ("codellama-13b-instruct-hf", "codellama:13b"),
    ("codellama-7b-instruct-hf", "codellama:7b"),
    // Google Gemma
    ("gemma-3-27b-it", "gemma3:27b"),
    ("gemma-3-12b-it", "gemma3:12b"),
    ("gemma-3-4b-it", "gemma3:4b"),
    ("gemma-3-1b-it", "gemma3:1b"),
    ("gemma-2-27b-it", "gemma2:27b"),
    ("gemma-2-9b-it", "gemma2:9b"),
    ("gemma-2-2b-it", "gemma2:2b"),
    // Microsoft Phi
    ("phi-4", "phi4"),
    ("phi-4-mini-instruct", "phi4-mini"),
    ("phi-3.5-mini-instruct", "phi3.5"),
    ("phi-3-mini-4k-instruct", "phi3"),
    ("phi-3-medium-14b-instruct", "phi3:14b"),
    ("phi-2", "phi"),
    ("orca-2-7b", "orca2:7b"),
    ("orca-2-13b", "orca2:13b"),
    // Mistral
    ("mistral-7b-instruct-v0.3", "mistral:7b"),
    ("mistral-7b-instruct-v0.2", "mistral:7b"),
    ("mistral-nemo-instruct-2407", "mistral-nemo"),
    ("mistral-small-24b-instruct-2501", "mistral-small:24b"),
    ("mistral-small-3.1-24b-instruct-2503", "mistral-small3.1"),
    ("mistral-large-instruct-2407", "mistral-large"),
    ("devstral-small-2505", "devstral"),
    ("mixtral-8x7b-instruct-v0.1", "mixtral:8x7b"),
    ("mixtral-8x22b-instruct-v0.1", "mixtral:8x22b"),
    // Qwen 2 / 2.5
    ("qwen2-1.5b-instruct", "qwen2:1.5b"),
    ("qwen2.5-72b-instruct", "qwen2.5:72b"),
    ("qwen2.5-32b-instruct", "qwen2.5:32b"),
    ("qwen2.5-14b-instruct", "qwen2.5:14b"),
    ("qwen2.5-7b-instruct", "qwen2.5:7b"),
    ("qwen2.5-7b", "qwen2.5:7b"),
    ("qwen2.5-3b-instruct", "qwen2.5:3b"),
    ("qwen2.5-1.5b-instruct", "qwen2.5:1.5b"),
    ("qwen2.5-1.5b", "qwen2.5:1.5b"),
    ("qwen2.5-0.5b-instruct", "qwen2.5:0.5b"),
    ("qwen2.5-0.5b", "qwen2.5:0.5b"),
    ("qwen2.5-coder-32b-instruct", "qwen2.5-coder:32b"),
    ("qwen2.5-coder-14b-instruct", "qwen2.5-coder:14b"),
    ("qwen2.5-coder-7b-instruct", "qwen2.5-coder:7b"),
    ("qwen2.5-coder-1.5b-instruct", "qwen2.5-coder:1.5b"),
    ("qwen2.5-coder-0.5b-instruct", "qwen2.5-coder:0.5b"),
    ("qwen2.5-vl-72b-instruct", "qwen2.5vl:72b"),
    ("qwen2.5-vl-7b-instruct", "qwen2.5vl:7b"),
    ("qwen2.5-vl-3b-instruct", "qwen2.5vl:3b"),
    ("qwq-32b", "qwq"),
    // Qwen 3
    ("qwen3-235b-a22b", "qwen3:235b"),
    ("qwen3-32b", "qwen3:32b"),
    ("qwen3-30b-a3b", "qwen3:30b-a3b"),
    ("qwen3-30b-a3b-instruct-2507", "qwen3:30b-a3b"),
    ("qwen3-14b", "qwen3:14b"),
    ("qwen3-8b", "qwen3:8b"),
    ("qwen3-4b", "qwen3:4b"),
    ("qwen3-4b-instruct-2507", "qwen3:4b"),
    ("qwen3-1.7b-base", "qwen3:1.7b"),
    ("qwen3-0.6b", "qwen3:0.6b"),
    ("qwen3-coder-30b-a3b-instruct", "qwen3-coder"),
    // Qwen 3.5
    ("qwen3.5-27b", "qwen3.5"),
    ("qwen3.5-35b-a3b", "qwen3.5:35b"),
    ("qwen3.5-122b-a10b", "qwen3.5:122b"),
    // Qwen 3.8 — 27B is the only size Ollama publishes; the 2.4T-A95B MoE
    // has no library entry.
    ("qwen3.8-27b", "qwen3.8:27b"),
    // Qwen3-Coder-Next
    ("qwen3-coder-next", "qwen3-coder-next"),
    // DeepSeek
    ("deepseek-v3", "deepseek-v3"),
    ("deepseek-v3.2", "deepseek-v3"),
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek-r1-0528", "deepseek-r1"),
    ("deepseek-r1-distill-qwen-32b", "deepseek-r1:32b"),
    ("deepseek-r1-distill-qwen-14b", "deepseek-r1:14b"),
    ("deepseek-r1-distill-qwen-7b", "deepseek-r1:7b"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-r1:1.5b"),
    ("deepseek-r1-distill-llama-70b", "deepseek-r1:70b"),
    ("deepseek-r1-distill-llama-8b", "deepseek-r1:8b"),
    ("deepseek-coder-v2-lite-instruct", "deepseek-coder-v2:16b"),
    // Community / other
    ("tinyllama-1.1b-chat-v1.0", "tinyllama"),
    ("stablelm-2-1_6b-chat", "stablelm2:1.6b"),
    ("yi-6b-chat", "yi:6b"),
    ("yi-34b-chat", "yi:34b"),
    ("starcoder2-7b", "starcoder2:7b"),
    ("starcoder2-15b", "starcoder2:15b"),
    ("falcon-7b-instruct", "falcon:7b"),
    ("falcon-40b-instruct", "falcon:40b"),
    ("falcon-180b-chat", "falcon:180b"),
    ("falcon3-1b-instruct", "falcon3:1b"),
    ("falcon3-3b-instruct", "falcon3:3b"),
    ("falcon3-7b-instruct", "falcon3:7b"),
    ("openchat-3.5-0106", "openchat:7b"),
    ("vicuna-7b-v1.5", "vicuna:7b"),
    ("vicuna-13b-v1.5", "vicuna:13b"),
    ("glm-4-9b-chat", "glm4:9b"),
    ("solar-10.7b-instruct-v1.0", "solar:10.7b"),
    ("zephyr-7b-beta", "zephyr:7b"),
    ("c4ai-command-r-v01", "command-r"),
    ("c4ai-command-r-plus-08-2024", "command-r-plus"),
    ("c4ai-command-a-03-2025", "command-a"),
    (
        "nous-hermes-2-mixtral-8x7b-dpo",
        "nous-hermes2-mixtral:8x7b",
    ),
    ("hermes-3-llama-3.1-8b", "hermes3:8b"),
    ("nomic-embed-text-v1.5", "nomic-embed-text"),
    ("bge-large-en-v1.5", "bge-large"),
    ("smollm2-1.7b-instruct", "smollm2:1.7b"),
    ("smollm2-135m-instruct", "smollm2:135m"),
    ("smollm2-135m", "smollm2:135m"),
    // Google Gemma 3n
    ("gemma-3n-e4b-it", "gemma3n:e4b"),
    ("gemma-3n-e2b-it", "gemma3n:e2b"),
    // Microsoft Phi-4 reasoning
    ("phi-4-reasoning", "phi4-reasoning"),
    ("phi-4-mini-reasoning", "phi4-mini-reasoning"),
    // NVIDIA Nemotron
    ("llama-3.1-nemotron-70b-instruct-hf", "nemotron:70b"),
    ("llama-3.3-nemotron-super-49b-v1", "nemotron:49b"),
    // EXAONE Deep reasoning
    ("exaone-deep-2.4b", "exaone-deep:2.4b"),
    ("exaone-deep-7.8b", "exaone-deep:7.8b"),
    ("exaone-deep-32b", "exaone-deep:32b"),
    // OLMo 2
    ("olmo-2-1124-7b-instruct", "olmo2:7b"),
    ("olmo-2-1124-13b-instruct", "olmo2:13b"),
    ("olmo-2-0325-32b-instruct", "olmo2:32b"),
    // DeepSeek V3.2 Speciale (no local Ollama tag yet, maps to v3)
    ("deepseek-v3.2-speciale", "deepseek-v3"),
    // Liquid AI LFM2
    ("lfm2-350m", "lfm2:350m"),
    ("lfm2-700m", "lfm2:700m"),
    ("lfm2-1.2b", "lfm2:1.2b"),
    ("lfm2-2.6b", "lfm2:2.6b"),
    ("lfm2-2.6b-exp", "lfm2:2.6b"),
    ("lfm2-8b-a1b", "lfm2:8b-a1b"),
    ("lfm2-24b-a2b", "lfm2:24b"),
    // Liquid AI LFM2.5
    ("lfm2.5-1.2b-instruct", "lfm2.5:1.2b"),
    ("lfm2.5-1.2b-thinking", "lfm2.5-thinking:1.2b"),
];

/// Split a lowercased model name into (family_name, size_tag) by finding
/// the rightmost segment that looks like a parameter size (e.g. "7b", "70b",
/// "30b-a3b" for MoE).  Returns `None` if no size-like segment is found.
///
/// Examples:
///   "qwen2.5-coder-14b"       → Some(("qwen2.5-coder", "14b"))
///   "deepseek-r1-distill-qwen-32b" → Some(("deepseek-r1-distill-qwen", "32b"))
///   "qwen3-coder-30b-a3b"     → Some(("qwen3-coder", "30b-a3b"))
///   "phi-4"                    → None (no "b" suffix — "4" isn't a size tag)
fn split_name_and_size(name: &str) -> Option<(&str, &str)> {
    // Walk segments from the right looking for one that matches a size
    // pattern like "7b", "70b", "1.7b", "30b-a3b" (MoE active params).
    let segments: Vec<&str> = name.split('-').collect();
    for i in (0..segments.len()).rev() {
        let seg = segments[i];
        // Check for a segment ending in 'b' with digits (e.g. "7b", "70b", "1.7b")
        if seg.ends_with('b') && seg.len() > 1 {
            let before_b = &seg[..seg.len() - 1];
            if before_b.chars().all(|c| c.is_ascii_digit() || c == '.') {
                // Include any trailing MoE segment like "-a3b"
                let size_start = segments[..i]
                    .iter()
                    .map(|s| s.len() + 1) // +1 for the '-'
                    .sum::<usize>();
                if size_start == 0 || size_start > name.len() {
                    return None;
                }
                let family = &name[..size_start - 1]; // trim trailing '-'
                let size = &name[size_start..];
                if !family.is_empty() && !size.is_empty() {
                    return Some((family, size));
                }
            }
        }
    }
    None
}

/// Look up the Ollama tag for an HF repo name. Returns the first match
/// from `OLLAMA_MAPPINGS`, or `None` if the model has no known Ollama equivalent.
fn lookup_ollama_tag(hf_name: &str) -> Option<&'static str> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    OLLAMA_MAPPINGS
        .iter()
        .find(|&&(hf_suffix, _)| repo == hf_suffix)
        .map(|&(_, tag)| tag)
}

/// Map a HuggingFace model name to Ollama candidate tags for install checking.
/// Tries the authoritative mapping table first, then falls back to heuristic
/// candidate generation so models without explicit mappings can still be
/// detected as installed.
pub fn hf_name_to_ollama_candidates(hf_name: &str) -> Vec<String> {
    if let Some(tag) = lookup_ollama_tag(hf_name) {
        return vec![tag.to_string()];
    }

    // Fallback: generate candidates from the HF repo name convention.
    // e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct" → ["qwen3-coder-30b-a3b", "qwen3-coder:30b-a3b", ...]
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();

    let base = strip_trailing_common_model_suffixes(&repo);

    let mut candidates = Vec::new();

    // Try to split off the size tag (e.g. "qwen3-coder-30b-a3b" → ("qwen3-coder", "30b-a3b"))
    // Ollama uses "name:size" format, so we look for a size-like segment.
    if let Some((name, size)) = split_name_and_size(&base) {
        // "name:size" is the primary Ollama format. Deliberately *not* the bare
        // family name: this model announces its size, so an install of a
        // different size in the same family is a different model. Adding the
        // stem here is what let one `qwen3:8b` mark every `Qwen3-*` entry in
        // the catalog installed (#861).
        candidates.push(format!("{}:{}", name, size));
    }

    // Also try the full lowered name and stripped name as-is
    candidates.push(base.clone());
    if base != repo {
        candidates.push(repo);
    }

    candidates.dedup();
    candidates
}

/// Returns `true` if this HF model has a known Ollama registry entry
/// and can be pulled.
pub fn has_ollama_mapping(hf_name: &str) -> bool {
    lookup_ollama_tag(hf_name).is_some()
}

/// Largest ratio between a catalog entry's parameter count and an installed
/// tag's size before the two are treated as different models. Deliberately
/// loose: Ollama tags round (`phi4:14b` for a 14.7B model), so the check is
/// only meant to catch mismatches of a different order, like a 14B tag
/// standing in for a 684B model.
const OLLAMA_SIZE_MISMATCH_RATIO: f64 = 2.0;

/// Parse the size out of an Ollama tag: `deepseek-r1:14b` -> `14.0`,
/// `qwen2.5-coder:7b-instruct-q4_K_M` -> `7.0`. Returns `None` for tags with
/// no parseable size, including `8x7b`-style MoE tags and bare family stems.
fn ollama_tag_size_b(installed_name: &str) -> Option<f64> {
    let size = installed_name.split(':').nth(1)?;
    let head = size.split('-').next()?;
    head.strip_suffix('b')?.parse::<f64>().ok()
}

/// Does an installed tag's size agree with the catalog entry's parameter
/// count? Unknown on either side means "no opinion", which keeps the previous
/// permissive behaviour rather than dropping a real install.
fn ollama_size_is_compatible(installed_name: &str, catalog_params_b: Option<f64>) -> bool {
    let (Some(catalog), Some(tag)) = (catalog_params_b, ollama_tag_size_b(installed_name)) else {
        return true;
    };
    if catalog <= 0.0 || tag <= 0.0 {
        return true;
    }
    let ratio = if catalog > tag {
        catalog / tag
    } else {
        tag / catalog
    };
    ratio <= OLLAMA_SIZE_MISMATCH_RATIO
}

fn ollama_installed_matches_candidate(
    installed_name: &str,
    candidate: &str,
    catalog_params_b: Option<f64>,
) -> bool {
    if installed_name == candidate {
        return true;
    }

    // Allow variant tags reported by `ollama list`, e.g.
    // candidate: "qwen2.5-coder:7b"
    // installed: "qwen2.5-coder:7b-instruct-q4_K_M"
    if candidate.contains(':') {
        return installed_name.starts_with(&format!("{candidate}-"));
    }

    // A size-less candidate is family-level by construction — either an
    // `OLLAMA_MAPPINGS` entry whose tag carries no size (`phi-4` → `phi4`,
    // `qwq-32b` → `qwq`) or an HF name with no size to parse. Any tag of that
    // family is *usually* the model in question, so match `phi4:14b` too.
    //
    // "Usually" is why the size check is here. Some families publish one bare
    // tag for wildly different models: `deepseek-r1` covers both the 684B
    // original and the 14B Qwen distill, so `deepseek-r1:14b` would otherwise
    // mark the 397 GB entry installed. Candidates derived from a *sized* HF
    // name never reach here (see `hf_name_to_ollama_candidates`).
    installed_name.starts_with(&format!("{candidate}:"))
        && ollama_size_is_compatible(installed_name, catalog_params_b)
}

/// Check if any of the Ollama candidates for an HF model appear in the
/// installed set.
pub fn is_model_installed(hf_name: &str, installed: &HashSet<String>) -> bool {
    is_model_installed_sized(hf_name, None, installed)
}

/// Like [`is_model_installed`], but takes the catalog entry's parameter count
/// so a family-level tag match can be rejected when the sizes disagree. Pass
/// `None` when the size is unknown; the match is then as permissive as before.
pub fn is_model_installed_sized(
    hf_name: &str,
    catalog_params_b: Option<f64>,
    installed: &HashSet<String>,
) -> bool {
    // Quick check: the installed set may contain the full HF name (lowercased)
    // from providers that report it verbatim (e.g. MLX server, /api/v1/installed).
    if installed.contains(&hf_name.to_lowercase()) {
        return true;
    }

    let candidates = hf_name_to_ollama_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed.iter().any(|installed_name| {
            ollama_installed_matches_candidate(installed_name, candidate, catalog_params_b)
        })
    })
}

/// Match a running provider's model tag (an Ollama-style id, or a GGUF file
/// path/stem as reported by llama-server) against an HF-style model name,
/// reusing the installed-column heuristics.
///
/// Two deliberately separate passes: Ollama-style candidate matching runs
/// only against the verbatim id, while file paths (".../gemma-3.Q8_0.gguf")
/// get exact stem matching only — feeding a bare stem into the Ollama
/// candidate heuristics would match whole families.
pub fn tag_matches_model(tag: &str, hf_name: &str) -> bool {
    let lower = tag.to_lowercase();

    let mut tag_set = HashSet::new();
    tag_set.insert(lower.clone());
    if is_model_installed(hf_name, &tag_set) {
        return true;
    }

    let stem = lower
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(&lower)
        .trim_end_matches(".gguf")
        .to_string();
    let mut stem_set = HashSet::new();
    if let Some(base) = strip_gguf_quant_suffix(&stem) {
        stem_set.insert(base);
    }
    // Third arm: MLX community tags carry mlx-community basenames
    // (`llama-3.2-1b-instruct-4bit`); strip the quant tail so they reduce to
    // catalog slugs through the same exact-stem path as GGUF stems (#854).
    if let Some(base) = strip_mlx_quant_suffix(&stem) {
        stem_set.insert(base);
    }
    stem_set.insert(stem);
    is_model_installed_llamacpp(hf_name, &stem_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Install layouts from issue #731 (Windows, LM Studio + Docker Desktop
    // installed but their servers not running) must be recognized. Expected
    // paths are built with join() so separators stay portable across the
    // 3-OS CI matrix.
    #[test]
    fn test_lmstudio_install_candidates_windows_layouts() {
        let pf = Path::new(r"C:\Program Files");
        let lad = Path::new(r"C:\Users\ben\AppData\Local");
        let home = Path::new(r"C:\Users\ben");
        let candidates = lmstudio_install_candidates(pf.to_str(), lad.to_str(), Some(home));
        // Reporter's per-machine install.
        assert!(candidates.contains(&pf.join("LM Studio").join("LM Studio.exe")));
        // Installer's per-user default.
        assert!(candidates.contains(&lad.join("Programs").join("LM Studio").join("LM Studio.exe")));
        // First-run data dir (any OS).
        assert!(candidates.contains(&home.join(".lmstudio")));
    }

    #[test]
    fn test_lmstudio_install_candidates_unix_layouts() {
        let home = Path::new("/home/ben");
        let candidates = lmstudio_install_candidates(None, None, Some(home));
        assert!(candidates.contains(&PathBuf::from("/Applications/LM Studio.app")));
        assert!(candidates.contains(&home.join("Applications").join("LM Studio.app")));
        assert!(candidates.contains(&home.join(".lmstudio")));
    }

    #[test]
    fn test_docker_desktop_install_candidates_windows_layouts() {
        let pf = Path::new(r"C:\Program Files");
        let home = Path::new(r"C:\Users\ben");
        let candidates = docker_desktop_install_candidates(pf.to_str(), Some(home));
        let docker = pf.join("Docker").join("Docker");
        // Classic exe location.
        assert!(candidates.contains(&docker.join("Docker Desktop.exe")));
        // Reporter's frontend\ layout from newer Docker Desktop releases.
        assert!(candidates.contains(&docker.join("frontend").join("Docker Desktop.exe")));
        assert!(
            candidates.contains(
                &home
                    .join(".docker")
                    .join("cli-plugins")
                    .join("docker-model.exe")
            )
        );
    }

    #[test]
    fn test_docker_desktop_install_candidates_unix_layouts() {
        let home = Path::new("/home/ben");
        let candidates = docker_desktop_install_candidates(None, Some(home));
        assert!(candidates.contains(&PathBuf::from("/Applications/Docker.app")));
        assert!(candidates.contains(&PathBuf::from("/opt/docker-desktop")));
        assert!(candidates.contains(&home.join(".docker").join("desktop")));
        assert!(
            candidates.contains(
                &home
                    .join(".docker")
                    .join("cli-plugins")
                    .join("docker-model")
            )
        );
    }

    #[test]
    fn test_hf_name_to_mlx_candidates() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            candidates
                .iter()
                .any(|c| c.contains("llama-3.1-8b-instruct"))
        );
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
        assert!(candidates.iter().any(|c| c.ends_with("-8bit")));

        let qwen = hf_name_to_mlx_candidates("Qwen/Qwen2.5-Coder-14B-Instruct");
        assert!(
            qwen.iter()
                .any(|c| c.contains("qwen2.5-coder-14b-instruct"))
        );
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_qwen35() {
        let candidates = hf_name_to_mlx_candidates("Qwen/Qwen3.5-9B");
        assert!(candidates.iter().any(|c| c == "qwen3.5-9b-4bit"));
        assert!(candidates.iter().any(|c| c == "qwen3.5-9b-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_llama4() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-4-Scout-17B-16E-Instruct");
        assert!(candidates.iter().any(|c| c.contains("llama-4-scout")));
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_gemma3() {
        let candidates = hf_name_to_mlx_candidates("google/gemma-3-27b-it");
        assert!(candidates.iter().any(|c| c == "gemma-3-27b-it-4bit"));
        assert!(candidates.iter().any(|c| c == "gemma-3-27b-it-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_fallback_generates_mlx_infix_candidates() {
        // For models not in the explicit mapping, the fallback should also
        // generate candidates with the -mlx- infix pattern
        let candidates = hf_name_to_mlx_candidates("SomeOrg/SomeNewModel-7B");
        assert!(candidates.iter().any(|c| c == "somenewmodel-7b-mlx-4bit"));
        assert!(candidates.iter().any(|c| c == "somenewmodel-7b-mlx-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_normalizes_explicit_mlx_repo() {
        let candidates =
            hf_name_to_mlx_candidates("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit");

        assert!(
            candidates
                .contains(&"lmstudio-community/qwen3-coder-30b-a3b-instruct-mlx-8bit".to_string())
        );
        assert!(candidates.contains(&"qwen3-coder-30b-a3b-instruct-4bit".to_string()));
        assert!(candidates.contains(&"qwen3-coder-30b-a3b-instruct-8bit".to_string()));
        assert!(!candidates.iter().any(|c| c.contains("-8bit-4bit")));
        assert!(!candidates.iter().any(|c| c.contains("-8bit-8bit")));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_normalizes_dwq_repo() {
        // #869: a DWQ repo id must generate candidates from its catalog
        // base, not carry the unstripped quant tail into every variant.
        let candidates = hf_name_to_mlx_candidates("mlx-community/Qwen3-8B-4bit-DWQ");

        assert!(candidates.contains(&"qwen3-8b-4bit".to_string()));
        assert!(candidates.contains(&"qwen3-8b".to_string()));
        assert!(!candidates.iter().any(|c| c.contains("-dwq-4bit")));
        assert!(!candidates.iter().any(|c| c.contains("-dwq-8bit")));
        assert!(!candidates.iter().any(|c| c.contains("-dwq-mlx")));
    }

    #[test]
    fn test_mlx_cache_scan_parsing() {
        // Test that the candidate matching works with cache-style names
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b-instruct-4bit".to_string());

        assert!(is_model_installed_mlx(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
        // Should not match unrelated model
        assert!(!is_model_installed_mlx(
            "Qwen/Qwen2.5-7B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_mlx() {
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder-14b-instruct-8bit".to_string());

        assert!(is_model_installed_mlx(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
        assert!(!is_model_installed_mlx(
            "Qwen/Qwen2.5-14B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_full_repo() {
        let candidates = hf_name_to_lmstudio_candidates("lmstudio-community/Qwen3-1.7B-GGUF");
        assert!(candidates.contains(&"lmstudio-community/qwen3-1.7b-gguf".to_string()));
        assert!(candidates.contains(&"qwen3-1.7b-gguf".to_string()));
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_strips_suffixes() {
        let candidates = hf_name_to_lmstudio_candidates("meta-llama/Llama-3-8B-Instruct");
        assert!(candidates.contains(&"meta-llama/llama-3-8b-instruct".to_string()));
        assert!(candidates.contains(&"llama-3-8b-instruct".to_string()));
        // Stripped variant (without -instruct)
        assert!(candidates.contains(&"llama-3-8b".to_string()));
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_bare_name() {
        let candidates = hf_name_to_lmstudio_candidates("qwen3");
        assert!(candidates.contains(&"qwen3".to_string()));
        // No slash, so repo == full name — no duplicate
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_lmstudio_api_key_filtering() {
        // Test the api_key filtering logic without mutating the process
        // environment. LmStudioProvider::default() applies
        // `.filter(|k| !k.is_empty())` to the env var value.
        fn filter_key(val: Option<&str>) -> Option<String> {
            val.map(String::from).filter(|k| !k.is_empty())
        }

        // Missing env var → None
        assert!(filter_key(None).is_none());
        // Real value → Some
        assert_eq!(
            filter_key(Some("my-secret-key")),
            Some("my-secret-key".to_string())
        );
        // Empty string → None (must not produce Some(""))
        assert!(filter_key(Some("")).is_none());
    }

    #[test]
    fn test_is_model_installed_mlx_with_owner_prefixed_repo_id() {
        let mut installed = HashSet::new();
        installed.insert("lmstudio-community/qwen3-coder-30b-a3b-instruct-mlx-8bit".to_string());

        assert!(is_model_installed_mlx(
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit",
            &installed
        ));
    }

    #[test]
    fn test_qwen_coder_14b_matches_coder_entry() {
        // "qwen2.5-coder:14b" from `ollama list` should match
        // the HF entry "Qwen/Qwen2.5-Coder-14B-Instruct", NOT
        // the base "Qwen/Qwen2.5-14B-Instruct".
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder:14b".to_string());
        installed.insert("qwen2.5-coder".to_string());

        assert!(is_model_installed(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
        // Must NOT match the non-coder model
        assert!(!is_model_installed("Qwen/Qwen2.5-14B-Instruct", &installed));
    }

    #[test]
    fn test_qwen_base_does_not_match_coder() {
        // "qwen2.5:14b" from `ollama list` should match the base model,
        // not the coder variant.
        let mut installed = HashSet::new();
        installed.insert("qwen2.5:14b".to_string());
        installed.insert("qwen2.5".to_string());

        assert!(is_model_installed("Qwen/Qwen2.5-14B-Instruct", &installed));
        assert!(!is_model_installed(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_installed_variant_suffix_matches_ollama_candidate() {
        // Real-world `ollama list` may include variant suffixes that still map
        // to the canonical pull tag in OLLAMA_MAPPINGS.
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder:7b-instruct".to_string());

        assert!(is_model_installed(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_candidates_for_coder_model() {
        let candidates = hf_name_to_ollama_candidates("Qwen/Qwen2.5-Coder-14B-Instruct");
        assert!(candidates.contains(&"qwen2.5-coder:14b".to_string()));
    }

    #[test]
    fn test_candidates_for_base_model() {
        let candidates = hf_name_to_ollama_candidates("Qwen/Qwen2.5-14B-Instruct");
        assert!(candidates.contains(&"qwen2.5:14b".to_string()));
    }

    #[test]
    fn test_qwen3_8_resolves_to_its_ollama_tag() {
        assert_eq!(
            hf_name_to_ollama_candidates("Qwen/Qwen3.8-27B"),
            vec!["qwen3.8:27b".to_string()]
        );
    }

    // Regression for #866: every gemma3 size Ollama ships must resolve
    // through the explicit mapping, so an `ollama pull gemma3:4b` install
    // is detected and the model stays pullable. Only the 12B was mapped.
    #[test]
    fn test_gemma3_family_resolves_to_its_ollama_tags() {
        for (hf_name, tag) in [
            ("google/gemma-3-1b-it", "gemma3:1b"),
            ("google/gemma-3-4b-it", "gemma3:4b"),
            ("google/gemma-3-12b-it", "gemma3:12b"),
            ("google/gemma-3-27b-it", "gemma3:27b"),
        ] {
            assert_eq!(
                hf_name_to_ollama_candidates(hf_name),
                vec![tag.to_string()],
                "candidates for {hf_name}"
            );
            assert!(has_ollama_mapping(hf_name), "mapping for {hf_name}");
            let installed = HashSet::from([tag.to_string()]);
            assert!(
                is_model_installed(hf_name, &installed),
                "install of {tag} must mark {hf_name} installed"
            );
        }
    }

    #[test]
    fn test_llama_mapping() {
        let candidates = hf_name_to_ollama_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(candidates.contains(&"llama3.1:8b".to_string()));
    }

    #[test]
    fn test_deepseek_coder_mapping() {
        let candidates =
            hf_name_to_ollama_candidates("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct");
        assert!(candidates.contains(&"deepseek-coder-v2:16b".to_string()));
    }

    #[test]
    fn test_normalize_ollama_host_with_scheme() {
        assert_eq!(
            normalize_ollama_host("https://ollama.example.com:11434"),
            Some("https://ollama.example.com:11434".to_string())
        );
    }

    #[test]
    fn test_normalize_ollama_host_without_scheme() {
        assert_eq!(
            normalize_ollama_host("ollama.example.com:11434"),
            Some("http://ollama.example.com:11434".to_string())
        );
    }

    #[test]
    fn test_normalize_ollama_host_rejects_unsupported_scheme() {
        assert_eq!(
            normalize_ollama_host("ftp://ollama.example.com:11434"),
            None
        );
    }

    #[test]
    fn test_is_wildcard_bind_address_ipv4() {
        assert!(is_wildcard_bind_address("0.0.0.0"));
        assert!(is_wildcard_bind_address("0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("http://0.0.0.0"));
        assert!(is_wildcard_bind_address("http://0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("https://0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("http://0.0.0.0:11434/api/tags"));
    }

    #[test]
    fn test_is_wildcard_bind_address_ipv6() {
        assert!(is_wildcard_bind_address("[::]"));
        assert!(is_wildcard_bind_address("[::]:11434"));
        assert!(is_wildcard_bind_address("http://[::]:11434"));
        assert!(is_wildcard_bind_address("http://[0:0:0:0:0:0:0:0]:11434"));
    }

    #[test]
    fn test_is_wildcard_bind_address_rejects_routable_hosts() {
        assert!(!is_wildcard_bind_address("localhost"));
        assert!(!is_wildcard_bind_address("http://localhost:11434"));
        assert!(!is_wildcard_bind_address("127.0.0.1"));
        assert!(!is_wildcard_bind_address("http://127.0.0.1:11434"));
        assert!(!is_wildcard_bind_address("http://[::1]:11434"));
        assert!(!is_wildcard_bind_address("http://ollama.example.com:11434"));
        // Hostnames or IPs that merely contain "0.0.0.0" as a substring must not match.
        assert!(!is_wildcard_bind_address("http://10.0.0.0.example.com"));
        assert!(!is_wildcard_bind_address("http://10.0.0.1:11434"));
    }

    // ── is_model_installed_llamacpp ──────────────────────────────────

    #[test]
    fn test_is_model_installed_llamacpp_exact() {
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b-instruct".to_string());
        assert!(is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_stripped_suffixes() {
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b".to_string());
        assert!(is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_strip_gguf_quant_suffix_unsloth_ud_marker() {
        // Unsloth "Dynamic" GGUFs carry a `-ud` marker before the quant; it
        // must be stripped alongside the quant so the stem reduces to the
        // canonical model name.
        assert_eq!(
            strip_gguf_quant_suffix("qwen3.6-35b-a3b-ud-q4_k_m").as_deref(),
            Some("qwen3.6-35b-a3b")
        );
        // Non-Unsloth files are unaffected.
        assert_eq!(
            strip_gguf_quant_suffix("qwen2.5-7b-instruct-q4_k_m").as_deref(),
            Some("qwen2.5-7b-instruct")
        );
    }

    #[test]
    fn test_strip_gguf_quant_suffix_covers_every_k_and_i_variant() {
        // Publishers ship far more variants than Q4_K_M: bartowski `_L`,
        // Unsloth Dynamic `_XL`, and the IQ family's `_NL`/`_XS`/`_XXS`.
        // Every one must reduce to the same base name.
        for stem in [
            "mymodel-q3_k_l",
            "mymodel-q4_k_l",
            "mymodel-q4_k_xl",
            "mymodel-q5_k_xl",
            "mymodel-q6_k_l",
            "mymodel-q8_k_xl",
            "mymodel-iq4_nl",
            "mymodel-iq4_xs",
            "mymodel-iq3_xxs",
            "mymodel-iq2_m",
            // Same set behind the Unsloth `-ud` marker.
            "mymodel-ud-q4_k_xl",
            "mymodel-ud-iq4_nl",
        ] {
            assert_eq!(
                strip_gguf_quant_suffix(stem).as_deref(),
                Some("mymodel"),
                "failed to strip quant from {stem}"
            );
        }
    }

    #[test]
    fn test_tag_matches_model_unsloth_dynamic_quants() {
        // Community submissions record the on-disk file name verbatim. These
        // three tags from the NVIDIA GB10 set (#872) were silently dropped
        // during lookup because their quants had no matching pattern.
        assert!(tag_matches_model(
            "Step-3.7-Flash-UD-IQ4_NL.gguf",
            "stepfun-ai/Step-3.7-Flash"
        ));
        assert!(tag_matches_model(
            "Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf",
            "Qwen/Qwen-AgentWorld-35B-A3B"
        ));
        assert!(tag_matches_model(
            "LFM2.5-8B-A1B-UD-Q4_K_XL.gguf",
            "LiquidAI/LFM2.5-8B-A1B"
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_unsloth_ud() {
        // `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` on disk yields these set entries
        // (see `installed_models_counted`) and must mark the catalog model as
        // installed despite the embedded `-ud` marker.
        let installed: HashSet<String> = ["qwen3.6-35b-a3b-ud-q4_k_m", "qwen3.6-35b-a3b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(is_model_installed_llamacpp(
            "Qwen/Qwen3.6-35B-A3B",
            &installed
        ));
    }

    #[test]
    fn test_tag_matches_model_unsloth_ud_gguf() {
        // End-to-end: a llama-server serving an Unsloth UD GGUF reports the
        // file name as the model id; it must match the catalog HF name so the
        // model is benchmarkable (regression: `-ud` broke the exact stem match).
        assert!(tag_matches_model(
            "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "Qwen/Qwen3.6-35B-A3B"
        ));
    }

    #[test]
    fn test_strip_mlx_quant_suffix_patterns() {
        // Plain bit-widths.
        assert_eq!(
            strip_mlx_quant_suffix("llama-3.2-1b-instruct-4bit").as_deref(),
            Some("llama-3.2-1b-instruct")
        );
        assert_eq!(
            strip_mlx_quant_suffix("internlm2_5-20b-chat-8bit").as_deref(),
            Some("internlm2_5-20b-chat")
        );
        assert_eq!(
            strip_mlx_quant_suffix("phi-4-2bit").as_deref(),
            Some("phi-4")
        );
        assert_eq!(
            strip_mlx_quant_suffix("gemma-2-2b-it-6bit").as_deref(),
            Some("gemma-2-2b-it")
        );
        // Variant markers after the bit-width, including date-stamped DWQ.
        assert_eq!(
            strip_mlx_quant_suffix("meta-llama-3.1-8b-instruct-4bit-dwq").as_deref(),
            Some("meta-llama-3.1-8b-instruct")
        );
        assert_eq!(
            strip_mlx_quant_suffix("qwen3-8b-4bit-dwq-05082025").as_deref(),
            Some("qwen3-8b")
        );
        // Compound schemes stripped as whole units.
        assert_eq!(
            strip_mlx_quant_suffix("gpt-oss-20b-mxfp4-q4").as_deref(),
            Some("gpt-oss-20b")
        );
        assert_eq!(
            strip_mlx_quant_suffix("mistral-7b-v0.1-fp16").as_deref(),
            Some("mistral-7b-v0.1")
        );
        // #869: odd bit-widths and mxfp4 variant markers.
        assert_eq!(
            strip_mlx_quant_suffix("minimax-m2.1-3bit").as_deref(),
            Some("minimax-m2.1")
        );
        assert_eq!(
            strip_mlx_quant_suffix("glm-5.2-mxfp4").as_deref(),
            Some("glm-5.2")
        );
        assert_eq!(
            strip_mlx_quant_suffix("gpt-oss-20b-mxfp4-q8").as_deref(),
            Some("gpt-oss-20b")
        );
        assert_eq!(
            strip_mlx_quant_suffix("gpt-oss-120b-mxfp4-bf16").as_deref(),
            Some("gpt-oss-120b")
        );
        // #869: trailing -mlx marker stripped once the quant is gone.
        assert_eq!(
            strip_mlx_quant_suffix("qwen3-coder-30b-a3b-instruct-mlx-8bit").as_deref(),
            Some("qwen3-coder-30b-a3b-instruct")
        );
        assert_eq!(
            strip_mlx_quant_suffix("bge-m3-mlx-fp16").as_deref(),
            Some("bge-m3")
        );
        assert_eq!(
            strip_mlx_quant_suffix("bge-m3-mlx").as_deref(),
            Some("bge-m3")
        );
        // Mid-name fragments and bare widths are not suffixes.
        assert_eq!(strip_mlx_quant_suffix("some-4bitish-model"), None);
        assert_eq!(strip_mlx_quant_suffix("4bit"), None);
        assert_eq!(strip_mlx_quant_suffix("-4bit"), None);
    }

    #[test]
    fn test_tag_matches_model_mlx_community_tags() {
        // End-to-end: verbatim `model` tags from the apple-m4-pro MLX
        // community submissions (#853) must resolve to their catalog HF
        // names, exactly like GGUF stems do (#854).
        assert!(tag_matches_model(
            "Llama-3.2-1B-Instruct-4bit",
            "meta-llama/Llama-3.2-1B-Instruct"
        ));
        assert!(tag_matches_model("Qwen3-14B-4bit", "Qwen/Qwen3-14B"));
        assert!(tag_matches_model(
            "gpt-oss-20b-MXFP4-Q4",
            "openai/gpt-oss-20b"
        ));
        // #869: compound mxfp4 markers and the -mlx infix now resolve too.
        assert!(tag_matches_model(
            "gpt-oss-20b-MXFP4-Q8",
            "openai/gpt-oss-20b"
        ));
        assert!(tag_matches_model(
            "Qwen3-Coder-30B-A3B-Instruct-MLX-8bit",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        ));
        // One model's basename must not match another model.
        assert!(!tag_matches_model(
            "Qwen3-8B-4bit",
            "meta-llama/Llama-3.2-1B-Instruct"
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_not_installed() {
        let installed = HashSet::new();
        assert!(!is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_no_family_false_positives() {
        // A single "gemma-3.Q8_0.gguf" on disk yields these stems — it must
        // mark ONLY repos actually named "gemma-3" as installed, not the
        // whole gemma-3 family (regression: substring matching ticked every
        // gemma-3-* model in the table).
        let installed: HashSet<String> = ["gemma-3.q8_0", "gemma-3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(is_model_installed_llamacpp(
            "tiny-random/gemma-3",
            &installed
        ));
        assert!(!is_model_installed_llamacpp(
            "google/gemma-3-27b-it",
            &installed
        ));
        assert!(!is_model_installed_llamacpp(
            "google/gemma-3-4b-it",
            &installed
        ));
        assert!(!is_model_installed_llamacpp(
            "unsloth/gemma-3-270m-it",
            &installed
        ));
    }

    // ── has_ollama_mapping ───────────────────────────────────────────

    #[test]
    fn test_has_ollama_mapping_known() {
        assert!(has_ollama_mapping("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(has_ollama_mapping("Qwen/Qwen2.5-7B-Instruct"));
    }

    #[test]
    fn test_has_ollama_mapping_unknown() {
        assert!(!has_ollama_mapping("totally-unknown/model-xyz"));
    }

    // ── ollama_installed_matches_candidate ────────────────────────────

    #[test]
    fn test_ollama_installed_matches_exact() {
        assert!(ollama_installed_matches_candidate(
            "llama3.1:8b",
            "llama3.1:8b",
            None
        ));
    }

    #[test]
    fn test_ollama_installed_matches_variant_suffix() {
        assert!(ollama_installed_matches_candidate(
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b",
            None
        ));
    }

    #[test]
    fn test_ollama_installed_no_match() {
        assert!(!ollama_installed_matches_candidate(
            "qwen2.5:7b",
            "llama3.1:8b",
            None
        ));
    }

    // ── size-aware family matching ───────────────────────────────────

    fn deepseek_14b_installed() -> HashSet<String> {
        let mut installed = HashSet::new();
        installed.insert("deepseek-r1:14b".to_string());
        installed
    }

    #[test]
    fn test_ollama_tag_size_parsing() {
        assert_eq!(ollama_tag_size_b("deepseek-r1:14b"), Some(14.0));
        assert_eq!(ollama_tag_size_b("deepseek-r1:1.5b"), Some(1.5));
        assert_eq!(
            ollama_tag_size_b("qwen2.5-coder:7b-instruct-q4_K_M"),
            Some(7.0)
        );
        // No size to read: bare family stem, MoE-style tag, or a plain name.
        assert_eq!(ollama_tag_size_b("glm-4.7-flash"), None);
        assert_eq!(ollama_tag_size_b("mixtral:8x7b"), None);
        assert_eq!(ollama_tag_size_b("deepseek-r1:latest"), None);
    }

    #[test]
    fn test_sized_family_tag_does_not_mark_much_larger_model_installed() {
        // `deepseek-r1` is the mapped tag for the 684B original *and* the
        // family prefix of the 14B distill. Having the distill must not claim
        // the 397 GB model is on disk.
        let installed = deepseek_14b_installed();
        assert!(!is_model_installed_sized(
            "deepseek-ai/DeepSeek-R1",
            Some(684.5),
            &installed
        ));
        assert!(!is_model_installed_sized(
            "deepseek-ai/DeepSeek-R1-0528",
            Some(684.5),
            &installed
        ));
    }

    #[test]
    fn test_sized_family_tag_still_matches_the_model_it_belongs_to() {
        let installed = deepseek_14b_installed();
        // The distill maps to `deepseek-r1:14b` outright.
        assert!(is_model_installed_sized(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            Some(14.8),
            &installed
        ));
        // And the real 684B model is matched by a tag of its own size.
        let mut big = HashSet::new();
        big.insert("deepseek-r1:671b".to_string());
        assert!(is_model_installed_sized(
            "deepseek-ai/DeepSeek-R1",
            Some(684.5),
            &big
        ));
    }

    #[test]
    fn test_size_rounding_in_tags_still_matches() {
        // Ollama tags round: phi4:14b is a 14.7B model, qwq:32b is 32.8B.
        let mut installed = HashSet::new();
        installed.insert("phi4:14b".to_string());
        installed.insert("qwq:32b".to_string());
        assert!(is_model_installed_sized(
            "microsoft/phi-4",
            Some(14.7),
            &installed
        ));
        assert!(is_model_installed_sized(
            "Qwen/QwQ-32B",
            Some(32.8),
            &installed
        ));
    }

    #[test]
    fn test_unknown_catalog_size_stays_permissive() {
        // With no parameter count recorded there is nothing to compare, so
        // behaviour must be exactly what it was before the size check.
        let installed = deepseek_14b_installed();
        assert!(is_model_installed_sized(
            "deepseek-ai/DeepSeek-R1",
            None,
            &installed
        ));
        assert!(is_model_installed("deepseek-ai/DeepSeek-R1", &installed));
    }

    // ── recursive gguf scan / LM Studio models directory ─────────────

    /// Minimal scratch directory that removes itself. Avoids a `tempfile`
    /// dev-dependency for the handful of tests that need a real tree.
    struct TempTree(PathBuf);

    impl TempTree {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "llmfit-{tag}-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create temp tree");
            Self(dir)
        }

        fn touch(&self, rel: &str) {
            let path = self.0.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create parents");
            }
            std::fs::write(&path, b"").expect("write file");
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn names_of(paths: &[PathBuf]) -> Vec<String> {
        let mut names: Vec<String> = paths
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn test_collect_gguf_files_descends_and_caps_depth() {
        let tree = TempTree::new("scan");
        tree.touch("root.gguf");
        tree.touch("publisher/mid.gguf");
        tree.touch("publisher/repo/leaf.gguf");
        tree.touch("publisher/repo/notes.txt");
        // One level past the cap.
        tree.touch("publisher/repo/nested/toodeep.gguf");

        let found = names_of(&collect_gguf_files(tree.path(), GGUF_SCAN_MAX_DEPTH));
        assert_eq!(found, vec!["leaf.gguf", "mid.gguf", "root.gguf"]);
    }

    #[test]
    fn test_collect_gguf_files_flat_when_depth_is_one() {
        let tree = TempTree::new("flat");
        tree.touch("root.gguf");
        tree.touch("publisher/mid.gguf");

        assert_eq!(
            names_of(&collect_gguf_files(tree.path(), 1)),
            vec!["root.gguf"]
        );
        assert!(collect_gguf_files(tree.path(), 0).is_empty());
    }

    #[test]
    fn test_collect_gguf_files_missing_root_is_empty() {
        let missing = std::env::temp_dir().join("llmfit-does-not-exist-9f3c1a");
        assert!(collect_gguf_files(&missing, GGUF_SCAN_MAX_DEPTH).is_empty());
    }

    #[cfg(unix)]
    fn symlink_file(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn symlink_file(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }

    #[test]
    fn test_collect_gguf_files_follows_symlinked_models() {
        // `DirEntry::file_type` describes the link rather than its target, so
        // a type check that does not resolve silently drops symlinked models.
        // Creating a symlink needs elevation on Windows, so skip there when
        // it is not permitted rather than failing the run.
        // The target sits outside the scanned tree, so the link is the only
        // way in and there is no alias for the dedupe pass to collapse.
        let store = TempTree::new("symlink-store");
        store.touch("real-model.gguf");

        let tree = TempTree::new("symlink");
        tree.touch("plain-model.gguf");
        let link = tree.path().join("linked-model.gguf");
        if symlink_file(&store.path().join("real-model.gguf"), &link).is_err() {
            return;
        }

        let found = names_of(&collect_gguf_files(tree.path(), GGUF_SCAN_MAX_DEPTH));
        assert_eq!(found, vec!["linked-model.gguf", "plain-model.gguf"]);
    }

    #[cfg(unix)]
    fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
    }

    #[test]
    fn test_collect_gguf_files_does_not_follow_directory_symlinks() {
        // A directory symlink can point anywhere, including outside the tree
        // being scanned. Following one would pull unrelated files into
        // installed detection and put them in front of callers that act on
        // the returned paths.
        let outside = TempTree::new("outside");
        outside.touch("secret-model.gguf");

        let tree = TempTree::new("escape");
        tree.touch("inside.gguf");
        if symlink_dir(outside.path(), &tree.path().join("escape-hatch")).is_err() {
            return;
        }

        let found = names_of(&collect_gguf_files(tree.path(), GGUF_SCAN_MAX_DEPTH));
        assert_eq!(found, vec!["inside.gguf"]);
    }

    #[test]
    fn test_collect_gguf_files_collapses_symlink_aliases() {
        // A symlink beside the file it points at is one model, not two.
        let tree = TempTree::new("alias");
        tree.touch("real-model.gguf");
        if symlink_file(
            &tree.path().join("real-model.gguf"),
            &tree.path().join("alias-model.gguf"),
        )
        .is_err()
        {
            return;
        }

        let found = collect_gguf_files(tree.path(), GGUF_SCAN_MAX_DEPTH);
        assert_eq!(found.len(), 1, "expected one model, got {found:?}");
        assert_eq!(names_of(&found), vec!["real-model.gguf"]);
    }

    #[test]
    fn test_dedupe_prefers_target_even_when_alias_is_seen_first() {
        // Under a symlinked root neither path equals its canonical form, so a
        // tie-break based on path equality silently never fires and the
        // survivor comes down to directory order. `dedupe_by_target` is called
        // directly with the alias first so the ordering is not left to the
        // filesystem: whichever order they arrive in, the real file must win,
        // because callers look models up by stem.
        let store = TempTree::new("realroot");
        store.touch("model.gguf");
        if symlink_file(
            &store.path().join("model.gguf"),
            &store.path().join("alias.gguf"),
        )
        .is_err()
        {
            return;
        }

        let links = TempTree::new("linkroot");
        let root_link = links.path().join("root");
        if symlink_dir(store.path(), &root_link).is_err() {
            return;
        }

        let alias_first = vec![root_link.join("alias.gguf"), root_link.join("model.gguf")];
        let kept = dedupe_by_target(alias_first);
        assert_eq!(kept.len(), 1, "expected one model, got {kept:?}");
        assert_eq!(names_of(&kept), vec!["model.gguf"]);

        let target_first = vec![root_link.join("model.gguf"), root_link.join("alias.gguf")];
        assert_eq!(
            names_of(&dedupe_by_target(target_first)),
            vec!["model.gguf"]
        );
    }

    #[test]
    fn test_lmstudio_models_dir_layout() {
        let home = PathBuf::from("/home/someone");
        assert_eq!(
            lmstudio_models_dir_for(Some(&home)),
            Some(home.join(".lmstudio").join("models"))
        );
        assert_eq!(lmstudio_models_dir_for(None), None);
    }

    fn lmstudio_tree() -> TempTree {
        let tree = TempTree::new("lms");
        tree.touch(
            "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        );
        tree.touch("lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q8_0.gguf");
        tree.touch("lmstudio-community/gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-BF16.gguf");
        tree
    }

    #[test]
    fn test_scan_lmstudio_models_dir_names_and_projector_handling() {
        let tree = lmstudio_tree();
        let (set, count) = scan_lmstudio_models_dir_at(Some(tree.path()));

        // Two models, not three: the mmproj file is a projector, not a model.
        assert_eq!(count, 2);
        assert!(set.contains("lmstudio-community/meta-llama-3.1-8b-instruct-gguf"));
        assert!(set.contains("meta-llama-3.1-8b-instruct"));
        assert!(set.contains("gemma-4-12b-it"));
        assert!(!set.iter().any(|n| n.starts_with("mmproj-")));
    }

    #[test]
    fn test_lmstudio_disk_match_is_exact_not_substring() {
        let tree = lmstudio_tree();
        let (set, _) = scan_lmstudio_models_dir_at(Some(tree.path()));

        assert!(is_model_installed_lmstudio_disk(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            &set
        ));
        assert!(is_model_installed_lmstudio_disk(
            "google/gemma-4-12B-it",
            &set
        ));

        // A catalog entry literally named "llama" is a substring of
        // "meta-llama-3.1-8b-instruct". Equality is what keeps it out.
        assert!(!is_model_installed_lmstudio_disk(
            "Resilient-Coders/llama",
            &set
        ));
        // The base model is a different model from the Instruct one.
        assert!(!is_model_installed_lmstudio_disk(
            "meta-llama/Llama-3.1-8B",
            &set
        ));
    }

    #[test]
    fn test_lmstudio_disk_base_model_does_not_claim_variants() {
        // Only the base model is on disk. The `-it`, `-instruct` and `-chat`
        // entries are distinct catalog models and must stay uninstalled.
        let tree = TempTree::new("variant");
        tree.touch("unsloth/gemma-4-12b-GGUF/gemma-4-12b-Q8_0.gguf");
        let (set, _) = scan_lmstudio_models_dir_at(Some(tree.path()));

        assert!(is_model_installed_lmstudio_disk(
            "unsloth/gemma-4-12b",
            &set
        ));
        assert!(!is_model_installed_lmstudio_disk(
            "google/gemma-4-12B-it",
            &set
        ));
        assert!(!is_model_installed_lmstudio_disk(
            "google/gemma-4-12B-instruct",
            &set
        ));
        assert!(!is_model_installed_lmstudio_disk(
            "google/gemma-4-12B-chat",
            &set
        ));
    }

    #[test]
    fn test_collect_gguf_files_ignores_directories_named_like_models() {
        // A directory called `something.gguf` is not a model, and whatever is
        // inside it still needs scanning.
        let tree = TempTree::new("ggufdir");
        tree.touch("decoy.gguf/real.gguf");

        let found = collect_gguf_files(tree.path(), GGUF_SCAN_MAX_DEPTH);
        assert_eq!(names_of(&found), vec!["real.gguf"]);
        assert!(found.iter().all(|p| p.is_file()));
    }

    #[test]
    fn test_scan_lmstudio_models_dir_without_home_is_empty() {
        let (set, count) = scan_lmstudio_models_dir_at(None);
        assert!(set.is_empty());
        assert_eq!(count, 0);
    }

    // ── hf_name_to_mlx_candidates edge cases ─────────────────────────

    #[test]
    fn test_hf_name_to_mlx_candidates_bare_model_name() {
        let candidates = hf_name_to_mlx_candidates("Phi-4");
        assert!(candidates.iter().any(|c| c.contains("phi-4")));
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_no_duplicates() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-3.1-8B-Instruct");
        let unique: HashSet<_> = candidates.iter().collect();
        assert_eq!(
            unique.len(),
            candidates.len(),
            "candidates should have no duplicates: {:?}",
            candidates
        );
    }

    // ── hf_name_to_ollama_candidates edge cases ──────────────────────

    #[test]
    fn test_hf_name_to_ollama_candidates_unknown_generates_fallback() {
        // Models without an explicit mapping should still generate
        // heuristic candidates so installed detection has something to match.
        let candidates = hf_name_to_ollama_candidates("totally-unknown/model-xyz");
        assert!(
            !candidates.is_empty(),
            "fallback candidate generation should produce at least one entry"
        );
        // All candidates should be lowercased
        for c in &candidates {
            assert_eq!(c, &c.to_lowercase(), "candidate should be lowercase: {c}");
        }
    }

    #[test]
    fn test_hf_name_to_ollama_candidates_multiple_models() {
        // Test a variety of known models
        assert!(!hf_name_to_ollama_candidates("meta-llama/Llama-3.1-8B-Instruct").is_empty());
        assert!(!hf_name_to_ollama_candidates("Qwen/Qwen2.5-Coder-7B-Instruct").is_empty());
        assert!(!hf_name_to_ollama_candidates("google/gemma-2-9b-it").is_empty());
    }

    // ── split_name_and_size ───────────────────────────────────────

    #[test]
    fn test_split_name_and_size_basic() {
        assert_eq!(
            split_name_and_size("qwen2.5-coder-14b"),
            Some(("qwen2.5-coder", "14b"))
        );
    }

    #[test]
    fn test_split_name_and_size_moe() {
        assert_eq!(
            split_name_and_size("qwen3-coder-30b-a3b"),
            Some(("qwen3-coder", "30b-a3b"))
        );
    }

    #[test]
    fn test_split_name_and_size_no_size() {
        // "phi-4" has no "b" suffix — "4" is not a size tag
        assert_eq!(split_name_and_size("phi-4"), None);
    }

    #[test]
    fn test_split_name_and_size_deepseek() {
        assert_eq!(
            split_name_and_size("deepseek-r1-distill-qwen-32b"),
            Some(("deepseek-r1-distill-qwen", "32b"))
        );
    }

    #[test]
    fn test_split_name_and_size_fractional() {
        assert_eq!(split_name_and_size("qwen3-1.7b"), Some(("qwen3", "1.7b")));
    }

    // ── fallback ollama candidate matching ──────────────────────────

    #[test]
    fn test_fallback_ollama_candidates_match_installed() {
        // Simulate a model NOT in OLLAMA_MAPPINGS but running in Ollama
        let candidates = hf_name_to_ollama_candidates("SomeOrg/CoolModel-13B-Instruct");
        // Should generate "coolmodel:13b" as a candidate
        assert!(
            candidates.contains(&"coolmodel:13b".to_string()),
            "expected 'coolmodel:13b' in candidates: {:?}",
            candidates
        );

        // Verify it matches against an installed set
        let mut installed = HashSet::new();
        installed.insert("coolmodel:13b".to_string());
        installed.insert("coolmodel".to_string());
        assert!(is_model_installed(
            "SomeOrg/CoolModel-13B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_fallback_ollama_moe_candidate() {
        // Use a fictitious MoE model that is NOT in OLLAMA_MAPPINGS
        let candidates = hf_name_to_ollama_candidates("FakeOrg/FakeModel-30B-A3B-Instruct");
        assert!(
            candidates.contains(&"fakemodel:30b-a3b".to_string()),
            "expected 'fakemodel:30b-a3b' in candidates: {:?}",
            candidates
        );
    }

    #[test]
    fn test_installed_hf_name_direct_match() {
        // /api/v1/installed returns the full HF name lowercased
        let mut installed = HashSet::new();
        installed.insert("deepseek-ai/deepseek-r1-distill-qwen-32b".to_string());
        assert!(is_model_installed(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            &installed
        ));
    }

    // ── Docker Model Runner ─────────────────────────────────────────

    #[test]
    fn test_docker_mr_catalog_parses() {
        // The embedded catalog should parse without errors
        let catalog = docker_mr_catalog();
        assert!(!catalog.is_empty(), "Docker MR catalog should not be empty");
    }

    #[test]
    fn test_docker_mr_pull_tag_returns_ai_prefixed() {
        let tag = docker_mr_pull_tag("meta-llama/Llama-3.1-70B-Instruct");
        assert!(tag.is_some());
        assert!(tag.unwrap().starts_with("ai/"));
    }

    #[test]
    fn test_docker_mr_candidates_includes_ai_prefix() {
        let candidates = hf_name_to_docker_mr_candidates("meta-llama/Llama-3.1-70B-Instruct");
        assert!(candidates.iter().any(|c| c.starts_with("ai/")));
    }

    #[test]
    fn test_docker_mr_candidates_unknown_returns_empty() {
        let candidates = hf_name_to_docker_mr_candidates("totally-unknown/model-xyz");
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_is_model_installed_docker_mr_exact() {
        let mut installed = HashSet::new();
        installed.insert("ai/llama3.1:70b".to_string());
        installed.insert("llama3.1:70b".to_string());
        installed.insert("llama3.1".to_string());
        assert!(is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_docker_mr_variant_suffix() {
        let mut installed = HashSet::new();
        installed.insert("ai/llama3.1:70b-q4_k_m".to_string());
        assert!(is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_docker_mr_not_installed() {
        let installed = HashSet::new();
        assert!(!is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_normalize_docker_mr_host_with_scheme() {
        assert_eq!(
            normalize_docker_mr_host("https://docker.example.com:12434"),
            Some("https://docker.example.com:12434".to_string())
        );
    }

    #[test]
    fn test_normalize_docker_mr_host_without_scheme() {
        assert_eq!(
            normalize_docker_mr_host("docker.example.com:12434"),
            Some("http://docker.example.com:12434".to_string())
        );
    }

    #[test]
    fn test_normalize_docker_mr_host_rejects_unsupported_scheme() {
        assert_eq!(
            normalize_docker_mr_host("ftp://docker.example.com:12434"),
            None
        );
    }

    // ── OpenAI-compatible identity disambiguation ─────────────────────

    #[test]
    fn test_omlx_status_payload_detected() {
        let payload = serde_json::json!({
            "status": "ok",
            "version": "0.4.4",
            "models_discovered": 1,
            "model_memory_max": 12_649_259_752u64,
            "cache_efficiency": 0.0
        });

        assert!(is_omlx_status_payload(&payload));
    }

    #[test]
    fn test_omlx_status_payload_rejects_generic_status() {
        let payload = serde_json::json!({
            "status": "ok",
            "version": "1.0.0"
        });

        assert!(!is_omlx_status_payload(&payload));
    }

    #[test]
    fn test_openai_model_list_detects_omlx_owner() {
        let list: OpenAiModelList = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "Qwen2.5-0.5B-Instruct-4bit",
                    "object": "model",
                    "owned_by": "omlx",
                    "max_model_len": 32768
                }
            ]
        }))
        .expect("test payload should parse");

        assert!(openai_model_list_is_omlx(&list));
    }

    #[test]
    fn test_openai_model_list_keeps_regular_vllm_available() {
        let list: OpenAiModelList = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "meta-llama/Llama-3.1-8B-Instruct",
                    "object": "model",
                    "owned_by": "vllm"
                }
            ]
        }))
        .expect("test payload should parse");

        assert!(!openai_model_list_is_omlx(&list));
    }

    /// Verbatim `/v1/models` body captured from llama-swap v251 (4ec3175)
    /// on 2026-08-23. The regression tests below feed it to both providers.
    const LLAMA_SWAP_MODELS_FIXTURE: &str = r#"{"data":[{"id":"llama-3.2-1b-instruct","object":"model","created":1787482072,"owned_by":"llama-swap","meta":{"llamaswap":{"type":"model"}},"status":{"value":"unloaded"}}],"object":"list"}"#;

    /// Verbatim `/v1/models` body captured from llama-server build 10520
    /// (cd644c395) on 2026-08-23.
    const LLAMA_SERVER_MODELS_FIXTURE: &str = r#"{"models":[{"name":"models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","model":"models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","aliases":["models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"],"tags":[],"object":"model","created":1787482072,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":128256,"n_ctx":131072,"n_ctx_train":131072,"n_embd":2048,"n_params":1235814432,"size":799862912,"ftype":"Q4_K - Medium"}}]}"#;

    /// Verbatim `/v1/models` body captured from mlx_lm.server 0.31.3 on
    /// 2026-08-23: no `owned_by` field at all.
    const MLX_LM_MODELS_FIXTURE: &str = r#"{"object": "list", "data": [{"id": "mlx-community/Llama-3.2-1B-Instruct-4bit", "object": "model", "created": 1787482072}]}"#;

    /// Verbatim `/v1/models` body captured from vLLM 0.28.0 on 2026-09-01.
    const VLLM_MODELS_FIXTURE: &str = r#"{"object":"list","data":[{"id":"facebook/opt-125m","object":"model","created":1788290125,"owned_by":"vllm","root":"facebook/opt-125m","parent":null,"max_model_len":512}]}"#;
    const FERRUM_MODELS_FIXTURE: &str = r#"{"data":[{"id":"ferrum","owned_by":"ferrum"}]}"#;

    /// Verbatim `/v1/models` body captured from Docker Model Runner on
    /// 2026-09-01: owned_by "docker" plus a per-model `dmr` object.
    const DOCKER_MR_MODELS_FIXTURE: &str = r#"{"object":"list","data":[{"id":"docker.io/ai/smollm2:360M-Q4_K_M","object":"model","created":1742816981,"owned_by":"docker","dmr":{"architecture":"llama","parameters":"361.82 M","quantization":"IQ2_XXS/Q4_K_M","size":"256.35 MiB"}}]}"#;

    /// Verbatim `/api/v0/models` body captured from LM Studio 0.4.23 on
    /// 2026-09-01. The `compatibility_type`/`state` fields are LM Studio-native
    /// and absent from the OpenAI schema, which is how the runtime is identified.
    const LM_STUDIO_MODELS_FIXTURE: &str = r#"{"data":[{"id":"text-embedding-nomic-embed-text-v1.5","object":"model","type":"embeddings","publisher":"nomic-ai","arch":"nomic-bert","compatibility_type":"gguf","quantization":"Q4_K_M","state":"loaded","max_context_length":2048}],"object":"list"}"#;

    #[test]
    fn test_classify_openai_endpoint_measured_payloads() {
        let swap: OpenAiModelList =
            serde_json::from_str(LLAMA_SWAP_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(None, &swap),
            OpenAiEndpointIdentity::LlamaSwap
        );

        let llamacpp: OpenAiModelList =
            serde_json::from_str(LLAMA_SERVER_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(Some("llama.cpp"), &llamacpp),
            OpenAiEndpointIdentity::LlamaCpp
        );
        // The body alone is enough: the entry carries `owned_by: "llamacpp"`.
        assert_eq!(
            classify_openai_endpoint(None, &llamacpp),
            OpenAiEndpointIdentity::LlamaCpp
        );

        let mlx: OpenAiModelList =
            serde_json::from_str(MLX_LM_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(Some("BaseHTTP/0.6 Python/3.14.7"), &mlx),
            OpenAiEndpointIdentity::Unrecognized
        );
        // A llama.cpp Server header marks the endpoint even when the body
        // carries no owner, as during model load.
        let empty: OpenAiModelList = serde_json::from_str(r#"{"object":"list","data":[]}"#)
            .expect("empty list should parse");
        assert_eq!(
            classify_openai_endpoint(Some("llama.cpp"), &empty),
            OpenAiEndpointIdentity::LlamaCpp
        );
        assert_eq!(
            classify_openai_endpoint(None, &empty),
            OpenAiEndpointIdentity::Unrecognized
        );

        // Follow-up providers, measured 2026-09-01.
        let vllm: OpenAiModelList =
            serde_json::from_str(VLLM_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(Some("uvicorn"), &vllm),
            OpenAiEndpointIdentity::Vllm
        );

        let ferrum: OpenAiModelList =
            serde_json::from_str(FERRUM_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(None, &ferrum),
            OpenAiEndpointIdentity::Ferrum
        );

        let docker: OpenAiModelList =
            serde_json::from_str(DOCKER_MR_MODELS_FIXTURE).expect("fixture should parse");
        assert_eq!(
            classify_openai_endpoint(None, &docker),
            OpenAiEndpointIdentity::DockerModelRunner
        );
    }

    /// Serve one HTTP response carrying the given body on an ephemeral
    /// loopback port, and return its base URL.
    fn serve_fixture(body: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("test listener addr");
        std::thread::spawn(move || {
            // Serve every connection: LM Studio detection probes both
            // /v1/models and the native /api/v0/models on the same base URL.
            while let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    // Regression for #791: a llama-swap response on the probed port must not
    // be imported as MLX models. Goes through the real probe, which applies
    // classify_openai_endpoint, the same gate the RamaLama path uses below.
    // On non-macOS the MLX probe skips the network path entirely, which
    // satisfies the same assertion.
    #[test]
    fn test_llama_swap_endpoint_not_imported_as_mlx() {
        let provider = MlxProvider::with_server_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        let (_, installed) = provider.detect_with_installed();
        assert!(!installed.contains("llama-3.2-1b-instruct"));
    }

    // Regression for #790: the same llama-swap response must not mark the
    // RamaLama provider as serving models. Same gate as the MLX path above.
    #[test]
    fn test_llama_swap_endpoint_not_imported_as_ramalama() {
        let provider = RamaLamaProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        let (_, installed, _count) = provider.detect_with_installed();
        assert!(!installed.contains("llama-3.2-1b-instruct"));
    }

    // Control: the same harness with a payload carrying no foreign marker IS
    // imported, so the rejection above comes from the identity gate and not
    // from a failed fetch.
    #[test]
    fn test_unmarked_endpoint_still_imported_as_ramalama() {
        let provider = RamaLamaProvider::with_base_url(&serve_fixture(MLX_LM_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(available);
        assert_eq!(count, 1);
        assert!(installed.contains("mlx-community/llama-3.2-1b-instruct-4bit"));
    }

    // #790 follow-up, vLLM. Positive identification: only an endpoint that
    // identifies as vLLM is imported, so both a llama-swap proxy and a plain
    // llama.cpp server answering on the port are rejected.
    #[test]
    fn test_llama_swap_endpoint_not_imported_as_vllm() {
        let provider = VllmProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
        assert!(!installed.contains("llama-3.2-1b-instruct"));
    }

    #[test]
    fn test_llama_cpp_endpoint_not_imported_as_vllm() {
        let provider = VllmProvider::with_base_url(&serve_fixture(LLAMA_SERVER_MODELS_FIXTURE));
        let (available, _installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
    }

    // Control: an unmarked OpenAI payload is NOT imported. Under positive
    // identification, vLLM imports only an endpoint that identifies as vLLM,
    // so a foreign server on the port is rejected rather than trusted.
    #[test]
    fn test_unmarked_endpoint_not_imported_as_vllm() {
        let provider = VllmProvider::with_base_url(&serve_fixture(MLX_LM_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
        assert!(!installed.contains("mlx-community/llama-3.2-1b-instruct-4bit"));
    }

    // A genuine vLLM endpoint (owned_by "vllm", measured 2026-09-01) IS
    // imported.
    #[test]
    fn test_vllm_endpoint_imported_as_vllm() {
        let provider = VllmProvider::with_base_url(&serve_fixture(VLLM_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(available);
        assert_eq!(count, 1);
        assert!(installed.contains("facebook/opt-125m"));
    }

    // Availability must match the identity gate, not mere reachability:
    // a foreign OpenAI server on the port is not "available".
    #[test]
    fn test_vllm_not_available_for_foreign_endpoint() {
        let provider = VllmProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        assert!(!provider.is_available());
    }

    #[test]
    fn test_vllm_available_for_vllm_endpoint() {
        let provider = VllmProvider::with_base_url(&serve_fixture(VLLM_MODELS_FIXTURE));
        assert!(provider.is_available());
    }

    #[test]
    fn test_lmstudio_not_available_for_foreign_endpoint() {
        let provider = LmStudioProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        assert!(!provider.is_available());
    }

    #[test]
    fn test_lmstudio_available_for_lmstudio_endpoint() {
        let provider = LmStudioProvider::with_base_url(&serve_fixture(LM_STUDIO_MODELS_FIXTURE));
        assert!(provider.is_available());
    }

    // An empty `data` array on /api/v0/models carries no native field, so a
    // foreign service exposing that path with an empty list is not identified
    // as LM Studio.
    #[test]
    fn test_empty_native_list_not_identified_as_lmstudio() {
        let provider =
            LmStudioProvider::with_base_url(&serve_fixture(r#"{"object":"list","data":[]}"#));
        assert!(!provider.is_available());
        let (available, _installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
    }

    // #790 follow-up, LM Studio. Positive identification via the native
    // /api/v0 API: only an endpoint that identifies as LM Studio is imported.
    #[test]
    fn test_llama_swap_endpoint_not_imported_as_lmstudio() {
        let provider = LmStudioProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
        assert!(!installed.contains("llama-3.2-1b-instruct"));
    }

    #[test]
    fn test_unmarked_endpoint_not_imported_as_lmstudio() {
        let provider = LmStudioProvider::with_base_url(&serve_fixture(MLX_LM_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
        assert!(!installed.contains("mlx-community/llama-3.2-1b-instruct-4bit"));
    }

    // A genuine LM Studio endpoint is identified by its native /api/v0/models
    // route (measured 2026-09-01) and IS imported.
    #[test]
    fn test_lmstudio_endpoint_imported_as_lmstudio() {
        // Same body served on /v1/models (for ids) and the native /api/v0/models
        // probe (for identity); the native fields make endpoint_is_lmstudio true.
        let provider = LmStudioProvider::with_base_url(&serve_fixture(LM_STUDIO_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(available);
        assert_eq!(count, 1);
        assert!(installed.contains("text-embedding-nomic-embed-text-v1.5"));
    }

    // #790 follow-up, Docker Model Runner. Positive identification: only an
    // endpoint that identifies as Docker Model Runner (owned_by "docker") is
    // imported. The OS gate short-circuits the network probe on Linux, so
    // these exercise the identity gate on macOS/Windows only.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_llama_swap_endpoint_not_imported_as_docker_mr() {
        let provider =
            DockerModelRunnerProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
        assert!(!installed.contains("llama-3.2-1b-instruct"));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_llama_cpp_endpoint_not_imported_as_docker_mr() {
        let provider =
            DockerModelRunnerProvider::with_base_url(&serve_fixture(LLAMA_SERVER_MODELS_FIXTURE));
        let (available, _installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
    }

    // A genuine Docker Model Runner endpoint (owned_by "docker", measured
    // 2026-09-01) IS imported.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_docker_mr_endpoint_imported_as_docker_mr() {
        let provider =
            DockerModelRunnerProvider::with_base_url(&serve_fixture(DOCKER_MR_MODELS_FIXTURE));
        let (available, installed, count) = provider.detect_with_installed();
        assert!(available);
        assert_eq!(count, 1);
        assert!(installed.contains("docker.io/ai/smollm2:360m-q4_k_m"));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_docker_mr_not_available_for_foreign_endpoint() {
        let provider =
            DockerModelRunnerProvider::with_base_url(&serve_fixture(LLAMA_SWAP_MODELS_FIXTURE));
        assert!(!provider.is_available());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_docker_mr_available_for_docker_mr_endpoint() {
        let provider =
            DockerModelRunnerProvider::with_base_url(&serve_fixture(DOCKER_MR_MODELS_FIXTURE));
        assert!(provider.is_available());
    }

    /// Serve different bodies per path prefix on an ephemeral loopback port,
    /// so a probe hitting both `/v1/models` and `/` can be exercised.
    #[cfg(not(target_os = "linux"))]
    fn serve_routes(routes: &'static [(&'static str, &'static str, &'static str)]) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("test listener addr");
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (_, content_type, body) = routes
                    .iter()
                    .find(|(prefix, _, _)| path.starts_with(prefix))
                    .or_else(|| routes.last())
                    .expect("routes not empty");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    content_type,
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    // A running Docker Model Runner with no models yet returns an empty
    // /v1/models list with no owned_by marker; the root banner identifies it,
    // so the provider stays visible (measured root body, 2026-09-01).
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_empty_docker_mr_endpoint_still_available() {
        let base = serve_routes(&[
            (
                "/v1/models",
                "application/json",
                r#"{"object":"list","data":[]}"#,
            ),
            ("/", "text/plain", "Docker Model Runner is running"),
        ]);
        let provider = DockerModelRunnerProvider::with_base_url(&base);
        assert!(provider.is_available());
        let (available, installed, count) = provider.detect_with_installed();
        assert!(available);
        assert_eq!(count, 0);
        assert!(installed.is_empty());
    }

    // An empty model list on a server whose root does NOT carry the banner
    // stays unidentified: reachability alone is not evidence.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_empty_unmarked_endpoint_not_available_as_docker_mr() {
        let base = serve_routes(&[
            (
                "/v1/models",
                "application/json",
                r#"{"object":"list","data":[]}"#,
            ),
            ("/", "text/plain", "some other server"),
        ]);
        let provider = DockerModelRunnerProvider::with_base_url(&base);
        assert!(!provider.is_available());
        let (available, _installed, count) = provider.detect_with_installed();
        assert!(!available);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_openai_model_list_without_owner_is_not_omlx() {
        let list: OpenAiModelList = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "meta-llama/Llama-3.1-8B-Instruct",
                    "object": "model"
                }
            ]
        }))
        .expect("test payload should parse");

        assert!(!openai_model_list_is_omlx(&list));
    }

    // ── vLLM ──────────────────────────────────────────────────────────

    #[test]
    fn test_hf_name_to_vllm_candidates() {
        let candidates = hf_name_to_vllm_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            candidates
                .iter()
                .any(|c| c == "meta-llama/llama-3.1-8b-instruct")
        );
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b-instruct"));
        // stripped variant (without -instruct)
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b"));
    }

    #[test]
    fn test_is_model_installed_vllm() {
        let mut installed = HashSet::new();
        installed.insert("meta-llama/llama-3.1-8b-instruct".to_string());
        assert!(is_model_installed_vllm(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
        assert!(!is_model_installed_vllm(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_normalize_vllm_host_with_scheme() {
        assert_eq!(
            normalize_vllm_host("http://myhost:8000"),
            Some("http://myhost:8000".to_string())
        );
    }

    #[test]
    fn test_normalize_vllm_host_without_scheme() {
        assert_eq!(
            normalize_vllm_host("myhost:8000"),
            Some("http://myhost:8000".to_string())
        );
    }

    #[test]
    fn test_normalize_vllm_host_rejects_unsupported_scheme() {
        assert_eq!(normalize_vllm_host("ftp://myhost:8000"), None);
    }

    #[test]
    fn test_normalize_vllm_host_empty() {
        assert_eq!(normalize_vllm_host(""), None);
        assert_eq!(normalize_vllm_host("  "), None);
    }

    #[test]
    fn test_hf_name_to_ramalama_candidates() {
        let candidates = hf_name_to_ramalama_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            candidates
                .iter()
                .any(|c| c == "meta-llama/llama-3.1-8b-instruct")
        );
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b-instruct"));
        // stripped variant (without -instruct)
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b"));
    }

    #[test]
    fn test_is_model_installed_ramalama() {
        let mut installed = HashSet::new();
        installed.insert("meta-llama/llama-3.1-8b-instruct".to_string());
        assert!(is_model_installed_ramalama(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
        assert!(!is_model_installed_ramalama(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_parse_ramalama_store_extracts_names() {
        let json = br#"[
            {"shortname":"granite","name":"ollama://granite-code:8b","modified":"2026-01-01","size":123},
            {"shortname":"","name":"huggingface://meta-llama/Llama-3.1-8B-Instruct","modified":"2026-01-01","size":456}
        ]"#;
        let (set, count) = parse_ramalama_store(json).expect("valid json parses");
        assert_eq!(count, 2);
        // Full transport-qualified names, lowercased.
        assert!(set.contains("ollama://granite-code:8b"));
        assert!(set.contains("huggingface://meta-llama/llama-3.1-8b-instruct"));
        // Trailing path components, for substring matching.
        assert!(set.contains("granite-code:8b"));
        assert!(set.contains("llama-3.1-8b-instruct"));
        // Shortname included when present.
        assert!(set.contains("granite"));
    }

    #[test]
    fn test_parse_ramalama_store_matches_hf_model() {
        let json = br#"[{"shortname":"","name":"huggingface://meta-llama/Llama-3.1-8B-Instruct","modified":"x","size":1}]"#;
        let (set, _) = parse_ramalama_store(json).expect("valid json parses");
        // Store-detected models resolve through the same matcher as the server path.
        assert!(is_model_installed_ramalama(
            "meta-llama/Llama-3.1-8B-Instruct",
            &set
        ));
    }

    #[test]
    fn test_parse_ramalama_store_empty_and_invalid() {
        let (set, count) = parse_ramalama_store(b"[]").expect("empty array parses");
        assert_eq!(count, 0);
        assert!(set.is_empty());
        assert!(parse_ramalama_store(b"not json").is_none());
    }

    #[test]
    fn test_normalize_ramalama_host_with_scheme() {
        assert_eq!(
            normalize_ramalama_host("http://myhost:8080"),
            Some("http://myhost:8080".to_string())
        );
    }

    #[test]
    fn test_normalize_ramalama_host_without_scheme() {
        assert_eq!(
            normalize_ramalama_host("myhost:8080"),
            Some("http://myhost:8080".to_string())
        );
    }

    #[test]
    fn test_normalize_ramalama_host_rejects_unsupported_scheme() {
        assert_eq!(normalize_ramalama_host("ftp://myhost:8080"), None);
    }

    #[test]
    fn test_normalize_ramalama_host_empty() {
        assert_eq!(normalize_ramalama_host(""), None);
        assert_eq!(normalize_ramalama_host("  "), None);
    }

    #[test]
    fn test_docker_model_runner_host_filtering() {
        // Test the DOCKER_MODEL_RUNNER_HOST filtering logic without mutating the
        // process environment. is_docker_desktop_running() applies
        // `!v.trim().is_empty()` to the env var value.
        fn host_is_set(val: Option<&str>) -> bool {
            val.map(|v| !v.trim().is_empty()).unwrap_or(false)
        }

        // Non-empty value should count as set
        assert!(host_is_set(Some("localhost:12434")));
        // Empty string should NOT count
        assert!(!host_is_set(Some("")));
        // Whitespace-only should NOT count
        assert!(!host_is_set(Some("   ")));
        // Missing env var should NOT count
        assert!(!host_is_set(None));
    }

    #[test]
    fn test_ollama_build_installed_set_skips_cloud_models() {
        let models = vec![
            ollama_entry("qwen3-coder:480b-cloud", 0), // cloud: -cloud suffix + size 0
            ollama_entry("gpt-oss:120b-cloud", 0),     // cloud
            ollama_entry("llama3.1:8b-instruct-q4_K_M", 4_700_000_000), // local
        ];

        let (set, count) = build_installed_set(models);

        // Only the local model is counted and inserted.
        assert_eq!(count, 1, "cloud models must not count as installed");
        assert!(set.contains("llama3.1:8b-instruct-q4_k_m"));
        // The tag is sized, so no family stem — see #861.
        assert!(!set.contains("llama3.1"));

        // The cloud family stem must NOT leak in — that was the #619 false positive.
        assert!(
            !set.contains("qwen3-coder"),
            "cloud family stem must not mark unrelated models installed"
        );
        assert!(!set.contains("gpt-oss"));
        assert!(!set.contains("qwen3-coder:480b-cloud"));
    }

    #[test]
    fn test_ollama_is_cloud_detection() {
        assert!(ollama_entry("qwen3-coder:480b-cloud", 0).is_cloud());

        // A local model with a real on-disk size is not cloud.
        assert!(!ollama_entry("llama3.1:8b", 4_700_000_000).is_cloud());

        // Defensive: a zero-size entry is treated as not-local even without the suffix.
        assert!(ollama_entry("mystery:latest", 0).is_cloud());
    }

    // ── installed-set breadth (#861) ─────────────────────────────────

    fn ollama_entry(name: &str, size: u64) -> OllamaModel {
        OllamaModel {
            name: name.to_string(),
            size,
            ..Default::default()
        }
    }

    fn ollama_entry_sized(name: &str, parameter_size: &str) -> OllamaModel {
        OllamaModel {
            name: name.to_string(),
            size: 4_700_000_000,
            details: OllamaModelDetails {
                parameter_size: parameter_size.to_string(),
            },
        }
    }

    #[test]
    fn sized_install_does_not_mark_the_family_installed() {
        // The #861 report: three models installed, dozens shown as installed.
        let (installed, _) = build_installed_set(vec![ollama_entry_sized("qwen3:8b", "8.2B")]);

        assert!(is_model_installed("Qwen/Qwen3-8B", &installed));
        for sibling in [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "unsloth/Qwen3-30B-A3B-GGUF",
        ] {
            assert!(
                !is_model_installed(sibling, &installed),
                "{sibling} must not look installed because qwen3:8b is"
            );
        }
    }

    #[test]
    fn latest_install_resolves_to_its_parameter_size() {
        // `ollama pull qwen3` leaves a `:latest` tag whose size only the
        // reported parameter count reveals.
        let (installed, _) = build_installed_set(vec![ollama_entry_sized("qwen3:latest", "8.2B")]);

        assert!(installed.contains("qwen3:8b"));
        assert!(is_model_installed("Qwen/Qwen3-8B", &installed));
        assert!(!is_model_installed("Qwen/Qwen3-32B", &installed));
    }

    #[test]
    fn latest_install_without_a_parameter_size_stays_family_level() {
        // No parameter count to work with: the family stem is all we have, and
        // it only matches catalog entries that carry no size of their own.
        let (installed, _) = build_installed_set(vec![ollama_entry("qwq:latest", 4_700_000_000)]);

        assert!(is_model_installed("Qwen/QwQ-32B", &installed));
    }

    #[test]
    fn sizeless_mapping_still_matches_a_sized_install() {
        // `OLLAMA_MAPPINGS` resolves microsoft/phi-4 to the size-less tag
        // `phi4`, which Ollama stores as `phi4:14b`.
        let (installed, _) = build_installed_set(vec![ollama_entry_sized("phi4:14b", "14.7B")]);

        assert!(is_model_installed("microsoft/phi-4", &installed));
    }

    #[test]
    fn every_mapped_model_is_still_detected_from_its_own_tag() {
        // Narrowing what counts as installed must not cost us any model in the
        // authoritative table: pulling exactly the tag a model maps to has to
        // mark that model, and only that model, installed.
        for (hf_suffix, tag) in OLLAMA_MAPPINGS {
            let (installed, _) = build_installed_set(vec![ollama_entry(tag, 4_700_000_000)]);
            assert!(
                is_model_installed(hf_suffix, &installed),
                "{hf_suffix} not detected from its own tag {tag}"
            );
        }
    }

    #[test]
    fn parameter_size_yields_marketing_and_verbatim_tags() {
        // Most tags carry the marketing size: qwen2.5:14b reports "14.8B".
        assert_eq!(size_tokens_from_parameter_size("14.8B"), ["14b", "14.8b"]);
        assert_eq!(size_tokens_from_parameter_size("8.2B"), ["8b", "8.2b"]);
        // Decimal-tagged families (solar:10.7b) need the verbatim form.
        assert_eq!(size_tokens_from_parameter_size("10.7B"), ["10b", "10.7b"]);
        // A whole number yields one token, not a duplicate.
        assert_eq!(size_tokens_from_parameter_size("8B"), ["8b"]);
        // Sub-1B counts are reported in M and have no usable tag form.
        assert!(size_tokens_from_parameter_size("596.05M").is_empty());
        assert!(size_tokens_from_parameter_size("").is_empty());
    }

    #[test]
    fn latest_install_of_a_decimal_tagged_family_is_detected() {
        // `solar:latest` is `solar:10.7b`; the truncated "10b" alias alone
        // would miss it.
        let (installed, _) = build_installed_set(vec![ollama_entry_sized("solar:latest", "10.7B")]);

        assert!(is_model_installed(
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            &installed
        ));
    }
}