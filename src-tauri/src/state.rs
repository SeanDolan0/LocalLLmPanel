//! Persisted configuration (JSON in %APPDATA%) and live server registry.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlamaCppChannel {
    #[default]
    Upstream,
    Prism,
}

impl LlamaCppChannel {
    pub fn repo(&self) -> (&'static str, &'static str) {
        match self {
            LlamaCppChannel::Upstream => ("ggml-org", "llama.cpp"),
            LlamaCppChannel::Prism => ("PrismML-Eng", "llama.cpp"),
        }
    }

    pub fn branch(&self) -> &'static str {
        match self {
            LlamaCppChannel::Upstream => "master",
            LlamaCppChannel::Prism => "prism",
        }
    }

    pub fn dir_suffix(&self) -> &'static str {
        match self {
            LlamaCppChannel::Upstream => "upstream",
            LlamaCppChannel::Prism => "prism",
        }
    }
}

pub const CONFIG_DIR_NAME: &str = "local-llm-panel";
pub const CONFIG_FILE_NAME: &str = "config.json";
pub const CONVERSATIONS_FILE_NAME: &str = "conversations.json";
pub const BENCHMARKS_FILE_NAME: &str = "benchmarks.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Base64-encoded data URLs of attached images for vision-capable (VLM) models.
    /// When present, the message is sent to the backend as a multimodal content array.
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

impl ChatMessage {
    /// Build the OpenAI-compatible message body. When `images` are present, `content`
    /// becomes a multimodal array of `{"type": "text"}` + `{"type": "image_url"}` parts.
    pub fn payload_body(&self) -> serde_json::Value {
        if let Some(imgs) = &self.images {
            let mut parts = vec![serde_json::json!({"type": "text", "text": self.content})];
            for img in imgs {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": img}
                }));
            }
            serde_json::json!({"role": self.role, "content": parts})
        } else {
            serde_json::json!({"role": self.role, "content": self.content})
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Conversation {
    pub id: String,
    pub server_id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<ChatMessage>,
}

impl Conversation {
    pub fn path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join(CONVERSATIONS_FILE_NAME)
    }

    pub fn load_all() -> Vec<Conversation> {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    pub fn save_all(convs: &[Conversation]) -> Result<(), String> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(convs).map_err(|e| format!("serialize: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename: {e}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    pub id: String,
    pub server_id: String,
    pub model_id: String,
    pub quant: Option<String>,
    pub timestamp: u64,
    pub prompt_tok_s: f64,
    pub gen_tok_s: f64,
    pub latency_ms: f64,
    pub prompt_count: usize,
}

impl BenchmarkRun {
    pub fn path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join(BENCHMARKS_FILE_NAME)
    }

    pub fn load_all() -> Vec<BenchmarkRun> {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    pub fn save_all(runs: &[BenchmarkRun]) -> Result<(), String> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(runs).map_err(|e| format!("serialize: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename: {e}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CachedEnrichment {
    pub stats: crate::hf::EnrichedStats,
    pub fetched_at: Instant,
}

#[derive(Debug, Clone)]
pub struct CachedQuants {
    pub variants: Vec<crate::hf::QuantVariant>,
    pub fetched_at: Instant,
}

// ---------------------------------------------------------------------------
// Server definitions & measured stats (persisted)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerDef {
    #[serde(default = "default_backend")]
    pub backend: String,
    pub id: String,
    pub name: String,
    pub model_id: String,
    /// "instruct" | "embed"
    pub task: String,
    pub port: u16,
    pub gpu_mem_util: f64,
    pub max_model_len: Option<usize>,
    /// "auto" | "fp16" | "fp8" | "awq" | "gptq" (native values are GGUF metadata)
    pub quant: String,
    /// Public API model name. vLLM maps it to --served-model-name and
    /// llama.cpp maps it to --alias.
    pub served_model_name: Option<String>,
    /// Per-server vLLM KV-cache dtype. `None` inherits the global setting.
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    /// llama.cpp binary channel: "upstream" (ggml-org) or "prism" (PrismML-Eng/llama.cpp@prism)
    #[serde(default)]
    pub llamacpp_channel: LlamaCppChannel,
    /// Skip CUDA-graph capture (`--enforce-eager`). WSL2's full-graph
    /// capture can hang for minutes or stall; eager mode starts reliably.
    /// Default true; throughput is slightly lower but startup is robust.
    #[serde(default = "default_true")]
    pub enforce_eager: bool,
    /// Estimated params (billions) memoized at create time (for VRAM guidance).
    pub params_b: Option<f64>,
    #[serde(default)]
    pub swap_space_gb: Option<usize>,
    #[serde(default)]
    pub cpu_offload_gb: Option<usize>,
    #[serde(default)]
    pub was_running: bool,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub mmproj_path: Option<String>,
    #[serde(default)]
    pub ctx_size: Option<usize>,
    /// Maximum GPU layers. `None` lets llama.cpp/`--fit` choose automatically.
    /// Older configurations commonly persisted `99`; serde keeps that value
    /// as `Some(99)` so those recipes retain their explicit behavior.
    #[serde(default)]
    pub n_gpu_layers: Option<usize>,
    #[serde(default)]
    pub n_cpu_moe: Option<usize>,
    /// Enable llama.cpp automatic device-memory fitting when supported.
    #[serde(default = "default_true")]
    pub fit: bool,
    /// Per-device memory margin passed to llama.cpp `--fit-target` (MiB).
    #[serde(default)]
    pub fit_target: Option<usize>,
    /// Comma-separated native llama.cpp device identifiers.
    #[serde(default)]
    pub device: Option<String>,
    /// Per-server OpenAI-compatible API key. It overrides the global vLLM key
    /// and is also passed directly to llama-server.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Native llama.cpp log verbosity (0..5).
    #[serde(default)]
    pub log_verbosity: Option<u8>,
    #[serde(default = "default_true")]
    pub flash_attn: bool,
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub ubatch_size: Option<usize>,
    #[serde(default = "default_parallel")]
    pub parallel: usize,
    #[serde(default = "default_true")]
    pub jinja: bool,
    #[serde(default)]
    pub no_kv_offload: bool,
    #[serde(default = "default_true")]
    pub metrics: bool,
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Per-server environment variables (merged with global default_env, per-server wins).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerBackend {
    Vllm,
    Llamacpp,
}

impl std::str::FromStr for ServerBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "vllm" => Ok(Self::Vllm),
            "llamacpp" => Ok(Self::Llamacpp),
            other => Err(format!("unsupported server backend '{other}'")),
        }
    }
}

fn default_backend() -> String {
    "vllm".to_string()
}

fn default_cache_type() -> String {
    "q8_0".to_string()
}

fn default_parallel() -> usize {
    0
}

fn default_true() -> bool {
    true
}

impl ServerDef {
    pub fn backend_kind(&self) -> Result<ServerBackend, String> {
        self.backend.parse()
    }

    /// Clear state that belongs only to the other backend before persisting a
    /// newly-created or explicitly-edited server. Legacy config loading remains
    /// non-destructive; normalization runs only at mutation boundaries.
    pub fn normalize_for_backend(&mut self) -> Result<(), String> {
        match self.backend_kind()? {
            ServerBackend::Vllm => {
                self.llamacpp_channel = LlamaCppChannel::Upstream;
                self.model_path = None;
                self.mmproj_path = None;
                self.ctx_size = None;
                self.n_gpu_layers = None;
                self.n_cpu_moe = None;
                self.fit = false;
                self.fit_target = None;
                self.device = None;
                self.log_verbosity = None;
                self.flash_attn = false;
                self.cache_type_k.clear();
                self.cache_type_v.clear();
                self.threads = None;
                self.batch_size = None;
                self.ubatch_size = None;
                self.parallel = 0;
                self.jinja = false;
                self.no_kv_offload = false;
                self.extra_args.clear();
            }
            ServerBackend::Llamacpp => {
                self.gpu_mem_util = 0.0;
                self.max_model_len = None;
                self.enforce_eager = false;
                self.swap_space_gb = None;
                self.cpu_offload_gb = None;
                self.kv_cache_dtype = None;
                self.task = "instruct".into();
                self.env.clear();
            }
        }
        Ok(())
    }

    pub fn effective_model_name(&self) -> String {
        self.served_model_name
            .clone()
            .unwrap_or_else(|| self.model_id.clone())
    }
}

/// Marker used in public/portable views when a secret exists but must not be
/// sent to the webview or written to an export file.  It is deliberately not
/// a valid credential and is stripped again at mutation boundaries.
pub const SECRET_PLACEHOLDER: &str = "__LLM_PANEL_SECRET_REDACTED__";

pub fn is_secret_placeholder(value: &str) -> bool {
    value == SECRET_PLACEHOLDER
}

/// Return a server definition safe to expose through the webview.  Keep the
/// keys so the settings UI can explain what is configured, but never return
/// their values.  Environment values are all redacted because a custom name
/// such as `FOO` can still carry a secret.
pub fn redact_server_def(def: &ServerDef) -> ServerDef {
    let mut redacted = def.clone();
    if redacted.api_key.as_deref().is_some_and(|value| !value.trim().is_empty()) {
        redacted.api_key = Some(SECRET_PLACEHOLDER.to_string());
    }
    redacted.env = redacted
        .env
        .keys()
        .map(|name| (name.clone(), SECRET_PLACEHOLDER.to_string()))
        .collect();
    redacted
}

/// Preserve secrets represented by [`SECRET_PLACEHOLDER`] when applying a
/// redacted view back to an existing server.  A new server never receives the
/// marker as a real environment value.
pub fn merge_server_secret_placeholders(incoming: &mut ServerDef, existing: Option<&ServerDef>) {
    if incoming.api_key.as_deref() == Some(SECRET_PLACEHOLDER) {
        incoming.api_key = existing.and_then(|server| server.api_key.clone());
    }
    let old_env = existing.map(|server| &server.env);
    let placeholder_keys: Vec<String> = incoming
        .env
        .iter()
        .filter(|(_, value)| is_secret_placeholder(value))
        .map(|(name, _)| name.clone())
        .collect();
    incoming
        .env
        .retain(|_, value| !is_secret_placeholder(value));
    if let Some(old_env) = old_env {
        for name in placeholder_keys {
            if let Some(previous) = old_env.get(&name) {
                incoming.env.insert(name, previous.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasuredStats {
    pub tokens_per_sec: Option<f64>,
    pub prompt_tokens_per_sec: Option<f64>,
    pub total_prompt_tokens: u64,
    pub total_generation_tokens: u64,
    pub requests: u64,
    pub measured_at_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Memory settings (persisted)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySettings {
    #[serde(default = "default_gpu_mem_util")]
    pub default_gpu_mem_util: f64,
    #[serde(default = "default_vram_overhead")]
    pub vram_overhead_mb: f64,
    #[serde(default = "default_true")]
    pub enable_ram_overflow: bool,
    #[serde(default)]
    pub manual_ram_limit_mb: Option<u64>,
    #[serde(default = "default_safety_reserve")]
    pub safety_reserve_mb: u64,
    #[serde(default = "default_true")]
    pub offload_weights_allowed: bool,
    #[serde(default)]
    pub max_context_cap: Option<usize>,
}

fn default_gpu_mem_util() -> f64 {
    0.85
}

fn default_vram_overhead() -> f64 {
    2500.0
}

fn default_safety_reserve() -> u64 {
    4096
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            default_gpu_mem_util: 0.85,
            vram_overhead_mb: 2500.0,
            enable_ram_overflow: true,
            manual_ram_limit_mb: None,
            safety_reserve_mb: 4096,
            offload_weights_allowed: true,
            max_context_cap: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Advanced developer settings (persisted)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdvancedSettings {
    /// Custom HF cache directory (e.g. /mnt/d/ai-models/hf)
    #[serde(default)]
    pub hf_home: Option<String>,
    /// Offline mode (HF_HUB_OFFLINE=1)
    #[serde(default)]
    pub hf_offline: bool,
    /// Default host binding: "127.0.0.1" (local) or "0.0.0.0" (LAN/remote)
    #[serde(default = "default_host")]
    pub host: String,
    /// Optional global API key for OpenAI-compatible endpoint
    #[serde(default)]
    pub api_key: Option<String>,
    /// Emit an OpenAI-compatible gateway router on 127.0.0.1:<gateway_port>.
    #[serde(default)]
    pub gateway_enabled: bool,
    /// Port for the built-in OpenAI-compatible gateway router.
    #[serde(default = "default_gateway_port")]
    pub gateway_port: u16,
    /// Default KV cache data type: "auto", "fp8", "fp8_e5m2", "fp8_e4m3"
    #[serde(default = "default_auto")]
    pub kv_cache_dtype: String,
    /// Enable prefix caching (KV cache reuse across prompts/turns)
    #[serde(default = "default_true")]
    pub enable_prefix_caching: bool,
    /// Enable chunked prefill (better interleave of prompt & decode)
    #[serde(default)]
    pub enable_chunked_prefill: bool,
    /// Max concurrency / sequence count limit (None = vLLM default 256)
    #[serde(default)]
    pub max_num_seqs: Option<usize>,
    /// Disable custom P2P all-reduce (often required for multi-GPU on consumer RTX or WSL2)
    #[serde(default)]
    pub disable_custom_all_reduce: bool,
    /// Logging level: "INFO", "DEBUG", "WARNING", "ERROR"
    #[serde(default = "default_info")]
    pub log_level: String,
    /// Extra arbitrary CLI arguments appended to vLLM launch (e.g. "--tensor-parallel-size 2")
    #[serde(default)]
    pub extra_vllm_args: Option<String>,
    /// Custom environment variables (KEY=VAL lines)
    #[serde(default)]
    pub custom_env_vars: Option<String>,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_auto() -> String {
    "auto".to_string()
}

fn default_gateway_port() -> u16 {
    crate::gateway::DEFAULT_GATEWAY_PORT
}

fn default_info() -> String {
    "INFO".to_string()
}

impl Default for AdvancedSettings {
    fn default() -> Self {
        Self {
            hf_home: None,
            hf_offline: false,
            host: default_host(),
            api_key: None,
            gateway_enabled: false,
            gateway_port: default_gateway_port(),
            kv_cache_dtype: default_auto(),
            enable_prefix_caching: true,
            enable_chunked_prefill: false,
            max_num_seqs: None,
            disable_custom_all_reduce: false,
            log_level: default_info(),
            extra_vllm_args: None,
            custom_env_vars: None,
        }
    }
}

pub fn redact_advanced_settings(settings: &AdvancedSettings) -> AdvancedSettings {
    let mut redacted = settings.clone();
    if redacted.api_key.as_deref().is_some_and(|value| !value.trim().is_empty()) {
        redacted.api_key = Some(SECRET_PLACEHOLDER.to_string());
    }
    // Custom environment text is intentionally omitted: parsing it here would
    // risk returning values whose names do not look secret-like.
    redacted.custom_env_vars = None;
    redacted
}

pub fn redact_config_for_display(config: &PersistedConfig) -> PersistedConfig {
    let mut redacted = config.clone();
    redacted.hf_token.clear();
    redacted.github_token.clear();
    redacted.advanced_settings = redact_advanced_settings(&config.advanced_settings);
    // The API-key field is represented by `api_key_configured`; do not put a
    // marker in a password input or send a marker back to the webview.
    redacted.advanced_settings.api_key = None;
    redacted.default_env = config
        .default_env
        .keys()
        .map(|name| (name.clone(), SECRET_PLACEHOLDER.to_string()))
        .collect();
    redacted.servers = config.servers.iter().map(redact_server_def).collect();
    redacted
}

// ---------------------------------------------------------------------------
// Persisted config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppChannelConfig {
    #[serde(default)]
    pub installed_tag: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default)]
    pub dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedConfig {
    pub distro: String,
    pub llm_dir: String,
    pub venv_dir: String,
    pub llamacpp_dir: String,
    pub gguf_dir: String,
    /// Per-channel llama.cpp installation state
    #[serde(default)]
    pub llamacpp_channels: BTreeMap<LlamaCppChannel, LlamaCppChannelConfig>,
    /// Legacy fields (for back-compat with config.json written by older versions)
    #[serde(default)]
    pub llamacpp_executable: Option<String>,
    #[serde(default)]
    pub llamacpp_installed_tag: Option<String>,
    #[serde(default)]
    pub llamacpp_version: Option<String>,
    #[serde(default)]
    pub llamacpp_help: Option<String>,
    pub hf_token: String,
    #[serde(default)]
    pub github_token: String,
    pub default_quant: String,
    pub servers: Vec<ServerDef>,
    pub measured: HashMap<String, MeasuredStats>,
    #[serde(default)]
    pub memory_settings: MemorySettings,
    #[serde(default)]
    pub advanced_settings: AdvancedSettings,
    #[serde(default = "default_true")]
    pub minimize_to_tray: bool,
    #[serde(default = "default_true")]
    pub auto_restart_crashed: bool,
    #[serde(default)]
    pub launch_at_login: bool,
    /// WSL paths of locally imported model folders (Task 9 local import).
    #[serde(default)]
    pub imported_local_models: Vec<String>,
    /// Global default environment variables for vLLM servers (merged under each server's env).
    #[serde(default)]
    pub default_env: BTreeMap<String, String>,
}

pub type AppConfig = PersistedConfig;

impl Default for LlamaCppChannelConfig {
    fn default() -> Self {
        let base_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join("llama.cpp");
        LlamaCppChannelConfig {
            installed_tag: None,
            version: None,
            help: None,
            executable: None,
            dir: base_dir.to_string_lossy().into_owned(),
        }
    }
}

impl Default for PersistedConfig {
    fn default() -> Self {
        let distro = crate::wsl::detect_default_distro()
            .unwrap_or_else(|| crate::wsl::APP_DISTRO_NAME.to_string());
        let mut default_env = BTreeMap::new();
        default_env.insert("VLLM_USE_FLASHINFER_SAMPLER".to_string(), "0".to_string());
        let mut llamacpp_channels = BTreeMap::new();
        llamacpp_channels.insert(LlamaCppChannel::Upstream, LlamaCppChannelConfig::default());
        llamacpp_channels.insert(LlamaCppChannel::Prism, LlamaCppChannelConfig::default());
        PersistedConfig {
            distro,
            llm_dir: "~/llm-lp".to_string(),
            venv_dir: "~/llm-lp/.venv".to_string(),
            llamacpp_dir: dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(CONFIG_DIR_NAME)
                .join("llama.cpp")
                .to_string_lossy()
                .into_owned(),
            gguf_dir: dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(CONFIG_DIR_NAME)
                .join("gguf")
                .to_string_lossy()
                .into_owned(),
            llamacpp_channels,
            llamacpp_executable: None,
            llamacpp_installed_tag: None,
            llamacpp_version: None,
            llamacpp_help: None,
            hf_token: String::new(),
            github_token: String::new(),
            default_quant: "fp16".to_string(),
            servers: Vec::new(),
            measured: HashMap::new(),
            memory_settings: MemorySettings::default(),
            advanced_settings: AdvancedSettings::default(),
            minimize_to_tray: true,
            auto_restart_crashed: true,
            launch_at_login: false,
            imported_local_models: Vec::new(),
            default_env,
        }
    }
}

fn replace_file_atomic(tmp: &PathBuf, destination: &PathBuf) -> Result<(), String> {
    match std::fs::rename(tmp, destination) {
        Ok(()) => Ok(()),
        Err(first_error) => {
            // Windows does not replace an existing destination with rename.
            // Keep a rollback copy while performing the replacement so a
            // failed write cannot destroy the last known-good config.
            let backup = destination.with_extension("json.previous");
            let _ = std::fs::remove_file(&backup);
            if destination.exists() {
                std::fs::rename(destination, &backup)
                    .map_err(|e| format!("backup existing config: {e}"))?;
            }
            match std::fs::rename(tmp, destination) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&backup);
                    Ok(())
                }
                Err(second_error) => {
                    let _ = std::fs::rename(&backup, destination);
                    Err(format!(
                        "replace config failed ({first_error}); rollback failed: {second_error}"
                    ))
                }
            }
        }
    }
}

impl PersistedConfig {
    pub fn path() -> PathBuf {
        #[cfg(test)]
        if let Ok(test_root) = std::env::var("LLM_TEST_CONFIG_DIR") {
            if !test_root.trim().is_empty() {
                return PathBuf::from(test_root).join(CONFIG_FILE_NAME);
            }
        }
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join(CONFIG_FILE_NAME)
    }

    pub fn load() -> Self {
        let path = Self::path();
        let mut cfg = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<PersistedConfig>(&text) {
                Ok(cfg) => cfg,
                Err(_) => {
                    // Corrupt config: back it up and start fresh.
                    let _ = std::fs::rename(&path, path.with_extension("json.bak"));
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        };
        for secret in [&mut cfg.hf_token, &mut cfg.github_token] {
            if secret.starts_with("dpapi:") {
                match crate::security::decrypt_token(secret) {
                    Ok(plain) => *secret = plain,
                    Err(error) => {
                        // Never use a still-encrypted value as a bearer token.  A
                        // failed DPAPI unlock is surfaced in the log and the
                        // unusable secret is cleared rather than leaked to a
                        // subprocess or API request.
                        eprintln!("[config] could not decrypt a stored token: {error}");
                        secret.clear();
                    }
                }
            }
        }
        // A persisted `was_running` flag is only a hint from a previous
        // process.  Process handles do not survive an app crash, so never
        // present stale state as live; explicit start/recovery is required.
        for server in &mut cfg.servers {
            server.was_running = false;
        }
        // Ensure default_env has the FlashInfer sampler disabled by default for existing configs.
        if cfg.default_env.is_empty() {
            cfg.default_env.insert("VLLM_USE_FLASHINFER_SAMPLER".to_string(), "0".to_string());
        }
        // Back-compat: migrate legacy single-install fields to Upstream channel config
        if cfg.llamacpp_channels.is_empty() {
            let mut channels = BTreeMap::new();
            let mut upstream = LlamaCppChannelConfig::default();
            upstream.installed_tag = cfg.llamacpp_installed_tag.clone();
            upstream.version = cfg.llamacpp_version.clone();
            upstream.help = cfg.llamacpp_help.clone();
            upstream.executable = cfg.llamacpp_executable.clone();
            upstream.dir = cfg.llamacpp_dir.clone();
            channels.insert(LlamaCppChannel::Upstream, upstream);
            channels.insert(LlamaCppChannel::Prism, LlamaCppChannelConfig::default());
            cfg.llamacpp_channels = channels;
        }
        cfg
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
        let mut on_disk = self.clone();
        if on_disk.hf_token == SECRET_PLACEHOLDER {
            on_disk.hf_token.clear();
        }
        if on_disk.github_token == SECRET_PLACEHOLDER {
            on_disk.github_token.clear();
        }
        if on_disk.advanced_settings.api_key.as_deref() == Some(SECRET_PLACEHOLDER) {
            on_disk.advanced_settings.api_key = None;
        }
        if !on_disk.hf_token.is_empty() && !on_disk.hf_token.starts_with("dpapi:") {
            on_disk.hf_token = crate::security::encrypt_token(&on_disk.hf_token)
                .map_err(|e| format!("encrypt Hugging Face token: {e}"))?;
        }
        if !on_disk.github_token.is_empty() && !on_disk.github_token.starts_with("dpapi:") {
            on_disk.github_token = crate::security::encrypt_token(&on_disk.github_token)
                .map_err(|e| format!("encrypt GitHub token: {e}"))?;
        }
        let text = serde_json::to_string_pretty(&on_disk).map_err(|e| format!("serialize: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        replace_file_atomic(&tmp, &path)?;
        Ok(())
    }

    pub fn find_server(&self, id: &str) -> Option<&ServerDef> {
        self.servers.iter().find(|s| s.id == id)
    }
}

// ---------------------------------------------------------------------------
// Portable Server Recipe & Full Config Export
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerRecipe {
    #[serde(default = "default_recipe_schema")]
    pub schema: String,
    pub model_id: String,
    pub task: String,
    pub port: u16,
    pub gpu_mem_util: f64,
    pub quant: String,
    #[serde(default)]
    pub max_model_len: Option<usize>,
    #[serde(default)]
    pub served_model_name: Option<String>,
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    #[serde(default)]
    pub enforce_eager: bool,
    #[serde(default)]
    pub swap_space_gb: Option<usize>,
    #[serde(default)]
    pub cpu_offload_gb: Option<usize>,
}

fn default_recipe_schema() -> String {
    "local-llm-panel/server-recipe/v1".to_string()
}

impl ServerRecipe {
    pub fn from_server_def(def: &ServerDef) -> Self {
        Self {
            schema: default_recipe_schema(),
            model_id: def.model_id.clone(),
            task: def.task.clone(),
            port: def.port,
            gpu_mem_util: def.gpu_mem_util,
            quant: def.quant.clone(),
            max_model_len: def.max_model_len,
            served_model_name: def.served_model_name.clone(),
            kv_cache_dtype: def.kv_cache_dtype.clone(),
            enforce_eager: def.enforce_eager,
            swap_space_gb: def.swap_space_gb,
            cpu_offload_gb: def.cpu_offload_gb,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigExportPackage {
    #[serde(default = "default_config_export_schema")]
    pub schema: String,
    pub exported_at: String,
    pub distro: String,
    pub llm_dir: String,
    pub venv_dir: String,
    pub default_quant: String,
    pub servers: Vec<ServerDef>,
    pub memory_settings: MemorySettings,
    pub advanced_settings: AdvancedSettings,
    pub minimize_to_tray: bool,
    pub auto_restart_crashed: bool,
    pub launch_at_login: bool,
    /// True when the package was produced by the redacted exporter.  Import
    /// uses this marker to preserve local secrets rather than replacing them
    /// with placeholder values.
    #[serde(default)]
    pub secrets_omitted: bool,
}

fn default_config_export_schema() -> String {
    "local-llm-panel/config-export/v1".to_string()
}

impl ConfigExportPackage {
    pub fn from_persisted(cfg: &PersistedConfig) -> Self {
        Self {
            schema: default_config_export_schema(),
            exported_at: chrono_or_simple_timestamp(),
            distro: cfg.distro.clone(),
            llm_dir: cfg.llm_dir.clone(),
            venv_dir: cfg.venv_dir.clone(),
            default_quant: cfg.default_quant.clone(),
            servers: cfg.servers.iter().map(redact_server_def).collect(),
            memory_settings: cfg.memory_settings.clone(),
            advanced_settings: redact_advanced_settings(&cfg.advanced_settings),
            minimize_to_tray: cfg.minimize_to_tray,
            auto_restart_crashed: cfg.auto_restart_crashed,
            launch_at_login: cfg.launch_at_login,
            secrets_omitted: true,
        }
    }
}

fn chrono_or_simple_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

// ---------------------------------------------------------------------------
// Live server registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Error,
}

impl ServerStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ServerStatus::Stopped => "stopped",
            ServerStatus::Starting => "starting",
            ServerStatus::Running => "running",
            ServerStatus::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub running: u64,
    pub waiting: u64,
    pub total_prompt_tokens: u64,
    pub total_generation_tokens: u64,
    pub requests: u64,
    pub measured: Option<MeasuredStats>,
}

/// Live (in-memory, not persisted) state of a running server.
pub struct LiveServer {
    pub def: ServerDef,
    pub status: ServerStatus,
    pub error: Option<String>,
    pub wsl_child: Option<crate::wsl::WslChild>,
    pub native_child: Option<crate::wsl::NativeChild>,
    /// WSL-side PID from the pidfile.
    pub wsl_pid: Option<u32>,
    pub log_ring: Mutex<VecDequeLog>,
    pub last_metrics: Option<MetricsSnapshot>,
    /// Set when we called stop() ourselves (so the monitor doesn't flag error).
    pub stopping: bool,
    pub crash_retry_count: u32,
}

pub struct VecDequeLog {
    pub lines: std::collections::VecDeque<String>,
    pub total: usize,
}

impl VecDequeLog {
    pub fn new() -> Self {
        VecDequeLog {
            lines: std::collections::VecDeque::new(),
            total: 0,
        }
    }
    pub fn push(&mut self, line: String) {
        if self.lines.len() >= 2000 {
            self.lines.pop_front();
        }
        self.total += 1;
        self.lines.push_back(line);
    }
    pub fn tail(&self, n: usize) -> String {
        let start = self.lines.len().saturating_sub(n);
        self.lines
            .iter()
            .skip(start)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
    pub fn since_line(&self, line: usize) -> String {
        let idx = line.saturating_sub(self.total.saturating_sub(self.lines.len()));
        let mut out = String::new();
        for (i, l) in self.lines.iter().enumerate() {
            if (self.total - self.lines.len()) + i >= idx {
                out.push_str(l);
                out.push('\n');
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub config: Mutex<PersistedConfig>,
    pub servers: Mutex<BTreeMap<String, LiveServer>>,
    /// IDs currently being spawned.  This closes the check-then-spawn race
    /// between two start requests for the same server.
    pub starting: Mutex<HashSet<String>>,
    pub http: reqwest::Client,
    /// In-flight model pulls: model_id → running flag.
    pub pulling: Arc<Mutex<HashMap<String, bool>>>,
    /// Latest GPU snapshot (polled by the monitor task, read by dashboard).
    pub gpu: Mutex<Option<GpuSnapshot>>,
    pub enrichment_cache: Mutex<HashMap<String, CachedEnrichment>>,
    pub quant_cache: Mutex<HashMap<String, CachedQuants>>,
    pub rec_cache: Mutex<Option<(Vec<crate::commands::ModelWithFit>, Instant, u64)>>,
    pub conversations: Mutex<Vec<Conversation>>,
    pub chat_cancels: Mutex<HashMap<String, Arc<tokio::sync::Notify>>>,
    pub benchmarks: Mutex<Vec<BenchmarkRun>>,
    pub benchmark_cancels: Mutex<HashMap<String, Arc<tokio::sync::Notify>>>,
    pub system_metrics: Mutex<VecDeque<SystemMetricPoint>>,
    pub server_metrics: Mutex<HashMap<String, VecDeque<ServerMetricPoint>>>,
}

pub const METRICS_SERIES_CAPACITY: usize = 60;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMetricPoint {
    pub timestamp: u64,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub gpu_util_pct: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerMetricPoint {
    pub timestamp: u64,
    pub tok_s: f64,
    pub prompt_tok_s: f64,
    pub requests_running: u64,
    pub requests_waiting: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuSnapshot {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub util_percent: u32,
}

impl AppState {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .user_agent("local-llm-panel/0.1")
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        let mut config = PersistedConfig::load();
        // Only resolve an empty distro at startup; never boot WSL just to validate it.
        if config.distro.is_empty() && !config.distro.starts_with("__test_") {
            if let Some(detected) = crate::wsl::detect_default_distro() {
                config.distro = detected;
                let _ = config.save();
            }
        }
        AppState {
            config: Mutex::new(config),
            servers: Mutex::new(BTreeMap::new()),
            starting: Mutex::new(HashSet::new()),
            http,
            pulling: Arc::new(Mutex::new(HashMap::new())),
            gpu: Mutex::new(None),
            enrichment_cache: Mutex::new(HashMap::new()),
            quant_cache: Mutex::new(HashMap::new()),
            rec_cache: Mutex::new(None),
            conversations: Mutex::new(Conversation::load_all()),
            chat_cancels: Mutex::new(HashMap::new()),
            benchmarks: Mutex::new(BenchmarkRun::load_all()),
            benchmark_cancels: Mutex::new(HashMap::new()),
            system_metrics: Mutex::new(VecDeque::new()),
            server_metrics: Mutex::new(HashMap::new()),
        }
    }

    pub fn record_system_metric(&self, snap: &GpuSnapshot) {
        let vram_used_mb = snap.vram_total_mb.saturating_sub(snap.vram_free_mb);
        let pt = SystemMetricPoint {
            timestamp: now_ms(),
            vram_used_mb,
            vram_total_mb: snap.vram_total_mb,
            vram_free_mb: snap.vram_free_mb,
            gpu_util_pct: snap.util_percent,
        };
        let mut sm = self.system_metrics.lock().unwrap();
        if sm.len() >= METRICS_SERIES_CAPACITY {
            sm.pop_front();
        }
        sm.push_back(pt);
    }

    pub fn record_server_metric(&self, server_id: &str, pt: ServerMetricPoint) {
        let mut sm = self.server_metrics.lock().unwrap();
        let q = sm
            .entry(server_id.to_string())
            .or_insert_with(VecDeque::new);
        if q.len() >= METRICS_SERIES_CAPACITY {
            q.pop_front();
        }
        q.push_back(pt);
    }

    pub fn config(&self) -> PersistedConfig {
        self.config.lock().unwrap().clone()
    }

    pub fn resolve_distro(&self) -> String {
        let current = self.config.lock().unwrap().distro.clone();
        if current.starts_with("__test_") {
            return current;
        }
        // If configured distro exists and is responsive, use it
        if !current.is_empty() && crate::wsl::run_script(&current, "echo ok").ok {
            return current;
        }
        // Otherwise detect and save the best available (prefers dedicated distro)
        if let Some(detected) = crate::wsl::detect_default_distro() {
            let mut cfg = self.config.lock().unwrap();
            cfg.distro = detected.clone();
            let _ = cfg.save();
            return detected;
        }
        current
    }

    pub fn hf_token(&self) -> Option<String> {
        let cfg = self.config();
        let t = cfg.hf_token.trim().to_string();
        if !t.is_empty() {
            Some(t)
        } else {
            std::env::var("HF_TOKEN")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_replace_replaces_existing_file() {
        let dir = std::env::temp_dir().join(format!(
            "llm-panel-config-replace-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("config.json");
        let temporary = dir.join("config.json.tmp");
        std::fs::write(&destination, b"old").unwrap();
        std::fs::write(&temporary, b"new").unwrap();
        replace_file_atomic(&temporary, &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"new");
        assert!(!temporary.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_round_trip() {
        let mut cfg = PersistedConfig::default();
        cfg.distro = "Ubuntu-22.04".to_string();
        cfg.hf_token = "hf_secret".to_string();
        cfg.servers.push(ServerDef {
            backend: "vllm".into(),
            id: "srv-1".into(),
            name: "coder".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            task: "instruct".into(),
            port: 8000,
            gpu_mem_util: 0.92,
            max_model_len: Some(4096),
            quant: "fp16".into(),
            served_model_name: None,
            kv_cache_dtype: None,
            llamacpp_channel: LlamaCppChannel::Upstream,
            enforce_eager: true,
            params_b: None,
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: false,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
        });
        cfg.measured.insert(
            "Qwen/Qwen2.5-0.5B-Instruct".into(),
            MeasuredStats {
                tokens_per_sec: Some(123.4),
                ..Default::default()
            },
        );
        let text = serde_json::to_string_pretty(&cfg).unwrap();
        let back: PersistedConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.distro, "Ubuntu-22.04");
        assert_eq!(back.servers.len(), 1);
        assert_eq!(back.servers[0].port, 8000);
        assert_eq!(
            back.measured["Qwen/Qwen2.5-0.5B-Instruct"].tokens_per_sec,
            Some(123.4)
        );
    }

    #[test]
    fn old_config_deserializes_with_llamacpp_defaults() {
        let old = r#"{
            "distro": "Ubuntu-22.04",
            "llm_dir": "~/llm-lp",
            "venv_dir": "~/llm-lp/.venv",
            "hf_token": "",
            "default_quant": "fp16",
            "servers": [{
                "id": "old",
                "name": "old server",
                "model_id": "org/model",
                "task": "instruct",
                "port": 8000,
                "gpu_mem_util": 0.92,
                "max_model_len": 4096,
                "quant": "fp16",
                "served_model_name": null,
                "params_b": null,
                "n_gpu_layers": 99
            }],
            "measured": {}
        }"#;
        let cfg: PersistedConfig = serde_json::from_str(old).unwrap();
        assert_eq!(cfg.servers[0].backend, "vllm");
        assert_eq!(cfg.servers[0].n_gpu_layers, Some(99));
        assert_eq!(cfg.servers[0].cache_type_k, "q8_0");
        assert!(cfg.servers[0].kv_cache_dtype.is_none());
        assert!(cfg.servers[0].jinja);
        assert!(!cfg.llamacpp_dir.is_empty());
        assert!(!cfg.gguf_dir.is_empty());
    }

    #[test]
    fn missing_gpu_layer_setting_defaults_to_auto_fit() {
        let cfg: PersistedConfig = serde_json::from_str(
            r#"{"servers":[{"id":"s","name":"s","model_id":"m","task":"instruct","port":8000,"gpu_mem_util":0.8,"quant":"fp16"}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.servers[0].n_gpu_layers, None);
        assert!(cfg.servers[0].fit);
    }

    #[test]
    fn backend_normalization_removes_opposite_settings() {
        let populated = r#"{
            "backend":"vllm",
            "id":"mixed",
            "name":"mixed",
            "model_id":"Qwen/test",
            "task":"instruct",
            "port":8000,
            "gpu_mem_util":0.85,
            "max_model_len":32768,
            "quant":"gptq",
            "served_model_name":"alias",
            "kv_cache_dtype":"fp8_e4m3",
            "swap_space_gb":8,
            "cpu_offload_gb":4,
            "model_path":"C:\\models\\model.gguf",
            "mmproj_path":"C:\\models\\mmproj.gguf",
            "ctx_size":4096,
            "n_gpu_layers":99,
            "n_cpu_moe":24,
            "fit":true,
            "fit_target":1024,
            "device":"CUDA0",
            "api_key":"server-key",
            "log_verbosity":4,
            "flash_attn":true,
            "cache_type_k":"q8_0",
            "cache_type_v":"q8_0",
            "threads":12,
            "batch_size":512,
            "ubatch_size":128,
            "parallel":2,
            "jinja":true,
            "no_kv_offload":true,
            "metrics":true,
            "extra_args":["--no-mmap"],
            "env":{"VLLM_LOGGING_LEVEL":"DEBUG"}
        }"#;

        let mut vllm: ServerDef = serde_json::from_str(populated).unwrap();
        vllm.normalize_for_backend().unwrap();
        assert_eq!(vllm.backend_kind().unwrap(), ServerBackend::Vllm);
        assert!(vllm.model_path.is_none());
        assert!(vllm.ctx_size.is_none());
        assert!(vllm.n_gpu_layers.is_none());
        assert!(vllm.cache_type_k.is_empty());
        assert!(vllm.threads.is_none());
        assert_eq!(vllm.parallel, 0);
        assert!(vllm.extra_args.is_empty());
        assert!(!vllm.fit);
        assert!(!vllm.flash_attn);
        assert!(!vllm.jinja);
        assert_eq!(vllm.effective_model_name(), "alias");
        assert_eq!(vllm.api_key.as_deref(), Some("server-key"));
        assert_eq!(vllm.kv_cache_dtype.as_deref(), Some("fp8_e4m3"));

        let mut native: ServerDef = serde_json::from_str(populated).unwrap();
        native.backend = "llamacpp".into();
        native.normalize_for_backend().unwrap();
        assert_eq!(native.backend_kind().unwrap(), ServerBackend::Llamacpp);
        assert!(native.max_model_len.is_none());
        assert!(native.swap_space_gb.is_none());
        assert!(native.cpu_offload_gb.is_none());
        assert!(native.env.is_empty());
        assert_eq!(native.gpu_mem_util, 0.0);
        assert!(!native.enforce_eager);
        assert_eq!(native.task, "instruct");
        assert_eq!(native.effective_model_name(), "alias");
        assert_eq!(native.api_key.as_deref(), Some("server-key"));
        assert!(native.kv_cache_dtype.is_none());
    }

    #[test]
    fn backend_normalization_rejects_unknown_backend() {
        let mut server: ServerDef = serde_json::from_str(
            r#"{"backend":"llamacpp ","id":"s","name":"s","model_id":"m","task":"instruct","port":8000,"gpu_mem_util":0.85,"quant":"GGUF"}"#,
        )
        .unwrap();
        assert!(server.normalize_for_backend().is_err());
    }

    #[test]
    fn vecdeque_log_caps_and_offsets() {
        let mut log = VecDequeLog::new();
        for i in 0..2050 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.lines.len(), 2000);
        let tail = log.tail(5);
        assert!(tail.contains("line 2049"));
        assert!(!tail.contains("line 0"));
        // since_line with an old offset returns everything after it
        let since = log.since_line(2050 - 3);
        assert!(since.contains("line 2047"));
        assert!(since.contains("line 2049"));
    }

    #[test]
    fn test_memory_settings_default_and_roundtrip() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.memory_settings.default_gpu_mem_util, 0.85);
        assert_eq!(cfg.memory_settings.vram_overhead_mb, 2500.0);
        assert!(cfg.memory_settings.enable_ram_overflow);
        assert_eq!(cfg.memory_settings.safety_reserve_mb, 4096);
        assert!(cfg.memory_settings.offload_weights_allowed);
        assert_eq!(cfg.memory_settings.manual_ram_limit_mb, None);
        assert_eq!(cfg.memory_settings.max_context_cap, None);

        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.memory_settings.default_gpu_mem_util, 0.85);
        assert_eq!(parsed.memory_settings, cfg.memory_settings);

        // Verify partial/missing JSON defaults
        let empty_ms: MemorySettings = serde_json::from_str("{}").unwrap();
        assert_eq!(empty_ms, MemorySettings::default());

        let legacy_config: AppConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy_config.memory_settings, MemorySettings::default());
    }

    #[test]
    fn test_serverdef_swap_offload_roundtrip() {
        let json = r#"{
            "id": "s1",
            "name": "test",
            "model_id": "test/model",
            "task": "instruct",
            "port": 8000,
            "gpu_mem_util": 0.9,
            "max_model_len": 2048,
            "quant": "fp16",
            "served_model_name": null,
            "enforce_eager": true,
            "params_b": 1.0,
            "swap_space_gb": 16,
            "cpu_offload_gb": 8
        }"#;
        let s: ServerDef = serde_json::from_str(json).unwrap();
        assert_eq!(s.swap_space_gb, Some(16));
        assert_eq!(s.cpu_offload_gb, Some(8));

        let mut server = s.clone();
        server.swap_space_gb = Some(8);
        server.cpu_offload_gb = Some(4);
        let serialized = serde_json::to_string(&server).unwrap();
        let deserialized: ServerDef = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.swap_space_gb, Some(8));
        assert_eq!(deserialized.cpu_offload_gb, Some(4));
        assert_eq!(deserialized, server);

        let legacy_json = r#"{
            "id": "s1",
            "name": "test",
            "model_id": "test/model",
            "task": "instruct",
            "port": 8000,
            "gpu_mem_util": 0.9,
            "max_model_len": 2048,
            "quant": "fp16",
            "served_model_name": null,
            "enforce_eager": true,
            "params_b": 1.0
        }"#;
        let s_legacy: ServerDef = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(s_legacy.swap_space_gb, None);
        assert_eq!(s_legacy.cpu_offload_gb, None);
    }

    #[test]
    fn test_advanced_settings_defaults_and_roundtrip() {
        let adv = AdvancedSettings::default();
        assert_eq!(adv.host, "127.0.0.1");
        assert_eq!(adv.kv_cache_dtype, "auto");
        assert!(adv.enable_prefix_caching);
        assert!(!adv.enable_chunked_prefill);
        assert_eq!(adv.log_level, "INFO");

        let json = serde_json::to_string(&adv).unwrap();
        let deserialized: AdvancedSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, adv);

        // Test empty json defaults
        let empty_json = "{}";
        let from_empty: AdvancedSettings = serde_json::from_str(empty_json).unwrap();
        assert_eq!(from_empty, adv);
    }

    #[test]
    fn test_conversation_serialization_and_roundtrip() {
        use super::{ChatMessage, Conversation};
        let c = Conversation {
            id: "conv-1".to_string(),
            server_id: "srv-1".to_string(),
            title: "Test Conversation".to_string(),
            created_at: 1000,
            updated_at: 2000,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are an assistant.".to_string(),
                    images: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello!".to_string(),
                    images: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hi there!".to_string(),
                    images: None,
                },
            ],
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
        assert_eq!(parsed.messages.len(), 3);
    }

    #[test]
    fn test_multimodal_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Describe this image".to_string(),
            images: Some(vec![
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    .to_string(),
            ]),
        };
        let val = serde_json::to_value(&msg).unwrap();
        assert!(val.get("images").is_some());

        let body = msg.payload_body();
        assert_eq!(body["role"], "user");
        assert!(body["content"].is_array());
        assert_eq!(body["content"][0]["type"], "text");
        assert_eq!(body["content"][1]["type"], "image_url");
        assert!(body["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_benchmark_run_serialization_and_roundtrip() {
        use super::BenchmarkRun;
        let b = BenchmarkRun {
            id: "bm-1".to_string(),
            server_id: "srv-1".to_string(),
            model_id: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            quant: Some("AWQ".to_string()),
            timestamp: 123456789,
            prompt_tok_s: 450.5,
            gen_tok_s: 85.2,
            latency_ms: 120.0,
            prompt_count: 3,
        };
        let json = serde_json::to_string(&b).unwrap();
        let parsed: BenchmarkRun = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, b);
        assert_eq!(parsed.gen_tok_s, 85.2);
    }

    #[test]
    fn test_system_and_server_metric_series_ring_buffer() {
        use super::{AppState, GpuSnapshot, ServerMetricPoint, METRICS_SERIES_CAPACITY};
        let app_state = AppState::new();

        // Test system metric ring buffer
        for i in 0..(METRICS_SERIES_CAPACITY + 10) {
            let snap = GpuSnapshot {
                name: "RTX 4090".to_string(),
                vram_total_mb: 24576,
                vram_free_mb: 24576 - (i as u64 * 100),
                util_percent: (i % 100) as u32,
            };
            app_state.record_system_metric(&snap);
        }

        let sm = app_state.system_metrics.lock().unwrap();
        assert_eq!(sm.len(), METRICS_SERIES_CAPACITY);
        assert_eq!(sm.back().unwrap().vram_total_mb, 24576);

        // Test server metric ring buffer
        for i in 0..(METRICS_SERIES_CAPACITY + 15) {
            let pt = ServerMetricPoint {
                timestamp: 1000 + i as u64,
                tok_s: 40.0 + (i as f64),
                prompt_tok_s: 150.0,
                requests_running: 1,
                requests_waiting: 0,
            };
            app_state.record_server_metric("test-server", pt);
        }

        let srv_m = app_state.server_metrics.lock().unwrap();
        let q = srv_m.get("test-server").unwrap();
        assert_eq!(q.len(), METRICS_SERIES_CAPACITY);
        assert_eq!(
            q.back().unwrap().tok_s,
            40.0 + ((METRICS_SERIES_CAPACITY + 14) as f64)
        );
    }

    #[test]
    fn test_appliance_settings_and_was_running_roundtrip() {
        use super::{PersistedConfig, ServerDef};
        let mut cfg = PersistedConfig::default();
        assert!(cfg.minimize_to_tray);
        assert!(cfg.auto_restart_crashed);
        assert!(!cfg.launch_at_login);

        cfg.minimize_to_tray = false;
        cfg.launch_at_login = true;
        cfg.servers.push(ServerDef {
            backend: "vllm".into(),
            id: "s1".into(),
            name: "test".into(),
            model_id: "m1".into(),
            task: "instruct".into(),
            port: 8000,
            gpu_mem_util: 0.9,
            max_model_len: None,
            quant: "fp16".into(),
            served_model_name: None,
            kv_cache_dtype: None,
            llamacpp_channel: LlamaCppChannel::Upstream,
            enforce_eager: true,
            params_b: None,
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: true,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
        });

        let text = serde_json::to_string(&cfg).unwrap();
        let parsed: PersistedConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.minimize_to_tray, false);
        assert_eq!(parsed.launch_at_login, true);
        assert_eq!(parsed.servers[0].was_running, true);
    }

    #[test]
    fn test_server_recipe_and_config_export_roundtrip() {
        use super::{ConfigExportPackage, ServerDef, ServerRecipe};
        let def = ServerDef {
            backend: "vllm".into(),
            id: "recipe-test".into(),
            name: "Qwen 7B".into(),
            model_id: "Qwen/Qwen2.5-7B-Instruct".into(),
            task: "instruct".into(),
            port: 8088,
            gpu_mem_util: 0.92,
            max_model_len: Some(8192),
            quant: "fp8".into(),
            served_model_name: Some("qwen-7b".into()),
            kv_cache_dtype: Some("fp8_e4m3".into()),
            llamacpp_channel: LlamaCppChannel::Upstream,
            enforce_eager: false,
            params_b: Some(7.6),
            swap_space_gb: Some(4),
            cpu_offload_gb: Some(2),
            was_running: false,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
        };

        let recipe = ServerRecipe::from_server_def(&def);
        assert_eq!(recipe.model_id, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(recipe.port, 8088);
        assert_eq!(recipe.swap_space_gb, Some(4));
        assert_eq!(recipe.kv_cache_dtype.as_deref(), Some("fp8_e4m3"));

        let text = serde_json::to_string_pretty(&recipe).unwrap();
        let parsed_recipe: ServerRecipe = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed_recipe, recipe);

        let mut cfg = super::PersistedConfig::default();
        cfg.servers.push(def);
        let export_pkg = ConfigExportPackage::from_persisted(&cfg);
        let export_json = serde_json::to_string_pretty(&export_pkg).unwrap();
        let parsed_pkg: ConfigExportPackage = serde_json::from_str(&export_json).unwrap();
        assert_eq!(parsed_pkg.servers.len(), 1);
        assert_eq!(parsed_pkg.servers[0].id, "recipe-test");

        let mut secret_cfg = super::PersistedConfig::default();
        secret_cfg.hf_token = "hf_export_secret".into();
        secret_cfg.github_token = "gh_export_secret".into();
        secret_cfg.advanced_settings.api_key = Some("gateway_export_secret".into());
        let mut secret_server = cfg.servers[0].clone();
        secret_server.api_key = Some("server_export_secret".into());
        secret_server
            .env
            .insert("CUSTOM_TOKEN".into(), "env_export_secret".into());
        secret_cfg.servers = vec![secret_server];
        let secret_json = serde_json::to_string(&ConfigExportPackage::from_persisted(&secret_cfg))
            .unwrap();
        for secret in [
            "hf_export_secret",
            "gh_export_secret",
            "gateway_export_secret",
            "server_export_secret",
            "env_export_secret",
        ] {
            assert!(!secret_json.contains(secret), "export leaked {secret}");
        }
        assert!(secret_json.contains(SECRET_PLACEHOLDER));
        let public_json = serde_json::to_string(&redact_config_for_display(&secret_cfg)).unwrap();
        assert!(!public_json.contains("hf_export_secret"));
        assert!(!public_json.contains("gh_export_secret"));
        assert!(!public_json.contains("gateway_export_secret"));
        assert!(!public_json.contains("server_export_secret"));
        assert!(!public_json.contains("env_export_secret"));

        let mut redacted = secret_cfg.servers[0].clone();
        let restored = secret_cfg.servers[0].clone();
        redacted = redact_server_def(&redacted);
        merge_server_secret_placeholders(&mut redacted, Some(&restored));
        assert_eq!(redacted.api_key, restored.api_key);
        assert_eq!(redacted.env.get("CUSTOM_TOKEN"), restored.env.get("CUSTOM_TOKEN"));
    }

    #[test]
    fn test_dpapi_persisted_config_disk_simulation() {
        use super::PersistedConfig;
        let mut cfg = PersistedConfig::default();
        cfg.hf_token = "hf_super_secret_test_token_9988".to_string();

        let tmp_dir = std::env::current_dir()
            .unwrap()
            .join(".localllm_dpapi_test");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let tmp_file = tmp_dir.join("test_config.json");

        // Manually do what save() does
        let mut on_disk = cfg.clone();
        if !on_disk.hf_token.is_empty() && !on_disk.hf_token.starts_with("dpapi:") {
            if let Ok(enc) = crate::security::encrypt_token(&on_disk.hf_token) {
                on_disk.hf_token = enc;
            }
        }
        let serialized = serde_json::to_string_pretty(&on_disk).unwrap();
        // File contents MUST NOT contain the plain token!
        assert!(!serialized.contains("hf_super_secret_test_token_9988"));
        assert!(serialized.contains("dpapi:"));
        std::fs::write(&tmp_file, &serialized).unwrap();

        // Manually do what load() does
        let read_back = std::fs::read_to_string(&tmp_file).unwrap();
        let mut loaded: PersistedConfig = serde_json::from_str(&read_back).unwrap();
        if loaded.hf_token.starts_with("dpapi:") {
            if let Ok(plain) = crate::security::decrypt_token(&loaded.hf_token) {
                loaded.hf_token = plain;
            }
        }

        assert_eq!(loaded.hf_token, "hf_super_secret_test_token_9988");
        let _ = std::fs::remove_file(&tmp_file);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_serverdef_env_field_roundtrip() {
        use super::ServerDef;
        let json = r#"{
            "id": "s1",
            "name": "test",
            "model_id": "test/model",
            "task": "instruct",
            "port": 8000,
            "gpu_mem_util": 0.9,
            "max_model_len": 2048,
            "quant": "fp16",
            "served_model_name": null,
            "enforce_eager": true,
            "params_b": 1.0,
            "env": {"CUSTOM_VAR": "custom_value", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        }"#;
        let s: ServerDef = serde_json::from_str(json).unwrap();
        assert_eq!(s.env.get("CUSTOM_VAR"), Some(&"custom_value".to_string()));
        assert_eq!(s.env.get("VLLM_USE_FLASHINFER_SAMPLER"), Some(&"0".to_string()));

        let serialized = serde_json::to_string(&s).unwrap();
        let deserialized: ServerDef = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.env, s.env);
    }

    #[test]
    fn test_persistedconfig_default_env() {
        use super::PersistedConfig;
        let cfg = PersistedConfig::default();
        assert_eq!(cfg.default_env.get("VLLM_USE_FLASHINFER_SAMPLER"), Some(&"0".to_string()));
    }

    #[test]
    fn test_old_config_loads_with_default_env() {
        use super::PersistedConfig;
        let old = r#"{
            "distro": "Ubuntu-22.04",
            "llm_dir": "~/llm-lp",
            "venv_dir": "~/llm-lp/.venv",
            "hf_token": "",
            "default_quant": "fp16",
            "servers": [{
                "id": "old",
                "name": "old server",
                "model_id": "org/model",
                "task": "instruct",
                "port": 8000,
                "gpu_mem_util": 0.92,
                "max_model_len": 4096,
                "quant": "fp16",
                "served_model_name": null,
                "params_b": null,
                "n_gpu_layers": 99
            }],
            "measured": {}
        }"#;
        let cfg: PersistedConfig = serde_json::from_str(old).unwrap();
        // Raw deserialization of old config has empty default_env.
        // The load() method adds the default VLLM_USE_FLASHINFER_SAMPLER=0.
        assert!(cfg.default_env.is_empty());
    }
}
