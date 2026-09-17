//! Persisted configuration (JSON in %APPDATA%) and live server registry.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const CONFIG_DIR_NAME: &str = "local-llm-panel";
pub const CONFIG_FILE_NAME: &str = "config.json";

#[derive(Debug, Clone)]
pub struct CachedEnrichment {
    pub stats: crate::hf::EnrichedStats,
    pub fetched_at: Instant,
}

// ---------------------------------------------------------------------------
// Server definitions & measured stats (persisted)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerDef {
    pub id: String,
    pub name: String,
    pub model_id: String,
    /// "instruct" | "embed"
    pub task: String,
    pub port: u16,
    pub gpu_mem_util: f64,
    pub max_model_len: Option<usize>,
    /// "fp16" | "fp8" | "awq" | "gptq"
    pub quant: String,
    pub served_model_name: Option<String>,
    /// Skip CUDA-graph capture (`--enforce-eager`). WSL2's full-graph
    /// capture can hang for minutes or stall; eager mode starts reliably.
    /// Default true; throughput is slightly lower but startup is robust.
    #[serde(default = "default_true")]
    pub enforce_eager: bool,
    /// Estimated params (billions) memoized at create time (for VRAM guidance).
    pub params_b: Option<f64>,
}

fn default_true() -> bool {
    true
}

impl ServerDef {
    pub fn effective_model_name(&self) -> String {
        self.served_model_name
            .clone()
            .unwrap_or_else(|| self.model_id.clone())
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
// Persisted config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedConfig {
    pub distro: String,
    pub llm_dir: String,
    pub venv_dir: String,
    pub hf_token: String,
    pub default_quant: String,
    pub servers: Vec<ServerDef>,
    pub measured: HashMap<String, MeasuredStats>,
}

impl Default for PersistedConfig {
    fn default() -> Self {
        PersistedConfig {
            distro: "Ubuntu".to_string(),
            llm_dir: "~/llm-lp".to_string(),
            venv_dir: "~/llm-lp/.venv".to_string(),
            hf_token: String::new(),
            default_quant: "fp16".to_string(),
            servers: Vec::new(),
            measured: HashMap::new(),
        }
    }
}

impl PersistedConfig {
    pub fn path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join(CONFIG_FILE_NAME)
    }

    pub fn load() -> Self {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str(&text) {
                Ok(cfg) => cfg,
                Err(_) => {
                    // Corrupt config: back it up and start fresh.
                    let _ = std::fs::rename(&path, path.with_extension("json.bak"));
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename: {e}"))?;
        Ok(())
    }

    pub fn find_server(&self, id: &str) -> Option<&ServerDef> {
        self.servers.iter().find(|s| s.id == id)
    }
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
    /// WSL-side PID from the pidfile.
    pub wsl_pid: Option<u32>,
    pub log_ring: Mutex<VecDequeLog>,
    pub last_metrics: Option<MetricsSnapshot>,
    /// Set when we called stop() ourselves (so the monitor doesn't flag error).
    pub stopping: bool,
}

pub struct VecDequeLog {
    pub lines: std::collections::VecDeque<String>,
    pub total: usize,
}

impl VecDequeLog {
    pub fn new() -> Self {
        VecDequeLog { lines: std::collections::VecDeque::new(), total: 0 }
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
        self.lines.iter().skip(start).cloned().collect::<Vec<_>>().join("\n")
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
    pub http: reqwest::Client,
    /// In-flight model pulls: model_id → running flag.
    pub pulling: Arc<Mutex<HashMap<String, bool>>>,
    /// Latest GPU snapshot (polled by the monitor task, read by dashboard).
    pub gpu: Mutex<Option<GpuSnapshot>>,
    pub enrichment_cache: Mutex<HashMap<String, CachedEnrichment>>,
    pub rec_cache: Mutex<Option<(Vec<crate::commands::ModelWithFit>, Instant)>>,
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
        AppState {
            config: Mutex::new(PersistedConfig::load()),
            servers: Mutex::new(BTreeMap::new()),
            http,
            pulling: Arc::new(Mutex::new(HashMap::new())),
            gpu: Mutex::new(None),
            enrichment_cache: Mutex::new(HashMap::new()),
            rec_cache: Mutex::new(None),
        }
    }

    pub fn config(&self) -> PersistedConfig {
        self.config.lock().unwrap().clone()
    }

    pub fn save_config(&self) {
        let cfg = self.config();
        if let Err(e) = cfg.save() {
            eprintln!("[state] failed to save config: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let mut cfg = PersistedConfig::default();
        cfg.distro = "Ubuntu-22.04".to_string();
        cfg.hf_token = "hf_secret".to_string();
        cfg.servers.push(ServerDef {
            id: "srv-1".into(),
            name: "coder".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            task: "instruct".into(),
            port: 8000,
            gpu_mem_util: 0.92,
            max_model_len: Some(4096),
            quant: "fp16".into(),
            served_model_name: None,
            enforce_eager: true,
            params_b: None,
        });
        cfg.measured.insert(
            "Qwen/Qwen2.5-0.5B-Instruct".into(),
            MeasuredStats { tokens_per_sec: Some(123.4), ..Default::default() },
        );
        let text = serde_json::to_string_pretty(&cfg).unwrap();
        let back: PersistedConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.distro, "Ubuntu-22.04");
        assert_eq!(back.servers.len(), 1);
        assert_eq!(back.servers[0].port, 8000);
        assert_eq!(back.measured["Qwen/Qwen2.5-0.5B-Instruct"].tokens_per_sec, Some(123.4));
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
}