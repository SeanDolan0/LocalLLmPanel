use crate::server::Metrics;
use crate::state::ServerDef;
use serde::{Deserialize, Serialize};

/// Fine-grained operational phase of a Live Server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", content = "details")]
pub enum LifecyclePhase {
    Stopped,
    AcquiringResources {
        port: u16,
        vram_mb: u64,
    },
    SpawningProcess {
        pid: Option<u32>,
    },
    WarmingUp {
        pid: u32,
        port: u16,
        last_log: Option<String>,
    },
    Healthy {
        pid: u32,
        port: u16,
    },
    Degraded {
        pid: u32,
        reason: String,
    },
    Stopping {
        pid: u32,
    },
    Failed {
        error: String,
    },
}

/// Point-in-time summary of a server's operational health, process liveliness, and runtime metrics.
/// (Corresponds to domain term: Server Snapshot in GLOSSARY.md).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSnapshot {
    pub def: ServerDef,
    pub status: String,
    pub phase: LifecyclePhase,
    pub port: u16,
    pub process_alive: bool,
    pub http_healthy: bool,
    pub error: Option<String>,
    pub metrics: Option<Metrics>,
    pub vram_reserved_mb: Option<u64>,
}

/// Payload for creating or updating a server.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateServerRequest {
    pub name: String,
    pub model_id: String,
    pub backend: Option<String>,
    pub task: Option<String>,
    pub port: Option<u16>,
    pub gpu_mem_util: Option<f64>,
    pub max_model_len: Option<usize>,
    pub quant: Option<String>,
    pub served_model_name: Option<String>,
    pub model_path: Option<String>,
    pub api_key: Option<String>,
    pub extra_args: Option<Vec<String>>,
}

/// Chunk of log lines with cursor metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogChunk {
    pub server_id: String,
    pub lines: Vec<String>,
    pub total_lines: usize,
}
