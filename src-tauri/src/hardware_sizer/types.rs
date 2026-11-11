use serde::{Deserialize, Serialize};

/// Target inference engine backend for memory sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetBackend {
    Vllm,
    LlamaCpp,
    Generic,
}

/// Verdict for model memory fit evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    FullGpu,
    HybridOffload,
    CpuOnly,
    TooTight,
}

/// Extracted architecture geometry for sizing a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelGeometry {
    pub params_b: f64,
    pub quant: String,
    pub context_length: Option<u32>,
    pub num_layers: Option<u32>,
    pub num_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
}

impl ModelGeometry {
    pub fn new(params_b: f64, quant: impl Into<String>) -> Self {
        Self {
            params_b,
            quant: quant.into(),
            context_length: None,
            num_layers: None,
            num_kv_heads: None,
            head_dim: None,
        }
    }
}

/// Static hardware capacity detected during application initialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticHardwareTopology {
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub host_ram_total_mb: u64,
    pub gpu_name: Option<String>,
    pub vram_total_mb: u64,
    pub is_unified_memory: bool,
    pub wsl_ram_total_mb: Option<u64>,
}

/// Dynamic memory and hardware telemetry sampled periodically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicTelemetry {
    pub host_ram_available_mb: u64,
    pub vram_free_mb: u64,
    pub wsl_ram_available_mb: Option<u64>,
    pub gpu_utilization_pct: Option<f32>,
}

/// Consolidated point-in-time snapshot of the execution environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareSnapshot {
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub host_ram_total_mb: u64,
    pub host_ram_available_mb: u64,
    pub wsl_ram_total_mb: Option<u64>,
    pub wsl_ram_available_mb: Option<u64>,
    pub gpu_name: Option<String>,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub vram_reserved_mb: u64,
    pub vram_effective_free_mb: u64,
    pub is_unified_memory: bool,
}

/// Comprehensive report of memory allocation across all configured and live servers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocationReport {
    pub running_servers_count: usize,
    pub total_weight_gb: f64,
    pub main_weight_gb: f64,
    pub spec_weight_gb: f64,
    pub ram_spill_gb: f64,
    pub total_reserved_vram_mb: u64,
    pub vram_pressure_pct: f64,
    pub is_low_vram: bool,
    pub warning_message: Option<String>,
}

/// Sizing recommendation and memory footprint verdict for a model variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitReport {
    pub verdict: FitVerdict,
    pub score: u32,
    pub max_context: u32,
    pub weight_gb: f64,
    pub required_vram_gb: f64,
    pub ram_offload_gb: f64,
    pub est_tokens_per_sec: f32,
    pub target_backend: TargetBackend,
    pub notes: Vec<String>,
}
