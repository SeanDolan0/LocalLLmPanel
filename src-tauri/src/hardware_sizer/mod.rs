pub mod sizing;
pub mod traits;
pub mod types;

#[cfg(test)]
pub mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::hardware_sizer::sizing::{calculate_allocation, calculate_fit};
use crate::hardware_sizer::traits::HardwareProbe;
pub use crate::hardware_sizer::types::*;
use crate::state::{AppState, MemorySettings, ServerDef};

/// Production probe accessing OS hardware via `llmfit_core` and `wsl`.
pub struct SystemHardwareProbe;

impl HardwareProbe for SystemHardwareProbe {
    fn probe_topology(&self) -> StaticHardwareTopology {
        let specs = llmfit_core::hardware::SystemSpecs::detect();
        let host_ram_mb = (specs.total_ram_gb * 1024.0).round() as u64;

        // Check Windows GPU first, then specs
        let (gpu_name, vram_total_mb) = if let Some(gpu) = crate::llamacpp_install::windows_gpu_snapshot() {
            (Some(gpu.name), gpu.vram_total_mb)
        } else if let Some(ref name) = specs.gpu_name {
            let mb = (specs.total_gpu_vram_gb.unwrap_or(specs.gpu_vram_gb.unwrap_or(0.0)) * 1024.0).round() as u64;
            (Some(name.clone()), mb)
        } else {
            (None, 0)
        };

        // Probe WSL RAM capacity if available
        let wsl_ram_mb = crate::wsl::detect_wsl_memory("Ubuntu")
            .or_else(|| crate::wsl::detect_wsl_memory("Debian"))
            .map(|(total, _)| total);

        StaticHardwareTopology {
            cpu_name: specs.cpu_name,
            cpu_cores: specs.total_cpu_cores,
            host_ram_total_mb: host_ram_mb,
            gpu_name,
            vram_total_mb,
            is_unified_memory: specs.unified_memory,
            wsl_ram_total_mb: wsl_ram_mb,
        }
    }

    fn probe_telemetry(&self) -> DynamicTelemetry {
        // Query current available host memory via llmfit-core
        let specs = llmfit_core::hardware::SystemSpecs::detect();
        let host_avail_mb = (specs.available_ram_gb * 1024.0).round() as u64;

        // Query GPU telemetry (try windows native nvidia-smi first, then WSL)
        let (vram_free_mb, gpu_util) = if let Some(gpu) = crate::llamacpp_install::windows_gpu_snapshot() {
            (gpu.vram_free_mb, Some(gpu.util_percent as f32))
        } else if let Some(gpu) = crate::wsl::gpu_snapshot("Ubuntu") {
            (gpu.vram_free_mb, Some(gpu.util_percent as f32))
        } else {
            (0, None)
        };

        // Query WSL memory
        let wsl_avail_mb = crate::wsl::detect_wsl_memory("Ubuntu")
            .or_else(|| crate::wsl::detect_wsl_memory("Debian"))
            .map(|(_, avail)| avail);

        DynamicTelemetry {
            host_ram_available_mb: host_avail_mb,
            vram_free_mb,
            wsl_ram_available_mb: wsl_avail_mb,
            gpu_utilization_pct: gpu_util,
        }
    }
}

/// The deep module for hardware detection, dynamic telemetry, VRAM reservations,
/// and model memory sizing.
pub struct HardwareSizer {
    probe: Arc<dyn HardwareProbe>,
    cached_topology: OnceLock<StaticHardwareTopology>,
    reservations: Mutex<HashMap<String, u64>>,
}

impl HardwareSizer {
    /// Create a new HardwareSizer with custom probe.
    pub fn new(probe: Arc<dyn HardwareProbe>) -> Self {
        Self {
            probe,
            cached_topology: OnceLock::new(),
            reservations: Mutex::new(HashMap::new()),
        }
    }

    /// Create with the default system probe.
    pub fn default_system() -> Self {
        Self::new(Arc::new(SystemHardwareProbe))
    }

    /// Get immutable physical hardware capacity (cached once on startup).
    pub fn topology(&self) -> &StaticHardwareTopology {
        self.cached_topology.get_or_init(|| self.probe.probe_topology())
    }

    /// Generate an authoritative point-in-time snapshot reconciling static topology,
    /// live dynamic telemetry, and active VRAM reservations.
    pub fn snapshot(&self) -> HardwareSnapshot {
        let topo = self.topology();
        let telemetry = self.probe.probe_telemetry();
        let reserved_mb = self.total_reserved_vram_mb();
        let effective_free_mb = telemetry.vram_free_mb.saturating_sub(reserved_mb);

        HardwareSnapshot {
            cpu_name: topo.cpu_name.clone(),
            cpu_cores: topo.cpu_cores,
            host_ram_total_mb: topo.host_ram_total_mb,
            host_ram_available_mb: telemetry.host_ram_available_mb,
            wsl_ram_total_mb: topo.wsl_ram_total_mb,
            wsl_ram_available_mb: telemetry.wsl_ram_available_mb,
            gpu_name: topo.gpu_name.clone(),
            vram_total_mb: topo.vram_total_mb,
            vram_free_mb: telemetry.vram_free_mb,
            vram_reserved_mb: reserved_mb,
            vram_effective_free_mb: effective_free_mb,
            is_unified_memory: topo.is_unified_memory,
        }
    }

    /// Track an exclusive in-flight VRAM reservation during server startup.
    pub fn reserve_vram(&self, server_id: &str, amount_mb: u64) {
        self.reservations
            .lock()
            .unwrap()
            .insert(server_id.to_string(), amount_mb);
    }

    /// Release a VRAM reservation once the process is up or startup fails.
    pub fn release_vram(&self, server_id: &str) {
        self.reservations.lock().unwrap().remove(server_id);
    }

    /// Sum of all active in-flight VRAM reservations.
    pub fn total_reserved_vram_mb(&self) -> u64 {
        self.reservations.lock().unwrap().values().sum()
    }

    /// Check reservation for a specific server.
    pub fn reserved_vram_mb(&self, server_id: &str) -> Option<u64> {
        self.reservations.lock().unwrap().get(server_id).copied()
    }

    /// Evaluate model memory fit against the target execution environment.
    pub fn evaluate_fit(
        &self,
        geom: &ModelGeometry,
        backend: TargetBackend,
        settings: &MemorySettings,
    ) -> FitReport {
        let hw = self.snapshot();
        calculate_fit(geom, backend, &hw, settings)
    }

    /// Compute allocation report across running servers.
    pub fn allocation_report(
        &self,
        running_defs: &[&ServerDef],
        settings: &MemorySettings,
    ) -> AllocationReport {
        let hw = self.snapshot();
        calculate_allocation(running_defs, &hw, settings)
    }

    // ------------------------------------------------------------------------
    // Strangler pattern delegators for AppState & commands.rs
    // ------------------------------------------------------------------------

    pub fn running_allocation_report(&self, state: &Arc<AppState>) -> AllocationReport {
        let servers = state.servers.lock().unwrap();
        let running: Vec<&ServerDef> = servers
            .values()
            .filter(|ls| ls.status == crate::state::ServerStatus::Running)
            .map(|ls| &ls.def)
            .collect();

        let mem_settings = state.config.lock().unwrap().memory_settings.clone();
        self.allocation_report(&running, &mem_settings)
    }
}
