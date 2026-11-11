use crate::hardware_sizer::types::{DynamicTelemetry, StaticHardwareTopology};
use crate::state::MemorySettings;

/// Port/Adapter for reading hardware topology and dynamic telemetry.
/// This allows tests to substitute hardware environments deterministically.
pub trait HardwareProbe: Send + Sync {
    /// Probe immutable physical hardware specs (DIMMs, GPU model, total VRAM, CPU cores).
    fn probe_topology(&self) -> StaticHardwareTopology;

    /// Sample transient dynamic metrics (free VRAM, available RAM).
    fn probe_telemetry(&self) -> DynamicTelemetry;
}

/// Provider for user memory configuration settings.
pub trait MemorySettingsProvider: Send + Sync {
    fn get_settings(&self) -> MemorySettings;
}

impl<F> MemorySettingsProvider for F
where
    F: Fn() -> MemorySettings + Send + Sync,
{
    fn get_settings(&self) -> MemorySettings {
        (self)()
    }
}
