#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::hardware_sizer::traits::HardwareProbe;
    use crate::hardware_sizer::types::*;
    use crate::hardware_sizer::HardwareSizer;
    use crate::state::{MemorySettings, ServerDef};

    struct MockHardwareProbe {
        topology_probes: AtomicUsize,
        telemetry_probes: AtomicUsize,
        telemetry_free_vram: u64,
        telemetry_host_ram_avail: u64,
        telemetry_wsl_ram_avail: Option<u64>,
    }

    impl MockHardwareProbe {
        fn new(free_vram: u64, host_ram_avail: u64, wsl_ram_avail: Option<u64>) -> Self {
            Self {
                topology_probes: AtomicUsize::new(0),
                telemetry_probes: AtomicUsize::new(0),
                telemetry_free_vram: free_vram,
                telemetry_host_ram_avail: host_ram_avail,
                telemetry_wsl_ram_avail: wsl_ram_avail,
            }
        }
    }

    impl HardwareProbe for MockHardwareProbe {
        fn probe_topology(&self) -> StaticHardwareTopology {
            self.topology_probes.fetch_add(1, Ordering::SeqCst);
            StaticHardwareTopology {
                cpu_name: "AMD Ryzen 9 7950X".to_string(),
                cpu_cores: 16,
                host_ram_total_mb: 65536,
                gpu_name: Some("NVIDIA GeForce RTX 4090".to_string()),
                vram_total_mb: 24576,
                is_unified_memory: false,
                wsl_ram_total_mb: Some(32768),
            }
        }

        fn probe_telemetry(&self) -> DynamicTelemetry {
            self.telemetry_probes.fetch_add(1, Ordering::SeqCst);
            DynamicTelemetry {
                host_ram_available_mb: self.telemetry_host_ram_avail,
                vram_free_mb: self.telemetry_free_vram,
                wsl_ram_available_mb: self.telemetry_wsl_ram_avail,
                gpu_utilization_pct: Some(15.0),
            }
        }
    }

    #[test]
    fn test_two_tier_caching_probes_static_topology_once() {
        let probe = Arc::new(MockHardwareProbe::new(20000, 30000, Some(16000)));
        let sizer = HardwareSizer::new(probe.clone());

        assert_eq!(probe.topology_probes.load(Ordering::SeqCst), 0);
        assert_eq!(probe.telemetry_probes.load(Ordering::SeqCst), 0);

        // First snapshot triggers topology probe + telemetry probe
        let snap1 = sizer.snapshot();
        assert_eq!(probe.topology_probes.load(Ordering::SeqCst), 1);
        assert_eq!(probe.telemetry_probes.load(Ordering::SeqCst), 1);
        assert_eq!(snap1.vram_total_mb, 24576);

        // Second snapshot reuses cached topology and only probes telemetry
        let _snap2 = sizer.snapshot();
        assert_eq!(probe.topology_probes.load(Ordering::SeqCst), 1);
        assert_eq!(probe.telemetry_probes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_coalesced_effective_free_vram_accounts_for_reservations() {
        let probe = Arc::new(MockHardwareProbe::new(12000, 30000, Some(16000)));
        let sizer = HardwareSizer::new(probe);

        let snap_initial = sizer.snapshot();
        assert_eq!(snap_initial.vram_free_mb, 12000);
        assert_eq!(snap_initial.vram_reserved_mb, 0);
        assert_eq!(snap_initial.vram_effective_free_mb, 12000);

        // Reserve 5000 MB for starting server-1
        sizer.reserve_vram("server-1", 5000);
        let snap_res1 = sizer.snapshot();
        assert_eq!(snap_res1.vram_free_mb, 12000);
        assert_eq!(snap_res1.vram_reserved_mb, 5000);
        assert_eq!(snap_res1.vram_effective_free_mb, 7000);

        // Over-reserve beyond free VRAM (should saturate to 0)
        sizer.reserve_vram("server-2", 10000);
        let snap_res2 = sizer.snapshot();
        assert_eq!(snap_res2.vram_reserved_mb, 15000);
        assert_eq!(snap_res2.vram_effective_free_mb, 0);

        // Release server-1
        sizer.release_vram("server-1");
        let snap_release = sizer.snapshot();
        assert_eq!(snap_release.vram_reserved_mb, 10000);
        assert_eq!(snap_release.vram_effective_free_mb, 2000);
    }

    #[test]
    fn test_dual_pool_fit_scoring_wsl_vs_host() {
        // Host RAM avail = 40000 MB (~39 GB); WSL RAM avail = 8000 MB (~7.8 GB)
        // Free VRAM = 2000 MB (~1.9 GB)
        let probe = Arc::new(MockHardwareProbe::new(2000, 40000, Some(8000)));
        let sizer = HardwareSizer::new(probe);
        let settings = MemorySettings {
            offload_weights_allowed: true,
            vram_overhead_mb: 500.0,
            ..MemorySettings::default()
        };

        // Model: 14B parameter model in Q4_K_M ~ 7.7 GB
        let geom = ModelGeometry::new(14.0, "q4_k_m");

        // When targeting vLLM (WSL pool: 8 GB avail, minus safety margins):
        // 7.7 GB + KV cache is TooTight for WSL's 8 GB budget
        let fit_vllm = sizer.evaluate_fit(&geom, TargetBackend::Vllm, &settings);
        assert_eq!(fit_vllm.target_backend, TargetBackend::Vllm);

        // When targeting llama.cpp (Host pool: 40 GB avail):
        // 7.7 GB fits into Host RAM hybrid offload
        let fit_llama = sizer.evaluate_fit(&geom, TargetBackend::LlamaCpp, &settings);
        assert_eq!(fit_llama.target_backend, TargetBackend::LlamaCpp);
        assert_eq!(fit_llama.verdict, FitVerdict::HybridOffload);
    }

    #[test]
    fn test_allocation_report_computes_weights_spill_and_pressure() {
        let probe = Arc::new(MockHardwareProbe::new(16000, 30000, Some(16000)));
        let sizer = HardwareSizer::new(probe);
        let settings = MemorySettings {
            vram_overhead_mb: 1024.0,
            ..MemorySettings::default()
        };

        let mut def = ServerDef {
            id: "srv-1".to_string(),
            model_id: "meta-llama/Llama-3-8B-Instruct".to_string(),
            params_b: Some(8.0),
            quant: "q4_k_m".to_string(),
            backend: "llamacpp".to_string(),
            n_gpu_layers: Some(33),
            ..ServerDef::default()
        };

        let report = sizer.allocation_report(&[&def], &settings);
        assert_eq!(report.running_servers_count, 1);
        // 8B * 0.55 = 4.4 GB
        assert!((report.total_weight_gb - 4.4).abs() < 0.1);
        assert_eq!(report.ram_spill_gb, 0.0);
        assert!(!report.is_low_vram);

        // Make it CPU only (-ngl 0)
        def.n_gpu_layers = Some(0);
        let report_cpu = sizer.allocation_report(&[&def], &settings);
        assert!((report_cpu.ram_spill_gb - 4.4).abs() < 0.1);
        assert!(report_cpu.warning_message.is_some());
    }

    #[test]
    fn test_fit_verdict_full_gpu_for_small_model() {
        let probe = Arc::new(MockHardwareProbe::new(20000, 30000, Some(16000)));
        let sizer = HardwareSizer::new(probe);
        let settings = MemorySettings::default();

        // 3B model in Q4 ~ 1.65 GB fits easily into 20 GB free VRAM
        let geom = ModelGeometry::new(3.0, "q4_k_m");
        let fit = sizer.evaluate_fit(&geom, TargetBackend::Generic, &settings);
        assert_eq!(fit.verdict, FitVerdict::FullGpu);
        assert!(fit.score >= 90);
        assert!(fit.max_context > 8192);
    }
}
