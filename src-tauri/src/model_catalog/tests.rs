#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::hardware_sizer::traits::HardwareProbe;
    use crate::hardware_sizer::types::*;
    use crate::hardware_sizer::HardwareSizer;
    use crate::model_catalog::traits::{LocalInventorySource, RemoteModelSource};
    use crate::model_catalog::types::*;
    use crate::model_catalog::ModelCatalog;

    struct MockRemoteSource {
        feed_calls: AtomicUsize,
        search_calls: AtomicUsize,
        detail_calls: AtomicUsize,
    }

    impl MockRemoteSource {
        fn new() -> Self {
            Self {
                feed_calls: AtomicUsize::new(0),
                search_calls: AtomicUsize::new(0),
                detail_calls: AtomicUsize::new(0),
            }
        }
    }

    impl RemoteModelSource for MockRemoteSource {
        fn fetch_category_feed(
            &self,
            _category: DiscoveryCategory,
        ) -> Result<Vec<CatalogModel>, String> {
            self.feed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![CatalogModel {
                model_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                name: "Llama-3.1-8B-Instruct".to_string(),
                author: "meta-llama".to_string(),
                downloads: 1_000_000,
                likes: 15_000,
                trending_score: 99.0,
                params_b: Some(8.0),
                family: Some("Llama-3".to_string()),
                publisher_tier: PublisherTier::OfficialOrg,
                available_quants: vec!["Q4_K_M".to_string(), "Q8_0".to_string()],
                best_quant: Some("Q4_K_M".to_string()),
                fit_score: 95,
                is_installed: false,
            }])
        }

        fn search_models(&self, query: &str) -> Result<Vec<CatalogModel>, String> {
            self.search_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![CatalogModel {
                model_id: format!("test/{query}"),
                name: query.to_string(),
                author: "test".to_string(),
                downloads: 500,
                likes: 10,
                trending_score: 50.0,
                params_b: Some(7.0),
                family: None,
                publisher_tier: PublisherTier::Community,
                available_quants: vec!["Q4_K_M".to_string()],
                best_quant: Some("Q4_K_M".to_string()),
                fit_score: 80,
                is_installed: false,
            }])
        }

        fn fetch_model_details(&self, model_id: &str) -> Result<CatalogModelDetails, String> {
            self.detail_calls.fetch_add(1, Ordering::SeqCst);
            Ok(CatalogModelDetails {
                model_id: model_id.to_string(),
                base_model_id: None,
                author: "meta-llama".to_string(),
                description: Some("Model description".to_string()),
                native_max_context: 131_072,
                recommended_context: 0, // Populated by ModelCatalog
                runtime_active_context: None,
                params_b: Some(70.0),
                num_layers: Some(80),
                num_kv_heads: Some(8),
                head_dim: Some(128),
                publisher_tier: PublisherTier::OfficialOrg,
                verified_variants: vec!["Q4_K_M".to_string(), "IQ3_M".to_string()],
                compatible_draft_models: Vec::new(),
                reasoning_effort_supported: false,
                chat_template_preview: Some(
                    "{{ bos_token }}<think>reasoning</think>{{ eos_token }}".to_string(),
                ),
            })
        }
    }

    struct MockInventorySource {
        scan_calls: AtomicUsize,
        current_mtime: AtomicU64,
    }

    impl MockInventorySource {
        fn new(mtime: u64) -> Self {
            Self {
                scan_calls: AtomicUsize::new(0),
                current_mtime: AtomicU64::new(mtime),
            }
        }
    }

    impl LocalInventorySource for MockInventorySource {
        fn scan_installed_models(&self) -> Result<Vec<InstalledModel>, String> {
            self.scan_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![InstalledModel {
                id: "llama-3-8b.gguf".to_string(),
                name: "llama-3-8b.gguf".to_string(),
                backend: "llamacpp".to_string(),
                quant: "Q4_K_M".to_string(),
                size_bytes: 4_500_000_000,
                local_path: "C:\\models\\llama-3-8b.gguf".to_string(),
                in_use: false,
                is_split: false,
            }])
        }

        fn directory_mtime_ms(&self, _path: &Path) -> Option<u64> {
            Some(self.current_mtime.load(Ordering::SeqCst))
        }
    }

    struct MockProbe;
    impl HardwareProbe for MockProbe {
        fn probe_topology(&self) -> StaticHardwareTopology {
            StaticHardwareTopology {
                cpu_name: "Ryzen 9".to_string(),
                cpu_cores: 16,
                host_ram_total_mb: 65536,
                gpu_name: Some("RTX 4090".to_string()),
                vram_total_mb: 24576,
                is_unified_memory: false,
                wsl_ram_total_mb: Some(32768),
            }
        }
        fn probe_telemetry(&self) -> DynamicTelemetry {
            DynamicTelemetry {
                host_ram_available_mb: 32000,
                vram_free_mb: 20000, // ~19.5 GB free
                wsl_ram_available_mb: Some(16000),
                gpu_utilization_pct: Some(10.0),
            }
        }
    }

    #[test]
    fn test_two_phase_hydration_caches_feed_and_hydrates_details_on_demand() {
        let remote = Arc::new(MockRemoteSource::new());
        let inventory = Arc::new(MockInventorySource::new(1000));
        let sizer = Arc::new(HardwareSizer::new(Arc::new(MockProbe)));
        let catalog = ModelCatalog::new(remote.clone(), inventory, sizer);

        assert_eq!(remote.feed_calls.load(Ordering::SeqCst), 0);
        let feed1 = catalog.get_feed(DiscoveryCategory::ForYou, false).unwrap();
        assert_eq!(feed1.len(), 1);
        assert_eq!(remote.feed_calls.load(Ordering::SeqCst), 1);

        // Second call hits cache
        let _feed2 = catalog.get_feed(DiscoveryCategory::ForYou, false).unwrap();
        assert_eq!(remote.feed_calls.load(Ordering::SeqCst), 1);

        // Deep details hydrated on demand
        assert_eq!(remote.detail_calls.load(Ordering::SeqCst), 0);
        let details = catalog
            .get_details("meta-llama/Llama-3.1-70B-Instruct")
            .unwrap();
        assert_eq!(remote.detail_calls.load(Ordering::SeqCst), 1);
        assert_eq!(details.native_max_context, 131_072);
    }

    #[test]
    fn test_three_tier_context_separation_calculates_recommended_ceiling() {
        let remote = Arc::new(MockRemoteSource::new());
        let inventory = Arc::new(MockInventorySource::new(1000));
        let sizer = Arc::new(HardwareSizer::new(Arc::new(MockProbe)));
        let catalog = ModelCatalog::new(remote, inventory, sizer);

        let details = catalog
            .get_details("meta-llama/Llama-3.1-70B-Instruct")
            .unwrap();
        // Model natively supports 128k
        assert_eq!(details.native_max_context, 131_072);
        // With ~19.5 GB free VRAM (>16 GB tier), recommended context is clamped to 32k
        assert_eq!(details.recommended_context, 32_768);
        // No server running yet, so runtime active context is None
        assert_eq!(details.runtime_active_context, None);
    }

    #[test]
    fn test_cached_local_inventory_avoids_rescans_until_mtime_changes() {
        let remote = Arc::new(MockRemoteSource::new());
        let inventory = Arc::new(MockInventorySource::new(5000));
        let sizer = Arc::new(HardwareSizer::new(Arc::new(MockProbe)));
        let catalog = ModelCatalog::new(remote, inventory.clone(), sizer);

        let dummy_path = Path::new("C:\\models");
        assert_eq!(inventory.scan_calls.load(Ordering::SeqCst), 0);

        // First listing scans disk
        let list1 = catalog.list_installed(dummy_path, false).unwrap();
        assert_eq!(list1.len(), 1);
        assert_eq!(inventory.scan_calls.load(Ordering::SeqCst), 1);

        // Second listing reuses cache because mtime is unchanged
        let _list2 = catalog.list_installed(dummy_path, false).unwrap();
        assert_eq!(inventory.scan_calls.load(Ordering::SeqCst), 1);

        // Directory mtime updates (e.g. model downloaded or deleted)
        inventory.current_mtime.store(6000, Ordering::SeqCst);
        let _list3 = catalog.list_installed(dummy_path, false).unwrap();
        assert_eq!(inventory.scan_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_speculative_draft_model_recommendations() {
        let remote = Arc::new(MockRemoteSource::new());
        let inventory = Arc::new(MockInventorySource::new(1000));
        let sizer = Arc::new(HardwareSizer::new(Arc::new(MockProbe)));
        let catalog = ModelCatalog::new(remote, inventory, sizer);

        // 70B Llama-3 model recommends 8B draft
        let drafts_70b =
            catalog.get_draft_suggestions("meta-llama/Llama-3.1-70B-Instruct", Some(70.0));
        assert!(!drafts_70b.is_empty());
        assert_eq!(
            drafts_70b[0].draft_model_id,
            "meta-llama/Llama-3.1-8B-Instruct"
        );
        assert_eq!(drafts_70b[0].params_b, 8.0);

        // 8B Llama-3 model recommends 1B draft
        let drafts_8b =
            catalog.get_draft_suggestions("meta-llama/Llama-3.1-8B-Instruct", Some(8.0));
        assert!(!drafts_8b.is_empty());
        assert_eq!(
            drafts_8b[0].draft_model_id,
            "meta-llama/Llama-3.2-1B-Instruct"
        );

        // DeepSeek-R1 recommends Distill-Qwen draft
        let drafts_r1 = catalog.get_draft_suggestions("deepseek-ai/DeepSeek-R1", Some(671.0));
        assert!(!drafts_r1.is_empty());
        assert_eq!(
            drafts_r1[0].draft_model_id,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        );
    }

    #[test]
    fn test_model_centric_chat_template_and_reasoning_detection() {
        let remote = Arc::new(MockRemoteSource::new());
        let inventory = Arc::new(MockInventorySource::new(1000));
        let sizer = Arc::new(HardwareSizer::new(Arc::new(MockProbe)));
        let catalog = ModelCatalog::new(remote, inventory, sizer);

        // Details for model with `<think>` template is automatically flagged as supporting reasoning effort
        let details = catalog
            .get_details("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            .unwrap();
        assert!(details.reasoning_effort_supported);
    }
}
