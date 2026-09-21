//! Client for the localmaxxing.com benchmark API.
//!
//! Reads from the embedded cache and llmfit community submissions collected
//! at build time; no network calls happen at runtime.

use crate::hardware::SystemSpecs;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Upper bound for a plausible measured decode rate. Local inference runs far
/// below this; higher values come from degenerate timings, e.g. one token with
/// a `eval_duration` of a few microseconds (#1038).
pub const MAX_PLAUSIBLE_TPS: f64 = 10_000.0;

/// Whether a tok/s value can be used as a measurement.
pub fn is_plausible_tps(tps: f64) -> bool {
    tps.is_finite() && tps > 0.0 && tps <= MAX_PLAUSIBLE_TPS
}

// Embedded benchmark cache — scraped by scripts/scrape_benchmarks.py
const BENCHMARK_CACHE_JSON: &str = include_str!("../data/benchmark_cache.json");

// Community submissions contributed via `llmfit bench --share`, aggregated
// from data/community/ by build.rs. Merged submission → next release ships it.
const COMMUNITY_BENCH_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/community_benchmarks.json"));

// ── Response types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    #[serde(default)]
    pub hw_class: Option<String>,
    #[serde(default)]
    pub gpu_name: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub gpu_count: Option<u32>,
    #[serde(default)]
    pub chip_vendor: Option<String>,
    #[serde(default)]
    pub chip_family: Option<String>,
    #[serde(default)]
    pub chip_variant: Option<String>,
    #[serde(default)]
    pub unified_memory_gb: Option<f64>,
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEntry {
    pub id: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub model: Option<LeaderboardModel>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub engine: Option<LeaderboardEngine>,
    #[serde(default)]
    pub engine_flags: Option<LeaderboardEngineFlags>,
    #[serde(default)]
    pub user: Option<LeaderboardUser>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardModel {
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub params: Option<f64>,
    #[serde(default)]
    pub is_mo_e: Option<bool>,
}

/// Per-run engine acceleration flags. Speculative decoding and MTP runs
/// measure draft-accelerated throughput, which can exceed the memory
/// bandwidth roofline that plain autoregressive estimates model — consumers
/// comparing against `estimate_tps` must filter these out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngineFlags {
    #[serde(default)]
    pub spec_decoding: Option<bool>,
    #[serde(default)]
    pub mtp_enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngine {
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub engine_version: Option<String>,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardUser {
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
}

impl LeaderboardEntry {
    /// Helper to get the model HF ID.
    pub fn hf_id(&self) -> &str {
        self.model.as_ref().map(|m| m.hf_id.as_str()).unwrap_or("")
    }

    /// Helper to get the engine name.
    pub fn engine_name(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.engine_name.as_str())
            .unwrap_or("")
    }

    /// Helper to get the quantization.
    pub fn quantization(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.quantization.as_str())
            .unwrap_or("")
    }

    /// Helper to get the username.
    pub fn username(&self) -> &str {
        self.user
            .as_ref()
            .and_then(|u| u.username.as_deref())
            .unwrap_or("anon")
    }

    /// Helper to check verified status.
    pub fn verified(&self) -> bool {
        self.user.as_ref().and_then(|u| u.verified).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardResponse {
    pub rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

// ── Embedded cache ───────────────────────────────────────────────────

/// Cache structure matching the scraper output.
#[derive(Debug, Clone, Deserialize)]
struct BenchmarkCache {
    #[serde(default)]
    presets: std::collections::HashMap<String, CachedPreset>,
}

#[derive(Debug, Clone, Deserialize)]
struct CachedPreset {
    rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    total: u64,
}

/// Lazily parsed embedded benchmark cache.
fn embedded_cache() -> &'static BenchmarkCache {
    static CACHE: OnceLock<BenchmarkCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(BENCHMARK_CACHE_JSON).unwrap_or_else(|_| BenchmarkCache {
            presets: std::collections::HashMap::new(),
        })
    })
}

/// Look up cached leaderboard data for a hardware preset label.
pub fn cached_leaderboard_for_preset(label: &str) -> Option<LeaderboardResponse> {
    let cache = embedded_cache();
    cache.presets.get(label).map(|p| LeaderboardResponse {
        rows: p.rows.clone(),
        total: p.total,
        limit: p.rows.len() as u64,
        offset: 0,
    })
}

/// All preset labels present in the embedded benchmark cache. Used by the
/// estimate-calibration test to replay every cached measurement.
pub fn cached_preset_labels() -> Vec<&'static str> {
    let mut labels: Vec<&'static str> = embedded_cache()
        .presets
        .keys()
        .map(|s| s.as_str())
        .collect();
    labels.sort_unstable();
    labels
}

// ── Measured throughput lookup (provenance-weighted estimates) ──────

/// Where a measured throughput came from. Priority when annotating fits:
/// your own runs on this machine, then llmfit community submissions recorded
/// on identical hardware, then localmaxxing medians on matching presets —
/// all of which outrank the formula estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeasuredSource {
    /// localmaxxing.com leaderboard runs on hardware matching the detected GPU.
    #[default]
    Community,
    /// `llmfit bench --share` submissions merged into the llmfit repo,
    /// recorded on hardware identical to this machine's.
    CommunityLlmfit,
    /// `llmfit bench` runs recorded in this machine's local store.
    LocalBench,
}

/// A real measured throughput from benchmark runs — either the community
/// leaderboard on matching hardware, or the user's own `llmfit bench` runs on
/// this very machine. When present, this is ground truth and should be
/// displayed with priority over the formula estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredTps {
    /// Measured generation throughput (tok/s): community median, or the most
    /// recent local run.
    pub tok_s: f64,
    /// Number of runs behind the value.
    pub sample_count: u32,
    /// Hardware the runs come from (e.g. "RTX 3090 (24 GB)" or "this machine").
    pub hardware_label: String,
    #[serde(default)]
    pub source: MeasuredSource,
}

/// Find the embedded-cache hardware preset matching the detected GPU.
pub fn cached_preset_for_specs(specs: &SystemSpecs) -> Option<&'static HardwarePreset> {
    let gpu = specs.gpu_name.as_deref()?.to_lowercase();
    HardwarePreset::all().iter().find(|p| {
        p.hardware_name
            .is_some_and(|hw| gpu.contains(&hw.to_lowercase()))
    })
}

/// Prebuilt index of comparable measured runs for the detected hardware.
/// Build once per fit pass, then O(1) lookups per model.
pub struct MeasuredTpsIndex {
    hardware_label: &'static str,
    /// (model slug, quantization lowercase) -> sorted tok/s samples.
    samples: std::collections::HashMap<(String, String), Vec<f64>>,
}

impl MeasuredTpsIndex {
    /// Build the index from the embedded cache for the preset matching the
    /// detected GPU. Only rows comparable to `estimated_tps` are indexed:
    /// single-request generation (batch <= 1), no speculative decoding /
    /// MTP. Returns None when the GPU has no preset or no rows qualify.
    pub fn for_specs(specs: &SystemSpecs) -> Option<Self> {
        let preset = cached_preset_for_specs(specs)?;
        let resp = cached_leaderboard_for_preset(preset.label)?;
        Self::from_rows(&resp.rows, preset.label)
    }

    fn from_rows(rows: &[LeaderboardEntry], hardware_label: &'static str) -> Option<Self> {
        let mut samples: std::collections::HashMap<(String, String), Vec<f64>> =
            std::collections::HashMap::new();
        for row in rows {
            if row.batch_size.unwrap_or(1) > 1 {
                continue;
            }
            if row
                .engine_flags
                .as_ref()
                .is_some_and(|f| f.spec_decoding.unwrap_or(false) || f.mtp_enabled.unwrap_or(false))
            {
                continue;
            }
            let Some(tok_s) = row.tok_s_out.filter(|t| *t > 0.5) else {
                continue;
            };
            let hf_id = row.hf_id();
            let quant = row.quantization();
            if hf_id.is_empty() || quant.is_empty() {
                continue;
            }
            samples
                .entry((crate::models::canonical_slug(hf_id), quant.to_lowercase()))
                .or_default()
                .push(tok_s);
        }
        if samples.is_empty() {
            return None;
        }
        for s in samples.values_mut() {
            s.sort_by(|a, b| a.partial_cmp(b).expect("tok/s samples are finite"));
        }
        Some(Self {
            hardware_label,
            samples,
        })
    }

    /// Median measured tok/s for this model+quant, if any comparable runs
    /// exist on the matched hardware.
    pub fn lookup(&self, model_hf_id: &str, quant: &str) -> Option<MeasuredTps> {
        let key = (
            crate::models::canonical_slug(model_hf_id),
            quant.to_lowercase(),
        );
        let s = self.samples.get(&key)?;
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        Some(MeasuredTps {
            tok_s: median,
            sample_count: n as u32,
            hardware_label: self.hardware_label.to_string(),
            source: MeasuredSource::Community,
        })
    }
}

/// One-shot lookup: median measured tok/s for `model_hf_id` at `quant` on
/// hardware matching the detected specs. Prefer [`MeasuredTpsIndex`] when
/// annotating many fits.
pub fn measured_tps_for(
    specs: &SystemSpecs,
    model_hf_id: &str,
    quant: &str,
) -> Option<MeasuredTps> {
    MeasuredTpsIndex::for_specs(specs)?.lookup(model_hf_id, quant)
}

// ── Embedded llmfit community submissions ───────────────────────────

/// All community benchmark submissions embedded at build time (see
/// build.rs). Each element is a full submission payload conforming to
/// `data/community/schema.json`.
pub fn community_submissions() -> &'static [serde_json::Value] {
    static CACHE: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    CACHE.get_or_init(|| serde_json::from_str(COMMUNITY_BENCH_JSON).unwrap_or_default())
}

/// Whether a submission's recorded `hardware` object matches `specs` (same
/// CPU and GPU name). Shared by the local store and the embedded community
/// data: measurements only transfer between identical configurations.
pub fn hardware_payload_matches(hw: &serde_json::Value, specs: &SystemSpecs) -> bool {
    let cpu_ok = hw["cpu"]
        .as_str()
        .is_some_and(|c| c.eq_ignore_ascii_case(&specs.cpu_name));
    let gpu_ok = match (&specs.gpu_name, hw["hardwareName"].as_str()) {
        (Some(now), Some(then)) => now.eq_ignore_ascii_case(then),
        (None, None) => true,
        _ => false,
    };
    cpu_ok && gpu_ok
}

/// One benchmark result from a community submission, for leaderboard display.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    pub model: String,
    pub provider: String,
    pub avg_tps: f64,
    pub ttft_ms: Option<f64>,
}

/// Community results recorded on hardware matching `specs`, newest
/// submission first.
pub fn community_results_for_specs(specs: &SystemSpecs) -> Vec<CommunityResult> {
    let mut subs: Vec<&serde_json::Value> = community_submissions()
        .iter()
        .filter(|s| hardware_payload_matches(&s["hardware"], specs))
        .collect();
    subs.sort_by_key(|s| std::cmp::Reverse(s["submittedAtUnix"].as_u64().unwrap_or(0)));

    let mut out = Vec::new();
    for s in subs {
        for r in s["results"]
            .as_array()
            .map(|v| v.as_slice())
            .unwrap_or_default()
        {
            let Some(tps) = r["avgTps"].as_f64().filter(|t| is_plausible_tps(*t))
            else {
                continue;
            };
            out.push(CommunityResult {
                model: r["model"].as_str().unwrap_or("?").to_string(),
                provider: r["provider"].as_str().unwrap_or("").to_string(),
                avg_tps: tps,
                ttft_ms: r["avgTtftMs"].as_f64(),
            });
        }
    }
    out
}

/// Index of embedded community submissions recorded on hardware identical to
/// the detected machine, for annotating fit rows. Sits between the user's
/// own local runs and localmaxxing preset medians in trust order — and gives
/// a fresh install measured numbers (and calibration anchors) from day one
/// when someone already contributed on the same hardware.
pub struct CommunityBenchIndex {
    /// (provider model tag, tok/s), newest submission first.
    entries: Vec<(String, f64)>,
}

impl CommunityBenchIndex {
    pub fn for_specs(specs: &SystemSpecs) -> Option<Self> {
        let entries: Vec<(String, f64)> = community_results_for_specs(specs)
            .into_iter()
            .map(|r| (r.model, r.avg_tps))
            .collect();
        (!entries.is_empty()).then_some(Self { entries })
    }

    /// Median community-measured tok/s for a catalog model, resolved through
    /// the same tag-matching heuristics as installed detection.
    pub fn lookup(&self, model_hf_name: &str) -> Option<MeasuredTps> {
        let mut matches: Vec<f64> = self
            .entries
            .iter()
            .filter(|(tag, _)| crate::providers::tag_matches_model(tag, model_hf_name))
            .map(|(_, tps)| *tps)
            .collect();
        if matches.is_empty() {
            return None;
        }
        matches.sort_by(|a, b| a.partial_cmp(b).expect("tok/s values are finite"));
        let n = matches.len();
        let median = if n % 2 == 1 {
            matches[n / 2]
        } else {
            (matches[n / 2 - 1] + matches[n / 2]) / 2.0
        };
        Some(MeasuredTps {
            tok_s: median,
            sample_count: n as u32,
            hardware_label: "identical hardware".to_string(),
            source: MeasuredSource::CommunityLlmfit,
        })
    }
}

// ── Hardware presets ─────────────────────────────────────────────────

/// A selectable hardware configuration for browsing benchmarks.
#[derive(Debug, Clone)]
pub struct HardwarePreset {
    pub label: &'static str,
    pub hw_class: &'static str,
    pub hardware_name: Option<&'static str>,
    pub mem_tier: Option<u32>,
}

impl HardwarePreset {
    /// Returns the built-in list of popular hardware presets.
    pub fn all() -> &'static [HardwarePreset] {
        &HARDWARE_PRESETS
    }
}

static HARDWARE_PRESETS: [HardwarePreset; 27] = [
    // NVIDIA consumer
    HardwarePreset {
        label: "RTX 5090 (32 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5090"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "RTX 5080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 4080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4070 Ti (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4070"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 3080 (10 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3080"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3060 (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3060"),
        mem_tier: Some(12),
    },
    // NVIDIA datacenter
    HardwarePreset {
        label: "A100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "A100 (40 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "H100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("H100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "L40S (48 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("L40S"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "T4 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("T4"),
        mem_tier: Some(16),
    },
    // AMD
    HardwarePreset {
        label: "RX 7900 XTX (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XTX"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RX 7900 XT (20 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XT"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "MI300X (192 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("MI300X"),
        mem_tier: Some(128),
    },
    // Apple Silicon
    HardwarePreset {
        label: "Apple M4 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M4 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (48 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (24 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "Apple M3 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M3 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Ultra (192 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Ultra"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M2 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Pro (32 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Pro"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "Apple M1 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M1 Max"),
        mem_tier: Some(48),
    },
    // CPU only
    HardwarePreset {
        label: "CPU Only",
        hw_class: "CPU_ONLY",
        hardware_name: None,
        mem_tier: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::GpuBackend;

    fn row(json: &str) -> LeaderboardEntry {
        serde_json::from_str(json).expect("valid test row")
    }

    fn specs(cpu: &str, gpu: Option<&str>) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: cpu.to_string(),
            has_gpu: gpu.is_some(),
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_available_gb: None,
            gpu_name: gpu.map(str::to_string),
            gpu_count: u32::from(gpu.is_some()),
            unified_memory: false,
            backend: GpuBackend::CpuX86,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    #[test]
    fn community_embed_parses_and_is_seeded() {
        let subs = community_submissions();
        // data/community/ ships with at least the first genuine submission;
        // this breaks if build.rs stops aggregating it into the binary.
        assert!(!subs.is_empty(), "embedded community aggregate is empty");
        for s in subs {
            assert_eq!(s["schemaVersion"], 1, "unexpected submission shape");
            assert!(s["results"].as_array().is_some_and(|r| !r.is_empty()));
            assert!(s["hardware"]["cpu"].is_string());
        }
    }

    #[test]
    fn community_results_filter_by_hardware() {
        // The seeded submission was recorded on this exact configuration.
        let seeded = specs(
            "Intel(R) Core(TM) Ultra 7 258V",
            Some("Intel Arc Graphics 130V/140V (integrated)"),
        );
        let rows = community_results_for_specs(&seeded);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|r| r.avg_tps > 0.0));

        // A different machine gets nothing.
        let other = specs("AMD Ryzen 9 7950X", Some("NVIDIA GeForce RTX 4090"));
        assert!(community_results_for_specs(&other).is_empty());
    }

    #[test]
    fn hardware_payload_matching_rules() {
        use serde_json::json;
        let hw = json!({"cpu": "Test CPU", "hardwareName": "Test GPU"});
        assert!(hardware_payload_matches(
            &hw,
            &specs("Test CPU", Some("Test GPU"))
        ));
        assert!(hardware_payload_matches(
            &hw,
            &specs("test cpu", Some("test gpu"))
        ));
        assert!(!hardware_payload_matches(&hw, &specs("Test CPU", None)));
        assert!(!hardware_payload_matches(
            &hw,
            &specs("Other CPU", Some("Test GPU"))
        ));

        let cpu_only = json!({"cpu": "Test CPU", "hardwareName": null});
        assert!(hardware_payload_matches(
            &cpu_only,
            &specs("Test CPU", None)
        ));
        assert!(!hardware_payload_matches(
            &cpu_only,
            &specs("Test CPU", Some("Test GPU"))
        ));
    }

    #[test]
    fn test_measured_index_median_and_comparability_filters() {
        let rows = vec![
            row(
                r#"{"id":"1","tokSOut":100.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            row(
                r#"{"id":"2","tokSOut":120.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            // Batched serving measures a different quantity — excluded.
            row(
                r#"{"id":"3","tokSOut":900.0,"batchSize":8,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"vllm","quantization":"Q4_K_M"}}"#,
            ),
            // Draft-accelerated runs beat the roofline — excluded.
            row(
                r#"{"id":"4","tokSOut":500.0,"engineFlags":{"specDecoding":true},"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
        ];
        let idx = MeasuredTpsIndex::from_rows(&rows, "RTX 3090 (24 GB)")
            .expect("comparable rows should build an index");

        let m = idx
            .lookup("unsloth/Qwen3-4B-GGUF", "q4_k_m")
            .expect("quant match is case-insensitive");
        assert_eq!(
            m.sample_count, 2,
            "batched/spec-decoding rows must not count"
        );
        assert!((m.tok_s - 110.0).abs() < 1e-9, "median of 100 and 120");
        assert_eq!(m.hardware_label, "RTX 3090 (24 GB)");

        assert!(
            idx.lookup("unsloth/Qwen3-4B-GGUF", "Q8_0").is_none(),
            "different quant must not match"
        );
    }

    #[test]
    fn test_measured_index_none_when_no_comparable_rows() {
        let rows = vec![row(
            r#"{"id":"1","tokSOut":900.0,"batchSize":16,"model":{"hfId":"a/b"},"engine":{"engineName":"vllm","quantization":"FP8"}}"#,
        )];
        assert!(MeasuredTpsIndex::from_rows(&rows, "A100 (80 GB)").is_none());
    }
}
