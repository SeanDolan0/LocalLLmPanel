//! Local benchmark store.
//!
//! Every completed bench run is recorded as a ready-to-upload submission
//! payload under the local data directory (overridable with
//! `LLMFIT_BENCH_STORE`). [`LocalBenchIndex`] reads those stored runs so fit
//! estimates can prefer measurements taken on this exact machine over
//! community medians and formula estimates.

use crate::hardware::SystemSpecs;
use serde_json::Value;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Local benchmark store
// ---------------------------------------------------------------------------

/// A submission payload recorded in the local benchmark store.
#[derive(Clone)]
pub struct StoredBenchmark {
    pub path: PathBuf,
    pub payload: Value,
}

impl StoredBenchmark {
    /// Whether this run was recorded on hardware matching `specs` (same CPU
    /// and GPU). Measurements from a previous machine configuration must not
    /// override or calibrate estimates for the current one.
    pub fn matches_hardware(&self, specs: &SystemSpecs) -> bool {
        crate::benchmarks::hardware_payload_matches(&self.payload["hardware"], specs)
    }
}

/// Root of the local store. Overridable with `LLMFIT_BENCH_STORE` (useful for
/// tests and for keeping the store on a shared volume).
fn store_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LLMFIT_BENCH_STORE")
        && !dir.trim().is_empty()
    {
        return Some(PathBuf::from(dir));
    }
    Some(dirs::data_local_dir()?.join("llmfit").join("benchmarks"))
}

/// llama-server reports the value of its `-m/--model` argument, usually an
/// absolute filesystem path to a GGUF, as the model id in its
/// OpenAI-compatible `/v1/models` listing, and that id ends up verbatim in
/// `BenchResult::model`. Keep only the file name when the value is a path to
/// a `.gguf` file, so no machine-specific directory (often a username) leaks
/// into a stored submission (#819). Every other id shape (Ollama tags,
/// HF-style `org/model` ids from vLLM or MLX, bare file names) passes
/// through unchanged.
///
/// Only ids that look like absolute paths are stripped (leading `/` or `\\`,
/// or a Windows drive letter), so Hub-style references such as
/// `hf.co/org/repo/file.gguf` keep their namespace: they contain separators
/// and end in `.gguf` but carry nothing machine-specific. Splits on both
/// separator kinds, like `tag_matches_model` does: a Windows path can show
/// up verbatim in the listing.
fn strip_gguf_path(id: &str) -> String {
    let is_absolute = id.starts_with('/')
        || id.starts_with('\\')
        || matches!(id.as_bytes(), [_, b':', b'/' | b'\\', ..]);
    if is_absolute && id.to_ascii_lowercase().ends_with(".gguf") {
        id.rsplit(['/', '\\']).next().unwrap_or(id).to_string()
    } else {
        id.to_string()
    }
}

/// Rewrite absolute GGUF paths left in the `model` field of a stored payload.
/// New payloads are normalised at write time, but stores written by older
/// binaries still carry paths (#819). Scrubbing at load time means the
/// listing, the `--dry-run` preview and the upload all agree, and no
/// machine-specific path leaves the machine.
fn sanitize_stored_payload(payload: &mut Value) {
    if let Some(results) = payload.get_mut("results").and_then(Value::as_array_mut) {
        for r in results {
            if let Some(model) = r["model"].as_str() {
                let stripped = strip_gguf_path(model);
                if stripped != model {
                    r["model"] = Value::String(stripped);
                }
            }
        }
    }
}

fn read_store(subdir: &str) -> Vec<StoredBenchmark> {
    let Some(dir) = store_root().map(|r| r.join(subdir)) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<StoredBenchmark> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                return None;
            }
            let mut payload: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
            sanitize_stored_payload(&mut payload);
            Some(StoredBenchmark { path, payload })
        })
        .collect();
    // Filenames start with the unix timestamp, so path order is record order.
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Stored benchmarks not yet contributed upstream, oldest first.
pub fn pending_benchmarks() -> Vec<StoredBenchmark> {
    read_store("pending")
}

/// Stored benchmarks already contributed upstream, oldest first.
pub fn shared_benchmarks() -> Vec<StoredBenchmark> {
    read_store("shared")
}

/// Index of the user's own benchmark runs (pending and shared), used to
/// annotate fit rows: a throughput measured on THIS machine is ground truth
/// and takes priority over community medians and formula estimates.
pub struct LocalBenchIndex {
    /// (provider model tag, tok/s), newest run first.
    entries: Vec<(String, f64)>,
}

impl LocalBenchIndex {
    /// Load every stored benchmark result recorded on hardware matching
    /// `specs`. Returns `None` when nothing qualifies so callers can skip
    /// per-model lookups entirely.
    pub fn load(specs: &SystemSpecs) -> Option<Self> {
        let mut entries: Vec<(String, f64)> = Vec::new();
        for s in shared_benchmarks().into_iter().chain(pending_benchmarks()) {
            if !s.matches_hardware(specs) {
                continue;
            }
            let Some(results) = s.payload["results"].as_array() else {
                continue;
            };
            for r in results {
                if let (Some(model), Some(tps)) = (r["model"].as_str(), r["avgTps"].as_f64())
                    && crate::benchmarks::is_plausible_tps(tps)
                {
                    entries.push((model.to_string(), tps));
                }
            }
        }
        // Store reads are oldest-first; prefer the newest measurement.
        entries.reverse();
        (!entries.is_empty()).then_some(Self { entries })
    }

    /// Most recent locally measured tok/s for a catalog model, if any stored
    /// run's provider tag matches it.
    pub fn lookup(&self, model_hf_name: &str) -> Option<crate::benchmarks::MeasuredTps> {
        let matches: Vec<f64> = self
            .entries
            .iter()
            .filter(|(tag, _)| crate::providers::tag_matches_model(tag, model_hf_name))
            .map(|(_, tps)| *tps)
            .collect();
        Some(crate::benchmarks::MeasuredTps {
            tok_s: *matches.first()?,
            sample_count: matches.len() as u32,
            hardware_label: "this machine".to_string(),
            source: crate::benchmarks::MeasuredSource::LocalBench,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn specs_with_gpu(name: &str) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(24.0),
            total_gpu_vram_gb: Some(24.0),
            gpu_available_gb: None,
            gpu_name: Some(name.to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: crate::hardware::GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    #[test]
    fn local_store_reads_and_indexes_stored_runs() {
        // LLMFIT_BENCH_STORE scopes the store to a temp dir. Env vars are
        // process-global, so this is the only test that may touch the store.
        let dir = std::env::temp_dir().join(format!("llmfit-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::set_var("LLMFIT_BENCH_STORE", &dir) };

        let specs = specs_with_gpu("NVIDIA GeForce RTX 4090");
        let pending = dir.join("pending");
        std::fs::create_dir_all(&pending).unwrap();
        let path = pending.join("1752100000-abcd1234.json");
        std::fs::write(
            &path,
            r#"{"schemaVersion":1,"hardware":{"cpu":"Test CPU","hardwareName":"NVIDIA GeForce RTX 4090"},"results":[{"model":"llama3.1:8b","provider":"ollama","avgTps":128.44},{"model":"ferrum","provider":"ferrum","avgTps":96.7}]}"#,
        )
        .unwrap();

        let stored = pending_benchmarks();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].payload["results"][1]["provider"], "ferrum");
        assert!(shared_benchmarks().is_empty());

        // #819: a payload written by an older binary can still carry an
        // absolute GGUF path in `model`. The store scrubs it at load time, so
        // the listing, the dry-run preview and the upload all see the file
        // name.
        let legacy = pending.join("1700000000-legacy00.json");
        std::fs::write(
            &legacy,
            r#"{"schemaVersion":1,"hardware":{"cpu":"Test CPU","hardwareName":"NVIDIA GeForce RTX 4090"},"results":[{"model":"/home/user/gguf/phi-4-Q4_K_M.gguf","provider":"llamacpp","avgTps":10.0}]}"#,
        )
        .unwrap();
        let with_legacy = pending_benchmarks();
        assert_eq!(with_legacy.len(), 2);
        assert_eq!(
            with_legacy[0].payload["results"][0]["model"],
            "phi-4-Q4_K_M.gguf"
        );
        std::fs::remove_file(&legacy).unwrap();

        // The local index resolves the stored run for the matching catalog
        // model (ollama tag "llama3.1:8b" ↔ HF-style name) and outranks
        // nothing else: unknown models get no local measurement.
        let idx = LocalBenchIndex::load(&specs).expect("store has one run");
        let m = idx.lookup("test/llama-3.1-8b").expect("tag should match");
        assert_eq!(m.tok_s, 128.44);
        assert_eq!(m.sample_count, 1);
        assert_eq!(m.source, crate::benchmarks::MeasuredSource::LocalBench);
        assert!(idx.lookup("test/qwen2.5-7b").is_none());

        // Runs recorded on different hardware never leak into the index.
        let other_gpu = specs_with_gpu("NVIDIA GeForce RTX 3060");
        assert!(LocalBenchIndex::load(&other_gpu).is_none());
        assert!(!with_legacy[0].matches_hardware(&other_gpu));

        unsafe { std::env::remove_var("LLMFIT_BENCH_STORE") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strip_gguf_path_keeps_only_the_file_name_for_gguf_paths() {
        assert_eq!(
            strip_gguf_path("/home/user/gguf/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            "SmolLM2-135M-Instruct-Q4_K_M.gguf"
        );
        assert_eq!(
            strip_gguf_path(r"C:\models\phi-4-Q4_K_M.GGUF"),
            "phi-4-Q4_K_M.GGUF"
        );
    }

    #[test]
    fn strip_gguf_path_leaves_non_path_ids_untouched() {
        // Ollama tag.
        assert_eq!(strip_gguf_path("llama3.1:8b"), "llama3.1:8b");
        // HF-style id from vLLM or MLX: contains a slash but is not a file.
        assert_eq!(
            strip_gguf_path("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama/Llama-3.1-8B-Instruct"
        );
        // GGUF repo id: ends in "GGUF" but not ".gguf".
        assert_eq!(
            strip_gguf_path("unsloth/Qwen3-4B-GGUF"),
            "unsloth/Qwen3-4B-GGUF"
        );
        // Already a bare file name.
        assert_eq!(strip_gguf_path("model.gguf"), "model.gguf");
        // Hub-style reference: separators and a .gguf suffix, but relative,
        // so the org and repo context must survive.
        assert_eq!(
            strip_gguf_path(
                "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
            ),
            "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
        );
        // Relative filesystem path: nothing machine-specific to hide.
        assert_eq!(strip_gguf_path("models/foo.gguf"), "models/foo.gguf");
    }

    #[test]
    fn sanitize_stored_payload_scrubs_absolute_model_paths() {
        let mut payload = json!({
            "results": [
                { "model": "/home/alice/models/phi-4-Q4_K_M.gguf" },
                { "model": "llama3.1:8b" }
            ]
        });
        sanitize_stored_payload(&mut payload);
        assert_eq!(payload["results"][0]["model"], "phi-4-Q4_K_M.gguf");
        assert_eq!(payload["results"][1]["model"], "llama3.1:8b");
    }

    #[test]
    fn sanitize_stored_payload_ignores_non_object_payloads() {
        // read_store treats every pending file as untrusted: a hand-edited
        // file whose top level is not a JSON object must not panic the scrub.
        for mut payload in [json!("not an object"), json!([1, 2, 3]), Value::Null] {
            let before = payload.clone();
            sanitize_stored_payload(&mut payload);
            assert_eq!(payload, before);
        }
    }

    #[test]
    fn sanitize_stored_payload_leaves_objects_without_results_untouched() {
        // Indexing through IndexMut would insert a null `results` key here.
        let mut payload = json!({ "hardware": { "gpuModel": "RTX 2080" } });
        sanitize_stored_payload(&mut payload);
        assert_eq!(payload, json!({ "hardware": { "gpuModel": "RTX 2080" } }));
    }
}