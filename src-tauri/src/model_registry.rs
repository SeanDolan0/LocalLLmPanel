//! Local model registry: durable inventory of discovered, downloading,
//! and installed model artifacts (both native GGUF and WSL vLLM models).

use crate::state::CONFIG_DIR_NAME;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

pub const REGISTRY_FILE_NAME: &str = "model_registry.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegistryModelEntry {
    pub model_id: String,
    pub repo_id: String,
    pub backend: String, // "llamacpp" | "vllm"
    pub quant: String,   // "Q4_K_M", "AWQ", "FP16", etc.
    pub files: Vec<String>,
    pub size_bytes: u64,
    pub installed_at: u64,
    pub local_path: String,
    #[serde(default)]
    pub is_split: bool,
    #[serde(default)]
    pub companion_mmproj: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistryData {
    pub models: HashMap<String, RegistryModelEntry>,
}

pub struct ModelRegistry {
    pub data: Mutex<ModelRegistryData>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        let registry = Self {
            data: Mutex::new(ModelRegistryData::default()),
        };
        registry.load_from_disk();
        registry
    }

    pub fn file_path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR_NAME)
            .join(REGISTRY_FILE_NAME)
    }

    pub fn load_from_disk(&self) {
        let path = Self::file_path();
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(data) = serde_json::from_str::<ModelRegistryData>(&text) {
                if let Ok(mut guard) = self.data.lock() {
                    *guard = data;
                }
            }
        }
    }

    pub fn save_to_disk(&self) -> Result<(), String> {
        let path = Self::file_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let guard = self.data.lock().map_err(|e| e.to_string())?;
        let text = serde_json::to_string_pretty(&*guard).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("write error: {e}"))?;
        let _ = std::fs::remove_file(&path);
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename error: {e}"))?;
        Ok(())
    }

    pub fn is_installed(&self, repo_or_model_id: &str, quant: Option<&str>) -> bool {
        let guard = match self.data.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let lower = repo_or_model_id.to_ascii_lowercase();
        let lower_norm = lower.replace("__", "/");

        for entry in guard.models.values() {
            let entry_id = entry.model_id.to_ascii_lowercase();
            let entry_repo = entry.repo_id.to_ascii_lowercase();
            let entry_id_norm = entry_id.replace("__", "/");
            let entry_repo_norm = entry_repo.replace("__", "/");

            let id_matches = entry_id == lower
                || entry_repo == lower
                || entry_id_norm == lower_norm
                || entry_repo_norm == lower_norm
                || entry_id.ends_with(&format!("/{lower}"))
                || entry_repo.ends_with(&format!("/{lower}"))
                || entry_id_norm.ends_with(&format!("/{lower_norm}"))
                || entry_repo_norm.ends_with(&format!("/{lower_norm}"));

            if id_matches {
                let entry_quant_lower = entry.quant.to_ascii_lowercase();
                let entry_is_generic = entry_quant_lower.is_empty() || entry_quant_lower == "gguf";

                if let Some(q) = quant {
                    let q_lower = q.to_ascii_lowercase();
                    let q_is_generic = q_lower.is_empty() || q_lower == "gguf";

                    if q_is_generic || entry_is_generic || entry_quant_lower == q_lower {
                        return true;
                    }
                } else {
                    return true;
                }
            }
        }
        false
    }

    pub fn register(&self, entry: RegistryModelEntry) -> Result<(), String> {
        let key = format!("{}:{}", entry.repo_id, entry.quant).to_ascii_lowercase();
        {
            let mut guard = self.data.lock().map_err(|e| e.to_string())?;
            guard.models.insert(key, entry);
        }
        self.save_to_disk()
    }

    pub fn unregister(&self, repo_or_model_id: &str, quant: &str) -> Result<(), String> {
        let direct_key = format!("{}:{}", repo_or_model_id, quant).to_ascii_lowercase();
        let target_quant = quant.to_ascii_lowercase();
        let target_id = repo_or_model_id.to_ascii_lowercase();
        let target_norm = target_id.replace("__", "/");
        {
            let mut guard = self.data.lock().map_err(|e| e.to_string())?;
            guard.models.remove(&direct_key);
            guard.models.retain(|_, entry| {
                let e_quant = entry.quant.to_ascii_lowercase();
                let e_repo = entry.repo_id.to_ascii_lowercase();
                let e_model = entry.model_id.to_ascii_lowercase();
                let e_repo_norm = e_repo.replace("__", "/");
                let e_model_norm = e_model.replace("__", "/");

                let quant_matches = if target_quant.is_empty() || target_quant == "gguf" {
                    e_quant.is_empty() || e_quant == "gguf"
                } else {
                    e_quant == target_quant
                };

                let id_matches = e_repo == target_id
                    || e_model == target_id
                    || e_repo_norm == target_norm
                    || e_model_norm == target_norm
                    || e_repo_norm.ends_with(&format!("/{target_norm}"))
                    || e_model_norm.ends_with(&format!("/{target_norm}"));

                !(id_matches && quant_matches)
            });
        }
        self.save_to_disk()
    }

    pub fn list(&self) -> Vec<RegistryModelEntry> {
        self.data
            .lock()
            .map(|g| g.models.values().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_registration_and_query() {
        let reg = ModelRegistry {
            data: Mutex::new(ModelRegistryData::default()),
        };

        let entry = RegistryModelEntry {
            model_id: "Qwen/Qwen2.5-Coder-7B-Instruct".into(),
            repo_id: "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF".into(),
            backend: "llamacpp".into(),
            quant: "Q4_K_M".into(),
            files: vec!["Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".into()],
            size_bytes: 4_500_000_000,
            installed_at: 123456789,
            local_path: "C:/gguf/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".into(),
            is_split: false,
            companion_mmproj: None,
        };

        reg.register(entry).unwrap();

        // Q4_K_M installed => another Q4_K_M pull is blocked as a duplicate (is_installed == true)
        assert!(reg.is_installed("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q4_K_M")));
        assert!(reg.is_installed("Qwen/Qwen2.5-Coder-7B-Instruct", Some("Q4_K_M")));

        // Q4_K_M installed => Q8_0 is still downloadable (is_installed == false)
        assert!(!reg.is_installed("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q8_0")));
        assert!(!reg.is_installed("Qwen/Qwen2.5-Coder-7B-Instruct", Some("Q8_0")));
    }

    #[test]
    fn test_two_quantizations_coexist_and_isolated_deletion() {
        let reg = ModelRegistry {
            data: Mutex::new(ModelRegistryData::default()),
        };

        let repo = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF";
        let model = "Qwen/Qwen2.5-Coder-7B-Instruct";

        let entry_q4 = RegistryModelEntry {
            model_id: model.into(),
            repo_id: repo.into(),
            backend: "llamacpp".into(),
            quant: "Q4_K_M".into(),
            files: vec!["Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".into()],
            size_bytes: 4_500_000_000,
            installed_at: 1000,
            local_path: "C:/gguf/Q4_K_M.gguf".into(),
            is_split: false,
            companion_mmproj: None,
        };

        let entry_q8 = RegistryModelEntry {
            model_id: model.into(),
            repo_id: repo.into(),
            backend: "llamacpp".into(),
            quant: "Q8_0".into(),
            files: vec!["Qwen2.5-Coder-7B-Instruct-Q8_0.gguf".into()],
            size_bytes: 8_000_000_000,
            installed_at: 2000,
            local_path: "C:/gguf/Q8_0.gguf".into(),
            is_split: false,
            companion_mmproj: None,
        };

        // Two quantizations of the same repo coexist in the registry
        reg.register(entry_q4).unwrap();
        reg.register(entry_q8).unwrap();

        assert_eq!(reg.list().len(), 2);
        assert!(reg.is_installed(repo, Some("Q4_K_M")));
        assert!(reg.is_installed(repo, Some("Q8_0")));
        assert!(!reg.is_installed(repo, Some("Q6_K")));

        // Deleting one quant leaves the other installed
        reg.unregister(repo, "Q4_K_M").unwrap();

        assert_eq!(reg.list().len(), 1);
        assert!(!reg.is_installed(repo, Some("Q4_K_M")));
        assert!(reg.is_installed(repo, Some("Q8_0")));
    }

    #[test]
    fn test_generic_quant_compatibility() {
        let reg = ModelRegistry {
            data: Mutex::new(ModelRegistryData::default()),
        };

        let entry_generic = RegistryModelEntry {
            model_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
            repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
            backend: "llamacpp".into(),
            quant: "GGUF".into(),
            files: vec!["tinyllama.gguf".into()],
            size_bytes: 1_000_000_000,
            installed_at: 1000,
            local_path: "C:/gguf/tinyllama.gguf".into(),
            is_split: false,
            companion_mmproj: None,
        };

        reg.register(entry_generic).unwrap();

        // Generic quant should be recognized for generic query or specific query fallback
        assert!(reg.is_installed("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", None));
        assert!(reg.is_installed("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", Some("GGUF")));
        assert!(reg.is_installed("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", Some("Q4_K_M")));
    }
}
