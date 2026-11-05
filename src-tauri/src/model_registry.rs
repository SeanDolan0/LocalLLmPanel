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
    data: Mutex<ModelRegistryData>,
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

        for entry in guard.models.values() {
            let entry_id = entry.model_id.to_ascii_lowercase();
            let entry_repo = entry.repo_id.to_ascii_lowercase();

            let id_matches = entry_id == lower
                || entry_repo == lower
                || entry_id.ends_with(&format!("/{lower}"))
                || entry_repo.ends_with(&format!("/{lower}"));

            if id_matches {
                if let Some(q) = quant {
                    if entry.quant.eq_ignore_ascii_case(q) {
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

    pub fn unregister(&self, repo_id: &str, quant: &str) -> Result<(), String> {
        let key = format!("{}:{}", repo_id, quant).to_ascii_lowercase();
        {
            let mut guard = self.data.lock().map_err(|e| e.to_string())?;
            guard.models.remove(&key);
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

        assert!(reg.is_installed("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q4_K_M")));
        assert!(reg.is_installed("Qwen/Qwen2.5-Coder-7B-Instruct", Some("Q4_K_M")));
        assert!(!reg.is_installed("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q8_0")));
    }
}
