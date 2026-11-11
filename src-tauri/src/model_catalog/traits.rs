use std::path::Path;
use crate::model_catalog::types::{CatalogModel, CatalogModelDetails, DiscoveryCategory, InstalledModel};

/// Remote data source abstraction for querying Hugging Face API and model registries.
pub trait RemoteModelSource: Send + Sync {
    /// Fetch candidates for a discovery category feed.
    fn fetch_category_feed(&self, category: DiscoveryCategory) -> Result<Vec<CatalogModel>, String>;

    /// Search remote models matching query string.
    fn search_models(&self, query: &str) -> Result<Vec<CatalogModel>, String>;

    /// Hydrate deep model details including repo tree files and architecture config.
    fn fetch_model_details(&self, model_id: &str) -> Result<CatalogModelDetails, String>;
}

/// Local inventory source abstraction for scanning downloaded models and GGUF files.
pub trait LocalInventorySource: Send + Sync {
    /// Scan all locally available models across WSL cache and native Windows folders.
    fn scan_installed_models(&self) -> Result<Vec<InstalledModel>, String>;

    /// Check if target directory has been modified since last known timestamp.
    fn directory_mtime_ms(&self, path: &Path) -> Option<u64>;
}
