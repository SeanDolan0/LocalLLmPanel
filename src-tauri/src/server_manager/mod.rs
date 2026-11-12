pub mod ports;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

use crate::server_manager::ports::allocate_port_held;
pub use crate::server_manager::traits::{
    BackendDriver, ConfigStore, EventEmitter, SupervisedProcess,
};
pub use crate::server_manager::types::{
    CreateServerRequest, LifecyclePhase, LogChunk, ServerSnapshot,
};
use crate::state::{AppState, ServerDef, ServerStatus};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct LiveState {
    definitions: BTreeMap<String, ServerDef>,
    running: HashMap<String, Box<dyn SupervisedProcess>>,
    health_states: HashMap<String, (bool, bool)>, // (process_alive, http_healthy)
    errors: HashMap<String, String>,
    vram_reservations: HashMap<String, u64>,
}

/// Server Manager: The central authority governing server lifecycle,
/// port allocation, and hardware resource reservations. (GLOSSARY.md)
pub struct ServerManager {
    config_store: Option<Arc<dyn ConfigStore>>,
    drivers: HashMap<String, Arc<dyn BackendDriver>>,
    events: Option<Arc<dyn EventEmitter>>,
    state: Mutex<LiveState>,
}

impl Default for ServerManager {
    fn default() -> Self {
        Self::new_app()
    }
}

impl ServerManager {
    /// Constructor for production app use.
    pub fn new_app() -> Self {
        Self {
            config_store: None,
            drivers: HashMap::new(),
            events: None,
            state: Mutex::new(LiveState {
                definitions: BTreeMap::new(),
                running: HashMap::new(),
                health_states: HashMap::new(),
                errors: HashMap::new(),
                vram_reservations: HashMap::new(),
            }),
        }
    }

    /// Constructor accepting injected adapters for headless testing.
    pub fn new(
        config_store: Arc<dyn ConfigStore>,
        drivers: HashMap<String, Arc<dyn BackendDriver>>,
        events: Arc<dyn EventEmitter>,
    ) -> Self {
        Self {
            config_store: Some(config_store),
            drivers,
            events: Some(events),
            state: Mutex::new(LiveState {
                definitions: BTreeMap::new(),
                running: HashMap::new(),
                health_states: HashMap::new(),
                errors: HashMap::new(),
                vram_reservations: HashMap::new(),
            }),
        }
    }

    // ========================================================================
    // Standalone / Headless Testing Methods
    // ========================================================================

    /// Load existing definitions from persistent store into memory.
    pub fn init_from_store(&self) -> Result<(), String> {
        if let Some(ref store) = self.config_store {
            let loaded = store.load_servers()?;
            let mut st = self.state.lock().unwrap();
            for def in loaded {
                st.definitions.insert(def.id.clone(), def);
            }
        }
        Ok(())
    }

    /// List all server snapshots (Live and Stopped) from standalone state.
    pub fn list(&self) -> Vec<ServerSnapshot> {
        let st = self.state.lock().unwrap();
        st.definitions
            .values()
            .map(|def| {
                let is_running = st.running.contains_key(&def.id);
                let (alive, healthy) = st
                    .health_states
                    .get(&def.id)
                    .copied()
                    .unwrap_or((false, false));
                let error = st.errors.get(&def.id).cloned();
                let vram = st.vram_reservations.get(&def.id).copied();

                let phase = if is_running {
                    if healthy {
                        LifecyclePhase::Healthy {
                            pid: st.running.get(&def.id).and_then(|p| p.pid()).unwrap_or(0),
                            port: def.port,
                        }
                    } else {
                        LifecyclePhase::WarmingUp {
                            pid: st.running.get(&def.id).and_then(|p| p.pid()).unwrap_or(0),
                            port: def.port,
                            last_log: None,
                        }
                    }
                } else if let Some(ref err) = error {
                    LifecyclePhase::Failed { error: err.clone() }
                } else {
                    LifecyclePhase::Stopped
                };

                ServerSnapshot {
                    def: def.clone(),
                    status: if is_running {
                        "running".to_string()
                    } else {
                        "stopped".to_string()
                    },
                    phase,
                    port: def.port,
                    process_alive: alive,
                    http_healthy: healthy,
                    error,
                    metrics: None,
                    vram_reserved_mb: vram,
                }
            })
            .collect()
    }

    /// Get a snapshot for a specific server from standalone state.
    pub fn get(&self, id: &str) -> Option<ServerSnapshot> {
        self.list().into_iter().find(|s| s.def.id == id)
    }

    /// Create and persist a new Server Definition (standalone).
    pub async fn create(&self, req: CreateServerRequest) -> Result<ServerSnapshot, String> {
        let existing_ports: Vec<u16> = {
            let st = self.state.lock().unwrap();
            st.definitions.values().map(|d| d.port).collect()
        };

        let port = match req.port {
            Some(p) if p > 0 => p,
            _ => {
                let held = allocate_port_held(&existing_ports)?;
                held.release()
            }
        };

        let id = format!(
            "{}-{}",
            req.name
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric(), "-"),
            port
        );

        let def = ServerDef {
            backend: req.backend.unwrap_or_else(|| "vllm".to_string()),
            id: id.clone(),
            name: req.name,
            model_id: req.model_id,
            task: req.task.unwrap_or_else(|| "instruct".to_string()),
            port,
            gpu_mem_util: req.gpu_mem_util.unwrap_or(0.9),
            max_model_len: req.max_model_len,
            quant: req.quant.unwrap_or_else(|| "auto".to_string()),
            served_model_name: req.served_model_name,
            kv_cache_dtype: None,
            llamacpp_channel: "upstream".to_string(),
            enforce_eager: true,
            params_b: None,
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: false,
            model_path: req.model_path,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: None,
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: req.api_key,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 0,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: req.extra_args.unwrap_or_default(),
            env: BTreeMap::new(),
            speculative_model: None,
            speculative_model_path: None,
            speculative_num_tokens: None,
            speculative_enabled: false,
            speculative_draft_ngl: None,
            speculative_draft_n_min: None,
            speculative_draft_p_min: None,
            speculative_draft_p_split: None,
            speculative_type: None,
            speculative_draft_device: None,
            speculative_draft_sampling: None,
            speculative_draft_backend_sampling: None,
            speculative_draft_cache_type_k: None,
            speculative_draft_cache_type_v: None,
            speculative_draft_n_ubatch: None,
            speculative_draft_threads: None,
            speculative_draft_threads_batch: None,
            speculative_draft_cpu_moe: None,
            speculative_draft_n_cpu_moe: None,
            speculative_draft_prio: None,
            speculative_draft_poll: None,
        };

        if let Some(ref store) = self.config_store {
            store.save_server(&def)?;
        }

        {
            let mut st = self.state.lock().unwrap();
            st.definitions.insert(id.clone(), def);
        }

        self.get(&id)
            .ok_or_else(|| "Server created but not found in state".to_string())
    }

    /// Start a server in standalone mode.
    pub async fn start(&self, id: &str) -> Result<ServerSnapshot, String> {
        let (def, driver) = {
            let mut st = self.state.lock().unwrap();
            if st.running.contains_key(id) {
                return Err(format!("Server '{id}' is already running"));
            }
            let def = st
                .definitions
                .get(id)
                .cloned()
                .ok_or_else(|| format!("Server '{id}' not found"))?;
            let driver = self
                .drivers
                .get(&def.backend)
                .cloned()
                .ok_or_else(|| format!("No driver for backend '{}'", def.backend))?;

            st.errors.remove(id);
            st.vram_reservations.insert(id.to_string(), 4096);
            (def, driver)
        };

        let mut process = match driver.spawn(&def, def.port) {
            Ok(p) => p,
            Err(spawn_err) => {
                {
                    let mut st = self.state.lock().unwrap();
                    st.vram_reservations.remove(id);
                    st.errors.insert(id.to_string(), spawn_err.clone());
                }
                if let Some(snap) = self.get(id) {
                    if let Some(ref ev) = self.events {
                        ev.emit_status(&snap);
                    }
                }
                return Err(spawn_err);
            }
        };

        let healthy = driver.check_health(def.port);
        if !healthy {
            let _ = process.terminate(Duration::from_secs(5));
            let err = format!("Server '{}' failed health check on port {}", id, def.port);
            {
                let mut st = self.state.lock().unwrap();
                st.vram_reservations.remove(id);
                st.errors.insert(id.to_string(), err.clone());
            }
            if let Some(snap) = self.get(id) {
                if let Some(ref ev) = self.events {
                    ev.emit_status(&snap);
                }
            }
            return Err(err);
        }

        {
            let mut st = self.state.lock().unwrap();
            st.health_states.insert(id.to_string(), (true, true));
            st.running.insert(id.to_string(), process);
            if let Some(def_mut) = st.definitions.get_mut(id) {
                def_mut.was_running = true;
            }
        }

        let snap = self.get(id).unwrap();
        if let Some(ref ev) = self.events {
            ev.emit_status(&snap);
        }
        Ok(snap)
    }

    /// Stop a running server in standalone mode.
    pub async fn stop(&self, id: &str) -> Result<ServerSnapshot, String> {
        let process = {
            let mut st = self.state.lock().unwrap();
            st.vram_reservations.remove(id);
            st.health_states.insert(id.to_string(), (false, false));
            if let Some(def_mut) = st.definitions.get_mut(id) {
                def_mut.was_running = false;
            }
            st.running.remove(id)
        };

        if let Some(mut proc) = process {
            proc.terminate(Duration::from_secs(10))?;
        }

        let snap = self
            .get(id)
            .ok_or_else(|| format!("Server '{id}' not found"))?;
        if let Some(ref ev) = self.events {
            ev.emit_status(&snap);
        }
        Ok(snap)
    }

    /// Restart a server in standalone mode.
    pub async fn restart(&self, id: &str) -> Result<ServerSnapshot, String> {
        let _ = self.stop(id).await;
        self.start(id).await
    }

    /// Delete a server definition in standalone mode.
    pub async fn delete(&self, id: &str) -> Result<(), String> {
        {
            let st = self.state.lock().unwrap();
            if st.running.contains_key(id) {
                return Err(format!(
                    "Cannot delete running server '{id}'; stop it first"
                ));
            }
        }

        if let Some(ref store) = self.config_store {
            store.remove_server(id)?;
        }

        {
            let mut st = self.state.lock().unwrap();
            st.definitions.remove(id);
            st.health_states.remove(id);
            st.errors.remove(id);
            st.vram_reservations.remove(id);
        }

        Ok(())
    }

    // ========================================================================
    // Strangler Integration Methods (Delegates to AppState & server.rs)
    // ========================================================================

    /// List all servers formatted as backward-compatible ServerListRow for Tauri commands.
    pub fn list_server_rows(&self, state: &Arc<AppState>) -> Vec<crate::commands::ServerListRow> {
        let base_servers = crate::server::list_servers(state);
        let servers = state.servers.lock().unwrap();
        base_servers
            .into_iter()
            .map(|(def, status, error, metrics)| {
                let (process_alive, http_healthy) = match crate::server::get_server_health(&def.id)
                {
                    Some(h) => (Some(h.process_alive), Some(h.http_healthy)),
                    None => (None, None),
                };
                let runtime_memory = servers.get(&def.id).map(|ls| {
                    let mut rm = ls.runtime_memory.lock().unwrap().clone();
                    #[cfg(windows)]
                    if let Some(ref child) = ls.native_child {
                        if let Some(ram) = crate::runtime_memory::process_working_set_mb(child.pid()) {
                            rm.process_ram_mb = Some(ram);
                        }
                    }
                    rm
                });
                crate::commands::ServerListRow {
                    def,
                    status: status.label().to_string(),
                    error,
                    metrics,
                    runtime_memory,
                    process_alive,
                    http_healthy,
                }
            })
            .collect()
    }

    /// List all servers as modern rich ServerSnapshots.
    pub fn list_snapshots(&self, state: &Arc<AppState>) -> Vec<ServerSnapshot> {
        let defs = state.config().servers.clone();
        let servers = state.servers.lock().unwrap();
        defs.into_iter()
            .map(|def| {
                let id = &def.id;
                let vram = crate::server::reserved_vram_mb(id);
                let (alive, healthy) = match crate::server::get_server_health(id) {
                    Some(h) => (h.process_alive, h.http_healthy),
                    None => (false, false),
                };

                let (status, phase, error) = if let Some(ls) = servers.get(id) {
                    let pid = ls.wsl_pid.unwrap_or(0);
                    let phase = match ls.status {
                        ServerStatus::Running => LifecyclePhase::Healthy {
                            pid,
                            port: def.port,
                        },
                        ServerStatus::Starting => LifecyclePhase::WarmingUp {
                            pid,
                            port: def.port,
                            last_log: None,
                        },
                        ServerStatus::Error => LifecyclePhase::Failed {
                            error: ls.error.clone().unwrap_or_else(|| "Unknown error".into()),
                        },
                        ServerStatus::Stopped => LifecyclePhase::Stopped,
                    };
                    (ls.status.label().to_string(), phase, ls.error.clone())
                } else {
                    ("stopped".to_string(), LifecyclePhase::Stopped, None)
                };

                ServerSnapshot {
                    def: crate::state::redact_server_def(&def),
                    status,
                    phase,
                    port: def.port,
                    process_alive: alive,
                    http_healthy: healthy,
                    error,
                    metrics: None,
                    vram_reserved_mb: vram,
                }
            })
            .collect()
    }

    /// Start a server through the unified ServerManager interface.
    pub fn start_server(
        &self,
        state: &Arc<AppState>,
        app: Option<&tauri::AppHandle>,
        id: &str,
    ) -> Result<ServerSnapshot, String> {
        crate::server::start_server(state, app, id).map_err(|e| e.to_string())?;
        self.list_snapshots(state)
            .into_iter()
            .find(|s| s.def.id == id)
            .ok_or_else(|| format!("Server '{id}' started but not found in snapshot list"))
    }

    /// Stop a server through the unified ServerManager interface.
    pub fn stop_server(
        &self,
        state: &Arc<AppState>,
        app: Option<&tauri::AppHandle>,
        id: &str,
    ) -> Result<ServerSnapshot, String> {
        crate::server::stop_server(state, app, id).map_err(|e| e.to_string())?;
        self.list_snapshots(state)
            .into_iter()
            .find(|s| s.def.id == id)
            .ok_or_else(|| format!("Server '{id}' stopped but not found in snapshot list"))
    }

    /// Restart a server through the unified ServerManager interface.
    pub fn restart_server(
        &self,
        state: &Arc<AppState>,
        app: Option<&tauri::AppHandle>,
        id: &str,
    ) -> Result<ServerSnapshot, String> {
        crate::server::restart_server(state, app, id).map_err(|e| e.to_string())?;
        self.list_snapshots(state)
            .into_iter()
            .find(|s| s.def.id == id)
            .ok_or_else(|| format!("Server '{id}' restarted but not found in snapshot list"))
    }

    /// Delete a server definition atomically: stopping, unregistering metrics, and persisting.
    pub fn delete_server(&self, state: &Arc<AppState>, id: &str) -> Result<(), String> {
        let was_running = {
            let servers = state.servers.lock().unwrap();
            servers.contains_key(id)
        };
        if was_running {
            let _ = crate::server::stop_server(state, None, id);
        }
        let mut cfg = state.config.lock().unwrap();
        let mut next = cfg.clone();
        next.servers.retain(|s| s.id != id);
        next.save().map_err(|e| e.to_string())?;
        *cfg = next;
        let mut m = state.server_metrics.lock().unwrap();
        m.remove(id);
        crate::server::release_vram(id);
        crate::server::remove_server_health(id);
        Ok(())
    }

    /// Clean shutdown of all running servers on application exit.
    pub fn shutdown_all(&self, state: &Arc<AppState>) -> usize {
        crate::server::shutdown_all_servers(state)
    }
}
