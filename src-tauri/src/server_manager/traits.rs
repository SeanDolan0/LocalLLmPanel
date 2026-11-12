use crate::server_manager::types::ServerSnapshot;
use crate::state::ServerDef;
use std::time::Duration;

/// Seam 1: Configuration Store
/// Persists and loads server definitions and their operational flags.
pub trait ConfigStore: Send + Sync {
    fn load_servers(&self) -> Result<Vec<ServerDef>, String>;
    fn save_server(&self, def: &ServerDef) -> Result<(), String>;
    fn remove_server(&self, id: &str) -> Result<(), String>;
}

/// Seam 2: Backend Driver
/// Isolates OS-specific process mechanics between Linux (WSL2) and native Windows.
pub trait BackendDriver: Send + Sync {
    fn backend_kind(&self) -> &'static str;
    fn spawn(&self, def: &ServerDef, port: u16) -> Result<Box<dyn SupervisedProcess>, String>;
    fn check_health(&self, port: u16) -> bool;
}

/// A running supervised child process.
pub trait SupervisedProcess: Send + Sync {
    fn pid(&self) -> Option<u32>;
    fn is_alive(&self) -> bool;
    fn terminate(&mut self, timeout: Duration) -> Result<(), String>;
    fn read_new_logs(&mut self) -> Vec<String>;
}

/// Seam 3: Event Emitter
/// Broadcasts lifecycle state updates and metrics to presentation layers.
pub trait EventEmitter: Send + Sync {
    fn emit_status(&self, snapshot: &ServerSnapshot);
    fn emit_log_line(&self, server_id: &str, line: &str);
}
