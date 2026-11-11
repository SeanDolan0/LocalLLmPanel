#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use crate::server_manager::traits::{BackendDriver, ConfigStore, EventEmitter, SupervisedProcess};
    use crate::server_manager::types::{CreateServerRequest, LifecyclePhase, ServerSnapshot};
    use crate::server_manager::ServerManager;
    use crate::state::ServerDef;

    // Test In-Memory Config Store
    struct InMemoryConfigStore {
        servers: Mutex<Vec<ServerDef>>,
    }

    impl InMemoryConfigStore {
        fn new() -> Self {
            Self {
                servers: Mutex::new(Vec::new()),
            }
        }
    }

    impl ConfigStore for InMemoryConfigStore {
        fn load_servers(&self) -> Result<Vec<ServerDef>, String> {
            Ok(self.servers.lock().unwrap().clone())
        }

        fn save_server(&self, def: &ServerDef) -> Result<(), String> {
            let mut list = self.servers.lock().unwrap();
            list.retain(|s| s.id != def.id);
            list.push(def.clone());
            Ok(())
        }

        fn remove_server(&self, id: &str) -> Result<(), String> {
            let mut list = self.servers.lock().unwrap();
            list.retain(|s| s.id != id);
            Ok(())
        }
    }

    // Test Event Emitter that records events
    struct RecordingEventEmitter {
        events: Mutex<Vec<ServerSnapshot>>,
    }
    impl RecordingEventEmitter {
        fn new() -> Self {
            Self { events: Mutex::new(Vec::new()) }
        }
    }
    impl EventEmitter for RecordingEventEmitter {
        fn emit_status(&self, snapshot: &ServerSnapshot) {
            self.events.lock().unwrap().push(snapshot.clone());
        }
        fn emit_log_line(&self, _server_id: &str, _line: &str) {}
    }

    // Test Mock Process
    struct MockProcess {
        pid: u32,
        alive: AtomicBool,
    }
    impl SupervisedProcess for MockProcess {
        fn pid(&self) -> Option<u32> {
            Some(self.pid)
        }
        fn is_alive(&self) -> bool {
            self.alive.load(Ordering::SeqCst)
        }
        fn terminate(&mut self, _timeout: Duration) -> Result<(), String> {
            self.alive.store(false, Ordering::SeqCst);
            Ok(())
        }
        fn read_new_logs(&mut self) -> Vec<String> {
            vec!["mock log line".into()]
        }
    }

    // Test Mock Backend Driver
    struct MockBackendDriver {
        should_fail_spawn: bool,
        should_fail_health: bool,
    }
    impl BackendDriver for MockBackendDriver {
        fn backend_kind(&self) -> &'static str {
            "mock"
        }
        fn spawn(&self, _def: &ServerDef, _port: u16) -> Result<Box<dyn SupervisedProcess>, String> {
            if self.should_fail_spawn {
                return Err("Simulated process spawn failure".into());
            }
            Ok(Box::new(MockProcess {
                pid: 12345,
                alive: AtomicBool::new(true),
            }))
        }
        fn check_health(&self, _port: u16) -> bool {
            !self.should_fail_health
        }
    }

    #[tokio::test]
    async fn test_create_server_auto_allocates_port_and_persists() {
        let store = Arc::new(InMemoryConfigStore::new());
        let events = Arc::new(RecordingEventEmitter::new());
        let mut drivers: HashMap<String, Arc<dyn BackendDriver>> = HashMap::new();
        drivers.insert("mock".to_string(), Arc::new(MockBackendDriver { should_fail_spawn: false, should_fail_health: false }));

        let manager = ServerManager::new(store.clone(), drivers, events);

        let req = CreateServerRequest {
            name: "Test Qwen".to_string(),
            model_id: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            backend: Some("mock".to_string()),
            task: None,
            port: None, // Auto-allocate
            gpu_mem_util: None,
            max_model_len: None,
            quant: None,
            served_model_name: None,
            model_path: None,
            api_key: None,
            extra_args: None,
        };

        let snapshot = manager.create(req).await.expect("create server must succeed");
        assert_eq!(snapshot.def.name, "Test Qwen");
        assert!(snapshot.def.port >= 8000, "Port must be auto-allocated >= 8000");
        assert_eq!(snapshot.phase, LifecyclePhase::Stopped);

        // Verify persisted in store
        let persisted = store.load_servers().unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].id, snapshot.def.id);
    }

    #[tokio::test]
    async fn test_delete_server_removes_from_store() {
        let store = Arc::new(InMemoryConfigStore::new());
        let events = Arc::new(RecordingEventEmitter::new());
        let drivers: HashMap<String, Arc<dyn BackendDriver>> = HashMap::new();

        let manager = ServerManager::new(store.clone(), drivers, events);

        let req = CreateServerRequest {
            name: "To Delete".to_string(),
            model_id: "test/model".to_string(),
            backend: None,
            task: None,
            port: Some(8123),
            gpu_mem_util: None,
            max_model_len: None,
            quant: None,
            served_model_name: None,
            model_path: None,
            api_key: None,
            extra_args: None,
        };

        let snapshot = manager.create(req).await.unwrap();
        assert_eq!(manager.list().len(), 1);

        manager.delete(&snapshot.def.id).await.expect("delete must succeed");
        assert_eq!(manager.list().len(), 0);

        let persisted = store.load_servers().unwrap();
        assert_eq!(persisted.len(), 0);
    }

    #[tokio::test]
    async fn test_start_and_stop_lifecycle_success() {
        let store = Arc::new(InMemoryConfigStore::new());
        let events = Arc::new(RecordingEventEmitter::new());
        let mut drivers: HashMap<String, Arc<dyn BackendDriver>> = HashMap::new();
        drivers.insert("mock".to_string(), Arc::new(MockBackendDriver { should_fail_spawn: false, should_fail_health: false }));

        let manager = ServerManager::new(store.clone(), drivers, events.clone());

        let req = CreateServerRequest {
            name: "Lifecycle Model".to_string(),
            model_id: "test/model".to_string(),
            backend: Some("mock".to_string()),
            task: None,
            port: Some(8200),
            gpu_mem_util: Some(0.5),
            max_model_len: None,
            quant: None,
            served_model_name: None,
            model_path: None,
            api_key: None,
            extra_args: None,
        };

        let created = manager.create(req).await.unwrap();
        let server_id = &created.def.id;

        // 1. Start server
        let running_snap = manager.start(server_id).await.expect("start must succeed");
        assert_eq!(running_snap.status, "running");
        assert!(matches!(running_snap.phase, LifecyclePhase::Healthy { .. }));
        assert_eq!(running_snap.process_alive, true);
        assert_eq!(running_snap.http_healthy, true);

        // 2. Stop server
        let stopped_snap = manager.stop(server_id).await.expect("stop must succeed");
        assert_eq!(stopped_snap.status, "stopped");
        assert_eq!(stopped_snap.phase, LifecyclePhase::Stopped);
        assert_eq!(stopped_snap.process_alive, false);
    }

    #[tokio::test]
    async fn test_start_failure_triggers_atomic_rollback() {
        let store = Arc::new(InMemoryConfigStore::new());
        let events = Arc::new(RecordingEventEmitter::new());
        let mut drivers: HashMap<String, Arc<dyn BackendDriver>> = HashMap::new();
        // Driver configured to fail spawn
        drivers.insert("mock".to_string(), Arc::new(MockBackendDriver { should_fail_spawn: true, should_fail_health: false }));

        let manager = ServerManager::new(store.clone(), drivers, events.clone());

        let req = CreateServerRequest {
            name: "Failing Model".to_string(),
            model_id: "test/failing".to_string(),
            backend: Some("mock".to_string()),
            task: None,
            port: Some(8205),
            gpu_mem_util: None,
            max_model_len: None,
            quant: None,
            served_model_name: None,
            model_path: None,
            api_key: None,
            extra_args: None,
        };

        let created = manager.create(req).await.unwrap();
        let server_id = &created.def.id;

        // Start should fail and cleanly rollback
        let err = manager.start(server_id).await;
        assert!(err.is_err(), "Start must return error when driver fails");

        // Server state must be Stopped, not corrupted or running
        let snap = manager.get(server_id).expect("snapshot must exist");
        assert_eq!(snap.status, "stopped");
        assert_eq!(snap.process_alive, false);
        assert!(snap.error.is_some(), "Snapshot must record failure error hint");
    }
}
