# 0001: Centralize Server Lifecycle into ServerManager

Server process supervision, port binding, and VRAM management were previously coordinated procedurally across `commands.rs`, `server.rs`, `state.rs`, and out-of-band global maps (`VRAM_RESERVATIONS_MB`, `SERVER_HEALTH`). We decided to collapse these responsibilities into an encapsulated `ServerManager` module inside `src-tauri/src/server_manager/`, isolating WSL2 and native Windows process mechanics behind an internal `BackendDriver` seam and abstracting UI event emission behind an `EventEmitter` trait.

## Status

Accepted

## Context

In the previous design, `commands.rs` acted as a shallow imperative workflow coordinator. Starting a server required callers to acquire multiple distinct mutexes in `AppState`, allocate and hold TCP ports, modify configuration files, invoke low-level process launchers, register into global static maps, and spawn uncoordinated background polling tasks. This design caused race conditions on startup/shutdown, leaked out-of-band state, and made server lifecycle logic difficult to test without a running Tauri window and live hardware.

## Decision

1. **Encapsulated State**: `ServerManager` privately owns live process handles, port allocations, health states, and VRAM reservations. `AppState` references a single `Arc<ServerManager>`.
2. **BackendDriver Seam**: Divergent process mechanics between vLLM (WSL2 Linux bash script, daemons, PID files) and llama.cpp (Windows native processes) are concealed behind an internal `BackendDriver` trait.
3. **Atomic Lifecycle**: `ServerManager::start()` encapsulates port allocation, health verification, and automatic rollback unwinding on startup failure.
4. **Decoupled Event Emission**: Lifecycle and metric events are dispatched via an `EventEmitter` trait, allowing comprehensive headless lifecycle unit and integration tests.
5. **Strangler Migration**: Transition existing IPC commands in `commands.rs` incrementally to `ServerManager` without disrupting surrounding subsystems.
