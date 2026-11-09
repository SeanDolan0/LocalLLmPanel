# Stream C: WSL Process & Server Lifecycle (Wave 2)

**Ownership:** `src-tauri/src/wsl.rs`, `src-tauri/src/server.rs` (process/lifecycle parts), `src-tauri/src/commands.rs` (minimal additive edits), `src-tauri/src/hf.rs` (coordinated edit for C10 token via stdin).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| C1 | WSL activity counter leak | VERIFIED | COMPLETED | `src-tauri/src/wsl.rs` | `wsl::tests::test_wsl_exec_guard_lifecycle`, `test_wsl_exec_guard_normal_exit`, `test_wsl_exec_guard_try_wait_exit`, `test_wsl_exec_guard_kill`, `test_wsl_exec_guard_drop`, `test_wsl_exec_guard_spawn_failure` | RAII `WslExecGuard` in `WslChild::exec_guard` ensures activity counter releases on wait, try_wait, kill, and drop. |
| C2 | Discarded script-write errors | VERIFIED | COMPLETED | `src-tauri/src/wsl.rs` | `wsl::tests::test_wsl_exec_guard_spawn_failure` | Stdin write error in `spawn_script_via_stdin` propagates as `Err`, immediately kills spawned process via taskkill, reaps child, and removes temp script. |
| C3 | Dedicated distro install compatibility | VERIFIED | COMPLETED | `src-tauri/src/wsl.rs` | `wsl::tests::test_parse_wsl_version` | Parses WSL version across UTF-16LE, localized labels, and checks WSL >= 2.4.4 before using `--name` flag. |
| C4 | Localized WSL state parsing | VERIFIED | COMPLETED | `src-tauri/src/wsl.rs` | `wsl::tests::test_verbose_state_is_running_localized` | Localized header matching ("STATE", "ESTADO", "STATUT", etc.) and non-stopped value detection across German, French, Spanish, Japanese, Chinese, Russian, with exit code probe fallback. |
| C5 | GPU snapshots & memory detection | VERIFIED | COMPLETED | `src-tauri/src/wsl.rs`, `src-tauri/src/commands.rs` | `wsl::tests::test_parse_gpu_snapshots_multi_gpu`, `wsl::tests::test_detect_wsl_memory_nonexistent_distro_fallback`, `commands::tests::test_hardware_profile_falls_back_to_detected_specs` | Multi-GPU snapshot parsing (removed `| head -1`), preserves `Option<(u64, u64)>` in `try_detect_wsl_memory`, and uses host-estimated WSL memory fallback (50% physical RAM / 8GB max limit) via `llmfit_core` when distro is inaccessible. |
| C6 | Concurrent server starts overcommit VRAM | VERIFIED | COMPLETED | `src-tauri/src/server.rs` | `server::tests::test_vram_reservation_tracking` | Global `VRAM_RESERVATIONS` mutex registry tracks model VRAM allocation and active reservations to prevent concurrent startup overcommits. |
| C7 | Port allocation check-to-use race | VERIFIED | COMPLETED | `src-tauri/src/server.rs` | `server::tests::test_alloc_port_held_bind_and_hold` | `alloc_port_held` binds a TCP socket on `127.0.0.1` and holds it until right before server process launch, eliminating TOCTOU port hijacking. |
| C8 | Health vs liveness separation | VERIFIED | COMPLETED | `src-tauri/src/server.rs` | `server::tests::test_server_health_state_separation` | Separates process liveness (`is_alive` / PID checking) from HTTP endpoint health (`is_healthy` via `/health`), with granular server lifecycle states (`Starting`, `AliveNotReady`, `Ready`, `Degraded`, `Terminated`). |
| C9 | Hard-crash reconciliation | VERIFIED | COMPLETED | `src-tauri/src/server.rs` | `server::tests::test_reconcile_server_pidfile_alive_vs_dead` | PID-file management and hard-crash reconciliation detecting orphaned llama-server / WSL processes on startup, verifying PID liveness, and cleaning up stale state. |
| C10 | HF token visible in cmdline | VERIFIED | COMPLETED | `src-tauri/src/hf.rs` | `hf::tests::test_pull_hf_token_passed_via_stdin_script` | Passes HF authentication token via stdin to python/curl downloader, completely removing secrets from process command line arguments. |
| C11 | Pull cancel by exact PID | VERIFIED | COMPLETED | `src-tauri/src/hf.rs` | `hf::tests::test_pull_cancellation_map` | Replaced wildcard `pkill -f` with thread-safe PID registry and exact child process tree termination on Windows/WSL. |

## Quality Gate Checklist
- [x] `cargo test` passes (412 unittests + contract tests + integration tests pass, 0 failed)
- [x] `cargo clippy --workspace --all-targets` has no new warnings in owned files
- [x] `npm run build` passes cleanly (Vite + TypeScript)
