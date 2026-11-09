# Stream C: WSL Process & Server Lifecycle (Wave 2)

**Ownership:** `src-tauri/src/wsl.rs`, `src-tauri/src/server.rs` (process/lifecycle parts), `src-tauri/src/commands.rs` (minimal additive edits).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| C1 | WSL activity counter leak | PENDING | PENDING | | | RAII guard for WslChild |
| C2 | Discarded script-write errors | PENDING | PENDING | | | Propagate errors, reap child |
| C3 | Dedicated distro install compatibility | PENDING | PENDING | | | Check WSL >= 2.4.4 for --name |
| C4 | Localized WSL state parsing | PENDING | PENDING | | | Probe via exit status, not English 'Running' |
| C5 | GPU snapshots & memory detection | PENDING | PENDING | | | Return all GPUs, preserve Option memory |
| C6 | Concurrent server starts overcommit VRAM | PENDING | PENDING | | | Global GPU memory reservation |
| C7 | Port allocation check-to-use race | PENDING | PENDING | | | Bind-and-hold or launch retry |
| C8 | Health vs liveness separation | PENDING | PENDING | | | Separate process alive from HTTP healthy |
| C9 | Hard-crash reconciliation | PENDING | PENDING | | | Orphan recovery test & handling |
| C10 | HF token visible in cmdline | PENDING | PENDING | | | Pass via stdin |
| C11 | Pull cancel by exact PID | PENDING | PENDING | | | Fix pkill regex issue |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
