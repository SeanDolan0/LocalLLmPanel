# Cross-Stream Requests

This file tracks requests from sub-agents needing modifications in files owned by another stream. The orchestrator routes or merges these changes safely across waves.

| Request ID | From Stream | Target File / Area | Target Stream / Wave | Description | Status |
|---|---|---|---|---|---|
| XREQ-01 | Stream A (A12) | `commands.rs` / `wsl.rs` | Stream C (C11) | Exact child PID tracking for WSL download cancel (`pkill` wildcard fix) | Queued for Wave 2 |
| XREQ-02 | Stream B (B3) | `wsl.rs` | Stream C (C5) | Multi-GPU telemetry (`head -1` -> all GPUs) | Queued for Wave 2 |
| XREQ-03 | Stream B (B5) | `wsl.rs` | Stream C (C5) | Preserve Option for WSL memory detection (remove synthetic 16/12GB default) | Queued for Wave 2 |
| XREQ-04 | Stream D (D4) | `provision.rs` / cleanup | Stream F (F6) / Stream C | Sudoers rule scoping and removal | RESOLVED (Wave 3) |
| XREQ-05 | Stream A (A5) / B (B8) | Frontend UI | Stream G (G6) | Disk space preflight UI presentation | Queued for Wave 3 |
## Detailed Request Specifications

### XREQ-02 (Stream B -> Stream C): Multi-GPU Telemetry in WSL
- **Target:** `src-tauri/src/wsl.rs` (owned by Stream C)
- **Problem:** `wsl.rs` uses `nvidia-smi ... | head -1`, returning only the primary GPU and dropping all secondary GPUs.
- **Request:**
  1. Remove `| head -1` and query `nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,utilization.gpu --format=csv,noheader,nounits`.
  2. Parse each line independently (refer to `llamacpp_install::parse_nvidia_gpus`).
  3. Expose enumerated GPU devices list and apply minimum compute capability policy across selected devices.

### XREQ-03 (Stream B -> Stream C): Preserve Option for WSL Memory Detection
- **Target:** `src-tauri/src/wsl.rs` and `src-tauri/src/commands.rs` (owned by Stream C)
- **Problem:** `detect_wsl_memory()` falls back to synthetic `(16_384, 12_288)` when WSL memory is not detected or when WSL is stopped.
- **Request:**
  1. In `commands.rs` / `wsl.rs`, use `try_detect_wsl_memory(distro)` which returns `Option<(u64, u64)>`.
  2. When WSL is stopped, call `crate::estimate::estimate_wsl_memory_from_host(windows_ram_mb, wslconfig_text)` provided in Stream B. This respects `.wslconfig` and 50% Windows physical RAM.
  3. If memory is genuinely unknown, preserve `None` (or set `ram_total_mb: 0` in `HardwareProfile`), which `fit.rs` now handles by marking RAM-overflow fit uncertain/disabled.
