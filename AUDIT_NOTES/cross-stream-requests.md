# Cross-Stream Requests

This file tracks requests from sub-agents needing modifications in files owned by another stream. The orchestrator routes or merges these changes safely across waves.

| Request ID | From Stream | Target File / Area | Target Stream / Wave | Description | Status |
|---|---|---|---|---|---|
| XREQ-01 | Stream A (A12) | `commands.rs` / `wsl.rs` | Stream C (C11) | Exact child PID tracking for WSL download cancel (`pkill` wildcard fix) | Queued for Wave 2 |
| XREQ-02 | Stream B (B3) | `wsl.rs` | Stream C (C5) | Multi-GPU telemetry (`head -1` -> all GPUs) | Queued for Wave 2 |
| XREQ-03 | Stream B (B5) | `wsl.rs` | Stream C (C5) | Preserve Option for WSL memory detection (remove synthetic 16/12GB default) | Queued for Wave 2 |
| XREQ-04 | Stream D (D4) | `provision.rs` / cleanup | Stream F (F6) / Stream C | Sudoers rule scoping and removal | Queued for Wave 3 |
| XREQ-05 | Stream A (A5) / B (B8) | Frontend UI | Stream G (G6) | Disk space preflight UI presentation | Queued for Wave 3 |
