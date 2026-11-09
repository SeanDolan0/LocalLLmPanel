# Stream G: First-Run UX & Windows Trust (Wave 3)

**Ownership:** frontend pages/components not owned elsewhere (`src/pages/*`, `src/components/*`), diagnostic bundle export and test runners in frontend, and backend helpers in `src-tauri/src/commands.rs` and `src-tauri/src/diagnostics.rs`.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| G1 | Unified first-run diagnostic | YES | COMPLETED | `src-tauri/src/diagnostics.rs`, `src/components/FirstRunDiagnosticCard.tsx`, `src/pages/Dashboard.tsx` | `test_diagnostic_ordering_and_steps` | Ordered 8-step diagnostic check (Windows -> WSL -> WSL2 -> NVIDIA -> WSL GPU -> CUDA -> Disk -> Backend). Beginners never see success when GPU is unusable. |
| G2 | Distinguish error sources in UI | YES | COMPLETED | `src/components/ErrorAlert.tsx`, `src/pages/Dashboard.tsx`, `src/pages/Servers.tsx`, `src/pages/Search.tsx`, `src/pages/Library.tsx` | Component build and contract compliance | Dedicated `ErrorAlert` component explicitly categorizing Model Downloads, Runtime Provisioning, and Server Execution with tailored styling and contextual repair hints. |
| G3 | Windows 10 support clarity | YES | COMPLETED | `src-tauri/src/diagnostics.rs`, `src/components/FirstRunDiagnosticCard.tsx` | `test_windows_build_support_threshold` | Explicit build check (Build 19041+ / Version 2004+ required for WSL2 CUDA vs Windows 11). Explicit status, labels, and update guidance in UI. |
| G4 | SmartScreen & installer trust | YES | COMPLETED | `src/components/SmartScreenTrustModal.tsx`, `src/pages/Dashboard.tsx` | Interactive modal verification | Guidance explaining Windows SmartScreen warnings on open-source binaries, "More info" -> "Run anyway", and copyable PowerShell SHA-256 release checksum verification. |
| G5 | Exportable diagnostic bundle | YES | COMPLETED | `src-tauri/src/diagnostics.rs`, `src-tauri/src/commands.rs`, `src/components/FirstRunDiagnosticCard.tsx` | `test_secret_redaction_huggingface_token`, `test_secret_redaction_github_tokens`, `test_secret_redaction_openai_api_keys`, `test_secret_redaction_bearer_headers`, `test_secret_redaction_env_and_query_params`, `test_diagnostic_bundle_secret_redaction` | Full diagnostic bundle export (JSON format, clipboard copy & file download) with rigorous redaction of HF tokens, GitHub tokens, API keys, Bearer headers, and config passwords. |
| G6 | Storage planning UI | YES | COMPLETED | `src-tauri/src/diagnostics.rs`, `src-tauri/src/commands.rs`, `src/components/StoragePlanner.tsx`, `src/pages/Search.tsx`, `src/pages/Library.tsx`, `src/pages/Servers.tsx` | Disk preflight evaluation & volume tests | Storage planning component displaying model weight + staging space margin + KV cache/swap vs volume free space (consuming A5 disk preflight) with Headroom status indicators. |
| G7 | Per-server compatibility test button | YES | COMPLETED | `src-tauri/src/diagnostics.rs`, `src-tauri/src/commands.rs`, `src/components/ServerCompatibilityModal.tsx`, `src/pages/Servers.tsx` | `servers_compatibility_test` runner & modal integration | Interactive "Test" button on live server cards running 5 automated prompts (basic chat, streaming SSE, tool calling, structured JSON output, system prompt) reporting latency, pass/fail status, and exportable report. |

## Quality Gate Checklist
- [x] `cargo test` passes (503 unit + integration + schema tests passed, 0 failed)
- [x] `cargo clippy --workspace --all-targets` has no new warnings in owned files
- [x] `npm run build` passes (TypeScript compilation & Vite production bundle clean in ~1.2s)

## Verification & Implementation Summary

### G1 & G3: Unified 8-Step First-Run Diagnostic & Windows 10/11 Support
- Added `src-tauri/src/diagnostics.rs` with `run_ordered_diagnostics` executing the strict 8-step sequence:
  1. `windows_version`: Verifies Windows 10 Build >= 19041 (Version 2004) or Windows 11 Build >= 22000. Rejects older builds with direct link and guidance for Windows Update.
  2. `wsl_installed`: Checks `wsl.exe` and modern WSL package version (>= 2.0.0).
  3. `wsl2_working`: Probes WSL2 container runtime and kernel execution responsiveness.
  4. `nvidia_driver`: Probes Windows host NVIDIA graphics driver (>= 525) and VRAM capacity.
  5. `gpu_in_wsl`: Validates GPU paravirtualization inside WSL (`/usr/lib/wsl/lib/libcuda.so.1` and `gpu_snapshot`).
  6. `cuda_runtime`: Validates CUDA acceleration in WSL PyTorch or native llama.cpp.
  7. `disk_space`: Validates volume free space on model cache drive (warns < 20 GB, fails < 10 GB).
  8. `backend_ready`: Checks vLLM or llama.cpp readiness.
- **Beginner Safety Rule:** `gpu_usable` flag is explicitly computed; overall report status is strictly failure/action required if GPU cannot be used, preventing novice users from experiencing silent fallback to unusable CPU inference.
- Created `FirstRunDiagnosticCard.tsx` on the Dashboard with collapsible actionable fix recommendations and copyable terminal commands.

### G2: Distinguish Error Sources in UI
- Created `ErrorAlert.tsx` with dedicated sources: `"download"`, `"provisioning"`, `"server"`, `"system"`.
- Each source features distinct icons, color themes, badges, and contextual troubleshooting hints.
- Applied across `Dashboard.tsx` (WSL & llama.cpp setup errors), `Servers.tsx` (server crash, port, and inference errors), `Search.tsx` (catalog and Hugging Face errors), and `Library.tsx` (import and deletion errors).

### G4: Windows Trust & SmartScreen Guidance
- Added `SmartScreenTrustModal.tsx` accessible directly from the Dashboard diagnostic card.
- Provides clear explanation of unsigned community releases and why Microsoft Defender SmartScreen warns.
- Documents the exact sequence: "More info" -> "Run anyway".
- Includes one-click copyable PowerShell command for verifying SHA-256 release checksums:
  `Get-FileHash "$HOME\Downloads\LocalLLmPanel-setup.exe" -Algorithm SHA256`.

### G5: Exportable Diagnostic Bundle with Secret Redaction
- Added `generate_redacted_bundle` in `src-tauri/src/diagnostics.rs` and `diagnostics_export_bundle` command.
- Assembles app version, Windows build, WSL version/state, GPU snapshots, backend versions, provision reports, server configurations, and recent log ring buffers.
- Thorough secret redaction: Hugging Face tokens (`hf_*`), GitHub tokens (`ghp_*`, `github_pat_*`), OpenAI/generic API keys (`sk-*`), Bearer tokens, and query/env parameters (`password=`, `token=`, `api_key=`).
- Automated tests verify zero leakage of sensitive keys or tokens in exported bundle JSON.

### G6: Storage Planning UI
- Added `evaluate_disk_preflight` in `diagnostics.rs` and `disk_preflight` command (consuming Stream A A5 Win32 `GetDiskFreeSpaceExW`).
- Created `StoragePlanner.tsx` displaying:
  - Model weights size
  - Temporary staging margin (model size + 512 MB buffer)
  - KV Cache / Swap allocation
  - Total Required vs Available volume free space
  - Headroom status badge (`Sufficient Space`, `Storage Tight`, `Insufficient Space`).
- Integrated into `Search.tsx` (Model Detail modal), `Library.tsx` (cache header), and `Servers.tsx` (New Server configuration).

### G7: Per-Server Compatibility Test Runner
- Added `run_server_compatibility_test` in `src-tauri/src/diagnostics.rs` and `servers_compatibility_test` command.
- Runs 5 live compatibility test prompts against the running server endpoint (`/v1/chat/completions`):
  1. Basic Chat Completion
  2. Token Streaming (SSE)
  3. Tool / Function Calling (`get_weather`)
  4. Structured JSON Output (`response_format` and JSON validation)
  5. System / Developer Prompt Normalization
- Measures per-test latency (ms), pass/fail status, response snippets, and failure reasons.
- Added interactive "🧪 Test" button on server cards and `ServerCompatibilityModal.tsx` with full report export.
