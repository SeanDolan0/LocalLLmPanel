# Stream B: NVIDIA Runtime, Provisioning, Estimator

**Ownership:** `src-tauri/src/provision.rs`, `src-tauri/src/llamacpp_install.rs`, `src-tauri/src/llmfit_adapter.rs`, `src-tauri/src/estimate.rs`, `src-tauri/src/fit.rs`.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| B1 | Provisioning reports success without GPU | VERIFIED | FIXED | `src-tauri/src/provision.rs` | `test_parse_verify_result_success`, `test_parse_verify_result_driver_not_visible`, `test_parse_verify_result_no_gpu_in_wsl`, `test_parse_verify_result_torch_import_failed`, `test_parse_verify_result_cuda_unavailable` | Distinct failure states in `phase_verify` with targeted error messages and diagnostics; bails on failure so `.provisioned` is not written. Live WSL GPU pass unverified (needs live test). |
| B2 | CUDA asset selection fails open | VERIFIED | FIXED | `src-tauri/src/llamacpp_install.rs` | `test_cuda_selection_refuses_when_driver_unknown`, `test_cuda_selection_allows_with_override_and_warns` | `driver_cuda_version` now checks exit status; `choose_assets_for_machine` stops when driver unknown; override allowed only with explicit user acknowledgment (`LLP_ALLOW_UNKNOWN_CUDA_DRIVER=1` or explicit flag). |
| B3 | Multi-GPU parsing & fit | VERIFIED | FIXED | `src-tauri/src/llamacpp_install.rs`, `src-tauri/src/fit.rs` | `test_parse_compute_capabilities_multiline_and_min_policy`, `test_parse_nvidia_gpus_multiline`, `test_fit_labels_estimates_and_single_gpu` | Enumerate all GPUs (index, name, VRAM, free VRAM, compute_cap, util); per-line parsing; minimum compute capability policy across devices; labeled as "single-GPU estimate". Multi-GPU WSL telemetry queued in XREQ-02. |
| B4 | Estimates presented as guarantees | VERIFIED | FIXED | `src-tauri/src/fit.rs`, `src-tauri/src/estimate.rs` | `test_vllm_kv_offloading_flag_version_gating`, `test_fit_labels_estimates_and_single_gpu` | Context and capacity figures labeled as heuristic estimates in verdict, reason, and notes; full swap buffer labeled as estimate; version-gating helper for `--kv-offloading-size` (valid for vLLM >= 0.6.0). |
| B5 | Synthetic RAM values | VERIFIED | FIXED | `src-tauri/src/fit.rs`, `src-tauri/src/estimate.rs` | `test_parse_wslconfig_memory`, `test_estimate_wsl_memory_from_host`, `test_fit_ram_unknown_disables_ram_overflow` | Preserves `Option`; when RAM unknown (`is_ram_known() == false`), RAM overflow fit is disabled and marked uncertain; added Windows physical RAM query and `.wslconfig` memory parsing for stopped WSL. Queued XREQ-03 for `commands.rs`/`wsl.rs`. |
| B6 | `uv` install pipeline | VERIFIED | FIXED | `src-tauri/src/provision.rs` | `test_venv_assignment_ends_with_newline` | `phase_uv` uses `set -euo pipefail`, verifies `command -v uv` and version, and executes fallback package install if verification fails. Live WSL run unverified (needs live test). |
| B7 | Unpinned runtimes | VERIFIED | FIXED | `src-tauri/src/provision.rs` | `test_runtime_manifest_consistency` | Added tested `RuntimeCompatibilityManifest` (Python 3.12, vLLM 0.7.3, torch 2.5.1, huggingface_hub 0.29.1, CUDA >= 12.0); install pins constrained versions; added `transactional_venv_operation` with snapshot and automatic rollback on failure. Live upgrade unverified (needs live test). |
| B8 | Native llama.cpp installer reliability | VERIFIED | FIXED | `src-tauri/src/llamacpp_install.rs` | `test_staging_dir_promotion_and_atomic_rollback`, `test_compute_file_sha256` | Added disk space preflight (1.5 GB minimum), request timeouts (300s/30s) and retry with backoff, extract to staging directory, executable verification before atomic promotion, rollback to previous working install on failure. Live download unverified (needs live test). |
| B9 | FlashInfer build-tool phase hardcoding | VERIFIED | FIXED | `src-tauri/src/provision.rs` | Unit tests in `provision.rs` | Configurable `venv_dir` (`phase_cuda_build_tools_with_venv`); dynamic querying of Python `sys.version_info` for package names and header detection; dynamic PyTorch CUDA query. Live JIT build unverified (needs live test). |

## Quality Gate Checklist
- [x] `cargo test` passes
- [x] `cargo clippy --workspace --all-targets` has no new warnings
- [x] `npm run build` passes
