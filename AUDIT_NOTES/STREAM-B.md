# Stream B: NVIDIA Runtime, Provisioning, Estimator

**Ownership:** src-tauri/src/provision.rs, src-tauri/src/llamacpp_install.rs, src-tauri/src/llmfit_adapter.rs, src-tauri/src/estimate.rs, src-tauri/src/fit.rs.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| B1 | Provisioning reports success without GPU | VERIFIED | IN_PROGRESS | src-tauri/src/provision.rs | Distinct failure states in phase_verify | Verified: phase_verify returns Ok(report) even if cuda_available is false. Must bail with distinct failure states. |
| B2 | CUDA asset selection fails open | VERIFIED | IN_PROGRESS | src-tauri/src/llamacpp_install.rs | driver_cuda_version exit check & unknown driver stop | Verified: driver.is_none_or passes all assets when driver is None. Output exit status not checked. |
| B3 | Multi-GPU parsing & fit | VERIFIED | IN_PROGRESS | src-tauri/src/llamacpp_install.rs, src-tauri/src/fit.rs | Per-line GPU parsing, min-capability policy | Verified: compute_capability and gpu snapshot take first line or split dots across lines. Fit must label single-GPU estimate. |
| B4 | Estimates presented as guarantees | VERIFIED | IN_PROGRESS | src-tauri/src/fit.rs, src-tauri/src/estimate.rs | Label context as estimates, vLLM flag version gating | Verified: Reasons presented context as hard limits. Must label as estimates and version-gate vLLM flags. |
| B5 | Synthetic RAM values | VERIFIED | IN_PROGRESS | src-tauri/src/fit.rs, src-tauri/src/estimate.rs | Unknown RAM handling, .wslconfig parsing | Verified: Unknown WSL memory defaulted to 16/12GB. Must mark RAM overflow uncertain/disabled and support .wslconfig. |
| B6 | uv install pipeline | VERIFIED | IN_PROGRESS | src-tauri/src/provision.rs | set -euo pipefail and post-install command verification | Verified: Missing pipefail and post-install uv verification. |
| B7 | Unpinned runtimes | VERIFIED | IN_PROGRESS | src-tauri/src/provision.rs | Runtime compatibility manifest, transactional rollback | Verified: Unpinned vllm and huggingface_hub installs. Must constrain versions and support rollback. |
| B8 | Native llama.cpp installer reliability | VERIFIED | IN_PROGRESS | src-tauri/src/llamacpp_install.rs | Disk space preflight, retry/timeout, staging & promotion | Verified: Direct extraction to install dir, no disk check, no download timeout/retry. |
| B9 | FlashInfer build-tool phase hardcoding | VERIFIED | IN_PROGRESS | src-tauri/src/provision.rs | Dynamic python version & venv_dir detection | Verified: Hardcoded python3.12 and ~/.venv in phase_cuda_build_tools. |

## Quality Gate Checklist
- [ ] cargo test passes
- [ ] cargo clippy --workspace --all-targets has no new warnings
- [ ] npm run build passes
