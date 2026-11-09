# Stream B: NVIDIA Runtime, Provisioning, Estimator

**Ownership:** `src-tauri/src/provision.rs`, `src-tauri/src/llamacpp_install.rs`, `src-tauri/src/llmfit_adapter.rs`, `src-tauri/src/estimate.rs`, `src-tauri/src/fit.rs`.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| B1 | Provisioning reports success without GPU | PENDING | PENDING | | | Distinct failure states in phase_verify |
| B2 | CUDA asset selection fails open | PENDING | PENDING | | | Strict driver exit status & parse checking |
| B3 | Multi-GPU parsing & fit | PENDING | PENDING | | | Enumerate all GPUs, evaluate selected devices |
| B4 | Estimates presented as guarantees | PENDING | PENDING | | | Label as estimates, version-gate vLLM flags |
| B5 | Synthetic RAM values | PENDING | PENDING | | | Preserve Option from detect_wsl_memory |
| B6 | `uv` install pipeline | PENDING | PENDING | | | set -euo pipefail, verify command |
| B7 | Unpinned runtimes | PENDING | PENDING | | | Compatibility manifest, constrained versions |
| B8 | Native llama.cpp installer reliability | PENDING | PENDING | | | Staging dir, atomic promotion, checksums |
| B9 | FlashInfer build-tool phase hardcoding | PENDING | PENDING | | | Use configured venv_dir and sys.version_info |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
