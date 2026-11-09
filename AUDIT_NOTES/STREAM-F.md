# Stream F: CI, Testing, Docs, Ops

**Ownership:** `.github/workflows/*`, `README.md`, `ABOUT.md`, `docs/`, test infrastructure.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| F1 | Security audit must gate releases | YES | RESOLVED | `audit-allowlist.json`, `scripts/security-audit.mjs`, `src-tauri/audit.toml`, `src-tauri/.cargo/audit.toml`, `package.json`, `.github/workflows/security-audit.yml`, `.github/workflows/ci.yml`, `.github/workflows/draft-release.yml` | `audit:check` test in `scripts/frontend-smoke.mjs`, standalone `node scripts/security-audit.mjs` execution | Blocking cargo/npm audit with allowlist mechanism; all `continue-on-error: true` removed; gate added to release workflow. |
| F2 | Installer packages in CI | YES | RESOLVED | `src-tauri/tauri.conf.json`, `.github/workflows/ci.yml` | CI build:all artifact validation step | NSIS (`.exe`) and MSI (`.msi`) bundles enabled in `tauri.conf.json`; CI executes `npm run build:all` and verifies all artifacts and `.sha256` checksums. |
| F3 | GPU/WSL integration test harness | YES | RESOLVED | `src-tauri/tests/gpu_wsl_integration.rs` | 5 integration tests (`test_live_wsl_provisioning_verify`, `test_live_vllm_start_stop_lifecycle`, `test_live_llamacpp_start_stop_lifecycle`, `test_live_context_length_ramp`, `test_live_crash_recovery`) | All tests strictly guarded by `#[ignore]` and `require_live_gpu_env()` checking `LLP_LIVE_GPU_TESTS=1` + physical NVIDIA hardware. |
| F4 | Rust/frontend contract drift | YES | RESOLVED | `src-tauri/tests/contract_drift.rs`, `scripts/frontend-smoke.mjs`, `src-tauri/src/lib.rs`, `src-tauri/src/commands.rs` | 3 Rust contract tests in `contract_drift.rs`, contract drift suites in `scripts/frontend-smoke.mjs` (13 tests total) | Discovered and fixed unregistered command `servers_update_env` and non-Send MutexGuard across await in `commands.rs`. |
| F5 | README/ABOUT corrections | YES | RESOLVED | `README.md`, `ABOUT.md` | Verification of Markdown formatting, link validation, and clean UTF-8 encoding | Corrected Windows 10 (2004+) & 11 platform coverage; documented chunked downloads, `.part` staging, resume, and shard verification; added preflight disk checks; documented DPAPI encryption; clarified analytical fit projections vs benchmarks; added LAN access requirements; added SmartScreen warning & SHA-256 verification instructions; eliminated mojibake. |
| F6 | Guided uninstall / environment cleanup | YES | RESOLVED | `src-tauri/src/cleanup.rs`, `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs`, `src/types.ts`, `src/api.ts` | 4 unit tests in `src-tauri/src/cleanup.rs` (dry run, empty confirmation, selective `.part` file deletion, serialization) | Full dry-run itemization and safe confirmed execution across WSL distro, venv, HF cache, GGUF dir, partial downloads, and sudoers rule. |

## Quality Gate Checklist
- [x] `cargo test` passes (500+ tests pass across workspace; 5 live GPU integration tests properly ignored)
- [x] `cargo clippy --workspace --all-targets` has no new warnings (clean exit code 0)
- [x] `npm run build` passes (`tsc --noEmit && vite build` bundled successfully in ~3s)
- [x] `node scripts/security-audit.mjs` passes with zero unallowlisted advisories
- [x] `node --test scripts/frontend-smoke.mjs` passes (13/13 tests)

## Verification Evidence

1. **Security Audit (F1)**:
   - Command: `node scripts/security-audit.mjs`
   - Output: Loaded allowlist with 2 accepted advisories (`RUSTSEC-2024-0370`, `RUSTSEC-2024-0429`). npm audit and cargo audit both passed with code 0.
   - Workflows: Removed `continue-on-error: true` in `security-audit.yml` and `ci.yml`. Added `security-audit-gate` job in `draft-release.yml`.

2. **Installer Packages (F2)**:
   - Config: `src-tauri/tauri.conf.json` updated with `"targets": ["nsis", "msi"]`.
   - CI: `.github/workflows/ci.yml` builds with `npm run build:all` and validates presence of standalone exe, NSIS installer, MSI installer, and SHA-256 checksums.

3. **GPU / WSL Live Test Harness (F3)**:
   - Command: `cargo test --test gpu_wsl_integration`
   - Result: 5 tests reported `ignored` (`require_live_gpu_env()` check active). In live runs with `LLP_LIVE_GPU_TESTS=1`, tests exercise WSL provisioning, vLLM / llama.cpp lifecycle, context length ramping, and crash recovery.

4. **Contract Drift (F4)**:
   - Command: `cargo test --test contract_drift`
   - Result: 3 passed in 0.06s.
   - Command: `node --test scripts/frontend-smoke.mjs`
   - Result: 13 passed in 127ms.
   - Contract Fix: Discovered `servers_update_env` in `commands.rs` was missing from `generate_context!` and `invoke_handler!` in `lib.rs`, and had a non-Send `MutexGuard` held across an `.await`. Fixed scope and registered command.

5. **Guided Cleanup (F6)**:
   - Unit tests: 4 passed in `src-tauri/src/cleanup.rs`.
   - IPC Contract: Verified TypeScript interfaces `CleanupTarget`, `CleanupItem`, `CleanupInspectionResult`, `CleanupExecutionRequest`, `CleanupExecutionResult` in `src/types.ts` and `src/api.ts`.

6. **Documentation & Encoding (F5)**:
   - Verified clean UTF-8 without BOM or mojibake in `README.md` and `ABOUT.md`.
   - Corrected platform specs, download details, security disclosures, sizing explanations, LAN configuration, and SmartScreen guidance.
