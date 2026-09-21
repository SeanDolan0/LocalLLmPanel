# AGENTS.md — Local LLM Panel

Tauri 2 (Rust + React/TypeScript) desktop app managing vLLM in WSL2 and native llama.cpp on Windows.

## Quick Commands

| Task | Command | Time |
|------|---------|------|
| Dev server | `npm run tauri dev` | — |
| Preflight (typecheck + cargo check) | `npm run check` | ~3s |
| Fast binary | `npm run build:fast` | ~20s |
| Standard release (NSIS installer) | `npm run build:setup` | ~40s |
| Full dist (NSIS + MSI) | `npm run build:all` | ~2–3m |
| Unit tests | `cd src-tauri && cargo test` | ~5–10s |
| WSL integration test (gated) | `cd src-tauri && $env:LLM_TEST_WSL="1"; cargo test --lib -- --ignored --nocapture wsl_it` | — |

## Version Synchronization (4 files)

Update **all four** on version bump:
1. `package.json` → `"version": "x.y.z"`
2. `package-lock.json` → `"version": "x.y.z"` + root package version
3. `src-tauri/Cargo.toml` → `version = "x.y.z"`
4. `src-tauri/tauri.conf.json` → `"version": "x.y.z"`

Build script (`scripts/build.mjs`) warns on mismatch.

## Build Gotchas

- **lld-link.exe not found**: Run via npm (`npm run build:*`), not raw `cargo build`. Build script auto-injects LLVM path (`C:\Program Files\LLVM\bin` or VS VC Tools).
- **Windows 11 24H2+ crash (0xc0000139)**: Fixed by `.cargo/config.toml` forcing `getrandom_backend="windows_legacy"`.
- **WiX MSI slow**: Only enabled via `npm run build:all`; default is NSIS only.
- **Release profile**: `Cargo.toml` uses `lto = false`, `incremental = false`, `codegen-units = 16` for fast linking.

## Test Notes

- Unit tests run via `cargo test` (estimate math, port alloc, config round-trip, llmfit-core).
- WSL integration test is `#[ignore]`-gated; requires WSL2 Ubuntu + NVIDIA GPU + ~5GB disk.
- Set `LLM_TEST_WSL=1` to run; skips if `nvidia-smi` unavailable in WSL.

## Architecture Highlights

- **Rust core**: `wsl.rs` (distro/run), `provision.rs` (idempotent phases), `server.rs` (lifecycle/logs/metrics), `estimate.rs`/`fit.rs` (pure heuristics), `hf.rs` (search/pull/quant discovery), `state.rs` (config + live state), `llamacpp_install.rs` (native binary mgmt).
- **Frontend**: Vite + React 18 + TS; Tailwind v4; pages in `src/pages/`; `api.ts` wraps Tauri invoke + events.
- **Config**: `%APPDATA%\local-llm-panel\config.json` (atomic write).
- **vLLM env**: per-server `env` merged with global `default_env`; launch scripts written to WSL files to avoid quoting issues; default `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **llama.cpp**: native Windows process; GGUF from configured dir; capability-gated flags (`--fit`, `n_cpu_moe`).

## Release Workflow

1. `npm run check` + `cd src-tauri && cargo test`
2. Bump version in 4 files
3. `npm run build:setup` (or `build:all`)
4. Verify `dist-release/` artifacts
5. `git add package.json package-lock.json src-tauri/Cargo.toml src-tauri/tauri.conf.json src-tauri/Cargo.lock dist-release/`
6. `git commit -m "release: vX.Y.Z" && git tag -a "vX.Y.Z" -m "Release vX.Y.Z"`
7. `git push origin main && git push origin "vX.Y.Z"`
8. `gh release create "vX.Y.Z" dist-release/* --title "vX.Y.Z" --generate-notes`

## Key Files

- `scripts/build.mjs` — build runner with linker auto-resolution & artifact staging
- `src-tauri/.cargo/config.toml` — Windows 11 getrandom workaround + lld-link
- `src-tauri/Cargo.toml` — release profile tuned for speed
- `docs/DEPLOYMENT.md` — full SOP
- `docs/ARCHITECTURE.md` — system diagram & module details