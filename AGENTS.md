# AGENTS.md — Local LLM Panel

Tauri 2 (Rust + React/TypeScript) desktop app managing vLLM in WSL2 and native llama.cpp on Windows.

## Quick Commands

| Task | Command |
|------|---------|
| Dev server | `npm run tauri dev` |
| Preflight (type check + cargo check + frontend contracts) | `npm run check` |
| Release metadata | `npm run release:check` |
| Frontend contract tests | `npm run test:contract` |
| Frontend build | `npm run build` |
| Rust formatting | `cd src-tauri && cargo fmt --all -- --check` |
| Rust lint | `cd src-tauri && cargo clippy --workspace --all-targets --locked` |
| Unit tests | `cd src-tauri && cargo test --workspace --locked` |
| Fast binary | `npm run build:fast` |
| Standard release (NSIS installer) | `npm run build:setup` |
| Full dist (NSIS + MSI) | `npm run build:all` |
| WSL integration test (gated) | `$env:LLM_TEST_WSL="1"; cd src-tauri; cargo test --lib -- --ignored --nocapture wsl_it` |

## Version Synchronization (4 files)

Update **all four** on version bump:
1. `package.json` → `"version": "x.y.z"`
2. `package-lock.json` → `"version": "x.y.z"` + root package version
3. `src-tauri/Cargo.toml` → `version = "x.y.z"`
4. `src-tauri/tauri.conf.json` → `"version": "x.y.z"`

Build script and `npm run release:check` validate version synchronization.

## Build Gotchas

- **lld-link.exe not found**: Run via npm (`npm run build:*`), not raw `cargo build`. Build script auto-injects LLVM path (`C:\Program Files\LLVM\bin` or VS VC Tools).
- **Windows 11 24H2+ crash (0xc0000139)**: Fixed by `.cargo/config.toml` forcing `getrandom_backend="windows_legacy"`.
- **WiX MSI slow**: Only enabled via `npm run build:all`; default is NSIS only.
- **Release profile**: `Cargo.toml` uses `lto = false`, `incremental = false`, `codegen-units = 16` for fast linking.

## Test Notes

- Frontend contract checks run with `npm run test:contract`.
- Rust unit tests run with `cargo test --workspace --locked` (estimate math, port allocation, config round-trip, llmfit-core, and related modules).
- CI runs on Windows and validates metadata, contracts, Rust formatting/lint/tests, frontend build, standalone executable build, and npm/Rust dependency audits.
- WSL integration tests are `#[ignore]`-gated; they require WSL2 Ubuntu + NVIDIA GPU + ~5GB disk.
- Set `LLM_TEST_WSL=1` to run; the test may skip if `nvidia-smi` is unavailable in WSL.
- Native llama.cpp integration tests require a compatible local executable and a small GGUF; see the README test instructions.
- GPU/WSL end-to-end tests are intentionally not part of standard hosted CI.

## Architecture Highlights

- **Rust core**: `wsl.rs` (distro/run), `provision.rs` (idempotent phases), `server.rs` (lifecycle/logs/metrics), `estimate.rs`/`fit.rs` (pure heuristics), `hf.rs` (search/pull/quant discovery), `state.rs` (config + live state), `llamacpp_install.rs` (llama.cpp fork install/update; built-in upstream, user-defined forks via Settings).
- **Frontend**: Vite + React 18 + TS; Tailwind v4; pages in `src/pages/`; `api.ts` wraps Tauri invoke + events.
- **Config**: `%APPDATA%\local-llm-panel\config.json` (atomic write).
- **vLLM env**: per-server `env` merged with global `default_env`; launch scripts written to WSL files to avoid quoting issues; default `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **llama.cpp**: native Windows process; GGUF from configured dir; capability-gated flags (`--fit`, `n_cpu_moe`).

## Release Workflow

1. Run `npm run check`, `npm run test:contract`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked`, and `cargo test --workspace --locked`.
2. Bump the version in all four files listed above.
3. Run `npm run release:check`.
4. Build with `npm run build:setup` (or `npm run build:all` for the MSI).
5. Verify the contents of `dist-release/` and the adjacent SHA-256 checksums.
6. Commit and push the versioned source, then create and push an annotated `vX.Y.Z` tag.
7. The `Draft release` workflow builds and uploads the Windows executable, NSIS installer, and checksums as a **draft** GitHub Release. Review the draft and publish it when ready.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full manual release SOP.

## Key Files

- `scripts/build.mjs` — build runner with linker auto-resolution & artifact staging
- `scripts/validate-release.mjs` — manifest and artifact validation
- `src-tauri/.cargo/config.toml` — Windows 11 getrandom workaround + lld-link
- `src-tauri/Cargo.toml` — workspace and release profile
- `.github/workflows/ci.yml` — Windows checks and artifact build
- `.github/workflows/codeql.yml` — scheduled static analysis (JavaScript/TypeScript, Rust, Actions)
- `.github/workflows/draft-release.yml` — tag-triggered draft releases
- `docs/DEPLOYMENT.md` — full SOP
- `docs/ARCHITECTURE.md` — system diagram & module details

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` and `docs/adr/`). See `docs/agents/domain.md`.

