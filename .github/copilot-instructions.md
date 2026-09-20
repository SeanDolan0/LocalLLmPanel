# Copilot instructions for Local LLM Panel

## Project shape

Local LLM Panel is a Windows 11 desktop application built with Tauri 2. The React 18/TypeScript/Vite frontend runs in WebView2; the Rust backend owns Windows/WSL process control, persistence, network calls, metrics, and all Tauri commands/events.

- `src/` contains the UI shell and feature pages: Dashboard, Search, Library, Servers, and Settings.
- `src/api.ts` is the typed frontend boundary for `@tauri-apps/api/core.invoke` and Tauri event subscriptions; `src/types.ts` mirrors the Rust command payloads.
- `src-tauri/src/commands.rs` is the public command layer. `lib.rs` registers commands and initializes shared `AppState`.
- `src-tauri/src/state.rs` owns persisted configuration, live server state, caches, conversations, benchmarks, and time-series metrics. Configuration is JSON under `%APPDATA%\local-llm-panel\`; saves use a temporary file and rename.
- `wsl.rs` is the process boundary for `wsl.exe`; `provision.rs` idempotently installs the WSL venv/vLLM environment; `server.rs` launches, monitors, logs, measures, and stops multiple vLLM instances.
- `hf.rs` searches/enriches Hugging Face models and downloads them into the WSL Hugging Face cache. `estimate.rs` and `fit.rs` are pure estimation/ranking logic; estimates are replaced by measured vLLM `/metrics` data when available.
- `src-tauri/crates/llmfit-core` is a path dependency providing hardware detection, model fitting, provider detection, and related data validation. Keep its public API concerns separate from the app-specific vLLM/WSL orchestration.
- `gateway.rs` is a small Tokio TCP HTTP proxy on the configured local gateway port (default `11434`) that routes OpenAI-compatible chat and embedding requests to running servers.

The runtime model is: Windows app -> `wsl.exe -d <distro> -- bash -lc ...` -> Ubuntu venv -> one foreground vLLM process per configured server. Servers bind to the configured host/port, normally `127.0.0.1`; logs and status/metrics are streamed back through Tauri events. The frontend uses `MemoryRouter`, so navigation state is in-process rather than URL-based.

## Build, run, and test

Run commands from the repository root unless noted.

```bash
npm install
npm run dev                 # Vite frontend only
npm run tauri dev           # development desktop app; builds frontend and Rust
npm run check               # fast preflight: TypeScript checks + cargo check (~3s)
npm run build:fast          # fast standalone executable (.exe only, ~20s, no bundling)
npm run build:setup         # standard release: standalone + NSIS setup installer (~40s)
npm run build:all           # full distribution: standalone + NSIS + WiX MSI package
npm run release             # alias for npm run build:setup
```

See [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) for the complete AI agent deployment SOP, 4-file version synchronization rules, and `skills/github-deployment/SKILL.md` for GitHub release workflows.


Rust commands run from `src-tauri`:

```bash
cargo test                  # app unit tests plus llmfit-core tests
cargo test -p llmfit-core   # only the path dependency's tests
cargo test test_name       # one matching unit test by name
cargo test --lib -- --ignored --nocapture wsl_it
                             # real WSL/vLLM test; set LLM_TEST_WSL=1 first
```

The WSL test is intentionally an ignored library test in `src-tauri/src/it.rs`, not a `tests/integration_wsl.rs` target:

```bash
set LLM_TEST_WSL=1 && cargo test --lib -- --ignored --nocapture wsl_it
```

It needs an Ubuntu WSL2 distro, internet access, an NVIDIA GPU visible inside WSL, and several GB for the vLLM/torch installation. There is no npm lint script or separate frontend test runner in `package.json`; `npm run build` is the frontend type/build check.

## Conventions and invariants

- Keep the frontend/backend contract synchronized. When changing a Tauri command, event, or serialized structure, update the Rust definition, command registration if needed, `src/api.ts`, and the corresponding interfaces in `src/types.ts`.
- Use Tauri commands for privileged/native work rather than browser APIs. Long-running work belongs in Tokio tasks or `spawn_blocking`; use named events such as `wsl-log`, `server-status`, `server-log`, `pull-progress`, and metrics events for streaming progress.
- Treat `AppState` as shared concurrent state. Follow the existing `Arc<AppState>` plus `Mutex` pattern and avoid holding a lock across blocking I/O, HTTP, process waits, or `.await`.
- Preserve persisted-config compatibility. New fields should use `serde(default)` or an explicit default function, and config/conversation/benchmark writes should remain atomic. Do not silently discard unknown or older settings.
- WSL paths may be Windows paths, WSL paths, or `~` paths. Reuse `wsl::windows_to_wsl_path` and the existing shell-quoting helpers; do not interpolate untrusted model IDs, tokens, paths, or extra arguments into `bash -lc` scripts without the same escaping/validation treatment.
- Server lifecycle is process- and health-driven: use the pidfile written in WSL, poll `/health`, and emit status transitions. Preserve per-server isolation, local binding defaults, log ring-buffer behavior, and graceful-stop/taskkill fallback.
- Model-fit math is deliberately pure and unit-tested. Keep context parsing, quantization byte assumptions, VRAM/RAM tiering, tok/s estimates, and variant ranking deterministic and free of I/O. Label heuristic values as estimates; measured metrics win.
- Quantization is not interchangeable: pre-quantized model repositories may reject an explicit `--quantization` flag. Preserve the existing detection and per-server quantization behavior when changing launch arguments.
- Keep errors visible across boundaries. Rust command failures return descriptive `Result<_, String>` errors to the UI; do not turn failed provisioning, server startup, HTTP requests, or persistence into success-shaped defaults.
- Use the existing dark Tailwind v4 styling and small page/component structure. Prefer typed API helpers over direct `invoke` calls in pages, and subscribe/unsubscribe to Tauri events with the same lifecycle pattern already used by the UI.
- The gateway routes only running servers of the requested task (`instruct` versus `embed`), preferring exact served-name/Hugging Face ID matches before suffix matches. Changes to routing should preserve model-listing and CORS/preflight behavior.

## Documentation sources

`README.md` contains the supported Windows/WSL2 prerequisites, setup walkthrough, production output locations, configuration path, and operational gotchas. `docs/ARCHITECTURE.md` describes module responsibilities and data flow; update it when a cross-layer behavior or process model changes.

## Optional MCP server

For browser-level testing of the React/WebView surface, configure the Playwright MCP server in the local Copilot/IDE MCP settings rather than committing credentials or client-specific configuration to this repository. Start the server with `npx @playwright/mcp@latest`; use it against the Vite dev server (`npm run dev`, port `1420`) and focus on navigation, Tauri-backed loading/error states, server-form interactions, and event-driven log/metric updates.
