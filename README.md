# Local LLM Panel

A native Windows desktop app (**Tauri 2** + React/TypeScript) for running local LLMs on your own hardware. It manages **vLLM inside WSL2** and **native Windows llama.cpp** servers from one UI, and exposes everything through a single OpenAI-compatible endpoint.

No cloud, no containers, no per-server port juggling. Estimates are estimates — measured data wins.

## Highlights

### Find & prepare models
- **Search Hugging Face** live, with real `config.json` context length, architecture, parameter count, and per-GPU **estimated tok/s**.
- **Quant-aware fit scoring** — AWQ/GPTQ/FP8/BNB repos and GGUF quants are discovered per model and ranked against your actual VRAM. Verdicts: **Comfortable / Constrained / DoesNotFit**.
- **VRAM → RAM overflow planning** — see usable context on GPU alone and the extended context available by swapping KV cache into WSL2 RAM, plus weight-offload modes. Configurable GPU utilization, overhead, safety reserve, and context cap in **Settings → Advanced Mode → Memory**.
- **Context fit test** — check a planned context length against current free VRAM/RAM before starting a server.
- **Download with control** — background pulls with progress, ETA, byte counts, and cancel. GGUF downloads go to a native Windows directory and handle split shards and `mmproj` companions.
- **Library management** — scan the HF cache and GGUF folders, see per-model disk usage, delete safely (blocked while a server is using the model), or import a local GGUF.

### Run & observe servers
- **Run multiple servers concurrently** — instruct and embedding tasks side by side, with per-server quantization, GPU memory utilization, max model length, port, served model name, and API key.
- **vLLM auto-provisioning (WSL2)** — idempotent phases: apt basics → `uv` venv (CPython 3.12) → vLLM CUDA wheels → torch/GPU verification. Re-running repairs or upgrades.
- **Native llama.cpp** — install CUDA builds of `llama-server.exe`, pick upstream or the **PrismML fork** (ternary formats / Bonsai 2 preset), use GGUF and `mmproj` vision files, MoE CPU offload, `--fit` auto-fit, Jinja tool calling, and per-channel installs.
- **Environment variables** — global defaults merged with per-server overrides. Launch scripts are written to WSL files to avoid quoting issues; boolean `VLLM_USE_*` flags are validated.
- **Live logs and metrics** — streamed logs, request/queue/KV-cache/TTFT/tok-s metrics, per-server metrics panel, and VRAM/GPU sparklines on the Dashboard.
- **Chat playground** — streaming responses, cancellation, saved conversations, markdown rendering, and an OpenAI tool-call smoke test.
- **Benchmark suite** — standardized prompt set, prompt/decode tok-s, latency, and history across runs and quants.
- **Appliance mode** — tray icon (open panel, stop all, quit), minimize-to-tray, start at login, automatic restart of crashed servers, and optional **WSL shutdown when idle**.

### One endpoint for every client
- **OpenAI-compatible gateway** on port `11434` by default:
  - `GET /v1/models` and `GET /v1/models/{id}` — running instruct and embedding servers with context limits, capabilities, quantization, and thinking metadata.
  - `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings` — routed by model name to the correct vLLM/llama.cpp server.
  - `GET /health` — gateway liveness (no authentication).
  - SSE streaming, CORS handling, and mandatory `Authorization: Bearer <key>` authentication on all `/v1/*` routes.
- Works with Cursor, Continue.dev, LibreChat, and any OpenAI-compatible client.
- **Harness-controlled thinking** — the gateway normalizes `reasoning_effort`, `thinking`, Ollama `think`, and `chat_template_kwargs` variants so the client's request wins over launch defaults, mapping standard effort labels onto the target backend generically across all model types.
- **LAN opt-in** — localhost-only by default; exposing servers or the gateway on `0.0.0.0` requires an explicit consent toggle.

### Security & portability
- **DPAPI-encrypted tokens** for Hugging Face and GitHub (Windows Credential Guard); plaintext is never written to disk.
- **Config export/import** and copy-to-clipboard **server recipes**.
- **Atomic config writes** and isolated test configs, so a failed save can't destroy the last known-good state.

## Requirements

- Windows 11 (WebView2), NVIDIA GPU and driver
- **For vLLM:** WSL2 with an Ubuntu distro and a CUDA ≥ 12.4 driver visible inside WSL (`.wslconfig` memory of `24GB` is what the dev machine uses)
- **For llama.cpp:** a recent CUDA-enabled `llama-server.exe`; WSL provisioning is not required for this backend
- Rust toolchain (MSVC target; VS Build Tools with "Desktop development with C++"), LLVM/LLD on `PATH`, Node ≥ 20
- Optional: GitHub token (llama.cpp release downloads), Hugging Face token (gated models)

## Development setup

```bash
npm install
npm run tauri dev        # builds the web UI + Rust, launches the app
```

Build and release commands:

```bash
npm run check              # fast preflight: type check + cargo check (~3s)
npm run build              # frontend-only TypeScript + Vite build
npm run build:fast         # fast standalone executable (~20s, no bundling)
npm run build:setup        # standard release: standalone + NSIS installer (~40s)
npm run build:all          # full distribution: standalone + NSIS + WiX MSI
npm run release:check      # validate release metadata/version sync
npm run test:contract      # frontend API-contract smoke tests
```

Staged binaries and installers are written to `dist-release/`.

The build runner (`scripts/build.mjs`) locates LLVM's `lld-link.exe` (from `C:\Program Files\LLVM\bin` or Visual Studio VC tools) automatically.

For the release SOP, version-sync rules, and GitHub publishing steps, see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and [`skills/github-deployment/SKILL.md`](skills/github-deployment/SKILL.md). Agent quick reference: [`AGENTS.md`](AGENTS.md).

## Tests

```bash
cd src-tauri
cargo test                                   # unit tests across the workspace
cargo test --workspace --locked              # same suite as CI
```

Environment-gated integration tests are `#[ignore]`d by default. They are compiled as lib tests because the standalone integration-test harness has link issues on Windows.

**vLLM/WSL end-to-end** (provisions an isolated venv, starts two servers, checks `/health`, `/v1/models`, chat, `/metrics`, then stops both):

```powershell
$env:LLM_TEST_WSL = "1"
cargo test --lib -- --ignored --nocapture wsl_it
```

Requires ~5 GB disk and network; first run can take 20–60 minutes. It skips if `nvidia-smi` is unavailable in WSL. Server state is written to a temporary config directory, never your real config.

**Native llama.cpp end-to-end** (starts `llama-server.exe` with a tiny GGUF, checks health and chat, then verifies shutdown):

```powershell
$env:LLM_TEST_LLAMACPP = "1"
$env:LLM_TEST_LLAMACPP_EXE = "C:\path\to\llama-server.exe"
$env:LLM_TEST_LLAMACPP_MODEL = "C:\path\to\tiny.gguf"
cargo test --lib -- --ignored --nocapture llamacpp_it
```

CI (`.github/workflows/ci.yml`) runs release-metadata validation, frontend contract tests, TypeScript/Vite build, `cargo check`, workspace tests, a standalone artifact build, and dependency audits.

## First-run walkthrough

1. Open the app → **Dashboard** → **Provision WSL** (or **Install llama.cpp** if you only want the Windows backend). Progress streams into the provisioning log.
2. **Search** for a model, e.g. `Qwen2.5-0.5B`. Context, fit verdict, and tok/s are estimates based on your detected hardware.
3. Review the **quant variants** and optionally **Pull** one first. vLLM can also download at first start.
4. **Servers → New server** → choose the backend, model, and task (`instruct` or `embed`). Use **Context fit test** to validate your planned context length.
5. Press **Start**. Logs stream live, `/health` turns green, and the chat drawer becomes available. Measured tok/s replaces the estimate after the first generation run.
6. **Metrics** shows request, KV-cache, latency, and throughput series; the Dashboard shows VRAM/GPU sparklines.
7. Optional: **Settings → Advanced Mode → Network** → set an API key, enable the **OpenAI-Compatible Gateway**, and point any client at `http://127.0.0.1:11434/v1`.

### vLLM FlashInfer sampler workaround

vLLM's FlashInfer top-k/top-p sampler JIT-compiles a CUDA kernel on first use, requiring `nvcc`, a C compiler (`gcc`), and `ninja` inside WSL. Without them vLLM fails with `RuntimeError: Could not find nvcc`.

- **Recommended fix (default):** the panel sets `VLLM_USE_FLASHINFER_SAMPLER=0` globally, so vLLM uses its built-in PyTorch sampler and avoids JIT entirely.
- **Optional full toolkit:** **Dashboard → FlashInfer JIT Ready → Check → Install CUDA Build Tools** installs `gcc`, `python3.12-dev`, `ninja-build`, and a matching NVIDIA CUDA toolkit from the WSL-Ubuntu repo. This downloads several GB and takes 5–15 minutes.
- When a startup log contains a known failure pattern, the server card shows an actionable hint — including a one-click restart with the FlashInfer sampler disabled, and a pointer to the environment editor for malformed values.

### Native llama.cpp first run

1. **Dashboard → Install llama.cpp**. The installer lists recent releases, selects the highest Windows x64 CUDA build supported by your NVIDIA driver, installs the matching `cudart` archive, and probes `--version` / `--help`. Vulkan and CPU-only archives are never silently substituted.
2. In **Settings**, review the llama.cpp install directories and GGUF directory, or set a custom `llama-server.exe` path.
3. Download a GGUF from **Search** (native Windows download) or place an existing `.gguf` in the GGUF directory, or import it from the **Library** page.
4. In **Servers → New server**, select **llama.cpp (Windows)**, choose the channel (**Upstream** or **PrismML**), and pick the first shard for split files. Use the **MoE auto-fit** preset when the binary supports `--fit`, or set `n_cpu_moe` manually. The PrismML channel adds a **Bonsai 2 / ternary** preset.
5. Point an external harness at `http://127.0.0.1:<port>/v1`, or use the gateway for stable model discovery. Native logs persist under `%APPDATA%\local-llm-panel\logs`.
6. Use **Copy log** on a server's log pane when reporting startup failures.

GitHub API access uses a separate optional token; the HF token is never sent to GitHub. A rejected GitHub token is retried anonymously, with diagnostics explaining the fallback. If GitHub is unreachable, download the Windows x64 CUDA zip plus the matching `cudart` zip manually, extract them together, and set **Custom llama-server.exe** in Settings.

## Using the gateway

Enable it in **Settings → Advanced Mode → Network → OpenAI-Compatible Gateway**, and configure an **API Key** in the same section. The key is required for every `/v1/*` route; a gateway without one returns `503` rather than exposing models to any local process.

```bash
curl http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer $LOCAL_LLM_KEY"

curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $LOCAL_LLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

- Routing is exact-first (served model name or HF id), then fuzzy suffix, so `Qwen2.5-7B-Instruct` finds `Qwen/Qwen2.5-7B-Instruct`.
- `/v1/models` reports the effective context limit, quantization, task, and capability/thinking metadata for each running server, including hints when a capability needs a launch flag. The advertised `id` is the served model name or, for llama.cpp, the GGUF file stem rather than a full file path.
- Thinking metadata exposes a boolean `reasoning` plus a detailed `reasoning_config` (aliased as `thinking`) with standard `reasoning_effort` controls, levels, and OpenRouter/Pi metadata (`supported_efforts`, `mandatory`, `default_effort`, `thinkingLevelMap`). The gateway canonicalizes standard and extended effort labels across all model types.
- Authentication uses the `Authorization: Bearer <key>` header only. `/health` stays open for liveness checks.
- Requests that don't touch thinking or token-limit aliases are forwarded byte-for-byte — the gateway does not inject defaults.

## Configuration & data locations

| Data | Location |
|------|----------|
| Config, measured stats, benchmarks | `%APPDATA%\local-llm-panel\config.json` (atomic write) |
| Conversations | `%APPDATA%\local-llm-panel\conversations.json` |
| Native llama.cpp logs | `%APPDATA%\local-llm-panel\logs\<server-id>.log` |
| GGUF models (Windows) | `%APPDATA%\local-llm-panel\gguf` (configurable) |
| llama.cpp installs | `%APPDATA%\local-llm-panel\llama.cpp\<release-tag>` (configurable per channel) |
| vLLM venv / logs (WSL) | `~/llm-lp/.venv`, `~/llm-lp/logs` |
| HF cache (WSL) | `~/.cache/huggingface` (custom `HF_HOME` supported in Advanced) |

Tokens for Hugging Face and GitHub are stored as DPAPI-encrypted blobs and redacted from UI/config-export payloads.

## Notes & gotchas

- **Version sync:** bump all four files together — `package.json`, `package-lock.json`, `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`. The build script warns on mismatch.
- **Windows 11 24H2+ crash (0xc0000139):** handled by `src-tauri/.cargo/config.toml` forcing `getrandom_backend="windows_legacy"`.
- **`lld-link.exe` not found:** run builds through npm scripts, not raw `cargo build`; the build runner injects the LLVM path.
- **MSVC:** if `link.exe` is not on `PATH`, cargo finds it via vswhere. The `x86_64-pc-windows-gnu` fallback exists but is not needed on the dev machine.
- **Unknown GPUs** default to ~700 GB/s bandwidth in estimates; the estimate is always labeled and measured data wins.
- **WSL memory** is controlled by `.wslconfig`, which the app never edits silently. Settings shows it read-only with a copy action.
- **Gateway binds localhost** unless LAN access is explicitly enabled with the consent warning. Use an API key on any shared network.
- **Auto-shutdown WSL** only runs when no server, pull, provisioning job, or recent WSL command is active.
- **Release profile** keeps builds fast (`lto = false`, `codegen-units = 16`); MSI packaging is only included in `npm run build:all`.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for modules, data flow, commands, events, and security notes, and [`docs/PLAN.md`](docs/PLAN.md) / [`docs/ROADMAP.md`](docs/ROADMAP.md) for design history and future work.
