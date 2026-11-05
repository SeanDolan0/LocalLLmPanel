# Local LLM Panel

[![CI](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/codeql.yml)

A native Windows desktop control panel for discovering, downloading, running, and monitoring local LLMs. It brings **vLLM in WSL2** and **llama.cpp on Windows** together behind one interface and an optional OpenAI-compatible API.

> Local inference, hardware-aware model selection, and one place to manage your servers. Estimates are estimates — measured performance wins.

## About

Local LLM Panel is designed for people who want to run capable language models on their own NVIDIA-powered Windows machines without manually coordinating model files, Python environments, inference servers, ports, and monitoring tools.

The application combines a React/TypeScript interface with a Rust backend in a Tauri 2 desktop app. Model discovery and fit estimates help narrow down what a machine can realistically run; server management handles the inference processes; and the optional gateway gives other applications one stable API endpoint.

Model discovery and downloads use online services such as Hugging Face and GitHub. Once a model is available locally, inference runs through the configured local backend.

See [ABOUT.md](ABOUT.md) for the project's goals and design principles.

## What it does

| Area | Capabilities |
|---|---|
| **Discover models** | Search Hugging Face, inspect model metadata and quantizations, compare hardware fit, and review estimated throughput. |
| **Plan memory** | Estimate weight and KV-cache use, account for GPU/RAM offload, and test a proposed context length before launch. |
| **Manage downloads** | Track background downloads, progress, speed, ETA, pause/cancel operations, GGUF shards, and companion vision files. |
| **Run inference** | Manage multiple vLLM and llama.cpp servers with per-server settings, logs, health checks, and restart controls. |
| **Use one API** | Route OpenAI-compatible chat, completion, and embedding requests through a single optional gateway. |
| **Measure performance** | View live server/GPU metrics and compare measured benchmark results with estimates. |
| **Manage your setup** | Maintain the local model library, export/import configuration, manage llama.cpp forks, and use tray/appliance mode. |

## Supported backends

| Backend | Where it runs | Model formats |
|---|---|---|
| **vLLM** | Ubuntu under WSL2 | Hugging Face model repositories and supported quantizations |
| **llama.cpp** | Native Windows process | GGUF, including split models and compatible vision projectors |

The application is localhost-only by default. LAN access is an explicit opt-in setting.

## Requirements

- Windows 11 x64 with WebView2.
- NVIDIA GPU and a current NVIDIA driver.
- **For vLLM:** WSL2 with an Ubuntu distribution and CUDA access from inside WSL2.
- **For llama.cpp:** a compatible Windows CUDA build of `llama-server.exe`; WSL2 is not required for this backend.
- **For development:** Node.js 22 or newer, stable Rust with the MSVC target, Visual Studio C++ Build Tools, and LLVM/LLD.

A GitHub token is optional for some llama.cpp release downloads. A Hugging Face token is optional unless you need gated repositories.

## Quick start

### Run from source

Install the development prerequisites, then run:

```powershell
npm ci
npm run tauri dev
```

The first launch opens the desktop app. From the Dashboard, provision WSL for vLLM or install a native llama.cpp build.

### First run

1. Open **Dashboard** and choose **Provision WSL** or **Install llama.cpp**, depending on the backend you plan to use.
2. Open **Search**, find a model, and review its available quantizations and hardware-fit estimates.
3. Pull a model when you want to download it ahead of time. vLLM can also download Hugging Face models when a server starts.
4. Go to **Servers → New server**, select a backend, model, and task, then review the context fit.
5. Start the server. Use its logs and health status to diagnose startup issues, and use Chat or Benchmarks to exercise it.

For detailed setup notes, FlashInfer troubleshooting, GGUF/fork setup, speculative decoding, and gateway behavior, see the [User Guide](docs/USER_GUIDE.md).

## OpenAI-compatible gateway

Enable the gateway and configure an API key under **Settings → Advanced Mode → Network**. The default endpoint is:

```text
http://127.0.0.1:11434/v1
```

Example request:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $LOCAL_LLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

The gateway supports chat/completions, legacy completions, and embeddings, and routes requests to the appropriate running server. Its `/health` endpoint is unauthenticated for liveness checks; all `/v1/*` routes require the configured bearer key.

## Development and tests

Install dependencies with `npm ci`. The most useful commands are:

| Task | Command |
|---|---|
| Start the desktop app | `npm run tauri dev` |
| Fast preflight checks | `npm run check` |
| Frontend contract tests | `npm run test:contract` |
| TypeScript + Vite build | `npm run build` |
| Rust formatting check | `cd src-tauri; cargo fmt --all -- --check` |
| Rust linting | `cd src-tauri; cargo clippy --workspace --all-targets --locked -- -D warnings` |
| Rust unit tests | `cd src-tauri; cargo test --workspace --locked` |
| Standalone Windows executable | `npm run build:fast` |
| NSIS installer + executable | `npm run build:setup` |
| Full distribution including MSI | `npm run build:all` |
| Release metadata validation | `npm run release:check` |

Build outputs are staged in `dist-release/`.

The GitHub Actions CI workflow runs on Windows and validates release metadata, frontend contracts, formatting, Clippy, the TypeScript/Vite build, Rust checks/tests, a standalone executable build, and npm/Rust dependency audits. A successful CI build also uploads the standalone executable and SHA-256 checksum as a short-lived workflow artifact.

### Hardware-dependent integration tests

The default Rust test suite does not provision WSL or launch real inference servers. Those tests are intentionally ignored unless explicitly enabled.

**vLLM / WSL2** (requires WSL2 Ubuntu, an NVIDIA GPU visible inside WSL, network access, and several GB of free disk):

```powershell
$env:LLM_TEST_WSL = "1"
cd src-tauri
cargo test --lib -- --ignored --nocapture wsl_it
```

**Native llama.cpp** (requires a compatible executable and a small local GGUF):

```powershell
$env:LLM_TEST_LLAMACPP = "1"
$env:LLM_TEST_LLAMACPP_EXE = "C:\path\to\llama-server.exe"
$env:LLM_TEST_LLAMACPP_MODEL = "C:\path\to\tiny.gguf"
cd src-tauri
cargo test --lib -- --ignored --nocapture llamacpp_it
```

These tests are not run on standard hosted CI because they require real local GPU, WSL, and model-file prerequisites.

## Documentation

- [About the project](ABOUT.md) — purpose, audience, architecture, and design principles.
- [User Guide](docs/USER_GUIDE.md) — first-run details, advanced features, and troubleshooting.
- [Architecture](docs/ARCHITECTURE.md) — modules, data flow, Tauri commands/events, and security notes.
- [Roadmap](docs/ROADMAP.md) — completed work and planned improvements.
- [Deployment Guide](docs/DEPLOYMENT.md) — versioning, packaging, and release SOP.
- [Contributing](CONTRIBUTING.md) — development workflow and pull request expectations.
- [Agent reference](AGENTS.md) — repository-specific commands and implementation notes.

## Security and privacy

- Hugging Face and GitHub tokens are protected with Windows DPAPI and redacted from exported configuration.
- The gateway and servers bind to localhost by default; LAN exposure requires an explicit opt-in.
- The application manages local inference processes; model discovery and downloads still require network access.
- Do not publish configuration files, authentication tokens, private model paths, or unredacted logs in issues.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.
