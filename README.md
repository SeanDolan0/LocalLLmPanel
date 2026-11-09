# Local LLM Panel

[![CI](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/codeql.yml)
[![Security Audit](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/security-audit.yml/badge.svg)](https://github.com/SeanDolan0/LocalLLmPanel/actions/workflows/security-audit.yml)
[![Release](https://img.shields.io/github/v/release/SeanDolan0/LocalLLmPanel?include_prereleases&style=flat-square&color=indigo)](https://github.com/SeanDolan0/LocalLLmPanel/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Platform: Windows 10 / 11](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20x64-0078D6?style=flat-square&logo=windows)](https://www.microsoft.com/windows)
[![Tauri v2](https://img.shields.io/badge/Tauri-v2-FFC131?style=flat-square&logo=tauri&logoColor=black)](https://v2.tauri.app)

A native Windows desktop control panel for discovering, sizing, downloading, orchestrating, and monitoring local large language models. It brings **high-throughput vLLM in WSL2** and **native llama.cpp on Windows** together behind one unified interface and an OpenAI-compatible API gateway.

> **Local inference, hardware-aware memory planning, and unified server orchestration.** Estimates are estimates - measured performance wins.

---

## Architecture Overview

```mermaid
graph TD
    UI["Local LLM Panel UI<br/>(React 18 + TypeScript + Vite)"]
    IPC["Tauri 2 IPC Bridge<br/>(Type-safe Commands and Events)"]
    
    subgraph Core ["Rust Desktop Core (src-tauri)"]
        SPV["Process Supervisor & Health Watcher"]
        CALC["Hardware Detector & KV-Cache Sizer"]
        GW["OpenAI API Reverse Proxy (:11434/v1)"]
        VAULT["DPAPI Encrypted Credential Store"]
        DL["Multi-part Download Orchestrator"]
        CLN["Guided Cleanup & Storage Manager"]
    end

    subgraph Backends ["Local Inference Engines"]
        VLLM["vLLM Engine (WSL2 Ubuntu)<br/>PagedAttention · AWQ · Safetensors"]
        LLAMA["llama.cpp Server (Native Windows CUDA)<br/>GGUF · GPU/CPU Offloading · Vision"]
    end

    UI --> IPC
    IPC --> Core
    SPV --> VLLM
    SPV --> LLAMA
    GW --> VLLM
    GW --> LLAMA
```

Read more about the design principles and comparative analysis in [ABOUT.md](ABOUT.md).

---

## What It Does

| Area | Capabilities |
|---|---|
| **Model Discovery & Shards** | Search Hugging Face and GGUF repositories, inspect parameter counts, quantization formats, context lengths, and verify multi-file GGUF / safetensors shard sets. |
| **Hardware Fit Planning** | Real-time memory accounting estimating weights, KV-cache footprint across context lengths, and GPU/RAM offload boundaries. Analytical estimates help avoid OOM crashes prior to launch. |
| **Dual Inference Engines** | Run high-throughput **vLLM** via WSL2 or lightweight **llama.cpp** native Windows CUDA processes side-by-side. |
| **Unified OpenAI Gateway** | Route standard OpenAI `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` requests to active servers with API key protection. Bound to localhost (`127.0.0.1`) by default. |
| **Resilient Download Manager** | Resilient background downloads with multi-connection chunked streaming, pause/resume staging via `.part` files, preflight disk space checks, ETA tracking, and shard integrity verification. |
| **Guided Environment Cleanup** | Safe dry-run inspection and itemized cleanup to reclaim disk space across WSL distributions, virtual environments, Hugging Face caches, GGUF models, and partial downloads. |
| **Live Telemetry & Benchmarks** | Monitor real-time GPU VRAM, compute usage, process logs, token generation rates, and conduct repeatable benchmarks. |
| **Zero Telemetry & DPAPI Security** | 100% localhost-bound by default. Sensitive tokens (Hugging Face, GitHub, Gateway keys) protected at rest using Windows DPAPI encryption. Zero remote telemetry. |

---

## Hardware & Sizing Reference

Local LLM Panel automatically reads your GPU's physical VRAM and calculates achievable context sizes. Below is a general guide for NVIDIA RTX workstations:

| GPU Tier | Typical VRAM | Recommended Models & Formats | Context Capacity |
|---|---|---|---|
| **Entry** | 8 GB | Llama 3.2 3B (Q8 / FP16), Qwen 2.5 7B (Q4_K_M), Mistral 7B (Q4_K_M) | 4K – 8K tokens |
| **Mid-Range** | 12 GB | Qwen 2.5 7B (Q8 / AWQ), Llama 3.1 8B (Q8 / FP16), Gemma 2 9B (Q5_K_M) | 8K – 32K tokens |
| **Enthusiast** | 16 GB | Qwen 2.5 14B (Q4_K_M / AWQ), DeepSeek-R1-Distill-Qwen-14B (Q4_K_M), Mixtral 8x7B (Q3_K_M) | 8K – 64K tokens |
| **High-End** | 24 GB | Qwen 2.5 32B (Q4_K_M / AWQ), Llama 3.1 70B (IQ2_XS offload), DeepSeek-R1-Distill-32B | 16K – 128K tokens |

> **Understanding Sizing Estimates:**
> Fit calculations provide analytical guidance based on model architecture metadata (layers, attention heads, head dimension, quantization bits per weight, and KV-cache precision) plus runtime framework overhead:
>
> $$\text{VRAM}_{\text{projected}} \approx \text{Weights} + \text{KV-Cache}(L, H, D, C_{\text{ctx}}, P) + \text{CUDA Runtime Overhead}$$
>
> These numbers are planning estimates to prevent out-of-memory crashes on server launch, rather than hard runtime guarantees. Measured benchmark throughput and empirical memory residency take precedence.

---

## Quick Start

### Prerequisites

- **OS:** Windows 10 (version 2004+ / build 19041+) or Windows 11 x64 with WebView2 Evergreen Runtime installed.
- **Hardware:** NVIDIA GPU with current Game Ready or Studio Drivers (CUDA 12.x compatible).
- **For vLLM:** WSL2 with Ubuntu 22.04 / 24.04 and NVIDIA CUDA WSL drivers.
- **For llama.cpp:** No WSL required; panel can download compatible Windows CUDA `llama-server.exe` binaries directly.
- **For Development:** Node.js 22+, Rust 1.80+ (MSVC target), Visual Studio 2022 C++ Build Tools, and LLVM/LLD.

### Running from Source

```powershell
# Clone the repository
git clone https://github.com/SeanDolan0/LocalLLmPanel.git
cd LocalLLmPanel

# Install frontend dependencies
npm ci

# Launch desktop app in development mode
npm run tauri dev
```

### Building Standalone Executable & Installers

```powershell
# Build fast standalone Windows executable (dist-release/LocalLLMPanel-v*.exe)
npm run build:fast

# Build complete NSIS setup installer (.exe)
npm run build:setup

# Build all release packages (standalone exe, NSIS setup, and MSI package)
npm run build:all
```

### Windows SmartScreen & Binary Verification

Community and draft release binaries are unsigned (they do not carry a paid commercial Authenticode EV certificate). When executing a downloaded binary or installer on Windows:

1. **SmartScreen Warning:** Windows Defender SmartScreen may display **"Windows protected your PC"**.
2. **Bypassing the Warning:** Click **More info**, then click **Run anyway**.
3. **Verifying Cryptographic Integrity:** You can independently verify that your downloaded binary has not been tampered with by checking its SHA-256 hash using PowerShell:

```powershell
# Calculate SHA-256 hash of downloaded executable or installer
Get-FileHash -Algorithm SHA256 .\dist-release\LocalLLMPanel-*.exe

# Compare against published checksum file
Get-Content .\dist-release\LocalLLMPanel-*.exe.sha256
```

---

## OpenAI-Compatible Gateway

The panel includes a high-performance reverse proxy that presents a standard OpenAI-compatible API surface for external applications (Continue, Cursor, Open WebUI, LangChain, LlamaIndex, etc.).

### Endpoint Compatibility

| Endpoint | Method | Status | Notes |
|---|---|---|---|
| `/v1/responses` | POST | **Supported** | OpenAI Responses API compatibility layer. Maps `input` (text or message array) and `instructions` to active models. Returns Responses-shaped JSON (`id: resp_...`, `object: response`, `status`, `output`, `output_text`, `usage`, `incomplete_details`). Supports streaming SSE events (`response.created`, `response.in_progress`, `response.output_item.added`, `response.output_text.delta`, `response.output_item.done`, `response.completed`/`response.incomplete`, `data: [DONE]`). |
| `/v1/chat/completions` | POST | **Supported** | Full Chat Completions API with model-aware routing, streaming SSE, tool calling, reasoning/thinking normalizations, `stream_options.include_usage`, and authoritative terminal `finish_reason`. |
| `/v1/completions` | POST | **Supported** | Legacy text/FIM completions (prompt, suffix). Explicitly validates and rejects `messages` with 400. |
| `/v1/embeddings` | POST | **Supported** | Generates vector embeddings. Accepts `input` as string, string array, 1D token array, or 2D batch token array. Supports `encoding_format: "float"` and `"base64"` (little-endian IEEE 754), `dimensions` truncation, and usage. |
| `/v1/models` | GET | **Supported** | Lists all running instruct and embedding models with context windows, permissions, and modality capabilities. |
| `/v1/models/{model}` | GET | **Supported** | Retrieves metadata for a specific running model. Every model returned by `/v1/models` is retrievable. Non-GET methods return 405. |
| `/health`, `/v1/health` | GET | **Supported** | Open liveness check endpoint (no auth required). |
| `/v1/responses/{id}` | GET / DELETE | **Unsupported (404/405)** | Response retrieval, cancellation, and deletion lifecycles are not supported on local engines. |
| `/v1/conversations/*` | ALL | **Unsupported (404)** | Hosted conversational state persistence is unsupported; manage history client-side. |
| `previous_response_id`, `store: true` | - | **Unsupported (400)** | Returns structured 400 error. The gateway does not silently claim support for hosted state storage. |

### Configuration & Authentication

- **Base URL:** `http://127.0.0.1:11434/v1` (or custom port configured under **Settings → Advanced Mode → Network**).
- **Authentication:** All `/v1/*` routes require `Authorization: Bearer YOUR_GATEWAY_API_KEY` (or `api-key: YOUR_GATEWAY_API_KEY`).
  > **Note:** The gateway API key is configured locally in LocalLLmPanel (**Settings → Gateway API Key**). It is a local access token designed to protect loopback endpoints from unauthorized local processes and malicious browser scripts. It is **not** an OpenAI hosted platform key.
- **Request IDs:** Every response includes an `x-request-id` header (generated or forwarded from `X-Client-Request-Id`).
- **Standardized Errors:** All errors return JSON matching the OpenAI error schema: `{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}`.

### LAN Access Requirements

By default, the gateway and all inference servers bind strictly to `127.0.0.1` (localhost). Exposing services to your local area network (`0.0.0.0`) is an explicit administrative opt-in configured under **Settings → Advanced Mode → Network**:

1. **Mandatory Gateway API Key:** A non-empty Gateway API Key must be set before binding to `0.0.0.0`.
2. **Windows Defender Firewall Rule:** Windows will block incoming network requests unless an inbound rule is created:
   ```powershell
   New-NetFirewallRule -DisplayName "Local LLM Panel Gateway" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
   ```
3. **WSL2 Port Proxy (for vLLM):** If external devices need to connect directly to vLLM running inside WSL2, configure a Windows port forwarding rule:
   ```powershell
   # Find WSL IP via 'wsl hostname -I'
   netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=<WSL_IP>
   ```

### Example: Responses API (cURL)

```bash
curl http://127.0.0.1:11434/v1/responses \
  -H "Authorization: Bearer YOUR_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "input": "Write a Rust function to calculate Fibonacci numbers.",
    "instructions": "You are an expert systems programmer."
  }'
```

### Example: Streaming Chat Completion (cURL)

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful programming assistant."},
      {"role": "user", "content": "Write a Rust function to calculate Fibonacci numbers."}
    ],
    "stream": true
  }'
```

### Example: Python (`openai` SDK)

```python
from openai import OpenAI

# Connect to the local LocalLLmPanel gateway
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="YOUR_GATEWAY_API_KEY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain how PagedAttention works in vLLM."}],
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Development & Test Commands

| Target | Command | Purpose |
|---|---|---|
| **Development App** | `npm run tauri dev` | Starts Vite hot-reload frontend with debug Rust binary |
| **Preflight Check** | `npm run check` | Validates TypeScript, manifests, and Rust workspace |
| **Frontend Smoke Tests** | `npm run test:contract` | Verifies Tauri command contracts, routes, and security |
| **Frontend Build** | `npm run build` | Compiles TypeScript and runs Vite production bundling |
| **Rust Code Formatting** | `cd src-tauri; cargo fmt --all -- --check` | Validates formatting against Rust style guide |
| **Rust Linter** | `cd src-tauri; cargo clippy --workspace --all-targets --locked` | Runs Clippy static analysis across all crates |
| **Rust Unit Tests** | `cd src-tauri; cargo test --workspace --locked` | Executes comprehensive workspace test suite (500+ tests) |
| **Security Audit Check** | `npm run audit:check` | Verifies Cargo and npm vulnerabilities against allowlist |
| **Standalone Executable** | `npm run build:fast` | High-speed release build using LLVM lld-link |
| **NSIS Setup Installer** | `npm run build:setup` | Generates Windows installer and SHA-256 checksums |
| **All Release Packages** | `npm run build:all` | Compiles standalone exe, NSIS setup, and MSI package |
| **Release Validation** | `npm run release:check` | Confirms synchronized versioning across all manifests |

---

## Repository Structure

```text
LocalLLmPanel/
├── .github/
│   ├── ISSUE_TEMPLATE/        # Structured GitHub Issue Forms (bug, feature, model)
│   ├── workflows/             # GitHub Actions CI, CodeQL, release, and audit pipelines
│   ├── dependabot.yml         # Automated weekly dependency updates
│   └── pull_request_template.md
├── docs/
│   ├── USER_GUIDE.md          # Comprehensive user manual and troubleshooting
│   ├── ARCHITECTURE.md        # Technical architectural design document
│   ├── ROADMAP.md             # Project roadmap and milestones
│   └── DEPLOYMENT.md          # Packaging and release procedure
├── scripts/
│   ├── build.mjs              # Multi-target build runner with LLD optimization
│   ├── frontend-smoke.mjs     # Automated contract and security test suite
│   ├── security-audit.mjs     # Automated cargo and npm audit gate with allowlist
│   └── validate-release.mjs   # Release manifest and artifact validator
├── src/                       # React 18 frontend (Vite + TypeScript + Tailwind)
│   ├── components/            # Reusable UI components and modal dialogs
│   ├── pages/                 # Dashboard, Search, Library, Servers, Settings
│   └── api.ts                 # Type-safe Tauri IPC wrapper
├── src-tauri/                 # Rust application backend (Tauri 2)
│   ├── crates/llmfit-core/    # Hardware detection, model fitting, and context sizing library
│   └── src/                   # Server supervisor, WSL bridge, gateway, DPAPI vault, cleanup
├── ABOUT.md                   # Project background and comparative analysis
├── CONTRIBUTING.md            # Contributor guidelines and workflow
├── SECURITY.md                # Security reporting and cryptographic guarantees
├── LICENSE                    # MIT License
└── README.md                  # Project landing page
```

---

## Documentation Links

- [User Guide](docs/USER_GUIDE.md) — Step-by-step first run, FlashInfer setup, GGUF offloading, and troubleshooting.
- [About & Principles](ABOUT.md) — Design philosophy, architectural goals, and ecosystem comparisons.
- [Architecture](docs/ARCHITECTURE.md) — Deep dive into modules, state machines, and Tauri IPC commands.
- [Contributing](CONTRIBUTING.md) — Pull request expectations, style rules, and developer guidelines.
- [Security Policy](SECURITY.md) — Vulnerability reporting, encryption details, and localhost assurances.
- [License](LICENSE) — MIT License.
