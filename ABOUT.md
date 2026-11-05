# About Local LLM Panel

Local LLM Panel is an open-source Windows desktop application that unifies local large language model (LLM) discovery, download orchestration, server lifecycle, and API routing behind a responsive, hardware-aware desktop control panel.

Running large language models locally on consumer workstations typically requires wrestling with fragmented tools: manual WSL2 environment scripts, bespoke Python virtualenvs, mismatched CUDA driver bindings, CLI command flags, separate Hugging Face download managers, and ad-hoc reverse proxies.

**Local LLM Panel brings order to the local inference ecosystem** by combining **vLLM (under WSL2)** and **native llama.cpp (on Windows)** with an intuitive interface, realistic VRAM and context planning, and an OpenAI-compatible gateway.

---

## How It Compares

| Feature | Local LLM Panel | Ollama | LM Studio | Raw CLI (vLLM / llama-server) |
|---|---|---|---|---|
| **Target Platform** | Windows 11 (NVIDIA CUDA) | Cross-platform | Cross-platform | Linux / Windows CLI |
| **Inference Engines** | **Dual:** vLLM (WSL2) + llama.cpp (Windows) | Custom llama.cpp fork | llama.cpp (bundled) | Single engine per command |
| **High-Throughput vLLM** | Yes (PagedAttention, FlashInfer, AWQ, FP8) | No | No | Yes (requires manual Linux/WSL setup) |
| **GGUF / Offloading** | Yes (native CUDA `llama-server.exe`) | Yes | Yes | Yes (manual CLI flags) |
| **Hardware Fit Estimator** | Real-time VRAM, weights + KV-cache math | Basic heuristic | Context slider estimate | None (OOM on launch if oversized) |
| **OpenAI Gateway** | Built-in reverse proxy with bearer auth | Built-in proxy | Built-in local server | Manual port coordination |
| **Configuration Model** | Single JSON config with DPAPI-encrypted secrets | Environment variables / CLI | Internal application DB | Scripts / shell variables |
| **Open Source** | 100% MIT Licensed | MIT Licensed | Proprietary | Apache 2.0 / MIT |

---

## What the Project Brings Together

1. **Hardware-Aware Discovery & Planning:**
   - Real-time catalog search via Hugging Face and GGUF repositories.
   - Exact memory accounting: weights footprint, KV cache requirements per context window, and attention head overhead.
   - Distinguishes theoretical estimates from measured benchmark throughput.

2. **Dual Local Inference Backends:**
   - **vLLM in WSL2:** Maximum token throughput for batched, concurrent, and high-performance Hugging Face / Safetensors models with PagedAttention and FlashInfer.
   - **llama.cpp on Windows:** Native GGUF inference, partial GPU/CPU offloading, split tensors, and vision projector support without WSL overhead.

3. **Resilient Download & Model Lifecycle:**
   - Multi-connection Hugging Face snapshots and GGUF shard downloads with pause/resume support.
   - Disk space verification and automatic model library indexing.

4. **Shared API Surface:**
   - Optional OpenAI-compatible gateway routing `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` to currently running servers.
   - Bearer token authentication and strict CORS policy for security.

5. **Operational Diagnostics & Telemetry:**
   - Real-time server process logs, health check monitoring, VRAM/GPU utilization telemetry, and reproducible benchmarks.

---

## Architectural Principles

```
┌────────────────────────────────────────────────────────┐
│               Local LLM Panel UI (React/TS)            │
└───────────────────────────┬────────────────────────────┘
                            │ Tauri IPC (Commands & Events)
┌───────────────────────────▼────────────────────────────┐
│                  Rust Backend (Tauri 2)                │
│  - Process Supervisor     - Model Registry / Catalog   │
│  - Hardware Detector      - Memory & KV Fit Calculator │
│  - OpenAI Reverse Proxy   - DPAPI Credential Vault     │
└──────────────┬────────────────────────────┬────────────┘
               │ Process Control            │ Process Control
┌──────────────▼────────────┐  ┌────────────▼────────────┐
│      WSL2 Ubuntu Subsys   │  │   Native Windows Space  │
│   vLLM Inference Server   │  │   llama-server.exe CUDA │
│  (HF / AWQ / Safetensors) │  │   (GGUF / Vision / MoE) │
└───────────────────────────┘  └─────────────────────────┘
```

1. **Measured Performance Over False Precision:**
   Calculated fit models and token-per-second projections are starting estimates. Actual performance is measured and recorded in repeatable local benchmarks.
2. **Make Resource Constraints Transparent:**
   Model parameters, quantization format, KV-cache context length, and system memory overhead are surfaced prior to launching inference, preventing unexpected CUDA out-of-memory crashes.
3. **Local by Default:**
   All services bind strictly to `127.0.0.1`. LAN exposure is an explicit, opt-in administrative toggle.
4. **Zero Telemetry & User Privacy:**
   No telemetry, tracking, or external analytic pings. Credentials (Hugging Face tokens, GitHub tokens) are sealed in Windows DPAPI storage.
5. **Composability:**
   Local LLM Panel manages and monitors standard, unadulterated inference binaries rather than wrapping proprietary closed engines.

---

## Repository Documentation

- [User Guide](docs/USER_GUIDE.md) — Comprehensive operational instructions, provisioning, and troubleshooting.
- [Architecture Guide](docs/ARCHITECTURE.md) — Detailed internal subsystem design, Tauri command map, and state machine.
- [Contributing Guide](CONTRIBUTING.md) — Development setup, coding guidelines, and pull request procedures.
- [Security Policy](SECURITY.md) — Responsible disclosure policy and security protections.
- [Roadmap](docs/ROADMAP.md) — Current release status and planned enhancements.
