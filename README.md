# Local LLM Panel

A native Windows desktop app (**Tauri 2** + React/TypeScript) that manages a **vLLM** installation inside **WSL2**:

- **Auto-provisions** WSL2 (idempotent): apt basics → uv venv (CPython 3.12) → `vllm` CUDA wheels → torch/GPU verification.
- **Search & estimate** any Hugging Face model: live `config.json` context, VRAM context-fit, and per-GPU **estimated max tok/s** for your hardware.
- **Pull models** in the background (`hf download`, progress streamed to the UI).
- **Run multiple servers** concurrently (instruct + embedding) with per-server quantization (`--quantization`), GPU memory util, max-model-len, ports, and live log tails, metrics, and a minimal chat playground.

## Requirements

- Windows 11 (WebView2), NVIDIA GPU + driver supporting CUDA ≥ 12.4 inside WSL2
- WSL2 with an Ubuntu distro (`.wslconfig` memory recommended: `memory=24GB` as the dev machine uses)
- Rust toolchain (MSVC target; VS Build Tools "Desktop development with C++" — cargo picks VS up via vswhere), Node ≥ 20

## Setup (dev)

```bash
npm install
npm run tauri dev        # builds web UI + Rust, launches app
```

Production bundle:

```bash
npm run tauri build
```

Binary outputs to `src-tauri/target/release/bundle/msi|nsis/`.

## Test

```bash
cd src-tauri
cargo test                      # unit tests (estimate math, port alloc, config round-trip)
LLM_TEST_WSL=1 cargo test --test integration_wsl -- --nocapture --ignored
                                # environment-gated integration: real vLLM install +
                                #   Qwen2.5-0.5B-Instruct + bge-small embedding concurrently
```

The integration test is `#[ignore]`-gated: it provisions WSL (heavy first run), starts two servers,
checks `/health`, `/v1/models`, chat completion, `/metrics` counters, then stops both and verifies the PIDs die.
Requires ~5 GB disk + network. If `nvidia-smi` is unavailable inside WSL, the test skips.

## First run walkthrough

1. Open the app → **Dashboard** → click **Provision WSL** (watch `wsl-log` lines stream in).
2. **Search** for a model, e.g. `Qwen2.5-0.5B` — context/tok-s columns are estimates.
3. Optionally **Pull** it first (or let vLLM auto-download at first start).
4. **Servers → New server** → pick the model, task `instruct` (or `embed`), press Start.
5. Logs stream live; the chat drawer works once `/health` is green; VRAM gauge updates on Dashboard.
6. Measured tok/s replaces the estimate after the first generation run.

## Notes & gotchas

- MSVC: if `link.exe` is not on PATH but VS is installed, cargo finds it via vswhere. Fallback target `x86_64-pc-windows-gnu` if linking ever fails (not needed on this machine).
- GPU bandwidth for unknown GPUs defaults to ~700 GB/s; the estimate is always labeled and measured data wins.
- Config lives at `%APPDATA%\local-llm-panel\config.json` (distro, HF token, server definitions, measured stats).
- The app never edits `.wslconfig` silently — Settings shows it read-only (copy to apply tweaks).

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/PLAN.md`](docs/PLAN.md).