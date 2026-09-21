# Local LLM Panel

A native Windows desktop app (**Tauri 2** + React/TypeScript) that manages **vLLM** inside WSL2 and native Windows **llama.cpp** servers:

- **Auto-provisions** WSL2 (idempotent): apt basics → uv venv (CPython 3.12) → `vllm` CUDA wheels → torch/GPU verification.
- **Search & estimate** any Hugging Face model: live `config.json` context, VRAM context-fit, and per-GPU **estimated max tok/s** for your hardware.
- **Pull models** in the background (`hf download`, progress streamed to the UI).
- **Run multiple servers** concurrently (instruct + embedding) with per-server quantization (`--quantization`), GPU memory util, max-model-len, ports, and live log tails, metrics, and a minimal chat playground.
- **Run llama.cpp `llama-server.exe` natively on Windows**, including GGUF models, split-shard downloads, CUDA builds, Jinja tool calling, and MoE expert CPU offload.
- **Per-server & global environment variables** for vLLM servers (merged with per-server overriding global defaults).
- **FlashInfer JIT workaround**: `VLLM_USE_FLASHINFER_SAMPLER=0` disables vLLM's FlashInfer sampler (which requires nvcc/CUDA toolkit in WSL) and uses the built-in PyTorch sampler instead.
- **One-click CUDA build tools installer** for FlashInfer JIT readiness (installs gcc, python3.12-dev, ninja-build, and NVIDIA CUDA toolkit from WSL-Ubuntu repo).

## Requirements

- Windows 11 (WebView2), NVIDIA GPU + driver supporting CUDA ≥ 12.4 inside WSL2
- WSL2 with an Ubuntu distro (`.wslconfig` memory recommended: `memory=24GB` as the dev machine uses)
- For llama.cpp: an NVIDIA Windows driver and a recent CUDA-enabled llama.cpp release. WSL provisioning is not required for this backend.
- Rust toolchain (MSVC target; VS Build Tools "Desktop development with C++" — cargo picks VS up via vswhere), LLVM/LLD on PATH, Node ≥ 20

## Setup (dev)

```bash
npm install
npm run tauri dev        # builds web UI + Rust, launches app
```

Production builds & releases:

```bash
npm run check              # fast preflight type check + cargo check (~3s)
npm run build:fast         # fast standalone executable (~20s, no bundling)
npm run build:setup        # standard release: standalone + NSIS installer (~40s)
npm run build:all          # full distribution: standalone + NSIS + WiX MSI
```

Staged release binaries and installers output automatically to `dist-release/`.

The build runner (`scripts/build.mjs`) automatically locates LLVM's `lld-link.exe` (from `C:\Program Files\LLVM\bin` or Visual Studio VC tools) and enables high-speed linking.

For the full cross-agent deployment SOP, version synchronization guidelines, and GitHub release instructions, see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and [`skills/github-deployment/SKILL.md`](skills/github-deployment/SKILL.md).

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

### vLLM FlashInfer sampler workaround

vLLM's FlashInfer-based top-k/top-p sampler JIT-compiles a CUDA kernel on first use, which requires `nvcc` (CUDA toolkit), a C compiler (`gcc`), and `ninja` in WSL. If these are not installed, vLLM crashes with `RuntimeError: Could not find nvcc`.

**Quick fix (recommended):** The panel sets `VLLM_USE_FLASHINFER_SAMPLER=0` globally by default, which makes vLLM use its built-in PyTorch sampler and avoids the JIT entirely. This works on any WSL2 setup without additional tooling.

**Full CUDA toolkit (optional):** If you want FlashInfer's optimized sampling, use **Dashboard → FlashInfer JIT Ready → Check → Install CUDA Build Tools**. This installs `gcc`, `python3.12-dev`, `ninja-build`, and the NVIDIA CUDA toolkit (matching your PyTorch CUDA version) from the official WSL-Ubuntu repository. **Warning:** downloads several GB and takes 5–15 minutes.

### Native llama.cpp first run

1. Open **Dashboard** and use **Install / update llama.cpp**. The app selects the latest Windows CUDA release, verifies `llama-server.exe --version`, and records its help capabilities.
2. In **Settings**, review the native llama.cpp and GGUF directories, or set a custom `llama-server.exe` path.
3. Download a GGUF from **Search** or place an existing `.gguf` in the GGUF directory.
4. In **Servers → New server**, select **llama.cpp (Windows)**, choose the first shard for split files, and choose either the **MoE auto-fit** or **MoE manual offload** preset. Automatic fit is enabled by default when the installed binary supports `--fit`; otherwise the configured `n_cpu_moe` value is used as the manual fallback.
5. Point an external harness at the displayed `http://127.0.0.1:<port>/v1` URL. Native logs persist under `%APPDATA%\local-llm-panel\logs`.

The llama.cpp backend binds to localhost only. The installer uses a separate optional GitHub token for `api.github.com`; the Hugging Face token is never sent to GitHub. A rejected GitHub token is retried anonymously, while rate-limit and network diagnostics explain the fallback. If GitHub is unavailable, download the Windows x64 CUDA zip plus matching cudart zip manually, extract them together, and use **Custom llama-server.exe** in Settings. The installer lists recent GitHub releases, parses exact Windows x64 CUDA and matching untagged `cudart` archives, and chooses the highest build supported by the Windows NVIDIA driver. It prefers CUDA 12.8+ for Blackwell GPUs, but allows an older compatible build with an explicit warning; it never silently substitutes Vulkan or CPU-only archives. If a newer release is still uploading assets, the installer falls back to the newest release with a complete CUDA pair and reports the skipped tags. Settings exposes GitHub access testing, token clearing, the native CUDA/device probe, and each server can select device IDs, an API key, fit target (MiB), and log verbosity. `n_cpu_moe` is a manual fallback/tuning value: increase it if VRAM is exhausted, and leave RAM headroom for the operating system and context/KV cache. Use **Copy log** on a server's log pane when reporting startup failures.

## Notes & gotchas

- MSVC: if `link.exe` is not on PATH but VS is installed, cargo finds it via vswhere. Fallback target `x86_64-pc-windows-gnu` if linking ever fails (not needed on this machine).
- GPU bandwidth for unknown GPUs defaults to ~700 GB/s; the estimate is always labeled and measured data wins.
- Config lives at `%APPDATA%\local-llm-panel\config.json` (distro, HF token, server definitions, measured stats).
- The app never edits `.wslconfig` silently — Settings shows it read-only (copy to apply tweaks).

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/PLAN.md`](docs/PLAN.md).