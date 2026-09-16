# Local vLLM Panel for Windows (Tauri 2 + WSL2)

## Summary
A native Windows desktop app that manages a bare-bones **vLLM installation inside WSL2**: provisions it automatically, lets you **search/estimate/pull any Hugging Face model** (showing a per-model **estimated max tokens/s** and **max context window** for *your* GPU), and runs **multiple vLLM servers concurrently** (e.g., an embedding model + an instruct model) with per-server flags tuned for max throughput. Development is **git-tracked** from day one, and the repo contains `docs/PLAN.md` (this plan) plus `docs/ARCHITECTURE.md`.

Locked decisions (from user):
1. **Tauri 2** (Rust core + React/TS web UI)
2. **App provisions everything** in WSL2 (not just guide/verify)
3. **Live HF API + heuristic** for model stats (real context from `config.json`; tok/s + context-fit estimated from GPU specs)

## Environment facts (verified at implementation start)
- WSL2 with `.wslconfig` `memory=24GB`; default distro **Ubuntu-22.04** (`wsl -l -q`; works via `wsl -e`)
- NVIDIA driver 610.62 + CUDA 12.8 toolkit on Windows; inside WSL: `nvidia-smi` reports **NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB** (Blackwell sm120)
- Rust 1.95.0 / cargo 1.95.0, target `x86_64-pc-windows-msvc`; node 24 / npm 11
- MSVC: VS 2022 Community present at `...\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` (found via vswhere by cargo, not on PATH)
- WebView2 153.x present (Windows 11)
- WSL Ubuntu: Python 3.10.12, `uv` at `~/.local/bin/uv`, `curl` — but no `python3-venv`/`pip` guarantee → provision installs them. System python 3.10 may be too old for latest vLLM → provision prefers a **uv-managed CPython 3.12** (`uv python install 3.12 && uv venv --python 3.12`); fallback to system python venv.
- HF API reachable from this machine (200 in ~0.3s, no auth needed for search).
- GPU bandwidth entry for estimate table: RTX 5070 Ti Laptop ⇒ **672 GB/s** (GDDR7 192-bit); fallback ~700 GB/s with label.

## Architecture & behavior

### App layout
- `src-tauri/` — Rust core: `wsl.rs` (distro detection via `wsl -l -q`, command runner), `provision.rs`, `server.rs` (multi-instance lifecycle), `hf.rs` (search/config/params/pull), `estimate.rs`, `state.rs` (persisted JSON config + measured stats), Tauri commands + events.
- `src/` — React + TypeScript + Vite + Tailwind (dark dashboard). Pages: **Dashboard** (GPU/WSL health, VRAM gauge), **Search** (HF results with estimate columns), **Library** (pulled models), **Servers** (create/manage multiple servers, live log tail, stats, minimal chat playground per instruct server), **Settings** (WSL distro, dirs, HF token, default quantization, optional `.wslconfig` tweaks).

### Process model (multi-model)
- One vLLM process per server instance, each spawned as a child of `wsl.exe -d <distro> -- bash -lc '<script>'`; launcher script starts vLLM in the foreground of that console (`exec` not used because we also write a PID file), writes a PID file.
- Default port auto-allocation from 8000; stop = `kill -TERM <pidfile pid>` (fallback: kill the wsl.exe process tree); status from `/health`, `/v1/models`, `/metrics` polling.
- Per-server options: model id, task (instruct / embed), port, `--gpu-memory-utilization` (default 0.92), `--max-model-len` (default = model's context, capped by VRAM fit), quantization (fp16 default / fp8 / AWQ / GPTQ), `--served-model-name`. Embedding servers default `--task embed` (tiny KV, trivially co-runs with an instruct server).
- VRAM guidance in UI: sum of estimated weight footprint of running servers vs. current free VRAM (from `nvidia-smi --query-gpu=...`).

### Bare-bones WSL2 provisioning (app-driven, idempotent)
1. Detect default distro (assume Ubuntu; else guide `wsl --install -d Ubuntu`).
2. `apt update`; install `python3-venv python3-pip curl` only.
3. `uv venv ~/llm-lp/.venv` (prefer `--python 3.12` via `uv python install`; fallback `python -m venv`).
4. `uv pip install vllm` (CUDA wheels; verify `nvidia-smi` works inside WSL and driver/CUDA compatibility).
5. Verify: `vllm --version`, `python -c "import torch, torch.cuda..."`, report GPU name/VRAM/BF16 support (sm80+).
6. Offer optional `.wslconfig` tuning (documented; never silently overwrite).

All steps are idempotent: re-running skips completed phases. **No systemd, no conda, no extra packages** — servers are foreground children of the app-managed `wsl` process; logs stream to the panel via Tauri events.

### Model search → stats pipeline
`HF search API` → enrich each result by fetching `config.json` (+ `safetensors.index.json` for param count) → estimate:
- **Context window**: real value from `config.json` (`max_position_embeddings` / `model_max_length` / `n_positions` / `max_seq_len` / nested `text_config`); family fallbacks when absent; source shown.
- **Context fit (VRAM limit)**: `kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × 2` (KV cache stays fp16 in vLLM; head_dim = hidden/n_heads or explicit) ⇒ max context ≈ usable KV VRAM ÷ kv_bytes_per_token; reported context = min(config value, VRAM limit) at chosen quantization (weight bytes = params × bytes/param; usable KV VRAM = VRAM×gpu_util − weights − ~2.5GB runtime overhead).
- **Max tok/s (decode)**: `bandwidth × utilization ÷ bytes_per_token`, where bytes/param: fp16=2, fp8=1, GPTQ/AWQ≈1.1 (int4 weights + overhead); bandwidth from GPU-model map with a conservative default (~700 GB/s) for unknowns; utilization ≈ 0.5 for single-GPU decode.
- All numbers clearly labeled **estimate** with the quantization assumption shown; after a real run, `measured` tok/s from `/metrics` replaces the estimate (badge).

### Pulling models
"Pull" = background `HF_TOKEN=<token> hf download <model-id>` in the shared venv with progress lines streamed to UI; models cached in `~/.cache/huggingface` inside WSL (vLLM also auto-downloads at first start, so pull is optional).

## Git & documentation
- `git init` at repo root; `.gitignore` (node_modules, target, dist, logs); milestone commits with conventional messages on `main` (local-only unless a remote is added later).
- `docs/PLAN.md` (this plan), `docs/ARCHITECTURE.md`, `README.md` (setup + usage). Planning doc updated whenever the plan changes, committed with the change.

## Important behavior/API changes
- New public surface = the Tauri command set: `env_status`, `provision`, `search_models`, `model_stats`, `pull_model`, `pull_status`, `servers.list/create/delete/start/stop/restart/logs/metrics/chat`, `settings.get/set`, `measured_stats`. Events: `wsl-log`, `server-status`, `pull-progress`. No other existing codebase — greenfield.
- Config persisted at `%APPDATA%/local-llm-panel/config.json` (distro, paths, HF token, server definitions, measured stats).

## Test cases & verification
- **Unit (Rust)**: estimate math (context parsing incl. fallbacks, kv-bytes-per-token, tok/s with known bandwidth/params), port allocator, config persistence round-trip.
- **Integration (environment-gated, run in WSL)**: provisioning is idempotent; start `Qwen2.5-0.5B-Instruct` (instruct) + a small embedding model concurrently; `/health` 200 on both ports; `/v1/models` lists both; chat completion returns; `/metrics` shows request counters; stop terminates both processes (pid gone).
- **Manual walkthrough**: fresh-machine provision → search → pull → run 2 servers → logs stream → VRAM gauge updates → measured stats appear after a run.
- **Acceptance**: all of the above green against the real environment before the MVP is considered done.

## Assumptions & defaults chosen
- MSVC build tools verified at implementation start (VS 2022 present; cargo resolves via vswhere); fallback target `x86_64-pc-windows-gnu` recorded in README if linking fails.
- WebView2 present (Windows 11 default). English UI, dark theme. HF API reachable (system proxy respected).
- Default distro Ubuntu; no systemd dependency. vLLM latest stable pinned to uv-managed CPython 3.12 (exact version recorded in docs at first provision).
- Estimates are explicitly estimates; measured data always wins. Minimal chat playground included in MVP. No model upload/rehost, no scheduling, single Windows host.

## Milestones (commit points)
1. `docs: scaffold plan/architecture/readme` — repo init, docs, .gitignore
2. `feat(rust): wsl, state, estimate, hf modules + unit tests` — core logic, no UI
3. `feat(rust): provision + server lifecycle` — provisioning + multi-server lifecycle with events
4. `feat(ui): dashboard/search/library/servers/settings` — full React UI wired to commands
5. `test: environment-gated integration in WSL` — real vLLM install + 2 concurrent servers + health/models/chat/metrics + stop
6. `chore: bump versions, README polish` — final pass