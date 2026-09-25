# Architecture — Local LLM Panel

## Overview

```
┌──────────────────────────── Windows ────────────────────────────┐
│  Tauri 2 app (Rust core) ── React/TS UI (WebView2)              │
│    ├─ wsl.rs           distro detect + WSL/native child runners │
│    ├─ provision.rs     idempotent WSL2 provisioning             │
│    ├─ server.rs        multi-instance lifecycle, logs, metrics  │
│    ├─ gateway.rs       OpenAI-compatible reverse proxy (11434)  │
│    ├─ hf.rs            HF search / quant discovery / pull       │
│    ├─ estimate.rs      tok/s + context-fit heuristics           │
│    ├─ fit.rs           hardware fit scoring + variant ranking   │
│    ├─ state.rs         persisted config + measured stats        │
│    ├─ llamacpp_install.rs native binary mgmt + CUDA probing     │
│    ├─ security.rs      DPAPI encryption for HF token            │
│    └─ llmfit_adapter.rs llmfit-core ONNX integration            │
│         │  spawns wsl.exe -d <distro> -- bash -lc '<script>'   │
└─────────┼───────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────── WSL2 Ubuntu ────────────────────────┐
│  ~/llm-lp/.venv (uv, CPython 3.12)  vllm (CUDA wheels)          │
│  Server A: python -m vllm.entrypoints.openai.api_server ...     │
│  Server B: python -m vllm.entrypoints.openai.api_server --task  │
│             embed ...                                            │
│  ~/llm-lp/run/<id>.pid   ~/llm-lp/logs/<id>.log                 │
│  HF cache: ~/.cache/huggingface                                 │
└─────────────────────────────────────────────────────────────────┘
```

llama.cpp uses a separate native Windows process boundary:

```
Tauri Rust core ── llama-server.exe (NativeChild)
       │             ├─ --host 127.0.0.1
       │             ├─ GGUF weights from gguf_dir
       │             └─ Windows CUDA runtime / nvidia-smi
       └─ %APPDATA%\local-llm-panel\logs\<id>.log
```

Gateway (OpenAI-compatible reverse proxy):

```
External Client (Cursor, Continue, etc.)
         │  http://127.0.0.1:11434/v1/chat/completions
         ▼
┌─────────────────────────────────────────────────────────────┐
│  gateway.rs — single port, routes by model name             │
│    • GET  /v1/models          → lists running instruct servers │
│                                  + effective context metadata   │
│    • GET  /v1/models/{id}     → model detail + same metadata    │
│    • POST /v1/chat/completions→ find_server_port_for(model)   │
│    • Streams SSE frames back to client                       │
└─────────────────────────────────────────────────────────────┘
         │  forwards to 127.0.0.1:<server_port>
         ▼
   vLLM / llama.cpp server
```

WSL/vLLM provisioning and native llama.cpp installation are independent. The app never
changes `.wslconfig`; llama.cpp remains usable when WSL has not been provisioned.

## Rust core modules

### `gateway.rs`
- **Unified OpenAI-compatible reverse proxy** listening on port 11434 (default, Ollama-compatible).
- `find_server_port_for(servers, model, task)` — routes by exact or fuzzy match on served model name or HF id against running `instruct` servers.
- `model_rows(servers)` — emits OpenAI `/v1/models` rows with the served id, backend metadata, and effective context fields (`max_model_len`, `context_length`, `max_context_length`, `context_window`, `n_ctx`). The configured launch value is preferred; implicit vLLM/llama.cpp limits are read from the running backend.
- Hand-rolled HTTP/1.1 framing (Tokio `TcpListener` + `AsyncReadExt`/`AsyncWriteExt`) — no external HTTP deps.
- Streams SSE `data:` frames bidirectionally; copies headers, handles `Content-Length` and chunked encoding.
- `spawn_supervisor(app_handle)` — background task binds port, accepts connections, routes per-request.
- `gateway_status` command returns `{running: bool, port: u16, error?: string}`.

### `wsl.rs`
- `detect_default_distro() -> Option<String>` — parse `wsl -l -q` (first line, strip BOM/zeros), default to `Ubuntu`.
- `run_script(distro, script) -> RunOutput` — sync run of `wsl.exe -d <distro> -- bash -lc <script>`, captures stdout/stderr+exit code.
- `spawn_script(distro, script) -> Child` — async/fire-and-forget variant with streaming stdout/stderr chunks emitted as `wsl-log` events (used by provisioning and by server launchers; server logs go to `server-log` events per server id).

### `provision.rs`
Phases, each idempotent:
1. `phase_distro` — ensure default distro is Ubuntu-ish; else return guidance error.
2. `phase_apt` — `apt-get update` + `apt-get install -y python3-venv python3-pip curl`.
3. `phase_venv` — prefer `uv python install 3.12` + `uv venv --python 3.12 ~/llm-lp/.venv`; fallback `python3 -m venv ~/llm-lp/.venv`.
4. `phase_vllm` — `uv pip install --python ~/llm-lp/.venv/bin/python vllm huggingface_hub` (CUDA wheels from PyPI).
5. `phase_verify` — `vllm --version` + torch CUDA probe (gpu name, VRAM, bf16 support, driver/CUDA compat check via `nvidia-smi`).
Each phase emits `wsl-log` lines; a phase that already succeeded is skipped (marker file `~/llm-lp/.provisioned` with phase list + recorded vLLM version).

### `estimate.rs` (pure, unit-tested)
- `parse_context(config_json) -> (usize, ContextSource)` — walk `max_position_embeddings`, `model_max_length`, `n_positions`, `max_seq_len`, `*_sequence_length`; recurse into `text_config` / `config`; family fallback table (qwen2=32768, llama=4096, mistral=32768, gemma=8192, etc.) when absent.
- `kv_bytes_per_token(n_layers, n_kv_heads, head_dim) = 2 × n × n_kv × head_dim × 2` (fp16 KV cache).
- `weight_bytes(params, quant) = params × bytes_per_param` (fp16=2, fp8=1, awq/gptq=1.1).
- `context_fit(vram_mb, gpu_util, wb, kvb, overhead_mb) -> usize` — usable KV VRAM = vram×util − wb − overhead(2500 MB); clamp ≥512.
- `tokens_per_sec(bandwidth_gbs, params, quant) = bw × 0.5 / weight-bytes-per-token`.
- `gpu_bandwidth(name) -> (GB/s, known: bool)` — table: RTX 5090=1792, 5080=960, 5070 Ti=896, 5070 Ti Laptop=672, 5070=448, 5060 Ti=448, 5060=288, 4090=1008, 4080=716, 4070 Ti/S=672/504, 4070=504, 4060 Ti=288, 4060=272, 3090=936, 3080=760, 3070=448, 3060=360, 6080/6090 future → default 700.

### `fit.rs` (pure, unit-tested)
- `score_variant(hw, variant, arch, measured) -> FitResult` — per-variant hardware fit scoring composing `estimate.rs` functions.
- `rank_variants(results)` & `compare_variant_fit(a, b)` — sort variants by composite score, native formats preferred over GGUF at equal score.
- `best_variant(results, preferred_format) -> usize` — pick optimal variant respecting user preference (`default_quant`).
- Verdicts: `Comfortable` (≤60% VRAM), `Constrained` (60–95%), `DoesNotFit` (>95%).

### `hf.rs`
- `search(query, limit) -> Vec<HfModel>` — GET `https://huggingface.co/api/models?search=…&limit=N`.
- `enrich(client, model_id, cache) -> Option<EnrichedStats>` — fetches config.json, queries `?expand[]=safetensors` API with index file fallback, cached in `AppState.enrichment_cache` (1h TTL).
- `discover_quant_variants(client, base_model_id, semaphore) -> Vec<QuantVariant>` — discovers AWQ/GPTQ/FP8/BNB repos by naming convention and known publishers; discovers GGUF variants by parsing repo siblings.
- `parse_gguf_quant_label(filename) -> Option<String>` — extracts quant label from GGUF filenames.
- `pull_model(id, token)` — background thread runs `HF_TOKEN=… hf download <id>` in venv, lines parsed (`Fetching`, `Downloading`, % progress with filenames) → `pull-progress` events; `pull_status()` returns current in-flight state.
- `list_gguf_repo_files` and `group_gguf_files` — discover GGUF files, combine split shards, and separate optional `mmproj` companions.
- `download_gguf` — native Windows, bearer-authenticated, resumable shard download into `gguf_dir`, emitting `pull-progress`.

### `llamacpp_install.rs`
- Lists and caches recent `ggml-org/llama.cpp` releases, parses anchored Windows x64 CUDA/cudart names, selects a driver-compatible numeric CUDA version (with Blackwell warnings), falls back across incomplete releases, extracts both archives into a tag/version directory, and probes `--version`/`--help`.
- Stores the installed tag, version, executable override, and help text in the persisted config. Windows `nvidia-smi` is used as a native GPU fallback.

### `security.rs`
- **DPAPI encryption** for HF token at rest: `protect_data()` / `unprotect_data()` wrap `CryptProtectData`/`CryptUnprotectData` (Windows Credential Guard).
- Token stored encrypted in `config.json`; plaintext never written to disk.

### `llmfit_adapter.rs`
- Bridge to `llmfit-core` crate (ONNX-based model fit estimation).
- Provides async `estimate_fit()` using embedded ONNX models for more accurate VRAM/tok-s predictions.

### `server.rs`
- `alloc_port()` — starting 8000 (+ existing server defs excluded), bind `127.0.0.1:port` to prove free.
- `start(def)` — compose launch script; spawn via `wsl.rs::spawn_script`; write `~/llm-lp/run/<id>.pid` with the WSL-side PID (script prints `$$` after `exec`-less start); poll `/health` until 200 or timeout (300s); emit `server-status`.
- `stop(id)` — read pidfile → `wsl -d <distro> -- kill -TERM <pid>`; poll health down (30s); fallback terminate child process tree (`taskkill /T /F /PID <wslpid>`); emit status.
- `logs(id)` — streaming: child stdout/stderr chunks accumulated in an in-memory ring buffer (per server) AND appended to `~/llm-lp/logs/<id>.log` in WSL (via `tee` in launcher) for restart-persistence.
- `metrics(id)` — GET `:port/metrics`, parse Prometheus counters `vllm:generation_tokens_total`, `vllm:prompt_tokens_total`, `vllm:num_requests_running`; delta between polls → measured tok/s (prompt+generation split), persisted to `state.rs`.
- `chat(id, messages)` — POST `:port/v1/chat/completions` (instruct servers only) from Rust (avoids webview CORS).
- `build_llamacpp_args` — pure mapping of persisted llama.cpp settings to `llama-server` flags; the runtime filters flags against the installed `--help` output.
- Native launch fitting is capability-gated: `--fit on/off` and `--fit-target` are emitted only when supported, while `n_cpu_moe` remains the manual fallback. Structured settings take precedence over arbitrary extra flags, and all `--override-kv` entries (including the AgentWorld metadata workaround) are merged into one last-wins list.
- `llamacpp_install.rs` selects live Windows CUDA release assets using `nvidia-smi`, installs the matching runtime archive, and probes `--list-devices`; Vulkan/CPU archives are not fallback builds. GitHub diagnostics include examined tags and rejected Windows assets. Native server requests use per-server API keys when configured.
- Native servers use `NativeChild`, localhost health polling, persisted Windows logs, Prometheus metrics, and `servers_test_tool_call` for a small OpenAI tool-call smoke test.
- **Environment variables for vLLM servers**: Per-server `env: BTreeMap<String, String>` in `ServerDef` merged with global `default_env` in `PersistedConfig` (per-server wins). Launch script is written to a file in WSL (`~/.local/share/local-llm-panel/launch-<id>.sh`) via stdin pipe to avoid nested quoting issues with `bash -lc`, then executed with `bash -l`. The script emits `export NAME='value'` lines with validation (`^[A-Za-z_][A-Za-z0-9_]*$`) and safe single-quote escaping before `exec python -m vllm...`. Default global `VLLM_USE_FLASHINFER_SAMPLER=0` avoids FlashInfer JIT compilation requirement for nvcc/CUDA toolkit in WSL. At startup the script logs each applied env var as `[LocalLLmPanel] env NAME=<VALUE>` for debugging quote issues.
- **Startup failure detection & one-click fix**: `startup_failure_hint()` scans logs for patterns (`Could not find nvcc`, `cuda_home`, `Failed to find C compiler`, `Python.h`, `ninja not found`, `invalid literal for int() with base 10` with `os.environ["VLLM_` → malformed env var value) and returns actionable hints. When the nvcc pattern is detected, the UI shows a "Restart with FlashInfer sampler disabled" button that sets `VLLM_USE_FLASHINFER_SAMPLER=0` and restarts. The `invalid literal for int()` pattern specifically indicates an environment variable has a malformed value (stray quotes or whitespace); the UI points the user at the Environment Variables editor.
- **CUDA build tools provisioning (optional)**: `phase_cuda_build_tools` installs `gcc`, `python3.12-dev`, `ninja-build`, and the NVIDIA CUDA toolkit (matching PyTorch CUDA version) from NVIDIA's WSL-Ubuntu repo. Runs as root via `wsl --user root`. Streamed via `wsl-log` events. Idempotent and user-initiated only.

### `state.rs`
- `AppState { config: Mutex<PersistedConfig>, servers: Mutex<BTreeMap<Id, LiveServer>>, http: Client, pulling: Arc<Mutex<HashMap<model_id, bool>>>, gpu: Mutex<Option<GpuSnapshot>>, enrichment_cache: Mutex<HashMap<model_id, CachedEnrichment>>, rec_cache: Mutex<Option<(Vec<ModelWithFit>, Instant)>>, conversations: Mutex<HashMap<String, Conversation>>, benchmarks: Mutex<Vec<BenchmarkRun>> }`.
- `PersistedConfig { distro, llm_dir, venv_dir, hf_token_encrypted, default_quant, servers: Vec<ServerDef>, measured: HashMap<model_id, MeasuredStats>, minimize_to_tray, launch_at_login, resume_servers_on_launch, gateway_enabled, gateway_port, lan_access_enabled, system_metrics_retention_secs }` — JSON at `%APPDATA%/local-llm-panel/config.json`; atomic save (write temp + rename). `hf_token_encrypted` is DPAPI blob.
- `LiveServer { def, child: ChildGuard (Windows process handle), wsl_pid: Option<u32>, status, log_ring: VecDeque<String>, health_since, last_metrics, was_running: bool }`.
- `Conversation { id, server_id, title, created, updated, messages: Vec<Message> }` — persisted to `conversations.json` (atomic).
- `BenchmarkRun { id, server_id, model_id, timestamp, prompts: Vec<BenchmarkPromptResult> }` — persisted in config.

## Tauri commands (public surface)

**Core/WSL:** `env_status`, `provision`, `wsl_distros`, `wslconfig_get`, `system_metrics_series`, `get_system_memory`, `get_memory_settings`, `update_memory_settings`.

**HF/Models:** `search_models`, `search_models_with_fit`, `recommended_models`, `model_stats`, `pull_model`, `pull_status`, `pull_cancel`, `gguf_files`, `download_gguf`, `library_list`, `library_remove`, `library_disk_usage`, `library_import_local`.

**llama.cpp:** `install_llamacpp`, `llamacpp_status`, `github_access`, `clear_github_token`.

**Servers:** `servers_list`, `servers_create`, `servers_delete`, `servers_start`, `servers_stop`, `servers_restart`, `servers_logs`, `servers_metrics`, `server_metrics_series`.

**Chat/Conversations:** `servers_chat`, `servers_chat_stream`, `servers_chat_cancel`, `servers_test_tool_call`, `conversations_list`, `conversations_save`, `conversations_delete`.

**Benchmarks:** `benchmarks_run`, `benchmarks_cancel`, `benchmarks_history`.

**Gateway:** `gateway_status`.

**Settings/Config:** `settings_get`, `settings_set`, `measured_stats`, `autostart_get`, `autostart_set`, `config_export`, `config_import`, `server_recipe_export`, `server_recipe_parse`, `open_url`.

Events: `wsl-log {phase,line}`, `llamacpp-install-progress {file,done,total?}`, `server-status {id,status,error?}`, `server-log {id,line}`, `pull-progress {model,state,file?,percent?}`, `chat-token {request_id,server_id,token}`, `chat-done {request_id}`, `chat-cancel {request_id}`, `gpu-status`, `gateway-status`.

## Frontend (`src/`)
- Vite + React 18 + TS; Tailwind v4 (`@tailwindcss/vite`); dark theme (zinc/indigo).
- `App.tsx` — shell: sidebar nav (Dashboard/Search/Library/Servers/Settings) + `<Routes>`.
- `api.ts` — typed wrappers over `@tauri-apps/api/core.invoke` + event subscriptions.
- Pages: `Dashboard.tsx` (env status card, GPU/VRAM gauge from nvidia-smi poll, VRAM/tok-s sparklines, gateway status toggle), `Search.tsx` (dynamic recommendations, debounced auto-search, quant-aware fit verdicts [Comfortable/Constrained/DoesNotFit], est tok/s per quant, GGUF experimental flagging, pull button), `Library.tsx` (pulled = HF-cache scan via `env_status` cache path + `hf cache list` source; per-model disk size, delete with in-use guard), `Servers.tsx` (create/edit/delete rows; per-row status dot, port, log tail pane, metrics, chat drawer with streaming + conversation persistence, benchmark runner, tool-call test), `Settings.tsx` (distro, dirs, HF token, default quant, `.wslconfig` read-only + copy, gateway enable/port, LAN access, autostart/minimize-to-tray/resume, memory limits, config export/import, server recipe share).

## Data flow
- Poll loops in Rust (tokio tasks, 5s): `nvidia-smi` snapshot → `gpu-status` event; per-server `/health`+`/metrics` → `server-status`/`server-metrics` events; UI keeps local copies. No DB — config JSON only.
- Measured stats: on each metrics poll compute deltas; persist into `config.measured` so they survive restart; `measured_stats` command returns latest.
- Gateway supervisor task accepts connections on 11434, routes each request to the correct server port based on model name, streams SSE bidirectionally.
- Conversations & benchmarks persisted atomically alongside config.

## Security notes
- HF token encrypted at rest via Windows DPAPI (`CryptProtectData`); never plaintext on disk.
- Ports bound on WSL 127.0.0.1 via `--host 127.0.0.1` (vLLM default 0.0.0.0 — we pass `--host 127.0.0.1` so servers aren't exposed on LAN).
- Gateway listens on 127.0.0.1:11434 only; LAN access requires explicit opt-in (`lan_access_enabled` in config) with consent warning in UI.
- Chat endpoint called from Rust (no webview CORS exposure).
- Config export omits encrypted HF token by default; import validates schema.