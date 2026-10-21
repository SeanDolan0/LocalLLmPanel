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
│    ├─ estimate.rs      tok/s + tiered context-fit heuristics    │
│    ├─ fit.rs           hardware fit scoring + variant ranking   │
│    ├─ context_fit.rs   pre-start context/VRAM/RAM analysis      │
│    ├─ gguf.rs          GGUF metadata reader (ctx/KV geometry)   │
│    ├─ state.rs         persisted config + measured stats        │
│    ├─ llamacpp_install.rs native binary mgmt + CUDA probing     │
│    ├─ security.rs      DPAPI encryption for tokens              │
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
│    • GET  /v1/models          → running instruct + embedding│
│                                  servers, with context/     │
│                                  capability metadata        │
│    • GET  /v1/models/{id}     → model detail + same metadata│
│    • POST /v1/chat/completions→ find_server_port_for(model) │
│    • POST /v1/completions     → instruct servers            │
│    • POST /v1/embeddings      → embedding servers           │
│    • GET  /health             → gateway liveness            │
│    • Bearer auth required on /v1/*                          │
│    • Streams SSE frames back to client                      │
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
- **Routes:** `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, and `GET /health`. Chat and completions route to running instruct servers; embeddings route to running embedding servers. `OPTIONS` is handled with an origin allowlist for local browser clients.
- **Authentication:** every `/v1/*` route requires `Authorization: Bearer <key>` matching the panel's global API key. A gateway with no key configured returns `503` instead of exposing models; `/health` is unauthenticated. Comparison is constant-time.
- `find_server_port_for(...)` — routes by exact or fuzzy match on served model name or HF id against running servers of the requested task.
- `model_rows(servers)` — emits OpenAI `/v1/models` rows for every running server (instruct first, then embeddings). Each row includes the served id plus `name`, `root`, `publisher`, `type`, `state`, `quantization`, `compatibility_type`, and the effective context under `max_model_len` / `context_length` / `max_context_length` / `context_window` / `n_ctx`. The configured launch value is preferred; implicit vLLM/llama.cpp limits are read from the running backend. Rows also report the output limit as `max_output_tokens` / `max_completion_tokens` / `max_tokens`, the modalities under `architecture.input_modalities` and the top-level `input` array (`["text","image"]` for vision), and capability aliases (`supports_tools`, `supports_vision`, `supports_reasoning`, `supports_embeddings`). The `max_tokens` and `input` aliases exist because Pi's `pi-openai-compat` catalog parser reads only `context_window`, `max_tokens`, boolean `reasoning`, and `input`; without them Pi falls back to a 4096-token output cap and text-only input. Pi caches that catalog, so metadata changes require `/compat-refresh` (or `/compat-logout` + `/compat-login`) on the Pi side.
- **Model naming** — the advertised `id` is the served model name when set. For llama.cpp the id otherwise falls back to the GGUF file stem (last path component, without `.gguf` or a `-00001-of-00002` split suffix), so clients never see a full Windows/UNC path. The full path stays in `root`/`model_id` so existing clients can still route by it. `effectiveModelName()` in `src/types.ts` mirrors this for the UI.
- **Thinking metadata** — each row carries a boolean `reasoning` (for OpenAI-compatible harnesses such as Pi's `pi-openai-compat`, which only recognizes a boolean) plus a detailed `reasoning_config` object (aliased as `thinking`) with `supported`, `output`, `enabled_by_default`, `default_enabled`, `mandatory`, `mode` (`effort` / `always_on`), standard `levels` (`["none","low","medium","high"]` or `["low","medium","high"]` for always-on models), `supported_efforts` (`["none","minimal","low","medium","high","xhigh","max"]`), `default_level` / `default_effort` (`"medium"`), `thinkingLevelMap`, the request `control` (`"reasoning_effort"`), the full `controls` list, and `request_precedence: "harness"`. `supported` means the model family can reason; `output` means the running backend will actually return `reasoning_content`. vLLM needs `--reasoning-parser` for the latter and `--enable-auto-tool-choice` + `--tool-call-parser` for tools; llama.cpp needs `--jinja` for tools. `configuration_hints` lists the exact missing flags when a capability is disabled by launch configuration.
- **Harness precedence** — panel launch flags (and environment variables) only provide defaults. Before forwarding `/v1/chat/completions`, the gateway canonicalizes every harness thinking shape (`reasoning_effort`, `reasoning.effort`, `thinking.type`/`budget_tokens`, Ollama `think`, `chat_template_kwargs.enable_thinking`/`thinking`, and the various budget spellings) into explicit request-level fields. Effort labels are normalized generically across all model types: for vLLM, extended labels collapse to `low`/`high` while preserving standard `low`/`medium`/`high`; for llama.cpp, labels are rounded onto the vocabulary read from the running template via `/props` (`chat_template`), so a Qwen3.8-style template that only accepts `xhigh`/`medium`/`low` receives `xhigh` for a harness `high` instead of raising a Jinja 500. Both backends rank explicit `chat_template_kwargs` above their launch defaults, llama.cpp ranks request `reasoning_format` above `--reasoning-format`, and request `reasoning_budget_tokens` above `--reasoning-budget`; when thinking is enabled without a harness budget, the gateway supplies the context window so a panel `--reasoning-budget 0` cannot suppress it. Requests that touch neither thinking nor token-limit aliases are forwarded byte-for-byte, so no gateway defaults are injected. `max_completion_tokens` / `max_output_tokens` are mapped to `max_tokens` so output limits also survive the proxy.
- Hand-rolled HTTP/1.1 framing (Tokio `TcpListener` + `AsyncReadExt`/`AsyncWriteExt`) — no external HTTP deps.
- Streams SSE `data:` frames bidirectionally; copies headers, handles `Content-Length` and chunked encoding.
- **Long-running chat requests** use a separate `AppState::chat_http` client with only a 30 s connect timeout. The shared `http` client carries a total 120 s deadline that reqwest applies until the response body finishes, so reusing it for chat would cut streams during long prompt prefill or generation and harnesses would report `stream ended without finish reason`. Upstream stream errors are logged instead of being silently converted into a clean EOF, and a trailing partial SSE line is newline-terminated before forwarding.
- `spawn_supervisor(app_handle)` — background task binds port, accepts connections, routes per-request.
- `gateway_status` command returns `{enabled, port, running, auth_configured}`; `running` is a live TCP probe and is false until both enabled and an API key exist.

### `wsl.rs`
- `detect_default_distro() -> Option<String>` — parse `wsl -l -q` (first line, strip BOM/zeros), default to `Ubuntu`. `ensure_app_distro` / `installed_distros` / `is_running` / `is_apt_distro` cover the surrounding lifecycle.
- `run_script(distro, script) -> RunOutput` — sync run of `wsl.exe -d <distro> -- bash -lc <script>`, captures stdout/stderr + exit code. `run_script_root` adds `--user root`.
- `run_script_stream(distro, script, on_line)` — streaming variant used by provisioning and CUDA build-tool installs; lines are parsed from both `\n` and `\r` so progress bars arrive per frame.
- `spawn_script_via_stdin(...)` — writes the launch script to a WSL file and runs it with `bash -l`, avoiding nested quoting. Returns a `WslChild` guard with PID/wait/kill.
- `NativeChild` — Windows process guard for `llama-server.exe` (spawn with args, wait, kill, join).
- `detect_wsl_memory` / `try_detect_wsl_memory` / `parse_meminfo` — WSL2 `MemTotal`/`MemAvailable` for RAM-overflow planning.
- `terminate_distro` + `active_wsl_execs` / `secs_since_last_exec` — idle-reaper support and clean WSL shutdown on app exit.
- `windows_to_wsl_path`, `shell_quote_wsl` — path and shell-quoting helpers.

### `provision.rs`
Phases, each idempotent:
1. `phase_distro` — validate that the configured distro is apt-based (Ubuntu/Debian/Kali/Mint) and responsive; otherwise return install guidance.
2. `phase_sudo` — ensure passwordless sudo; if missing, writes a `/etc/sudoers.d/llm-panel-<user>` rule via `wsl --user root` and verifies it.
3. `phase_apt` — `apt-get update` + `apt-get install -y python3-venv python3-pip curl`.
4. `phase_uv` — install `uv` if missing.
5. `phase_venv` — prefer `uv python install 3.12` + `uv venv --python 3.12 ~/llm-lp/.venv`; fall back to `python3 -m venv`.
6. `phase_vllm` — `uv pip install --python ~/llm-lp/.venv/bin/python vllm huggingface_hub` (CUDA wheels from PyPI).
7. `phase_verify` — `vllm --version` + torch CUDA probe (GPU name, VRAM, bf16 support, driver/CUDA compatibility via `nvidia-smi`).
Each phase emits `wsl-log` lines; a phase that already succeeded is skipped (marker file `~/llm-lp/.provisioned` with phase list + recorded vLLM version).

**Optional CUDA build tools:** `phase_cuda_build_tools` installs `gcc`, `python3.12-dev`, `ninja-build`, and the NVIDIA CUDA toolkit (matching the PyTorch CUDA version) from NVIDIA's WSL-Ubuntu repo. Runs as root, streamed via `wsl-log`, idempotent, and user-initiated only.

### `estimate.rs` (pure, unit-tested)
- `parse_context(config_json) -> (usize, ContextSource)` — walk `max_position_embeddings`, `model_max_length`, `n_positions`, `max_seq_len`, `*_sequence_length`; recurse into `text_config` / `config`; family fallback table (qwen2=32768, llama=4096, mistral=32768, gemma=8192, etc.) when absent.
- `kv_bytes_per_token(n_layers, n_kv_heads, head_dim) = 2 × n × n_kv × head_dim × 2` (fp16 KV cache); `kv_bytes_per_token_hybrid` and `estimate_kv_bytes_per_token` cover models whose config omits KV geometry.
- `weight_gb(params, quant)` / `bytes_per_param(quant)` — fp16/bf16=2.0, fp8/int8=1.0, Q8=1.05, Q6=0.80, Q5=0.68, Q4/AWQ/GPTQ/GGUF≈0.55, Q3=0.45, Q2=0.35, ternary (PQ/PTQ)≈0.28 bytes per parameter.
- `context_fit_tiered(...) -> TieredContextFit` — computes three budgets: `vram_context`, `extended_context` using RAM swap, and the required `swap_space_gb` / `cpu_offload_gb`. Pure-GPU `context_fit` / `context_fit_with_weight` remain for simple cases; usable KV VRAM is `vram × util − weights − overhead`, clamped to ≥512 tokens.
- `tokens_per_sec` / `tokens_per_sec_tiered` — bandwidth-bound decode estimate; the tiered variant applies PCIe/DDR degradation when weights or KV spill into RAM.
- `gpu_bandwidth(name) -> (GB/s, known: bool)` — lookup table covering RTX 50/40/30-series desktop and laptop parts (e.g. 5090=1792, 4090=1008, 3090=936, 3060=360), with ~700 GB/s fallback and `known=false` for unrecognized NVIDIA/GeForce/RTX/Quadro/Tesla names.
- `parse_params_from_config` / `parse_params_from_name` / `parse_params_from_safetensors_api` — parameter-count discovery for fit scoring.

### `fit.rs` (pure, unit-tested)
- `HardwareProfile` — GPU name, VRAM, bandwidth, plus `ram_total_mb`, `ram_usable_mb`, `ram_potential_mb`, and `ram_bandwidth_gbs` for overflow planning.
- `score_variant(hw, variant, arch, measured) -> FitResult` — per-variant hardware fit scoring composing `estimate.rs`.
- `RunMode` — `Gpu`, `GpuRamSwap`, `CpuOffload`, or `DoesNotFit`; `FitResult` carries both `vram_context` and `extended_context`, plus recommended `swap_space_gb` / `cpu_offload_gb`.
- `rank_variants(results)` & `compare_variant_fit(a, b)` — sort variants by composite score, native formats preferred over GGUF at equal score.
- `best_variant(results, preferred_format) -> usize` — pick optimal variant respecting user preference (`default_quant`).
- Verdicts: `Comfortable` (weights + overhead ≤60% of VRAM, fully on GPU), `Constrained` (≤95% or RAM swap/offload needed), `DoesNotFit` (>95%), via `COMFORTABLE_MAX_RATIO` / `CONSTRAINED_MAX_RATIO`.

### `context_fit.rs` (unit-tested)
- `analyze_vllm_context` / `analyze_llama_context` — pre-start analysis of a requested context against model weight, KV-cache geometry, current free VRAM, available WSL RAM, GPU utilization, and offload settings.
- `ContextFitReport` — status (`Fits`, `FitsWithCpu`, `Tight`, `DoesNotFit`, `Unknown`), weight source, KV bytes/token, projected VRAM and headroom, max context with/without RAM overflow, recommended context and KV-cache dtype, required CPU offload, plus warnings.
- Backs the editable **Context fit test** in the server editor; recommendations can be applied to the form without starting the server.

### `gguf.rs`
- Minimal GGUF v2/v3 metadata reader (bounds-checked, no external parser dependency).
- Extracts `context_length`, `block_count`, `head_count`, `head_count_kv`, and key/value head dimensions so llama.cpp servers get the same KV/fit analysis as vLLM models.

### `hf.rs`
- `search(query, limit) -> Vec<HfModel>` — GET `https://huggingface.co/api/models?search=…&limit=N`.
- `enrich(client, model_id, cache) -> Option<EnrichedStats>` — fetches config.json, queries `?expand[]=safetensors` API with index file fallback, cached in `AppState.enrichment_cache` (1h TTL).
- `discover_quant_variants(client, base_model_id, semaphore) -> Vec<QuantVariant>` — discovers AWQ/GPTQ/FP8/BNB repos by naming convention and known publishers; discovers GGUF variants by parsing repo siblings.
- `parse_gguf_quant_label(filename)` / `quant_requires_prism_channel(label)` — extract the quant label and flag ternary formats that need the PrismML llama.cpp channel.
- `pull_model(id, token)` — background thread runs `HF_TOKEN=… hf download <id>` in the venv, parses progress lines (including `\r`-delimited tqdm frames) into `pull-progress` events with file, percent, bytes, and ETA. Cancellable via `pull_cancel`; `pull_status` reports in-flight state.
- `list_gguf_repo_files`, `check_gguf_repo`, and grouping helpers — discover GGUF files, combine split shards, and separate optional `mmproj` companions.
- Native GGUF downloads run on Windows (not WSL), are bearer-authenticated, resumable per shard, and write into the configured `gguf_dir`.

### `llamacpp_install.rs`
- **Per-channel installs** — `Upstream` (`ggml-org/llama.cpp`) and `PrismML` (`PrismML-Eng/llama.cpp@prism`, needed for ternary formats). Each channel keeps its own installed tag, version, executable, help text, and directory in config.
- Lists and caches recent releases, parses anchored Windows x64 CUDA/`cudart` names, selects a driver-compatible numeric CUDA version (with Blackwell warnings), falls back across incomplete releases, extracts both archives into a tag directory, and probes `--version` / `--help`.
- `test_github_access` → structured diagnostics (token used, rate limit, rejected token retry); `list_devices` / `parse_devices` validate the CUDA backend. `windows_gpu_snapshot` provides a native `nvidia-smi` fallback.

### `security.rs`
- **DPAPI encryption** for Hugging Face and GitHub tokens at rest: `encrypt_token()` / `decrypt_token()` wrap `CryptProtectData` / `CryptUnprotectData` (Windows Credential Guard). Stored values carry a `dpapi:` prefix.
- Secrets are redacted from webview payloads (`SECRET_PLACEHOLDER`), preserved across settings updates when untouched, and stripped from config exports by default.

### `llmfit_adapter.rs`
- Bridge to the `llmfit-core` crate: `SystemSpecs::detect`, the embedded model database, and llmfit's fit/run-mode enums.
- `recommend_models(limit)` / `search_models_local(query, limit)` power offline recommendations; converters map llmfit verdicts and run modes onto the panel's `FitVerdict` / `RunMode`.

### `server.rs`
- `alloc_port()` — starting 8000 (+ existing server defs excluded), bind `127.0.0.1:port` to prove free.
- `start_server(id)` — compose the launch script; vLLM runs via `wsl.rs::spawn_script_via_stdin` and writes `~/llm-lp/run/<id>.pid`; llama.cpp spawns `NativeChild`. Polls `/health` until 200 or timeout; emits `server-status`. vLLM GPU utilization is clamped against current free VRAM to avoid startup OOM.
- `stop_server(id)` — read the vLLM pidfile → `kill -TERM`, or kill the native child; poll health down; fall back to `taskkill /T /F`; remove the pidfile only when WSL is running; emit status.
- **Crash recovery** — the monitor detects an unexpected exit and, when `auto_restart_crashed` is set, restarts up to 3 times with backoff (`crash_retry_count`).
- **Idle reaper** — `should_terminate_wsl(IdleCheck)` + `spawn_wsl_idle_reaper` shut down the distro after inactivity only when no server, pull, provisioning job, starting server, or recent WSL command is active.
- `server_logs(id, since)` — streaming logs in a per-server 2000-line ring buffer; vLLM also appends `~/llm-lp/logs/<id>.log` in WSL via `tee` for restart persistence, while native logs persist on Windows.
- `server_metrics(id)` — GET `:port/metrics` (with the server API key when configured), strip Prometheus label suffixes, and sum per-label series. Supports vLLM (`vllm:*`) and llama.cpp (`llamacpp:*`): requests running/waiting, prompt/generation totals, KV-cache usage, prefix-cache hit rate, TTFT/E2E/queue/prefill latency histograms, preemptions, and instant throughput gauges. Counter deltas between polls produce decode/prefill tok/s and request rate, persisted to `state.rs` and streamed to the Servers metrics panel.
- `chat` / `chat_stream` — POST `:port/v1/chat/completions` from Rust (avoids webview CORS); SSE deltas become `chat-token` events, with cancellation through `chat_cancels` (`Notify`) and `chat-done` / `chat-cancel` / `chat-error`. `chat_with_tools` backs the tool-call smoke test.
- `run_benchmark` — standardized prompts, per-step `benchmark-step` events, persisted `BenchmarkRun` history.
- `build_llamacpp_args_with_help` — pure mapping of persisted llama.cpp settings to `llama-server` flags; `filter_llamacpp_args` drops flags the installed `--help` doesn't advertise.
- Native launch fitting is capability-gated: `--fit on/off` and `--fit-target` are emitted only when supported, while `n_cpu_moe` remains the manual fallback. Structured settings take precedence over arbitrary extra flags, and all `--override-kv` entries are merged into one last-wins list.
- Native servers use `NativeChild`, localhost health polling, persisted Windows logs, Prometheus metrics, and per-server API keys.
- **Environment variables for vLLM servers**: per-server `env` merged with global `default_env` (per-server wins). The launch script is written to `~/.local/share/local-llm-panel/launch-<id>.sh` via stdin to avoid nested quoting, then executed with `bash -l`. It emits `export NAME='value'` lines with validation (`^[A-Za-z_][A-Za-z0-9_]*$`) and safe quoting before `exec python -m vllm...`. The default global `VLLM_USE_FLASHINFER_SAMPLER=0` avoids the FlashInfer JIT requirement.
- **Startup failure detection & one-click fix**: `startup_failure_hint()` scans logs for patterns (`Could not find nvcc`, `cuda_home`, `Failed to find C compiler`, `Python.h`, `ninja not found`, malformed env values) and returns actionable hints. The nvcc hint offers a "Restart with FlashInfer sampler disabled" action; malformed-value hints point at the environment editor.

### `state.rs`
- `AppState { config: Mutex<PersistedConfig>, servers: Mutex<BTreeMap<Id, LiveServer>>, starting: Mutex<HashSet<Id>>, http: Client, pulling: Arc<Mutex<HashMap<model_id, bool>>>, gpu: Mutex<Option<GpuSnapshot>>, enrichment_cache, quant_cache, rec_cache, conversations: Mutex<Vec<Conversation>>, chat_cancels, benchmarks: Mutex<Vec<BenchmarkRun>>, benchmark_cancels, system_metrics: VecDeque<SystemMetricPoint>, server_metrics: BTreeMap<Id, VecDeque<ServerMetricPoint>>, provisioning: AtomicBool }`.
- `PersistedConfig { distro, llm_dir, venv_dir, llamacpp_dir, gguf_dir, llamacpp_channels, hf_token, github_token, default_quant, servers, measured, memory_settings, advanced_settings, minimize_to_tray, auto_restart_crashed, launch_at_login, auto_shutdown_wsl, imported_local_models, default_env }` — JSON at `%APPDATA%/local-llm-panel/config.json`; atomic save (temp + rename with rollback). `hf_token` / `github_token` are DPAPI blobs; legacy llama.cpp fields are retained for back-compat.
- `MemorySettings` — GPU utilization, VRAM overhead, RAM-overflow toggle, manual RAM limit, safety reserve, weight-offload allowance, and context cap.
- `AdvancedSettings` — `HF_HOME`, offline mode, host binding, global API key, gateway enable/port, KV-cache dtype, prefix caching, chunked prefill, max sequences, custom all-reduce toggle, log level, extra vLLM args, custom env text.
- `LiveServer { def, status, error, wsl_child, native_child, wsl_pid, log_ring: Mutex<VecDequeLog>, last_metrics, stopping, crash_retry_count }` (log ring caps at 2000 lines).
- `Conversation { id, server_id, title, created_at, updated_at, messages }` — persisted to `conversations.json` (atomic).
- `BenchmarkRun { id, server_id, model_id, quant, timestamp, prompt_tok_s, gen_tok_s, latency_ms, prompt_count }` — persisted in config.
- `METRICS_SERIES_CAPACITY = 60` — ring-buffer depth for system and per-server sparklines.

## Tauri commands (public surface)

**Core/WSL:** `env_status`, `provision`, `wsl_distros`, `wslconfig_get`, `gpu_status`, `system_metrics_series`, `get_system_memory`, `get_memory_settings`, `update_memory_settings`, `check_flashinfer_ready`, `install_cuda_build_tools`.

**HF/Models:** `search_models`, `search_models_with_fit`, `recommended_models`, `model_stats`, `analyze_context_fit`, `pull_model`, `pull_status`, `pull_cancel`, `gguf_files`, `download_gguf`, `library_list`, `library_remove`, `library_disk_usage`, `library_import_local`.

**llama.cpp:** `install_llamacpp`, `install_llamacpp_channel`, `llamacpp_status`, `github_access`, `clear_github_token`.

**Servers:** `servers_list`, `servers_create`, `servers_update`, `servers_delete`, `servers_start`, `servers_stop`, `servers_restart`, `servers_logs`, `servers_metrics`, `server_metrics_series`.

**Chat/Conversations:** `servers_chat`, `servers_chat_stream`, `servers_chat_cancel`, `servers_test_tool_call`, `conversations_list`, `conversations_save`, `conversations_delete`.

**Benchmarks:** `benchmarks_run`, `benchmarks_cancel`, `benchmarks_history`.

**Gateway:** `gateway_status`.

**Settings/Config:** `settings_get`, `settings_set`, `autostart_get`, `autostart_set`, `config_export`, `config_import`, `server_recipe_export`, `server_recipe_parse`, `open_url`.

Events: `wsl-log {phase,line}`, `llamacpp-install-progress {file,done,total?}`, `pull-progress {model,state,file?,percent?,speed_bps?,eta_seconds?,bytes_downloaded?,bytes_total?}`, `server-status {id,status,error?}`, `server-log {id,line}`, `server-metrics`, `chat-token {request_id,server_id,token}`, `chat-done {request_id}`, `chat-cancel {request_id}`, `chat-error {request_id,error}`, `benchmark-step`, `benchmark-done`, `benchmark-cancel`, `benchmark-error`.

## Frontend (`src/`)
- Vite + React 18 + TS; Tailwind v4 (`@tailwindcss/vite`); dark theme (zinc/indigo).
- `App.tsx` — shell: sidebar nav (Dashboard/Search/Library/Servers/Settings) + `<Routes>` in a `MemoryRouter`.
- `api.ts` — typed wrappers over `@tauri-apps/api/core.invoke` + event subscriptions; `types.ts` mirrors the Rust payloads; `ui.tsx` holds shared UI primitives.
- Pages:
  - `Dashboard.tsx` — env status, provisioning + FlashInfer actions, llama.cpp install/update, GPU/VRAM gauge, system sparklines, server summary, VRAM-pressure warning.
  - `Search.tsx` — debounced search, dynamic recommendations, quant-aware fit verdicts, estimated tok/s per quant, dual VRAM/RAM context badges, GGUF/PrismML flagging, pull controls.
  - `Library.tsx` — HF-cache scan + local GGUF imports, per-model disk usage, delete with in-use guard, active-download progress.
  - `Servers.tsx` — create/edit/delete servers; per-row status/port/log tail; `ServerMetricsPanel`; chat drawer with streaming, cancellation, and conversation persistence; benchmark runner; tool-call test; context fit test; per-backend advanced fields.
  - `Settings.tsx` — Simple/Advanced mode. Covers distro, directories, tokens, default quant, `.wslconfig` read-only + copy, gateway + API key, LAN consent, autostart/minimize-to-tray/auto-restart/auto-shutdown, memory budgets, engine flags, custom env, config export/import, and recipe sharing.
- `components/Sparkline.tsx` and `components/ServerMetricsPanel.tsx` render inline SVG series without charting dependencies.

## Data flow
- UI poll loops call Tauri commands every 4–5s: the Dashboard refreshes `env_status` and `system_metrics_series`; the Servers page refreshes `servers_list`, `server_metrics_series`, and `gpu_status`. Each running server also has a Rust monitor task (`HEALTH_POLL` 2s, `METRICS_POLL` 5s) that emits `server-status` and `server-metrics` events. No DB — config JSON only.
- Measured stats: each metrics poll computes counter deltas and persists them into `config.measured` so they survive restart; `servers_list` and metrics payloads surface the latest values.
- Gateway supervisor task accepts connections on 11434, authenticates `/v1/*`, routes each request to the correct server port based on model name, and streams SSE bidirectionally.
- Conversations persist to `conversations.json`; benchmark runs persist in `config.json`; both use atomic replacement.
- On app exit, every server the app launched is stopped first (SIGTERM → bounded wait → SIGKILL for vLLM, `taskkill /T /F` for llama.cpp), and the persisted `was_running` hint is cleared so the crash monitor cannot restart anything mid-exit. If `auto_shutdown_wsl` is enabled, the distro is then terminated. The idle reaper does the same after a quiet period while the app keeps running.

## Security notes
- Hugging Face and GitHub tokens are encrypted at rest via Windows DPAPI (`CryptProtectData`); plaintext is never written to disk. Secrets are redacted from the webview and from config exports.
- Ports bind to `127.0.0.1` by default; vLLM is launched with `--host 127.0.0.1` so it is not exposed on the LAN. Binding `0.0.0.0` requires the explicit LAN consent flow.
- Gateway listens on `127.0.0.1:11434` and requires a Bearer API key for every `/v1/*` route; without a configured key it returns `503`. `/health` is unauthenticated.
- Per-server API keys are applied to chat/metrics requests; native llama.cpp requests use the server's key when configured.
- Chat is called from Rust, so the webview is not exposed to backend CORS.
- Config import validates schema and never trusts placeholder secrets; config export omits tokens by default.
