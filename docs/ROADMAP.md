# Local LLM Panel — Product Roadmap

The repo is at **v1.4.0** (WSL2 provisioning, HF search + fit estimation, model
pull, multi-instance vLLM/llama.cpp servers with logs/metrics, real-time chat playground with streaming & persistence, standardized benchmark suite, complete model cache management & uninstall, time-series GPU/VRAM sparklines, tray appliance mode with auto-resume & crash recovery, DPAPI token encryption, recipe & config export/import, LAN access controls, **OpenAI-compatible Gateway (port 11434)**). This roadmap orders the features by **value ÷ effort**, consistent with the existing design ethos
("estimates are estimates — measured data wins", idempotent + minimal deps).

Legend: ✅ done · 🔵 planned · work item split into backend (Rust, `src-tauri`)
and frontend (React, `src`).

---

## Phase 1 — Real chat: streaming + conversation memory  ✅ (v0.6.0)

The chat playground is an assistant — streams tokens live, keeps
conversations, and persists them.

- **Backend**
  - [x] `server.rs::chat_stream` — POST `/v1/chat/completions` with
        `"stream": true`, parse SSE `data:` frames via `reqwest` bytes_stream;
        emit `chat-token {request_id, server_id, token}` for each delta.
  - [x] Cancellation — in-flight map `request_id → cancel flag`; abort the
        stream task on cancel (kill reqwest body read) and emit
        `chat-done`/`chat-cancel`.
  - [x] Conversation store in `AppState` + persistence
        (`%APPDATA%/local-llm-panel/conversations.json`, atomic save, same
        pattern as `config.json`): id, server_id, title, created/updated,
        messages `[{role, content}]`.
  - [x] Commands: `servers_chat_stream` (fire-and-forget spawn + return), 
        `servers_chat_cancel`, `conversations_list/save/delete`.
- **Frontend**
  - [x] `types.ts`/`api.ts`: `Conversation`, `ChatStreamEvent`; streaming
        command + event subscription.
  - [x] Replace the minimal chat modal in `Servers.tsx` with a full
        conversation drawer: chat list (select/new/delete/rename), system
        prompt, temperature, streaming token-by-token rendering, markdown
        (code blocks, bold, inline code), auto-scroll, stop button,
        auto-save on completion.

**Acceptance:** start instruct server → send message → tokens render live →
stop mid-stream works → close/reopen app → conversation still there.

---

## Phase 2 — Benchmark suite (measured data wins)  ✅ (v0.7.0)

Take the "measure bare-minimum" idea further: make **measured tok/s** a
first-class feature instead of a byproduct of chat traffic.

- [x] Backend: `benchmarks_run` — send a standardized prompt set
      (small/medium/large context) to a running instruct server, measure
      prompt + generation tok/s per run and aggregate; persist to
      `measured` (reuse existing measurement plumbing).
- [x] Backend: `benchmarks_history(state) -> Vec<BenchmarkRun>` — persist
      runs so results are comparable over time / across quants.
- [x] Frontend: a **Benchmarks** surface (tab or drawer per server) —
      run + stop, stream tok/s live, table of past runs (date, quant,
      prompt/generation tok/s, decode latency).
- [x] Show a "measured vs estimated" chip on the model card (already has
      measured stats — surface the delta prominently).

**Acceptance:** run a 3-prompt benchmark on two quants → see comparable
numbers; tok/s estimates replaced by measured on dashboard.

---

## Phase 3 — Model management (library delete + disk + uninstall)  ✅ (v0.8.0)

`Library` provides full model lifecycle management inside the HF cache.

- [x] Backend: `library_remove(model_id)` — delete from `~/.cache/huggingface`
      (WSL-side `rm -rf models--...`), plus `library_disk_usage` 
      (per-model MB, du 1 cache dir).
- [x] Backend: extend `LibraryEntry` with `quant`, `params_b`, `installed`
      vs `mark_for_deletion`; expose a `servers`-cross-ref so a model in use
      is flagged ("in use — stop first").
- [x] Frontend: Library rows = pull status + disk size + **Delete** button
      (confirm), `installed` badge, filter by task (instruct/embed), optional
      "open in browser" (already exists).

**Acceptance:** pull two models → Library shows sizes → delete one → HF cache
size updates and row disappears.

---

## Phase 4 — Time-series VRAM / GPU monitoring  ✅ (v0.9.0)

Dashboard + Servers graph real-time resource utilization.

- [x] Backend: ring buffer of `MetricsSnapshot`s per running server with
      timestamps (60-120 samples); `servers_metrics_series(id)` returns a
      `Vec<(timestamp, vram_used, gpu_util, tok_s)>`.
- [x] Frontend: mini line chart (no new dep — inline `<svg>` polyline) on
      the dashboard GPU card + per-server metrics card showing VRAM used %
      and tok/s over the last 5 min; add VRAM-pressure warning when
      `vram_used + in-flight weights > free`.

**Acceptance:** run a server + benchmark → see tok/s + VRAM sparkline move in
real time; peaks align with benchmark activity.

---

## Phase 5 — Appliance-ization: tray + autostart + auto-resume  ✅ (v0.10.0)

A true "always-on local LLM appliance" rather than a panel you open.

- [x] Tray icon (Tauri 2) with menu: show/hide, per-server quick Start/Stop,
      Quit.
- [x] `minimize_to_tray` setting; closing the window keeps servers running.
- [x] `launch_at_login` (auto-start app on Windows login, registry or
      `autostart` crate).
- [x] `resume_servers_on_launch` — on app start, auto-start servers that were
      running when the app last shut down (persist `was_running` flag).
- [x] Auto-restart a crashed server (with backoff + max retries).

**Acceptance:** start a server → close window → tray icon remains → click
→ server still up; relaunch app → servers auto-start.

---

## Phase 6 — Token security + export/share  ✅ (v1.0.0)

- [x] Encrypt `hf_token` with Windows DPAPI (CryptProtectData) instead of
      plaintext in config.json.
- [x] Export/import full config (distro, servers, settings) as a single JSON
      file; "share server recipe" copy-to-clipboard.
- [x] LAN toggle for vLLM `--host 0.0.0.0` (consent + warning) for testing
      from other devices.

---

## Phase 7 — OpenAI-compatible Gateway  ✅ (v1.2.0)

Single local endpoint (`http://127.0.0.1:11434/v1`) that routes to the correct
running vLLM/llama.cpp server by model name. Enables Cursor, Continue.dev,
LibreChat, and any OpenAI-compatible client to work without per-server port
configuration.

- [x] Backend: `gateway.rs` — Tokio `TcpListener` on 11434, hand-rolled HTTP/1.1
      framing, `find_server_port_for()` exact/fuzzy routing, SSE streaming proxy,
      `GET /v1/models` listing running instruct servers with ports.
- [x] Backend: `gateway_status` command + `gateway-status` event; supervisor
      task spawned at startup when enabled.
- [x] Frontend: Settings → Gateway toggle + port override; Dashboard shows
      gateway status badge.
- [x] LAN access opt-in (binds 0.0.0.0 with consent warning).

**Acceptance:** start two instruct servers (different models) → enable Gateway
→ from Cursor pointed at 11434, chat with both models by name → tokens stream
correctly.

---

## Stretch (not yet prioritized)

- Multi-GPU tensor-parallel (fit.rs already has a `TensorParallel` run mode),
  prompt-library templates, embedding similarity playground.

---

## Process notes

- Each phase should be its own commit (or a small series) with the version
  bump, following the existing conventional-commit style.
- Unit tests stay Rust-side with the same patterns as `server.rs`/`commands.rs`
  tests. Integration tests remain WSL-gated.
- Keep dependency footprint minimal; inline SVG for charts, no UI libs.
