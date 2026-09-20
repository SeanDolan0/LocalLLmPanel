# Complete Roadmap Implementation Plan (Phases 1 - 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 6 phases of `docs/ROADMAP.md` (Real chat streaming & memory, Benchmark suite, Model management & delete, Time-series VRAM/GPU monitoring, Appliance-ization tray & auto-resume, Token security & export/share), verifying Rust tests and frontend build at every stage.

**Architecture:** 
- Backend: Tauri 2 Rust core (`src-tauri/src/`) extending `server.rs`, `state.rs`, `commands.rs`, `lib.rs` with robust SSE streaming, cancellation maps, persistent JSON stores, tray support, DPAPI encryption, and WSL integration.
- Frontend: React + TypeScript + Tailwind CSS (`src/`) extending `api.ts`, `types.ts`, `Servers.tsx`, `Library.tsx`, `Dashboard.tsx`, `Settings.tsx` with streaming chat UI, benchmark drawer & history, disk usage & uninstall controls, inline SVG time-series sparklines, tray/autostart toggles, recipe export, and LAN access controls.

**Tech Stack:** Tauri 2, Rust 2021, Tokio, Reqwest, React 18, TypeScript 5, Tailwind CSS 4, Vite 6.

**Spec:** `docs/ROADMAP.md`

## Global Constraints
- Minimal dependencies: use inline SVG for charts, standard library and existing crates where possible.
- Idempotent and clean: unit tests in Rust for new functionality, safe handling of system clocks, no breaking regressions.
- Version bumps and conventional commits for each phase.

---

### Task 1: Phase 1 — Real Chat Streaming & Conversation Memory (Backend)
**Files:**
- Modify: `src-tauri/src/state.rs` (ChatMessage, Conversation, persistence in %APPDATA%/conversations.json, cancellation map)
- Modify: `src-tauri/src/server.rs` (chat_stream with SSE parsing, cancellation check, event emission)
- Modify: `src-tauri/src/commands.rs` (servers_chat_stream, servers_chat_cancel, conversations_list, conversations_save, conversations_delete)
- Modify: `src-tauri/src/lib.rs` (register new commands)
- Test: `src-tauri/src/server.rs` / `src-tauri/src/state.rs` unit tests

### Task 2: Phase 1 — Real Chat Streaming & Conversation Memory (Frontend)
**Files:**
- Modify: `src/types.ts` (Conversation, ChatMessage, events)
- Modify: `src/api.ts` (chatStream, cancelChatStream, conversations APIs, event listeners)
- Modify: `src/pages/Servers.tsx` (Conversation drawer, streaming tokens, stop button, markdown, persistent chats)
- Test: `npm run build` verification

### Task 3: Phase 2 — Benchmark Suite (Backend & Frontend)
**Files:**
- Modify: `src-tauri/src/state.rs` (BenchmarkRun, persistence in %APPDATA%/benchmarks.json)
- Modify: `src-tauri/src/server.rs` (benchmarks_run execution with standardized prompts)
- Modify: `src-tauri/src/commands.rs` (benchmarks_run, benchmarks_history, benchmarks_cancel)
- Modify: `src-tauri/src/lib.rs` (register commands)
- Modify: `src/types.ts` & `src/api.ts` (BenchmarkRun types and APIs)
- Modify: `src/pages/Servers.tsx` / `src/pages/Dashboard.tsx` (Benchmark drawer/modal, history table, measured vs estimated chips)
- Test: `cargo test` and `npm run build`

### Task 4: Phase 3 — Model Management (Library Delete, Disk Usage & Uninstall)
**Files:**
- Modify: `src-tauri/src/wsl.rs` (library_remove `rm -rf`, library_disk_usage `du -sm`)
- Modify: `src-tauri/src/commands.rs` (library_remove, library_disk_usage, extend LibraryEntry with in_use check)
- Modify: `src-tauri/src/lib.rs` (register commands)
- Modify: `src/types.ts` & `src/api.ts`
- Modify: `src/pages/Library.tsx` (Disk size display, Delete button with confirmation, in-use flag, task filters)
- Test: `cargo test` and `npm run build`

### Task 5: Phase 4 — Time-Series VRAM & GPU Monitoring
**Files:**
- Modify: `src-tauri/src/state.rs` (Ring buffer for metrics history per server and system)
- Modify: `src-tauri/src/commands.rs` (servers_metrics_series, system_metrics_series)
- Modify: `src-tauri/src/lib.rs` (register commands)
- Modify: `src/types.ts` & `src/api.ts`
- Modify: `src/pages/Dashboard.tsx` & `src/pages/Servers.tsx` (Inline SVG sparkline polylines, VRAM pressure warnings)
- Test: `cargo test` and `npm run build`

### Task 6: Phase 5 — Appliance-ization: Tray, Autostart & Auto-Resume
**Files:**
- Modify: `src-tauri/Cargo.toml` (Tauri tray feature / windows registry)
- Modify: `src-tauri/src/state.rs` (app settings: minimize_to_tray, launch_at_login, resume_servers_on_launch, was_running)
- Modify: `src-tauri/src/lib.rs` (Tray icon setup, menu, window close interceptor, resume on startup)
- Modify: `src-tauri/src/commands.rs` (app settings commands, registry autostart)
- Modify: `src/types.ts`, `src/api.ts`, `src/pages/Settings.tsx` (Toggles for minimize to tray, launch at login, resume servers)
- Test: `cargo test` and `npm run build`

### Task 7: Phase 6 — Token Security & Export/Share
**Files:**
- Modify: `src-tauri/src/state.rs` & `src-tauri/src/commands.rs` (Windows DPAPI encryption for hf_token, config export/import, server recipe export/import, LAN toggle `--host 0.0.0.0`)
- Modify: `src-tauri/src/server.rs` (LAN host binding)
- Modify: `src/types.ts`, `src/api.ts`, `src/pages/Settings.tsx`, `src/pages/Servers.tsx` (Recipe copy button, export/import buttons, LAN access toggle with consent warning)
- Test: `cargo test` and `npm run build`

### Task 8: Verification & Roadmap Update
**Files:**
- Update: `docs/ROADMAP.md` (Update status markers to ✅)
- Run full test suite (`cargo test` & `npm run build`)
- Final review against all acceptance criteria
