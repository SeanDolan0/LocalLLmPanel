# P0 Gateway Completeness — Design

Date: 2026-09-20
Status: approved in-chat, pending spec review
Scope: coding-backend P0 only (gateway completeness + client configs + safe fallback)

## 1. Intent

Make Local LLM Panel a reliable always-on backend for coding tools
(Continue.dev, Cursor, Cline) via the unified gateway on
`127.0.0.1:<gateway_port>` (default 11434).

Decisions locked with user:
- Unknown model → `404 + hint list` of running models. No auto-start.
- Auth → forward-only. Gateway does not enforce `api_key`; it forwards
  `Authorization: Bearer <api_key>` to vLLM via existing `apply_vllm_auth`.
- Keep hand-rolled `TcpListener` HTTP layer. No new web framework.

## 2. Current state

- `src-tauri/src/gateway.rs::route` handles only:
  - `GET /v1/models`
  - `POST /v1/chat/completions` (instruct servers only)
- `find_server_port_for_model` matches served name / HF id exact, then
  fuzzy suffix. Ignores embed + stopped servers.
- `proxy_chat` hardcodes upstream path `/v1/chat/completions`, supports
  SSE chunked streaming.
- `write_response` emits no CORS headers; no `OPTIONS` handling.
- Settings gateway card (`src/pages/Settings.tsx`) shows port + base URL
  copy only. No Continue/Cursor snippets.

## 3. Changes

### 3.1 Backend (`src-tauri/src/gateway.rs` only)

Add routes in `route()`:
- `GET /health` → `200 application/json {"status":"ok"}`. No model lookup.
- `OPTIONS *` → `204` with CORS headers, empty body. Matches any target
  when method is `OPTIONS`.
- `POST /v1/embeddings` → route via new
  `find_server_port_for_embed(model)` (mirrors instruct logic, filters
  `task == "embed"`, `Running` only). Proxy to upstream
  `/v1/embeddings`, non-streaming.
- `POST /v1/completions` → route via existing
  `find_server_port_for_model` (instruct only). Proxy to upstream
  `/v1/completions`, body passed through verbatim (preserves FIM fields
  `suffix`, `stop`, `max_tokens`). Support streaming flag if present.
- Keep `POST /v1/chat/completions` behavior unchanged.

Generalize proxy:
- Rename/extract `proxy_chat(stream, state, port, body)` → generic
  `proxy_post(stream, state, port, upstream_path, body)` handling both
  streaming (chunked SSE) and non-streaming JSON. Chat/completions use
  streaming detection; embeddings use non-streaming.

404 payload:
- `not_found_payload(model, running: &[String])` →
  `{"error":{"message":"no running <kind> server for model 'X'. Running: [a, b]","type":"invalid_request_error"},"running":[...]}`.
  `kind` is `instruction` or `embedding` depending on route.

CORS:
- Append to every `write_response` + streaming head:
  `Access-Control-Allow-Origin: *`,
  `Access-Control-Allow-Methods: GET, POST, OPTIONS`,
  `Access-Control-Allow-Headers: Content-Type, Authorization`.

No changes to: `start_gateway`, `spawn_supervisor`, `state.rs`,
`server.rs`, ports, DPAPI, supervisor polling.

### 3.2 Frontend (`src/pages/Settings.tsx` gateway card)

Extend existing gateway card (no new Tauri commands):
- Data: `adv.gateway_port` + `api.serversList()` (first running instruct
  + embed names) to fill snippets.
- Buttons:
  - Copy Continue `config.yaml` (provider `openai`, baseURL
    `http://127.0.0.1:<port>/v1`, chat + autocomplete + embeddings blocks).
  - Copy Cursor / Cline baseURL + model line.
  - Copy `curl` for chat and embeddings.
- Show hint when no instruct/embed running: "start a server first".

### 3.3 Error handling

- Missing/invalid `model` field → `500 gateway error: request body must
  be JSON with a model field` (existing behavior preserved).
- Unknown model → `404` with hint list (section 3.1). No auto-start, no
  VRAM check needed.
- Upstream vLLM non-2xx → status + body passed through unchanged.
- Body > 32MB / head > 32KB → existing `500` limits unchanged.

## 4. Testing

Rust unit (`gateway.rs::tests`):
- `find_server_port_for_embed` exact + fuzzy + ignores instruct/stopped.
- completions routes to instruct, embeddings routes to embed.
- 404 payload contains `running` list.
- `OPTIONS` returns CORS headers; `GET /health` shape.
- Existing chat routing tests keep passing.

Manual acceptance:
1. Start 1 instruct + 1 embed server.
2. `GET /health` → ok; `GET /v1/models` lists both instruct names.
3. `POST /v1/chat/completions`, `/v1/completions`,
   `/v1/embeddings` via `curl` through gateway succeed.
4. Unknown model → 404 with running list.
5. Paste copied Continue config → chat + autocomplete + embeddings work.
6. `cargo test` green.

## 5. Out of scope

- Gateway-level API key enforcement.
- Auto-start stopped servers on 404.
- `/v1/audio/*`, rerank, files, fine-tuning routes.
- axum migration, new dependencies.
- `api_key` DPAPI encryption (separate hardening item).
- CSP, updater, CI (separate hardening items).

## 6. Self-review

- No TBDs. All routes, payloads, files explicitly named.
- Consistent with locked decisions (404+hint, forward-only auth).
- Single-plan scope: `gateway.rs` + Settings card only.
- Ambiguities resolved: FIM passthrough verbatim; CORS `*`;
  embeddings non-streaming; completions reuse chat streaming detection.
