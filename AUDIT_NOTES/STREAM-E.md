# Stream E: Gateway & Agentic Compatibility

**Ownership:** src-tauri/src/gateway.rs (+ gateway tests).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| E1 | Responses API tool-call streaming | YES | IN_PROGRESS | src-tauri/src/gateway.rs | Pending | Tool calls were dropped in proxy_responses_stream; implementing full item and delta lifecycle. |
| E2 | Port conflict loop | YES | IN_PROGRESS | src-tauri/src/gateway.rs, src-tauri/src/commands.rs | Pending | Supervisor loops every second on bind failure; gateway_status checked TCP only without identity check. |
| E3 | Model-name routing | YES | IN_PROGRESS | src-tauri/src/gateway.rs | Pending | Multiple fuzzy matches picked first in map order; implementing route_server_for with ambiguity error. |
| E4 | Concurrency policy | YES | IN_PROGRESS | src-tauri/src/gateway.rs | Pending | No in-flight tracking; implementing per-server concurrency limiter with 429/503 responses. |
| E5 | Conformance tests | YES | IN_PROGRESS | src-tauri/src/gateway.rs | Pending | Only 1 basic mock test existed; adding comprehensive mock test suite. |

## Verification Details

- **E1**: Verified lines 4771-4860 in gateway.rs. proxy_responses_stream only processed delta.content. 	ool_calls in stream chunks were completely ignored.
- **E2**: Verified lines 4946-5003 in gateway.rs and 3056-3082 in commands.rs. Bind failure caused immediate supervisor retry every 1 second without persistent state or backoff. gateway_status checked TCP socket connection only.
- **E3**: Verified lines 233-262 in gateway.rs. ind_server_port_for returned the first matching server when multiple fuzzy matches were present.
- **E4**: Verified proxy_post and related proxy handlers. There was no per-server concurrency tracking or semaphore/in-flight accounting, risking backend starvation without 429/503.
- **E5**: Verified existing tests only had one mock upstream integration test. Conformance coverage for streaming, tools, structured output, system prompts, Responses tool streaming, and CORS is needed.

## Quality Gate Checklist
- [ ] cargo test passes
- [ ] cargo clippy --workspace --all-targets has no new warnings
- [ ] 
pm run build passes
