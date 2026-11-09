# Stream E: Gateway & Agentic Compatibility

**Ownership:** src-tauri/src/gateway.rs (+ gateway tests, minimal surgical support in commands.rs, 	ypes.ts, Settings.tsx).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| E1 | Responses API tool-call streaming | YES | RESOLVED | src-tauri/src/gateway.rs | 	est_responses_stream_tool_calls, 	est_conformance_responses_api_streaming_tool_call | Full tool call lifecycle emitted: esponse.output_item.added, esponse.function_call_arguments.delta, esponse.function_call_arguments.done, esponse.output_item.done, and final esponse.completed with output items preserving call IDs/names. |
| E2 | Port conflict loop & health verification | YES | RESOLVED | src-tauri/src/gateway.rs, src-tauri/src/commands.rs, src/types.ts, src/pages/Settings.tsx | 	est_port_conflict_state, 	est_verify_gateway_health, 	est_conformance_port_conflict_and_identity | Halts restart loop on persistent port conflict (SupervisorState::PortConflict), bounded exponential backoff for transient errors, and erify_gateway_health verifies service header and body ({"service":"llm-panel-gateway"}). UI displays conflict warning. |
| E3 | Model-name routing ambiguity | YES | RESOLVED | src-tauri/src/gateway.rs | 	est_route_server_for_exact_match_wins, 	est_route_server_fuzzy_matches, 	est_route_server_ambiguous_fuzzy_fails | Exact model ID matches always win. Ambiguous fuzzy matches return ModelRouteError::Ambiguous with candidate model IDs and HTTP 400 Bad Request (mbiguous_model). |
| E4 | Concurrency policy & rate limiting | YES | RESOLVED | src-tauri/src/gateway.rs | 	est_concurrency_manager_limit_and_release, 	est_concurrency_manager_offline_server, 	est_conformance_concurrency_limits_and_backpressure | ServerConcurrencyManager with RAII ServerPermit tracking in-flight requests per server. Returns HTTP 429 Too Many Requests (etry-after: 1) on limit exceeded (default 16 or parallel), and HTTP 503 when server is starting/offline. |
| E5 | End-to-end conformance test suite | YES | RESOLVED | src-tauri/src/gateway.rs | 10 comprehensive mock integration tests in 	ests module | Conformance test suite covers: non-streaming chat completions, streaming chat completions, tool call arguments parsing, structured JSON output (esponse_format), developer/system prompt normalization, Responses API streaming with tool calls, auth rejection (401/503), CORS preflight & origin rejection (403), port conflict detection, and concurrency limiting. |

## Verification & Resolution Details

- **E1: Responses API Tool-Call Streaming**
  - **Root Cause:** In proxy_responses_stream, chunks were inspected only for delta.content. delta.tool_calls arrays from OpenAI-compatible chat completion backends were discarded.
  - **Fix:** Added accumulator state tracking in-flight tool calls by index. Translates tool calls to Responses API events: esponse.output_item.added (type unction_call), esponse.function_call_arguments.delta for streaming argument fragments, esponse.function_call_arguments.done, and esponse.output_item.done. The terminal esponse.completed payload packages full output items with tool call IDs and arguments intact.
  - **Tests:** 	est_responses_stream_tool_calls, 	est_conformance_responses_api_streaming_tool_call.

- **E2: Port Conflict Loop & Identity Verification**
  - **Root Cause:** When gateway port was occupied by an existing process, supervisor looped every 1000ms attempting to bind, consuming CPU and spamming error logs. Furthermore, gateway_status checked TCP socket reachability without verifying that the listener was actually llm-panel-gateway.
  - **Fix:** Introduced SupervisorState::PortConflict and GATEWAY_PORT_CONFLICT atomic. Port bind errors immediately halt retries until port reconfiguration or explicit user intervention. Transient errors use exponential backoff (1s -> 2s -> 4s -> max 16s). Added erify_gateway_health(port) which checks /health for X-Gateway-Service: llm-panel-gateway and JSON {"status":"ok","service":"llm-panel-gateway"}. Exposed port_conflict in gateway_status command, GatewayStatus TypeScript type, and added an amber port conflict alert banner in Settings.tsx.
  - **Tests:** 	est_port_conflict_state, 	est_verify_gateway_health, 	est_conformance_port_conflict_and_identity.

- **E3: Model-Name Routing & Ambiguity Handling**
  - **Root Cause:** ind_server_port_for matched exact name or alias, but on substring/prefix match, returned the first hit based on HashMap iteration order, leading to non-deterministic routing when multiple models shared prefixes.
  - **Fix:** Implemented oute_server_for(requested_model, servers) with deterministic two-tier resolution:
    1. Exact match on model_id or 
ame immediately wins.
    2. Fuzzy match (case-insensitive substring/prefix): if exactly 1 match, routes to it; if 2 or more matches, returns ModelRouteError::Ambiguous { requested, candidates }.
    3. If 0 matches, returns ModelRouteError::NotFound.
    4. HTTP proxy layers map ModelRouteError::Ambiguous to HTTP 400 Bad Request with error code mbiguous_model and list of candidate models in the JSON error response.
  - **Tests:** 	est_route_server_for_exact_match_wins, 	est_route_server_fuzzy_matches, 	est_route_server_ambiguous_fuzzy_fails.

- **E4: Gateway Concurrency Policy & In-flight Limiting**
  - **Root Cause:** Requests were forwarded directly to backends without gateway-level concurrency tracking. Overloaded backends would hang or crash rather than providing early backpressure.
  - **Fix:** Added ServerConcurrencyManager backed by Arc<ServerConcurrencyManager> (CONCURRENCY_MANAGER). Tracks in-flight request count per server.
    - If active requests >= server limit (configured via server.def.parallel, default 16), immediately returns HTTP 429 Too Many Requests with Retry-After: 1 and JSON error code server_concurrency_limit_exceeded.
    - If server is in Starting or Stopped status, returns HTTP 503 Service Unavailable with descriptive retry guidance.
    - Uses RAII ServerPermit to guarantee counter decrement on normal completion, stream termination, client disconnect, or error.
  - **Tests:** 	est_concurrency_manager_limit_and_release, 	est_concurrency_manager_offline_server, 	est_conformance_concurrency_limits_and_backpressure.

- **E5: End-to-End Conformance Test Suite**
  - **Root Cause:** Prior to this work, gateway.rs had only 1 mock HTTP upstream test, leaving streaming, tool calls, structured outputs, prompt normalization, auth, and CORS without automated regression testing.
  - **Fix:** Added comprehensive test suite with simulated mock upstream servers:
    1. 	est_conformance_chat_completions_basic: OpenAI chat completions request/response schema.
    2. 	est_conformance_chat_completions_streaming: SSE stream forwarding and terminal chunk handling.
    3. 	est_conformance_tool_calling: Chat completion tool call invocation with function name and JSON arguments.
    4. 	est_conformance_structured_output: Validates esponse_format: { type: "json_object" } forwarding.
    5. 	est_conformance_developer_prompt_normalization: Normalizes developer role to system for backend compatibility.
    6. 	est_conformance_responses_api_streaming_tool_call: Full Responses API SSE stream translation with tool call lifecycle.
    7. 	est_conformance_auth_rejection: HTTP 401 on missing/invalid bearer token; HTTP 503 if secret unconfigured.
    8. 	est_conformance_cors_handling: HTTP 204 on OPTIONS preflight with Access-Control-Allow-Origin: *, HTTP 403 on untrusted origin with strict policy.
    9. 	est_conformance_port_conflict_and_identity: Tests gateway /health identity verification against mock foreign servers.
    10. 	est_conformance_concurrency_limits_and_backpressure: Asserts HTTP 429 when max in-flight permits are held.

## Quality Gate Checklist
- [x] cargo test --workspace passes (503 passed, 0 failed, 1 ignored)
- [x] cargo clippy --workspace --all-targets has no new warnings in owned code (src/gateway.rs has 0 clippy warnings)
- [x] 
pm run build passes with 0 errors
