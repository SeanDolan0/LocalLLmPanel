# Stream E: Gateway & Agentic Compatibility

**Ownership:** `src-tauri/src/gateway.rs` (+ gateway tests).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| E1 | Responses API tool-call streaming | PENDING | PENDING | | | Streamed tool-call delta lifecycle |
| E2 | Port conflict loop | PENDING | PENDING | | | Persistent PortConflict state & health check |
| E3 | Model-name routing | PENDING | PENDING | | | Exact match priority, ambiguity error |
| E4 | Concurrency policy | PENDING | PENDING | | | Enforce or document per-server policy |
| E5 | Conformance tests | PENDING | PENDING | | | Mock backend test suite |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
