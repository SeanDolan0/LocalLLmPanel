# Stream D: Security & LAN (Wave 3)

**Ownership:** `src-tauri/src/state.rs`, `src/pages/Settings.tsx`, key/bind sections of `server.rs::launch_script` and `server_api_key()`.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| D1 | API keys in plaintext | PENDING | PENDING | | | DPAPI encrypt API keys, migrate existing |
| D2 | LAN access | PENDING | PENDING | | | Require key for 0.0.0.0, clarify WSL NAT/portproxy |
| D3 | Global key vs native llama.cpp | PENDING | PENDING | | | Apply global key consistently unless overridden |
| D4 | Passwordless sudo scoping | PENDING | PENDING | | | Scope or use wsl --user root |
| D5 | Native llama.cpp API key argument | PENDING | PENDING | | | Environment variable or key file |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
