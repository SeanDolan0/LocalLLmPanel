# Stream D: Security & LAN (Wave 3)

**Ownership:** `src-tauri/src/state.rs`, `src/pages/Settings.tsx`, key/bind sections of `server.rs::launch_script` and `server_api_key()`.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| D1 | API keys in plaintext | YES | FIXED | `src-tauri/src/state.rs`, `src-tauri/src/gateway.rs`, `src/pages/Settings.tsx` | `state::tests::test_dpapi_api_key_roundtrip_and_migration`, `state::tests::test_server_recipe_and_config_export_roundtrip` | DPAPI encryption applied to `advanced_settings.api_key` and `ServerDef.api_key` on save/load with automatic plaintext migration; decrypted only when building launch args; redacted in exports and UI copy updated. |
| D2 | LAN access | YES | FIXED | `src-tauri/src/server.rs`, `src/pages/Settings.tsx` | `server::tests::test_start_server_refuses_0_0_0_0_without_api_key` | Start refused on `0.0.0.0` without configured API key. UI blocks enabling LAN without key. Copy documents WSL2 NAT isolation requiring mirrored mode (`networkingMode=mirrored`) or `netsh interface portproxy`, noting native llama.cpp binds 127.0.0.1. |
| D3 | Global key vs native llama.cpp | YES | FIXED | `src-tauri/src/server.rs`, `src/pages/Servers.tsx` | `server::tests::test_server_api_key_applies_to_vllm_and_llamacpp`, `server::tests::test_llamacpp_args_decrypts_api_key` | `server_api_key()` updated so global key applies to both vLLM and llama.cpp unless overridden by per-server key; llama.cpp startup inherits global key. |
| D4 | Passwordless sudo scoping | YES | FIXED | `src-tauri/src/provision.rs`, `src-tauri/src/cleanup.rs` | `cleanup::tests::test_cleanup_plan_itemization_dry_run` | Sudoers rule scoped to `/etc/sudoers.d/llm-panel-sudoers` limited to `/usr/bin/apt-get, /usr/bin/apt`; legacy `99-llm-lp` removed; cleanup workflow configured to remove both. |
| D5 | Native llama.cpp API key argument | YES | FIXED | `src-tauri/src/server.rs` | `server::tests::test_llamacpp_args_decrypts_api_key` | Prefers `--api-key-file` when advertised by binary `--help` (written to secure run directory); falls back to `--api-key` with logged security diagnostic warning regarding process argument visibility. |

## Quality Gate Checklist
- [x] `cargo test` passes (417 library tests passed, 0 failed)
- [x] `cargo clippy --workspace --all-targets` has no new warnings (0 errors)
- [x] `npm run build` passes (TypeScript and Vite production bundle clean)
