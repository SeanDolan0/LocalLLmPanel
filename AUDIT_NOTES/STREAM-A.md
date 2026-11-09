# Stream A: Download Integrity & Reliability

**Ownership:** `src-tauri/src/download_manager.rs`, `src-tauri/src/hf.rs`, `src/context/DownloadContext.tsx`, download-related UI components.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| A1 | Duplicate-job race | PENDING | PENDING | | | Atomic check+insert, UUID IDs |
| A2 | No content integrity | PENDING | PENDING | | | SHA-256 verification, HF LFS digests |
| A3 | Range resume validation | PENDING | PENDING | | | Content-Range start validation |
| A4 | Read timeout / retry / cancel responsiveness | PENDING | PENDING | | | Exponential backoff, timeout, cancel aware |
| A5 | Disk-space preflight | PENDING | PENDING | | | Pre-check destination volume |
| A6 | Unbounded thread-per-job | PENDING | PENDING | | | Bounded worker pool, Queued state |
| A7 | Pinned revision | PENDING | PENDING | | | Commit SHA pinning |
| A8 | HEAD dependence | PENDING | PENDING | | | Ranged GET fallback |
| A9 | Registry failure swallowed | PENDING | PENDING | | | Surface library registration error |
| A10 | Frontend job-state bugs | PENDING | PENDING | | | Fix duplicate IDs, clear completed, controller cleanup |
| A11 | vLLM/WSL download parity | PENDING | PENDING | | | Capability flags in job model |
| A12 | `pull_cancel` regex bug | PENDING | PENDING | | | Exact child PID tracking (coord with Stream C) |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
