# Stream A: Download Integrity & Reliability

**Ownership:** `src-tauri/src/download_manager.rs`, `src-tauri/src/hf.rs`, `src/context/DownloadContext.tsx`, download-related UI components (`src/components/ActiveDownloadsPanel.tsx`).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| A1 | Duplicate-job race | YES | COMPLETED | `src-tauri/src/download_manager.rs` | `test_atomic_job_registration`, `test_uuid_v4_generation` | Atomic duplicate check and job insertion under single Mutex lock; RFC 4122 UUID v4 IDs generated; worker thread dispatched after registration. |
| A2 | No content integrity | YES (needs live test) | COMPLETED | `src-tauri/src/download_manager.rs`, `src/components/ActiveDownloadsPanel.tsx`, `src/context/DownloadContext.tsx` | `test_checksum_mismatch_not_promoted` | Fetches Git LFS SHA-256 digests from repo tree API; validates hash before `.part` promotion; re-verifies pre-existing files; UI distinguishes 'SHA-256 Verified' from 'Size Checked (no checksum)' and handles verifying status. |
| A3 | Range resume validation | YES | COMPLETED | `src-tauri/src/download_manager.rs` | `test_content_range_mismatch_rejection` | Validates HTTP 206 `Content-Range` start offset equals local `.part` size; resets offset and truncates `.part` on mismatch or HTTP 200 response; never appends mismatched data. |
| A4 | Read timeout / retry / cancel responsiveness | YES (needs live test) | COMPLETED | `src-tauri/src/download_manager.rs` | `test_retry_backoff_against_mock_server`, `test_stalled_read_cancel` | 30s timeout configured; checks cancel/pause flags inside chunk read loop; exponential backoff + jitter up to 5 retries; honors `Retry-After`; retries 429/500/502/503/504 and network errors; fails immediately on 401/403/404; resumes from verified offset. |
| A5 | Disk-space preflight | YES | COMPLETED | `src-tauri/src/download_manager.rs`, `src/components/ActiveDownloadsPanel.tsx` | Unit verified via Win32 `GetDiskFreeSpaceExW` | Evaluates free disk space on destination drive against aggregate size + 512MB safety margin prior to downloading; surfaces required, available, and path in error banner. |
| A6 | Unbounded thread-per-job | YES | COMPLETED | `src-tauri/src/download_manager.rs` | `test_bounded_worker_pool_and_queue` | Bounded worker pool with `max_concurrent` limit (default 2); `Queued` is an authentic state; queued jobs automatically promoted as slots open; cancellation of queued jobs requires no worker thread cleanup. |
| A7 | Pinned revision | YES (needs live test) | COMPLETED | `src-tauri/src/download_manager.rs`, `src-tauri/src/hf.rs`, `src/components/ActiveDownloadsPanel.tsx`, `src/context/DownloadContext.tsx` | Verified via commit SHA pinning in URL builders & WSL args | Resolves 40-character commit SHA at inspection/start; pins shard download URLs to `resolve/{commit_sha}/{file}`; stores revision in job model & UI; passes `--revision {sha}` to vLLM download CLI. |
| A8 | HEAD dependence | YES | COMPLETED | `src-tauri/src/download_manager.rs` | Verified via `resolve_remote_file_size` | Falls back to ranged `GET bytes=0-0` to extract total size from `Content-Range` or `Content-Length` if HEAD fails/unsupported; fails immediately on 401/403/404 without retries. |
| A9 | Registry failure swallowed | YES | COMPLETED | `src-tauri/src/download_manager.rs`, `src/components/ActiveDownloadsPanel.tsx` | Verified via error check & UI presentation | Checks `self.registry.register()`; surfaces failure as `"downloaded, library registration failed: {error}"` state with an interactive "Rescan Library" repair action in UI. |
| A10 | Frontend job-state bugs | YES | COMPLETED | `src/context/DownloadContext.tsx`, `src-tauri/src/download_manager.rs` | Verified via `clearCompleted` & `resumeDownload` & controller cleanup | `resumeDownload` captures new job ID and deletes old key preventing duplicate entries; `clearCompleted` retains `Failed` jobs matching backend; backend cleans up controller map on all terminal job states. |
| A11 | vLLM/WSL download parity | YES | COMPLETED | `src-tauri/src/download_manager.rs`, `src/context/DownloadContext.tsx`, `src/components/ActiveDownloadsPanel.tsx` | Verified capability flags | Explicit `supports_pause`, `supports_resume`, `supports_cancel` flags in job model and progress payloads; UI hides/disables pause and resume for vLLM downloads while preserving cancellation. |
| A12 | `pull_cancel` regex bug | YES | SUBMITTED | `AUDIT_NOTES/cross-stream-requests.md` | Request XREQ-01 | Submitted cross-stream request XREQ-01 for Stream C (C11) to track and kill exact child PID in WSL/commands.rs instead of wildcard process regex. |

## Quality Gate Checklist
- [x] `cargo test` passes (503 passed, 0 failed, 1 ignored)
- [x] `cargo clippy --workspace --all-targets` has no new warnings (0 warnings in owned files)
- [x] `npm run build` passes (TypeScript compilation & Vite production bundle clean)

## Automated Tests Added (src-tauri/src/download_manager.rs)
1. `test_uuid_v4_generation`: Verifies generation of valid RFC 4122 Version 4 UUIDs with correct version and variant bits.
2. `test_atomic_job_registration`: Spawns 20 concurrent threads attempting to register the same model and quant; verifies exactly one succeeds under Mutex lock and 19 are rejected with duplicate error.
3. `test_content_range_mismatch_rejection`: Tests range resume validation; validates 206 with matched start, rejects mismatched start, handles 200 full content restart, and handles 416 range not satisfiable.
4. `test_checksum_mismatch_not_promoted`: Confirms that a corrupt shard failing SHA-256 digest comparison is NOT promoted to the final destination file and `.part` is removed.
5. `test_retry_backoff_against_mock_server`: Spins up mock HTTP server returning 503 and 429 before succeeding; verifies exponential backoff, Retry-After compliance, and successful retry recovery.
6. `test_stalled_read_cancel`: Simulates a stalled socket stream; tests cancel signal triggering immediate, responsive termination of the download read loop.
7. `test_bounded_worker_pool_and_queue`: Verifies that exceeding concurrency cap leaves subsequent jobs in `Queued` state and that completing active jobs dispatches next queued job.
8. `test_multi_quant_downloads_coexist_and_cancel_isolation`: Verifies multiple quant downloads of the same repository coexist cleanly and can be cancelled independently.
