# Baseline Test & Build Status (Phase 0)

Recorded: 2026-11-09

## Rust Backend
- `cargo test --workspace --manifest-path src-tauri/Cargo.toml`:
  - Result: 503 passed, 0 failed, 1 ignored (doc tests 0).
  - Clean test suite.
- `cargo clippy --workspace --all-targets --manifest-path src-tauri/Cargo.toml`:
  - Result: 0 errors, 49 warnings (36 duplicates, primarily `cloned_ref_to_slice_refs`, `bool_assert_comparison`, `field_reassign_with_default`).

## Frontend
- `npm run build` (`tsc --noEmit && vite build`):
  - Result: Clean build, 0 errors, output generated in `dist/`.
- `npm run test:contract` (`node --test scripts/frontend-smoke.mjs`):
  - Result: 10 passed, 0 failed.
- `npm run release:check` (`node scripts/validate-release.mjs --setup`):
  - Result: Validated v2.5.0.
