# Stream F: CI, Testing, Docs, Ops

**Ownership:** `.github/workflows/*`, `README.md`, `ABOUT.md`, `docs/`, test infrastructure.

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| F1 | Security audit must gate releases | PENDING | PENDING | | | Blocking cargo/npm audit with allowlist |
| F2 | Installer packages in CI | PENDING | PENDING | | | NSIS/MSI in Windows CI path |
| F3 | GPU/WSL integration test harness | PENDING | PENDING | | | #[ignore] live tests |
| F4 | Rust/frontend contract drift | PENDING | PENDING | | | IPC contract drift check |
| F5 | README/ABOUT corrections | PENDING | PENDING | | | Align documentation with verified reality |
| F6 | Guided uninstall / environment cleanup | PENDING | PENDING | | | Distro/venv/cache cleanup command |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
