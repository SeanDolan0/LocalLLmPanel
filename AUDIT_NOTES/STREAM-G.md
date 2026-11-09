# Stream G: First-Run UX & Windows Trust (Wave 3)

**Ownership:** frontend pages/components not owned elsewhere (`src/pages/*`, `src/components/*`).

## Status Table

| ID | Issue | Verified? | Status | Files Changed | Tests Added | Notes |
|---|---|---|---|---|---|---|
| G1 | Unified first-run diagnostic | PENDING | PENDING | | | Ordered diagnostic chain |
| G2 | Distinguish error sources | PENDING | PENDING | | | Download vs install vs start separation |
| G3 | Windows 10 support clarity | PENDING | PENDING | | | Explicit version check & docs |
| G4 | SmartScreen/installer trust guidance | PENDING | PENDING | | | First-run warning guidance |
| G5 | Exportable diagnostic bundle | PENDING | PENDING | | | Redacted diagnostic bundle export |
| G6 | Storage planning UI | PENDING | PENDING | | | Consumes A5 disk preflight |
| G7 | Per-server compatibility test button | PENDING | PENDING | | | Interactive test using E5 runner |

## Quality Gate Checklist
- [ ] `cargo test` passes
- [ ] `cargo clippy --workspace --all-targets` has no new warnings
- [ ] `npm run build` passes
