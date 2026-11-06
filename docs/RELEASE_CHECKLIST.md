# Production Release Checklist

Every release must pass all gates below before being tagged and published.

---

## 1. Automated Verification Gates

Run the following test suites and deterministic checks from a clean working copy:

- [ ] **Dependency check**: `npm ci` completes successfully with clean audit.
- [ ] **Contract test suite**: `npm run test:contract` passes (verifies manifest schema, target architecture filtering, IPC signatures, and config sanity).
- [ ] **Frontend build**: `npm run build` passes with zero TypeScript and bundling errors.
- [ ] **Rust check**: `cargo check --workspace --locked --manifest-path src-tauri/Cargo.toml` passes.
- [ ] **Rust unit test suite**: `cargo test --workspace --locked --manifest-path src-tauri/Cargo.toml` passes all tests (including updater version comparison, serialization, downgrade rejection, and minisign signature verification tests).
- [ ] **Formatting check**: `cargo fmt --all -- --check` reports zero formatting discrepancies.

---

## 2. Key Management & Signing Gates

- [ ] **Public Key Integrity**: Verify `src-tauri/tauri.conf.json` contains the expected Minisign public key in `plugins.updater.pubkey`.
- [ ] **No Secret Keys in Git**: Verify that neither `llm_panel.key`, `.tauri/`, nor any raw private keys or certificates are tracked in Git (`git status --ignored`).
- [ ] **GitHub Secrets Verification**: Ensure CI repository secrets are active:
  - `TAURI_SIGNING_PRIVATE_KEY` (Minisign secret key string)
  - `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (Secret key password, if configured)
  - `WINDOWS_CERTIFICATE_BASE64` & `WINDOWS_CERTIFICATE_PASSWORD` (If Authenticode signing is active)
- [ ] **Key Continuity**: Confirm that the private key matches the public key embedded in previous releases. *(Warning: rotating the private key breaks updates for all existing installations!)*

---

## 3. Artifact & Metadata Validation (`scripts/validate-release.mjs`)

Build the staged release artifacts:
```bash
npm run build:setup
node scripts/validate-release.mjs --setup --artifacts
```

Ensure all 8 required release files exist in `dist-release/`:
- [ ] `LocalLLMPanel-v<version>.exe` (Standalone executable)
- [ ] `LocalLLMPanel-v<version>.exe.sha256`
- [ ] `Local.LLM.Panel_<version>_x64-setup.exe` (NSIS setup installer)
- [ ] `Local.LLM.Panel_<version>_x64-setup.exe.sha256`
- [ ] `Local.LLM.Panel_<version>_x64-setup.nsis.zip` (Tauri updater zip bundle)
- [ ] `Local.LLM.Panel_<version>_x64-setup.nsis.zip.sha256`
- [ ] `Local.LLM.Panel_<version>_x64-setup.nsis.zip.sig` (Minisign signature file)
- [ ] `latest.json` (Updater manifest)

Validate `latest.json` contents:
- [ ] `version` matches the bumped release version.
- [ ] `platforms["windows-x86_64"]` is defined with valid `url` and base64-encoded `signature`.
- [ ] Signature decodes and starts with Minisign untrusted comment header.
- [ ] No unwanted architectures (ARM, macOS, Linux) or debug builds are referenced.

---

## 4. End-to-End Installation & Upgrade Testing

Perform these manual tests on a Windows 11 x64 test machine:

### A. Clean Installation
- [ ] Run `Local.LLM.Panel_<version>_x64-setup.exe` on a machine without Local LLM Panel installed.
- [ ] Verify the installer completes cleanly and installs to `%LOCALAPPDATA%\Programs\Local LLM Panel` without admin elevation prompts (per-user install).
- [ ] Launch the app and verify the version displayed in the title bar or settings matches `<version>`.

### B. Upgrade from Previous Release (N-1 to N)
- [ ] Install previous production version (e.g., v2.2.2).
- [ ] Trigger an update check or stage `latest.json` pointing to version N.
- [ ] Verify the update-available banner appears in the UI.
- [ ] Click "Update Now" and verify the download progress bar, followed by the "Verifying cryptographic signature..." phase.
- [ ] Confirm the app prompts to restart, closes active background servers gracefully, runs the NSIS installer in passive mode, and relaunches into version N.
- [ ] Verify that user settings, model catalog, and API keys are completely preserved across the upgrade.

### C. Tamper Rejection Test (Signature Verification)
- [ ] Modify a byte inside `Local.LLM.Panel_<version>_x64-setup.nsis.zip` while keeping the original `.sig` intact.
- [ ] Trigger an update pointing to the tampered bundle.
- [ ] Verify that Tauri rejects the update during signature verification.
- [ ] Confirm the UI displays "Signature Verification Failed" with guidance to download directly from GitHub.
- [ ] Verify the currently running app remains fully functional and undamaged.

### D. Downgrade Rejection Test
- [ ] Attempt to trigger an update where `latest.json` specifies a lower version number (e.g., v1.0.0).
- [ ] Verify that Tauri ignores or rejects the downgrade and reports that the app is already up to date.

### E. Rollback / Interrupted Update Recovery
- [ ] Simulate network interruption during download (e.g., disconnect network or kill the process mid-download).
- [ ] Verify that restarting the application succeeds without corruption and leaves the previous version intact and operational.

---

## 5. Release Publication & Attestation

- [ ] Tag the release: `git tag -a "v<version>" -m "Release v<version>"` and push to `origin`.
- [ ] Ensure GitHub Actions draft release workflow completes:
  - Artifact attestation generated via `actions/attest-build-provenance`.
  - All 8 artifacts attached to the GitHub Release.
- [ ] Review release notes and publish release from draft to public.

