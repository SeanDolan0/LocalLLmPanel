# Deployment & Build Standardization Guide

This guide establishes the standard operating procedure (SOP) for building, packaging, and deploying **Local LLM Panel** across different AI agents (Copilot, OpenCode, Claude Code, Cursor, Antigravity, Aider, Windsurf) and human contributors.

---

## 1. Quick Reference Commands

| Goal | Command | Expected Time | Output Artifacts |
|---|---|---|---|
| **Preflight check** | `npm run check` | ~3–5s | Type checks & cargo verification |
| **Fast dev binary** | `npm run build:fast` | ~15–25s | `dist-release/LocalLLMPanel-vX.X.X.exe` |
| **Standard signed release** | `npm run build:setup` *(or `npm run release`)* | ~35–45s | Standalone EXE, NSIS installer, `.nsis.zip` updater bundle, `.sig` signature, `latest.json` manifest, `.sha256` files |
| **Full distribution** | `npm run build:all` | ~2–3m | Standalone + NSIS installer + WiX MSI package + updater artifacts |
| **Contract tests** | `npm run test:contract` | ~2–5s | Release schema, manifest contract, and IPC mock tests |
| **Release validation** | `npm run release:check` | ~3–5s | Validates all staged release artifacts, signatures, and `latest.json` |
| **Unit tests** | `cd src-tauri && cargo test` | ~5–10s | Rust unit tests across workspace |

---

## 2. Architecture of the Signed Auto-Update System

Local LLM Panel uses the official **Tauri 2 signed updater plugin** (`tauri-plugin-updater` and `@tauri-apps/plugin-updater`).

### Security Architecture & Trust Root
1. **Minisign Cryptographic Verification**: Updates are signed with an ed25519/minisign private key. The public key is embedded in `src-tauri/tauri.conf.json` under `plugins.updater.pubkey`.
2. **Signature Verification**: Tauri's updater core downloads `latest.json`, fetches the `.nsis.zip` bundle, and cryptographically validates the minisign signature BEFORE any extraction or execution. Tampered or corrupted bundles are rejected immediately.
3. **SHA-256 Checksums**: Produced alongside release files as supplemental integrity checks for human users and package managers, but **not** used as the security trust root.
4. **Authenticode (Optional)**: If `WINDOWS_CERTIFICATE_BASE64` and `WINDOWS_CERTIFICATE_PASSWORD` are configured in CI secrets, binaries and installers are signed and timestamped via `signtool.exe` (or Azure Trusted Signing).
5. **No Telemetry**: The update check only queries the static release manifest endpoint over HTTPS; no telemetry or tracking data is sent.

### Artifact Contract (8 Release Files)
Every production release publishes 8 artifacts to GitHub Releases:
1. `LocalLLMPanel-v<version>.exe` (Standalone portable executable)
2. `LocalLLMPanel-v<version>.exe.sha256`
3. `Local.LLM.Panel_<version>_x64-setup.exe` (NSIS installer)
4. `Local.LLM.Panel_<version>_x64-setup.exe.sha256`
5. `Local.LLM.Panel_<version>_x64-setup.nsis.zip` (Tauri updater zip bundle containing installer)
6. `Local.LLM.Panel_<version>_x64-setup.nsis.zip.sha256`
7. `Local.LLM.Panel_<version>_x64-setup.nsis.zip.sig` (Minisign detached signature for updater)
8. `latest.json` (Tauri updater manifest with platform target `windows-x86_64`)

---

## 3. Key Management & Migration Safety

### Generating the Tauri Signing Keypair
To generate a new keypair using the Tauri CLI:
```bash
npx @tauri-apps/cli signer generate -w ~/.tauri/llm_panel.key
```
You will be prompted for an optional password.
- **Public Key**: Placed in `src-tauri/tauri.conf.json` under `plugins.updater.pubkey`:
  ```json
  "plugins": {
    "updater": {
      "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEZDRDE3QzBGODFFNUUzOUYKUldTZjQrV0JEM3pSL05MeWIwb29sRnU5WkoxeStKcTh4N0F6T0lReVVaNHM0dG01cXZtYTA2V1kK",
      "endpoints": [
        "https://github.com/SeanDolan0/LocalLLmPanel/releases/latest/download/latest.json"
      ],
      "windows": {
        "installMode": "quiet"
      }
    }
  }
  ```
- **Private Key**: NEVER committed to Git. Stored securely in GitHub Actions Secrets as `TAURI_SIGNING_PRIVATE_KEY` with password in `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.

> [!CAUTION]
> **Key Retention & Breaking Changes**:
> Losing or replacing the private key without a planned migration will break auto-updates for all existing installed users, because their installed binary will reject releases signed by a different key.
> If key rotation is ever required, a bridge release must be published with the new public key embedded, giving users a window to update before the private key is retired.

### Backward Compatibility & Migration from v2.2.2 and Older
- Older installations (v2.2.2 and earlier) used a custom updater that parsed GitHub Releases for `Local.LLM.Panel_<version>_x64-setup.exe` and checked `.sha256`.
- By continuing to publish `Local.LLM.Panel_<version>_x64-setup.exe` and `Local.LLM.Panel_<version>_x64-setup.exe.sha256` in every release, existing v2.2.2 installations can successfully update to the new version.
- Once updated, installations transition to the official signed updater plugin (`latest.json` + minisign). Unsigned fallback installation is strictly forbidden.

---

## 4. Required GitHub Secrets for CI/CD

Configure these secrets in GitHub Repository Settings (`Settings` -> `Secrets and variables` -> `Actions`):

| Secret Name | Required | Description |
|---|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | **Yes** | Minisign private key string (single-line or escaped multiline) |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | **Yes** | Password used when generating the key (empty string if passwordless) |
| `WINDOWS_CERTIFICATE_BASE64` | Optional | Base64-encoded PFX code signing certificate for Authenticode |
| `WINDOWS_CERTIFICATE_PASSWORD` | Optional | Password for the PFX certificate |

---

## 5. The 4-File Version Synchronization Rule

When releasing a new version (e.g. bumping from `2.2.2` to `2.3.0`), **all AI agents must update the version in exactly 4 files**:

1. **`package.json`**:
   ```json
   "version": "2.3.0"
   ```
2. **`package-lock.json`**:
   ```json
   "version": "2.3.0",
   "packages": {
     "": {
       "version": "2.3.0"
     }
   }
   ```
3. **`src-tauri/Cargo.toml`**:
   ```toml
   [package]
   name = "llm-panel"
   version = "2.3.0"
   ```
4. **`src-tauri/tauri.conf.json`**:
   ```json
   "version": "2.3.0"
   ```

> [!TIP]
> `scripts/build.mjs` reads `package.json` and emits a warning if `Cargo.toml` or `tauri.conf.json` do not match. Always ensure all 4 are in sync before building.

---

## 6. Standard Agent Release Workflow

Follow these steps in sequence when preparing and deploying a release:

### Step 1: Pre-flight Verification
Run tests and contract checks:
```bash
npm run check
npm run test:contract
cd src-tauri && cargo test --workspace --locked
```

### Step 2: Version Bump
Synchronize the version string across `package.json`, `package-lock.json`, `src-tauri/Cargo.toml`, and `src-tauri/tauri.conf.json`.

### Step 3: Run the Build
For standard releases:
```bash
npm run build:setup
```
If signing locally for validation, ensure `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` are set in the environment.

### Step 4: Verify Staged Artifacts
Run the deterministic release verification script:
```bash
node scripts/validate-release.mjs --setup --artifacts
```
This validates:
- Presence of all 8 required release files in `dist-release/`.
- Format and contents of `latest.json` (correct version, valid signature string, download URL).
- Cryptographic structure of `.sig` files.
- Validity of `.sha256` checksums.

### Step 5: Git Commit & Tag
```bash
git add package.json package-lock.json src-tauri/Cargo.toml src-tauri/tauri.conf.json src-tauri/Cargo.lock
git commit -m "release: v<version>"
git tag -a "v<version>" -m "Release v<version>"
```

### Step 6: GitHub Deployment
Push code and tags to trigger the GitHub Actions release workflow:
```bash
git push origin main
git push origin "v<version>"
```
The `.github/workflows/draft-release.yml` workflow will:
1. Validate tag and version consistency across all 4 manifest files.
2. Build the Tauri frontend and Rust binary.
3. Sign the NSIS updater bundle using `TAURI_SIGNING_PRIVATE_KEY`.
4. Sign executables if Authenticode credentials exist.
5. Generate `latest.json` and SHA-256 checksums.
6. Verify all 8 artifacts using `scripts/validate-release.mjs`.
7. Generate GitHub Artifact Attestations for supply chain security.
8. Create a draft release with all 8 assets attached.

---

## 7. Agent Troubleshooting & Known Gotchas

### 1. `lld-link.exe not found`
- **Cause**: Running raw `cargo build` in a subshell where LLVM is not in `PATH`.
- **Fix**: Run through npm (`npm run build:fast` or `npm run release`). The Node script automatically detects and injects `C:\Program Files\LLVM\bin` into PATH.

### 2. Windows 11 `ProcessPrng` / `0xc0000139` Error
- **Cause**: Windows 11 24H2+ / 25H2 removed `ProcessPrng` from `bcryptprimitives.dll`. Any binary linking against `getrandom >= 0.3` without the legacy flag will crash on launch with `STATUS_ENTRYPOINT_NOT_FOUND`.
- **Fix**: Keep `.cargo/config.toml` with:
  ```toml
  [build]
  rustflags = ["--cfg", "getrandom_backend=\"windows_legacy\""]
  ```

### 3. WSL Integration Test Gating
- **Behavior**: `cargo test` in `src-tauri` runs unit tests and skips slow WSL tests.
- **Running WSL tests**: Set `LLM_TEST_WSL=1`:
  ```powershell
  $env:LLM_TEST_WSL="1"; cargo test --lib -- --ignored --nocapture wsl_it
  ```
  *Do not run this on CI or without an active WSL2 Ubuntu distribution and NVIDIA GPU.*

