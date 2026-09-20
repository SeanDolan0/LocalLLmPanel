# Deployment & Build Standardization Guide

This guide establishes the standard operating procedure (SOP) for building, packaging, and deploying **Local LLM Panel** across different AI agents (Copilot, OpenCode, Claude Code, Cursor, Antigravity, Aider, Windsurf) and human contributors.

---

## 1. Quick Reference Commands

| Goal | Command | Expected Time | Output Artifacts |
|---|---|---|---|
| **Preflight check** | `npm run check` | ~3–5s | Type checks & cargo verification |
| **Fast dev binary** | `npm run build:fast` | ~15–25s | `dist-release/LocalLLMPanel-vX.X.X.exe` |
| **Standard release** | `npm run build:setup` *(or `npm run release`)* | ~35–45s | `dist-release/LocalLLMPanel-vX.X.X.exe`<br>`dist-release/Local.LLM.Panel_X.X.X_x64-setup.exe` |
| **Full distribution** | `npm run build:all` | ~2–3m | Standalone + NSIS installer + WiX MSI package |
| **Unit tests** | `cd src-tauri && cargo test` | ~5–10s | Rust unit tests across workspace |

---

## 2. Architecture of Build Optimizations

### Why Builds Took "Forever" Before
1. **Thin LTO Overhead**: `Cargo.toml` previously had `lto = "thin"` with `incremental = true`. Thin LTO performs cross-crate link-time optimization by passing bitcode across all ~300 crates. For Local LLM Panel, where compute is handled by external background processes (`vllm` / `llama-server.exe`), LTO provides 0% user-perceptible UI performance, but cost 3–8 minutes on every link.
2. **WiX MSI Bottleneck**: `tauri.conf.json` bundled both `["nsis", "msi"]` by default. WiX decompresses and recompiles massive XML schemas and cabinet archives, adding several minutes to every build.
3. **Linker Missing on PATH**: Subshells spawned by AI agents often lacked `C:\Program Files\LLVM\bin` in their environment `PATH`, triggering immediate failures with `error: linker lld-link.exe not found`.

### Optimizations Applied
- **Rust Release Profile**: In `src-tauri/Cargo.toml`, `profile.release` has `lto = false` and `incremental = false`. Code compiles directly into object files and links in ~10–15 seconds.
- **Linker Auto-Resolution**: `scripts/build.mjs` automatically inspects standard LLVM paths (`C:\Program Files\LLVM\bin`, VS 2022 VC Tools, etc.) and injects `lld-link.exe` into the build process environment. If absent, it gracefully falls back to `link.exe`.
- **Target Tiering**: Default bundle target is `["nsis"]`. WiX MSI is reserved for explicit full distribution builds (`npm run build:all`).
- **Automated Staging**: Built binaries and installers are automatically collected, stamped, and staged in `dist-release/`.

---

## 3. The 4-File Version Synchronization Rule

When releasing a new version (e.g. bumping from `1.2.2` to `1.2.3`), **all AI agents must update the version in exactly 4 files**:

1. **`package.json`**:
   ```json
   "version": "1.2.3"
   ```
2. **`package-lock.json`**:
   ```json
   "version": "1.2.3",
   "packages": {
     "": {
       "version": "1.2.3"
     }
   }
   ```
3. **`src-tauri/Cargo.toml`**:
   ```toml
   [package]
   name = "llm-panel"
   version = "1.2.3"
   ```
4. **`src-tauri/tauri.conf.json`**:
   ```json
   "version": "1.2.3"
   ```

> [!TIP]
> `scripts/build.mjs` reads `package.json` and emits a warning if `Cargo.toml` or `tauri.conf.json` do not match. Always ensure all 4 are in sync before building.

---

## 4. Standard Agent Release Workflow

Follow these steps in sequence when preparing and deploying a release:

### Step 1: Pre-flight Verification
Run type checks and unit tests before any build:
```bash
npm run check
cd src-tauri && cargo test
```
*Ensure there are no compilation errors or broken tests.*

### Step 2: Version Bump
Synchronize the version string across `package.json`, `package-lock.json`, `src-tauri/Cargo.toml`, and `src-tauri/tauri.conf.json`.

### Step 3: Run the Build
For standard releases:
```bash
npm run build:setup
```
*Or for a full distribution bundle with MSI:*
```bash
npm run build:all
```

### Step 4: Verify Staged Artifacts
Check the contents of `dist-release/`:
```powershell
Get-ChildItem dist-release
```
Confirm the following files exist and match the bumped version:
- `dist-release/LocalLLMPanel-v<version>.exe`
- `dist-release/Local.LLM.Panel_<version>_x64-setup.exe`
- *(Optional)* `dist-release/Local.LLM.Panel_<version>_x64_en-US.msi`

### Step 5: Git Commit & Tag
```bash
git add package.json package-lock.json src-tauri/Cargo.toml src-tauri/tauri.conf.json src-tauri/Cargo.lock dist-release/
git commit -m "release: v<version>"
git tag -a "v<version>" -m "Release v<version>"
```

### Step 6: GitHub Deployment
Push code, tags, and publish the release via GitHub CLI or Web:
```bash
git push origin main
git push origin "v<version>"
gh release create "v<version>" dist-release/LocalLLMPanel-v<version>.exe dist-release/Local.LLM.Panel_<version>_x64-setup.exe --title "v<version>" --generate-notes
```

---

## 5. Agent Troubleshooting & Known Gotchas

### 1. `lld-link.exe not found`
- **Cause**: Running raw `cargo build` in a subshell where LLVM is not in `PATH`.
- **Fix**: Run through npm (`npm run build:fast` or `npm run release`). The Node script automatically detects and injects `C:\Program Files\LLVM\bin` into PATH.
- **Manual override**: In PowerShell:
  ```powershell
  $env:PATH = "C:\Program Files\LLVM\bin;$env:PATH"
  ```

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
