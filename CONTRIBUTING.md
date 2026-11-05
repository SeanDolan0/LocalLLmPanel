# Contributing

Thanks for helping improve Local LLM Panel. Contributions that make local inference easier to discover, configure, operate, or troubleshoot are welcome.

## Before you start

- Search existing issues and pull requests to avoid duplicating work.
- For larger changes, open an issue or discussion first so the intended behavior can be agreed on.
- Keep changes focused. Avoid unrelated refactors in feature or bug-fix pull requests.
- Do not commit model weights, installers, build output, local configuration, tokens, or unredacted logs.

## Development environment

The app is a Windows desktop application built with Tauri 2, Rust, React, TypeScript, and Vite.

Required development tools:

- Windows 11 and WebView2
- Node.js 22+
- Stable Rust with the `x86_64-pc-windows-msvc` target
- Visual Studio C++ Build Tools
- LLVM/LLD on PATH (the npm build scripts also search standard installation locations)
- An NVIDIA GPU and WSL2 Ubuntu for exercising the vLLM path

Install dependencies and run the app:

```powershell
npm ci
npm run tauri dev
```

The native llama.cpp backend can be developed and tested without provisioning vLLM in WSL2, provided a compatible local `llama-server.exe` and GGUF are available.

## Before opening a pull request

Run the checks relevant to your change:

```powershell
npm run release:check
npm run test:contract
npm run build
cd src-tauri
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked
cargo test --workspace --locked
```

For a full Windows executable build:

```powershell
cd ..
npm run build:fast
```

Hardware-dependent WSL and llama.cpp end-to-end tests are opt-in and require local services, model files, and (for WSL/vLLM) an NVIDIA GPU. See the [README test section](README.md#hardware-dependent-integration-tests).

## Pull request expectations

A good pull request includes:

- A clear description of the user problem and the intended behavior.
- Screenshots or a short recording for visible UI changes.
- Tests run and any checks that could not be run locally.
- Backend and hardware details for inference-related changes (Windows version, GPU/VRAM, backend/version, and model/quant where relevant).
- Documentation updates when setup, configuration, commands, or behavior changes.

Avoid posting secrets. Redact Hugging Face/GitHub tokens, private paths, and personal data from logs and screenshots.

## Code and design conventions

- Preserve the distinction between estimated and measured performance.
- Keep network, filesystem, and process operations recoverable and report actionable errors.
- Keep localhost as the default bind address; LAN exposure must remain explicit.
- Prefer capability detection over assuming a particular vLLM or llama.cpp release supports a command-line flag.
- Keep dependencies focused and explain why new runtime dependencies are needed.
- Add or update Rust unit tests for deterministic backend logic and frontend contract tests for Tauri API surface changes.

## Version changes

Do not bump the application version for ordinary feature pull requests unless the change is explicitly preparing a release. When preparing a release, keep these files synchronized:

1. `package.json`
2. `package-lock.json` (top-level and root package version)
3. `src-tauri/Cargo.toml`
4. `src-tauri/tauri.conf.json`

The release validation script checks this metadata.
