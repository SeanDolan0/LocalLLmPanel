# Release checklist

The v1.8.1 release is publishable after these gates pass from the committed
checkout:

- `npm ci` completes from the committed `package-lock.json`.
- `node scripts/validate-release.mjs --setup --artifacts` passes.
- `npm run test:contract` passes.
- `npm run build` passes.
- `cargo check --workspace --locked --manifest-path src-tauri/Cargo.toml` passes.
- `cargo test --workspace --locked --manifest-path src-tauri/Cargo.toml` passes.
- `npm run build:setup` produces a non-empty standalone executable, NSIS
  installer, and matching `.sha256` files.
- The release artifact is installed and smoke-tested on a clean Windows
  machine.

The validation scripts only inspect or build local artifacts; publishing,
tagging, and pushing remain explicit release actions.
