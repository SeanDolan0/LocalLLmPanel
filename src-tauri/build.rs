fn main() {
    // Embed the common-controls v6 manifest into EVERY binary this package
    // produces (app exe, integration tests, unit-test exes). On Windows 11
    // 25H2 the System32 comctl32.dll is still v5.82 (does not export
    // TaskDialogIndirect); tauri's `common-controls-v6` feature links a raw
    // import of it, so without the manifest the loader fails with
    // 0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND. Tauri's own manifest covers the
    // app exe; this covers `cargo test` binaries. /MANIFESTINPUT merges with
    // the tauri-winres manifest, so there is no conflict.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources/common-controls-v6.manifest");
    println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
    println!("cargo:rustc-link-arg=/MANIFESTINPUT:{}", manifest.display());
    tauri_build::build();
}
