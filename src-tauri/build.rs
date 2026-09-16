fn main() {
    let is_release = std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false);
    if !is_release {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("resources/common-controls-v6.manifest");
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg=/MANIFESTINPUT:{}", manifest.display());
    }
    tauri_build::build();
}
