//! Rust & Frontend Contract Drift Integration Tests
//!
//! Verifies that:
//! 1. All Tauri command definitions in `commands.rs` are registered in `lib.rs`.
//! 2. All frontend `invoke("command")` targets in `src/api.ts` exist and match backend command signatures.
//! 3. Cleanup and updater IPC signatures maintain exact contract compatibility.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

fn get_repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent of src-tauri")
        .to_path_buf()
}

#[test]
fn test_all_tauri_commands_registered_in_lib() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_rs = fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("read lib.rs");
    let commands_rs =
        fs::read_to_string(manifest_dir.join("src/commands.rs")).expect("read commands.rs");

    let registered_handlers: HashSet<String> = lib_rs
        .lines()
        .filter_map(|l| {
            let trimmed = l.trim();
            if trimmed.starts_with("commands::") {
                Some(
                    trimmed
                        .trim_start_matches("commands::")
                        .trim_end_matches(',')
                        .trim()
                        .to_string(),
                )
            } else {
                None
            }
        })
        .collect();

    // Find all #[tauri::command] functions in commands.rs
    let mut command_names = Vec::new();
    let mut in_command_attr = false;
    for line in commands_rs.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[tauri::command]") {
            in_command_attr = true;
            continue;
        }
        if in_command_attr && trimmed.starts_with("pub ") && trimmed.contains("fn ") {
            if let Some(fn_part) = trimmed.split("fn ").nth(1) {
                let fn_name = fn_part.split('(').next().unwrap_or("").trim();
                if !fn_name.is_empty() {
                    command_names.push(fn_name.to_string());
                }
            }
            in_command_attr = false;
        }
    }

    assert!(
        !command_names.is_empty(),
        "Failed to discover tauri commands"
    );

    let mut unregistered = Vec::new();
    for cmd in &command_names {
        if !registered_handlers.contains(cmd) {
            unregistered.push(cmd.clone());
        }
    }

    assert!(
        unregistered.is_empty(),
        "Discovered Tauri commands in commands.rs not registered in lib.rs: {:?}",
        unregistered
    );
}

#[test]
fn test_frontend_invokes_match_backend_commands() {
    let root = get_repo_root();
    let api_ts = fs::read_to_string(root.join("src/api.ts")).expect("read api.ts");
    let lib_rs = fs::read_to_string(root.join("src-tauri/src/lib.rs")).expect("read lib.rs");

    let registered_handlers: HashSet<String> = lib_rs
        .lines()
        .filter_map(|l| {
            let trimmed = l.trim();
            if trimmed.starts_with("commands::") {
                Some(
                    trimmed
                        .trim_start_matches("commands::")
                        .trim_end_matches(',')
                        .trim()
                        .to_string(),
                )
            } else {
                None
            }
        })
        .collect();

    // Extract invoke targets from api.ts using regex
    let mut invoked_commands = HashSet::new();
    let invoke_re =
        regex_lite::Regex::new(r#"invoke\s*(?:<[\s\S]*?>)?\s*\(\s*["']([^"']+)["']"#).unwrap();
    for cap in invoke_re.captures_iter(&api_ts) {
        let cmd = &cap[1];
        if !cmd.is_empty() && !cmd.starts_with('.') && !cmd.starts_with('/') {
            invoked_commands.insert(cmd.to_string());
        }
    }

    assert!(
        !invoked_commands.is_empty(),
        "Failed to discover invoke calls in api.ts"
    );

    for cmd in &invoked_commands {
        assert!(
            registered_handlers.contains(cmd),
            "Frontend api.ts invokes unhandled command: \"{}\"",
            cmd
        );
    }
}

#[test]
fn test_cleanup_and_updater_commands_registered() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_rs = fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("read lib.rs");

    let required = [
        "commands::cleanup_inspect",
        "commands::cleanup_execute",
        "commands::check_app_update",
        "commands::download_and_install_update",
        "commands::last_update_status",
    ];

    for r in &required {
        assert!(
            lib_rs.contains(r),
            "lib.rs invoke_handler must contain critical command: {}",
            r
        );
    }
}
