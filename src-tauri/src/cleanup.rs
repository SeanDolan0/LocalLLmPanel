//! Guided Environment Cleanup & Uninstallation Module
//!
//! Provides two-phase, dry-run itemized environment cleanup:
//! 1. `inspect_cleanup`: dry-run discovery of all components, paths, and byte sizes.
//! 2. `execute_cleanup`: requires explicit, itemized confirmation of each target.
//!    NEVER deletes any component without its explicit inclusion in `confirmed_targets`.

use crate::state::AppState;
use crate::wsl;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupTarget {
    WslDistro,
    Venv,
    HfCache,
    GgufDir,
    PartialDownloads,
    SudoersRule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupItem {
    pub target: CleanupTarget,
    pub label: String,
    pub description: String,
    pub path_or_target: String,
    pub exists: bool,
    pub estimated_bytes: u64,
    pub is_wsl: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPlan {
    pub distro: String,
    pub items: Vec<CleanupItem>,
    pub total_estimated_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupItemResult {
    pub target: CleanupTarget,
    pub success: bool,
    pub message: String,
    pub freed_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupExecutionReport {
    pub results: Vec<CleanupItemResult>,
    pub total_freed_bytes: u64,
}

/// Calculate directory size in bytes on the local Windows filesystem.
pub fn calculate_dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut total: u64 = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += calculate_dir_size(&p);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Find all partial `.part` download files and calculate their total size.
pub fn find_partial_downloads(root: &Path) -> (Vec<PathBuf>, u64) {
    let mut files = Vec::new();
    let mut bytes: u64 = 0;
    if !root.exists() {
        return (files, bytes);
    }
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let (sub_files, sub_bytes) = find_partial_downloads(&p);
                files.extend(sub_files);
                bytes += sub_bytes;
            } else if p.extension().and_then(|e| e.to_str()) == Some("part")
                || p.file_name().and_then(|n| n.to_str()).map(|n| n.ends_with(".part")).unwrap_or(false)
            {
                let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
                bytes += len;
                files.push(p);
            }
        }
    }
    (files, bytes)
}

/// Inspect the current environment and produce an itemized dry-run cleanup plan.
pub fn inspect_cleanup(state: &AppState) -> CleanupPlan {
    let distro = state.resolve_distro();
    let cfg = state.config();
    let mut items = Vec::new();
    let mut total_bytes: u64 = 0;

    // 1. Partial downloads in GGUF dir
    let gguf_path = PathBuf::from(&cfg.gguf_dir);
    let (part_files, part_bytes) = find_partial_downloads(&gguf_path);
    items.push(CleanupItem {
        target: CleanupTarget::PartialDownloads,
        label: "Partial Downloads (.part)".to_string(),
        description: format!(
            "Incomplete staged model downloads ({count} file{s})",
            count = part_files.len(),
            s = if part_files.len() == 1 { "" } else { "s" }
        ),
        path_or_target: cfg.gguf_dir.clone(),
        exists: !part_files.is_empty(),
        estimated_bytes: part_bytes,
        is_wsl: false,
    });
    total_bytes += part_bytes;

    // 2. Local GGUF Directory
    let gguf_exists = gguf_path.exists();
    let gguf_bytes = if gguf_exists { calculate_dir_size(&gguf_path) } else { 0 };
    items.push(CleanupItem {
        target: CleanupTarget::GgufDir,
        label: "Local GGUF Models".to_string(),
        description: "Locally stored GGUF model files and metadata".to_string(),
        path_or_target: cfg.gguf_dir.clone(),
        exists: gguf_exists && gguf_bytes > 0,
        estimated_bytes: gguf_bytes.saturating_sub(part_bytes),
        is_wsl: false,
    });
    total_bytes += gguf_bytes.saturating_sub(part_bytes);

    // Check WSL items if distro is reachable
    let distros = wsl::installed_distros();
    let distro_exists = !distro.is_empty() && distros.iter().any(|d| d.eq_ignore_ascii_case(&distro));

    // 3. WSL Distro
    items.push(CleanupItem {
        target: CleanupTarget::WslDistro,
        label: format!("WSL2 Distribution ({distro})"),
        description: "Entire dedicated WSL2 Ubuntu environment (destroys all Linux files)".to_string(),
        path_or_target: distro.clone(),
        exists: distro_exists,
        estimated_bytes: 0, // Managed by virtual disk (vhdx)
        is_wsl: true,
    });

    // 4. vLLM Python venv inside WSL (~/llm-lp)
    let mut venv_exists = false;
    let mut venv_bytes: u64 = 0;
    if distro_exists {
        let check = wsl::run_script(&distro, "[ -d ~/llm-lp ] && du -sb ~/llm-lp 2>/dev/null | awk '{print $1}'");
        if check.ok && !check.stdout.trim().is_empty() {
            if let Ok(b) = check.stdout.trim().parse::<u64>() {
                venv_exists = true;
                venv_bytes = b;
            }
        }
    }
    items.push(CleanupItem {
        target: CleanupTarget::Venv,
        label: "vLLM Virtual Environment".to_string(),
        description: "Python 3.12 venv, uv cache, and vLLM dependencies (~/llm-lp)".to_string(),
        path_or_target: "~/llm-lp".to_string(),
        exists: venv_exists,
        estimated_bytes: venv_bytes,
        is_wsl: true,
    });
    total_bytes += venv_bytes;

    // 5. Hugging Face Cache inside WSL (~/.cache/huggingface)
    let mut hf_exists = false;
    let mut hf_bytes: u64 = 0;
    if distro_exists {
        let check = wsl::run_script(&distro, "[ -d ~/.cache/huggingface ] && du -sb ~/.cache/huggingface 2>/dev/null | awk '{print $1}'");
        if check.ok && !check.stdout.trim().is_empty() {
            if let Ok(b) = check.stdout.trim().parse::<u64>() {
                hf_exists = true;
                hf_bytes = b;
            }
        }
    }
    items.push(CleanupItem {
        target: CleanupTarget::HfCache,
        label: "Hugging Face Model Cache".to_string(),
        description: "Downloaded Hugging Face model snapshots and weights (~/.cache/huggingface)".to_string(),
        path_or_target: "~/.cache/huggingface".to_string(),
        exists: hf_exists,
        estimated_bytes: hf_bytes,
        is_wsl: true,
    });
    total_bytes += hf_bytes;

    // 6. Sudoers Rule inside WSL (/etc/sudoers.d/llm-panel-*)
    let mut sudoers_exists = false;
    if distro_exists {
        let check = wsl::run_script_root(&distro, "ls /etc/sudoers.d/llm-panel-* /etc/sudoers.d/99-llm-lp 2>/dev/null && echo found");
        if check.ok && check.stdout.contains("found") {
            sudoers_exists = true;
        }
    }
    items.push(CleanupItem {
        target: CleanupTarget::SudoersRule,
        label: "Passwordless Sudo Rule".to_string(),
        description: "Scoped sudoers permission rule created during provisioning (/etc/sudoers.d/llm-panel-*)".to_string(),
        path_or_target: "/etc/sudoers.d/llm-panel-*".to_string(),
        exists: sudoers_exists,
        estimated_bytes: 0,
        is_wsl: true,
    });

    CleanupPlan {
        distro,
        items,
        total_estimated_bytes: total_bytes,
    }
}

/// Execute cleanup for explicitly confirmed targets.
/// NEVER deletes any target that is not present in `confirmed_targets`.
pub fn execute_cleanup(
    state: &AppState,
    confirmed_targets: &[CleanupTarget],
) -> CleanupExecutionReport {
    let mut results = Vec::new();
    let mut total_freed: u64 = 0;
    let distro = state.resolve_distro();
    let cfg = state.config();

    // Guard: Empty confirmation list does nothing
    if confirmed_targets.is_empty() {
        return CleanupExecutionReport {
            results,
            total_freed_bytes: 0,
        };
    }

    // 1. Partial downloads
    if confirmed_targets.contains(&CleanupTarget::PartialDownloads) {
        let gguf_path = PathBuf::from(&cfg.gguf_dir);
        let (part_files, part_bytes) = find_partial_downloads(&gguf_path);
        let mut deleted_count = 0;
        let mut failed = false;
        for f in &part_files {
            if fs::remove_file(f).is_ok() {
                deleted_count += 1;
            } else {
                failed = true;
            }
        }
        total_freed += part_bytes;
        results.push(CleanupItemResult {
            target: CleanupTarget::PartialDownloads,
            success: !failed,
            message: format!("Removed {deleted_count} of {} partial download file(s)", part_files.len()),
            freed_bytes: part_bytes,
        });
    }

    // 2. GGUF Directory
    if confirmed_targets.contains(&CleanupTarget::GgufDir) {
        let gguf_path = PathBuf::from(&cfg.gguf_dir);
        let size = calculate_dir_size(&gguf_path);
        let mut ok = true;
        let mut msg = String::new();
        if gguf_path.exists() {
            // Remove GGUF files inside directory while preserving root if possible
            if let Ok(entries) = fs::read_dir(&gguf_path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        if let Err(e) = fs::remove_dir_all(&p) {
                            ok = false;
                            msg.push_str(&format!("Failed to remove {}: {e}; ", p.display()));
                        }
                    } else if let Err(e) = fs::remove_file(&p) {
                        ok = false;
                        msg.push_str(&format!("Failed to remove {}: {e}; ", p.display()));
                    }
                }
            }
        }
        if ok {
            msg = format!("Cleaned GGUF directory ({size} bytes freed)");
            total_freed += size;
        }
        results.push(CleanupItemResult {
            target: CleanupTarget::GgufDir,
            success: ok,
            message: msg,
            freed_bytes: if ok { size } else { 0 },
        });
    }

    // WSL targets
    let distros = wsl::installed_distros();
    let distro_exists = !distro.is_empty() && distros.iter().any(|d| d.eq_ignore_ascii_case(&distro));

    // 3. Sudoers Rule
    if confirmed_targets.contains(&CleanupTarget::SudoersRule) {
        let mut ok = true;
        let mut msg = "Sudoers rule removed".to_string();
        if distro_exists {
            let out = wsl::run_script_root(&distro, "rm -f /etc/sudoers.d/llm-panel-* /etc/sudoers.d/99-llm-lp && echo ok");
            if !out.ok || !out.stdout.contains("ok") {
                ok = false;
                msg = format!("Failed to remove sudoers rule: {}", out.combined());
            }
        } else {
            msg = "WSL distro not found; sudoers rule already absent".to_string();
        }
        results.push(CleanupItemResult {
            target: CleanupTarget::SudoersRule,
            success: ok,
            message: msg,
            freed_bytes: 0,
        });
    }

    // 4. vLLM venv
    if confirmed_targets.contains(&CleanupTarget::Venv) {
        let mut ok = true;
        let mut msg = "vLLM environment directory (~/llm-lp) removed".to_string();
        let mut size = 0;
        if distro_exists {
            let du = wsl::run_script(&distro, "du -sb ~/llm-lp 2>/dev/null | awk '{print $1}'");
            if let Ok(b) = du.stdout.trim().parse::<u64>() {
                size = b;
            }
            let out = wsl::run_script(&distro, "rm -rf ~/llm-lp && echo ok");
            if !out.ok || !out.stdout.contains("ok") {
                ok = false;
                msg = format!("Failed to remove ~/llm-lp: {}", out.combined());
            } else {
                total_freed += size;
            }
        } else {
            msg = "WSL distro not found; venv already absent".to_string();
        }
        results.push(CleanupItemResult {
            target: CleanupTarget::Venv,
            success: ok,
            message: msg,
            freed_bytes: if ok { size } else { 0 },
        });
    }

    // 5. Hugging Face Cache
    if confirmed_targets.contains(&CleanupTarget::HfCache) {
        let mut ok = true;
        let mut msg = "Hugging Face cache (~/.cache/huggingface) removed".to_string();
        let mut size = 0;
        if distro_exists {
            let du = wsl::run_script(&distro, "du -sb ~/.cache/huggingface 2>/dev/null | awk '{print $1}'");
            if let Ok(b) = du.stdout.trim().parse::<u64>() {
                size = b;
            }
            let out = wsl::run_script(&distro, "rm -rf ~/.cache/huggingface && echo ok");
            if !out.ok || !out.stdout.contains("ok") {
                ok = false;
                msg = format!("Failed to remove Hugging Face cache: {}", out.combined());
            } else {
                total_freed += size;
            }
        } else {
            msg = "WSL distro not found; cache already absent".to_string();
        }
        results.push(CleanupItemResult {
            target: CleanupTarget::HfCache,
            success: ok,
            message: msg,
            freed_bytes: if ok { size } else { 0 },
        });
    }

    // 6. WSL Distro Deregistration (must be last among WSL items)
    if confirmed_targets.contains(&CleanupTarget::WslDistro) {
        let mut ok = true;
        let mut msg = format!("WSL distribution '{distro}' unregistered successfully");
        if distro_exists {
            // First terminate any running instances
            wsl::terminate_distro(&distro);
            // Unregister distro
            let status = std::process::Command::new("wsl.exe")
                .args(["--unregister", &distro])
                .status();
            match status {
                Ok(s) if s.success() => {}
                Ok(s) => {
                    ok = false;
                    msg = format!("wsl --unregister exited with code {:?}", s.code());
                }
                Err(e) => {
                    ok = false;
                    msg = format!("Failed to execute wsl --unregister: {e}");
                }
            }
        } else {
            msg = format!("WSL distribution '{distro}' was not registered");
        }
        results.push(CleanupItemResult {
            target: CleanupTarget::WslDistro,
            success: ok,
            message: msg,
            freed_bytes: 0,
        });
    }

    CleanupExecutionReport {
        results,
        total_freed_bytes: total_freed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    #[test]
    fn test_cleanup_plan_itemization_dry_run() {
        let state = AppState::default();
        let plan = inspect_cleanup(&state);
        assert!(!plan.distro.is_empty());
        assert_eq!(plan.items.len(), 6);

        let targets: Vec<CleanupTarget> = plan.items.iter().map(|i| i.target).collect();
        assert!(targets.contains(&CleanupTarget::PartialDownloads));
        assert!(targets.contains(&CleanupTarget::GgufDir));
        assert!(targets.contains(&CleanupTarget::WslDistro));
        assert!(targets.contains(&CleanupTarget::Venv));
        assert!(targets.contains(&CleanupTarget::HfCache));
        assert!(targets.contains(&CleanupTarget::SudoersRule));
    }

    #[test]
    fn test_cleanup_execute_requires_explicit_confirmation() {
        let state = AppState::default();
        // Empty confirmation list must do zero operations
        let report = execute_cleanup(&state, &[]);
        assert_eq!(report.results.len(), 0);
        assert_eq!(report.total_freed_bytes, 0);
    }

    #[test]
    fn test_cleanup_execute_selective_partial_downloads() {
        let dir_path = std::env::temp_dir().join(format!("llp-cleanup-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir_path);
        let valid_file = dir_path.join("model-q4.gguf");
        let part_file = dir_path.join("model-q8.gguf.part");

        File::create(&valid_file).unwrap().write_all(b"valid content").unwrap();
        File::create(&part_file).unwrap().write_all(b"partial chunk").unwrap();

        let state = AppState::default();
        state.config.lock().unwrap().gguf_dir = dir_path.to_string_lossy().to_string();

        // Dry-run inspection should detect the .part file
        let plan = inspect_cleanup(&state);
        let part_item = plan.items.iter().find(|i| i.target == CleanupTarget::PartialDownloads).unwrap();
        assert!(part_item.exists);
        assert_eq!(part_item.estimated_bytes, 13); // "partial chunk" len = 13

        // Execute cleanup ONLY for PartialDownloads
        let report = execute_cleanup(&state, &[CleanupTarget::PartialDownloads]);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].target, CleanupTarget::PartialDownloads);
        assert!(report.results[0].success);

        // Verify that the .part file was deleted, but the valid .gguf file was PRESERVED
        assert!(!part_file.exists(), ".part file must be deleted");
        assert!(valid_file.exists(), "valid .gguf file must NOT be deleted");
        let _ = fs::remove_dir_all(&dir_path);
    }

    #[test]
    fn test_cleanup_target_serialization() {
        let targets = [
            CleanupTarget::WslDistro,
            CleanupTarget::Venv,
            CleanupTarget::HfCache,
            CleanupTarget::GgufDir,
            CleanupTarget::PartialDownloads,
            CleanupTarget::SudoersRule,
        ];
        for t in &targets {
            let json = serde_json::to_string(t).unwrap();
            let parsed: CleanupTarget = serde_json::from_str(&json).unwrap();
            assert_eq!(*t, parsed);
        }
    }
}
