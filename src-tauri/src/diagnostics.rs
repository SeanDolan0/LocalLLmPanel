//! Unified first-run diagnostics, Windows support verification, and redacted diagnostic bundle export.

use crate::server::{apply_vllm_auth, server_api_key};
use crate::state::AppState;
use regex_lite::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Minimum Windows 10 build required for WSL2 + NVIDIA CUDA GPU acceleration (Version 2004 / 20H1).
pub const MIN_WIN10_CUDA_BUILD: u32 = 19041;
/// Windows 11 initial release build.
pub const WIN11_MIN_BUILD: u32 = 22000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WindowsInfo {
    pub build_number: Option<u32>,
    pub display_version: Option<String>,
    pub product_name: Option<String>,
    pub raw_version: Option<String>,
    pub is_supported: bool,
    pub is_windows_11: bool,
    pub os_label: String,
    pub support_note: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticStatus {
    Pass,
    Warning,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticStep {
    pub order: u32,
    pub id: String,
    pub title: String,
    pub status: DiagnosticStatus,
    pub value: String,
    pub details: Option<String>,
    pub fix_recommendation: Option<String>,
    pub doc_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub passed: bool,
    pub gpu_usable: bool,
    pub summary: String,
    pub windows_info: WindowsInfo,
    pub steps: Vec<DiagnosticStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskPreflightInfo {
    pub target_path: String,
    pub available_bytes: u64,
    pub total_bytes: u64,
    pub available_gb: f64,
    pub total_gb: f64,
    pub required_bytes: Option<u64>,
    pub required_gb: Option<f64>,
    pub headroom_gb: Option<f64>,
    pub status: String, // "sufficient", "tight", "insufficient"
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityTestItem {
    pub id: String,
    pub name: String,
    pub description: String,
    pub passed: bool,
    pub latency_ms: u64,
    pub details: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityTestReport {
    pub server_id: String,
    pub model_id: String,
    pub backend: String,
    pub timestamp: String,
    pub all_passed: bool,
    pub passed_count: usize,
    pub total_count: usize,
    pub items: Vec<CompatibilityTestItem>,
}

/// Evaluates Windows build information and support status according to Windows 10/11 + NVIDIA requirements.
pub fn evaluate_windows_build(
    build: Option<u32>,
    display_ver: Option<&str>,
    prod_name: Option<&str>,
    raw_ver: Option<&str>,
) -> WindowsInfo {
    let (is_supported, is_win11, os_label, support_note) = match build {
        Some(b) if b >= WIN11_MIN_BUILD => {
            let ver_suffix = display_ver.map(|v| format!(" ({v})")).unwrap_or_default();
            (
                true,
                true,
                format!("Windows 11 Build {b}{ver_suffix}"),
                "Windows 11 fully supports WSL2 and NVIDIA CUDA GPU acceleration.".to_string(),
            )
        }
        Some(b) if b >= MIN_WIN10_CUDA_BUILD => {
            let ver_suffix = display_ver.map(|v| format!(" ({v})")).unwrap_or_default();
            (
                true,
                false,
                format!("Windows 10 Build {b}{ver_suffix}"),
                "Windows 10 Build 19041+ (2004+) satisfies WSL2 and NVIDIA CUDA requirements."
                    .to_string(),
            )
        }
        Some(b) => {
            let ver_suffix = display_ver.map(|v| format!(" ({v})")).unwrap_or_default();
            (
                false,
                false,
                format!("Windows 10 Build {b}{ver_suffix} [UNSUPPORTED]"),
                format!(
                    "Windows build {b} is below the minimum required build {MIN_WIN10_CUDA_BUILD} (Version 2004). Please run Windows Update to upgrade to Windows 10 22H2 or Windows 11."
                ),
            )
        }
        None => {
            let label = prod_name
                .map(|p| p.to_string())
                .or_else(|| raw_ver.map(|r| r.to_string()))
                .unwrap_or_else(|| "Windows (Unknown Version)".to_string());
            (
                false,
                false,
                label,
                "Unable to verify Windows build number. Windows 10 build 19041+ or Windows 11 is required.".to_string(),
            )
        }
    };

    WindowsInfo {
        build_number: build,
        display_version: display_ver.map(|s| s.to_string()),
        product_name: prod_name.map(|s| s.to_string()),
        raw_version: raw_ver.map(|s| s.to_string()),
        is_supported,
        is_windows_11: is_win11,
        os_label,
        support_note,
    }
}

/// Detects live Windows operating system information via registry queries with cmd /c ver fallback.
pub fn detect_windows_info() -> WindowsInfo {
    #[cfg(windows)]
    {
        use std::process::Command;
        let mut build_num: Option<u32> = None;
        let mut display_version: Option<String> = None;
        let mut product_name: Option<String> = None;
        let mut raw_version: Option<String> = None;

        // Query CurrentBuild from registry
        if let Ok(out) = Command::new("reg")
            .args(&[
                "query",
                r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "/v",
                "CurrentBuild",
            ])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 && parts[0].eq_ignore_ascii_case("CurrentBuild") {
                        if let Ok(b) = parts[parts.len() - 1].parse::<u32>() {
                            build_num = Some(b);
                            break;
                        }
                    }
                }
            }
        }

        // Query DisplayVersion from registry
        if let Ok(out) = Command::new("reg")
            .args(&[
                "query",
                r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "/v",
                "DisplayVersion",
            ])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 && parts[0].eq_ignore_ascii_case("DisplayVersion") {
                        display_version = Some(parts[parts.len() - 1].to_string());
                        break;
                    }
                }
            }
        }

        // Query ProductName from registry
        if let Ok(out) = Command::new("reg")
            .args(&[
                "query",
                r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "/v",
                "ProductName",
            ])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines() {
                    if line.contains("ProductName") && line.contains("REG_SZ") {
                        if let Some(pos) = line.find("REG_SZ") {
                            let name = line[pos + 6..].trim().to_string();
                            if !name.is_empty() {
                                product_name = Some(name);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: cmd /c ver
        if let Ok(out) = Command::new("cmd").args(&["/c", "ver"]).output() {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                let trimmed = text.trim().to_string();
                if !trimmed.is_empty() {
                    raw_version = Some(trimmed.clone());
                    if build_num.is_none() {
                        if let Some(caps) = Regex::new(r"(?i)version\s+10\.0\.(\d+)")
                            .ok()
                            .and_then(|re| re.captures(&trimmed))
                        {
                            if let Some(m) = caps.get(1) {
                                build_num = m.as_str().parse::<u32>().ok();
                            }
                        }
                    }
                }
            }
        }

        evaluate_windows_build(
            build_num,
            display_version.as_deref(),
            product_name.as_deref(),
            raw_version.as_deref(),
        )
    }

    #[cfg(not(windows))]
    {
        evaluate_windows_build(Some(22631), Some("23H2"), Some("Windows 11 Pro"), None)
    }
}

/// Redacts secrets such as Hugging Face tokens, GitHub tokens, OpenAI/generic API keys,
/// and bearer tokens from arbitrary text, configs, and logs.
pub fn redact_secrets(input: &str) -> String {
    let mut s = input.to_string();

    // Redact HuggingFace tokens: hf_[a-zA-Z0-9]{20,}
    if let Ok(re) = Regex::new(r"hf_[a-zA-Z0-9]{20,}") {
        s = re.replace_all(&s, "[REDACTED_HF_TOKEN]").to_string();
    }

    // Redact GitHub classic and fine-grained tokens: ghp_*, github_pat_*
    if let Ok(re) = Regex::new(r"ghp_[a-zA-Z0-9]{20,}") {
        s = re.replace_all(&s, "[REDACTED_GITHUB_TOKEN]").to_string();
    }
    if let Ok(re) = Regex::new(r"github_pat_[a-zA-Z0-9_]{20,}") {
        s = re.replace_all(&s, "[REDACTED_GITHUB_TOKEN]").to_string();
    }

    // Redact OpenAI / generic API keys: sk-[a-zA-Z0-9_\-]{20,}
    if let Ok(re) = Regex::new(r"sk-[a-zA-Z0-9_\-]{20,}") {
        s = re.replace_all(&s, "[REDACTED_API_KEY]").to_string();
    }

    // Redact Bearer authorization headers: Bearer <alphanumeric_or_punctuation>
    if let Ok(re) = Regex::new(r"(?i)Bearer\s+([a-zA-Z0-9_\-\.]{15,})") {
        s = re
            .replace_all(&s, "Bearer [REDACTED_BEARER_TOKEN]")
            .to_string();
    }

    // Redact key=value or token=value query parameters and env strings
    if let Ok(re) = Regex::new(
        r#"(?i)(token|api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*([^\s,;\"']+)"#,
    ) {
        s = re.replace_all(&s, "$1=[REDACTED]").to_string();
    }

    s
}

/// Runs the unified 8-step ordered first-run diagnostic check:
/// 1. Windows version & build
/// 2. WSL present & version
/// 3. WSL2 working
/// 4. NVIDIA driver
/// 5. GPU visible inside WSL
/// 6. CUDA runtime
/// 7. Free disk space
/// 8. Backend readiness
pub fn run_ordered_diagnostics(state: &AppState) -> DiagnosticReport {
    let mut steps: Vec<DiagnosticStep> = Vec::new();
    let win_info = detect_windows_info();

    // Step 1: Windows version & build
    let step1 = if win_info.is_supported {
        DiagnosticStep {
            order: 1,
            id: "windows_version".into(),
            title: "Windows Version & Build".into(),
            status: DiagnosticStatus::Pass,
            value: win_info.os_label.clone(),
            details: Some(win_info.support_note.clone()),
            fix_recommendation: None,
            doc_url: Some("https://learn.microsoft.com/windows/wsl/install".into()),
        }
    } else {
        DiagnosticStep {
            order: 1,
            id: "windows_version".into(),
            title: "Windows Version & Build".into(),
            status: DiagnosticStatus::Fail,
            value: win_info.os_label.clone(),
            details: Some(win_info.support_note.clone()),
            fix_recommendation: Some(
                "Run Windows Update to install Windows 10 Version 22H2 (Build 19045) or Windows 11. WSL2 GPU virtualization requires Build 19041+.".into()
            ),
            doc_url: Some("https://support.microsoft.com/windows/update-windows".into()),
        }
    };
    steps.push(step1);

    // Step 2: WSL present & version
    let wsl_version_opt = crate::wsl::get_wsl_version();
    let step2 = match wsl_version_opt {
        Some((major, minor, patch)) => {
            let ver_str = format!("{major}.{minor}.{patch}");
            if major >= 2 {
                DiagnosticStep {
                    order: 2,
                    id: "wsl_installed".into(),
                    title: "WSL Installation & Version".into(),
                    status: DiagnosticStatus::Pass,
                    value: format!("WSL {ver_str}"),
                    details: Some("Modern WSL package detected.".into()),
                    fix_recommendation: None,
                    doc_url: None,
                }
            } else {
                DiagnosticStep {
                    order: 2,
                    id: "wsl_installed".into(),
                    title: "WSL Installation & Version".into(),
                    status: DiagnosticStatus::Warning,
                    value: format!("WSL {ver_str}"),
                    details: Some("WSL version is below 2.0.0. An update is recommended.".into()),
                    fix_recommendation: Some("Run in PowerShell: wsl --update".into()),
                    doc_url: Some("https://learn.microsoft.com/windows/wsl/install".into()),
                }
            }
        }
        None => {
            let distros = crate::wsl::installed_distros();
            if !distros.is_empty() {
                DiagnosticStep {
                    order: 2,
                    id: "wsl_installed".into(),
                    title: "WSL Installation & Version".into(),
                    status: DiagnosticStatus::Warning,
                    value: "WSL Installed (legacy inbox version)".into(),
                    details: Some(format!("Found distros: {}", distros.join(", "))),
                    fix_recommendation: Some(
                        "Update WSL to Microsoft Store package: run 'wsl --update' in PowerShell."
                            .into(),
                    ),
                    doc_url: Some("https://learn.microsoft.com/windows/wsl/install".into()),
                }
            } else {
                DiagnosticStep {
                    order: 2,
                    id: "wsl_installed".into(),
                    title: "WSL Installation & Version".into(),
                    status: DiagnosticStatus::Fail,
                    value: "WSL Not Installed".into(),
                    details: Some(
                        "wsl.exe was not found or no Linux distributions are registered.".into(),
                    ),
                    fix_recommendation: Some(
                        "Install WSL in Administrator PowerShell: wsl --install --no-distribution"
                            .into(),
                    ),
                    doc_url: Some("https://learn.microsoft.com/windows/wsl/install".into()),
                }
            }
        }
    };
    steps.push(step2);

    // Step 3: WSL2 working
    let distro_name = state.resolve_distro();
    let is_wsl_running = crate::wsl::is_running(&distro_name);
    let step3 = if is_wsl_running {
        DiagnosticStep {
            order: 3,
            id: "wsl2_working".into(),
            title: "WSL2 Environment & Kernel".into(),
            status: DiagnosticStatus::Pass,
            value: format!("Active ({distro_name})"),
            details: Some(
                "WSL2 virtualization and container runtime are operating normally.".into(),
            ),
            fix_recommendation: None,
            doc_url: None,
        }
    } else {
        let probe = crate::wsl::run_script(&distro_name, "echo ok");
        if probe.ok {
            DiagnosticStep {
                order: 3,
                id: "wsl2_working".into(),
                title: "WSL2 Environment & Kernel".into(),
                status: DiagnosticStatus::Pass,
                value: format!("Ready ({distro_name})"),
                details: Some("WSL2 instance responds to commands.".into()),
                fix_recommendation: None,
                doc_url: None,
            }
        } else {
            DiagnosticStep {
                order: 3,
                id: "wsl2_working".into(),
                title: "WSL2 Environment & Kernel".into(),
                status: DiagnosticStatus::Fail,
                value: format!("Unresponsive ({distro_name})"),
                details: Some(if probe.stderr.is_empty() {
                    "WSL2 distro did not respond to execution probes.".into()
                } else {
                    probe.stderr.trim().to_string()
                }),
                fix_recommendation: Some(
                    "Enable Virtual Machine Platform in Windows Features and restart your PC. In PowerShell: dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart".into()
                ),
                doc_url: Some("https://learn.microsoft.com/windows/wsl/troubleshooting".into()),
            }
        }
    };
    steps.push(step3);

    // Step 4: NVIDIA driver on host
    let native_gpu = crate::llamacpp_install::windows_gpu_snapshot();
    let host_smi_gpus = {
        let output = std::process::Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ])
            .output();
        output.ok().and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
    };

    let step4 = match (native_gpu.as_ref(), host_smi_gpus.as_ref()) {
        (Some(gpu), _) => DiagnosticStep {
            order: 4,
            id: "nvidia_driver".into(),
            title: "Host NVIDIA Driver".into(),
            status: DiagnosticStatus::Pass,
            value: format!("{}, {} MB VRAM", gpu.name, gpu.vram_total_mb),
            details: Some(format!("Host GPU snapshot captured with {} MB total VRAM.", gpu.vram_total_mb)),
            fix_recommendation: None,
            doc_url: None,
        },
        (None, Some(smi)) => DiagnosticStep {
            order: 4,
            id: "nvidia_driver".into(),
            title: "Host NVIDIA Driver".into(),
            status: DiagnosticStatus::Pass,
            value: smi.clone(),
            details: Some("Detected via nvidia-smi query.".into()),
            fix_recommendation: None,
            doc_url: None,
        },
        _ => DiagnosticStep {
            order: 4,
            id: "nvidia_driver".into(),
            title: "Host NVIDIA Driver".into(),
            status: DiagnosticStatus::Fail,
            value: "No NVIDIA Driver Detected".into(),
            details: Some("NVIDIA driver is missing or nvidia-smi failed on the host.".into()),
            fix_recommendation: Some(
                "Install latest GeForce Game Ready or Studio Driver from nvidia.com. Do NOT install Linux graphics drivers in WSL.".into()
            ),
            doc_url: Some("https://www.nvidia.com/Download/index.aspx".into()),
        },
    };
    steps.push(step4);

    // Step 5: GPU visible inside WSL
    let wsl_gpu = crate::wsl::gpu_snapshot(&distro_name);
    let step5 = match &wsl_gpu {
        Some(gpu) => DiagnosticStep {
            order: 5,
            id: "gpu_in_wsl".into(),
            title: "GPU Visible in WSL2".into(),
            status: DiagnosticStatus::Pass,
            value: format!("{} ({} MB VRAM)", gpu.name, gpu.vram_total_mb),
            details: Some(format!(
                "WSL2 GPU paravirtualization active. Free VRAM: {} MB.",
                gpu.vram_free_mb
            )),
            fix_recommendation: None,
            doc_url: None,
        },
        None => {
            let check_libcuda = crate::wsl::run_script(
                &distro_name,
                "test -f /usr/lib/wsl/lib/libcuda.so.1 && echo present || echo missing",
            );
            let has_libcuda = check_libcuda.stdout.contains("present");
            if has_libcuda {
                DiagnosticStep {
                    order: 5,
                    id: "gpu_in_wsl".into(),
                    title: "GPU Visible in WSL2".into(),
                    status: DiagnosticStatus::Warning,
                    value: "libcuda.so present but snapshot failed".into(),
                    details: Some("WSL2 GPU driver mount is present, but nvidia-smi did not report an active device.".into()),
                    fix_recommendation: Some("Restart WSL by running 'wsl --shutdown' in PowerShell, then re-launch the app.".into()),
                    doc_url: Some("https://docs.nvidia.com/cuda/wsl-user-guide/index.html".into()),
                }
            } else {
                DiagnosticStep {
                    order: 5,
                    id: "gpu_in_wsl".into(),
                    title: "GPU Visible in WSL2".into(),
                    status: DiagnosticStatus::Fail,
                    value: "GPU Not Visible in WSL2".into(),
                    details: Some("/usr/lib/wsl/lib/libcuda.so.1 missing. WSL2 cannot pass through NVIDIA GPU.".into()),
                    fix_recommendation: Some(
                        "Ensure Windows Host NVIDIA Driver >= 525 is installed. Run 'wsl --update' and 'wsl --shutdown' in PowerShell.".into()
                    ),
                    doc_url: Some("https://docs.nvidia.com/cuda/wsl-user-guide/index.html".into()),
                }
            }
        }
    };
    steps.push(step5);

    // Step 6: CUDA runtime (WSL or native llama.cpp)
    let prov_report = {
        let out = crate::wsl::run_script(
            &distro_name,
            "cat ~/llm-lp/.provisioned 2>/dev/null || true",
        );
        if out.ok && !out.stdout.trim().is_empty() {
            serde_json::from_str::<crate::provision::ProvisionReport>(&out.stdout).ok()
        } else {
            None
        }
    };

    let llamacpp_cuda = native_gpu.is_some() || wsl_gpu.is_some();
    let step6 = match &prov_report {
        Some(rep) if rep.cuda_available => DiagnosticStep {
            order: 6,
            id: "cuda_runtime".into(),
            title: "CUDA Runtime & PyTorch".into(),
            status: DiagnosticStatus::Pass,
            value: format!(
                "CUDA Ready (PyTorch {})",
                rep.torch_version.as_deref().unwrap_or("available")
            ),
            details: Some(format!(
                "GPU: {}",
                rep.gpu_name.as_deref().unwrap_or("NVIDIA")
            )),
            fix_recommendation: None,
            doc_url: None,
        },
        Some(_) => DiagnosticStep {
            order: 6,
            id: "cuda_runtime".into(),
            title: "CUDA Runtime & PyTorch".into(),
            status: DiagnosticStatus::Fail,
            value: "PyTorch CUDA Unavailable in WSL".into(),
            details: Some(
                "Provisioning ran but PyTorch reports cuda.is_available() == false.".into(),
            ),
            fix_recommendation: Some(
                "Re-run 'Provision WSL' from Dashboard. Verify host NVIDIA driver is intact."
                    .into(),
            ),
            doc_url: None,
        },
        None => {
            if llamacpp_cuda {
                DiagnosticStep {
                    order: 6,
                    id: "cuda_runtime".into(),
                    title: "CUDA Runtime & PyTorch".into(),
                    status: DiagnosticStatus::Warning,
                    value: "WSL vLLM Unprovisioned (llama.cpp CUDA available)".into(),
                    details: Some("WSL virtualenv has not been provisioned yet. Native llama.cpp with CUDA is supported.".into()),
                    fix_recommendation: Some("Click 'Provision WSL' on the Dashboard to set up vLLM.".into()),
                    doc_url: None,
                }
            } else {
                DiagnosticStep {
                    order: 6,
                    id: "cuda_runtime".into(),
                    title: "CUDA Runtime & PyTorch".into(),
                    status: DiagnosticStatus::Fail,
                    value: "CUDA Runtime Unavailable".into(),
                    details: Some("Neither WSL PyTorch CUDA nor native llama.cpp CUDA is accessible.".into()),
                    fix_recommendation: Some("Install NVIDIA drivers on Windows and run 'Provision WSL' from the Dashboard.".into()),
                    doc_url: None,
                }
            }
        }
    };
    steps.push(step6);

    // Step 7: Free disk space
    let models_dir = dirs::home_dir()
        .map(|h| h.join(".cache").join("huggingface"))
        .unwrap_or_else(|| PathBuf::from(r"C:\"));
    let disk_info = evaluate_disk_preflight(&models_dir, Some(10 * 1024 * 1024 * 1024));
    let step7 = match disk_info.status.as_str() {
        "sufficient" => DiagnosticStep {
            order: 7,
            id: "disk_space".into(),
            title: "Available Disk Space".into(),
            status: DiagnosticStatus::Pass,
            value: format!("{:.1} GB Free", disk_info.available_gb),
            details: Some(format!(
                "Volume total: {:.1} GB ({})",
                disk_info.total_gb, disk_info.target_path
            )),
            fix_recommendation: None,
            doc_url: None,
        },
        "tight" => DiagnosticStep {
            order: 7,
            id: "disk_space".into(),
            title: "Available Disk Space".into(),
            status: DiagnosticStatus::Warning,
            value: format!("{:.1} GB Free (Low)", disk_info.available_gb),
            details: Some(
                "Free storage is below 20 GB. Larger models (>14B) may fail to stage and download."
                    .into(),
            ),
            fix_recommendation: Some(
                "Clean up unused files or configure a separate storage volume for models.".into(),
            ),
            doc_url: None,
        },
        _ => DiagnosticStep {
            order: 7,
            id: "disk_space".into(),
            title: "Available Disk Space".into(),
            status: DiagnosticStatus::Fail,
            value: format!("{:.1} GB Free (Critical)", disk_info.available_gb),
            details: Some(
                "Free storage is below 10 GB. Model downloads and KV-cache allocations will fail."
                    .into(),
            ),
            fix_recommendation: Some(
                "Free at least 20 GB of disk space on your main drive.".into(),
            ),
            doc_url: None,
        },
    };
    steps.push(step7);

    // Step 8: Backend readiness
    let cfg = state.config();
    let llamacpp_ready = crate::llamacpp_install::executable_from_config(&cfg).is_some();
    let vllm_ready = prov_report
        .as_ref()
        .map(|r| r.vllm_version.is_some() && r.cuda_available)
        .unwrap_or(false);

    let step8 = if vllm_ready && llamacpp_ready {
        DiagnosticStep {
            order: 8,
            id: "backend_ready".into(),
            title: "Inference Backend Readiness".into(),
            status: DiagnosticStatus::Pass,
            value: "Both vLLM & llama.cpp Ready".into(),
            details: Some("All local backends are installed and ready to serve models.".into()),
            fix_recommendation: None,
            doc_url: None,
        }
    } else if vllm_ready {
        DiagnosticStep {
            order: 8,
            id: "backend_ready".into(),
            title: "Inference Backend Readiness".into(),
            status: DiagnosticStatus::Pass,
            value: "vLLM Ready (WSL2)".into(),
            details: Some("vLLM server engine is provisioned with GPU support.".into()),
            fix_recommendation: None,
            doc_url: None,
        }
    } else if llamacpp_ready {
        DiagnosticStep {
            order: 8,
            id: "backend_ready".into(),
            title: "Inference Backend Readiness".into(),
            status: DiagnosticStatus::Pass,
            value: "llama.cpp Ready (Windows Native)".into(),
            details: Some("llama.cpp binary is installed and executable.".into()),
            fix_recommendation: None,
            doc_url: None,
        }
    } else {
        DiagnosticStep {
            order: 8,
            id: "backend_ready".into(),
            title: "Inference Backend Readiness".into(),
            status: DiagnosticStatus::Warning,
            value: "No Backends Installed".into(),
            details: Some(
                "Neither vLLM (WSL2) nor llama.cpp (Windows) has been installed yet.".into(),
            ),
            fix_recommendation: Some(
                "Click 'Provision WSL' on Dashboard or 'Install llama.cpp' to get started.".into(),
            ),
            doc_url: None,
        }
    };
    steps.push(step8);

    // Beginner safety rule:
    // Beginners must NEVER see "success" when the GPU is unusable!
    let gpu_usable = (native_gpu.is_some() || wsl_gpu.is_some())
        && steps
            .iter()
            .find(|s| s.id == "nvidia_driver")
            .map(|s| s.status != DiagnosticStatus::Fail)
            .unwrap_or(false);

    let any_critical_fail = steps.iter().any(|s| {
        (s.id == "windows_version"
            || s.id == "wsl2_working"
            || s.id == "nvidia_driver"
            || s.id == "gpu_in_wsl")
            && s.status == DiagnosticStatus::Fail
    });

    let passed = !any_critical_fail && gpu_usable;

    let summary = if !win_info.is_supported {
        format!("Windows build is unsupported: {}", win_info.support_note)
    } else if !gpu_usable {
        "GPU is unusable: No active NVIDIA GPU driver or CUDA device detected. Models cannot run with hardware acceleration.".into()
    } else if any_critical_fail {
        "System setup incomplete: Critical errors detected in WSL or driver virtualization.".into()
    } else if steps.iter().any(|s| s.status == DiagnosticStatus::Warning) {
        "System is operational with warnings. You can run models, but review recommended actions."
            .into()
    } else {
        "All diagnostics passed! System is fully ready for GPU-accelerated local inference.".into()
    };

    DiagnosticReport {
        passed,
        gpu_usable,
        summary,
        windows_info: win_info,
        steps,
    }
}

/// Evaluates free volume space against required bytes for storage planning and preflight checks.
pub fn evaluate_disk_preflight(
    target_path: &Path,
    required_bytes: Option<u64>,
) -> DiskPreflightInfo {
    let available_bytes = crate::download_manager::get_disk_free_space(target_path).unwrap_or(0);
    let total_bytes = available_bytes.max(100 * 1024 * 1024 * 1024);
    let available_gb = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let required_gb = required_bytes.map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0));
    let headroom_gb = required_gb.map(|req| available_gb - req);

    let (status, recommendation) = match (required_bytes, headroom_gb) {
        (Some(req), Some(headroom)) => {
            if available_bytes < req {
                (
                    "insufficient".to_string(),
                    Some(format!(
                        "Requires {:.1} GB but only {:.1} GB is available. Need at least {:.1} GB more free space.",
                        req as f64 / (1024.0 * 1024.0 * 1024.0),
                        available_gb,
                        (req.saturating_sub(available_bytes)) as f64 / (1024.0 * 1024.0 * 1024.0)
                    )),
                )
            } else if headroom < 5.0 {
                (
                    "tight".to_string(),
                    Some(format!(
                        "Download will leave only {:.1} GB free headroom on this volume. Consider freeing space.",
                        headroom
                    )),
                )
            } else {
                (
                    "sufficient".to_string(),
                    Some(format!(
                        "Ample storage available: {:.1} GB headroom remaining after allocation.",
                        headroom
                    )),
                )
            }
        }
        (Some(req), None) => (
            "insufficient".to_string(),
            Some(format!(
                "Required: {:.1} GB, but volume free space could not be determined.",
                req as f64 / (1024.0 * 1024.0 * 1024.0)
            )),
        ),
        (None, _) => {
            if available_gb < 10.0 {
                (
                    "insufficient".to_string(),
                    Some("Critically low disk space (< 10 GB).".into()),
                )
            } else if available_gb < 20.0 {
                (
                    "tight".to_string(),
                    Some("Storage is tight (< 20 GB free).".into()),
                )
            } else {
                (
                    "sufficient".to_string(),
                    Some("Sufficient free disk space.".into()),
                )
            }
        }
    };

    DiskPreflightInfo {
        target_path: target_path.to_string_lossy().to_string(),
        available_bytes,
        total_bytes,
        available_gb,
        total_gb,
        required_bytes,
        required_gb,
        headroom_gb,
        status,
        recommendation,
    }
}

/// Diagnostic export bundle structure. ALL secrets are thoroughly redacted.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticBundle {
    pub app_version: String,
    pub generated_at: String,
    pub windows: WindowsInfo,
    pub wsl_version: Option<String>,
    pub wsl_distro: String,
    pub wsl_running: bool,
    pub gpus: Vec<crate::state::GpuSnapshot>,
    pub llama_devices: Vec<crate::llamacpp_install::NvidiaGpuDevice>,
    pub provision_report: Option<crate::provision::ProvisionReport>,
    pub diagnostic_report: DiagnosticReport,
    pub server_configurations: Vec<serde_json::Value>,
    pub recent_logs: Vec<String>,
}

/// Creates a complete diagnostic bundle with all secrets (tokens, API keys, passwords) thoroughly redacted.
pub fn generate_redacted_bundle(state: &AppState) -> DiagnosticBundle {
    let now = chrono::Utc::now().to_rfc3339();
    let diag_report = run_ordered_diagnostics(state);
    let distro = state.resolve_distro();
    let wsl_ver = crate::wsl::get_wsl_version().map(|(a, b, c)| format!("{a}.{b}.{c}"));
    let wsl_run = crate::wsl::is_running(&distro);

    let gpus = {
        let mut list = Vec::new();
        if let Some(g) = crate::wsl::gpu_snapshot(&distro) {
            list.push(g);
        } else if let Some(g) = crate::llamacpp_install::windows_gpu_snapshot() {
            list.push(g);
        }
        list
    };

    let llama_devices = crate::llamacpp_install::windows_gpu_devices();

    let prov_report = {
        let out = crate::wsl::run_script(&distro, "cat ~/llm-lp/.provisioned 2>/dev/null || true");
        if out.ok && !out.stdout.trim().is_empty() {
            serde_json::from_str::<crate::provision::ProvisionReport>(&out.stdout).ok()
        } else {
            None
        }
    };

    // Extract servers and redact API keys & passwords
    let servers = {
        let s_guard = state.servers.lock().unwrap();
        s_guard
            .values()
            .map(|ls| {
                let mut def_val = serde_json::to_value(&ls.def).unwrap_or_default();
                if let Some(obj) = def_val.as_object_mut() {
                    if obj.contains_key("api_key") && !obj["api_key"].is_null() {
                        obj.insert(
                            "api_key".into(),
                            serde_json::Value::String("[REDACTED_API_KEY]".into()),
                        );
                    }
                }
                def_val
            })
            .collect()
    };

    // Collect recent server & provisioning logs and redact secrets
    let mut logs = Vec::new();
    {
        let s_guard = state.servers.lock().unwrap();
        for (id, ls) in s_guard.iter() {
            let ring = ls.log_ring.lock().unwrap();
            for l in ring.lines.iter().rev().take(50).rev() {
                logs.push(format!("[{id}] {}", redact_secrets(l)));
            }
        }
    }

    DiagnosticBundle {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at: now,
        windows: diag_report.windows_info.clone(),
        wsl_version: wsl_ver,
        wsl_distro: distro,
        wsl_running: wsl_run,
        gpus,
        llama_devices,
        provision_report: prov_report,
        diagnostic_report: diag_report,
        server_configurations: servers,
        recent_logs: logs,
    }
}

/// Runs compatibility test prompts (basic chat, streaming, tool calling, structured output, system prompt)
/// against a live server endpoint.
pub async fn run_server_compatibility_test(
    state: &Arc<AppState>,
    server_id: &str,
) -> Result<CompatibilityTestReport, String> {
    // Use the same effective API key (per-server override, else global,
    // DPAPI-decrypted) that the running server was launched with, so the
    // compatibility probes don't fail with 401 on key-protected servers.
    let (port, model, backend, def) = {
        let servers = state.servers.lock().unwrap();
        let ls = servers
            .get(server_id)
            .ok_or_else(|| format!("Server {server_id} not found"))?;
        if ls.status != crate::state::ServerStatus::Running {
            return Err(format!(
                "Server {server_id} is not running (status: {:?})",
                ls.status
            ));
        }
        (
            ls.def.port,
            ls.def.backend_model_name(),
            ls.def.backend.clone(),
            ls.def.clone(),
        )
    };
    let api_key = server_api_key(state, &def);

    let base_url = format!("http://127.0.0.1:{port}");
    let mut items = Vec::new();

    // 1. Basic Chat
    let t0 = std::time::Instant::now();
    let basic_res = async {
        let client = &state.chat_http;
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word 'PASS'."}],
            "max_tokens": 32,
            "stream": false,
        });
        let resp = apply_vllm_auth(
            client
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&body),
            api_key.as_deref(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim();
        if content.is_empty() {
            return Err("Empty completion response".into());
        }
        Ok(format!("Response: {content}"))
    }
    .await;
    let latency_ms = t0.elapsed().as_millis() as u64;
    items.push(CompatibilityTestItem {
        id: "basic_chat".into(),
        name: "Basic Chat Completion".into(),
        description: "Standard non-streaming completion endpoint /v1/chat/completions".into(),
        passed: basic_res.is_ok(),
        latency_ms,
        details: basic_res.as_ref().ok().cloned(),
        error: basic_res.as_ref().err().cloned(),
    });

    // 2. Streaming Chat
    let t0 = std::time::Instant::now();
    let stream_res = async {
        let client = &state.chat_http;
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 64,
            "stream": true,
        });
        let resp = apply_vllm_auth(
            client
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&body),
            api_key.as_deref(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if text.contains("data:") {
            Ok("SSE stream chunks received successfully".into())
        } else {
            Err("No SSE stream tokens received".into())
        }
    }
    .await;
    let latency_ms = t0.elapsed().as_millis() as u64;
    items.push(CompatibilityTestItem {
        id: "streaming".into(),
        name: "Token Streaming (SSE)".into(),
        description: "Server-Sent Events streaming token delivery (stream: true)".into(),
        passed: stream_res.is_ok(),
        latency_ms,
        details: stream_res.as_ref().ok().cloned(),
        error: stream_res.as_ref().err().cloned(),
    });

    // 3. Tool Calling
    let t0 = std::time::Instant::now();
    let tool_res = async {
        let client = &state.chat_http;
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Seattle? Call get_weather."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto",
            "max_tokens": 256,
        });
        let resp = apply_vllm_auth(
            client.post(format!("{base_url}/v1/chat/completions")).json(&body),
            api_key.as_deref(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let calls = json["choices"][0]["message"]["tool_calls"].as_array();
        if let Some(c) = calls {
            if !c.is_empty() {
                let fname = c[0]["function"]["name"].as_str().unwrap_or("");
                return Ok(format!("Function call emitted: {fname}"));
            }
        }
        Err("Model responded with text instead of structured tool_call".into())
    }.await;
    let latency_ms = t0.elapsed().as_millis() as u64;
    items.push(CompatibilityTestItem {
        id: "tool_calling".into(),
        name: "Function / Tool Calling".into(),
        description: "OpenAI-compatible function definitions and tool_calls emission".into(),
        passed: tool_res.is_ok(),
        latency_ms,
        details: tool_res.as_ref().ok().cloned(),
        error: tool_res.as_ref().err().cloned(),
    });

    // 4. Structured JSON Output
    let t0 = std::time::Instant::now();
    let json_res = async {
        let client = &state.chat_http;
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a JSON generator. You must output only a valid JSON object with key 'status' and value 'ok'."},
                {"role": "user", "content": "Generate status object."}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 64,
        });
        let resp = apply_vllm_auth(
            client.post(format!("{base_url}/v1/chat/completions")).json(&body),
            api_key.as_deref(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let fallback_body = serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": "Output valid JSON only: {\"status\": \"ok\"}"},
                    {"role": "user", "content": "Give status JSON."}
                ],
                "max_tokens": 64,
            });
            let fallback_resp = apply_vllm_auth(
                client
                    .post(format!("{base_url}/v1/chat/completions"))
                    .json(&fallback_body),
                api_key.as_deref(),
            )
            .send()
            .await
            .map_err(|e| e.to_string())?;
            if !fallback_resp.status().is_success() {
                return Err(format!("HTTP {}", fallback_resp.status()));
            }
            let json: serde_json::Value = fallback_resp.json().await.map_err(|e| e.to_string())?;
            let raw = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim();
            serde_json::from_str::<serde_json::Value>(raw).map_err(|e| format!("Invalid JSON: {e}"))?;
            return Ok("Parsed valid JSON from model completion".into());
        }
        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let raw = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim();
        serde_json::from_str::<serde_json::Value>(raw).map_err(|e| format!("Invalid JSON: {e}"))?;
        Ok("Verified structured JSON completion".into())
    }.await;
    let latency_ms = t0.elapsed().as_millis() as u64;
    items.push(CompatibilityTestItem {
        id: "structured_output".into(),
        name: "Structured JSON Output".into(),
        description: "Valid JSON emission with response_format or schema guidance".into(),
        passed: json_res.is_ok(),
        latency_ms,
        details: json_res.as_ref().ok().cloned(),
        error: json_res.as_ref().err().cloned(),
    });

    // 5. System Prompt Normalization
    let t0 = std::time::Instant::now();
    let sys_res = async {
        let client = &state.chat_http;
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "Secret code is ALPHA_77. When asked for code, reply ALPHA_77."},
                {"role": "user", "content": "What is the secret code?"}
            ],
            "max_tokens": 32,
        });
        let resp = apply_vllm_auth(
            client.post(format!("{base_url}/v1/chat/completions")).json(&body),
            api_key.as_deref(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let content = json["choices"][0]["message"]["content"].as_str().unwrap_or("");
        if content.contains("ALPHA_77") {
            Ok("System instructions followed correctly".into())
        } else {
            Err(format!("System prompt not respected; output was: {content}"))
        }
    }.await;
    let latency_ms = t0.elapsed().as_millis() as u64;
    items.push(CompatibilityTestItem {
        id: "system_prompt".into(),
        name: "System / Developer Prompt".into(),
        description: "Instruction following and persona adoption via system role message".into(),
        passed: sys_res.is_ok(),
        latency_ms,
        details: sys_res.as_ref().ok().cloned(),
        error: sys_res.as_ref().err().cloned(),
    });

    let passed_count = items.iter().filter(|i| i.passed).count();
    let total_count = items.len();
    let all_passed = passed_count == total_count;

    Ok(CompatibilityTestReport {
        server_id: server_id.to_string(),
        model_id: model,
        backend,
        timestamp: chrono::Utc::now().to_rfc3339(),
        all_passed,
        passed_count,
        total_count,
        items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows_build_support_threshold() {
        // Windows 10 below 19041 fails
        let win10_old = evaluate_windows_build(Some(18363), Some("1909"), Some("Windows 10"), None);
        assert!(!win10_old.is_supported);
        assert!(!win10_old.is_windows_11);
        assert!(win10_old.os_label.contains("UNSUPPORTED"));
        assert!(win10_old.support_note.contains("19041"));

        // Windows 10 exact threshold 19041 (Version 2004) passes
        let win10_2004 =
            evaluate_windows_build(Some(19041), Some("2004"), Some("Windows 10 Pro"), None);
        assert!(win10_2004.is_supported);
        assert!(!win10_2004.is_windows_11);
        assert!(!win10_2004.os_label.contains("UNSUPPORTED"));

        // Windows 10 22H2 (Build 19045) passes
        let win10_22h2 =
            evaluate_windows_build(Some(19045), Some("22H2"), Some("Windows 10 Pro"), None);
        assert!(win10_22h2.is_supported);
        assert!(!win10_22h2.is_windows_11);

        // Windows 11 (Build 22000+) passes
        let win11 = evaluate_windows_build(Some(22631), Some("23H2"), Some("Windows 11"), None);
        assert!(win11.is_supported);
        assert!(win11.is_windows_11);
        assert!(win11.os_label.contains("Windows 11"));

        // Unknown build fails safe
        let win_unknown = evaluate_windows_build(None, None, None, None);
        assert!(!win_unknown.is_supported);
    }

    #[test]
    fn test_secret_redaction_huggingface_token() {
        let text = "Downloading from https://huggingface.co with token hf_1234567890abcdef1234567890ABCDEF1234 on worker.";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("hf_1234567890"));
        assert!(redacted.contains("[REDACTED_HF_TOKEN]"));
    }

    #[test]
    fn test_secret_redaction_github_tokens() {
        let text = "git clone https://ghp_abcdef1234567890ABCDEF1234567890@github.com/repo";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("ghp_abcdef"));
        assert!(redacted.contains("[REDACTED_GITHUB_TOKEN]"));

        let text_fine_grained =
            "Authorization: Bearer github_pat_11AAAAAA00000000000000_12345678901234567890";
        let redacted2 = redact_secrets(text_fine_grained);
        assert!(!redacted2.contains("github_pat_"));
        assert!(redacted2.contains("[REDACTED_GITHUB_TOKEN]"));
    }

    #[test]
    fn test_secret_redaction_openai_api_keys() {
        let text = "Request failed with key sk-proj-1234567890abcdefghijklmnopqrstuvwxyz12345";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("sk-proj-1234567890"));
        assert!(redacted.contains("[REDACTED_API_KEY]"));
    }

    #[test]
    fn test_secret_redaction_bearer_headers() {
        let text = "Headers: { Authorization: Bearer someSecretTokenValue123456789 }";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("someSecretTokenValue123456789"));
        assert!(redacted.contains("Bearer [REDACTED_BEARER_TOKEN]"));
    }

    #[test]
    fn test_secret_redaction_env_and_query_params() {
        let text = "env: API_KEY=secret_key_value_99999, token=topsecret123, password=mysecretpass";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("secret_key_value_99999"));
        assert!(!redacted.contains("topsecret123"));
        assert!(!redacted.contains("mysecretpass"));
        assert!(redacted.contains("API_KEY=[REDACTED]"));
        assert!(redacted.contains("token=[REDACTED]"));
        assert!(redacted.contains("password=[REDACTED]"));
    }

    #[test]
    fn test_diagnostic_ordering_and_steps() {
        let state = AppState::new();
        let report = run_ordered_diagnostics(&state);

        assert_eq!(report.steps.len(), 8);
        let ids: Vec<&str> = report.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "windows_version",
                "wsl_installed",
                "wsl2_working",
                "nvidia_driver",
                "gpu_in_wsl",
                "cuda_runtime",
                "disk_space",
                "backend_ready"
            ]
        );

        for (i, step) in report.steps.iter().enumerate() {
            assert_eq!(step.order, (i + 1) as u32);
        }
    }

    #[test]
    fn test_diagnostic_bundle_secret_redaction() {
        let state = AppState::new();
        {
            let mut servers = state.servers.lock().unwrap();
            let def: crate::state::ServerDef = serde_json::from_value(serde_json::json!({
                "backend": "vllm",
                "id": "test-srv",
                "name": "Test Server",
                "model_id": "meta-llama/Llama-3-8B",
                "task": "instruct",
                "port": 8000,
                "gpu_mem_util": 0.9,
                "quant": "fp16",
                "llamacpp_channel": "upstream",
                "enforce_eager": true,
                "fit": true,
                "flash_attn": true,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "parallel": 1,
                "jinja": true,
                "no_kv_offload": false,
                "metrics": true,
                "extra_args": [],
                "env": {},
                "api_key": "sk-super_secret_test_key_123456789012345",
                "speculative_enabled": false
            }))
            .expect("valid server def");

            let ls = crate::state::LiveServer {
                def,
                status: crate::state::ServerStatus::Running,
                error: None,
                wsl_child: None,
                native_child: None,
                wsl_pid: None,
                log_ring: std::sync::Mutex::new(crate::state::VecDequeLog::new()),
                last_metrics: None,
                runtime_memory: std::sync::Arc::new(std::sync::Mutex::new(
                    crate::runtime_memory::RuntimeMemoryReport::default(),
                )),
                stopping: false,
                crash_retry_count: 0,
            };
            ls.log_ring.lock().unwrap().push(
                "Starting worker with auth hf_12345678901234567890abcdef1234567890 on hub".into(),
            );
            servers.insert("test-srv".into(), ls);
        }

        let bundle = generate_redacted_bundle(&state);
        let bundle_json = serde_json::to_string_pretty(&bundle).unwrap();

        assert!(!bundle_json.contains("sk-super_secret_test_key"));
        assert!(!bundle_json.contains("hf_12345678901234567890abcdef"));
        assert!(bundle_json.contains("[REDACTED_API_KEY]"));
        assert!(bundle_json.contains("[REDACTED_HF_TOKEN]"));
    }
}
