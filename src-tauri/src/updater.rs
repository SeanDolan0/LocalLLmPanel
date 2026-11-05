//! In-app GitHub release checker and auto-installer for Local LLM Panel.

use anyhow::{bail, Context, Result};
use regex_lite::Regex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::state::AppState;

pub const GITHUB_REPO: &str = "SeanDolan0/LocalLLmPanel";
pub const RELEASES_LATEST_API: &str =
    "https://api.github.com/repos/SeanDolan0/LocalLLmPanel/releases/latest";

static UPDATE_DOWNLOADING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUpdateInfo {
    pub available: bool,
    pub current_version: String,
    pub latest_version: String,
    pub release_name: String,
    pub release_notes: String,
    pub published_at: Option<String>,
    pub html_url: String,
    pub download_url: Option<String>,
    pub asset_name: Option<String>,
    pub asset_size: Option<u64>,
    pub sha256_url: Option<String>,
    pub is_installer: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUpdateProgress {
    pub percent: f32,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub speed_bps: f64,
    pub phase: String, // "downloading" | "verifying" | "ready" | "error"
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateLaunchResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NsisTarget {
    pub install_dir: PathBuf,
    pub executable_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdaterScriptParams {
    pub installer_path: PathBuf,
    pub app_pid: u32,
    pub log_path: PathBuf,
    pub status_path: PathBuf,
    pub install_dir: PathBuf,
    pub target_executable: PathBuf,
    pub expected_version: String,
    pub skip_ui: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LastUpdateStatus {
    pub status: String,
    pub message: String,
    pub exit_code: Option<i32>,
    pub expected_version: Option<String>,
    pub target_exe: Option<String>,
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubReleaseResponse {
    tag_name: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
    #[serde(default)]
    assets: Vec<GithubReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GithubReleaseAsset {
    pub name: String,
    pub size: u64,
    pub browser_download_url: String,
}

/// Parsed numerical semantic version: `(major, minor, patch, pre)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
}

impl ParsedVersion {
    pub fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim().trim_start_matches(|c| c == 'v' || c == 'V');
        if trimmed.is_empty() {
            return None;
        }

        let (core, pre) = match trimmed.split_once('-') {
            Some((c, p)) => (c, Some(p.to_string())),
            None => (trimmed, None),
        };

        let mut parts = core.split('.');
        let major = parts.next()?.parse::<u64>().ok()?;
        let minor = parts
            .next()
            .and_then(|p| p.parse::<u64>().ok())
            .unwrap_or(0);
        let patch = parts
            .next()
            .and_then(|p| p.parse::<u64>().ok())
            .unwrap_or(0);

        Some(Self {
            major,
            minor,
            patch,
            pre,
        })
    }

    /// Returns true if `self` is strictly newer than `other`.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        if (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch) {
            return true;
        }
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch) {
            return false;
        }

        // When major.minor.patch match:
        // A release without pre-release tags is newer than one with pre-release tags
        // e.g. 1.9.6 is newer than 1.9.6-rc1
        match (&self.pre, &other.pre) {
            (None, Some(_)) => true,
            (Some(s), Some(o)) => s > o,
            _ => false,
        }
    }
}

/// Select the preferred asset for installation on Windows.
///
/// We prioritize:
/// 1. NSIS setup installer (`Local.LLM.Panel_<version>_x64-setup.exe` or `*setup*.exe`)
/// 2. Standalone `.exe` (`LocalLLMPanel-*.exe`)
///
/// Also matches the corresponding `.sha256` checksum file if uploaded.
pub fn select_release_asset(
    assets: &[GithubReleaseAsset],
) -> (
    Option<&GithubReleaseAsset>,
    Option<&GithubReleaseAsset>,
    bool,
) {
    let mut installer: Option<&GithubReleaseAsset> = None;
    let mut standalone: Option<&GithubReleaseAsset> = None;

    let setup_re = Regex::new(r"(?i)setup.*\.exe$").unwrap();
    let exe_re = Regex::new(r"(?i)\.exe$").unwrap();

    for asset in assets {
        if asset.name.to_ascii_lowercase().ends_with(".sha256") {
            continue;
        }
        if setup_re.is_match(&asset.name) {
            installer = Some(asset);
            break;
        } else if exe_re.is_match(&asset.name) && standalone.is_none() {
            standalone = Some(asset);
        }
    }

    let (chosen, is_installer) = match installer {
        Some(inst) => (Some(inst), true),
        None => (standalone, false),
    };

    let sha256_asset = chosen.and_then(|c| {
        let expected_sha_name = format!("{}.sha256", c.name).to_ascii_lowercase();
        assets
            .iter()
            .find(|a| a.name.to_ascii_lowercase() == expected_sha_name)
    });

    (chosen, sha256_asset, is_installer)
}

/// Fetch the latest GitHub release info and evaluate if an update is available.
pub async fn fetch_latest_release(
    client: &reqwest::Client,
    token: Option<&str>,
) -> Result<AppUpdateInfo> {
    let current_version_str = env!("CARGO_PKG_VERSION");
    let current_parsed = ParsedVersion::parse(current_version_str).unwrap_or(ParsedVersion {
        major: 0,
        minor: 0,
        patch: 0,
        pre: None,
    });

    let mut req = client
        .get(RELEASES_LATEST_API)
        .header(reqwest::header::USER_AGENT, "Local-LLM-Panel-Updater");

    if let Some(t) = token.filter(|t| !t.trim().is_empty()) {
        req = req.bearer_auth(t.trim());
    }

    let resp = req
        .send()
        .await
        .context("Failed to connect to GitHub releases API")?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!("No published GitHub releases found for {}", GITHUB_REPO);
    }

    let resp = resp
        .error_for_status()
        .context("GitHub releases API returned an error")?;

    let release: GithubReleaseResponse = resp
        .json()
        .await
        .context("Failed to parse GitHub release JSON")?;

    let latest_version_str = release
        .tag_name
        .trim()
        .trim_start_matches(['v', 'V'])
        .to_string();
    let latest_parsed = ParsedVersion::parse(&latest_version_str).unwrap_or(ParsedVersion {
        major: 0,
        minor: 0,
        patch: 0,
        pre: None,
    });

    let available = latest_parsed.is_newer_than(&current_parsed);

    let (asset, sha256_asset, is_installer) = select_release_asset(&release.assets);

    Ok(AppUpdateInfo {
        available,
        current_version: format!("v{current_version_str}"),
        latest_version: format!("v{latest_version_str}"),
        release_name: release.name.unwrap_or_else(|| release.tag_name.clone()),
        release_notes: release.body.unwrap_or_default(),
        published_at: release.published_at,
        html_url: release
            .html_url
            .unwrap_or_else(|| format!("https://github.com/{GITHUB_REPO}/releases")),
        download_url: asset.map(|a| a.browser_download_url.clone()),
        asset_name: asset.map(|a| a.name.clone()),
        asset_size: asset.map(|a| a.size),
        sha256_url: sha256_asset.map(|a| a.browser_download_url.clone()),
        is_installer,
        error: None,
    })
}

/// Download an update asset and launch the installer detached.
pub async fn download_and_install_update_impl(
    app: AppHandle,
    st: Arc<AppState>,
    info: AppUpdateInfo,
) -> Result<UpdateLaunchResult> {
    if UPDATE_DOWNLOADING.swap(true, Ordering::SeqCst) {
        bail!("An update download is already in progress");
    }

    struct DownloadGuard;
    impl Drop for DownloadGuard {
        fn drop(&mut self) {
            UPDATE_DOWNLOADING.store(false, Ordering::SeqCst);
        }
    }
    let _guard = DownloadGuard;

    let download_url = match info.download_url {
        Some(url) => url,
        None => bail!(
            "No executable asset available for download in release {}",
            info.latest_version
        ),
    };
    let is_installer = info.is_installer;
    let asset_name = info
        .asset_name
        .unwrap_or_else(|| "LocalLLMPanel-setup.exe".into());

    let temp_dir = std::env::temp_dir().join("LocalLLMPanel-updates");
    std::fs::create_dir_all(&temp_dir)
        .context("Failed to create temporary directory for updater")?;

    let dest_file = temp_dir.join(&asset_name);

    let token = st.github_token();

    let _ = app.emit(
        "app-update-progress",
        AppUpdateProgress {
            percent: 0.0,
            downloaded_bytes: 0,
            total_bytes: info.asset_size,
            speed_bps: 0.0,
            phase: "downloading".into(),
            error: None,
        },
    );

    // 1. Download installer file
    let mut req = st
        .http
        .get(&download_url)
        .header(reqwest::header::USER_AGENT, "Local-LLM-Panel-Updater");

    if let Some(t) = token.as_deref().filter(|t| !t.trim().is_empty()) {
        req = req.bearer_auth(t.trim());
    }

    let resp = req
        .send()
        .await
        .context("Failed to connect to download URL")?;
    let resp = resp
        .error_for_status()
        .context("Download URL returned an error response")?;

    let total_bytes = resp.content_length().or(info.asset_size);
    let mut stream = resp.bytes_stream();

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::File::create(&dest_file)
        .await
        .context("Failed to create destination update file")?;

    let start_time = Instant::now();
    let mut downloaded: u64 = 0;
    let mut last_emit = Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error while receiving update download stream")?;
        file.write_all(&chunk)
            .await
            .context("Error writing update payload to disk")?;
        downloaded += chunk.len() as u64;

        if last_emit.elapsed() > Duration::from_millis(200) {
            let elapsed_sec = start_time.elapsed().as_secs_f64().max(0.001);
            let speed_bps = downloaded as f64 / elapsed_sec;
            let percent = match total_bytes {
                Some(total) if total > 0 => ((downloaded as f64 / total as f64) * 100.0) as f32,
                _ => 0.0,
            };
            let _ = app.emit(
                "app-update-progress",
                AppUpdateProgress {
                    percent,
                    downloaded_bytes: downloaded,
                    total_bytes,
                    speed_bps,
                    phase: "downloading".into(),
                    error: None,
                },
            );
            last_emit = Instant::now();
        }
    }

    file.flush()
        .await
        .context("Failed to flush installer file to disk")?;
    drop(file);

    // 2. Require SHA256 checksum verification before executing downloaded code.
    let sha_url = info
        .sha256_url
        .ok_or_else(|| anyhow::anyhow!("Release does not provide a SHA256 checksum"))?;
    let _ = app.emit(
        "app-update-progress",
        AppUpdateProgress {
            percent: 100.0,
            downloaded_bytes: downloaded,
            total_bytes,
            speed_bps: 0.0,
            phase: "verifying".into(),
            error: None,
        },
    );

    let expected_hash = st
        .http
        .get(&sha_url)
        .send()
        .await
        .context("Failed to download update checksum")?
        .error_for_status()
        .context("Update checksum URL returned an error")?
        .text()
        .await
        .context("Failed to read update checksum")?
        .split_whitespace()
        .next()
        .filter(|hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| anyhow::anyhow!("Update checksum is missing or malformed"))?;

    let file_bytes = tokio::fs::read(&dest_file)
        .await
        .context("Failed to read downloaded file for hash calculation")?;
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let computed = format!("{:x}", hasher.finalize()).to_ascii_lowercase();
    if computed != expected_hash {
        let err_msg = format!("Checksum mismatch: expected {expected_hash}, got {computed}");
        let _ = app.emit(
            "app-update-progress",
            AppUpdateProgress {
                percent: 100.0,
                downloaded_bytes: downloaded,
                total_bytes,
                speed_bps: 0.0,
                phase: "error".into(),
                error: Some(err_msg.clone()),
            },
        );
        bail!(err_msg);
    }

    let _ = app.emit(
        "app-update-progress",
        AppUpdateProgress {
            percent: 100.0,
            downloaded_bytes: downloaded,
            total_bytes,
            speed_bps: 0.0,
            phase: "ready".into(),
            error: None,
        },
    );

    let expected_version = info
        .latest_version
        .trim()
        .trim_start_matches(['v', 'V'])
        .to_string();

    let target = resolve_nsis_target(
        std::env::current_exe().ok().as_deref(),
        st.config().app_install_dir.as_deref(),
    );

    // Persist actual NSIS install directory to config if changed
    {
        let mut cfg = st.config.lock().unwrap();
        let target_dir_str = target.install_dir.to_string_lossy().to_string();
        if cfg.app_install_dir.as_deref() != Some(&target_dir_str) {
            cfg.app_install_dir = Some(target_dir_str);
            let _ = cfg.save();
        }
    }

    if is_installer {
        // Silent NSIS update: a detached PowerShell helper waits for this
        // process to exit (exiting triggers RunEvent::Exit, which stops every
        // AI server first), runs the installer with /S and /D, verifies the installed
        // binary and its version, and only then relaunches it.
        #[cfg(target_os = "windows")]
        {
            launch_silent_installer_and_exit(&app, &dest_file, &expected_version, &target)?;
            return Ok(UpdateLaunchResult {
                success: true,
                message: format!(
                    "Update to v{} downloaded and verified. Local LLM Panel will close now; the installer will update files at {} and relaunch automatically.",
                    expected_version,
                    target.install_dir.display()
                ),
            });
        }
        #[cfg(not(target_os = "windows"))]
        {
            launch_installer(&dest_file)?;
            Ok(UpdateLaunchResult {
                success: true,
                message: "Installer launched. You can exit Local LLM Panel to complete the update."
                    .to_string(),
            })
        }
    } else {
        // Portable standalone binary: it cannot replace the running process,
        // so fall back to letting the user restart.
        launch_installer(&dest_file)?;
        Ok(UpdateLaunchResult {
            success: true,
            message: "Standalone executable launched. Close Local LLM Panel and run it to apply the update.".to_string(),
        })
    }
}

/// Determine the actual NSIS install directory and target executable path.
pub fn resolve_nsis_target(
    current_exe: Option<&std::path::Path>,
    persisted_dir: Option<&str>,
) -> NsisTarget {
    // 1. If currently running executable has uninstall.exe in its parent directory,
    // it was launched directly from an active installation.
    if let Some(exe) = current_exe {
        if let Some(parent) = exe.parent() {
            if parent.join("uninstall.exe").exists() {
                let file_name = exe
                    .file_name()
                    .unwrap_or_else(|| std::ffi::OsStr::new("llm-panel.exe"));
                return NsisTarget {
                    install_dir: parent.to_path_buf(),
                    executable_path: parent.join(file_name),
                };
            }
        }
    }

    // 2. Query Windows Registry for Local LLM Panel uninstall entry.
    #[cfg(target_os = "windows")]
    if let Some(target) = query_registry_nsis_target() {
        return target;
    }

    // 3. Persisted directory in configuration if it exists on disk.
    if let Some(dir_str) = persisted_dir.map(str::trim).filter(|s| !s.is_empty()) {
        let dir = PathBuf::from(dir_str);
        if dir.exists() {
            let exe = dir.join("llm-panel.exe");
            return NsisTarget {
                install_dir: dir,
                executable_path: exe,
            };
        }
    }

    // 4. Standard Tauri install directories.
    if let Some(target) = check_standard_install_dirs() {
        return target;
    }

    // 5. Default fallback to per-user install directory.
    default_nsis_target()
}

#[cfg(target_os = "windows")]
pub fn query_registry_nsis_target() -> Option<NsisTarget> {
    use std::process::Command;

    let hives = [
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Local LLM Panel",
        r"HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\Local LLM Panel",
        r"HKLM\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Local LLM Panel",
    ];

    for key in &hives {
        let output = Command::new("reg.exe").args(["query", key]).output().ok()?;
        if !output.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mut install_location: Option<PathBuf> = None;
        let mut binary_name: Option<String> = None;

        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("InstallLocation") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "REG_SZ" {
                    let raw_val = rest.split("REG_SZ").nth(1).unwrap_or("").trim();
                    let clean = raw_val.trim_matches('"').trim();
                    if !clean.is_empty() {
                        install_location = Some(PathBuf::from(clean));
                    }
                }
            } else if let Some(rest) = line.strip_prefix("MainBinaryName") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "REG_SZ" {
                    let raw_val = rest.split("REG_SZ").nth(1).unwrap_or("").trim();
                    let clean = raw_val.trim_matches('"').trim();
                    if !clean.is_empty() {
                        binary_name = Some(clean.to_string());
                    }
                }
            }
        }

        if let Some(dir) = install_location {
            let bin = binary_name.unwrap_or_else(|| "llm-panel.exe".to_string());
            let exe = dir.join(&bin);
            return Some(NsisTarget {
                install_dir: dir,
                executable_path: exe,
            });
        }
    }

    None
}

fn check_standard_install_dirs() -> Option<NsisTarget> {
    let mut candidates = Vec::new();

    if let Some(local_app_data) = dirs::data_local_dir() {
        candidates.push(local_app_data.join("Local LLM Panel"));
        candidates.push(local_app_data.join("Programs").join("Local LLM Panel"));
    }

    if let Some(pf) = std::env::var_os("ProgramFiles") {
        candidates.push(PathBuf::from(pf).join("Local LLM Panel"));
    }
    if let Some(pf86) = std::env::var_os("ProgramFiles(x86)") {
        candidates.push(PathBuf::from(pf86).join("Local LLM Panel"));
    }

    for dir in candidates {
        let exe = dir.join("llm-panel.exe");
        if exe.exists() {
            return Some(NsisTarget {
                install_dir: dir,
                executable_path: exe,
            });
        }
    }

    None
}

fn default_nsis_target() -> NsisTarget {
    let dir = dirs::data_local_dir()
        .map(|d| d.join("Local LLM Panel"))
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\Default\AppData\Local\Local LLM Panel"));
    let exe = dir.join("llm-panel.exe");
    NsisTarget {
        install_dir: dir,
        executable_path: exe,
    }
}

pub fn generate_updater_script(params: &UpdaterScriptParams) -> String {
    let installer_str = params.installer_path.to_string_lossy().replace('\'', "''");
    let log_str = params.log_path.to_string_lossy().replace('\'', "''");
    let status_str = params.status_path.to_string_lossy().replace('\'', "''");
    let install_dir_str = params.install_dir.to_string_lossy().replace('\'', "''");
    let target_exe_str = params.target_executable.to_string_lossy().replace('\'', "''");
    let version_str = params.expected_version.replace('\'', "''");
    let app_pid = params.app_pid;
    let skip_ui = if params.skip_ui {
        "$true".to_string()
    } else {
        "($env:LOCALLLMPANEL_UPDATE_NO_UI -eq '1') -or ($env:CI -eq '1') -or ($env:TESTING -eq '1')".to_string()
    };

    format!(
        r#"$ErrorActionPreference = 'Continue'

$installer = $env:LOCALLLMPANEL_UPDATE_INSTALLER
if (-not $installer) {{ $installer = '{installer_str}' }}

$appPid = 0
if ($env:LOCALLLMPANEL_UPDATE_PID) {{
  $appPid = [int]$env:LOCALLLMPANEL_UPDATE_PID
}} else {{
  $appPid = {app_pid}
}}

$log = $env:LOCALLLMPANEL_UPDATE_LOG
if (-not $log) {{ $log = '{log_str}' }}

$installDir = $env:LOCALLLMPANEL_UPDATE_INSTALL_DIR
if (-not $installDir) {{ $installDir = '{install_dir_str}' }}

$targetExe = $env:LOCALLLMPANEL_UPDATE_TARGET_EXE
if (-not $targetExe) {{ $targetExe = '{target_exe_str}' }}

$expectedVersion = $env:LOCALLLMPANEL_UPDATE_VERSION
if (-not $expectedVersion) {{ $expectedVersion = '{version_str}' }}

$statusFile = $env:LOCALLLMPANEL_UPDATE_STATUS
if (-not $statusFile) {{ $statusFile = '{status_str}' }}

$skipUi = {skip_ui}

$logDir = [System.IO.Path]::GetDirectoryName($log)
if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {{
  New-Item -ItemType Directory -Path $logDir -Force -ErrorAction SilentlyContinue | Out-Null
}}

function Write-Log([string]$msg) {{
  $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffK")
  $entry = "[$timestamp] $msg"
  try {{
    $entry | Out-File -LiteralPath $log -Encoding utf8 -Append -ErrorAction SilentlyContinue
  }} catch {{}}
}}

function Write-Status([string]$status, [string]$msg, [int]$exitCode = 0) {{
  if ($statusFile) {{
    try {{
      $dir = [System.IO.Path]::GetDirectoryName($statusFile)
      if ($dir -and -not (Test-Path -LiteralPath $dir)) {{
        New-Item -ItemType Directory -Path $dir -Force -ErrorAction SilentlyContinue | Out-Null
      }}
      $obj = [ordered]@{{
        status = $status
        message = $msg
        exitCode = $exitCode
        expectedVersion = $expectedVersion
        targetExe = $targetExe
        timestamp = (Get-Date).ToString("o")
      }}
      $json = $obj | ConvertTo-Json -Compress
      $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
      [System.IO.File]::WriteAllText($statusFile, $json, $utf8NoBom)
    }} catch {{}}
  }}
}}

function Fail-Update([string]$reason, [int]$code) {{
  Write-Log "ERROR (code $code): $reason"
  Write-Status "failed" $reason $code
  if (-not $skipUi) {{
    try {{
      Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
      [System.Windows.Forms.MessageBox]::Show(
        "Local LLM Panel update to version $expectedVersion failed.`n`nReason: $reason`n`nDiagnostic log: $log",
        "Local LLM Panel Update Failed",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
      ) | Out-Null
    }} catch {{
      try {{
        $wsh = New-Object -ComObject WScript.Shell
        $wsh.Popup("Local LLM Panel update failed.`n`n$reason`n`nLog: $log", 0, "Update Failed", 16) | Out-Null
      }} catch {{}}
    }}
  }}
  exit $code
}}

Write-Log "=================================================="
Write-Log "Local LLM Panel Updater Helper Started"
Write-Log "Expected Version:  $expectedVersion"
Write-Log "Installer Path:    $installer"
Write-Log "Install Dir:       $installDir"
Write-Log "Target Executable: $targetExe"
Write-Log "Original PID:      $appPid"
Write-Log "Log File:          $log"
Write-Log "Status File:       $statusFile"
Write-Log "=================================================="

# 1. Wait for original process to exit
if ($appPid -gt 0) {{
  Write-Log "Waiting for application process (PID $appPid) to terminate..."
  $deadline = (Get-Date).AddSeconds(240)
  while ((Get-Process -Id $appPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {{
    Start-Sleep -Milliseconds 400
  }}
  if (Get-Process -Id $appPid -ErrorAction SilentlyContinue) {{
    Fail-Update "The application (PID $appPid) did not exit within 240 seconds." 1
  }}
  Write-Log "Process $appPid has exited."
}}
Start-Sleep -Milliseconds 500

# 2. Verify installer exists
if (-not $installer -or -not (Test-Path -LiteralPath $installer -PathType Leaf)) {{
  Fail-Update "Installer executable not found at: $installer" 2
}}
Write-Log "Verified installer exists at: $installer"

# 3. Run the NSIS installer with /S
Write-Log "Launching NSIS installer silently..."
$installerArgs = New-Object System.Collections.Generic.List[string]
$installerArgs.Add('/S')
if ($installDir) {{
  $installerArgs.Add("/D=$installDir")
}}

$argDisplay = $installerArgs -join ' '
Write-Log "Executing: Start-Process -FilePath '$installer' -ArgumentList '$argDisplay' -Wait"

$installerExitCode = -1
try {{
  $proc = Start-Process -FilePath $installer -ArgumentList $installerArgs -PassThru -Wait -ErrorAction Stop
  $installerExitCode = $proc.ExitCode
  Write-Log "Installer completed with exit code: $installerExitCode"
}} catch {{
  Fail-Update "Failed to launch installer process: $($_.Exception.Message)" 3
}}

if ($installerExitCode -ne 0) {{
  Fail-Update "Installer failed with non-zero exit code: $installerExitCode" $installerExitCode
}}

# 4. Verify installed executable exists
Write-Log "Verifying installed executable..."
if (-not $targetExe -and $installDir) {{
  $targetExe = Join-Path $installDir "llm-panel.exe"
}}

if (-not $targetExe -or -not (Test-Path -LiteralPath $targetExe -PathType Leaf)) {{
  if ($installDir) {{
    $cand = Join-Path $installDir "llm-panel.exe"
    if (Test-Path -LiteralPath $cand -PathType Leaf) {{
      $targetExe = $cand
      Write-Log "Found executable at install directory: $targetExe"
    }}
  }}
}}

if (-not $targetExe -or -not (Test-Path -LiteralPath $targetExe -PathType Leaf)) {{
  Fail-Update "Installed executable not found after installer completed. Looked at: $targetExe" 4
}}
Write-Log "Installed executable verified at: $targetExe"

# 5. Verify that its file version or embedded version matches the downloaded release version
Write-Log "Verifying installed executable version matches release '$expectedVersion'..."
$versionInfo = (Get-Item -LiteralPath $targetExe).VersionInfo
$prodVersion = if ($versionInfo.ProductVersion) {{ $versionInfo.ProductVersion.Trim() }} else {{ "" }}
$fileVersion = if ($versionInfo.FileVersion) {{ $versionInfo.FileVersion.Trim() }} else {{ "" }}

Write-Log "Target executable version info: ProductVersion='$prodVersion', FileVersion='$fileVersion'"

function Normalize-Ver([string]$v) {{
  if (-not $v) {{ return "" }}
  $t = $v.Trim().TrimStart('v','V')
  $parts = $t.Split('.')
  if ($parts.Length -ge 3) {{
    return "$($parts[0]).$($parts[1]).$($parts[2])"
  }}
  return $t
}}

$normExpected = Normalize-Ver $expectedVersion
$normProd = Normalize-Ver $prodVersion
$normFile = Normalize-Ver $fileVersion

Write-Log "Normalized versions: Expected='$normExpected', InstalledProd='$normProd', InstalledFile='$normFile'"

$versionMatch = ($normProd -and $normProd -eq $normExpected) -or ($normFile -and $normFile -eq $normExpected)

if (-not $versionMatch) {{
  Fail-Update "Installed executable version does not match expected version '$expectedVersion'. Found ProductVersion: '$prodVersion', FileVersion: '$fileVersion'." 5
}}
Write-Log "Version verification successful."

if ($env:LOCALLLMPANEL_UPDATE_NO_RELAUNCH -eq '1') {{
  Write-Log "Relaunch skipped by LOCALLLMPANEL_UPDATE_NO_RELAUNCH"
  Write-Status "success" "Update to $expectedVersion verified successfully (relaunch skipped in test)." 0
  exit 0
}}

# 6. Only then relaunch the installed executable
Write-Log "Relaunching updated application: $targetExe"
try {{
  $workDir = Split-Path -LiteralPath $targetExe -Parent
  Start-Process -FilePath $targetExe -WorkingDirectory $workDir -ErrorAction Stop | Out-Null
  Write-Log "Relaunch initiated successfully."
  Write-Status "success" "Update to $expectedVersion successfully installed and relaunched." 0
}} catch {{
  Fail-Update "Failed to relaunch $targetExe`: $($_.Exception.Message)" 6
}}

# 7. Clean up helper script
try {{
  Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
}} catch {{}}
"#
    )
}

pub fn read_last_update_status() -> Option<LastUpdateStatus> {
    let temp_dir = std::env::temp_dir().join("LocalLLMPanel-updates");
    let status_file = temp_dir.join("update-status.json");
    if !status_file.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&status_file).ok()?;
    let clean = content.strip_prefix('\u{feff}').unwrap_or(&content);
    let status: LastUpdateStatus = serde_json::from_str(clean).ok()?;
    let _ = std::fs::remove_file(&status_file);
    Some(status)
}

#[cfg(target_os = "windows")]
fn launch_silent_installer_and_exit(
    app: &AppHandle,
    installer_path: &PathBuf,
    expected_version: &str,
    target: &NsisTarget,
) -> Result<()> {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x00000008;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let temp_dir = installer_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(std::env::temp_dir);
    let script = temp_dir.join(format!("apply-update-{}.ps1", std::process::id()));
    let log = temp_dir.join("update-apply.log");
    let status_file = temp_dir.join("update-status.json");

    let params = UpdaterScriptParams {
        installer_path: installer_path.clone(),
        app_pid: std::process::id(),
        log_path: log.clone(),
        status_path: status_file.clone(),
        install_dir: target.install_dir.clone(),
        target_executable: target.executable_path.clone(),
        expected_version: expected_version.to_string(),
        skip_ui: false,
    };

    let ps = generate_updater_script(&params);
    std::fs::write(&script, ps).with_context(|| {
        format!(
            "Failed to write update helper script at {}",
            script.display()
        )
    })?;

    // Also persist update-target.json
    let target_json = temp_dir.join("update-target.json");
    let target_payload = serde_json::json!({
        "install_dir": target.install_dir.to_string_lossy(),
        "executable_path": target.executable_path.to_string_lossy(),
        "expected_version": expected_version,
        "installer_path": installer_path.to_string_lossy(),
    });
    let _ = std::fs::write(&target_json, target_payload.to_string());

    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.env("LOCALLLMPANEL_UPDATE_INSTALLER", installer_path.as_os_str())
        .env("LOCALLLMPANEL_UPDATE_PID", std::process::id().to_string())
        .env("LOCALLLMPANEL_UPDATE_LOG", log.as_os_str())
        .env("LOCALLLMPANEL_UPDATE_STATUS", status_file.as_os_str())
        .env(
            "LOCALLLMPANEL_UPDATE_INSTALL_DIR",
            target.install_dir.as_os_str(),
        )
        .env(
            "LOCALLLMPANEL_UPDATE_TARGET_EXE",
            target.executable_path.as_os_str(),
        )
        .env("LOCALLLMPANEL_UPDATE_VERSION", expected_version)
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-File",
        ])
        .arg(&script)
        .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW);

    cmd.spawn()
        .with_context(|| format!("Failed to launch update helper at {}", script.display()))?;

    // Give the helper a moment to start, then exit cleanly.
    std::thread::sleep(std::time::Duration::from_millis(300));
    app.exit(0);
    Ok(())
}

#[cfg(target_os = "windows")]
fn launch_installer(path: &PathBuf) -> Result<()> {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x00000008;

    std::process::Command::new(path)
        .creation_flags(DETACHED_PROCESS)
        .spawn()
        .with_context(|| {
            format!(
                "Failed to launch installer executable at {}",
                path.display()
            )
        })?;

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn launch_installer(path: &PathBuf) -> Result<()> {
    std::process::Command::new(path).spawn().with_context(|| {
        format!(
            "Failed to launch installer executable at {}",
            path.display()
        )
    })?;

    Ok(())
}

/// Background periodic or startup check for updates.
pub async fn background_check_update(app: AppHandle) {
    let state: State<'_, Arc<AppState>> = app.state();
    let auto_check = state.config().auto_check_updates;
    if !auto_check {
        return;
    }

    let token = state.inner().github_token();
    let res = fetch_latest_release(&state.http, token.as_deref()).await;
    match res {
        Ok(info) => {
            if info.available {
                let _ = app.emit("app-update-available", &info);
            }
            let mut cache = state.app_update_cache.lock().unwrap();
            *cache = Some((info, Instant::now()));
        }
        Err(e) => {
            eprintln!("[updater] background update check failed: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Implementation functions
// ---------------------------------------------------------------------------

pub async fn check_app_update(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    force: Option<bool>,
) -> Result<AppUpdateInfo, String> {
    let should_force = force.unwrap_or(false);

    // Check cache (10 min TTL) if not forced
    if !should_force {
        let cache = state.app_update_cache.lock().unwrap();
        if let Some((ref cached, fetched_at)) = *cache {
            if fetched_at.elapsed() < Duration::from_secs(600) {
                return Ok(cached.clone());
            }
        }
    }

    let token = state.inner().github_token();
    match fetch_latest_release(&state.http, token.as_deref()).await {
        Ok(info) => {
            if info.available {
                let _ = app.emit("app-update-available", &info);
            }
            let mut cache = state.app_update_cache.lock().unwrap();
            *cache = Some((info.clone(), Instant::now()));
            Ok(info)
        }
        Err(e) => {
            let current_version_str = env!("CARGO_PKG_VERSION");
            Ok(AppUpdateInfo {
                available: false,
                current_version: format!("v{current_version_str}"),
                latest_version: format!("v{current_version_str}"),
                release_name: String::new(),
                release_notes: String::new(),
                published_at: None,
                html_url: format!("https://github.com/{GITHUB_REPO}/releases"),
                download_url: None,
                asset_name: None,
                asset_size: None,
                sha256_url: None,
                is_installer: false,
                error: Some(e.to_string()),
            })
        }
    }
}

pub async fn download_and_install_update(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<UpdateLaunchResult, String> {
    let st = (*state).clone();

    // Get current update info from cache or fetch fresh
    let info = {
        let cache = state.app_update_cache.lock().unwrap();
        cache.as_ref().map(|(c, _)| c.clone())
    };

    let info = match info {
        Some(i) if i.available && i.download_url.is_some() => i,
        _ => {
            let token = state.inner().github_token();
            fetch_latest_release(&state.http, token.as_deref())
                .await
                .map_err(|e| e.to_string())?
        }
    };

    if !info.available {
        return Err("No new update version is available".into());
    }

    download_and_install_update_impl(app, st, info)
        .await
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsed_version_semantics() {
        let v1_9_6 = ParsedVersion::parse("1.9.6").unwrap();
        let v1_9_7 = ParsedVersion::parse("v1.9.7").unwrap();
        let v1_10_0 = ParsedVersion::parse("V1.10.0").unwrap();
        let v2_0_0 = ParsedVersion::parse("2.0.0").unwrap();
        let v1_9_5 = ParsedVersion::parse("1.9.5").unwrap();
        let v1_9_6_rc = ParsedVersion::parse("1.9.6-rc1").unwrap();

        assert!(v1_9_7.is_newer_than(&v1_9_6));
        assert!(v1_10_0.is_newer_than(&v1_9_6));
        assert!(v2_0_0.is_newer_than(&v1_9_6));
        assert!(!v1_9_5.is_newer_than(&v1_9_6));
        assert!(!v1_9_6.is_newer_than(&v1_9_6));

        // Release 1.9.6 is newer than release candidate 1.9.6-rc1
        assert!(v1_9_6.is_newer_than(&v1_9_6_rc));
        assert!(!v1_9_6_rc.is_newer_than(&v1_9_6));
    }

    #[test]
    fn test_select_release_asset() {
        let assets = vec![
            GithubReleaseAsset {
                name: "LocalLLMPanel-v1.9.7.exe.sha256".into(),
                size: 91,
                browser_download_url: "https://.../standalone.sha256".into(),
            },
            GithubReleaseAsset {
                name: "LocalLLMPanel-v1.9.7.exe".into(),
                size: 36000000,
                browser_download_url: "https://.../standalone.exe".into(),
            },
            GithubReleaseAsset {
                name: "Local.LLM.Panel_1.9.7_x64-setup.exe.sha256".into(),
                size: 102,
                browser_download_url: "https://.../setup.exe.sha256".into(),
            },
            GithubReleaseAsset {
                name: "Local.LLM.Panel_1.9.7_x64-setup.exe".into(),
                size: 5800000,
                browser_download_url: "https://.../setup.exe".into(),
            },
        ];

        let (chosen, sha, is_installer) = select_release_asset(&assets);
        assert!(chosen.is_some());
        let chosen = chosen.unwrap();
        assert_eq!(chosen.name, "Local.LLM.Panel_1.9.7_x64-setup.exe");
        assert!(is_installer);
        assert!(sha.is_some());
        assert_eq!(
            sha.unwrap().name,
            "Local.LLM.Panel_1.9.7_x64-setup.exe.sha256"
        );
    }

    #[test]
    fn test_select_release_asset_fallback_standalone() {
        let assets = vec![
            GithubReleaseAsset {
                name: "LocalLLMPanel-v1.9.7.exe".into(),
                size: 36000000,
                browser_download_url: "https://.../standalone.exe".into(),
            },
            GithubReleaseAsset {
                name: "LocalLLMPanel-v1.9.7.exe.sha256".into(),
                size: 91,
                browser_download_url: "https://.../standalone.sha256".into(),
            },
        ];

        let (chosen, sha, is_installer) = select_release_asset(&assets);
        assert!(chosen.is_some());
        assert_eq!(chosen.unwrap().name, "LocalLLMPanel-v1.9.7.exe");
        assert!(!is_installer);
        assert!(sha.is_some());
        assert_eq!(sha.unwrap().name, "LocalLLMPanel-v1.9.7.exe.sha256");
    }

    #[test]
    fn test_generated_updater_arguments_and_paths() {
        let params = UpdaterScriptParams {
            installer_path: PathBuf::from(r"C:\Custom Temp\Local.LLM.Panel_2.1.3_x64-setup.exe"),
            app_pid: 12345,
            log_path: PathBuf::from(r"C:\Custom Temp\update-apply.log"),
            status_path: PathBuf::from(r"C:\Custom Temp\update-status.json"),
            install_dir: PathBuf::from(r"C:\Program Files\Local LLM Panel"),
            target_executable: PathBuf::from(r"C:\Program Files\Local LLM Panel\llm-panel.exe"),
            expected_version: "2.1.3".to_string(),
            skip_ui: true,
        };

        let script = generate_updater_script(&params);

        assert!(script.contains(r"C:\Custom Temp\Local.LLM.Panel_2.1.3_x64-setup.exe"));
        assert!(script.contains(r"C:\Custom Temp\update-apply.log"));
        assert!(script.contains(r"C:\Custom Temp\update-status.json"));
        assert!(script.contains(r"C:\Program Files\Local LLM Panel"));
        assert!(script.contains(r"C:\Program Files\Local LLM Panel\llm-panel.exe"));
        assert!(script.contains("2.1.3"));
        assert!(script.contains("12345"));

        assert!(script.contains("'/S'"));
        assert!(script.contains(r#""/D=$installDir""#));
        assert!(script.contains("$installerExitCode = $proc.ExitCode"));
        assert!(script.contains("Fail-Update \"Installer failed with non-zero exit code: $installerExitCode\""));
        assert!(script.contains("Test-Path -LiteralPath $targetExe -PathType Leaf"));
        assert!(script.contains("Normalize-Ver $expectedVersion"));
        assert!(script.contains("Normalize-Ver $prodVersion"));
        assert!(script.contains("Normalize-Ver $fileVersion"));

        assert!(!script.contains("$candidates.Add($appExe)"));
    }

    #[cfg(target_os = "windows")]
    fn run_powershell_script(script_path: &PathBuf) -> std::process::Output {
        std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
            .arg(script_path)
            .env("LOCALLLMPANEL_UPDATE_NO_UI", "1")
            .env("LOCALLLMPANEL_UPDATE_NO_RELAUNCH", "1")
            .output()
            .expect("powershell execution failed")
    }

    #[cfg(target_os = "windows")]
    fn read_status_json(status_path: &PathBuf) -> serde_json::Value {
        let raw = std::fs::read_to_string(status_path).unwrap();
        let clean = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
        serde_json::from_str(clean).unwrap()
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_failed_installer_exit_codes() {
        let test_dir = std::env::temp_dir().join(format!("llm_update_test_fail_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let mock_installer = test_dir.join("mock_installer.cmd");
        std::fs::write(&mock_installer, "@echo off\r\nexit /b 42\r\n").unwrap();

        let log = test_dir.join("update-apply.log");
        let status = test_dir.join("update-status.json");
        let target_exe = test_dir.join("llm-panel.exe");
        let script = test_dir.join("apply.ps1");

        let params = UpdaterScriptParams {
            installer_path: mock_installer.clone(),
            app_pid: 0,
            log_path: log.clone(),
            status_path: status.clone(),
            install_dir: test_dir.clone(),
            target_executable: target_exe.clone(),
            expected_version: "2.1.3".into(),
            skip_ui: true,
        };

        let script_content = generate_updater_script(&params);
        std::fs::write(&script, script_content).unwrap();

        let output = run_powershell_script(&script);
        assert_eq!(output.status.code(), Some(42), "helper must propagate failed installer exit code");

        let status_json = read_status_json(&status);
        assert_eq!(status_json["status"], "failed");
        assert_eq!(status_json["exitCode"], 42);

        let log_content = std::fs::read_to_string(&log).unwrap();
        assert!(log_content.contains("Installer completed with exit code: 42"));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_missing_installed_executable() {
        let test_dir = std::env::temp_dir().join(format!("llm_update_test_missing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let mock_installer = test_dir.join("mock_installer.cmd");
        std::fs::write(&mock_installer, "@echo off\r\nexit /b 0\r\n").unwrap();

        let log = test_dir.join("update-apply.log");
        let status = test_dir.join("update-status.json");
        let target_exe = test_dir.join("nonexistent_llm_panel.exe");
        let script = test_dir.join("apply.ps1");

        let params = UpdaterScriptParams {
            installer_path: mock_installer.clone(),
            app_pid: 0,
            log_path: log.clone(),
            status_path: status.clone(),
            install_dir: test_dir.clone(),
            target_executable: target_exe.clone(),
            expected_version: "2.1.3".into(),
            skip_ui: true,
        };

        let script_content = generate_updater_script(&params);
        std::fs::write(&script, script_content).unwrap();

        let output = run_powershell_script(&script);
        assert_eq!(output.status.code(), Some(4), "helper must exit 4 when installed executable is missing");

        let status_json = read_status_json(&status);
        assert_eq!(status_json["status"], "failed");
        assert_eq!(status_json["exitCode"], 4);

        let log_content = std::fs::read_to_string(&log).unwrap();
        assert!(log_content.contains("Installed executable not found"));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_installed_version_mismatch() {
        let test_dir = std::env::temp_dir().join(format!("llm_update_test_mismatch_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let sys_exe = PathBuf::from(r"C:\Windows\System32\cmd.exe");
        let target_exe = test_dir.join("llm-panel.exe");
        std::fs::copy(&sys_exe, &target_exe).unwrap();

        let mock_installer = test_dir.join("mock_installer.cmd");
        std::fs::write(&mock_installer, "@echo off\r\nexit /b 0\r\n").unwrap();

        let log = test_dir.join("update-apply.log");
        let status = test_dir.join("update-status.json");
        let script = test_dir.join("apply.ps1");

        let params = UpdaterScriptParams {
            installer_path: mock_installer.clone(),
            app_pid: 0,
            log_path: log.clone(),
            status_path: status.clone(),
            install_dir: test_dir.clone(),
            target_executable: target_exe.clone(),
            expected_version: "99.0.0".into(),
            skip_ui: true,
        };

        let script_content = generate_updater_script(&params);
        std::fs::write(&script, script_content).unwrap();

        let output = run_powershell_script(&script);
        assert_eq!(output.status.code(), Some(5), "helper must exit 5 on version mismatch");

        let status_json = read_status_json(&status);
        assert_eq!(status_json["status"], "failed");
        assert_eq!(status_json["exitCode"], 5);

        let log_content = std::fs::read_to_string(&log).unwrap();
        assert!(log_content.contains("Installed executable version does not match expected version"));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_successful_installation_and_relaunch() {
        let test_dir = std::env::temp_dir().join(format!("llm_update_test_success_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let sys_exe = PathBuf::from(r"C:\Windows\System32\cmd.exe");
        let target_exe = test_dir.join("llm-panel.exe");
        std::fs::copy(&sys_exe, &target_exe).unwrap();

        let out = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-Command"])
            .arg(format!("(Get-Item '{}').VersionInfo.ProductVersion", target_exe.display()))
            .output()
            .unwrap();
        let raw_ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(!raw_ver.is_empty());

        let mock_installer = test_dir.join("mock_installer.cmd");
        std::fs::write(&mock_installer, "@echo off\r\nexit /b 0\r\n").unwrap();

        let log = test_dir.join("update-apply.log");
        let status = test_dir.join("update-status.json");
        let script = test_dir.join("apply.ps1");

        let params = UpdaterScriptParams {
            installer_path: mock_installer.clone(),
            app_pid: 0,
            log_path: log.clone(),
            status_path: status.clone(),
            install_dir: test_dir.clone(),
            target_executable: target_exe.clone(),
            expected_version: raw_ver,
            skip_ui: true,
        };

        let script_content = generate_updater_script(&params);
        std::fs::write(&script, script_content).unwrap();

        let output = run_powershell_script(&script);
        assert_eq!(output.status.code(), Some(0), "helper must succeed and exit 0 when version matches");

        let status_json = read_status_json(&status);
        assert_eq!(status_json["status"], "success");
        assert_eq!(status_json["exitCode"], 0);

        let log_content = std::fs::read_to_string(&log).unwrap();
        assert!(log_content.contains("Version verification successful"));

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
