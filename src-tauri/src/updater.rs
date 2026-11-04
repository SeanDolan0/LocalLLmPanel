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

    if is_installer {
        // Silent NSIS update: a detached PowerShell helper waits for this
        // process to exit (exiting triggers RunEvent::Exit, which stops every
        // AI server first), runs the installer with /S, then relaunches the
        // installed binary. The UI never needs to stay open for questions.
        #[cfg(target_os = "windows")]
        {
            launch_silent_installer_and_exit(&app, &dest_file)?;
            return Ok(UpdateLaunchResult {
                success: true,
                message: "Update downloaded. Local LLM Panel will close now; the updater will install it silently in the background and relaunch it automatically.".to_string(),
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

#[cfg(target_os = "windows")]
fn launch_silent_installer_and_exit(app: &AppHandle, path: &PathBuf) -> Result<()> {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x00000008;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let temp_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(std::env::temp_dir);
    let script = temp_dir.join(format!("apply-update-{}.ps1", std::process::id()));
    let log = temp_dir.join("update-apply.log");

    // Environment variables carry the parameters so paths with spaces or odd
    // characters never need quoting inside the script.
    let ps = r#"
$ErrorActionPreference = 'SilentlyContinue'
$installer = $env:LOCALLLMPANEL_UPDATE_INSTALLER
$appPid = [int]$env:LOCALLLMPANEL_UPDATE_PID
$log = $env:LOCALLLMPANEL_UPDATE_LOG
$appExe = $env:LOCALLLMPANEL_UPDATE_APP

function Write-Log([string]$msg) {
  "[$(Get-Date -Format o)] $msg" | Out-File $log -Encoding utf8 -Append
}

Write-Log "update helper started; waiting for pid $appPid to exit"
$deadline = (Get-Date).AddSeconds(240)
while ((Get-Process -Id $appPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 400
}
Start-Sleep -Milliseconds 400

if (-not (Test-Path $installer)) {
  Write-Log "installer not found: $installer"
  exit 2
}

Write-Log "app exited; running installer $installer /S"
try {
  $p = Start-Process -FilePath $installer -ArgumentList '/S' -PassThru -Wait -ErrorAction Stop
  Write-Log "installer exit code: $($p.ExitCode)"
  if ($p.ExitCode -ne 0) {
    Write-Log "installer failed; not relaunching the old executable"
    exit $p.ExitCode
  }
} catch {
  Write-Log "installer launch failed: $($_.Exception.Message)"
  exit 3
}

# Locate the installed executable for relaunch: prefer the value the
# installer registered, then fall back to the known per-user install dirs.
$candidates = New-Object System.Collections.Generic.List[string]
$candidates.Add($appExe)
foreach ($hive in 'HKCU:','HKLM:') {
  $reg = Get-ItemProperty "$hive\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Local LLM Panel" -ErrorAction SilentlyContinue
  if ($reg -and $reg.InstallLocation) {
    $candidates.Add((Join-Path $reg.InstallLocation.Trim('"') 'llm-panel.exe'))
  }
}
$candidates.Add("$env:LOCALAPPDATA\Local LLM Panel\llm-panel.exe")
$candidates.Add("$env:LOCALAPPDATA\Programs\Local LLM Panel\llm-panel.exe")
$candidates.Add("${env:ProgramFiles}\Local LLM Panel\llm-panel.exe")
$candidates.Add("${env:ProgramFiles(x86)}\Local LLM Panel\llm-panel.exe")

$relaunched = $false
foreach ($exe in $candidates) {
  if ($exe -and (Test-Path $exe)) {
    try {
      Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) -ErrorAction Stop | Out-Null
      Write-Log "relaunched $exe"
      $relaunched = $true
      break
    } catch {
      Write-Log "failed to relaunch $exe`: $($_.Exception.Message)"
    }
  }
}
if (-not $relaunched) {
  Write-Log "could not locate installed llm-panel.exe; user must start it manually"
}
Remove-Item $MyInvocation.MyCommand.Path -Force
"#;
    std::fs::write(&script, ps).with_context(|| {
        format!(
            "Failed to write update helper script at {}",
            script.display()
        )
    })?;

    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.env("LOCALLLMPANEL_UPDATE_INSTALLER", path.as_os_str())
        .env("LOCALLLMPANEL_UPDATE_PID", std::process::id().to_string())
        .env("LOCALLLMPANEL_UPDATE_LOG", log.as_os_str())
        .env(
            "LOCALLLMPANEL_UPDATE_APP",
            std::env::current_exe().context("Failed to determine the running app path")?,
        )
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

    // Give the helper a moment to start (it waits for this process to die),
    // then exit. app.exit() triggers RunEvent::Exit, whose handler stops all
    // AI servers (vLLM/llama.cpp) before the process disappears.
    std::thread::sleep(std::time::Duration::from_millis(250));
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
}
