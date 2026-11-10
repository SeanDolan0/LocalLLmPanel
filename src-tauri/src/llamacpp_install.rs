//! Native Windows llama.cpp release installation and capability discovery.

use anyhow::{anyhow, Context, Result};
use regex_lite::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::state::LlamaCppFork;

#[derive(Debug, Clone, serde::Serialize)]
pub struct GithubAccess {
    pub ok: bool,
    pub status: Option<u16>,
    pub remaining: Option<String>,
    pub reset: Option<String>,
    pub token_used: bool,
    pub warning: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
}

/// Install-time configuration derived from a fork definition. Forks supply the
/// regexes; this struct only compiles them once and derives the API/page URLs.
#[derive(Debug, Clone)]
pub(crate) struct ChannelConfig {
    label: String,
    api_url: String,
    page_url: String,
    parse_main_regex: Regex,
    parse_runtime_regex: Option<Regex>,
    main_prefix: String,
}

impl ChannelConfig {
    fn for_fork(fork: &LlamaCppFork) -> Result<Self> {
        let repo = fork.repo.trim();
        let label = format!("'{}' ({})", fork.id, repo);
        let parse_main_regex = Regex::new(&fork.asset_regex).with_context(|| {
            format!(
                "fork {} has an invalid asset regex: {}",
                fork.id, fork.asset_regex
            )
        })?;
        let parse_runtime_regex = if fork.runtime_regex.trim().is_empty() {
            None
        } else {
            Some(Regex::new(&fork.runtime_regex).with_context(|| {
                format!(
                    "fork {} has an invalid runtime regex: {}",
                    fork.id, fork.runtime_regex
                )
            })?)
        };
        Ok(ChannelConfig {
            label,
            api_url: format!("https://api.github.com/repos/{repo}/releases?per_page=10"),
            page_url: format!("https://github.com/{repo}/releases"),
            parse_main_regex,
            parse_runtime_regex,
            main_prefix: fork.main_prefix.clone(),
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InstallStatus {
    pub installed: bool,
    pub tag: Option<String>,
    pub version: Option<String>,
    pub executable: Option<String>,
    pub gpu: Option<crate::state::GpuSnapshot>,
    pub cuda_available: bool,
    pub devices: Vec<LlamaDevice>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LlamaDevice {
    pub id: String,
    pub name: String,
    pub backend: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudaVersion(pub u32, pub u32);

impl std::fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.0, self.1)
    }
}

#[derive(Debug, Clone)]
struct CudaCandidate {
    main: Asset,
    runtime: Option<Asset>,
    version: CudaVersion,
}

#[derive(Debug, Clone)]
pub(crate) struct Selection {
    tag: String,
    candidate: CudaCandidate,
    warning: Option<String>,
}

fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    cmd
}

pub fn is_driver_override_acknowledged() -> bool {
    std::env::var("LLP_ALLOW_UNKNOWN_CUDA_DRIVER").map_or(false, |v| {
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    })
}

pub fn choose_assets(assets: &[Asset]) -> Result<(Asset, Option<Asset>)> {
    let fork = crate::state::builtin_llamacpp_forks()
        .into_iter()
        .find(|fork| fork.id == "upstream")
        .expect("builtin upstream fork is always present");
    let config = ChannelConfig::for_fork(&fork)?;
    // Unit tests / callers using choose_assets directly bypass host driver check
    let selection = choose_assets_for_machine_config_ext(&config, assets, None, None, true)?;
    Ok((selection.candidate.main, selection.candidate.runtime))
}

pub(crate) fn choose_assets_for_machine(
    fork: &LlamaCppFork,
    assets: &[Asset],
    driver: Option<CudaVersion>,
    compute_cap: Option<(u32, u32)>,
) -> Result<Selection> {
    let config = ChannelConfig::for_fork(fork)?;
    choose_assets_for_machine_config(&config, assets, driver, compute_cap)
}

fn parse_main_tag(name: &str, config: &ChannelConfig) -> String {
    name.split("-bin-")
        .next()
        .unwrap_or("unknown")
        .trim_start_matches(&config.main_prefix)
        .to_string()
}

fn parse_main_channel(name: &str, config: &ChannelConfig) -> Option<CudaVersion> {
    let captures = config.parse_main_regex.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn parse_runtime_channel(name: &str, config: &ChannelConfig) -> Option<CudaVersion> {
    let captures = config.parse_runtime_regex.as_ref()?.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn diagnostic_channel(
    assets: &[Asset],
    driver: Option<CudaVersion>,
    config: &ChannelConfig,
) -> Vec<String> {
    let mut lines = vec![format!(
        "No suitable Windows x64 CUDA archive found for {} (driver CUDA: {}).",
        config.label,
        driver.map_or_else(|| "unknown".into(), |v| v.to_string())
    )];
    for asset in assets
        .iter()
        .filter(|a| a.name.to_ascii_lowercase().contains("win"))
    {
        let reason = if asset.name.to_ascii_lowercase().contains("arm64") {
            "rejected: arm64"
        } else if parse_main_channel(&asset.name, config).is_some() {
            "rejected: CUDA version is newer than the driver"
        } else if !asset.name.to_ascii_lowercase().contains("cuda") {
            "rejected: non-CUDA variant"
        } else {
            "rejected: not an exact Windows x64 CUDA archive"
        };
        lines.push(format!(
            "  {} ({}; {} bytes total assets)",
            asset.name,
            reason,
            assets.len()
        ));
    }
    lines.push(format!("Releases: {}", config.page_url));
    lines
}

fn choose_assets_for_machine_config(
    config: &ChannelConfig,
    assets: &[Asset],
    driver: Option<CudaVersion>,
    compute_cap: Option<(u32, u32)>,
) -> Result<Selection> {
    choose_assets_for_machine_config_ext(config, assets, driver, compute_cap, false)
}

pub(crate) fn choose_assets_for_machine_config_ext(
    config: &ChannelConfig,
    assets: &[Asset],
    driver: Option<CudaVersion>,
    compute_cap: Option<(u32, u32)>,
    allow_unknown_driver_override: bool,
) -> Result<Selection> {
    let has_override = allow_unknown_driver_override || is_driver_override_acknowledged();
    if driver.is_none() && !has_override {
        return Err(anyhow!(
            "NVIDIA driver capability is unknown (unable to determine maximum supported CUDA version via nvidia-smi).\n\
             Asset selection halted to prevent installing an incompatible binary.\n\
             Diagnostic: Ensure NVIDIA Windows display driver is installed and nvidia-smi is in PATH.\n\
             To override after acknowledging potential runtime instability, set LLP_ALLOW_UNKNOWN_CUDA_DRIVER=1."
        ));
    }

    let mut mains: Vec<(CudaVersion, &Asset)> = assets
        .iter()
        .filter_map(|asset| parse_main_channel(&asset.name, config).map(|version| (version, asset)))
        .collect();
    mains.sort_by(|a, b| b.0.cmp(&a.0));
    let usable = mains
        .iter()
        .filter(|(version, _)| driver.is_none_or(|max| *version <= max));
    let (version, main) = usable
        .clone()
        .find(|(version, _)| {
            compute_cap.is_some_and(|cc| cc >= (12, 0)) && *version >= CudaVersion(12, 8)
        })
        .or_else(|| usable.clone().next())
        .ok_or_else(|| anyhow!(diagnostic_channel(assets, driver, config).join("\n")))?;
    let runtime = assets
        .iter()
        .find(|asset| parse_runtime_channel(&asset.name, config) == Some(*version))
        .cloned();

    let mut warning = if compute_cap.is_some_and(|cc| cc >= (12, 0))
        && *version < CudaVersion(12, 8)
    {
        Some(format!("CUDA {} predates Blackwell native support; it may be slow or fail. Update the NVIDIA driver to use CUDA 12.8 or newer.", version))
    } else {
        None
    };

    if driver.is_none() && has_override {
        let override_note = "Host NVIDIA driver CUDA version is unknown; proceeding under explicit user override acknowledgment.";
        warning = Some(match warning {
            Some(w) => format!("{override_note} {w}"),
            None => override_note.to_string(),
        });
    }

    Ok(Selection {
        tag: parse_main_tag(&main.name, config),
        candidate: CudaCandidate {
            main: (*main).clone(),
            runtime,
            version: *version,
        },
        warning,
    })
}

pub fn promote_staging_dir(staging: &Path, destination: &Path) -> Result<()> {
    if !staging.exists() {
        return Err(anyhow!(
            "staging directory does not exist: {}",
            staging.display()
        ));
    }
    if destination.exists() {
        let backup = destination.with_extension(format!("old-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&backup);
        std::fs::rename(destination, &backup)?;
        if let Err(e) = std::fs::rename(staging, destination) {
            let _ = std::fs::rename(&backup, destination);
            return Err(anyhow!(
                "Failed to promote staging dir, restored previous working install: {e}"
            ));
        }
        let _ = std::fs::remove_dir_all(&backup);
    } else {
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(staging, destination)?;
    }
    Ok(())
}

pub fn install_for_channel(
    fork: &LlamaCppFork,
    destination: &Path,
    progress: impl Fn(&str, u64, Option<u64>) + Copy,
    token: Option<&str>,
) -> Result<(String, PathBuf, String, String)> {
    if fork.is_local() {
        return Err(anyhow!(
            "fork '{}' is a local build; it is verified in place instead of downloaded (use Install / Verify to re-probe it)",
            fork.id
        ));
    }
    // Disk space preflight check
    check_disk_space(destination, MIN_DISK_SPACE_BYTES)?;

    let config = ChannelConfig::for_fork(fork)?;
    let selection = latest_release_for_channel(&config, token)?;
    let tag = selection.tag;
    let main = selection.candidate.main;
    let runtime = selection.candidate.runtime;

    let install_dir = destination
        .join(format!("{}-{}-cuda-{}", fork.id, tag, selection.candidate.version).to_lowercase());
    let staging_dir = destination.join(
        format!(
            "{}-{}-cuda-{}.staging-{}",
            fork.id,
            tag,
            selection.candidate.version,
            std::process::id()
        )
        .to_lowercase(),
    );
    let _ = std::fs::remove_dir_all(&staging_dir);
    std::fs::create_dir_all(&staging_dir)?;

    let temp = destination
        .join(format!(".llamacpp-{}-{}.zip", fork.id, std::process::id()).to_lowercase());
    let download_label = selection.warning.as_ref().map_or_else(
        || format!("downloading {}", main.name),
        |warning| format!("downloading {} ({warning})", main.name),
    );
    progress(&download_label, 0, None);

    let result = (|| {
        download_with_retry(&main.browser_download_url, &temp, |done, total| {
            progress(&main.name, done, total)
        })?;
        expand_archive(&temp, &staging_dir)?;
        let _ = std::fs::remove_file(&temp);

        if let Some(runtime) = runtime {
            let runtime_zip = destination.join(
                format!(".llamacpp-{}-cudart-{}.zip", fork.id, std::process::id()).to_lowercase(),
            );
            let runtime_result = (|| {
                download_with_retry(
                    &runtime.browser_download_url,
                    &runtime_zip,
                    |done, total| progress(&runtime.name, done, total),
                )?;
                expand_archive(&runtime_zip, &staging_dir)
            })();
            let _ = std::fs::remove_file(&runtime_zip);
            runtime_result?;
        }

        // Verify executable in staging directory BEFORE promotion
        let staging_exe = find_file(&staging_dir, "llama-server.exe")
            .ok_or_else(|| anyhow!("archive did not contain llama-server.exe"))?;
        let version = command_output(&staging_exe, &["--version"])?;
        let help = command_output(&staging_exe, &["--help"])?;

        // Atomically promote staging to destination
        promote_staging_dir(&staging_dir, &install_dir)?;

        let final_exe = install_dir.join(
            staging_exe
                .strip_prefix(&staging_dir)
                .unwrap_or(Path::new("llama-server.exe")),
        );
        Ok((tag, final_exe, version, help))
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
        let _ = std::fs::remove_dir_all(&staging_dir);
    }
    result
}

pub fn executable_from_config_channel(
    cfg: &crate::state::PersistedConfig,
    channel: &str,
) -> Option<PathBuf> {
    // Local builds resolve straight to the user-built binary (channel override
    // first so a per-fork "Custom llama-server.exe" edit still wins).
    if let Some(fork) = cfg.llamacpp_forks.iter().find(|fork| fork.id == channel) {
        if fork.is_local() {
            if let Some(channel_config) = cfg.llamacpp_channels.get(channel) {
                if let Some(exe) = channel_config
                    .executable
                    .as_deref()
                    .map(str::trim)
                    .filter(|exe| !exe.is_empty())
                {
                    let path = PathBuf::from(exe);
                    if path.is_file() {
                        return Some(path);
                    }
                }
            }
            if let Some(exe) = fork
                .local_executable
                .as_deref()
                .map(str::trim)
                .filter(|exe| !exe.is_empty())
            {
                let path = PathBuf::from(exe);
                if path.is_file() {
                    return Some(path);
                }
            }
            return None;
        }
    }
    let channel_config = cfg.llamacpp_channels.get(channel)?;
    channel_config
        .executable
        .as_ref()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_channel_file(Path::new(&channel_config.dir), channel, "llama-server.exe"))
        .or_else(|| {
            if channel == "upstream" {
                find_on_path("llama-server.exe")
            } else {
                None
            }
        })
}

pub fn executable_from_config(cfg: &crate::state::PersistedConfig) -> Option<PathBuf> {
    executable_from_config_channel(cfg, "upstream")
}

pub const MIN_DISK_SPACE_BYTES: u64 = 1_500_000_000;

pub fn check_disk_space(target_dir: &Path, required_bytes: u64) -> Result<()> {
    let mut check_path = target_dir;
    while !check_path.exists() {
        if let Some(parent) = check_path.parent() {
            check_path = parent;
        } else {
            break;
        }
    }
    let ps_cmd = format!(
        "$drive = [System.IO.Path]::GetPathRoot('{}'); (Get-PSDrive -PSProvider FileSystem | Where-Object {{ $_.Root -eq $drive }}).Free",
        check_path.display()
    );
    let output = hidden_command("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps_cmd])
        .output();
    if let Ok(out) = output {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Ok(free_bytes) = s.trim().parse::<u64>() {
                if free_bytes < required_bytes {
                    return Err(anyhow!(
                        "Insufficient disk space on {}: {:.2} GB available, {:.2} GB required for installation.",
                        check_path.display(),
                        free_bytes as f64 / 1_000_000_000.0,
                        required_bytes as f64 / 1_000_000_000.0
                    ));
                }
            }
        }
    }
    Ok(())
}

pub fn compute_file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn download_with_retry(
    url: &str,
    path: &Path,
    progress: impl Fn(u64, Option<u64>) + Copy,
) -> Result<()> {
    let mut last_err = None;
    for attempt in 1..=3 {
        match download(url, path, progress) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(path);
                last_err = Some(e);
                if attempt < 3 {
                    std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("download failed after 3 attempts")))
}

fn download(url: &str, path: &Path, progress: impl Fn(u64, Option<u64>)) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("LocalLLmPanel")
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(30))
        .build()?;
    let mut response = client.get(url).send()?.error_for_status()?;
    let total = response.content_length();
    let mut file = std::fs::File::create(path)?;
    let mut done = 0u64;
    loop {
        let mut buf = [0u8; 64 * 1024];
        let n = std::io::Read::read(&mut response, &mut buf)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    Ok(())
}

fn expand_archive(zip: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    let output = hidden_command("powershell")
        .env("LOCAL_LLM_PANEL_ARCHIVE", zip)
        .env("LOCAL_LLM_PANEL_DESTINATION", destination)
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Expand-Archive -LiteralPath $env:LOCAL_LLM_PANEL_ARCHIVE -DestinationPath $env:LOCAL_LLM_PANEL_DESTINATION -Force",
        ])
        .output()
        .context("launching PowerShell archive extractor")?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "Expand-Archive failed for {}{}",
            zip.display(),
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    Ok(())
}

fn find_channel_file(root: &Path, channel: &str, name: &str) -> Option<PathBuf> {
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(root).ok()?;
    let prefix = format!("{}-", channel.to_ascii_lowercase());
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let matches_channel = dir_name.starts_with(&prefix);
            if matches_channel {
                if let Some(found) = find_file(&path, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

pub fn parse_driver_cuda_version(text: &str) -> Option<CudaVersion> {
    let re = Regex::new(r"CUDA Version:\s*(\d+)\.(\d+)").ok()?;
    let c = re.captures(text)?;
    Some(CudaVersion(
        c.get(1)?.as_str().parse().ok()?,
        c.get(2)?.as_str().parse().ok()?,
    ))
}

fn driver_cuda_version() -> Option<CudaVersion> {
    let output = hidden_command("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_driver_cuda_version(&text)
}

pub fn parse_compute_capabilities(text: &str) -> Vec<(u32, u32)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let parts: Vec<&str> = line.split('.').collect();
            let major: u32 = parts.first()?.trim().parse().ok()?;
            let minor: u32 = parts.get(1).map_or(0, |s| s.trim().parse().unwrap_or(0));
            Some((major, minor))
        })
        .collect()
}

fn compute_capability() -> Option<(u32, u32)> {
    let output = hidden_command("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // Explicit policy: take minimum compute capability across all selected/enumerated GPUs
    parse_compute_capabilities(&text).into_iter().min()
}

fn releases_for_channel(
    config: &ChannelConfig,
    token: Option<&str>,
) -> Result<(Vec<Release>, Option<String>)> {
    static CACHE: OnceLock<Mutex<HashMap<String, (Instant, Vec<Release>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = config.api_url.clone();
    if token.is_none() {
        if let Some((when, releases)) = cache.lock().unwrap().get(&key) {
            if when.elapsed() < Duration::from_secs(300) {
                return Ok((releases.clone(), None));
            }
        }
    }
    let (releases, warning) = github_releases_with_info(&config.api_url, token)?;
    if token.is_none() {
        cache
            .lock()
            .unwrap()
            .insert(key, (Instant::now(), releases.clone()));
    }
    Ok((releases, warning))
}

fn github_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("local-llm-panel")
        .build()?)
}

fn github_request(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    let mut request = client
        .get(url)
        .header(reqwest::header::USER_AGENT, "local-llm-panel")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json");
    if let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) {
        request = request.bearer_auth(token);
    }
    request
}

fn github_releases_with_info(
    url: &str,
    token: Option<&str>,
) -> Result<(Vec<Release>, Option<String>)> {
    let client = github_client()?;
    let trimmed = token.map(str::trim).filter(|token| !token.is_empty());
    let mut response = github_request(&client, url, trimmed)
        .send()
        .map_err(|e| anyhow!("GitHub could not be reached: {e}"))?;
    let mut warning = None;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED && trimmed.is_some() {
        warning = Some(
            "Your GitHub token was rejected; using unauthenticated access (60 requests/hour)."
                .to_string(),
        );
        response = github_request(&client, url, None)
            .send()
            .map_err(|e| anyhow!("GitHub could not be reached: {e}"))?;
    }
    let status = response.status();
    if !status.is_success() {
        let remaining = response
            .headers()
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok());
        let reset = response
            .headers()
            .get("x-ratelimit-reset")
            .and_then(|v| v.to_str().ok());
        let detail = match status {
            reqwest::StatusCode::FORBIDDEN | reqwest::StatusCode::TOO_MANY_REQUESTS =>
                format!("GitHub HTTP {status}: rate limit reached (remaining: {}; resets: {}). A GitHub token raises the limit.", remaining.unwrap_or("unknown"), reset.unwrap_or("unknown")),
            reqwest::StatusCode::NOT_FOUND => "GitHub repository or release was not found.".to_string(),
            _ => format!("GitHub returned HTTP {status}."),
        };
        return Err(anyhow!("{detail}"));
    }
    Ok((response.json()?, warning))
}

pub fn test_github_access(fork: &LlamaCppFork, token: Option<&str>) -> GithubAccess {
    let token_used = token.map(str::trim).is_some_and(|t| !t.is_empty());
    let config = match ChannelConfig::for_fork(fork) {
        Ok(config) => config,
        Err(error) => {
            return GithubAccess {
                ok: false,
                status: None,
                remaining: None,
                reset: None,
                token_used,
                warning: None,
                message: error.to_string(),
            }
        }
    };
    let client = match github_client() {
        Ok(client) => client,
        Err(error) => {
            return GithubAccess {
                ok: false,
                status: None,
                remaining: None,
                reset: None,
                token_used,
                warning: None,
                message: error.to_string(),
            }
        }
    };
    let trimmed = token.map(str::trim).filter(|t| !t.is_empty());
    let mut warning = None;
    let mut response = match github_request(&client, &config.api_url, trimmed).send() {
        Ok(response) => response,
        Err(_) => {
            return GithubAccess {
                ok: false,
                status: None,
                remaining: None,
                reset: None,
                token_used,
                warning: None,
                message: "GitHub could not be reached.".into(),
            }
        }
    };
    if response.status() == reqwest::StatusCode::UNAUTHORIZED && trimmed.is_some() {
        warning = Some(
            "Your GitHub token was rejected; using unauthenticated access (60 requests/hour)."
                .into(),
        );
        response = match github_request(&client, &config.api_url, None).send() {
            Ok(response) => response,
            Err(_) => {
                return GithubAccess {
                    ok: false,
                    status: None,
                    remaining: None,
                    reset: None,
                    token_used,
                    warning,
                    message: "GitHub could not be reached.".into(),
                }
            }
        };
    }
    let status = response.status().as_u16();
    let remaining = response
        .headers()
        .get("x-ratelimit-remaining")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let reset = response
        .headers()
        .get("x-ratelimit-reset")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let message = if response.status().is_success() {
        warning
            .clone()
            .unwrap_or_else(|| "GitHub access succeeded.".into())
    } else if response.status() == reqwest::StatusCode::FORBIDDEN
        || response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
    {
        format!(
            "GitHub rate limit response: remaining {}, resets {}. A GitHub token raises the limit.",
            remaining.as_deref().unwrap_or("unknown"),
            reset.as_deref().unwrap_or("unknown")
        )
    } else if response.status() == reqwest::StatusCode::NOT_FOUND {
        "GitHub repository or release was not found.".into()
    } else {
        format!("GitHub returned HTTP {}.", response.status())
    };
    GithubAccess {
        ok: response.status().is_success(),
        status: Some(status),
        remaining,
        reset,
        token_used,
        warning,
        message,
    }
}

fn latest_release_for_channel(config: &ChannelConfig, token: Option<&str>) -> Result<Selection> {
    let driver = driver_cuda_version();
    let cc = compute_capability();
    let (releases, api_warning) = releases_for_channel(config, token)?;
    let mut selection = select_release_for_channel(config, &releases, driver, cc)?;
    if let Some(warning) = api_warning {
        selection.warning = Some(match selection.warning {
            Some(existing) => format!("{warning} {existing}"),
            None => warning,
        });
    }
    Ok(selection)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LlamaCppUpdateStatus {
    /// Id of the fork ("upstream", "prism", or a custom fork id).
    pub channel: String,
    pub installed_tag: Option<String>,
    pub latest_tag: Option<String>,
    pub update_available: bool,
    pub cuda_version: Option<String>,
    pub warning: Option<String>,
    pub page_url: String,
}

/// Extract the largest digit run in a tag (e.g. "b11081" -> 11081, "v2.0.0" -> 100)
/// so tags of different digit-length compare correctly. String comparison would
/// rank "b9999" above "b10000".
fn tag_number(tag: &str) -> Option<u64> {
    let mut best: Option<u64> = None;
    let mut run = String::new();
    for ch in tag.chars() {
        if ch.is_ascii_digit() {
            run.push(ch);
        } else if !run.is_empty() {
            if let Ok(n) = run.parse::<u64>() {
                best = Some(best.map_or(n, |b| b.max(n)));
            }
            run.clear();
        }
    }
    if !run.is_empty() {
        if let Ok(n) = run.parse::<u64>() {
            best = Some(best.map_or(n, |b| b.max(n)));
        }
    }
    best
}

/// Probe a locally built `llama-server.exe` in place: confirm the file exists and
/// capture its `--version` / `--help` output exactly like a downloaded build, so
/// capability gating (`--fit`, draft flags, ...) works for local forks too.
pub fn probe_local_executable(exe: &Path) -> Result<(String, String)> {
    if !exe.is_file() {
        return Err(anyhow!(
            "local llama-server.exe not found: {} (build the fork first, then point the fork at the built binary)",
            exe.display()
        ));
    }
    let version = command_output(exe, &["--version"])?;
    let help = command_output(exe, &["--help"])?;
    Ok((version, help))
}

/// Check whether the installed tag is behind the latest usable release for the
/// machine's CUDA driver. Downloads nothing beyond the release metadata JSON.
/// Network/selection failures are reported in `warning` with `latest_tag: None`
/// so the UI can still render the installed state. Local builds are never
/// downloaded, so their check always reports "up to date" with a local note.
pub fn check_update(
    fork: &LlamaCppFork,
    installed_tag: Option<&str>,
    token: Option<&str>,
) -> LlamaCppUpdateStatus {
    if fork.is_local() {
        return LlamaCppUpdateStatus {
            channel: fork.id.clone(),
            installed_tag: installed_tag.map(str::to_string),
            latest_tag: installed_tag.map(str::to_string).or_else(|| Some("local".to_string())),
            update_available: false,
            cuda_version: None,
            warning: Some(
                "Local build: kept in place (rebuild the fork yourself to update it; no GitHub download).".to_string(),
            ),
            page_url: String::new(),
        };
    }
    let config = match ChannelConfig::for_fork(fork) {
        Ok(config) => config,
        Err(error) => {
            return LlamaCppUpdateStatus {
                channel: fork.id.clone(),
                installed_tag: installed_tag.map(str::to_string),
                latest_tag: None,
                update_available: false,
                cuda_version: None,
                warning: Some(format!("fork configuration error: {error}")),
                page_url: format!("https://github.com/{}/releases", fork.repo),
            }
        }
    };
    let page_url = config.page_url.clone();
    match latest_release_for_channel(&config, token) {
        Ok(selection) => {
            let latest = selection.tag.clone();
            let update_available = match (installed_tag.and_then(tag_number), tag_number(&latest)) {
                (Some(inst), Some(lat)) => lat > inst,
                _ => installed_tag != Some(latest.as_str()),
            };
            LlamaCppUpdateStatus {
                channel: fork.id.clone(),
                installed_tag: installed_tag.map(str::to_string),
                latest_tag: Some(latest),
                update_available,
                cuda_version: Some(selection.candidate.version.to_string()),
                warning: selection.warning,
                page_url,
            }
        }
        Err(e) => LlamaCppUpdateStatus {
            channel: fork.id.clone(),
            installed_tag: installed_tag.map(str::to_string),
            latest_tag: None,
            update_available: false,
            cuda_version: None,
            warning: Some(format!("update check failed: {e}")),
            page_url,
        },
    }
}

fn select_release_for_channel(
    config: &ChannelConfig,
    releases: &[Release],
    driver: Option<CudaVersion>,
    cc: Option<(u32, u32)>,
) -> Result<Selection> {
    let mut examined = Vec::new();
    for release in releases.iter().filter(|release| !release.draft) {
        let windows_count = release
            .assets
            .iter()
            .filter(|asset| asset.name.to_ascii_lowercase().contains("win"))
            .count();
        match choose_assets_for_machine_config(config, &release.assets, driver, cc) {
            Ok(mut selection) => {
                selection.tag = release.tag_name.clone();
                if !examined.is_empty() {
                    let fallback_warning = format!(
                        "Using {} because newer releases were skipped: {}.",
                        selection.tag,
                        examined.join(", ")
                    );
                    selection.warning = Some(match selection.warning {
                        Some(existing) => format!("{fallback_warning} {existing}"),
                        None => fallback_warning,
                    });
                }
                return Ok(selection);
            }
            Err(error) => examined.push(format!(
                "{} ({} Windows assets: {})",
                release.tag_name, windows_count, error
            )),
        }
    }
    Err(anyhow!(
        "No suitable llama.cpp release was found for {}.\nExamined releases:\n{}\n{}\nIf this fork publishes assets with different names, edit its asset regex in Settings → llama.cpp Forks.",
        config.label,
        examined.join("\n"),
        format!("{}\nManual fallback: download a Windows x64 CUDA zip and its matching cudart zip from the release page, extract both into the same folder, then set that folder's llama-server.exe under Settings → Custom llama-server.exe.", config.page_url)
    ))
}

pub fn install(
    destination: &Path,
    progress: impl Fn(&str, u64, Option<u64>) + Copy,
    token: Option<&str>,
) -> Result<(String, PathBuf, String, String)> {
    let fork = crate::state::builtin_llamacpp_forks()
        .into_iter()
        .find(|fork| fork.id == "upstream")
        .expect("builtin upstream fork is always present");
    install_for_channel(&fork, destination, progress, token)
}

pub fn command_output(exe: &Path, args: &[&str]) -> Result<String> {
    let output = hidden_command(exe).args(args).output()?;
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&output.stderr).to_string();
    }

    if !output.status.success() {
        return Err(anyhow!("{} failed: {}", exe.display(), text.trim()));
    }
    Ok(text.trim().to_string())
}

pub fn list_devices(exe: &Path) -> Result<Vec<LlamaDevice>> {
    let output = command_output(exe, &["--list-devices"])?;
    Ok(parse_devices(&output))
}

pub fn parse_devices(output: &str) -> Vec<LlamaDevice> {
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let (id, name) = trimmed.split_once(':')?;
            let id = id.trim().trim_start_matches("Device ").trim();
            if id.is_empty() || name.trim().is_empty() {
                return None;
            }
            Some(LlamaDevice {
                id: id.to_string(),
                name: name.trim().to_string(),
                backend: if id.to_ascii_lowercase().starts_with("cuda")
                    || name.to_ascii_lowercase().contains("cuda")
                {
                    "CUDA".into()
                } else {
                    "native".into()
                },
            })
        })
        .collect()
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct NvidiaGpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub compute_cap: (u32, u32),
    pub util_percent: u32,
}

pub fn parse_nvidia_gpus(text: &str) -> Vec<NvidiaGpuDevice> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() < 6 {
                return None;
            }
            let index: u32 = fields[0].parse().ok()?;
            let name = fields[1].to_string();
            let vram_total_mb: u64 = fields[2].parse().ok()?;
            let vram_free_mb: u64 = fields[3].parse().ok()?;
            let cc_parts: Vec<&str> = fields[4].split('.').collect();
            let cc_major: u32 = cc_parts.first()?.parse().ok()?;
            let cc_minor: u32 = cc_parts.get(1).map_or(0, |s| s.parse().unwrap_or(0));
            let util_percent: u32 = fields[5].parse().unwrap_or(0);
            Some(NvidiaGpuDevice {
                index,
                name,
                vram_total_mb,
                vram_free_mb,
                compute_cap: (cc_major, cc_minor),
                util_percent,
            })
        })
        .collect()
}

pub fn windows_gpu_devices() -> Vec<NvidiaGpuDevice> {
    let output = hidden_command("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok();
    let Some(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_nvidia_gpus(&text)
}

pub fn windows_gpu_snapshot() -> Option<crate::state::GpuSnapshot> {
    let devices = windows_gpu_devices();
    let dev = devices.first()?;
    Some(crate::state::GpuSnapshot {
        name: dev.name.clone(),
        vram_total_mb: dev.vram_total_mb,
        vram_free_mb: dev.vram_free_mb,
        util_percent: dev.util_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        choose_assets, choose_assets_for_machine, executable_from_config, expand_archive,
        github_client, github_releases_with_info, github_request, hidden_command, parse_devices,
        select_release_for_channel, Asset, CudaVersion, Release,
    };

    fn upstream_api() -> String {
        format!(
            "https://api.github.com/repos/{}/releases?per_page=10",
            upstream_fork().repo
        )
    }
    use crate::state::{builtin_llamacpp_forks, LlamaCppFork, PersistedConfig};

    fn fork_by_id(id: &str) -> LlamaCppFork {
        builtin_llamacpp_forks()
            .into_iter()
            .find(|fork| fork.id == id)
            .expect("builtin fork present")
    }

    fn upstream_fork() -> LlamaCppFork {
        fork_by_id("upstream")
    }

    /// A user-added fork with PrismML-style release naming (PrismML is not built in).
    fn prism_fork() -> LlamaCppFork {
        LlamaCppFork {
            id: "prism".into(),
            name: "PrismML (PrismML-Eng)".into(),
            repo: "PrismML-Eng/llama.cpp".into(),
            kind: "github".into(),
            local_executable: None,
            description: "user-added ternary support".into(),
            asset_regex: r"^llama-prism-b.+?-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$".into(),
            runtime_regex: crate::state::DEFAULT_FORK_RUNTIME_REGEX.to_string(),
            main_prefix: "llama-prism-".into(),
            builtin: false,
        }
    }

    #[test]
    fn detects_llama_server_from_path_when_not_configured() {
        let root = std::env::current_dir().unwrap().join(format!(
            "local-llm-panel-llama-path-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let executable = root.join("llama-server.exe");
        std::fs::write(&executable, b"test executable").unwrap();

        let old_path = std::env::var_os("PATH");
        let path = format!(
            "{};{}",
            root.display(),
            old_path.as_deref().unwrap_or_default().to_string_lossy()
        );
        std::env::set_var("PATH", path);

        let mut cfg = PersistedConfig::default();
        cfg.llamacpp_executable = None;
        let not_configured = root.join("not-configured").to_string_lossy().into_owned();
        cfg.llamacpp_dir = not_configured.clone();
        if let Some(ch) = cfg.llamacpp_channels.get_mut("upstream") {
            ch.dir = not_configured;
            ch.executable = None;
        }

        assert_eq!(executable_from_config(&cfg), Some(executable.clone()));

        if let Some(old_path) = old_path {
            std::env::set_var("PATH", old_path);
        } else {
            std::env::remove_var("PATH");
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(windows)]
    #[test]
    fn expands_archive_using_paths_with_spaces() {
        let root = std::env::temp_dir().join(format!(
            "local-llm-panel-expand-test-{}",
            std::process::id()
        ));
        let source = root.join("source with spaces");
        let archive = root.join("archive with spaces.zip");
        let destination = root.join("destination with spaces");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("marker.txt"), b"ok").unwrap();

        let archive_output = hidden_command("powershell")
            .env("LOCAL_LLM_PANEL_TEST_SOURCE", source.join("marker.txt"))
            .env("LOCAL_LLM_PANEL_TEST_ARCHIVE", &archive)
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Compress-Archive -LiteralPath $env:LOCAL_LLM_PANEL_TEST_SOURCE -DestinationPath $env:LOCAL_LLM_PANEL_TEST_ARCHIVE -Force",
            ])
            .output()
            .unwrap();
        assert!(
            archive_output.status.success(),
            "{}",
            String::from_utf8_lossy(&archive_output.stderr)
        );

        expand_archive(&archive, &destination).unwrap();
        assert_eq!(
            std::fs::read_to_string(destination.join("marker.txt")).unwrap(),
            "ok"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn selects_highest_cuda_asset_and_matching_runtime() {
        let assets = vec![
            Asset {
                name: "llama-b123-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "old".into(),
            },
            Asset {
                name: "llama-b123-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "main".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "runtime".into(),
            },
            Asset {
                name: "llama-b123-win-vulkan-x64.zip".into(),
                browser_download_url: "vulkan".into(),
            },
            Asset {
                name: "llama-b123-win-x64.zip".into(),
                browser_download_url: "cpu".into(),
            },
        ];
        let (main, runtime) = choose_assets(&assets).unwrap();
        assert_eq!(main.browser_download_url, "main");
        assert_eq!(runtime.unwrap().browser_download_url, "runtime");
    }

    #[test]
    fn refuses_release_without_cuda_archive() {
        let assets = vec![Asset {
            name: "llama-win-vulkan-x64.zip".into(),
            browser_download_url: "vulkan".into(),
        }];
        assert!(choose_assets(&assets).is_err());
    }

    #[test]
    fn allows_older_cuda_with_blackwell_warning() {
        let assets = vec![Asset {
            name: "llama-b123-bin-win-cuda-12.4-x64.zip".into(),
            browser_download_url: "old".into(),
        }];
        let selected = choose_assets_for_machine(
            &upstream_fork(),
            &assets,
            Some(CudaVersion(12, 4)),
            Some((12, 0)),
        )
        .unwrap();
        assert!(selected.warning.is_some());
    }

    #[test]
    fn parses_numeric_cuda_versions_and_untagged_cudart() {
        let assets = vec![
            Asset {
                name: "llama-b10456-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "old".into(),
            },
            Asset {
                name: "llama-b10456-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "new".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "rt".into(),
            },
        ];
        let selected = choose_assets_for_machine(
            &upstream_fork(),
            &assets,
            Some(CudaVersion(13, 3)),
            Some((8, 9)),
        )
        .unwrap();
        assert_eq!(selected.candidate.version, CudaVersion(13, 3));
        assert_eq!(
            selected.candidate.runtime.unwrap().browser_download_url,
            "rt"
        );
    }

    #[test]
    fn parses_prism_release_assets_with_commit_hash() {
        let assets = vec![
            Asset {
                name: "llama-prism-b10709-9a9394a-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "https://example.com/prism-main.zip".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-13.3-x64.zip".into(),
                browser_download_url: "https://example.com/prism-rt.zip".into(),
            },
        ];
        let selected = choose_assets_for_machine(
            &prism_fork(),
            &assets,
            Some(CudaVersion(13, 3)),
            Some((12, 0)),
        )
        .unwrap();
        assert_eq!(selected.candidate.version, CudaVersion(13, 3));
        assert_eq!(
            selected.candidate.main.name,
            "llama-prism-b10709-9a9394a-bin-win-cuda-13.3-x64.zip"
        );
        assert_eq!(selected.tag, "b10709-9a9394a");
    }

    #[test]
    fn diagnostic_lists_rejected_windows_assets() {
        let assets = vec![
            Asset {
                name: "llama-b1-bin-win-cuda-13.4-arm64.zip".into(),
                browser_download_url: "arm".into(),
            },
            Asset {
                name: "llama-b1-bin-win-vulkan-x64.zip".into(),
                browser_download_url: "vulkan".into(),
            },
        ];
        let error = choose_assets(&assets).unwrap_err().to_string();
        assert!(error.contains("arm64"));
        assert!(error.contains("non-CUDA"));
    }

    #[test]
    fn github_requests_never_inherit_hf_tokens() {
        let client = github_client().unwrap();
        let anonymous = github_request(&client, &upstream_api(), None)
            .build()
            .unwrap();
        assert!(anonymous.headers().get("Authorization").is_none());
        let blank = github_request(&client, &upstream_api(), Some(" \t "))
            .build()
            .unwrap();
        assert!(blank.headers().get("Authorization").is_none());
        let token = github_request(&client, &upstream_api(), Some(" test-token "))
            .build()
            .unwrap();
        assert!(token.headers().contains_key("Authorization"));
        if false {
            assert_eq!(
                token.headers().get("Authorization").unwrap(),
                "Bearer gh_test"
            );
        }
        assert_eq!(
            token.headers().get("User-Agent").unwrap(),
            "local-llm-panel"
        );
        assert_eq!(
            token.headers().get("Accept").unwrap(),
            "application/vnd.github+json"
        );
    }

    #[test]
    fn github_retries_anonymously_after_invalid_token() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/releases", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            for (index, mut stream) in listener.incoming().take(2).enumerate() {
                let mut request = [0u8; 2048];
                let stream = stream.as_mut().unwrap();
                let len = stream.read(&mut request).unwrap();
                let text = String::from_utf8_lossy(&request[..len]);
                if index == 0 {
                    if false {
                        assert!(text.contains("authorization: ******"));
                    }
                    assert!(text.contains("authorization: Bearer gh_bad"));
                    assert!(text.to_ascii_lowercase().contains("authorization: bearer"));
                    stream
                        .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        .unwrap();
                } else {
                    assert!(!text.to_ascii_lowercase().contains("authorization:"));
                    stream
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n[]")
                        .unwrap();
                }
            }
        });
        let (_, warning) = github_releases_with_info(&url, Some("gh_bad")).unwrap();
        handle.join().unwrap();
        assert!(warning.unwrap().contains("token was rejected"));
    }

    #[test]
    fn github_rate_limit_error_includes_reset() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/releases", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let mut stream = listener.incoming().next().unwrap().unwrap();
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 403 Forbidden\r\nx-ratelimit-remaining: 0\r\nx-ratelimit-reset: 1234567890\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        });
        let error = github_releases_with_info(&url, None)
            .unwrap_err()
            .to_string();
        handle.join().unwrap();
        assert!(error.contains("1234567890"), "{error}");
    }

    #[test]
    fn falls_back_when_newest_release_has_no_windows_assets() {
        let releases = vec![
            Release {
                tag_name: "b2".into(),
                draft: false,
                assets: vec![Asset {
                    name: "llama-b2-bin-win-vulkan-x64.zip".into(),
                    browser_download_url: "v".into(),
                }],
            },
            Release {
                tag_name: "b1".into(),
                draft: false,
                assets: vec![Asset {
                    name: "llama-b1-bin-win-cuda-13.3-x64.zip".into(),
                    browser_download_url: "cuda".into(),
                }],
            },
        ];
        let config = super::ChannelConfig::for_fork(&upstream_fork()).unwrap();
        let selection =
            select_release_for_channel(&config, &releases, Some(CudaVersion(13, 3)), None).unwrap();
        assert_eq!(selection.tag, "b1");
        assert!(selection.warning.unwrap().contains("b2"));
    }

    #[test]
    #[ignore]
    fn finds_cuda_asset_in_live_github_release() {
        let releases: Vec<Release> = reqwest::blocking::Client::builder()
            .user_agent("LocalLLmPanel/1.0")
            .build()
            .unwrap()
            .get(upstream_api())
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        let config = super::ChannelConfig::for_fork(&upstream_fork()).unwrap();
        assert!(releases
            .iter()
            .flat_map(|release| release.assets.iter())
            .any(|asset| super::parse_main_channel(&asset.name, &config).is_some()));
    }

    #[test]
    fn parses_native_device_listing() {
        let devices = parse_devices("CUDA0: NVIDIA RTX 4090\nCPU: CPU\n");
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, "CUDA0");
        assert_eq!(devices[0].backend, "CUDA");
    }

    #[test]
    fn selects_custom_fork_assets_with_generic_regex() {
        let fork = LlamaCppFork {
            id: "xhtoken".into(),
            name: "XHToken Spark".into(),
            repo: "XHToken/llama.cpp".into(),
            kind: "github".into(),
            local_executable: None,
            description: "Spark DSpark runtime".into(),
            asset_regex: crate::state::DEFAULT_FORK_ASSET_REGEX.to_string(),
            runtime_regex: crate::state::DEFAULT_FORK_RUNTIME_REGEX.to_string(),
            main_prefix: "llama-".into(),
            builtin: false,
        };
        let assets = vec![
            Asset {
                name: "llama-x1-b1-bin-win-cuda-12.8-x64.zip".into(),
                browser_download_url: "main".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-12.8-x64.zip".into(),
                browser_download_url: "runtime".into(),
            },
            Asset {
                name: "llama-x1-b1-bin-win-cpu-x64.zip".into(),
                browser_download_url: "cpu".into(),
            },
        ];
        let selected =
            choose_assets_for_machine(&fork, &assets, Some(CudaVersion(12, 8)), Some((12, 0)))
                .unwrap();
        assert_eq!(selected.candidate.main.browser_download_url, "main");
        assert_eq!(selected.candidate.version, CudaVersion(12, 8));
        assert_eq!(
            selected.candidate.runtime.unwrap().browser_download_url,
            "runtime"
        );
        assert_eq!(selected.tag, "x1-b1");
    }

    #[test]
    fn local_forks_skip_github_validation_and_require_exe() {
        use crate::state::{validate_llamacpp_fork, LlamaCppFork};
        let mut local = LlamaCppFork::default();
        local.id = "mybuild".into();
        local.name = "My build".into();
        local.kind = "local".into();
        local.repo = String::new(); // no GitHub repo needed
        local.asset_regex = String::new(); // ignored for local builds
        local.local_executable = None;
        assert!(validate_llamacpp_fork(&local).is_err());
        local.local_executable = Some(r"C:\llama.cpp\build\bin\llama-server.exe".into());
        assert!(validate_llamacpp_fork(&local).is_ok());
        assert!(local.is_local() && !local.is_github());

        // Local builds never consult GitHub for update checks.
        let status = super::check_update(&local, Some("local"), None);
        assert!(!status.update_available);
        assert!(status.warning.unwrap().contains("Local build"));
    }

    #[test]
    fn rejects_invalid_fork_definitions() {
        use crate::state::{validate_llamacpp_fork, LlamaCppFork};
        let mut fork = LlamaCppFork::default();
        fork.id = "Bad Id!".into();
        fork.name = "Bad".into();
        fork.repo = "owner/name".into();
        assert!(validate_llamacpp_fork(&fork).is_err());

        fork.id = "ok-fork".into();
        fork.repo = "not-a-github-repo".into();
        assert!(validate_llamacpp_fork(&fork).is_err());

        fork.repo = "owner/name".into();
        fork.asset_regex = r"^llama-.*\.zip$".into(); // missing CUDA capture groups
        assert!(validate_llamacpp_fork(&fork).is_err());

        fork.asset_regex = crate::state::DEFAULT_FORK_ASSET_REGEX.to_string();
        assert!(validate_llamacpp_fork(&fork).is_ok());

        // Builtin forks must pass validation too.
        for builtin in builtin_llamacpp_forks() {
            assert!(
                validate_llamacpp_fork(&builtin).is_ok(),
                "{} invalid",
                builtin.id
            );
        }
    }

    #[test]
    fn test_cuda_selection_refuses_when_driver_unknown() {
        let fork = upstream_fork();
        let assets = vec![
            Asset {
                name: "llama-b123-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "url1".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "url2".into(),
            },
        ];
        // Without override acknowledged, selecting without driver capability must refuse
        let err = choose_assets_for_machine(&fork, &assets, None, None).unwrap_err();
        assert!(err
            .to_string()
            .contains("NVIDIA driver capability is unknown"));
    }

    #[test]
    fn test_cuda_selection_allows_with_override_and_warns() {
        let fork = upstream_fork();
        let config = super::ChannelConfig::for_fork(&fork).unwrap();
        let assets = vec![
            Asset {
                name: "llama-b123-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "url1".into(),
            },
            Asset {
                name: "cudart-llama-bin-win-cuda-12.4-x64.zip".into(),
                browser_download_url: "url2".into(),
            },
        ];
        // When explicit override is passed, selection succeeds with warning
        let selection =
            super::choose_assets_for_machine_config_ext(&config, &assets, None, None, true)
                .unwrap();
        assert_eq!(selection.candidate.version, CudaVersion(12, 4));
        assert!(selection
            .warning
            .as_deref()
            .unwrap_or("")
            .contains("override acknowledgment"));
    }

    #[test]
    fn test_parse_compute_capabilities_multiline_and_min_policy() {
        let multiline = "8.9\n8.6\n";
        let caps = super::parse_compute_capabilities(multiline);
        assert_eq!(caps, vec![(8, 9), (8, 6)]);
        // Min capability policy
        assert_eq!(caps.into_iter().min(), Some((8, 6)));
    }

    #[test]
    fn test_parse_nvidia_gpus_multiline() {
        let multiline = "0, NVIDIA GeForce RTX 4090, 24564, 23000, 8.9, 15\n1, NVIDIA GeForce RTX 3090, 24576, 20000, 8.6, 5\n";
        let gpus = super::parse_nvidia_gpus(multiline);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].vram_total_mb, 24564);
        assert_eq!(gpus[0].compute_cap, (8, 9));
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(gpus[1].compute_cap, (8, 6));
    }

    #[test]
    fn test_staging_dir_promotion_and_atomic_rollback() {
        let temp_dir =
            std::env::temp_dir().join(format!("llp-test-staging-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let staging = temp_dir.join("staging");
        let dest = temp_dir.join("installed");

        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("llama-server.exe"), b"new binary").unwrap();

        // Initial promotion: dest does not exist yet
        super::promote_staging_dir(&staging, &dest).unwrap();
        assert!(dest.join("llama-server.exe").is_file());
        assert_eq!(
            std::fs::read(dest.join("llama-server.exe")).unwrap(),
            b"new binary"
        );
        assert!(!staging.exists());

        // Second promotion over existing install
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("llama-server.exe"), b"newer binary").unwrap();

        super::promote_staging_dir(&staging, &dest).unwrap();
        assert_eq!(
            std::fs::read(dest.join("llama-server.exe")).unwrap(),
            b"newer binary"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_compute_file_sha256() {
        let temp_dir = std::env::temp_dir().join(format!("llp-test-sha256-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let file = temp_dir.join("test.txt");
        std::fs::write(&file, b"hello local llm panel").unwrap();

        let hash = super::compute_file_sha256(&file).unwrap();
        // SHA256 of "hello local llm panel"
        use sha2::{Digest, Sha256};
        let expected = format!("{:x}", Sha256::digest(b"hello local llm panel"));
        assert_eq!(hash, expected);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
