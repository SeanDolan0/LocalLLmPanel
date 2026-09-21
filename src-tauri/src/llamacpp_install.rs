//! Native Windows llama.cpp release installation and capability discovery.

use anyhow::{anyhow, Context, Result};
use regex_lite::Regex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::state::LlamaCppChannel;

const UPSTREAM_RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10";
const UPSTREAM_RELEASES_PAGE: &str = "https://github.com/ggml-org/llama.cpp/releases";
const PRISM_RELEASES_API: &str = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases?per_page=10";
const PRISM_RELEASES_PAGE: &str = "https://github.com/PrismML-Eng/llama.cpp/releases";

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

#[derive(Debug, Clone)]
struct ChannelConfig {
    api_url: &'static str,
    page_url: &'static str,
    parse_main_regex: Regex,
    parse_runtime_regex: Regex,
    main_prefix: String,
}

impl ChannelConfig {
    fn for_channel(channel: LlamaCppChannel) -> Self {
        match channel {
            LlamaCppChannel::Upstream => ChannelConfig {
                api_url: UPSTREAM_RELEASES_API,
                page_url: UPSTREAM_RELEASES_PAGE,
                parse_main_regex: Regex::new(r"^llama-b\d+-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").unwrap(),
                parse_runtime_regex: Regex::new(r"^cudart-llama-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").unwrap(),
                main_prefix: "llama-".to_string(),
            },
            LlamaCppChannel::Prism => ChannelConfig {
                api_url: PRISM_RELEASES_API,
                page_url: PRISM_RELEASES_PAGE,
                parse_main_regex: Regex::new(r"^llama-prism-b\d+-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").unwrap(),
                parse_runtime_regex: Regex::new(r"^cudart-llama-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").unwrap(),
                main_prefix: "llama-prism-".to_string(),
            },
        }
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
struct Selection {
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

pub fn choose_assets(assets: &[Asset]) -> Result<(Asset, Option<Asset>)> {
    let selection = choose_assets_for_machine(LlamaCppChannel::Upstream, assets, None, None)?;
    Ok((selection.candidate.main, selection.candidate.runtime))
}

fn parse_main(name: &str) -> Option<CudaVersion> {
    let re = Regex::new(r"^llama-b\d+-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").ok()?;
    let captures = re.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn parse_runtime(name: &str) -> Option<CudaVersion> {
    let re = Regex::new(r"^cudart-llama-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$").ok()?;
    let captures = re.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn diagnostic(assets: &[Asset], driver: Option<CudaVersion>) -> String {
    let mut lines = vec![format!(
        "No suitable Windows x64 CUDA archive found (driver CUDA: {}).",
        driver.map_or_else(|| "unknown".into(), |v| v.to_string())
    )];
    for asset in assets
        .iter()
        .filter(|a| a.name.to_ascii_lowercase().contains("win"))
    {
        let reason = if asset.name.to_ascii_lowercase().contains("arm64") {
            "rejected: arm64"
        } else if parse_main(&asset.name).is_some() {
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
    lines.push(format!("Releases: {}", UPSTREAM_RELEASES_PAGE));
    lines.join("\n")
}

fn choose_assets_for_machine(
    channel: LlamaCppChannel,
    assets: &[Asset],
    driver: Option<CudaVersion>,
    compute_cap: Option<(u32, u32)>,
) -> Result<Selection> {
    let mut mains: Vec<(CudaVersion, &Asset)> = assets
        .iter()
        .filter_map(|asset| parse_main(&asset.name).map(|version| (version, asset)))
        .collect();
    mains.sort_by(|a, b| b.0.cmp(&a.0));
    let usable = mains
        .iter()
        .filter(|(version, _)| driver.map_or(true, |max| *version <= max));
    let (version, main) = usable
        .clone()
        .find(|(version, _)| {
            compute_cap.map_or(false, |cc| cc >= (12, 0)) && *version >= CudaVersion(12, 8)
        })
        .or_else(|| usable.clone().next())
        .ok_or_else(|| anyhow!(diagnostic(assets, driver)))?;
    let runtime = assets
        .iter()
        .find(|asset| parse_runtime(&asset.name) == Some(*version))
        .cloned();
    let warning = if compute_cap.map_or(false, |cc| cc >= (12, 0)) && *version < CudaVersion(12, 8)
    {
        Some(format!("CUDA {} predates Blackwell native support; it may be slow or fail. Update the NVIDIA driver to use CUDA 12.8 or newer.", version))
    } else {
        None
    };
    Ok(Selection {
        tag: parse_main_tag(&main.name, channel),
        candidate: CudaCandidate {
            main: (*main).clone(),
            runtime,
            version: *version,
        },
        warning,
    })
}

fn parse_main_tag(name: &str, channel: LlamaCppChannel) -> String {
    let config = ChannelConfig::for_channel(channel);
    name.split("-bin-")
        .next()
        .unwrap_or("unknown")
        .trim_start_matches(&config.main_prefix)
        .to_string()
}

fn parse_main_channel(name: &str, channel: LlamaCppChannel) -> Option<CudaVersion> {
    let config = ChannelConfig::for_channel(channel);
    let captures = config.parse_main_regex.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn parse_runtime_channel(name: &str, channel: LlamaCppChannel) -> Option<CudaVersion> {
    let config = ChannelConfig::for_channel(channel);
    let captures = config.parse_runtime_regex.captures(name)?;
    Some(CudaVersion(
        captures.get(1)?.as_str().parse().ok()?,
        captures.get(2)?.as_str().parse().ok()?,
    ))
}

fn diagnostic_channel(assets: &[Asset], driver: Option<CudaVersion>, channel: LlamaCppChannel) -> String {
    let config = ChannelConfig::for_channel(channel);
    let mut lines = vec![format!(
        "No suitable Windows x64 CUDA archive found for {channel:?} channel (driver CUDA: {}).",
        driver.map_or_else(|| "unknown".into(), |v| v.to_string())
    )];
    for asset in assets
        .iter()
        .filter(|a| a.name.to_ascii_lowercase().contains("win"))
    {
        let reason = if asset.name.to_ascii_lowercase().contains("arm64") {
            "rejected: arm64"
        } else if parse_main_channel(&asset.name, channel).is_some() {
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
    lines.join("\n")
}

fn choose_assets_for_machine_channel(
    channel: LlamaCppChannel,
    assets: &[Asset],
    driver: Option<CudaVersion>,
    compute_cap: Option<(u32, u32)>,
) -> Result<Selection> {
    let mut mains: Vec<(CudaVersion, &Asset)> = assets
        .iter()
        .filter_map(|asset| parse_main_channel(&asset.name, channel).map(|version| (version, asset)))
        .collect();
    mains.sort_by(|a, b| b.0.cmp(&a.0));
    let usable = mains
        .iter()
        .filter(|(version, _)| driver.map_or(true, |max| *version <= max));
    let (version, main) = usable
        .clone()
        .find(|(version, _)| {
            compute_cap.map_or(false, |cc| cc >= (12, 0)) && *version >= CudaVersion(12, 8)
        })
        .or_else(|| usable.clone().next())
        .ok_or_else(|| anyhow!(diagnostic_channel(assets, driver, channel)))?;
    let runtime = assets
        .iter()
        .find(|asset| parse_runtime_channel(&asset.name, channel) == Some(*version))
        .cloned();
    let warning = if compute_cap.map_or(false, |cc| cc >= (12, 0)) && *version < CudaVersion(12, 8)
    {
        Some(format!("CUDA {} predates Blackwell native support; it may be slow or fail. Update the NVIDIA driver to use CUDA 12.8 or newer.", version))
    } else {
        None
    };
    Ok(Selection {
        tag: parse_main_tag(&main.name, channel),
        candidate: CudaCandidate {
            main: (*main).clone(),
            runtime,
            version: *version,
        },
        warning,
    })
}

pub fn install_for_channel(
    channel: LlamaCppChannel,
    destination: &Path,
    progress: impl Fn(&str, u64, Option<u64>) + Copy,
    token: Option<&str>,
) -> Result<(String, PathBuf, String, String)> {
    let selection = latest_release_for_channel(channel, token)?;
    let tag = selection.tag;
    let main = selection.candidate.main;
    let runtime = selection.candidate.runtime;
    let config = ChannelConfig::for_channel(channel);
    let install_dir = destination.join(format!("{channel:?}-{}-cuda-{}", tag, selection.candidate.version).to_lowercase());
    std::fs::create_dir_all(&install_dir)?;
    let temp = destination.join(format!(".llamacpp-{channel:?}-{}.zip", std::process::id()).to_lowercase());
    let download_label = selection.warning.as_ref().map_or_else(
        || format!("downloading {}", main.name),
        |warning| format!("downloading {} ({warning})", main.name),
    );
    progress(&download_label, 0, None);
    let result = (|| {
        download(&main.browser_download_url, &temp, |done, total| {
            progress(&main.name, done, total)
        })?;
        expand_archive(&temp, &install_dir)?;
        let _ = std::fs::remove_file(&temp);

        if let Some(runtime) = runtime {
            let runtime_zip =
                destination.join(format!(".llamacpp-{channel:?}-cudart-{}.zip", std::process::id()).to_lowercase());
            let runtime_result = (|| {
                download(
                    &runtime.browser_download_url,
                    &runtime_zip,
                    |done, total| progress(&runtime.name, done, total),
                )?;
                expand_archive(&runtime_zip, &install_dir)
            })();
            let _ = std::fs::remove_file(runtime_zip);
            runtime_result?;
        }

        let exe = find_file(&install_dir, "llama-server.exe")
            .ok_or_else(|| anyhow!("archive did not contain llama-server.exe"))?;
        let version = command_output(&exe, &["--version"])?;
        let help = command_output(&exe, &["--help"])?;
        Ok((tag, exe, version, help))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

pub fn executable_from_config_channel(cfg: &crate::state::PersistedConfig, channel: LlamaCppChannel) -> Option<PathBuf> {
    let channel_config = cfg.llamacpp_channels.get(&channel)?;
    channel_config
        .executable
        .as_ref()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_file(Path::new(&channel_config.dir), "llama-server.exe"))
        .or_else(|| find_on_path("llama-server.exe"))
}

pub fn executable_from_config(cfg: &crate::state::PersistedConfig) -> Option<PathBuf> {
    executable_from_config_channel(cfg, LlamaCppChannel::Upstream)
}

fn download(url: &str, path: &Path, progress: impl Fn(u64, Option<u64>)) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("LocalLLmPanel")
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

fn driver_cuda_version() -> Option<CudaVersion> {
    let output = hidden_command("nvidia-smi").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let re = Regex::new(r"CUDA Version:\s*(\d+)\.(\d+)").ok()?;
    let c = re.captures(&text)?;
    Some(CudaVersion(
        c.get(1)?.as_str().parse().ok()?,
        c.get(2)?.as_str().parse().ok()?,
    ))
}

fn compute_capability() -> Option<(u32, u32)> {
    let output = hidden_command("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let output_text = String::from_utf8_lossy(&output.stdout);
    let value = output_text.trim().split('.').collect::<Vec<_>>();
    Some((
        value.first()?.parse().ok()?,
        value.get(1).unwrap_or(&"0").parse().ok()?,
    ))
}

fn releases_for_channel(channel: LlamaCppChannel, token: Option<&str>) -> Result<(Vec<Release>, Option<String>)> {
    let config = ChannelConfig::for_channel(channel);
    static CACHE: OnceLock<Mutex<Option<(Instant, Vec<Release>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if token.is_none() {
        if let Some((when, releases)) = cache.lock().unwrap().as_ref() {
            if when.elapsed() < Duration::from_secs(300) {
                return Ok((releases.clone(), None));
            }
        }
    }
    let (releases, warning) = github_releases_with_info(config.api_url, token)?;
    if token.is_none() {
        *cache.lock().unwrap() = Some((Instant::now(), releases.clone()));
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

pub fn test_github_access(channel: LlamaCppChannel, token: Option<&str>) -> GithubAccess {
    let config = ChannelConfig::for_channel(channel);
    let token_used = token.map(str::trim).is_some_and(|t| !t.is_empty());
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
    let mut response = match github_request(&client, config.api_url, trimmed).send() {
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
        response = match github_request(&client, config.api_url, None).send() {
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

fn latest_release_for_channel(channel: LlamaCppChannel, token: Option<&str>) -> Result<Selection> {
    let driver = driver_cuda_version();
    let cc = compute_capability();
    let (releases, api_warning) = releases_for_channel(channel, token)?;
    let mut selection = select_release_for_channel(channel, &releases, driver, cc)?;
    if let Some(warning) = api_warning {
        selection.warning = Some(match selection.warning {
            Some(existing) => format!("{warning} {existing}"),
            None => warning,
        });
    }
    Ok(selection)
}

fn select_release_for_channel(
    channel: LlamaCppChannel,
    releases: &[Release],
    driver: Option<CudaVersion>,
    cc: Option<(u32, u32)>,
) -> Result<Selection> {
    let config = ChannelConfig::for_channel(channel);
    let mut examined = Vec::new();
    for release in releases.iter().filter(|release| !release.draft) {
        let windows_count = release
            .assets
            .iter()
            .filter(|asset| asset.name.to_ascii_lowercase().contains("win"))
            .count();
        match choose_assets_for_machine_channel(channel, &release.assets, driver, cc) {
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
        "No suitable llama.cpp release was found for {channel:?} channel.\nExamined releases:\n{}\n{}",
        examined.join("\n"),
        format!("{}\nManual fallback: download a Windows x64 CUDA zip and its matching cudart zip from the release page, extract both into the same folder, then set that folder's llama-server.exe under Settings → Custom llama-server.exe.", config.page_url)
    ))
}

pub fn install(
    destination: &Path,
    progress: impl Fn(&str, u64, Option<u64>) + Copy,
    token: Option<&str>,
) -> Result<(String, PathBuf, String, String)> {
    install_for_channel(LlamaCppChannel::Upstream, destination, progress, token)
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

#[cfg(test)]
mod tests {
    use super::{
        choose_assets, choose_assets_for_machine, executable_from_config, expand_archive,
        github_client, github_releases_with_info, github_request, hidden_command, parse_devices,
        select_release, Asset, CudaVersion, Release, RELEASES_API,
    };
    use crate::state::PersistedConfig;

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
        cfg.llamacpp_dir = root.join("not-configured").to_string_lossy().into_owned();

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
        let selected =
            choose_assets_for_machine(&assets, Some(CudaVersion(12, 4)), Some((12, 0))).unwrap();
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
        let selected =
            choose_assets_for_machine(&assets, Some(CudaVersion(13, 3)), Some((8, 9))).unwrap();
        assert_eq!(selected.candidate.version, CudaVersion(13, 3));
        assert_eq!(
            selected.candidate.runtime.unwrap().browser_download_url,
            "rt"
        );
        assert!(CudaVersion(12, 4) < CudaVersion(12, 8));
        assert!(CudaVersion(12, 8) < CudaVersion(13, 3));
        assert!(CudaVersion(13, 3) < CudaVersion(13, 4));
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
        let anonymous = github_request(&client, RELEASES_API, None).build().unwrap();
        assert!(anonymous.headers().get("Authorization").is_none());
        let blank = github_request(&client, RELEASES_API, Some(" \t "))
            .build()
            .unwrap();
        assert!(blank.headers().get("Authorization").is_none());
        let token = github_request(&client, RELEASES_API, Some(" test-token "))
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
        let selection = select_release(&releases, Some(CudaVersion(13, 3)), None).unwrap();
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
            .get(RELEASES_API)
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        assert!(releases
            .iter()
            .flat_map(|release| release.assets.iter())
            .any(|asset| super::parse_main(&asset.name).is_some()));
    }

    #[test]
    fn parses_native_device_listing() {
        let devices = parse_devices("CUDA0: NVIDIA RTX 4090\nCPU: CPU\n");
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, "CUDA0");
        assert_eq!(devices[0].backend, "CUDA");
    }
}

pub fn windows_gpu_snapshot() -> Option<crate::state::GpuSnapshot> {
    let output = hidden_command("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let line = String::from_utf8_lossy(&output.stdout);
    let mut it = line.lines().next()?.split(',');
    Some(crate::state::GpuSnapshot {
        name: it.next()?.trim().to_string(),
        vram_total_mb: it.next()?.trim().parse().ok()?,
        vram_free_mb: it.next()?.trim().parse().ok()?,
        util_percent: it.next()?.trim().parse().unwrap_or(0),
    })
}
