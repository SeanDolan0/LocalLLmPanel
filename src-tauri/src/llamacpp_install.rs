//! Native Windows llama.cpp release installation and capability discovery.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

#[derive(Debug, Clone, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InstallStatus {
    pub installed: bool,
    pub tag: Option<String>,
    pub version: Option<String>,
    pub executable: Option<String>,
    pub gpu: Option<crate::state::GpuSnapshot>,
}

fn choose_assets(assets: &[Asset]) -> Result<(Asset, Option<Asset>)> {
    let mut candidates: Vec<&Asset> = assets
        .iter()
        .filter(|a| {
            let n = a.name.to_ascii_lowercase();
            n.ends_with(".zip")
                && (n.contains("win") || n.contains("windows"))
                && n.contains("cuda")
                && (n.contains("cu12") || n.contains("cu13") || n.contains("12.8"))
                && !n.contains("vulkan")
        })
        .collect();
    candidates.sort_by_key(|a| {
        let n = a.name.to_ascii_lowercase();
        (!(n.contains("cu13")), !(n.contains("cu128")), n.len())
    });
    let main = candidates
        .first()
        .ok_or_else(|| anyhow!("latest llama.cpp release has no Windows CUDA archive"))?;
    let runtime = assets.iter().find(|a| {
        let n = a.name.to_ascii_lowercase();
        n.ends_with(".zip")
            && (n.contains("cudart") || n.contains("cuda-runtime"))
            && (n.contains("win") || n.contains("windows"))
    });
    Ok(((*main).clone(), runtime.cloned()))
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
    let status = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force",
            zip.to_string_lossy().as_ref(),
            destination.to_string_lossy().as_ref(),
        ])
        .status()
        .context("launching PowerShell archive extractor")?;
    if !status.success() {
        return Err(anyhow!("Expand-Archive failed for {}", zip.display()));
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

fn latest_release() -> Result<(String, Asset, Option<Asset>)> {
    let release: Release = reqwest::blocking::Client::new()
        .get(RELEASES_API)
        .header("User-Agent", "LocalLLmPanel")
        .send()?
        .error_for_status()?
        .json()?;
    let (main, runtime) = choose_assets(&release.assets)?;
    Ok((release.tag_name, main, runtime))
}

pub fn install(
    destination: &Path,
    progress: impl Fn(&str, u64, Option<u64>) + Copy,
) -> Result<(String, PathBuf, String, String)> {
    let (tag, main, runtime) = latest_release()?;
    std::fs::create_dir_all(destination)?;
    let temp = std::env::temp_dir().join(format!("llamacpp-{}.zip", std::process::id()));
    progress(&format!("downloading {}", main.name), 0, None);
    download(&main.browser_download_url, &temp, |done, total| {
        progress(&main.name, done, total)
    })?;
    expand_archive(&temp, destination)?;
    let _ = std::fs::remove_file(&temp);

    if let Some(runtime) = runtime {
        let runtime_zip =
            std::env::temp_dir().join(format!("llamacpp-cudart-{}.zip", std::process::id()));
        download(&runtime.browser_download_url, &runtime_zip, |done, total| {
            progress(&runtime.name, done, total)
        })?;
        expand_archive(&runtime_zip, destination)?;
        let _ = std::fs::remove_file(runtime_zip);
    }

    let exe = find_file(destination, "llama-server.exe")
        .ok_or_else(|| anyhow!("archive did not contain llama-server.exe"))?;
    let version = command_output(&exe, &["--version"])?;
    let help = command_output(&exe, &["--help"])?;
    Ok((tag, exe, version, help))
}

pub fn command_output(exe: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new(exe).args(args).output()?;
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&output.stderr).to_string();
    }
    if !output.status.success() {
        return Err(anyhow!("{} failed: {}", exe.display(), text.trim()));
    }
    Ok(text.trim().to_string())
}

pub fn executable_from_config(cfg: &crate::state::PersistedConfig) -> Option<PathBuf> {
    cfg.llamacpp_executable
        .as_ref()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_file(Path::new(&cfg.llamacpp_dir), "llama-server.exe"))
}

pub fn windows_gpu_snapshot() -> Option<crate::state::GpuSnapshot> {
    let output = Command::new("nvidia-smi")
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
