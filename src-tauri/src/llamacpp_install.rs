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
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
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
    let mut candidates: Vec<&Asset> = assets
        .iter()
        .filter(|a| {
            let n = a.name.to_ascii_lowercase();
            n.ends_with(".zip")
                && (n.contains("win") || n.contains("windows"))
                && n.contains("cuda")
                && !n.contains("vulkan")
                && !n.contains("cpu")
        })
        .collect();
    candidates.sort_by(|a, b| cuda_asset_version(&b.name).cmp(&cuda_asset_version(&a.name)));
    let main = candidates
        .first()
        .ok_or_else(|| anyhow!("latest llama.cpp release has no Windows CUDA archive (Vulkan/CPU builds are not supported)"))?;
    let main_cuda = cuda_asset_version(&main.name);
    if main_cuda < (12, 8) {
        return Err(anyhow!(
            "latest llama.cpp release has no Windows CUDA 12.8+ archive required for Blackwell GPUs"
        ));
    }
    let mut runtimes: Vec<&Asset> = assets
        .iter()
        .filter(|a| {
            let n = a.name.to_ascii_lowercase();
            n.ends_with(".zip")
                && (n.contains("cudart") || n.contains("cuda-runtime"))
                && (n.contains("win") || n.contains("windows"))
        })
        .collect();
    runtimes.sort_by(|a, b| cuda_asset_version(&b.name).cmp(&cuda_asset_version(&a.name)));
    let runtime = runtimes
        .into_iter()
        .find(|asset| cuda_asset_version(&asset.name) <= main_cuda);
    Ok(((*main).clone(), runtime.cloned()))
}

fn cuda_asset_version(name: &str) -> (u32, u32) {
    let lower = name.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'c' && i + 4 < bytes.len() && bytes[i + 1] == b'u' {
            let digits = &lower[i + 2..];
            let digits: String = digits.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 2 {
                if let Ok(value) = digits.parse::<u32>() {
                    if digits.len() == 2 {
                        return (value, 0);
                    }
                    return (value / 10, value % 10);
                }
            }
        }
    }
    if let Some(pos) = lower.find("cuda") {
        let suffix = lower[pos + 4..].trim_start_matches(['-', '_', '.']);
        let digits: String = suffix
            .chars()
            .filter(|c| c.is_ascii_digit())
            .take(3)
            .collect();
        if let Ok(value) = digits.parse::<u32>() {
            return (value / 10, value % 10);
        }
    }
    (0, 0)
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
    let status = hidden_command("powershell")
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
    let temp = destination.join(format!(".llamacpp-{}.zip", std::process::id()));
    progress(&format!("downloading {}", main.name), 0, None);
    download(&main.browser_download_url, &temp, |done, total| {
        progress(&main.name, done, total)
    })?;
    expand_archive(&temp, destination)?;
    let _ = std::fs::remove_file(&temp);

    if let Some(runtime) = runtime {
        let runtime_zip =
            destination.join(format!(".llamacpp-cudart-{}.zip", std::process::id()));
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

pub fn executable_from_config(cfg: &crate::state::PersistedConfig) -> Option<PathBuf> {
    cfg.llamacpp_executable
        .as_ref()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_file(Path::new(&cfg.llamacpp_dir), "llama-server.exe"))
        .or_else(|| find_on_path("llama-server.exe"))
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
    use super::{choose_assets, executable_from_config, parse_devices, Asset};
    use crate::state::PersistedConfig;

    #[test]
    fn detects_llama_server_from_path_when_not_configured() {
        let root = std::env::current_dir()
            .unwrap()
            .join(format!(
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

    #[test]
    fn selects_highest_cuda_asset_and_matching_runtime() {
        let assets = vec![
            Asset { name: "llama-b123-win-cuda-12.4-x64.zip".into(), browser_download_url: "old".into() },
            Asset { name: "llama-b123-win-cuda-12.8-x64.zip".into(), browser_download_url: "main".into() },
            Asset { name: "cudart-12.8-win-x64.zip".into(), browser_download_url: "runtime".into() },
            Asset { name: "llama-b123-win-vulkan-x64.zip".into(), browser_download_url: "vulkan".into() },
            Asset { name: "llama-b123-win-x64.zip".into(), browser_download_url: "cpu".into() },
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
    fn refuses_cuda_archive_before_blackwell_minimum() {
        let assets = vec![Asset {
            name: "llama-win-cuda-12.4-x64.zip".into(),
            browser_download_url: "old".into(),
        }];
        assert!(choose_assets(&assets).is_err());
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
