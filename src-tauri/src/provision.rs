//! Idempotent WSL2 provisioning: distro check → apt → uv venv → vllm → verify.

use anyhow::{bail, Result};
use serde::Serialize;

use crate::wsl;

/// A single provisioning log line, emitted to the `wsl-log` event.
#[derive(Debug, Clone, Serialize)]
pub struct ProvisionLog {
    pub phase: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvisionReport {
    pub phases_completed: Vec<String>,
    pub distro: String,
    pub vllm_version: Option<String>,
    pub torch_version: Option<String>,
    pub cuda_available: bool,
    pub gpu_name: Option<String>,
    pub vram_mb: Option<u64>,
    pub bf16_supported: bool,
}

/// Run one provisioning phase; streams lines to `on_log`, returns report fields.
pub enum ProvisionTarget {
    All,
    EnvCheck,
}

/// Full provisioning pipeline. Every phase is idempotent — re-running skips
/// anything already done. This is the app-facing entry point.
pub fn provision_all(
    distro: &str,
    venv_dir: &str,
    mut on_log: impl FnMut(&str, &str),
) -> Result<ProvisionReport> {
    let mut phases = Vec::new();

    phases.push(phase_distro(distro, &mut on_log)?);
    phases.push(phase_apt(distro, &mut on_log)?);
    phases.push(phase_venv(distro, venv_dir, &mut on_log)?);
    phases.push(phase_vllm(distro, venv_dir, &mut on_log)?);
    let report = phase_verify(distro, venv_dir, &mut on_log)?;

    // Marker file so later skips are quick.
    let marker = format!(
        "mkdir -p ~/llm-lp && echo '{{\"provisioned\": true, \"vllm\": \"{}\"}}' > ~/llm-lp/.provisioned",
        report.vllm_version.clone().unwrap_or_default()
    );
    let _ = wsl::run_script(distro, &marker);

    Ok(ProvisionReport { phases_completed: phases, ..report })
}

fn phase_distro(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    on_log("distro", &format!("checking distro {distro}…"));
    if !wsl::is_apt_distro(distro) {
        bail!(
            "distro '{}' is not apt-based (Ubuntu/Debian/Kali/Mint). \
             Install an Ubuntu distro: `wsl --install -d Ubuntu`, then set it \
             as the default: `wsl --set-default Ubuntu`.",
            distro
        );
    }
    let out = wsl::run_script(distro, "echo ok");
    if !out.ok {
        bail!("wsl -d {distro} is not responsive: {}", out.combined());
    }
    Ok("distro".into())
}

fn phase_apt(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if !apt_done(distro) {
        on_log("apt", "running apt-get update…");
        let out = wsl::run_script_stream(
            distro,
            "sudo apt-get update -y -qq && sudo apt-get install -y -qq python3-venv python3-pip curl",
            |l| on_log("apt", l),
        );
        if !out.ok {
            bail!("apt phase failed: {}", out.combined());
        }
        mark_apt_done(distro);
    } else {
        on_log("apt", "apt packages already installed (skipping).");
    }
    Ok("apt".into())
}

fn apt_done(distro: &str) -> bool {
    wsl::run_script(distro, "command -v python3-venv >/dev/null && command -v pip3 >/dev/null && command -v curl >/dev/null && echo yes").stdout == "yes"
}

fn mark_apt_done(distro: &str) {
    let _ = wsl::run_script(distro, "mkdir -p ~/llm-lp && touch ~/llm-lp/.apt-done");
}

fn phase_venv(distro: &str, venv_dir: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if venv_ok(distro, venv_dir) {
        on_log("venv", "venv already present (skipping).");
        return Ok("venv".into());
    }
    on_log("venv", "creating venv…");
    let script = format!(
        r#"
set -e
mkdir -p ~/llm-lp/run ~/llm-lp/logs
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  echo "using uv: $(uv --version)"
  uv venv --python 3.12 {venv} 2>/dev/null || uv venv {venv}
else
  echo "uv not found; falling back to python3 -m venv"
  python3 -m venv {venv}
fi
{venv}/bin/python --version
{venv}/bin/python -m pip install --upgrade pip -q
"#,
        venv = venv_dir
    );
    let out = wsl::run_script_stream(distro, &script, |l| on_log("venv", l));
    if !out.ok {
        bail!("venv phase failed: {}", out.combined());
    }
    Ok("venv".into())
}

fn venv_ok(distro: &str, venv_dir: &str) -> bool {
    let out = wsl::run_script(distro, &format!("{venv}/bin/python --version 2>/dev/null || true", venv = venv_dir));
    out.stdout.contains("Python 3.")
}

fn phase_vllm(distro: &str, venv_dir: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if vllm_installed(distro, venv_dir) {
        on_log("vllm", "vllm already installed (skipping).");
        return Ok("vllm".into());
    }
    on_log("vllm", "installing vllm + huggingface_hub (CUDA wheels)…");
    let script = format!(
        "export PATH=\"$HOME/.local/bin:$PATH\"; cd ~/llm-lp && uv pip install --python {venv}/bin/python vllm 'huggingface_hub[cli]' 2>&1",
        venv = venv_dir
    );
    let out = wsl::run_script_stream(distro, &script, |l| on_log("vllm", l));
    if !out.ok || !vllm_installed(distro, venv_dir) {
        bail!("vllm phase failed: {}", out.combined());
    }
    Ok("vllm".into())
}

fn vllm_installed(distro: &str, venv_dir: &str) -> bool {
    let out = wsl::run_script(
        distro,
        &format!("{venv}/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true", venv = venv_dir),
    );
    out.ok && !out.stdout.trim().is_empty()
}

fn phase_verify(distro: &str, venv_dir: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<ProvisionReport> {
    on_log("verify", "verifying vllm + torch/CUDA…");
    let vllm_out = wsl::run_script(
        distro,
        &format!("{venv}/bin/python -c 'import vllm; print(vllm.__version__)'", venv = venv_dir),
    );
    let vllm_version = vllm_out.ok.then(|| vllm_out.stdout.trim().to_string());

    let torch_out = wsl::run_script(
        distro,
        &format!(
            "{venv}/bin/python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0); print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)\"",
            venv = venv_dir
        ),
    );
    let lines: Vec<&str> = torch_out.stdout.lines().collect();
    let torch_version = lines.first().map(|s| s.to_string());
    let cuda_available = lines.get(1).map(|s| *s == "True").unwrap_or(false);
    let gpu_name = lines.get(2).filter(|s| !s.is_empty() && **s != "n/a").map(|s| s.to_string());
    let vram_mb = lines
        .get(3)
        .and_then(|s| s.parse::<u64>().ok())
        .map(|b| b / (1024 * 1024));
    let bf16_supported = lines.get(4).map(|s| *s == "True").unwrap_or(false);

    // nvidia-smi from inside WSL as a driver-level cross-check
    let smi = wsl::run_script(distro, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1");
    on_log("verify", &format!("GPU: {}", smi.stdout.trim()));

    let report = ProvisionReport {
        phases_completed: Vec::new(), // filled by caller
        distro: distro.to_string(),
        vllm_version,
        torch_version,
        cuda_available,
        gpu_name,
        vram_mb,
        bf16_supported,
    };
    Ok(report)
}

/// Lightweight environment status (no provisioning). Used by `env_status`.
pub fn env_probe(distro: &str, venv_dir: &str) -> ProvisionReport {
    let vllm_out = wsl::run_script(
        distro,
        &format!("{venv}/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true", venv = venv_dir),
    );
    let vllm_version = vllm_out.ok.then(|| vllm_out.stdout.trim().to_string());
    let torch_out = wsl::run_script(
        distro,
        &format!(
            "{venv}/bin/python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)\" 2>/dev/null || true",
            venv = venv_dir
        ),
    );
    let lines: Vec<&str> = torch_out.stdout.lines().collect();
    let cuda_available = lines.get(1).map(|s| *s == "True").unwrap_or(false);
    let gpu_name = lines.get(2).filter(|s| !s.is_empty() && **s != "n/a").map(|s| s.to_string());
    let bf16_supported = lines.get(3).map(|s| *s == "True").unwrap_or(false);
    let smi = wsl::run_script(distro, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1");
    let mut vram_mb = None;
    if let Some((_, mem)) = smi.stdout.split_once(',') {
        vram_mb = mem.trim().split_whitespace().next().and_then(|s| s.parse::<u64>().ok());
    }
    ProvisionReport {
        phases_completed: Vec::new(),
        distro: distro.to_string(),
        vllm_version,
        torch_version: lines.first().map(|s| s.to_string()),
        cuda_available,
        gpu_name,
        vram_mb,
        bf16_supported,
    }
}