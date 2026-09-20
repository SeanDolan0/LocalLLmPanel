//! Idempotent WSL2 provisioning: distro check → apt → uv venv → vllm → verify.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::wsl;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Full provisioning pipeline. Every phase is idempotent — re-running skips
/// anything already done. This is the app-facing entry point.
pub fn provision_all(
    distro: &str,
    venv_dir: &str,
    mut on_log: impl FnMut(&str, &str),
) -> Result<ProvisionReport> {
    let mut phases = Vec::new();

    phases.push(phase_distro(distro, &mut on_log)?);
    phases.push(phase_sudo(distro, &mut on_log)?);
    phases.push(phase_apt(distro, &mut on_log)?);
    phases.push(phase_uv(distro, &mut on_log)?);
    phases.push(phase_venv(distro, venv_dir, &mut on_log)?);
    phases.push(phase_vllm(distro, venv_dir, &mut on_log)?);
    let report = phase_verify(distro, venv_dir, &mut on_log)?;

    // Marker file so later skips are quick.
    let full_report = ProvisionReport { phases_completed: phases, ..report };
    if let Ok(json) = serde_json::to_string(&full_report) {
        let marker = format!(
            "mkdir -p ~/llm-lp && cat << 'EOF' > ~/llm-lp/.provisioned\n{}\nEOF",
            json
        );
        let _ = wsl::run_script(distro, &marker);
    }

    Ok(full_report)
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

/// Make `sudo` passwordless for the distro's default user. WSL distros
/// usually require a sudo password; a non-interactive `wsl.exe` shell cannot
/// answer the prompt, so the app writes a scoped NOPASSWD rule via
/// `wsl --user root` (root needs no password). Idempotent: does nothing if
/// passwordless sudo already works.
fn phase_sudo(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    let probe = wsl::run_script(distro, "sudo -n true 2>/dev/null && echo ok || echo needs");
    if probe.stdout.trim() == "ok" {
        on_log("sudo", "passwordless sudo already configured (skipping).");
        return Ok("sudo".into());
    }
    let user = wsl::run_script(distro, "id -un");
    let user = user.stdout.trim().to_string();
    if user.is_empty() {
        bail!("could not determine WSL user for distro {distro}");
    }
    let safe_user: String = user
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    on_log("sudo", &format!("configuring passwordless sudo for user '{user}' (via wsl --user root)…"));
    let script = format!(
        "echo '{user} ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/llm-panel-{safe_user} && chmod 440 /etc/sudoers.d/llm-panel-{safe_user} && echo ok"
    );
    let out = wsl::run_script_root(distro, &script);
    if !out.ok || !out.stdout.trim().ends_with("ok") {
        bail!("sudo config failed: {}", out.combined());
    }
    let verify = wsl::run_script(distro, "sudo -n true 2>/dev/null && echo ok || echo needs");
    if verify.stdout.trim() != "ok" {
        bail!("passwordless sudo still not working after config: {}", verify.combined());
    }
    Ok("sudo".into())
}

/// Install `uv` inside the distro if missing (host uv is irrelevant — WSL is
/// a separate Linux). uv makes the big vLLM install dramatically faster and
/// the plan pins it as the venv/installer of choice. Idempotent.
fn phase_uv(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    let has = wsl::run_script(distro, "command -v uv >/dev/null 2>&1 || [ -x \"$HOME/.local/bin/uv\" ] && echo yes || echo no");
    if has.stdout.trim() == "yes" {
        on_log("uv", "uv already installed (skipping).");
        return Ok("uv".into());
    }
    on_log("uv", "installing uv via official installer…");
    let out = wsl::run_script_stream(
        distro,
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        |l| on_log("uv", l),
    );
    if !out.ok {
        // curl may be missing (apt phase normally runs first in fresh WSL).
        // Fall back to apt uv if available, else fail with clear message.
        let fallback = wsl::run_script(distro, "DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq uv 2>&1 | tail -1; command -v uv && echo ok");
        if fallback.stdout.trim() != "ok" {
            bail!("uv install failed (installer + apt): {}", out.combined());
        }
    }
    Ok("uv".into())
}

fn phase_apt(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if !apt_done(distro) {
        on_log("apt", "running apt-get update…");
        let out = wsl::run_script_stream(
            distro,
            "DEBIAN_FRONTEND=noninteractive sudo apt-get update -y -qq && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq python3-venv python3-pip curl",
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

pub fn apt_done_script() -> &'static str {
    "[ -f ~/llm-lp/.apt-done ] || (command -v pip3 >/dev/null && command -v curl >/dev/null && python3 -c 'import venv' 2>/dev/null && echo yes) || echo no"
}

fn apt_done(distro: &str) -> bool {
    let out = wsl::run_script(distro, apt_done_script());
    out.stdout.trim() == "yes" || out.stdout.trim().contains(".apt-done") || out.ok && !out.stdout.contains("no")
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
  python3 -m venv {venv} || {{ rm -rf {venv}; exit 1; }}
fi
# Ubuntu 22.04's python3-venv meta sometimes misses the version-specific
# package, leaving a pip-less venv. Repair with ensurepip, or apt-install the
# right python3.*-venv and recreate.
if ! {venv}/bin/python -m pip --version >/dev/null 2>&1; then
  echo "venv has no pip; running ensurepip…"
  if ! {venv}/bin/python -m ensurepip --upgrade >/dev/null 2>&1; then
    PYM=\$({venv}/bin/python --version | sed 's/Python \\([0-9]*\\.[0-9]*\\).*/python\\1-venv/')
    echo "ensurepip failed; apt-get installing $PYM…"
    DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq "$PYM" >/dev/null 2>&1
    rm -rf {venv}
    python3 -m venv {venv}
  fi
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
    let out = wsl::run_script(
        distro,
        &format!("{venv}/bin/python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null && {venv}/bin/python -m pip --version >/dev/null 2>&1 && echo ok || echo bad", venv = venv_dir),
    );
    out.stdout.contains("3.") && out.stdout.ends_with("ok")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apt_done_command_format() {
        let cmd = apt_done_script();
        assert!(cmd.contains(".apt-done"));
        assert!(cmd.contains("python3"));
    }
}