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
    crate::server::validate_venv_dir(venv_dir).map_err(|e| anyhow::anyhow!(e))?;
    let mut phases = Vec::new();

    // Determine which distro to actually use (auto-install dedicated if needed)
    let distros = wsl::installed_distros();
    let mut effective_distro = if distros.contains(&distro.to_string()) {
        distro.to_string()
    } else {
        on_log("distro", &format!("distro '{}' not found, installing dedicated distro…", distro));
        let installed = wsl::ensure_app_distro(|msg| on_log("distro", msg))
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        on_log("distro", &format!("using dedicated distro '{}'", installed));
        installed
    };

    // If the effective distro isn't apt-based (e.g., docker-desktop), install dedicated
    if !wsl::is_apt_distro(&effective_distro) {
        on_log("distro", &format!("distro '{}' is not apt-based, installing dedicated distro…", effective_distro));
        let installed = wsl::ensure_app_distro(|msg| on_log("distro", msg))
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        on_log("distro", &format!("using dedicated distro '{}'", installed));
        effective_distro = installed;
    }

    phases.push(phase_distro(&effective_distro, &mut on_log)?);
    phases.push(phase_sudo(&effective_distro, &mut on_log)?);
    phases.push(phase_apt(&effective_distro, &mut on_log)?);
    phases.push(phase_uv(&effective_distro, &mut on_log)?);
    phases.push(phase_venv(&effective_distro, venv_dir, &mut on_log)?);
    phases.push(phase_vllm(&effective_distro, venv_dir, &mut on_log)?);
    let report = phase_verify(&effective_distro, venv_dir, &mut on_log)?;

    // Marker file so later skips are quick.
    let full_report = ProvisionReport {
        phases_completed: phases,
        ..report
    };
    if let Ok(json) = serde_json::to_string(&full_report) {
        let marker = format!(
            "mkdir -p ~/llm-lp && cat << 'EOF' > ~/llm-lp/.provisioned\n{}\nEOF",
            json
        );
        let _ = wsl::run_script(&effective_distro, &marker);
    }

    Ok(full_report)
}

fn venv_assignment(venv_dir: &str) -> String {
    format!(
        "{}venv_dir=$(__llm_panel_expand_tilde {})",
        wsl::WSL_TILDE_EXPANSION_SNIPPET,
        wsl::shell_quote_wsl(venv_dir),
    )
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
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    on_log(
        "sudo",
        &format!("configuring passwordless sudo for user '{user}' (via wsl --user root)…"),
    );
    let rule = format!("{safe_user} ALL=(ALL) NOPASSWD:ALL");
    let rule_path = format!("/etc/sudoers.d/llm-panel-{safe_user}");
    let script = format!(
        "printf '%s\\n' {} > {} && chmod 440 {} && echo ok",
        wsl::shell_quote_wsl(&rule),
        wsl::shell_quote_wsl(&rule_path),
        wsl::shell_quote_wsl(&rule_path),
    );
    let out = wsl::run_script_root(distro, &script);
    if !out.ok || !out.stdout.trim().ends_with("ok") {
        bail!("sudo config failed: {}", out.combined());
    }
    let verify = wsl::run_script(distro, "sudo -n true 2>/dev/null && echo ok || echo needs");
    if verify.stdout.trim() != "ok" {
        bail!(
            "passwordless sudo still not working after config: {}",
            verify.combined()
        );
    }
    Ok("sudo".into())
}

/// Install `uv` inside the distro if missing (host uv is irrelevant — WSL is
/// a separate Linux). uv makes the big vLLM install dramatically faster and
/// the plan pins it as the venv/installer of choice. Idempotent.
fn phase_uv(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    let has = wsl::run_script(
        distro,
        "command -v uv >/dev/null 2>&1 || [ -x \"$HOME/.local/bin/uv\" ] && echo yes || echo no",
    );
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
    out.stdout.trim() == "yes"
        || out.stdout.trim().contains(".apt-done")
        || out.ok && !out.stdout.contains("no")
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
        r#"{venv_assignment}
set -e
mkdir -p ~/llm-lp/run ~/llm-lp/logs
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  echo "using uv: $(uv --version)"
  uv venv --python 3.12 "$venv_dir" 2>/dev/null || uv venv "$venv_dir"
else
  echo "uv not found; falling back to python3 -m venv"
  python3 -m venv "$venv_dir" || {{ echo "could not create venv at $venv_dir" >&2; exit 1; }}
fi
# Ubuntu 22.04's python3-venv meta sometimes misses the version-specific
# package, leaving a pip-less venv. Repair with ensurepip, or apt-install the
# right python3.*-venv and recreate.
if ! "$venv_dir/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "venv has no pip; running ensurepip…"
  if ! "$venv_dir/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "ensurepip failed; apt-get installing python3.12-venv…"
    DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq python3.12-venv >/dev/null 2>&1
    python3 -m venv "$venv_dir"
  fi
fi
"$venv_dir/bin/python" --version
"$venv_dir/bin/python" -m pip install --upgrade pip -q
"#,
        venv_assignment = venv_assignment(venv_dir),
    );
    let out = wsl::run_script_stream(distro, &script, |l| on_log("venv", l));
    if !out.ok {
        bail!("venv phase failed: {}", out.combined());
    }
    Ok("venv".into())
}

fn venv_ok(distro: &str, venv_dir: &str) -> bool {
    let script = format!(
        "{}\"$venv_dir/bin/python\" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null && \"$venv_dir/bin/python\" -m pip --version >/dev/null 2>&1 && echo ok || echo bad",
        venv_assignment(venv_dir),
    );
    let out = wsl::run_script(distro, &script);
    out.stdout.contains("3.") && out.stdout.ends_with("ok")
}

fn phase_vllm(distro: &str, venv_dir: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if vllm_installed(distro, venv_dir) {
        on_log("vllm", "vllm already installed (skipping).");
        return Ok("vllm".into());
    }
    on_log("vllm", "installing vllm + huggingface_hub (CUDA wheels)…");
    let script = format!(
        "{}export PATH=\"$HOME/.local/bin:$PATH\"; cd ~/llm-lp && uv pip install --python \"$venv_dir/bin/python\" vllm 'huggingface_hub[cli]' 2>&1",
        venv_assignment(venv_dir),
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
        &format!(
            "{}\"$venv_dir/bin/python\" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true",
            venv_assignment(venv_dir),
        ),
    );
    out.ok && !out.stdout.trim().is_empty()
}

fn phase_verify(
    distro: &str,
    venv_dir: &str,
    on_log: &mut impl FnMut(&str, &str),
) -> Result<ProvisionReport> {
    on_log("verify", "verifying vllm + torch/CUDA…");
    let vllm_out = wsl::run_script(
        distro,
        &format!(
            "{}\"$venv_dir/bin/python\" -c 'import vllm; print(vllm.__version__)'",
            venv_assignment(venv_dir),
        ),
    );
    let vllm_version = vllm_out.ok.then(|| vllm_out.stdout.trim().to_string());

    let torch_out = wsl::run_script(
        distro,
        &format!(
            "{}\"$venv_dir/bin/python\" -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0); print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)\"",
            venv_assignment(venv_dir),
        ),
    );
    let lines: Vec<&str> = torch_out.stdout.lines().collect();
    let torch_version = lines.first().map(|s| s.to_string());
    let cuda_available = lines.get(1).map(|s| *s == "True").unwrap_or(false);
    let gpu_name = lines
        .get(2)
        .filter(|s| !s.is_empty() && **s != "n/a")
        .map(|s| s.to_string());
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

/// Optional phase: Install CUDA build tools for FlashInfer JIT compilation.
/// This installs gcc, python3.12-dev, ninja-build, and the CUDA toolkit from NVIDIA's WSL-Ubuntu repo.
/// Must be run as root (via wsl --user root) because apt needs root.
/// Returns the phase name if successful, or an error if the user cancels or it fails.
pub fn phase_cuda_build_tools(
    distro: &str,
    on_log: &mut impl FnMut(&str, &str),
) -> Result<String> {
    on_log("cuda-tools", "Checking for existing CUDA build tools…");

    // Check what's already installed
    let check = wsl::run_script(
        distro,
        r#"
            which nvcc >/dev/null 2>&1 && echo "nvcc=yes" || echo "nvcc=no"
            which gcc >/dev/null 2>&1 && echo "gcc=yes" || echo "gcc=no"
            which ninja >/dev/null 2>&1 && echo "ninja=yes" || echo "ninja=no"
            test -f /usr/include/python3.12/Python.h && echo "python_dev=yes" || echo "python_dev=no"
        "#,
    );

    let mut nvcc = false;
    let mut gcc = false;
    let mut ninja = false;
    let mut python_dev = false;
    for line in check.stdout.lines() {
        if line.starts_with("nvcc=yes") { nvcc = true; }
        else if line.starts_with("gcc=yes") { gcc = true; }
        else if line.starts_with("ninja=yes") { ninja = true; }
        else if line.starts_with("python_dev=yes") { python_dev = true; }
    }

    if nvcc && gcc && ninja && python_dev {
        on_log("cuda-tools", "All CUDA build tools already installed (skipping).");
        return Ok("cuda-tools".into());
    }

    on_log("cuda-tools", "Installing CUDA build tools (gcc, python3.12-dev, ninja-build, CUDA toolkit)…");
    on_log("cuda-tools", "⚠ This downloads several GB and may take 5-15 minutes. First FlashInfer JIT compile will be slow.");

    // Get PyTorch CUDA version to match toolkit version
    let torch_cuda = wsl::run_script(
        distro,
        "~/llm-lp/.venv/bin/python -c \"import torch; print(torch.version.cuda)\" 2>/dev/null || echo 'unknown'",
    );
    let torch_cuda_version = torch_cuda.stdout.trim();
    let cuda_major = if torch_cuda_version != "unknown" && !torch_cuda_version.is_empty() {
        torch_cuda_version.split('.').next().unwrap_or("12")
    } else {
        "12"
    };

    on_log("cuda-tools", &format!("Targeting CUDA toolkit major version: {cuda_major} (matching PyTorch)"));

    // Install basic build tools first (gcc, python3.12-dev, ninja-build)
    let apt_script = format!(
        r#"
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y -qq
        apt-get install -y -qq gcc python3.12-dev ninja-build wget gpg
        "#
    );
    let out = wsl::run_script_root(distro, &apt_script);
    if !out.ok {
        on_log("cuda-tools", &format!("Failed to install base build tools: {}", out.combined()));
        bail!("CUDA build tools installation failed: {}", out.combined());
    }
    on_log("cuda-tools", "Base build tools (gcc, python3.12-dev, ninja-build) installed.");

    // Add NVIDIA CUDA repository for WSL-Ubuntu
    let repo_script = format!(
        r#"
        set -e
        export DEBIAN_FRONTEND=noninteractive
        # Determine Ubuntu version
        UBUNTU_VERSION=$(lsb_release -rs)
        # NVIDIA's WSL-Ubuntu repo
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/ubuntu${{UBUNTU_VERSION//./}}/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update -y -qq
        # Install CUDA toolkit (compiler, nvcc, libraries) - no driver
        apt-get install -y -qq cuda-toolkit-{cuda_major}-0
        "#
    );
    on_log("cuda-tools", "Adding NVIDIA CUDA repository and installing CUDA toolkit…");
    let out = wsl::run_script_root(distro, &repo_script);
    if !out.ok {
        on_log("cuda-tools", &format!("CUDA toolkit install failed (may need manual intervention): {}", out.combined()));
        // Don't bail - base tools are installed, user can retry
    } else {
        on_log("cuda-tools", "CUDA toolkit installed successfully.");
    }

    // Verify nvcc is now available
    let verify = wsl::run_script(distro, "which nvcc && nvcc --version | head -1");
    if verify.ok && verify.stdout.contains("nvcc") {
        on_log("cuda-tools", &format!("nvcc verified: {}", verify.stdout.lines().next().unwrap_or("ok")));
    } else {
        on_log("cuda-tools", "WARNING: nvcc not found in PATH after install. May need shell restart or manual PATH setup.");
    }

    // Set CUDA_HOME for future sessions
    let cuda_home_script = r#"
        echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
        echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    "#;
    let _ = wsl::run_script(distro, cuda_home_script);

    Ok("cuda-tools".into())
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
