//! Idempotent WSL2 provisioning: distro check -> apt -> uv venv -> vllm -> verify.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::wsl;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

/// Distinct failure states encountered when verifying GPU and runtime capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    DriverNotVisible { diagnostic: String },
    NoGpuInWsl { diagnostic: String },
    TorchImportFailed { diagnostic: String },
    CudaUnavailable { diagnostic: String, torch_version: Option<String> },
    NoWorkingGpu { diagnostic: String },
}

impl std::fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DriverNotVisible { diagnostic } => write!(
                f,
                "NVIDIA driver not visible in WSL2. Host NVIDIA Windows driver must be installed and WSL GPU paravirtualization active. Diagnostic: {diagnostic}"
            ),
            Self::NoGpuInWsl { diagnostic } => write!(
                f,
                "No NVIDIA GPU detected in WSL2. nvidia-smi ran but detected no GPU devices. Diagnostic: {diagnostic}"
            ),
            Self::TorchImportFailed { diagnostic } => write!(
                f,
                "PyTorch import failed in provisioned venv. Virtual environment cannot load torch. Diagnostic: {diagnostic}"
            ),
            Self::CudaUnavailable { diagnostic, torch_version } => write!(
                f,
                "CUDA unavailable in PyTorch (version: {}). PyTorch loaded but torch.cuda.is_available() is False. Diagnostic: {diagnostic}",
                torch_version.as_deref().unwrap_or("unknown")
            ),
            Self::NoWorkingGpu { diagnostic } => write!(
                f,
                "No working GPU found: GPU properties could not be queried or VRAM is 0. Diagnostic: {diagnostic}"
            ),
        }
    }
}

/// Tested compatibility manifest pinning vetted combinations of Python, vLLM, PyTorch,
/// CUDA wheels, and minimum host driver capabilities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeCompatibilityManifest {
    pub python_version: &'static str,
    pub vllm_version: &'static str,
    pub torch_version: &'static str,
    pub torch_cuda_version: &'static str,
    pub huggingface_hub_version: &'static str,
    pub min_cuda_driver: (u32, u32),
}

pub const TESTED_RUNTIME_MANIFEST: RuntimeCompatibilityManifest = RuntimeCompatibilityManifest {
    python_version: "3.12",
    vllm_version: "0.7.3",
    torch_version: "2.5.1",
    torch_cuda_version: "cu124",
    huggingface_hub_version: "0.29.1",
    min_cuda_driver: (12, 0),
};

/// Pure parser and classifier for verification command outputs.
pub fn parse_verify_result(
    smi_ok: bool,
    smi_stdout: &str,
    smi_stderr: &str,
    torch_ok: bool,
    torch_stdout: &str,
    torch_stderr: &str,
    vllm_version: Option<String>,
    distro: &str,
) -> Result<ProvisionReport, VerifyFailure> {
    // 1. Check WSL nvidia-smi driver visibility and GPU enumeration
    let trimmed_smi = smi_stdout.trim();
    if !smi_ok || trimmed_smi.is_empty() {
        let combined = format!("{smi_stdout}\n{smi_stderr}").trim().to_string();
        if combined.contains("NVIDIA-SMI has failed")
            || combined.contains("command not found")
            || combined.contains("not found")
            || !smi_ok
        {
            return Err(VerifyFailure::DriverNotVisible { diagnostic: combined });
        }
        return Err(VerifyFailure::NoGpuInWsl { diagnostic: combined });
    }

    let first_line = trimmed_smi.lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return Err(VerifyFailure::NoGpuInWsl {
            diagnostic: "nvidia-smi returned empty GPU query list".to_string(),
        });
    }

    // 2. Check PyTorch execution in the venv
    if !torch_ok {
        let combined = format!("{torch_stdout}\n{torch_stderr}").trim().to_string();
        return Err(VerifyFailure::TorchImportFailed { diagnostic: combined });
    }

    let lines: Vec<&str> = torch_stdout.lines().collect();
    let torch_version = lines.first().map(|s| s.trim().to_string());
    let cuda_available = lines.get(1).map(|s| s.trim() == "True").unwrap_or(false);

    if !cuda_available {
        return Err(VerifyFailure::CudaUnavailable {
            diagnostic: format!(
                "torch.cuda.is_available() was False; driver reported: {first_line}"
            ),
            torch_version,
        });
    }

    let gpu_name = lines
        .get(2)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "n/a");
    let vram_mb = lines
        .get(3)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b / (1024 * 1024));
    let bf16_supported = lines.get(4).map(|s| s.trim() == "True").unwrap_or(false);

    if gpu_name.is_none() || vram_mb.unwrap_or(0) == 0 {
        return Err(VerifyFailure::NoWorkingGpu {
            diagnostic: format!("gpu_name={gpu_name:?}, vram_mb={vram_mb:?}"),
        });
    }

    Ok(ProvisionReport {
        phases_completed: Vec::new(),
        distro: distro.to_string(),
        vllm_version,
        torch_version,
        cuda_available,
        gpu_name,
        vram_mb,
        bf16_supported,
    })
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
        on_log(
            "distro",
            &format!(
                "distro '{distro}' not found, installing dedicated distro…"
            ),
        );
        let installed = wsl::ensure_app_distro(|msg| on_log("distro", msg))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        on_log("distro", &format!("using dedicated distro '{installed}'"));
        installed
    };

    // If the effective distro isn't apt-based (e.g., docker-desktop), install dedicated
    if !wsl::is_apt_distro(&effective_distro) {
        on_log(
            "distro",
            &format!(
                "distro '{effective_distro}' is not apt-based, installing dedicated distro…"
            ),
        );
        let installed = wsl::ensure_app_distro(|msg| on_log("distro", msg))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        on_log("distro", &format!("using dedicated distro '{installed}'"));
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
            "mkdir -p ~/llm-lp && cat << 'EOF' > ~/llm-lp/.provisioned\n{json}\nEOF"
        );
        let _ = wsl::run_script(&effective_distro, &marker);
    }

    Ok(full_report)
}

fn venv_assignment(venv_dir: &str) -> String {
    format!(
        "{}\nvenv_dir=$(__llm_panel_expand_tilde {})\n",
        wsl::WSL_TILDE_EXPANSION_SNIPPET.trim(),
        wsl::shell_quote_wsl(venv_dir),
    )
}

fn phase_distro(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    on_log("distro", &format!("checking distro '{distro}'…"));
    let res = wsl::run_script(distro, "uname -a; lsb_release -d 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME");
    if !res.ok {
        bail!("failed to run basic commands inside WSL distro '{distro}': {}", res.combined());
    }
    let first = res.stdout.lines().next().unwrap_or("");
    on_log("distro", &format!("running on {first}"));
    Ok("distro".into())
}

fn phase_sudo(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if sudo_nopasswd_ok(distro) {
        on_log("sudo", "passwordless sudo already configured (skipping).");
        return Ok("sudo".into());
    }
    on_log("sudo", "configuring scoped passwordless sudo via root…");
    let script = r#"
        u="$(id -un)"
        echo "$u ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt" > /etc/sudoers.d/llm-panel-sudoers
        chmod 0440 /etc/sudoers.d/llm-panel-sudoers
        rm -f /etc/sudoers.d/99-llm-lp
    "#;
    let out = wsl::run_script_root(distro, script);
    if !out.ok {
        bail!("failed to configure scoped passwordless sudo via root: {}", out.combined());
    }
    if !sudo_nopasswd_ok(distro) {
        bail!("scoped passwordless sudo configured, but verify failed");
    }
    on_log("sudo", "scoped passwordless sudo active.");
    Ok("sudo".into())
}

fn sudo_nopasswd_ok(distro: &str) -> bool {
    let out = wsl::run_script(
        distro,
        "sudo -n apt-get -v 2>/dev/null && echo ok || (sudo -n true 2>/dev/null && echo ok || echo fail)",
    );
    out.stdout.trim() == "ok"
}

/// Idempotent uv install pipeline using set -euo pipefail, command verification,
/// and fallback package install if the standalone installer fails.
fn phase_uv(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    let check_script = r#"set -euo pipefail
UV_BIN=$(command -v uv 2>/dev/null || ( [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv" ) || true)
if [ -n "$UV_BIN" ] && "$UV_BIN" --version >/dev/null 2>&1; then
    "$UV_BIN" --version
else
    echo "NOT_INSTALLED"
fi
"#;
    let has = wsl::run_script(distro, check_script);
    if has.ok && !has.stdout.contains("NOT_INSTALLED") && has.stdout.contains("uv ") {
        on_log("uv", &format!("uv already installed: {} (skipping)", has.stdout.trim()));
        return Ok("uv".into());
    }

    on_log("uv", "installing uv via official installer…");
    let install_script = "set -euo pipefail\ncurl -LsSf https://astral.sh/uv/install.sh | sh";
    let out = wsl::run_script_stream(distro, install_script, |l| on_log("uv", l));

    let verify_script = r#"set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
UV_BIN=$(command -v uv 2>/dev/null || ( [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv" ))
"$UV_BIN" --version
"#;
    let verify = wsl::run_script(distro, verify_script);
    if verify.ok && verify.stdout.contains("uv ") {
        on_log("uv", &format!("uv verified: {}", verify.stdout.trim()));
        return Ok("uv".into());
    }

    on_log("uv", "official installer verification failed; attempting apt fallback…");
    let fallback_script = r#"set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y -qq
sudo apt-get install -y -qq uv
export PATH="$HOME/.local/bin:$PATH"
command -v uv
uv --version
"#;
    let fallback = wsl::run_script_stream(distro, fallback_script, |l| on_log("uv", l));
    let verify_fallback = wsl::run_script(distro, verify_script);
    if !verify_fallback.ok || !verify_fallback.stdout.contains("uv ") {
        bail!(
            "uv install verification failed (installer: {}, fallback: {}, detail: {})",
            out.combined().trim(),
            fallback.combined().trim(),
            verify_fallback.combined().trim()
        );
    }
    on_log("uv", &format!("uv installed via fallback: {}", verify_fallback.stdout.trim()));
    Ok("uv".into())
}

fn phase_apt(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if !apt_done(distro) {
        on_log("apt", "running apt-get update…");
        let script = apt_install_script();
        let out = wsl::run_script_stream(distro, &script, |l| on_log("apt", l));
        if !out.ok {
            bail!("apt-get phase failed: {}", out.combined());
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

fn apt_install_script() -> &'static str {
    r#"
        set -e
        export DEBIAN_FRONTEND=noninteractive
        sudo apt-get update -y -qq
        sudo apt-get install -y -qq             curl             ca-certificates             python3             python3-pip             python3-venv             procps             pciutils             bc             jq
    "#
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
  uv venv --seed --clear --python 3.12 "$venv_dir" 2>/dev/null || uv venv --seed --clear "$venv_dir"
else
  echo "uv not found; falling back to python3 -m venv"
  python3 -m venv --clear "$venv_dir" || {{ echo "could not create venv at $venv_dir" >&2; exit 1; }}
fi
if ! "$venv_dir/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "venv has no pip; running ensurepip…"
  if ! "$venv_dir/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "ensurepip failed; apt-get installing python3.12-venv…"
    DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq python3.12-venv >/dev/null 2>&1
    python3 -m venv --clear "$venv_dir"
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

/// Installs vLLM and HuggingFace Hub with constrained versions from TESTED_RUNTIME_MANIFEST.
fn phase_vllm(distro: &str, venv_dir: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    if vllm_installed(distro, venv_dir) {
        on_log("vllm", "vllm already installed (skipping).");
        return Ok("vllm".into());
    }
    on_log(
        "vllm",
        &format!(
            "installing constrained vllm=={} and huggingface_hub=={} (CUDA wheels)…",
            TESTED_RUNTIME_MANIFEST.vllm_version, TESTED_RUNTIME_MANIFEST.huggingface_hub_version
        ),
    );
    let script = format!(
        "{}set -euo pipefail\nexport PATH=\"$HOME/.local/bin:$PATH\"; cd ~/llm-lp && uv pip install --python \"$venv_dir/bin/python\" 'vllm=={}' 'huggingface_hub[cli]=={}' 2>&1",
        venv_assignment(venv_dir),
        TESTED_RUNTIME_MANIFEST.vllm_version,
        TESTED_RUNTIME_MANIFEST.huggingface_hub_version,
    );
    let out = wsl::run_script_stream(distro, &script, |l| on_log("vllm", l));
    if !out.ok || !vllm_installed(distro, venv_dir) {
        bail!("vllm phase failed: {}", out.combined());
    }
    Ok("vllm".into())
}

/// Transactional venv execution with backup snapshot and rollback on error.
pub fn transactional_venv_operation(
    distro: &str,
    venv_dir: &str,
    script_to_run: &str,
    on_log: &mut impl FnMut(&str, &str),
) -> Result<()> {
    on_log("transactional-venv", "creating backup snapshot of venv…");
    let backup_script = format!(
        "{}set -euo pipefail\nrm -rf \"$venv_dir.bak\" && cp -a \"$venv_dir\" \"$venv_dir.bak\"",
        venv_assignment(venv_dir)
    );
    let backup_out = wsl::run_script(distro, &backup_script);
    let backup_created = backup_out.ok;

    let op_out = wsl::run_script_stream(distro, script_to_run, |l| on_log("venv-op", l));
    if op_out.ok {
        if backup_created {
            let _ = wsl::run_script(distro, &format!("{}rm -rf \"$venv_dir.bak\"", venv_assignment(venv_dir)));
        }
        Ok(())
    } else {
        if backup_created {
            on_log("transactional-venv", "operation failed; rolling back venv to backup snapshot…");
            let rollback_script = format!(
                "{}rm -rf \"$venv_dir\" && mv \"$venv_dir.bak\" \"$venv_dir\"",
                venv_assignment(venv_dir)
            );
            let _ = wsl::run_script(distro, &rollback_script);
        }
        bail!("venv operation failed: {}", op_out.combined());
    }
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

    let smi = wsl::run_script(
        distro,
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1",
    );

    let torch_script = format!(
        r#"{venv_assignment}
"$venv_dir/bin/python" - << 'EOF'
import sys
try:
    import torch
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}", file=sys.stderr)
    sys.exit(10)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
EOF
"#,
        venv_assignment = venv_assignment(venv_dir)
    );
    let torch_out = wsl::run_script(distro, &torch_script);

    let report = match parse_verify_result(
        smi.ok,
        &smi.stdout,
        &smi.stderr,
        torch_out.ok,
        &torch_out.stdout,
        &torch_out.stderr,
        vllm_version,
        distro,
    ) {
        Ok(r) => {
            on_log(
                "verify",
                &format!(
                    "Verification passed: GPU '{}', VRAM {} MB, CUDA available, bf16: {}",
                    r.gpu_name.as_deref().unwrap_or("unknown"),
                    r.vram_mb.unwrap_or(0),
                    r.bf16_supported
                ),
            );
            r
        }
        Err(err) => {
            on_log("verify", &format!("Verification failed: {err}"));
            bail!("{err}");
        }
    };

    Ok(report)
}

/// Optional phase: Install CUDA build tools for FlashInfer JIT compilation.
/// This installs gcc, python dev headers, ninja-build, and the CUDA toolkit.
pub fn phase_cuda_build_tools(distro: &str, on_log: &mut impl FnMut(&str, &str)) -> Result<String> {
    let venv_dir = "~/llm-lp/.venv";
    phase_cuda_build_tools_with_venv(distro, venv_dir, on_log)
}

/// FlashInfer build tools phase with configurable venv_dir and dynamic sys.version_info querying.
pub fn phase_cuda_build_tools_with_venv(
    distro: &str,
    venv_dir: &str,
    on_log: &mut impl FnMut(&str, &str),
) -> Result<String> {
    on_log("cuda-tools", "Checking for existing CUDA build tools…");

    // Query python version dynamically from the configured venv
    let py_ver_cmd = format!(
        "{}\"$venv_dir/bin/python\" -c \"import sys; print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')\" 2>/dev/null || echo '3.12'",
        venv_assignment(venv_dir),
    );
    let py_ver_out = wsl::run_script(distro, &py_ver_cmd);
    let py_ver = py_ver_out.stdout.trim().to_string();
    let py_ver = if py_ver.is_empty() { "3.12".to_string() } else { py_ver };

    let check_script = format!(
        r#"{venv_assignment}
which nvcc >/dev/null 2>&1 && echo "nvcc=yes" || echo "nvcc=no"
which gcc >/dev/null 2>&1 && echo "gcc=yes" || echo "gcc=no"
which ninja >/dev/null 2>&1 && echo "ninja=yes" || echo "ninja=no"
test -f /usr/include/python{py_ver}/Python.h && echo "python_dev=yes" || ( "$venv_dir/bin/python" -c "import sysconfig, os; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('include'), 'Python.h')) else 1)" >/dev/null 2>&1 && echo "python_dev=yes" || echo "python_dev=no" )
"#,
        venv_assignment = venv_assignment(venv_dir),
        py_ver = py_ver
    );
    let check = wsl::run_script(distro, &check_script);

    let mut nvcc = false;
    let mut gcc = false;
    let mut ninja = false;
    let mut python_dev = false;
    for line in check.stdout.lines() {
        if line.starts_with("nvcc=yes") {
            nvcc = true;
        } else if line.starts_with("gcc=yes") {
            gcc = true;
        } else if line.starts_with("ninja=yes") {
            ninja = true;
        } else if line.starts_with("python_dev=yes") {
            python_dev = true;
        }
    }

    if nvcc && gcc && ninja && python_dev {
        on_log(
            "cuda-tools",
            "All CUDA build tools already installed (skipping).",
        );
        return Ok("cuda-tools".into());
    }

    on_log(
        "cuda-tools",
        &format!("Installing CUDA build tools (gcc, python{py_ver}-dev, ninja-build, CUDA toolkit)…"),
    );
    on_log("cuda-tools", "⚠ This downloads several GB and may take 5-15 minutes. First FlashInfer JIT compile will be slow.");

    // Query PyTorch CUDA version using configured venv
    let torch_cuda_cmd = format!(
        "{}\"$venv_dir/bin/python\" -c \"import torch; print(torch.version.cuda)\" 2>/dev/null || echo 'unknown'",
        venv_assignment(venv_dir),
    );
    let torch_cuda = wsl::run_script(distro, &torch_cuda_cmd);
    let torch_cuda_version = torch_cuda.stdout.trim();
    let cuda_major = if torch_cuda_version != "unknown" && !torch_cuda_version.is_empty() {
        torch_cuda_version.split('.').next().unwrap_or("12")
    } else {
        "12"
    };

    on_log(
        "cuda-tools",
        &format!("Targeting CUDA toolkit major version: {cuda_major} (matching PyTorch)"),
    );

    // Install base build tools with dynamic python dev package
    let apt_script = format!(
        r#"
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y -qq
        apt-get install -y -qq gcc python{py_ver}-dev ninja-build wget gpg
        "#
    );
    let out = wsl::run_script_root(distro, &apt_script);
    if !out.ok {
        on_log(
            "cuda-tools",
            &format!("Failed to install base build tools: {}", out.combined()),
        );
        bail!("CUDA build tools installation failed: {}", out.combined());
    }
    on_log(
        "cuda-tools",
        &format!("Base build tools (gcc, python{py_ver}-dev, ninja-build) installed."),
    );

    // Add NVIDIA CUDA repository for WSL-Ubuntu
    let repo_script = format!(
        r#"
        set -e
        export DEBIAN_FRONTEND=noninteractive
        UBUNTU_VERSION=$(lsb_release -rs)
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/ubuntu${{UBUNTU_VERSION//./}}/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update -y -qq
        apt-get install -y -qq cuda-toolkit-{cuda_major}-0
        "#
    );
    on_log(
        "cuda-tools",
        "Adding NVIDIA CUDA repository and installing CUDA toolkit…",
    );
    let out = wsl::run_script_root(distro, &repo_script);
    if !out.ok {
        on_log(
            "cuda-tools",
            &format!(
                "CUDA toolkit install failed (may need manual intervention): {}",
                out.combined()
            ),
        );
    } else {
        on_log("cuda-tools", "CUDA toolkit installed successfully.");
    }

    let verify = wsl::run_script(distro, "which nvcc && nvcc --version | head -1");
    if verify.ok && verify.stdout.contains("nvcc") {
        on_log(
            "cuda-tools",
            &format!(
                "nvcc verified: {}",
                verify.stdout.lines().next().unwrap_or("ok")
            ),
        );
    } else {
        on_log("cuda-tools", "WARNING: nvcc not found in PATH after install. May need shell restart or manual PATH setup.");
    }

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

    #[test]
    fn test_venv_assignment_ends_with_newline() {
        let snippet = venv_assignment("~/llm-lp/.venv");
        assert!(snippet.ends_with('\n'));
        assert!(snippet.contains("venv_dir=$(__llm_panel_expand_tilde '~/llm-lp/.venv')\n"));
    }

    #[test]
    fn test_parse_verify_result_success() {
        let smi_out = "NVIDIA GeForce RTX 4090, 24564 MiB, 560.94\n";
        let torch_out = "2.5.1+cu124\nTrue\nNVIDIA GeForce RTX 4090\n25757220864\nTrue\n";
        let res = parse_verify_result(
            true, smi_out, "",
            true, torch_out, "",
            Some("0.7.3".into()),
            "Ubuntu-24.04",
        ).expect("should succeed with working GPU");

        assert_eq!(res.vllm_version.as_deref(), Some("0.7.3"));
        assert_eq!(res.torch_version.as_deref(), Some("2.5.1+cu124"));
        assert!(res.cuda_available);
        assert_eq!(res.gpu_name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(res.vram_mb, Some(24564));
        assert!(res.bf16_supported);
    }

    #[test]
    fn test_parse_verify_result_driver_not_visible() {
        let smi_err = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.";
        let torch_out = "2.5.1+cu124\nFalse\nn/a\n0\nFalse\n";
        let err = parse_verify_result(
            false, "", smi_err,
            true, torch_out, "",
            None,
            "Ubuntu-24.04",
        ).unwrap_err();

        match err {
            VerifyFailure::DriverNotVisible { diagnostic } => {
                assert!(diagnostic.contains("NVIDIA-SMI has failed"));
            }
            other => panic!("expected DriverNotVisible, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_verify_result_no_gpu_in_wsl() {
        let smi_out = "";
        let torch_out = "2.5.1+cu124\nFalse\nn/a\n0\nFalse\n";
        let err = parse_verify_result(
            true, smi_out, "",
            true, torch_out, "",
            None,
            "Ubuntu-24.04",
        ).unwrap_err();

        match err {
            VerifyFailure::NoGpuInWsl { .. } => {}
            other => panic!("expected NoGpuInWsl, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_verify_result_torch_import_failed() {
        let smi_out = "NVIDIA GeForce RTX 4090, 24564 MiB, 560.94\n";
        let torch_err = "ModuleNotFoundError: No module named 'torch'";
        let err = parse_verify_result(
            true, smi_out, "",
            false, "", torch_err,
            None,
            "Ubuntu-24.04",
        ).unwrap_err();

        match err {
            VerifyFailure::TorchImportFailed { diagnostic } => {
                assert!(diagnostic.contains("No module named 'torch'"));
            }
            other => panic!("expected TorchImportFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_verify_result_cuda_unavailable() {
        let smi_out = "NVIDIA GeForce RTX 4090, 24564 MiB, 560.94\n";
        let torch_out = "2.5.1+cpu\nFalse\nn/a\n0\nFalse\n";
        let err = parse_verify_result(
            true, smi_out, "",
            true, torch_out, "",
            None,
            "Ubuntu-24.04",
        ).unwrap_err();

        match err {
            VerifyFailure::CudaUnavailable { torch_version, .. } => {
                assert_eq!(torch_version.as_deref(), Some("2.5.1+cpu"));
            }
            other => panic!("expected CudaUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn test_runtime_manifest_consistency() {
        assert_eq!(TESTED_RUNTIME_MANIFEST.python_version, "3.12");
        assert_eq!(TESTED_RUNTIME_MANIFEST.vllm_version, "0.7.3");
        assert_eq!(TESTED_RUNTIME_MANIFEST.torch_version, "2.5.1");
        assert!(TESTED_RUNTIME_MANIFEST.min_cuda_driver >= (12, 0));
    }
}
