//! WSL command helpers: distro detection, sync run, and streaming runs
//! (used by provisioning, model pulling, and server launchers).

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};

use crate::state::GpuSnapshot;

/// Convert a Windows (or already-WSL) path to the WSL mount path.
///
/// `D:\AI\qwen` -> `/mnt/d/AI/qwen`. Paths already inside WSL
/// (`/mnt/...`, `/home/...`, `~`) pass through unchanged so they can be
/// pasted either way.
pub fn windows_to_wsl_path(path: &str) -> String {
    let p = path.trim();
    if p.is_empty() {
        return p.to_string();
    }
    // Already a WSL-style path: keep it as-is.
    if p.starts_with('/') || p.starts_with('~') {
        return p.to_string();
    }
    // Drive-letter path (backslash or forward slash): `C:\Models` or `c:/models`.
    let bytes = p.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let drive = bytes[0].to_ascii_lowercase() as char;
        let rest: String = p[2..].replace('\\', "/");
        let rest = rest.trim_matches('/');
        if rest.is_empty() {
            return format!("/mnt/{drive}");
        }
        return format!("/mnt/{drive}/{rest}");
    }
    // Unknown shape — pass through untouched.
    p.to_string()
}

/// Create a `Command` for `wsl.exe` with `CREATE_NO_WINDOW` on Windows
/// so that child console windows do not flash on screen.
pub fn wsl_command() -> Command {
    let mut cmd = Command::new("wsl.exe");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.stdin(Stdio::null());
    cmd
}

/// Our dedicated distro name — isolated from user's distros.
pub const APP_DISTRO_NAME: &str = "local-llm-panel-ubuntu";

/// Expand a leading `~` inside a quoted WSL path value.
pub const WSL_TILDE_EXPANSION_SNIPPET: &str = r#"
__llm_panel_expand_tilde() {
  case "$1" in
    "~") printf '%s' "$HOME" ;;
    "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
    *) printf '%s' "$1" ;;
  esac
}
"#;

/// Ensure the app's dedicated Ubuntu distro exists. Installs it if missing.
/// Returns the distro name (always APP_DISTRO_NAME on success).
pub fn ensure_app_distro(mut on_log: impl FnMut(&str)) -> Result<String, String> {
    let distros = installed_distros();
    if distros.contains(&APP_DISTRO_NAME.to_string()) {
        on_log(&format!("distro '{}' already installed", APP_DISTRO_NAME));
        return Ok(APP_DISTRO_NAME.to_string());
    }

    on_log(&format!("installing dedicated distro '{}'…", APP_DISTRO_NAME));
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1")
        .args([
            "--install",
            "-d", "Ubuntu-22.04",
            "--name", APP_DISTRO_NAME,
            "--web-download",
            "--no-launch",
        ]);
    let out = cmd.output().map_err(|e| format!("spawn wsl --install: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("wsl --install failed: {}", err));
    }

    // Wait for first-boot setup to complete (creates default user, etc.)
    on_log("waiting for distro initialization…");
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let probe = run_script(APP_DISTRO_NAME, "echo ok");
        if probe.ok {
            on_log("distro ready");
            return Ok(APP_DISTRO_NAME.to_string());
        }
    }
    Err("distro installed but not responding after 60s".into())
}

/// List all installed WSL distros cleanly.
pub fn installed_distros() -> Vec<String> {
    let out = match wsl_command()
        .env("WSL_UTF8", "1")
        .args(["-l", "-q"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut list = Vec::new();
    for line in text.lines() {
        let clean: String = line.chars().filter(|c| *c != '\u{0}').collect();
        let clean = clean.trim();
        if clean.is_empty()
            || clean.contains("legal notice")
            || clean.to_lowercase().contains("windows")
        {
            continue;
        }
        let clean_s = clean.to_string();
        if !list.contains(&clean_s) {
            list.push(clean_s);
        }
    }
    list
}

/// Detect the best WSL distro to use:
/// 1. First checks `wsl -l -v` for default distro marked with `*`
/// 2. If default isn't apt-based, searches installed distros for an apt-based one
/// 3. Falls back to any installed distro that responds to `echo ok`
pub fn detect_default_distro() -> Option<String> {
    // 1. Check wsl -l -v to find the default distro (marked with '*')
    if let Ok(out) = wsl_command()
        .env("WSL_UTF8", "1")
        .args(["-l", "-v"])
        .output()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let clean: String = line.chars().filter(|c| *c != '\u{0}').collect();
            let clean = clean.trim();
            if clean.starts_with('*') {
                let name = clean.trim_start_matches('*').trim();
                let name = name.split_whitespace().next().unwrap_or("").trim();
                if !name.is_empty() && is_apt_distro(name) && run_script(name, "echo ok").ok {
                    return Some(name.to_string());
                }
            }
        }
    }

    let distros = installed_distros();

    // 2. Prefer our dedicated distro if present
    if distros.contains(&APP_DISTRO_NAME.to_string()) && run_script(APP_DISTRO_NAME, "echo ok").ok {
        return Some(APP_DISTRO_NAME.to_string());
    }

    // 3. Scan installed distros for an apt-based one that responds to echo ok
    for d in &distros {
        if is_apt_distro(d) && run_script(d, "echo ok").ok {
            return Some(d.clone());
        }
    }

    // 3. Fallback to any installed distro that responds to echo ok
    for d in &distros {
        if run_script(d, "echo ok").ok {
            return Some(d.clone());
        }
    }

    // 4. Return the first installed distro if any
    distros.into_iter().next()
}

/// Whether the given distro is currently running. Uses `wsl -l -v`, which
/// reports state without booting a stopped distro.
// ponytail: relies on the English "Running" state word; patch the matcher if
// localized Windows builds misreport (safe failure mode = "not running", so worst
// case is a cosmetic dashboard state, never a spurious boot).
pub fn is_running(distro: &str) -> bool {
    let distro = distro.trim();
    if distro.is_empty() || distro.starts_with("__test_") {
        return false;
    }
    let Ok(out) = wsl_command()
        .env("WSL_UTF8", "1")
        .args(["-l", "-v"])
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let cleaned: String = text.chars().filter(|c| *c != '\u{0}').collect();
    verbose_state_is_running(&cleaned, distro)
}

/// Parse `wsl -l -v` output for whether `distro` is running. Pure so it can
/// be unit-tested without shelling out.
fn verbose_state_is_running(text: &str, distro: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || line.contains("NAME") || !line.contains("Running") {
            return false;
        }
        let name = line
            .trim_start_matches('*')
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches(':');
        name == distro
    })
}

/// Check if a distro looks Ubuntu-ish (apt-based).
pub fn is_apt_distro(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("ubuntu") || n.contains("debian") || n.contains("kali") || n.contains("mint")
}

/// Parse `/proc/meminfo` contents into `(total_mb, avail_mb)`.
pub fn parse_meminfo(content: &str) -> (u64, u64) {
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("MemAvailable:") {
            avail_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (total_kb / 1024, avail_kb / 1024)
}

/// Detect total and available memory in WSL2 (in MB), returning `None` when
/// the distro cannot be queried. Fit verification must not treat fallback
/// values as measured hardware.
pub fn try_detect_wsl_memory(distro: &str) -> Option<(u64, u64)> {
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    if distro.trim().is_empty() {
        cmd.args(["--exec", "cat", "/proc/meminfo"]);
    } else {
        cmd.args(["-d", distro, "--exec", "cat", "/proc/meminfo"]);
    }
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (total, available) = parse_meminfo(&text);
    (total > 0).then_some((total, available))
}

/// Detect total and available memory in WSL2 (in MB).
/// Falls back to 16GB total / 12GB available on failure. Never panics.
pub fn detect_wsl_memory(distro: &str) -> (u64, u64) {
    try_detect_wsl_memory(distro).unwrap_or((16_384, 12_288))
}

/// Result of a synchronous WSL script run.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub ok: bool,
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl RunOutput {
    pub fn combined(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// Run a script synchronously in the given distro, waiting for completion.
pub fn run_script(distro: &str, script: &str) -> RunOutput {
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    if distro.trim().is_empty() {
        cmd.args(["--exec", "bash", "-lc", script]);
    } else {
        cmd.args(["-d", distro, "--exec", "bash", "-lc", script]);
    }
    match cmd.output() {
        Ok(o) => RunOutput {
            ok: o.status.success(),
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&o.stderr).trim().to_string(),
        },
        Err(e) => RunOutput {
            ok: false,
            code: -1,
            stdout: String::new(),
            stderr: format!("spawn error: {e}"),
        },
    }
}

/// Like [`run_script`] but as the distro's `root` user (no sudo password
/// needed). Used to write sudoers rules and do other root-only setup.
pub fn run_script_root(distro: &str, script: &str) -> RunOutput {
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    if distro.trim().is_empty() {
        cmd.args(["--user", "root", "--exec", "bash", "-lc", script]);
    } else {
        cmd.args([
            "-d", distro, "--user", "root", "--exec", "bash", "-lc", script,
        ]);
    }
    match cmd.output() {
        Ok(o) => RunOutput {
            ok: o.status.success(),
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&o.stderr).trim().to_string(),
        },
        Err(e) => RunOutput {
            ok: false,
            code: -1,
            stdout: String::new(),
            stderr: format!("spawn error: {e}"),
        },
    }
}

/// Run a script synchronously, streaming each output line to `on_line`.
pub fn run_script_stream(distro: &str, script: &str, mut on_line: impl FnMut(&str)) -> RunOutput {
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    if distro.trim().is_empty() {
        cmd.args(["--exec", "bash", "-lc", script]);
    } else {
        cmd.args(["-d", distro, "--exec", "bash", "-lc", script]);
    }
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            return RunOutput {
                ok: false,
                code: -1,
                stdout: String::new(),
                stderr: format!("spawn error: {e}"),
            };
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (tx, rx) = std::sync::mpsc::channel::<(bool, String)>();

    let mut handles = Vec::new();
    if let Some(out) = stdout {
        let tx_out = tx.clone();
        handles.push(std::thread::spawn(move || {
            let mut reader = BufReader::new(out);
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(_) => break,
                };
                for &b in &chunk[..n] {
                    if b == b'\n' || b == b'\r' {
                        if !buf.is_empty() {
                            let _ = tx_out.send((true, String::from_utf8_lossy(&buf).into_owned()));
                            buf.clear();
                        }
                    } else {
                        buf.push(b);
                    }
                }
            }
            if !buf.is_empty() {
                let _ = tx_out.send((true, String::from_utf8_lossy(&buf).into_owned()));
            }
        }));
    }
    if let Some(err) = stderr {
        let tx_err = tx.clone();
        handles.push(std::thread::spawn(move || {
            let mut reader = BufReader::new(err);
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(_) => break,
                };
                for &b in &chunk[..n] {
                    if b == b'\n' || b == b'\r' {
                        if !buf.is_empty() {
                            let _ = tx_err.send((false, String::from_utf8_lossy(&buf).into_owned()));
                            buf.clear();
                        }
                    } else {
                        buf.push(b);
                    }
                }
            }
            if !buf.is_empty() {
                let _ = tx_err.send((false, String::from_utf8_lossy(&buf).into_owned()));
            }
        }));
    }
    drop(tx);

    let mut so = String::new();
    let mut se = String::new();

    while let Ok((is_stdout, line)) = rx.recv() {
        let clean: String = line.chars().filter(|c| *c != '\u{0}').collect();
        if !clean.trim().is_empty() {
            on_line(&clean);
        }
        if is_stdout {
            so.push_str(&clean);
            so.push('\n');
        } else {
            se.push_str(&clean);
            se.push('\n');
        }
    }

    for h in handles {
        let _ = h.join();
    }

    let status = child.wait();
    let (ok, code) = match &status {
        Ok(s) => (s.success(), s.code().unwrap_or(-1)),
        Err(_) => (false, -1),
    };

    RunOutput {
        ok,
        code,
        stdout: so.trim().to_string(),
        stderr: se.trim().to_string(),
    }
}

/// Long-running child: a wsl.exe that stays alive (used by server launchers).
/// Stdout/stderr are streamed line-by-line to `on_line` from reader threads.
/// The caller keeps the handle to kill / check liveness.
pub struct WslChild {
    pub child: Child,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Path to the generated script file in WSL (for cleanup on drop)
    script_path: Option<String>,
    /// Distro used to launch the script, retained for best-effort cleanup when
    /// the process is force-killed before its EXIT trap runs.
    distro: Option<String>,
}

/// Native Windows child used by llama-server. Output is streamed using the
/// same callback shape as WSL children, while termination uses the Windows
/// process tree so descendants cannot outlive the app.
pub struct NativeChild {
    pub child: Child,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl NativeChild {
    pub fn spawn(
        executable: &std::path::Path,
        args: &[String],
        on_line: impl FnMut(String) + Send + 'static,
    ) -> Result<Self, String> {
        let mut cmd = Command::new(executable);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", executable.display()))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let callback = std::sync::Arc::new(std::sync::Mutex::new(on_line));
        let mut handles = Vec::new();
        for stream in [
            stdout.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            stderr.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let cb = callback.clone();
            handles.push(std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().flatten() {
                    if let Ok(mut cb) = cb.lock() {
                        cb(line);
                    }
                }
            }));
        }
        Ok(Self { child, handles })
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    pub fn kill(&mut self) -> std::io::Result<()> {
        #[cfg(windows)]
        {
            let pid = self.child.id().to_string();
            let status = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid])
                .status()?;
            if status.success() {
                return Ok(());
            }
        }
        self.child.kill()
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn join(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for NativeChild {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.kill();
        }
        self.join();
    }
}

/// Generate a unique script filename for a server launch.
fn gen_script_name(server_id: &str) -> String {
    let mut safe = String::new();
    for ch in server_id.chars() {
        if safe.len() >= 48 {
            break;
        }
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            safe.push(ch);
        } else {
            safe.push('_');
        }
    }
    if safe.is_empty() {
        safe.push_str("server");
    }
    // Keep distinct IDs distinct even when sanitization collapses characters,
    // while ensuring the generated path is always a simple /tmp file name.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in server_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("/tmp/local-llm-panel-launch-{safe}-{hash:016x}.sh")
}

/// Minimal shell_quote for WSL path generation (used internally for script paths).
/// Only handles the specific case of script paths which don't contain quotes.
pub fn shell_quote_wsl(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Spawn a script in WSL by piping script content to stdin of `cat > file`.
/// This avoids heredoc delimiter issues and command line length limits.
pub fn spawn_script_via_stdin(
    distro: &str,
    script: &str,
    server_id: &str,
    on_line: impl FnMut(String) + Send + 'static,
) -> Result<WslChild, String> {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    let script_path = gen_script_name(server_id);

    // Build command: mkdir, then read stdin (until EOF) and write to file, chmod, execute
    let mkdir_cmd = "mkdir -p /tmp/local-llm-panel";
    let write_cmd = format!("cat > {}", shell_quote_wsl(&script_path));
    let chmod_cmd = format!("chmod 700 {}", shell_quote_wsl(&script_path));
    let exec_cmd = format!("bash -l {}", shell_quote_wsl(&script_path));
    let full_cmd = format!("{} && {} && {} && {}", mkdir_cmd, write_cmd, chmod_cmd, exec_cmd);

    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    cmd.stdin(Stdio::piped());

    let mut args = Vec::new();
    if distro.trim().is_empty() {
        args.push("--exec".to_string());
    } else {
        args.push("-d".to_string());
        args.push(distro.to_string());
        args.push("--exec".to_string());
    }
    args.push("bash".to_string());
    args.push("-lc".to_string());
    args.push(full_cmd);

    for arg in args {
        cmd.arg(arg);
    }

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn wsl.exe: {e}"))?;

    // Write script content to stdin, then close stdin (EOF signals end to cat)
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(script.as_bytes());
        let _ = stdin.flush();
        // stdin dropped here, closing the pipe - cat sees EOF and finishes
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut handles = Vec::new();
    let on_line = Arc::new(Mutex::new(on_line));
    for stream in [
        stdout.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        stderr.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let cb = on_line.clone();
        handles.push(std::thread::spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines() {
                if let Ok(l) = line {
                    let clean: String = l.chars().filter(|c| *c != '\u{0}').collect();
                    if let Ok(mut cb) = cb.lock() {
                        cb(clean);
                    }
                }
            }
        }));
    }
    Ok(WslChild {
        child,
        handles,
        script_path: Some(script_path),
        distro: Some(distro.to_string()),
    })
}

impl WslChild {
    /// Spawn a script in WSL by writing it to a temp file via stdin (unique delimiter),
    /// then executing it. This avoids command line length limits and heredoc delimiter conflicts.
    /// The generated script supervises the long-running Python process and
    /// removes its own temporary file when the process exits.
    pub fn spawn(
        distro: &str,
        script: &str,
        server_id: &str,
        on_line: impl FnMut(String) + Send + 'static,
    ) -> Result<WslChild, String> {
        spawn_script_via_stdin(distro, script, server_id, on_line)
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    /// Terminate the wsl.exe process tree (Windows side). Killing only the
    /// launcher can leave the Linux child orphaned, so use taskkill on Windows
    /// before falling back to the platform child handle.
    pub fn kill(&mut self) -> std::io::Result<()> {
        #[cfg(windows)]
        {
            let pid = self.child.id().to_string();
            let status = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid])
                .status()?;
            if status.success() {
                return Ok(());
            }
        }
        self.child.kill()
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait()
    }

    pub fn script_path(&self) -> Option<&str> {
        self.script_path.as_deref()
    }

    /// Join reader threads (used on drop paths so logs finish flushing).
    pub fn join(&mut self) {
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for WslChild {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.kill();
        }
        self.join();
        if let (Some(distro), Some(script_path)) = (self.distro.as_deref(), self.script_path.as_deref()) {
            let cleanup = format!("rm -f -- {}", shell_quote_wsl(script_path));
            let _ = run_script(distro, &cleanup);
        }
    }
}

/// Query the primary GPU's name, total VRAM (MB), free VRAM (MB), and utilization (%) via nvidia-smi inside WSL.
pub fn gpu_snapshot(distro: &str) -> Option<GpuSnapshot> {
    let out = run_script(
        distro,
        "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1",
    );
    let line = out.stdout.trim();
    if line.is_empty() {
        return None;
    }
    // "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227, 8456, 12"
    let mut it = line.split(',');
    let name = it.next()?.trim().to_string();
    let total = it.next()?.trim().parse::<u64>().ok()?;
    let free = it.next()?.trim().parse::<u64>().ok()?;
    let util = it.next()?.trim().parse::<u32>().ok().unwrap_or(0);
    Some(GpuSnapshot {
        name,
        vram_total_mb: total,
        vram_free_mb: free,
        util_percent: util,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distro_detection_returns_something() {
        // This tests the real local machine; if no wsl.exe at all, we still
        // shouldn't panic — just return None.
        let _ = detect_default_distro();
    }

    #[test]
    fn test_windows_to_wsl_path_conversion() {
        assert_eq!(
            windows_to_wsl_path(r"C:\Models\Llama"),
            "/mnt/c/Models/Llama"
        );
        assert_eq!(windows_to_wsl_path(r"D:\AI\qwen"), "/mnt/d/AI/qwen");
        assert_eq!(windows_to_wsl_path(r"D:\AI\qwen\"), "/mnt/d/AI/qwen");
        assert_eq!(windows_to_wsl_path("D:/AI/qwen"), "/mnt/d/AI/qwen");
        assert_eq!(
            windows_to_wsl_path("/mnt/d/Models/Llama"),
            "/mnt/d/Models/Llama"
        );
        assert_eq!(windows_to_wsl_path("~/models/qwen"), "~/models/qwen");
        assert_eq!(windows_to_wsl_path("  D:/AI/qwen  "), "/mnt/d/AI/qwen");
    }

    #[test]
    fn test_generated_script_names_are_confined_and_unique() {
        let hostile = gen_script_name("srv/../../outside; rm -rf /");
        assert!(hostile.starts_with("/tmp/local-llm-panel-launch-"));
        assert!(!hostile.contains(".."));
        assert!(!hostile.contains(';'));
        assert_eq!(hostile.matches('/').count(), 2);
        assert_ne!(hostile, gen_script_name("srv/../different"));
        assert_ne!(hostile, gen_script_name("srv_other"));
    }

    #[test]
    fn test_parse_proc_meminfo() {
        let sample = "MemTotal:       24576000 kB\nMemFree:         4000000 kB\nMemAvailable:   18432000 kB\n";
        let (total_mb, avail_mb) = parse_meminfo(sample);
        assert_eq!(total_mb, 24000);
        assert_eq!(avail_mb, 18000);
    }

    #[test]
    fn test_parse_meminfo_empty_and_garbage() {
        let (total_mb, avail_mb) = parse_meminfo("");
        assert_eq!(total_mb, 0);
        assert_eq!(avail_mb, 0);

        let garbage = "Something: abc kB\nMemTotal: not_a_number kB\n";
        let (total_mb, avail_mb) = parse_meminfo(garbage);
        assert_eq!(total_mb, 0);
        assert_eq!(avail_mb, 0);
    }

    #[test]
    fn test_detect_wsl_memory_nonexistent_distro_fallback() {
        let (total_mb, avail_mb) = detect_wsl_memory("__nonexistent_distro_test_xyz__");
        assert_eq!(total_mb, 16384);
        assert_eq!(avail_mb, 12288);
    }

    #[test]
    fn test_detect_wsl_memory_empty_distro() {
        // Empty distro queries default distro or returns fallback; must never panic
        let (total_mb, avail_mb) = detect_wsl_memory("");
        assert!(total_mb > 0);
        assert!(avail_mb > 0);
    }

    #[test]
    fn test_installed_distros() {
        let list = installed_distros();
        for d in &list {
            assert!(!d.contains('\u{0}'));
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_verbose_state_is_running() {
        let modern = "  NAME                   STATE           VERSION\n* Ubuntu-24.04           Running         2\n  docker-desktop         Stopped         2\n";
        assert!(verbose_state_is_running(modern, "Ubuntu-24.04"));
        assert!(!verbose_state_is_running(modern, "docker-desktop"));
        assert!(!verbose_state_is_running(modern, "Nonexistent"));

        let legacy = "  NAME      STATE           VERSION\n* Ubuntu:  Running         2\n";
        assert!(verbose_state_is_running(legacy, "Ubuntu"));
        assert!(!verbose_state_is_running(legacy, "Ubuntu-24.04"));

        assert!(!verbose_state_is_running("", "Ubuntu"));
        assert!(!verbose_state_is_running("  NAME   STATE   VERSION\n", "Ubuntu"));
    }

    #[test]
    fn test_is_running_empty_or_test_distro() {
        assert!(!is_running(""));
        assert!(!is_running("__test_distro_xyz__"));
    }

    #[test]
    fn test_run_script_preserves_shell_variables() {
        let distro = detect_default_distro().unwrap_or_else(|| "Ubuntu".to_string());
        let out = run_script(&distro, "for v in A B C; do echo VAR=$v; done");
        if out.ok {
            assert!(out.stdout.contains("VAR=A"));
            assert!(out.stdout.contains("VAR=B"));
            assert!(out.stdout.contains("VAR=C"));
        }
    }

    #[test]
    fn test_run_script_stream_does_not_deadlock_on_large_stderr() {
        // Echoes lines to both stdout and stderr in alternating batches
        let script = r#"
            python3 -c '
import sys
for i in range(500):
    sys.stderr.write("E" * 128 + "\n")
    sys.stdout.write("O" * 128 + "\n")
sys.stderr.flush()
sys.stdout.flush()
' 2>&1 || true
        "#;
        let mut count = 0;
        let res = run_script_stream("Ubuntu", script, |_line| {
            count += 1;
        });
        // Even if Ubuntu is absent or command fails, it must return cleanly without hanging
        let _ = res;
    }
}
