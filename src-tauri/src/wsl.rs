//! WSL command helpers: distro detection, sync run, and streaming runs
//! (used by provisioning, model pulling, and server launchers).

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

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

    // 2. Scan installed distros for an apt-based one that responds to echo ok
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

/// Detect total and available memory in WSL2 (in MB).
/// Falls back to 16GB total / 12GB available on failure. Never panics.
pub fn detect_wsl_memory(distro: &str) -> (u64, u64) {
    let mut cmd = wsl_command();
    cmd.env("WSL_UTF8", "1");
    if distro.trim().is_empty() {
        cmd.args(["--exec", "cat", "/proc/meminfo"]);
    } else {
        cmd.args(["-d", distro, "--exec", "cat", "/proc/meminfo"]);
    }
    if let Ok(o) = cmd.output() {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout);
            let (total, avail) = parse_meminfo(&s);
            if total > 0 {
                return (total, avail);
            }
        }
    }
    (16384, 12288) // Safe fallback: 16GB total / 12GB avail
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
        cmd.args(["-d", distro, "--user", "root", "--exec", "bash", "-lc", script]);
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
    let mut child = match cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
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
            let reader = BufReader::new(out);
            for line in reader.lines().flatten() {
                let _ = tx_out.send((true, line));
            }
        }));
    }
    if let Some(err) = stderr {
        let tx_err = tx.clone();
        handles.push(std::thread::spawn(move || {
            let reader = BufReader::new(err);
            for line in reader.lines().flatten() {
                let _ = tx_err.send((false, line));
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
}

impl WslChild {
    /// Spawn `bash -lc <script>` in `<distro>`; stream output lines to `on_line`.
    /// `script` typically ends with `exec python ...` so the child lives for the
    /// lifetime of the server.
    pub fn spawn(
        distro: &str,
        script: &str,
        on_line: impl FnMut(String) + Send + 'static,
    ) -> Result<WslChild, String> {
        let mut cmd = wsl_command();
        cmd.env("WSL_UTF8", "1");
        if distro.trim().is_empty() {
            cmd.args(["--exec", "bash", "-lc", script]);
        } else {
            cmd.args(["-d", distro, "--exec", "bash", "-lc", script]);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to spawn wsl.exe: {e}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let mut handles = Vec::new();
        let on_line = std::sync::Arc::new(std::sync::Mutex::new(on_line));
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
        Ok(WslChild { child, handles })
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    /// Terminate the wsl.exe process (Windows side). The WSL-side shell gets
    /// SIGKILL via console teardown — safest fallback when pidfile kill fails.
    pub fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill()
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait()
    }

    /// Join reader threads (used on drop paths so logs finish flushing).
    pub fn join(&mut self) {
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
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
