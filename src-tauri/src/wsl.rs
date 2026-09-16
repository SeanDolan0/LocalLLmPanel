//! WSL command helpers: distro detection, sync run, and streaming runs
//! (used by provisioning, model pulling, and server launchers).

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

/// Detect the default WSL distro from `wsl -l -q`.
pub fn detect_default_distro() -> Option<String> {
    let out = Command::new("wsl.exe")
        .env("WSL_UTF8", "1")
        .args(["-l", "-q"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let clean: String = line.chars().filter(|c| *c != '\u{0}').collect();
        let clean = clean.trim();
        if clean.is_empty() || clean.contains("legal notice") || clean.to_lowercase().contains("windows") {
            continue;
        }
        return Some(clean.to_string());
    }
    None
}

/// Check if a distro looks Ubuntu-ish (apt-based).
pub fn is_apt_distro(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("ubuntu") || n.contains("debian") || n.contains("kali") || n.contains("mint")
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
    let mut cmd = Command::new("wsl.exe");
    cmd.env("WSL_UTF8", "1");
    cmd.args(["-d", distro, "--", "bash", "-lc", script]);
    match cmd.output() {
        Ok(o) => RunOutput {
            ok: o.status.success(),
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&o.stderr).trim().to_string(),
        },
        Err(e) => RunOutput { ok: false, code: -1, stdout: String::new(), stderr: format!("spawn error: {e}") },
    }
}

/// Like [`run_script`] but as the distro's `root` user (no sudo password
/// needed). Used to write sudoers rules and do other root-only setup.
pub fn run_script_root(distro: &str, script: &str) -> RunOutput {
    let mut cmd = Command::new("wsl.exe");
    cmd.env("WSL_UTF8", "1");
    cmd.args(["-d", distro, "--user", "root", "--", "bash", "-lc", script]);
    match cmd.output() {
        Ok(o) => RunOutput {
            ok: o.status.success(),
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&o.stderr).trim().to_string(),
        },
        Err(e) => RunOutput { ok: false, code: -1, stdout: String::new(), stderr: format!("spawn error: {e}") },
    }
}

/// Run a script synchronously, streaming each output line to `on_line`.
pub fn run_script_stream(
    distro: &str,
    script: &str,
    mut on_line: impl FnMut(&str),
) -> RunOutput {
    let mut child = match Command::new("wsl.exe")
        .env("WSL_UTF8", "1")
        .args(["-d", distro, "--", "bash", "-lc", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return RunOutput { ok: false, code: -1, stdout: String::new(), stderr: format!("spawn error: {e}") };
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (mut so, mut se) = (String::new(), String::new());

    // Drain one stream, splitting on \n (vLLM/HF progress uses \r — those
    // lines are flushed as-is so the UI can show them live).
    fn drain<R: std::io::Read>(reader: R, on_line: &mut dyn FnMut(&str)) -> String {
        let mut out = String::new();
        let mut partial = String::new();
        for b in reader.bytes() {
            let Ok(b) = b else { break };
            if b == b'\n' {
                out.push_str(&partial);
                out.push('\n');
                if !partial.trim().is_empty() {
                    on_line(&partial);
                }
                partial.clear();
            } else if b == b'\r' {
                if !partial.trim().is_empty() {
                    on_line(&partial);
                }
                partial.clear();
            } else {
                partial.push(b as char);
            }
        }
        if !partial.trim().is_empty() {
            out.push_str(&partial);
        }
        out
    }

    if let Some(out) = stdout {
        so = drain(out, &mut on_line);
    }
    if let Some(err) = stderr {
        se = drain(err, &mut on_line);
    }

    // Read the remainder (leftover lines not yet flushed after drain).
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
        let mut child = Command::new("wsl.exe")
            .env("WSL_UTF8", "1")
            .args(["-d", distro, "--", "bash", "-lc", script])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to spawn wsl.exe: {e}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let mut handles = Vec::new();
        let on_line = std::sync::Arc::new(std::sync::Mutex::new(on_line));
        for stream in [stdout.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), stderr.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)]
            .into_iter()
            .flatten()
        {
            let cb = on_line.clone();
            handles.push(std::thread::spawn(move || {
                let reader = BufReader::new(stream);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        if let Ok(mut cb) = cb.lock() {
                            cb(l);
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
}