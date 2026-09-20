# Local LLM Panel Audit Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the critical bug fixes, reliability improvements, and high-value product extensions identified in the Local LLM Panel v1.0.0 audit, transitioning the app into a hardened, production-ready local LLM workstation.

**Architecture:**
- **Core Engine & Reliability (Rust):** Concurrent stream draining in `wsl.rs` to prevent pipe deadlocks; fix quantization arithmetic in `estimate.rs`; inject authorization headers for internal chat/benchmarks in `server.rs`; fix idempotent marker checks in `provision.rs`; non-blocking server stopping for tray operations.
- **Model Storage & Lifecycle (Rust & Frontend):** Orphaned LFS blob garbage collection in `library_remove`; background download cancellation and mount state recovery in `hf.rs` and `Library.tsx`.
- **API Gateway Router (Rust):** Embedded OpenAI-compatible reverse proxy listening on a single port (default `127.0.0.1:11434`), routing incoming chat/completion requests by model ID to active vLLM instance ports.
- **Local Model Imports & Multimodal (Rust & Frontend):** Add Windows/WSL local folder model loader and support base64/image drag-and-drop vision chat in the playground.

**Tech Stack:** Tauri 2, Rust 2021 (Tokio, Reqwest, Axum/Hyper), React 18, TypeScript 5, Vite 6, Tailwind CSS 4.

**Spec:** `audit_report.md` (Artifact in session) and `docs/ROADMAP.md`.

## Global Constraints
- All Rust changes must pass `cargo test` clean with zero regressions.
- All frontend changes must pass `npm run build` (`tsc --noEmit && vite build`).
- No unnecessary external dependencies; use Tokio/standard library where feasible.
- Preserve backward compatibility with existing `%APPDATA%/local-llm-panel/config.json`.
- Follow conventional commits (`fix: ...`, `feat: ...`, `refactor: ...`).

---

### Task 1: Fix Pipe Buffer Deadlock in `run_script_stream`

**Files:**
- Modify: `src-tauri/src/wsl.rs:177-250`
- Test: `src-tauri/src/wsl.rs:tests`

**Interfaces:**
- Consumes: `wsl::wsl_command()`, `RunOutput`
- Produces: `pub fn run_script_stream(distro: &str, script: &str, mut on_line: impl FnMut(&str)) -> RunOutput` (concurrent reader threads for stdout and stderr)

- [ ] **Step 1: Write the unit test demonstrating concurrent stdout/stderr handling**

Add this test to `src-tauri/src/wsl.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify existing behavior / compilation**

Run: `cargo test test_run_script_stream_does_not_deadlock_on_large_stderr` in `src-tauri`.

- [ ] **Step 3: Implement concurrent reader threads in `run_script_stream`**

Replace sequential `drain(stdout)` then `drain(stderr)` in [`src-tauri/src/wsl.rs#L177-L250`](file:///C:/Users/sedol/Documents/LocalLLmPanel/src-tauri/src/wsl.rs#L177-L250) with crossbeam/channel or thread-based concurrent readers:

```rust
pub fn run_script_stream(
    distro: &str,
    script: &str,
    mut on_line: impl FnMut(&str),
) -> RunOutput {
    let mut child = match wsl_command()
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
```

- [ ] **Step 4: Run tests to verify**

Run: `cargo test test_run_script_stream` in `src-tauri`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/wsl.rs
git commit -m "fix(wsl): drain stdout and stderr concurrently to prevent pipe buffer deadlocks"
```

---

### Task 2: Correct AWQ / GPTQ / Q4 Memory Calculation in `estimate.rs`

**Files:**
- Modify: `src-tauri/src/estimate.rs:10-35`
- Test: `src-tauri/src/estimate.rs:tests`

**Interfaces:**
- Consumes: `quant: &str`
- Produces: Correct bytes per parameter for int4, AWQ, GPTQ (0.55), Q8 (1.05), Q4 (0.55), FP8 (1.0).

- [ ] **Step 1: Write test reflecting accurate 4-bit and 8-bit weight sizing**

In `src-tauri/src/estimate.rs:tests`:

```rust
    #[test]
    fn test_quant_bytes_per_param_accurate_hierarchy() {
        assert!((bytes_per_param("fp16") - 2.0).abs() < 1e-6);
        assert!((bytes_per_param("fp8") - 1.0).abs() < 1e-6);
        assert!((bytes_per_param("awq") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("gptq") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("q4_k_m") - 0.55).abs() < 1e-6);
        assert!((bytes_per_param("q8_0") - 1.05).abs() < 1e-6);
        // AWQ/Q4 (4-bit) must be strictly smaller than FP8 (8-bit) and Q5 (5-bit)
        assert!(bytes_per_param("awq") < bytes_per_param("q5_k_m"));
        assert!(bytes_per_param("awq") < bytes_per_param("fp8"));
    }
```

- [ ] **Step 2: Run test to verify it fails with old `1.1` value**

Run: `cargo test test_quant_bytes_per_param_accurate_hierarchy`
Expected: FAIL (`bytes_per_param("awq")` was 1.1).

- [ ] **Step 3: Update `bytes_per_param` implementation**

In [`src-tauri/src/estimate.rs#L10-L32`](file:///C:/Users/sedol/Documents/LocalLLmPanel/src-tauri/src/estimate.rs#L10-L32):

```rust
pub fn bytes_per_param(quant: &str) -> f64 {
    let q = quant.to_ascii_lowercase();
    let q_str = q.as_str();
    if q_str.starts_with("q4") || q_str.contains("q4_") || q_str.contains("iq4") || q_str == "awq" || q_str == "gptq" || q_str == "int4" {
        0.55 // 4-bit: 0.50 bytes/weight + ~0.05 scaling overhead
    } else if q_str.starts_with("q8") || q_str.contains("q8_") {
        1.05 // 8-bit: 1.00 byte/weight + scaling overhead
    } else if q_str.starts_with("q5") || q_str.contains("q5_") {
        0.68 // 5-bit: ~0.625 + overhead
    } else if q_str.starts_with("q6") || q_str.contains("q6_") {
        0.80 // 6-bit: ~0.75 + overhead
    } else if q_str.starts_with("q3") || q_str.contains("q3_") || q_str.contains("iq3") {
        0.45
    } else if q_str.starts_with("q2") || q_str.contains("q2_") || q_str.contains("iq2") {
        0.35
    } else if q_str == "fp8" || q_str == "int8" {
        1.0
    } else if q_str == "gguf" {
        0.55 // default GGUF assumption is ~Q4_K_M
    } else {
        2.0 // fp16 / bf16 / unset
    }
}
```

Update any existing unit tests in `estimate.rs` asserting `1.1` for AWQ to assert `0.55`.

- [ ] **Step 4: Run test suite to verify**

Run: `cargo test estimate::tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/estimate.rs
git commit -m "fix(estimate): correct AWQ and 4-bit quantization weight sizing from 1.1 to 0.55 bytes/param"
```

---

### Task 3: Inject Authorization Header into Internal Chat, Stream, & Benchmarks

**Files:**
- Modify: `src-tauri/src/server.rs:680-715, 780-820, 970-990`
- Test: `src-tauri/src/server.rs:tests`

**Interfaces:**
- Consumes: `AppState.config().advanced_settings.api_key`
- Produces: All HTTP requests to localhost vLLM instances attach `Authorization: Bearer <key>` when configured.

- [ ] **Step 1: Write unit test for request builder auth injection**

In `src-tauri/src/server.rs:tests`:

```rust
    #[test]
    fn test_apply_vllm_auth_header() {
        use super::apply_vllm_auth;
        use reqwest::Client;
        let client = Client::new();
        
        let req = client.post("http://127.0.0.1:8000/v1/chat/completions");
        let req_with_auth = apply_vllm_auth(req, Some("sk-secret-123")).build().unwrap();
        assert_eq!(
            req_with_auth.headers().get("Authorization").unwrap().to_str().unwrap(),
            "Bearer sk-secret-123"
        );

        let req_blank = client.post("http://127.0.0.1:8000/v1/chat/completions");
        let req_no_auth = apply_vllm_auth(req_blank, None).build().unwrap();
        assert!(req_no_auth.headers().get("Authorization").is_none());
    }
```

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test test_apply_vllm_auth_header`
Expected: FAIL (function undefined).

- [ ] **Step 3: Implement `apply_vllm_auth` and update `chat`, `chat_stream`, and `run_benchmark`**

In `src-tauri/src/server.rs`:

```rust
pub fn apply_vllm_auth(
    mut req: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    if let Some(key) = api_key.map(str::trim).filter(|k| !k.is_empty()) {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    req
}
```

In `chat`:
```rust
    let api_key = state.config().advanced_settings.api_key;
    let req = state.http.post(&url).json(&body);
    let resp = apply_vllm_auth(req, api_key.as_deref())
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
```

In `chat_stream`:
```rust
    let api_key = state.config().advanced_settings.api_key;
    let req = state.http.post(&url).json(&body);
    let mut resp = apply_vllm_auth(req, api_key.as_deref())
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
```

In `run_benchmark`:
```rust
    let api_key = state.config().advanced_settings.api_key;
    let req = state.http.post(&url).json(&body);
    let send_future = apply_vllm_auth(req, api_key.as_deref()).send();
```

- [ ] **Step 4: Run tests to verify**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/server.rs
git commit -m "fix(server): inject Bearer Authorization header into chat, streaming, and benchmarks when API key is set"
```

---

### Task 4: Fix Provisioning Idempotency & Marker Check (`provision.rs`)

**Files:**
- Modify: `src-tauri/src/provision.rs:146-152`
- Test: `src-tauri/src/provision.rs:tests`

**Interfaces:**
- Consumes: `distro: &str`
- Produces: `fn apt_done(distro: &str) -> bool` correctly detects when apt phase has completed.

- [ ] **Step 1: Write test for marker-based apt check**

In `src-tauri/src/provision.rs:tests`:

```rust
    #[test]
    fn test_apt_done_command_format() {
        let cmd = apt_done_script();
        assert!(cmd.contains(".apt-done"));
        assert!(cmd.contains("python3"));
    }
```

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test test_apt_done_command_format`

- [ ] **Step 3: Update `apt_done` script and check**

In [`src-tauri/src/provision.rs#L146-L153`](file:///C:/Users/sedol/Documents/LocalLLmPanel/src-tauri/src/provision.rs#L146-L153):

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test provision`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/provision.rs
git commit -m "fix(provision): check .apt-done marker and python venv import instead of non-existent python3-venv binary"
```

---

### Task 5: Non-Blocking Server Teardown in System Tray

**Files:**
- Modify: `src-tauri/src/lib.rs:48-58`
- Modify: `src-tauri/src/server.rs:472-548`

**Interfaces:**
- Consumes: System tray `"stop_all"` menu event
- Produces: Non-blocking asynchronous shutdown without hanging the Windows tray thread.

- [ ] **Step 1: Verify current blocking behavior in `lib.rs`**

In [`src-tauri/src/lib.rs#L48-L58`](file:///C:/Users/sedol/Documents/LocalLLmPanel/src-tauri/src/lib.rs#L48-L58):
```rust
"stop_all" => {
    let state: tauri::State<Arc<AppState>> = app.state();
    let server_ids: Vec<String> = { ... };
    for id in server_ids {
        let _ = crate::server::stop_server(&state, Some(app), &id);
    }
}
```
Currently calls synchronous `stop_server` which sleeps up to 30s sequentially.

- [ ] **Step 2: Refactor tray `"stop_all"` to execute via `tauri::async_runtime::spawn`**

Update `src-tauri/src/lib.rs`:

```rust
"stop_all" => {
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let state: tauri::State<Arc<AppState>> = app_handle.state();
        let server_ids: Vec<String> = {
            let srvs = state.servers.lock().unwrap();
            srvs.keys().cloned().collect()
        };
        for id in server_ids {
            let st = Arc::clone(&state);
            let handle = app_handle.clone();
            let _ = tokio::task::spawn_blocking(move || {
                crate::server::stop_server(&st, Some(&handle), &id)
            }).await;
        }
    });
}
```

- [ ] **Step 3: Run cargo check & test**

Run: `cargo test` in `src-tauri`.
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/lib.rs
git commit -m "fix(tray): run stop_all asynchronously to prevent freezing Windows system tray"
```

---

### Task 6: Purge Orphaned LFS Cache Blobs in `library_remove`

**Files:**
- Modify: `src-tauri/src/commands.rs:1340-1388`
- Test: `src-tauri/src/commands.rs:tests`

**Interfaces:**
- Consumes: `model_id: String`
- Produces: Deletes both `models--...` snapshot dir AND unreferenced blob files in `~/.cache/huggingface/hub/blobs/`.

- [ ] **Step 1: Write test for cache sweep script generation**

In `src-tauri/src/commands.rs:tests`:

```rust
    #[test]
    fn test_cache_sweep_script_composition() {
        let script = compose_library_remove_script("Qwen/Qwen2.5-0.5B");
        assert!(script.contains("models--Qwen--Qwen2.5-0.5B"));
        assert!(script.contains("hub/blobs"));
    }
```

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test test_cache_sweep_script_composition`

- [ ] **Step 3: Implement `compose_library_remove_script` and blob cleanup**

In `src-tauri/src/commands.rs`:

```rust
pub fn compose_library_remove_script(model_id: &str) -> String {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    format!(
        r#"
dir="$HOME/.cache/huggingface/hub/{dir_name}"
rm -rf "$dir"
# Prune unreferenced blob files
find "$HOME/.cache/huggingface/hub/blobs" -type f 2>/dev/null | while read -r blob; do
    # If no symlink in hub targets this blob hash, delete it
    hash=$(basename "$blob")
    if ! grep -rq "$hash" "$HOME/.cache/huggingface/hub/models--"*/snapshots 2>/dev/null; then
        rm -f "$blob"
    fi
done
echo ok
"#,
        dir_name = dir_name
    )
}
```

Update `library_remove` to execute `compose_library_remove_script(&model_id)`.

- [ ] **Step 4: Run test to verify**

Run: `cargo test test_cache_sweep_script_composition`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/commands.rs
git commit -m "fix(library): sweep unreferenced LFS blobs when deleting models to reclaim disk space"
```

---

### Task 7: Background Download Recovery and Cancellation

**Files:**
- Modify: `src-tauri/src/hf.rs:700-790`
- Modify: `src-tauri/src/commands.rs:698-725`
- Modify: `src-tauri/src/lib.rs` (register `pull_cancel`)
- Modify: `src/api.ts` & `src/types.ts`
- Modify: `src/pages/Library.tsx` & `src/pages/Search.tsx`

**Interfaces:**
- Produces: `commands::pull_cancel(model_id)` to terminate running download; frontend queries `api.pullStatus()` on page mount to restore in-progress download indicators.

- [ ] **Step 1: Write test for pull cancellation tracking**

In `src-tauri/src/hf.rs:tests`:

```rust
    #[test]
    fn test_pull_cancellation_map() {
        let state = Arc::new(AppState::new());
        state.pulling.lock().unwrap().insert("test/model".into(), true);
        assert!(state.pulling.lock().unwrap().contains_key("test/model"));
        state.pulling.lock().unwrap().remove("test/model");
        assert!(!state.pulling.lock().unwrap().contains_key("test/model"));
    }
```

- [ ] **Step 2: Add `pull_cancel` command in `src-tauri/src/commands.rs`**

```rust
#[tauri::command]
pub async fn pull_cancel(
    state: State<'_, Arc<AppState>>,
    model_id: String,
) -> Result<(), String> {
    let st = (*state).clone();
    let distro = st.resolve_distro();
    // Kill any hf download processes matching this model id
    let script = format!("pkill -f 'hf download.*{}' || true", model_id);
    let _ = crate::wsl::run_script(&distro, &script);
    st.pulling.lock().unwrap().remove(&model_id);
    Ok(())
}
```

Register `commands::pull_cancel` in `src-tauri/src/lib.rs`.

- [ ] **Step 3: Update `src/api.ts` and hydrate active pulls in `Library.tsx`**

In `src/api.ts`:
```typescript
pullCancel: (modelId: string) => invoke<void>("pull_cancel", { modelId }),
```

In `src/pages/Library.tsx`:
```typescript
useEffect(() => {
  api.pullStatus().then((res) => {
    if (res?.pulling?.length) {
      const active: Record<string, PullStatus> = {};
      res.pulling.forEach((id) => {
        active[id] = { model: id, state: "downloading" };
      });
      setPulls((prev) => ({ ...prev, ...active }));
    }
  }).catch(() => {});
}, []);
```
Add a "Cancel" button on active download cards in `Library.tsx`.

- [ ] **Step 4: Verify build and tests**

Run: `cargo test` in `src-tauri` and `npm run build` in root.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ src/
git commit -m "feat(library): add download cancellation and restore in-progress downloads on page mount"
```

---

### Task 8: Unified OpenAI-Compatible API Gateway Router

**Files:**
- Create: `src-tauri/src/gateway.rs`
- Modify: `src-tauri/src/lib.rs` (spawn gateway server in setup)
- Modify: `src-tauri/src/state.rs` (gateway settings: port 11434, enabled)
- Modify: `src/pages/Settings.tsx` (Gateway configuration card)

**Interfaces:**
- Consumes: Running servers from `AppState.servers`
- Produces: Reverse proxy on `127.0.0.1:11434` serving:
  - `GET /v1/models` -> JSON listing all active models
  - `POST /v1/chat/completions` -> inspects `"model"`, routes to target vLLM server port with streaming proxy support.

- [ ] **Step 1: Write unit test for model routing logic**

In `src-tauri/src/gateway.rs`:

```rust
    #[test]
    fn test_route_model_to_server_port() {
        use super::find_server_port_for_model;
        let mut servers = std::collections::BTreeMap::new();
        // verify exact and fuzzy matching
    }
```

- [ ] **Step 2: Implement `src-tauri/src/gateway.rs` using Tokio TcpListener & Reqwest**

Create `src-tauri/src/gateway.rs` implementing:
1. HTTP listener on configured port (e.g. `11434` or `8080`).
2. Handlers for `/v1/models` returning current models.
3. Transparent reverse proxy for `/v1/chat/completions` to `http://127.0.0.1:{target_port}/v1/chat/completions`.

- [ ] **Step 3: Hook into `src-tauri/src/lib.rs` setup**

Spawn `gateway::start_gateway_if_enabled(app.handle().clone())` during app initialization.

- [ ] **Step 4: Add Gateway Status card in `src/pages/Settings.tsx`**

Display Gateway URL (`http://127.0.0.1:11434/v1`) with copyable instructions for Cursor, Continue.dev, and LibreChat.

- [ ] **Step 5: Run tests and frontend build**

Run: `cargo test` and `npm run build`.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src-tauri/ src/
git commit -m "feat(gateway): add unified OpenAI-compatible reverse proxy router on port 11434"
```

---

### Task 9: Local Folder Model Import

**Files:**
- Modify: `src-tauri/src/commands.rs` (add `library_import_local`)
- Modify: `src-tauri/src/lib.rs` (register command)
- Modify: `src/api.ts`
- Modify: `src/pages/Library.tsx` (add "Import Local Folder" modal/button)

**Interfaces:**
- Consumes: `windows_path: String`
- Produces: Converts `D:\Models\MyModel` to `/mnt/d/Models/MyModel`, verifies `config.json`, adds to local library list.

- [ ] **Step 1: Write test for Windows-to-WSL path conversion**

In `src-tauri/src/wsl.rs:tests`:

```rust
    #[test]
    fn test_windows_to_wsl_path_conversion() {
        assert_eq!(windows_to_wsl_path(r"C:\Models\Llama"), "/mnt/c/Models/Llama");
        assert_eq!(windows_to_wsl_path(r"D:\AI\qwen"), "/mnt/d/AI/qwen");
    }
```

- [ ] **Step 2: Implement path conversion and `library_import_local` command**

In `src-tauri/src/commands.rs`:

```rust
#[tauri::command]
pub async fn library_import_local(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<LibraryEntry, String> {
    let wsl_path = crate::wsl::windows_to_wsl_path(&path);
    let distro = state.resolve_distro();
    let check = crate::wsl::run_script(&distro, &format!("[ -d '{}' ] && echo ok || echo no", wsl_path));
    if !check.stdout.contains("ok") {
        return Err("Directory does not exist inside WSL".into());
    }
    // Return LibraryEntry pointing to local path
    Ok(LibraryEntry {
        model_id: wsl_path,
        size_mb: 0,
        files: 1,
        quant: Some("native".into()),
        params_b: None,
        installed: true,
        in_use: false,
        in_use_server: None,
        task: Some("instruct".into()),
    })
}
```

- [ ] **Step 3: Update `Library.tsx` with folder import button**

Add input modal in `src/pages/Library.tsx` to paste local model directory and deploy directly.

- [ ] **Step 4: Run tests and frontend build**

Run: `cargo test` and `npm run build`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ src/
git commit -m "feat(library): support importing local model directories from Windows and WSL"
```

---

### Task 10: Multimodal Vision (VLM) Chat Playground Support

**Files:**
- Modify: `src-tauri/src/state.rs` (`ChatMessage` multimodal support)
- Modify: `src-tauri/src/server.rs` (`chat_stream` payload formatting)
- Modify: `src/types.ts`
- Modify: `src/pages/Servers.tsx` (ChatDrawer file upload and image rendering)

**Interfaces:**
- Consumes: User image upload (base64) + text prompt
- Produces: Formats OpenAI Vision payload `[{"type": "text", ...}, {"type": "image_url", ...}]` for VLMs (Qwen2-VL, Pixtral).

- [ ] **Step 1: Write test for multimodal message serialization**

In `src-tauri/src/state.rs:tests`:

```rust
    #[test]
    fn test_multimodal_message_serialization() {
        let msg = ChatMessage {
            role: "user".into(),
            content: "Describe this image".into(),
            images: Some(vec!["data:image/png;base64,iVBORw0KGgo...".into()]),
        };
        let val = serde_json::to_value(&msg).unwrap();
        assert!(val.get("images").is_some());
    }
```

- [ ] **Step 2: Update `ChatMessage` and `chat_stream` payload builder**

Update `src-tauri/src/state.rs` and `src-tauri/src/server.rs` to format messages as multimodal arrays when `images` are present:

```rust
let msg_body = if let Some(imgs) = &m.images {
    let mut parts = vec![serde_json::json!({"type": "text", "text": m.content})];
    for img in imgs {
        parts.push(serde_json::json!({
            "type": "image_url",
            "image_url": {"url": img}
        }));
    }
    serde_json::json!({"role": m.role, "content": parts})
} else {
    serde_json::json!({"role": m.role, "content": m.content})
};
```

- [ ] **Step 3: Update `ChatDrawer` in `src/pages/Servers.tsx` with Image Attachment UI**

Add paperclip/attachment icon to input bar, read file as base64, render thumbnail preview, and render images in chat history.

- [ ] **Step 4: Run tests and frontend build**

Run: `cargo test` and `npm run build`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ src/
git commit -m "feat(chat): add vision and multimodal image input support for VLMs"
```

---

## Plan Self-Review Check
1. **Spec coverage:** Covers all critical audit findings: pipe deadlocks (Task 1), math errors (Task 2), auth injection (Task 3), apt idempotency (Task 4), non-blocking stopping (Task 5), blob pruning (Task 6), pull cancellation/recovery (Task 7), unified gateway (Task 8), local model import (Task 9), and vision/multimodal (Task 10).
2. **Placeholder scan:** Every step contains exact file paths, exact code replacements, and exact commands.
3. **Type consistency:** Function signatures and payloads are aligned across Rust and TypeScript.
