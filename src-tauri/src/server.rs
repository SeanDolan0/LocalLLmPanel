//! Multi-instance vLLM server lifecycle: launch, monitor, stop, metrics, chat.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::estimate;
pub use crate::state::MetricsSnapshot;
use crate::state::{AppState, LiveServer, ServerDef, ServerStatus, VecDequeLog};
use crate::wsl;
use tauri::Emitter;

/// Validate environment variable name: must match ^[A-Za-z_][A-Za-z0-9_]*$
pub fn validate_env_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Environment variable name cannot be empty".into());
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err("Environment variable name must start with a letter or underscore".into());
    }
    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' {
            return Err("Environment variable name can only contain letters, numbers, and underscores".into());
        }
    }
    Ok(())
}

/// Safely single-quote a value for POSIX sh.
/// Wraps in single quotes and replaces each embedded ' with '\''.
/// This is the ONLY quoting function that should be used for shell values.
pub fn shell_quote(value: &str) -> String {
    // Simple and correct: wrap in single quotes, escape embedded quotes
    let mut result = String::with_capacity(value.len() + 2);
    result.push('\'');
    result.push_str(&value.replace('\'', "'\\''"));
    result.push('\'');
    result
}

/// Sanitize a user-supplied environment variable value:
/// - Trim whitespace
/// - Strip a matching pair of surrounding single or double quotes
/// This allows users to enter '0' or "0" and get 0.
pub fn sanitize_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let first = trimmed.chars().next().unwrap();
        let last = trimmed.chars().last().unwrap();
        if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
            return trimmed[1..trimmed.len() - 1].to_string();
        }
    }
    trimmed.to_string()
}

const DEFAULT_PORT_START: u16 = 8000;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const HEALTH_POLL: Duration = Duration::from_secs(2);
const METRICS_POLL: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Port allocation
// ---------------------------------------------------------------------------

/// Allocate a free localhost port, starting at 8000 (or after `min`),
/// skipping any ports already bound by existing server definitions.
pub fn alloc_port(existing: &[u16]) -> Result<u16> {
    let mut port = DEFAULT_PORT_START;
    loop {
        if existing.contains(&port) {
            port += 1;
            continue;
        }
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
        port = port.wrapping_add(1);
        if port == DEFAULT_PORT_START {
            bail!("no free port found");
        }
    }
}

// ---------------------------------------------------------------------------
// Launch script
// ---------------------------------------------------------------------------

/// Determine if a model is an already quantized checkpoint.
/// Such models contain their own `quantization_config` in `config.json` which vLLM
/// automatically resolves. Passing an explicit `--quantization` flag to them causes
/// mismatch errors (e.g. compressed-tensors vs awq).
fn is_prequantized_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("-awq")
        || lower.contains("_awq")
        || lower.contains("/awq")
        || lower.contains("-gptq")
        || lower.contains("_gptq")
        || lower.contains("/gptq")
        || lower.contains("-int4")
        || lower.contains("_int4")
        || lower.contains("-int8")
        || lower.contains("_int8")
        || lower.contains("-fp8")
        || lower.contains("_fp8")
        || lower.contains("-bnb")
        || lower.contains("_bnb")
        || lower.contains("compressed-tensors")
    {
        return true;
    }
    let p = std::path::Path::new(model_id);
    if p.is_dir() {
        if let Ok(content) = std::fs::read_to_string(p.join("config.json")) {
            if content.contains("\"quantization_config\"") || content.contains("\"quant_method\"") {
                return true;
            }
        }
    }
    false
}

/// Build the launch script that will be written to a file in WSL and executed.
/// This avoids nested quoting issues with `wsl.exe -d <distro> --exec bash -lc "<script>"`.
/// The script activates the venv, records the PID, exports env vars, then execs vLLM.
fn launch_script(
    venv_dir: &str,
    def: &ServerDef,
    hf_token: &str,
    adv: &crate::state::AdvancedSettings,
    global_default_env: &BTreeMap<String, String>,
) -> String {
    // Merge global default_env with per-server env (per-server wins)
    let mut merged_env = global_default_env.clone();
    merged_env.extend(def.env.clone());

    let mut parts: Vec<String> = vec![
        format!("mkdir -p {}/../run", shell_quote(venv_dir)),
        format!("cd {}/..", shell_quote(venv_dir)),
        format!(". {}/bin/activate", shell_quote(venv_dir)),
        format!("echo $$ > {}/../run/{}.pid", shell_quote(venv_dir), def.id),
    ];

    // Export environment variables (merged global defaults + per-server overrides)
    // Log applied env vars at startup for debugging quote issues
    parts.insert(0, format!("echo '[LocalLLmPanel] Starting server {} on port {}'", shell_quote(&def.id), def.port));
    for (name, value) in &merged_env {
        if let Err(e) = validate_env_name(name) {
            eprintln!("[launch_script] Invalid env name '{}': {}, skipping", name, e);
            continue;
        }
        let quoted = shell_quote(value);
        parts.insert(0, format!("export {}={}", name, quoted));
        // Debug log showing the exact value that will be seen by the process
        parts.insert(0, format!("printf '[LocalLLmPanel] env %s=<%s>\\n' {} {}", shell_quote(name), quoted));
    }

    let mut args: Vec<String> = Vec::new();
    args.push("--model".into());
    args.push(shell_quote(&def.model_id));
    args.push("--host".into());
    let host = if adv.host.trim().is_empty() {
        "127.0.0.1"
    } else {
        adv.host.trim()
    };
    args.push(host.into());
    args.push("--port".into());
    args.push(def.port.to_string());
    args.push("--gpu-memory-utilization".into());
    args.push(format!("{:.2}", def.gpu_mem_util));
    if def.task == "embed" {
        args.push("--runner".into());
        args.push("pooling".into());
    }
    // For pre-quantized models (AWQ, GPTQ, compressed-tensors, INT4, bitsandbytes,
    // pre-quantized FP8 checkpoints, etc.), vLLM automatically reads `quant_method`
    // from the model's `config.json`. Passing an explicit `--quantization` flag (such as
    // `--quantization awq` for a compressed-tensors model) causes vLLM to reject the model
    // with a ValidationError.
    // We only pass `--quantization fp8` if an unquantized model is explicitly requested
    // to be dynamically quantized to FP8 at runtime.
    if def.quant.eq_ignore_ascii_case("fp8") && !is_prequantized_model(&def.model_id) {
        args.push("--quantization".into());
        args.push("fp8".into());
    }
    let max_len = match def.max_model_len {
        Some(len) if len > 0 => len,
        _ => 4096,
    };
    args.push("--max-model-len".into());
    args.push(max_len.to_string());
    if def.enforce_eager {
        args.push("--enforce-eager".into());
    }
    if let Some(served) = &def.served_model_name {
        args.push("--served-model-name".into());
        args.push(shell_quote(served));
    }
    if let Some(swap) = def.swap_space_gb {
        if swap > 0 {
            args.push("--kv-offloading-size".into());
            args.push(swap.to_string());
        }
    }
    if let Some(offload) = def.cpu_offload_gb {
        if offload > 0 {
            args.push("--cpu-offload-gb".into());
            args.push(offload.to_string());
        }
    }

    // Advanced vLLM engine flags
    if let Some(key) = &adv.api_key {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            args.push("--api-key".into());
            args.push(shell_quote(trimmed));
        }
    }
    if adv.kv_cache_dtype.to_ascii_lowercase() != "auto" && !adv.kv_cache_dtype.trim().is_empty() {
        args.push("--kv-cache-dtype".into());
        args.push(adv.kv_cache_dtype.trim().to_string());
    }
    if adv.enable_prefix_caching {
        args.push("--enable-prefix-caching".into());
    }
    if adv.enable_chunked_prefill {
        args.push("--enable-chunked-prefill".into());
    }
    if let Some(seqs) = adv.max_num_seqs {
        if seqs > 0 {
            args.push("--max-num-seqs".into());
            args.push(seqs.to_string());
        }
    }
    if adv.disable_custom_all_reduce {
        args.push("--disable-custom-all-reduce".into());
    }
    if let Some(extra) = &adv.extra_vllm_args {
        for token in extra.split_whitespace() {
            if !token.is_empty() {
                args.push(token.to_string());
            }
        }
    }

    // Modern vLLM replaces legacy --swap-space with --kv-offloading-size.
    for arg in &mut args {
        if arg == "--swap-space" {
            *arg = "--kv-offloading-size".to_string();
        }
    }

    // Environment variables (WSL2 vLLM requirements: bypass UVA pin-memory bug & use spawn workers)
    parts.insert(0, format!("export VLLM_WORKER_MULTIPROC_METHOD={}", shell_quote("spawn")));
    parts.insert(0, format!("export VLLM_WSL2_ENABLE_PIN_MEMORY={}", shell_quote("1")));
    if !adv.log_level.trim().is_empty() {
        parts.insert(
            0,
            format!("export VLLM_LOGGING_LEVEL={}", shell_quote(&adv.log_level.trim())),
        );
    }
    if adv.hf_offline {
        parts.insert(0, format!("export HF_HUB_OFFLINE={}", shell_quote("1")));
    }
    if let Some(home) = &adv.hf_home {
        let home_trim = home.trim();
        if !home_trim.is_empty() {
            parts.insert(
                0,
                format!("mkdir -p {} && export HF_HOME={}", shell_quote(home_trim), shell_quote(home_trim)),
            );
        }
    }
    if let Some(custom_envs) = &adv.custom_env_vars {
        for line in custom_envs.lines() {
            let line_trim = line.trim();
            if !line_trim.is_empty() && !line_trim.starts_with('#') && line_trim.contains('=') {
                if let Some((key, value)) = line_trim.split_once('=') {
                    let key = key.trim();
                    let value = value.trim();
                    if validate_env_name(key).is_ok() {
                        let sanitized = sanitize_env_value(value);
                        let quoted = shell_quote(&sanitized);
                        parts.insert(0, format!("export {}={}", key, quoted));
                        parts.insert(0, format!("printf '[LocalLLmPanel] env %s=<%s>\\n' {} {}", shell_quote(key), quoted));
                    } else {
                        eprintln!("[launch_script] Invalid custom env name '{}', skipping", key);
                    }
                }
            }
        }
    }
    let token_ok = !hf_token.is_empty()
        && hf_token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || (c.is_ascii_punctuation() && c != '\''));
    if token_ok {
        parts.insert(0, format!("export HF_TOKEN={}", shell_quote(hf_token)));
        parts.insert(0, format!("printf '[LocalLLmPanel] env %s=<%s>\\n' {} {}", shell_quote("HF_TOKEN"), shell_quote(hf_token)));
    }
    parts.push(format!(
        "exec python -m vllm.entrypoints.openai.api_server {}",
        args.join(" ")
    ));
    parts.join(" && ")
}

pub fn build_start_command(def: &ServerDef, hf_token: &str) -> String {
    let adv = crate::state::AdvancedSettings::default();
    let global_default_env = BTreeMap::new();
    launch_script("~/llm-lp/.venv", def, hf_token, &adv, &global_default_env)
}

pub fn build_start_command_with_advanced(
    def: &ServerDef,
    hf_token: &str,
    adv: &crate::state::AdvancedSettings,
) -> String {
    let global_default_env = BTreeMap::new();
    launch_script("~/llm-lp/.venv", def, hf_token, adv, &global_default_env)
}

/// Resolve a user-supplied model identifier or path to an existing GGUF file.
/// Checks:
/// 1. Direct file path on disk
/// 2. Files inside `gguf_dir` directly or in subdirectories
/// 3. Matching downloaded repositories inside the WSL Hugging Face cache
pub fn resolve_gguf_model_path(
    model_raw: &str,
    gguf_dir: &str,
    distro: &str,
) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(model_raw);
    if p.is_file() {
        return Some(p.to_path_buf());
    }

    let gguf_root = std::path::Path::new(gguf_dir);
    if gguf_root.is_dir() {
        let direct = gguf_root.join(model_raw);
        if direct.is_file() {
            return Some(direct);
        }

        let leaf_name = model_raw.rsplit('/').next().unwrap_or(model_raw);
        let repo_dir = gguf_root.join(leaf_name);
        if repo_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&repo_dir) {
                let mut ggufs: Vec<_> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .and_then(|x| x.to_str())
                            .map(|x| x.eq_ignore_ascii_case("gguf"))
                            .unwrap_or(false)
                    })
                    .collect();
                ggufs.sort();
                if let Some(first) = ggufs.into_iter().next() {
                    return Some(first);
                }
            }
        }

        if let Ok(entries) = std::fs::read_dir(gguf_root) {
            for entry in entries.flatten().filter(|e| e.path().is_dir()) {
                let dir_name = entry.file_name().to_string_lossy().to_lowercase();
                let leaf_lower = leaf_name.to_lowercase();
                if dir_name.contains(&leaf_lower) || leaf_lower.contains(&dir_name) {
                    if let Ok(sub_entries) = std::fs::read_dir(entry.path()) {
                        let mut ggufs: Vec<_> = sub_entries
                            .flatten()
                            .map(|e| e.path())
                            .filter(|p| {
                                p.extension()
                                    .and_then(|x| x.to_str())
                                    .map(|x| x.eq_ignore_ascii_case("gguf"))
                                    .unwrap_or(false)
                            })
                            .collect();
                        ggufs.sort();
                        if let Some(first) = ggufs.into_iter().next() {
                            return Some(first);
                        }
                    }
                }
            }
        }
    }

    // Check WSL Hugging Face cache (UNC path)
    if !distro.is_empty() {
        let hf_folder_name = format!("models--{}", model_raw.replace('/', "--"));
        let wsl_unc_base = format!(r"\\wsl.localhost\{distro}");
        let wsl_home = std::path::PathBuf::from(&wsl_unc_base).join("home");
        if let Ok(users) = std::fs::read_dir(&wsl_home) {
            for user in users.flatten().filter(|e| e.path().is_dir()) {
                let model_hub = user
                    .path()
                    .join(".cache")
                    .join("huggingface")
                    .join("hub")
                    .join(&hf_folder_name);
                if model_hub.is_dir() {
                    let blobs_dir = model_hub.join("blobs");
                    let snapshots_dir = model_hub.join("snapshots");
                    if snapshots_dir.is_dir() {
                        if let Ok(snaps) = std::fs::read_dir(&snapshots_dir) {
                            for snap in snaps.flatten().filter(|e| e.path().is_dir()) {
                                if let Ok(snap_files) = std::fs::read_dir(snap.path()) {
                                    for sf in snap_files.flatten() {
                                        let sf_name =
                                            sf.file_name().to_string_lossy().to_lowercase();
                                        if sf_name.ends_with(".gguf") {
                                            if let Ok(target) = std::fs::read_link(sf.path()) {
                                                let full_target = snap.path().join(target);
                                                if full_target.is_file() {
                                                    return Some(full_target);
                                                }
                                            }
                                            if blobs_dir.is_dir() {
                                                if let Ok(blobs) = std::fs::read_dir(&blobs_dir) {
                                                    let mut sorted_blobs: Vec<_> =
                                                        blobs.flatten().map(|b| b.path()).collect();
                                                    sorted_blobs.sort_by_key(|p| {
                                                        p.metadata().map(|m| m.len()).unwrap_or(0)
                                                    });
                                                    if let Some(biggest) =
                                                        sorted_blobs.into_iter().next_back()
                                                    {
                                                        return Some(biggest);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Build native llama-server arguments without shell quoting.
///
/// Capability filtering is applied by the installer/runtime layer; this
/// function intentionally maps the persisted configuration deterministically.
pub fn build_llamacpp_args(def: &ServerDef) -> Vec<String> {
    build_llamacpp_args_with_help(def, "")
}

/// Build native llama-server arguments, enabling automatic fitting only when
/// the installed binary advertises the corresponding capability.
pub fn build_llamacpp_args_with_help(def: &ServerDef, help: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut structured_flags = Vec::new();
    let model = def.model_path.as_deref().unwrap_or(&def.model_id);
    args.extend(["-m".into(), model.into()]);
    if let Some(mmproj) = def.mmproj_path.as_deref().filter(|p| !p.is_empty()) {
        args.extend(["--mmproj".into(), mmproj.into()]);
        structured_flags.push("--mmproj");
    }
    args.extend([
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        def.port.to_string(),
    ]);
    structured_flags.extend(["-m", "--model", "--host", "--port"]);
    if let Some(ctx) = def.ctx_size.filter(|v| *v > 0) {
        args.extend(["-c".into(), ctx.to_string()]);
        structured_flags.extend(["-c", "--ctx-size"]);
    }
    if let Some(n_gpu_layers) = def.n_gpu_layers {
        args.extend(["-ngl".into(), n_gpu_layers.to_string()]);
        structured_flags.push("-ngl");
    }
    if let Some(n) = def.n_cpu_moe {
        args.extend(["--n-cpu-moe".into(), n.to_string()]);
        structured_flags.push("--n-cpu-moe");
        structured_flags.push("-ncmoe");
    }
    if !help.is_empty() && help.contains("--fit") {
        args.extend(["--fit".into(), if def.fit { "on" } else { "off" }.into()]);
        structured_flags.push("--fit");
        structured_flags.push("-fit");
        if let Some(target) = def.fit_target.filter(|v| *v > 0) {
            args.extend(["--fit-target".into(), target.to_string()]);
            structured_flags.push("--fit-target");
            structured_flags.push("-fitt");
        }
    }
    if let Some(device) = def.device.as_deref().filter(|v| !v.trim().is_empty()) {
        args.extend(["--device".into(), device.trim().into()]);
        structured_flags.push("--device");
        structured_flags.push("-dev");
    }
    if let Some(api_key) = def.api_key.as_deref().filter(|v| !v.trim().is_empty()) {
        args.extend(["--api-key".into(), api_key.to_string()]);
        structured_flags.push("--api-key");
    }
    if def.flash_attn {
        args.extend(["-fa".into(), "on".into()]);
        structured_flags.extend(["-fa", "--flash-attn"]);
    }
    if !def.cache_type_k.is_empty() {
        args.extend(["--cache-type-k".into(), def.cache_type_k.clone()]);
        structured_flags.extend(["--cache-type-k", "-ctk"]);
    }
    if !def.cache_type_v.is_empty() {
        args.extend(["--cache-type-v".into(), def.cache_type_v.clone()]);
        structured_flags.extend(["--cache-type-v", "-ctv"]);
    }
    if let Some(threads) = def.threads.filter(|v| *v > 0) {
        args.extend(["-t".into(), threads.to_string()]);
        structured_flags.extend(["-t", "--threads"]);
    }
    if let Some(batch) = def.batch_size.filter(|v| *v > 0) {
        args.extend(["-b".into(), batch.to_string()]);
        structured_flags.extend(["-b", "--batch-size"]);
    }
    if let Some(ubatch) = def.ubatch_size.filter(|v| *v > 0) {
        args.extend(["-ub".into(), ubatch.to_string()]);
        structured_flags.extend(["-ub", "--ubatch-size"]);
    }
    if def.parallel > 0 {
        args.extend(["-np".into(), def.parallel.to_string()]);
        structured_flags.extend(["-np", "--parallel"]);
    }
    if def.jinja {
        args.push("--jinja".into());
        structured_flags.push("--jinja");
    }
    if def.no_kv_offload {
        args.push("--no-kv-offload".into());
        structured_flags.extend(["--no-kv-offload", "-nkvo"]);
    }
    if def.metrics {
        args.push("--metrics".into());
        structured_flags.push("--metrics");
    }
    if let Some(verbosity) = def.log_verbosity.filter(|v| *v <= 5) {
        args.extend(["--log-verbosity".into(), verbosity.to_string()]);
        structured_flags.push("--log-verbosity");
        structured_flags.push("-lv");
    }

    // Workaround for upstream metadata bug in Qwen AgentWorld / qwen35moe models.
    // Keep it structured so it can be merged with user override-kv entries.
    let model_lower = model.to_lowercase();
    let mut overrides = Vec::new();
    if model_lower.contains("agentworld") || model_lower.contains("qwen35moe") {
        overrides.extend([
            "qwen35moe.block_count=int:40".to_string(),
            "qwen35moe.nextn_predict_layers=int:0".to_string(),
        ]);
    }
    let mut extra = def.extra_args.clone();
    let mut extra_overrides = Vec::new();
    let mut i = 0;
    while i < extra.len() {
        if extra[i] == "--override-kv" {
            if let Some(value) = extra.get(i + 1) {
                extra_overrides.extend(
                    value
                        .split(',')
                        .filter(|v| !v.trim().is_empty())
                        .map(str::to_string),
                );
                i += 2;
                continue;
            }
        } else if let Some(value) = extra[i].strip_prefix("--override-kv=") {
            extra_overrides.extend(
                value
                    .split(',')
                    .filter(|v| !v.trim().is_empty())
                    .map(str::to_string),
            );
            i += 1;
            continue;
        }
        i += 1;
    }
    extra.retain(|arg| arg != "--override-kv" && !arg.starts_with("--override-kv="));
    // Structured settings win over arbitrary extra args for the same option.
    let mut filtered_extra = Vec::with_capacity(extra.len());
    let mut i = 0;
    while i < extra.len() {
        let arg = &extra[i];
        if structured_flags.contains(&arg.as_str()) {
            i += 1;
            if matches!(
                arg.as_str(),
                "-m" | "--model"
                    | "--host"
                    | "--port"
                    | "-c"
                    | "--ctx-size"
                    | "-ngl"
                    | "--n-gpu-layers"
                    | "--gpu-layers"
                    | "--n-cpu-moe"
                    | "-ncmoe"
                    | "--fit"
                    | "-fit"
                    | "--fit-target"
                    | "-fitt"
                    | "--device"
                    | "-dev"
                    | "--api-key"
                    | "--log-verbosity"
                    | "-lv"
                    | "-fa"
                    | "--flash-attn"
                    | "--cache-type-k"
                    | "-ctk"
                    | "--cache-type-v"
                    | "-ctv"
                    | "-t"
                    | "--threads"
                    | "-b"
                    | "--batch-size"
                    | "-ub"
                    | "--ubatch-size"
                    | "-np"
                    | "--parallel"
            ) {
                i += 1;
            }
            continue;
        }
        filtered_extra.push(arg.clone());
        i += 1;
    }
    overrides.extend(extra_overrides);
    if !overrides.is_empty() {
        let mut merged: Vec<(String, String)> = Vec::new();
        for item in overrides {
            if let Some((key, value)) = item.split_once('=') {
                merged.retain(|(old, _)| old != key.trim());
                merged.push((key.trim().to_string(), value.trim().to_string()));
            }
        }
        if !merged.is_empty() {
            args.extend([
                "--override-kv".into(),
                merged
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ]);
        }
    }
    args.extend(filtered_extra);
    args
}

/// Remove options that are not present in the installed llama-server build.
/// llama.cpp changes option spellings between releases, so persisted configs
/// must not make an older binary fail before it can print a useful error.
pub fn filter_llamacpp_args(args: Vec<String>, help: &str) -> Vec<String> {
    let supports = |flag: &str| help.contains(flag);
    let mut filtered = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "-fa" && !supports("-fa") && supports("--flash-attn") {
            filtered.push("--flash-attn".into());
            if let Some(value) = args.get(i + 1) {
                filtered.push(value.clone());
                i += 1;
            }
            i += 1;
            continue;
        }
        if arg == "--device" && !supports("--device") && supports("--dev") {
            filtered.push("--dev".into());
            if let Some(value) = args.get(i + 1) {
                filtered.push(value.clone());
                i += 1;
            }
            i += 1;
            continue;
        }
        let supported = match arg.as_str() {
            "--n-cpu-moe" => supports("--n-cpu-moe"),
            "--fit" => supports("--fit"),
            "--fit-target" => supports("--fit-target"),
            "--device" => supports("--device") || supports("--dev"),
            "--api-key" => supports("--api-key"),
            "--log-verbosity" => supports("--log-verbosity"),
            "--mmproj" => supports("--mmproj"),
            "-fa" => supports("-fa") || supports("--flash-attn"),
            "--cache-type-k" | "--cache-type-v" => supports(arg),
            "--no-kv-offload" | "--metrics" | "--jinja" => supports(arg),
            _ => true,
        };
        if supported {
            filtered.push(arg.clone());
            if matches!(
                arg.as_str(),
                "--n-cpu-moe"
                    | "--fit"
                    | "--fit-target"
                    | "--device"
                    | "--log-verbosity"
                    | "--api-key"
                    | "--mmproj"
                    | "-fa"
                    | "--cache-type-k"
                    | "--cache-type-v"
                    | "-m"
                    | "--host"
                    | "--port"
                    | "-c"
                    | "-ngl"
                    | "-t"
                    | "-b"
                    | "-ub"
                    | "-np"
                    | "--override-kv"
            ) {
                if let Some(value) = args.get(i + 1) {
                    filtered.push(value.clone());
                    i += 1;
                }
            }
        } else if matches!(
            arg.as_str(),
            "--n-cpu-moe"
                | "--fit"
                | "--fit-target"
                | "--device"
                | "--api-key"
                | "--log-verbosity"
                | "--mmproj"
                | "-fa"
                | "--cache-type-k"
                | "--cache-type-v"
                | "--no-kv-offload"
                | "--metrics"
                | "--jinja"
        ) {
            if matches!(
                arg.as_str(),
                "--n-cpu-moe"
                    | "--fit"
                    | "--fit-target"
                    | "--device"
                    | "--api-key"
                    | "--log-verbosity"
                    | "--mmproj"
                    | "-fa"
                    | "--cache-type-k"
                    | "--cache-type-v"
            ) {
                i += 1;
            }
        }
        i += 1;
    }
    filtered
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub struct ServerStatusEvent {
    pub id: String,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Clone, Serialize)]
struct ServerLogEvent {
    id: String,
    line: String,
}

fn emit_status(
    app: Option<&tauri::AppHandle>,
    id: &str,
    status: ServerStatus,
    error: Option<String>,
) {
    let payload = ServerStatusEvent {
        id: id.to_string(),
        status: status.label().to_string(),
        error,
    };
    if let Some(app) = app {
        let _ = app.emit("server-status", payload);
    }
}

// ---------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------

/// Start one server. Health is polled asynchronously; a monitor task follows
/// lifecycle and emits events. `state` is an Arc so the monitor task can hold
/// it beyond the command call.
pub fn start_server(state: &Arc<AppState>, app: Option<&tauri::AppHandle>, id: &str) -> Result<()> {
    let cfg = state.config();
    let distro = state.resolve_distro();
    let mut def = cfg
        .find_server(id)
        .cloned()
        .ok_or_else(|| anyhow!("no server with id {id}"))?;
    {
        let servers = state.servers.lock().unwrap();
        if let Some(ls) = servers.get(id) {
            if ls.status == ServerStatus::Running || ls.status == ServerStatus::Starting {
                bail!("server {} already {}", def.name, ls.status.label());
            }
        }
    }

    // Dynamically clamp GPU memory utilization based on currently available VRAM
    // to prevent vLLM startup ValueError when Windows/desktop processes occupy VRAM.
    // When vLLM starts inside WSL2, PyTorch CUDA runtime context, primary context allocations,
    // and NCCL distributed environment buffers take ~1,300-1,400 MB of VRAM BEFORE vLLM checks
    // free memory against requested utilization (init_snapshot.free_memory >= total * util).
    let mut vram_notice: Option<String> = None;
    if def.backend == "vllm" {
        if let Some(snap) = crate::wsl::gpu_snapshot(&distro) {
            if snap.vram_total_mb > 0 && snap.vram_free_mb > 0 {
                let startup_overhead_mb = 1400.0;
                let available_mb = (snap.vram_free_mb as f64 - startup_overhead_mb).max(0.0);
                let safe_ratio = available_mb / snap.vram_total_mb as f64;
                let cap = if snap.vram_total_mb >= 20000 {
                    0.92
                } else if snap.vram_total_mb >= 15000 {
                    0.90
                } else {
                    0.86
                };
                let safe_max = safe_ratio.clamp(0.10, cap);
                let safe_max_rounded = (safe_max * 100.0).floor() / 100.0;
                if def.gpu_mem_util > safe_max_rounded {
                    let orig = def.gpu_mem_util;
                    def.gpu_mem_util = safe_max_rounded;
                    vram_notice = Some(format!(
                    "[LocalLLmPanel] Free VRAM is {} MB / {} MB ({:.1}%). Clamping GPU memory utilization from {:.2} to {:.2} (accounting for PyTorch/NCCL startup buffers) to prevent startup crash.",
                    snap.vram_free_mb, snap.vram_total_mb, (snap.vram_free_mb as f64 / snap.vram_total_mb as f64) * 100.0, orig, def.gpu_mem_util
                ));
                }
            }
        }
    }

    let id_log = id.to_string();
    let id_for_wsl = id_log.clone(); // Clone for WslChild::spawn call later
    let state_log = Arc::clone(state);
    let app_ev = app.map(|a| (*a).clone());
    let log_path = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("local-llm-panel")
        .join("logs")
        .join(format!("{id}.log"));
    let _ = std::fs::create_dir_all(log_path.parent().unwrap_or_else(|| Path::new(".")));
    let log_cb = move |line: String| {
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
        if let Some(ls) = state_log.servers.lock().unwrap().get_mut(&id_log) {
            ls.log_ring.lock().unwrap().push(line.clone());
        }
        if let Some(app) = &app_ev {
            let _ = app.emit(
                "server-log",
                ServerLogEvent {
                    id: id_log.clone(),
                    line,
                },
            );
        }
    };
    let (wsl_child, native_child, wsl_pid) = if def.backend == "llamacpp" {
        let exe = crate::llamacpp_install::executable_from_config(&cfg)
            .ok_or_else(|| anyhow!("llama-server.exe is not installed or configured"))?;
        let raw_model = def.model_path.as_deref().unwrap_or(&def.model_id);
        let mut def = def.clone();
        if let Some(resolved) = resolve_gguf_model_path(raw_model, &cfg.gguf_dir, &distro) {
            def.model_path = Some(resolved.to_string_lossy().into_owned());
        } else if !std::path::Path::new(raw_model).is_file() {
            bail!(
                "GGUF model file not found for '{raw_model}'. Please provide a valid path to a .gguf file or ensure the model is downloaded to your GGUF directory ({}).",
                cfg.gguf_dir
            );
        }
        let help = crate::llamacpp_install::command_output(&exe, &["--help"])
            .unwrap_or_else(|_| cfg.llamacpp_help.clone().unwrap_or_default());
        let args = filter_llamacpp_args(build_llamacpp_args_with_help(&def, &help), &help);
        let child = wsl::NativeChild::spawn(&exe, &args, log_cb)
            .map_err(|e| anyhow!("failed to launch llama-server: {e}"))?;
        (None, Some(child), None)
    } else {
        let script = launch_script(
            &cfg.venv_dir,
            &def,
            &cfg.hf_token,
            &cfg.advanced_settings,
            &cfg.default_env,
        );
        let child = wsl::WslChild::spawn(&distro, &script, &id_for_wsl, log_cb)
            .map_err(|e| anyhow!("failed to launch wsl: {e}"))?;
        let pid = child.pid();
        (Some(child), None, Some(pid))
    };

    let existing_retry = {
        let servers = state.servers.lock().unwrap();
        servers.get(id).map(|ls| ls.crash_retry_count).unwrap_or(0)
    };

    {
        let mut servers = state.servers.lock().unwrap();
        let mut log_ring = VecDequeLog::new();
        if let Some(ref notice) = vram_notice {
            log_ring.push(notice.clone());
            if let Some(app) = app {
                let _ = app.emit(
                    "server-log",
                    ServerLogEvent {
                        id: id.to_string(),
                        line: notice.clone(),
                    },
                );
            }
        }
        servers.insert(
            id.to_string(),
            LiveServer {
                def: def.clone(),
                status: ServerStatus::Starting,
                error: None,
                wsl_child,
                native_child,
                wsl_pid,
                log_ring: std::sync::Mutex::new(log_ring),
                last_metrics: None,
                stopping: false,
                crash_retry_count: existing_retry,
            },
        );
    }
    {
        let mut cfg = state.config.lock().unwrap();
        if let Some(d) = cfg.servers.iter_mut().find(|s| s.id == id) {
            d.was_running = true;
        }
        let _ = cfg.save();
    }
    emit_status(app, id, ServerStatus::Starting, None);

    // Async monitor: health → running, then metrics + liveness loop.
    let app = app.map(|a| (*a).clone());
    let http = state.http.clone();
    let state_task = Arc::clone(state);
    let id_task = id.to_string();
    let model_for_metrics = def.model_id.clone();
    let backend_for_monitor = def.backend.clone();
    tauri::async_runtime::spawn(async move {
        let url = format!("http://127.0.0.1:{}/health", def.port);
        let mut ok = false;
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            let exited = {
                let mut servers = state_task.servers.lock().unwrap();
                servers.get_mut(&id_task).and_then(|ls| {
                    ls.wsl_child
                        .as_mut()
                        .and_then(|c| c.try_wait().ok().flatten())
                        .or_else(|| {
                            ls.native_child
                                .as_mut()
                                .and_then(|c| c.try_wait().ok().flatten())
                        })
                })
            };
            if let Some(exit) = exited {
                let error = process_failure(
                    &state_task,
                    &id_task,
                    &backend_for_monitor,
                    &exit.to_string(),
                );
                emit_status(app.as_ref(), &id_task, ServerStatus::Error, Some(error));
                update_status(&state_task, &id_task, ServerStatus::Error);
                return;
            }
            if let Ok(resp) = http.get(&url).send().await {
                if resp.status().is_success() {
                    ok = true;
                    break;
                }
            }
            tokio::time::sleep(HEALTH_POLL).await;
        }
        if !ok {
            let error = process_failure(
                &state_task,
                &id_task,
                &backend_for_monitor,
                "health check timeout",
            );
            emit_status(app.as_ref(), &id_task, ServerStatus::Error, Some(error));
            update_status(&state_task, &id_task, ServerStatus::Error);
            return;
        }
        {
            let mut servers = state_task.servers.lock().unwrap();
            if let Some(ls) = servers.get_mut(&id_task) {
                ls.crash_retry_count = 0;
            }
        }
        emit_status(app.as_ref(), &id_task, ServerStatus::Running, None);
        update_status(&state_task, &id_task, ServerStatus::Running);

        // Metrics + liveness loop
        let metrics_url = format!("http://127.0.0.1:{}/metrics", def.port);
        let mut last_gen = 0u64;
        let mut last_prompt = 0u64;
        let mut last_measured_at = now_ms();
        let mut first_sample = true;
        loop {
            // Liveness: did the wsl process die on its own (and we're not stopping)?
            let (exited, stopping) = {
                let mut servers = state_task.servers.lock().unwrap();
                let mut ls = servers.get_mut(&id_task);
                let exit = ls.as_mut().and_then(|ls| {
                    ls.wsl_child
                        .as_mut()
                        .and_then(|c| c.try_wait().ok().flatten())
                        .or_else(|| {
                            ls.native_child
                                .as_mut()
                                .and_then(|c| c.try_wait().ok().flatten())
                        })
                });
                let stopping = ls.map(|ls| ls.stopping).unwrap_or(false);
                (exit, stopping)
            };
            if let Some(exit) = exited {
                if !stopping {
                    let auto_restart = {
                        let cfg = state_task.config.lock().unwrap();
                        cfg.auto_restart_crashed
                    };
                    let retry_count = {
                        let mut servers = state_task.servers.lock().unwrap();
                        if let Some(ls) = servers.get_mut(&id_task) {
                            ls.crash_retry_count += 1;
                            ls.crash_retry_count
                        } else {
                            4
                        }
                    };
                    if auto_restart && retry_count <= 3 {
                        let backoff = match retry_count {
                            1 => Duration::from_secs(2),
                            2 => Duration::from_secs(4),
                            _ => Duration::from_secs(8),
                        };
                        eprintln!(
                            "[server] auto-restarting crashed server {} (attempt {}/3 in {:?})",
                            id_task, retry_count, backoff
                        );
                        tokio::time::sleep(backoff).await;
                        let _ = start_server(&state_task, app.as_ref(), &id_task);
                        return;
                    }

                    let error = process_failure(
                        &state_task,
                        &id_task,
                        &backend_for_monitor,
                        &exit.to_string(),
                    );
                    emit_status(app.as_ref(), &id_task, ServerStatus::Error, Some(error));
                    update_status(&state_task, &id_task, ServerStatus::Error);
                } else {
                    update_status(&state_task, &id_task, ServerStatus::Stopped);
                }
                return;
            }

            if let Ok(resp) = http.get(&metrics_url).send().await {
                if let Ok(text) = resp.text().await {
                    if let Some(m) = parse_metrics(&text) {
                        let measured = if first_sample {
                            None
                        } else {
                            let dt = (now_ms() - last_measured_at).max(1000) as f64 / 1000.0;
                            Some(crate::state::MeasuredStats {
                                tokens_per_sec: Some(
                                    ((m.total_generation_tokens - last_gen) as f64) / dt,
                                ),
                                prompt_tokens_per_sec: Some(
                                    ((m.total_prompt_tokens - last_prompt) as f64) / dt,
                                ),
                                total_prompt_tokens: m.total_prompt_tokens,
                                total_generation_tokens: m.total_generation_tokens,
                                requests: m.requests,
                                measured_at_ms: Some(now_ms()),
                            })
                        };
                        last_gen = m.total_generation_tokens;
                        last_prompt = m.total_prompt_tokens;
                        last_measured_at = now_ms();
                        first_sample = false;
                        let snapshot = MetricsSnapshot {
                            running: m.running,
                            waiting: m.waiting,
                            total_prompt_tokens: m.total_prompt_tokens,
                            total_generation_tokens: m.total_generation_tokens,
                            requests: m.requests,
                            measured: measured.clone(),
                        };
                        let tok_s = measured
                            .as_ref()
                            .and_then(|ms| ms.tokens_per_sec)
                            .unwrap_or(0.0);
                        let prompt_tok_s = measured
                            .as_ref()
                            .and_then(|ms| ms.prompt_tokens_per_sec)
                            .unwrap_or(0.0);
                        state_task.record_server_metric(
                            &id_task,
                            crate::state::ServerMetricPoint {
                                timestamp: now_ms(),
                                tok_s,
                                prompt_tok_s,
                                requests_running: m.running,
                                requests_waiting: m.waiting,
                            },
                        );
                        {
                            let mut servers = state_task.servers.lock().unwrap();
                            if let Some(ls) = servers.get_mut(&id_task) {
                                ls.last_metrics = Some(snapshot.clone());
                            }
                            if let Some(app) = &app {
                                let _ = app.emit("server-metrics", snapshot);
                            }
                        }
                        if let Some(ms) = measured {
                            let mut cfg = state_task.config.lock().unwrap();
                            cfg.measured.insert(model_for_metrics.clone(), ms);
                            if let Err(e) = cfg.save() {
                                eprintln!("[server] save measured: {e}");
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(METRICS_POLL).await;
        }
    });

    Ok(())
}

fn update_status(state: &Arc<AppState>, id: &str, status: ServerStatus) {
    let mut servers = state.servers.lock().unwrap();
    if let Some(ls) = servers.get_mut(id) {
        ls.status = status;
    }
}

/// Turn an early backend exit into an actionable error instead of only
/// reporting an opaque exit code.
pub fn startup_failure_hint(backend: &str, exit: &str, log_tail: &str) -> String {
    let lower = log_tail.to_ascii_lowercase();
    let hint = if lower.contains("could not find nvcc")
        || lower.contains("cuda_home")
        || (lower.contains("flashinfer") && lower.contains("nvcc"))
    {
        "vLLM's FlashInfer sampler needs the CUDA toolkit (nvcc) in WSL. Disable it with VLLM_USE_FLASHINFER_SAMPLER=0, or install the CUDA toolkit in WSL."
    } else if lower.contains("failed to find c compiler")
        || lower.contains("compiler not found")
        || (lower.contains("c compiler") && lower.contains("not found"))
    {
        "Install a C compiler in WSL: run 'sudo apt update && sudo apt install -y gcc'."
    } else if lower.contains("python.h")
        || (lower.contains("python") && lower.contains("dev") && lower.contains("not found"))
    {
        "Install Python development headers in WSL: run 'sudo apt update && sudo apt install -y python3.12-dev'."
    } else if lower.contains("no such file or directory: 'ninja'")
        || lower.contains("ninja: command not found")
        || lower.contains("ninja not found")
    {
        "Install ninja build tool in WSL: run 'sudo apt update && sudo apt install -y ninja-build'."
    } else if lower.contains("erroroutofdevicememory")
        || lower.contains("unable to allocate")
        || lower.contains("failed to allocate")
        || lower.contains("failed to fit params")
        || lower.contains("out of memory")
        || lower.contains("cuda error")
        || lower.contains("insufficient memory")
    {
        "Remove explicit -ngl, enable automatic fit, lower the context size, or increase --n-cpu-moe."
    } else if lower.contains("unknown argument")
        || lower.contains("unrecognized option")
        || lower.contains("invalid option")
    {
        "The installed binary does not support one of the configured flags; reinstall/update llama.cpp or remove the unsupported extra argument."
    } else if lower.contains("no such file")
        || lower.contains("cannot open")
        || lower.contains("failed to load model")
    {
        "Check that the GGUF path exists and that all split shards and the optional mmproj file are available."
    } else if lower.contains("address already in use") {
        "Choose another port or stop the process currently listening on this port."
    } else {
        "Open the server log for the backend's full diagnostic output."
    };
    let tail = log_tail
        .lines()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    if tail.is_empty() {
        format!("{backend} process exited ({exit}). {hint}")
    } else {
        format!("{backend} process exited ({exit}). {hint}\n\nRecent log:\n{tail}")
    }
}

fn process_failure(state: &Arc<AppState>, id: &str, backend: &str, exit: &str) -> String {
    let tail = server_logs(state, id, 0);
    startup_failure_hint(backend, exit, &tail)
}

/// Stop a server: SIGTERM to the WSL-side PID via pidfile, then wait; on
/// timeout kill the wsl.exe process tree. Idempotent.
pub fn stop_server(state: &Arc<AppState>, app: Option<&tauri::AppHandle>, id: &str) -> Result<()> {
    let cfg = state.config();
    let distro = state.resolve_distro();
    let (mut wsl_child, mut native_child) = {
        let mut servers = state.servers.lock().unwrap();
        match servers.get_mut(id) {
            Some(ls) if ls.status != ServerStatus::Stopped => {
                ls.stopping = true;
                (ls.wsl_child.take(), ls.native_child.take())
            }
            // Already stopped — idempotent no-op.
            _ => (None, None),
        }
    };

    // 1) Ask the backend process to terminate gracefully.
    let mut term_ok = false;
    if wsl_child.is_some() {
        let pid_from_file = wsl::run_script(
            &distro,
            &format!("cat {}/../run/{id}.pid 2>/dev/null || true", cfg.venv_dir),
        );
        if let Some(pid) = pid_from_file.stdout.trim().parse::<u32>().ok() {
            let kill = wsl::run_script(
                &distro,
                &format!("kill -TERM {pid} 2>/dev/null && echo killed || echo nograb"),
            );
            term_ok = kill.stdout.contains("killed");
        }
    }

    // 2) Wait up to 30s for the child to exit on its own.
    if let Some(c) = wsl_child.as_mut() {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = c.try_wait() {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        let still_alive = c.try_wait().map(|r| r.is_none()).unwrap_or(false);
        if still_alive {
            // 3) Fallback: kill the wsl.exe tree from Windows.
            let mut cmd = std::process::Command::new("taskkill");
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x08000000);
            }
            let _ = cmd
                .args(["/T", "/F", "/PID", &c.pid().to_string()])
                .status();
        }
        if let Some(c) = wsl_child.take() {
            let mut c = c;
            c.join();
        }
    }
    if let Some(c) = native_child.as_mut() {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = c.try_wait() {
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        if c.try_wait().map(|r| r.is_none()).unwrap_or(false) {
            let _ = c.kill();
        }
        if let Some(mut c) = native_child.take() {
            c.join();
        }
    }

    // Mark stopped.
    {
        let mut servers = state.servers.lock().unwrap();
        if let Some(ls) = servers.get_mut(id) {
            ls.status = ServerStatus::Stopped;
            ls.wsl_child = None;
            ls.native_child = None;
            ls.wsl_pid = None;
            ls.error = None;
            ls.last_metrics = None;
            ls.crash_retry_count = 0;
        }
    }
    {
        let mut cfg = state.config.lock().unwrap();
        if let Some(d) = cfg.servers.iter_mut().find(|s| s.id == id) {
            d.was_running = false;
        }
        let _ = cfg.save();
    }
    let _ = term_ok;
    emit_status(app, id, ServerStatus::Stopped, None);
    Ok(())
}

/// Resume any servers that had was_running=true when the app last ran.
pub async fn resume_servers_if_configured(state: &Arc<AppState>, app: Option<&tauri::AppHandle>) {
    let (should_resume, servers_to_resume) = {
        let cfg = state.config.lock().unwrap();
        if !cfg.resume_servers_on_launch {
            (false, Vec::new())
        } else {
            let to_resume: Vec<String> = cfg
                .servers
                .iter()
                .filter(|s| s.was_running)
                .map(|s| s.id.clone())
                .collect();
            (true, to_resume)
        }
    };
    if should_resume && !servers_to_resume.is_empty() {
        let distro = state.resolve_distro();
        if crate::wsl::run_script(&distro, "echo ok").ok {
            for id in servers_to_resume {
                let _ = start_server(state, app, &id);
            }
        }
    }
}

/// Restart = tolerant stop then start.
pub fn restart_server(
    state: &Arc<AppState>,
    app: Option<&tauri::AppHandle>,
    id: &str,
) -> Result<()> {
    let _ = stop_server(state, app, id);
    start_server(state, app, id)
}

// ---------------------------------------------------------------------------
// Metrics parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub running: u64,
    pub waiting: u64,
    pub total_prompt_tokens: u64,
    pub total_generation_tokens: u64,
    pub requests: u64,
}

/// Parse the Prometheus text from vLLM `/metrics`.
pub fn parse_metrics(text: &str) -> Option<Metrics> {
    let mut m = Metrics::default();
    let mut saw = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((name, value)) = line.split_once(' ') else {
            continue;
        };
        let Ok(val) = value.trim().parse::<f64>() else {
            continue;
        };
        match name {
            "vllm:num_requests_running"
            | "vllm:num_requests_running_gauge"
            | "llama_requests_processing"
            | "llamacpp_requests_processing" => {
                m.running = val as u64;
                saw = true;
            }
            "vllm:num_requests_waiting" | "vllm:num_requests_waiting_gauge" => {
                m.waiting = val as u64;
                saw = true;
            }
            "vllm:prompt_tokens_total"
            | "vllm:prompt_tokens_succeeded_total"
            | "llama_prompt_tokens_total"
            | "llama_prompt_tokens"
            | "llamacpp_prompt_tokens_total"
            | "llamacpp_prompt_tokens" => {
                m.total_prompt_tokens = val as u64;
                saw = true;
            }
            "vllm:generation_tokens_total"
            | "vllm:generation_tokens_succeeded_total"
            | "llama_generation_tokens_total"
            | "llama_tokens_predicted_total"
            | "llama_generation_tokens"
            | "llamacpp_tokens_predicted_total"
            | "llamacpp_generation_tokens_total" => {
                m.total_generation_tokens = val as u64;
                saw = true;
            }
            "vllm:generation_tokens_failed_total" => {
                m.total_generation_tokens += val as u64;
                saw = true;
            }
            "vllm:request_success_total"
            | "vllm:requests_succeeded_total"
            | "llama_requests_total"
            | "llama_requests_completed_total"
            | "llamacpp_requests_total"
            | "llamacpp_requests_completed_total" => {
                m.requests = val as u64;
                saw = true;
            }
            _ => {}
        }
    }

    if saw {
        Some(m)
    } else {
        None
    }
}

pub async fn chat_with_tools(
    state: &Arc<AppState>,
    server_id: &str,
    messages: Vec<serde_json::Value>,
    max_tokens: u32,
    disable_thinking: bool,
) -> Result<serde_json::Value> {
    let (port, model, api_key) = {
        let servers = state.servers.lock().unwrap();
        let ls = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("unknown server {server_id}"))?;
        (
            ls.def.port,
            ls.def.effective_model_name(),
            server_api_key(state, &ls.def),
        )
    };
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "chat_template_kwargs": if disable_thinking {
            serde_json::json!({"enable_thinking": false})
        } else {
            serde_json::json!({})
        }
    });
    let mut request = state
        .http
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    Ok(request.send().await?.error_for_status()?.json().await?)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

pub fn list_servers(
    state: &Arc<AppState>,
) -> Vec<(
    ServerDef,
    ServerStatus,
    Option<String>,
    Option<MetricsSnapshot>,
)> {
    let defs = state.config().servers.clone();
    let servers = state.servers.lock().unwrap();
    defs.into_iter()
        .map(|def| {
            if let Some(ls) = servers.get(&def.id) {
                (def, ls.status, ls.error.clone(), ls.last_metrics.clone())
            } else {
                (def, ServerStatus::Stopped, None, None)
            }
        })
        .collect()
}

/// Estimated weight footprint (GB) of all RUNNING servers — for the VRAM
/// guidance bar vs. current free VRAM.
pub fn running_weight_gb(state: &Arc<AppState>) -> f64 {
    let servers = state.servers.lock().unwrap();
    servers
        .values()
        .filter(|ls| ls.status == ServerStatus::Running && ls.def.params_b.is_some())
        .map(|ls| estimate::weight_gb(ls.def.params_b.unwrap_or(0.0), &ls.def.quant))
        .sum()
}

pub fn apply_vllm_auth(
    mut req: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    if let Some(key) = api_key.map(str::trim).filter(|k| !k.is_empty()) {
        req = req.bearer_auth(key);
    }
    req
}

fn server_api_key(state: &Arc<AppState>, def: &ServerDef) -> Option<String> {
    def.api_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .or_else(|| state.config().advanced_settings.api_key.clone())
}

/// Issue a chat completion against an instruct server (server-side, no CORS).
pub async fn chat(
    state: &Arc<AppState>,
    server_id: &str,
    messages: Vec<serde_json::Value>,
) -> Result<serde_json::Value> {
    let def = state
        .config()
        .find_server(server_id)
        .cloned()
        .ok_or_else(|| anyhow!("no server {server_id}"))?;
    if def.task != "instruct" {
        bail!("server {server_id} is not an instruct server");
    }
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", def.port);
    let body = serde_json::json!({
        "model": def.effective_model_name(),
        "messages": messages,
        "stream": false,
    });
    let api_key = server_api_key(&state, &def);
    let req = state.http.post(&url).json(&body);
    let resp = apply_vllm_auth(req, api_key.as_deref())
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("chat failed ({}): {}", status, text);
    }
    Ok(resp.json().await.context("chat response JSON")?)
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenPayload {
    pub request_id: String,
    pub server_id: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatDonePayload {
    pub request_id: String,
    pub server_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCancelPayload {
    pub request_id: String,
    pub server_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatErrorPayload {
    pub request_id: String,
    pub server_id: String,
    pub error: String,
}

pub fn parse_sse_token(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return None;
    }
    let data = trimmed.trim_start_matches("data:").trim();
    if data == "[DONE]" || data.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
        if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }
        if let Some(text) = v["choices"][0]["text"].as_str() {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Stream chat completion tokens over SSE and emit Tauri events.
pub async fn chat_stream(
    app: Option<tauri::AppHandle>,
    state: Arc<AppState>,
    request_id: String,
    server_id: String,
    messages: Vec<crate::state::ChatMessage>,
    temperature: Option<f32>,
) {
    let cancel_notify = Arc::new(tokio::sync::Notify::new());
    state
        .chat_cancels
        .lock()
        .unwrap()
        .insert(request_id.clone(), cancel_notify.clone());

    let res: Result<bool> = async {
        let def = state
            .config()
            .find_server(&server_id)
            .cloned()
            .ok_or_else(|| anyhow!("no server {server_id}"))?;
        if def.task != "instruct" {
            bail!("server {server_id} is not an instruct server");
        }
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", def.port);
        let mut body = serde_json::json!({
            "model": def.effective_model_name(),
            "messages": messages
                .iter()
                .map(|m| m.payload_body())
                .collect::<Vec<_>>(),
            "stream": true,
        });
        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let api_key = server_api_key(&state, &def);
        let req = state.http.post(&url).json(&body);
        let mut resp = apply_vllm_auth(req, api_key.as_deref())
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("chat stream failed ({}): {}", status, text);
        }

        let mut buffer = String::new();
        loop {
            tokio::select! {
                _ = cancel_notify.notified() => {
                    return Ok(true); // cancelled
                }
                chunk_opt = resp.chunk() => {
                    match chunk_opt {
                        Ok(Some(chunk)) => {
                            buffer.push_str(&String::from_utf8_lossy(&chunk));
                            while let Some(newline_pos) = buffer.find('\n') {
                                let line = buffer[..newline_pos].to_string();
                                buffer.drain(..=newline_pos);
                                if let Some(token) = parse_sse_token(&line) {
                                    if let Some(ref a) = app {
                                        let _ = a.emit("chat-token", ChatTokenPayload {
                                            request_id: request_id.clone(),
                                            server_id: server_id.clone(),
                                            token,
                                        });
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            break;
                        }
                        Err(e) => {
                            bail!("stream read error: {e}");
                        }
                    }
                }
            }
        }

        if !buffer.is_empty() {
            if let Some(token) = parse_sse_token(&buffer) {
                if let Some(ref a) = app {
                    let _ = a.emit(
                        "chat-token",
                        ChatTokenPayload {
                            request_id: request_id.clone(),
                            server_id: server_id.clone(),
                            token,
                        },
                    );
                }
            }
        }

        Ok(false)
    }
    .await;

    state.chat_cancels.lock().unwrap().remove(&request_id);

    match res {
        Ok(true) => {
            if let Some(ref a) = app {
                let _ = a.emit(
                    "chat-cancel",
                    ChatCancelPayload {
                        request_id,
                        server_id,
                    },
                );
            }
        }
        Ok(false) => {
            if let Some(ref a) = app {
                let _ = a.emit(
                    "chat-done",
                    ChatDonePayload {
                        request_id,
                        server_id,
                    },
                );
            }
        }
        Err(e) => {
            if let Some(ref a) = app {
                let _ = a.emit(
                    "chat-error",
                    ChatErrorPayload {
                        request_id,
                        server_id,
                        error: e.to_string(),
                    },
                );
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkStepPayload {
    pub server_id: String,
    pub step: usize,
    pub total_steps: usize,
    pub prompt_tok_s: f64,
    pub gen_tok_s: f64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCancelPayload {
    pub server_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkErrorPayload {
    pub server_id: String,
    pub error: String,
}

pub fn standardized_benchmark_prompts() -> [&'static str; 3] {
    [
        "Explain the core difference between synchronous and asynchronous I/O in three brief bullet points.",
        "Write a Python function `lru_cache_custom(capacity)` that implements a simple Least Recently Used cache using a doubly linked list and hash map. Include brief docstrings and comments explaining how eviction works.",
        "Analyze the following technical architectural pattern: Event-driven architecture (EDA) decouples producers from consumers through asynchronous event brokers such as Apache Kafka, RabbitMQ, and AWS SQS/SNS. In high-throughput distributed systems, event-driven designs provide advantages such as horizontal scalability, resilient fault isolation, and temporal decoupling, while introducing challenges including eventual consistency, message ordering guarantees, distributed tracing complexity, and idempotent consumer handling. In contrast, synchronous RPC models such as gRPC and REST provide immediate request-response semantics, simpler error handling, and strict consistency at the cost of tighter coupling and susceptibility to cascading latency bottlenecks. Based on this, please provide: 1) A trade-off matrix comparing Event-Driven vs Synchronous RPC across latency, fault isolation, operational complexity, and data consistency. 2) Three specific scenarios where EDA should be preferred, and three where synchronous RPC is the superior choice. 3) Recommendations for mitigating eventual consistency anomalies."
    ]
}

pub async fn run_benchmark(app: Option<tauri::AppHandle>, state: Arc<AppState>, server_id: String) {
    let cancel_notify = Arc::new(tokio::sync::Notify::new());
    state
        .benchmark_cancels
        .lock()
        .unwrap()
        .insert(server_id.clone(), cancel_notify.clone());

    let res: Result<Option<crate::state::BenchmarkRun>> = async {
        let def = state
            .config()
            .find_server(&server_id)
            .cloned()
            .ok_or_else(|| anyhow!("no server {server_id}"))?;
        if def.task != "instruct" {
            bail!("server {server_id} is not an instruct server");
        }
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", def.port);
        let prompts = standardized_benchmark_prompts();
        let total_steps = prompts.len();

        let mut step_results: Vec<(f64, f64, f64)> = Vec::new();

        for (idx, prompt) in prompts.iter().enumerate() {
            let step = idx + 1;
            let body = serde_json::json!({
                "model": def.effective_model_name(),
                "messages": [
                    {"role": "system", "content": "You are a concise, accurate benchmark runner."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "stream": false
            });

            let t0 = Instant::now();
            let api_key = server_api_key(&state, &def);
            let req = state.http.post(&url).json(&body);
            let send_future = apply_vllm_auth(req, api_key.as_deref()).send();

            let resp = tokio::select! {
                _ = cancel_notify.notified() => {
                    return Ok(None);
                }
                res = send_future => {
                    res.with_context(|| format!("POST {url}"))?
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                bail!("benchmark step {step} failed ({status}): {text}");
            }

            let json: serde_json::Value = resp.json().await.context("read benchmark json")?;
            let elapsed_s = t0.elapsed().as_secs_f64();
            let latency_ms = t0.elapsed().as_millis() as f64;

            let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(60) as f64;
            let completion_tokens =
                json["usage"]["completion_tokens"].as_u64().unwrap_or(64) as f64;

            let gen_tok_s = if elapsed_s > 0.0 {
                completion_tokens / elapsed_s
            } else {
                0.0
            };
            let prompt_tok_s = if elapsed_s > 0.0 {
                prompt_tokens / (elapsed_s * 0.25).max(0.01)
            } else {
                0.0
            };

            step_results.push((prompt_tok_s, gen_tok_s, latency_ms));

            if let Some(ref a) = app {
                let _ = a.emit(
                    "benchmark-step",
                    BenchmarkStepPayload {
                        server_id: server_id.clone(),
                        step,
                        total_steps,
                        prompt_tok_s,
                        gen_tok_s,
                        latency_ms,
                    },
                );
            }
        }

        let n = step_results.len() as f64;
        let avg_prompt_tok_s = step_results.iter().map(|(p, _, _)| p).sum::<f64>() / n;
        let avg_gen_tok_s = step_results.iter().map(|(_, g, _)| g).sum::<f64>() / n;
        let avg_latency_ms = step_results.iter().map(|(_, _, l)| l).sum::<f64>() / n;

        let now_sec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let run = crate::state::BenchmarkRun {
            id: format!("bm_{now_sec}_{}", def.port),
            server_id: server_id.clone(),
            model_id: def.model_id.clone(),
            quant: Some(def.quant.clone()),
            timestamp: now_sec,
            prompt_tok_s: avg_prompt_tok_s,
            gen_tok_s: avg_gen_tok_s,
            latency_ms: avg_latency_ms,
            prompt_count: total_steps,
        };

        // Persist run
        {
            let mut bms = state.benchmarks.lock().unwrap();
            bms.insert(0, run.clone());
            let _ = crate::state::BenchmarkRun::save_all(&bms);
        }

        // Update measured stats in config
        {
            let mut cfg = state.config.lock().unwrap();
            let entry = cfg.measured.entry(def.model_id.clone()).or_default();
            entry.tokens_per_sec = Some(avg_gen_tok_s);
            entry.prompt_tokens_per_sec = Some(avg_prompt_tok_s);
            entry.measured_at_ms = Some(now_sec * 1000);
            let _ = cfg.save();
        }

        Ok(Some(run))
    }
    .await;

    state.benchmark_cancels.lock().unwrap().remove(&server_id);

    match res {
        Ok(Some(run)) => {
            if let Some(ref a) = app {
                let _ = a.emit("benchmark-done", run);
            }
        }
        Ok(None) => {
            if let Some(ref a) = app {
                let _ = a.emit("benchmark-cancel", BenchmarkCancelPayload { server_id });
            }
        }
        Err(e) => {
            if let Some(ref a) = app {
                let _ = a.emit(
                    "benchmark-error",
                    BenchmarkErrorPayload {
                        server_id,
                        error: e.to_string(),
                    },
                );
            }
        }
    }
}

pub fn server_logs(state: &Arc<AppState>, server_id: &str, since: usize) -> String {
    let servers = state.servers.lock().unwrap();
    match servers.get(server_id) {
        Some(ls) => ls.log_ring.lock().unwrap().since_line(since),
        None => String::new(),
    }
}

pub fn server_metrics(state: &Arc<AppState>, server_id: &str) -> Option<MetricsSnapshot> {
    let servers = state.servers.lock().unwrap();
    servers
        .get(server_id)
        .and_then(|ls| ls.last_metrics.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_allocator_skips_bound_ports() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let taken = listener.local_addr().unwrap().port();
        let p = alloc_port(&[taken]).unwrap();
        assert_ne!(p, taken);
        assert!(p >= DEFAULT_PORT_START);
    }

    #[test]
    fn port_allocator_respects_existing_defs() {
        let p = alloc_port(&[8000, 8001, 8002]).unwrap();
        assert!(p >= 8003);
    }

    fn def(model: &str, task: &str, port: u16, quant: &str, served: Option<&str>) -> ServerDef {
        ServerDef {
            backend: "vllm".into(),
            id: "s1".into(),
            name: "test".into(),
            model_id: model.into(),
            task: task.into(),
            port,
            gpu_mem_util: 0.92,
            max_model_len: Some(2048),
            quant: quant.into(),
            served_model_name: served.map(|s| s.into()),
            enforce_eager: true,
            params_b: None,
            swap_space_gb: None,
            cpu_offload_gb: None,
            was_running: false,
            model_path: None,
            mmproj_path: None,
            ctx_size: None,
            n_gpu_layers: Some(99),
            n_cpu_moe: None,
            fit: true,
            fit_target: None,
            device: None,
            api_key: None,
            log_verbosity: None,
            flash_attn: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            threads: None,
            batch_size: None,
            ubatch_size: None,
            parallel: 1,
            jinja: true,
            no_kv_offload: false,
            metrics: true,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
        }
    }

    #[test]
    fn launch_script_instruct() {
        let script = launch_script(
            "~/llm-lp/.venv",
            &def("Qwen/Qwen2.5-0.5B-Instruct", "instruct", 8010, "fp16", None),
            "",
            &crate::state::AdvancedSettings::default(),
            &BTreeMap::new(),
        );
        assert!(script.contains("exec python -m vllm.entrypoints.openai.api_server"));
        assert!(script.contains("--model 'Qwen/Qwen2.5-0.5B-Instruct'"));
        assert!(script.contains("--port 8010"));
        assert!(script.contains("--gpu-memory-utilization 0.92"));
        assert!(script.contains("--max-model-len 2048"));
        // Paths are now shell-quoted
        assert!(script.contains("$$ > '~/llm-lp/.venv'/../run/s1.pid"));
        assert!(!script.contains("--runner pooling"));
        assert!(!script.contains("--quantization"));
        assert!(!script.contains("export HF_TOKEN="));
    }

    #[test]
    fn llamacpp_args_include_defaults_and_optionals() {
        let mut d = def("model.gguf", "instruct", 8123, "gguf", None);
        d.backend = "llamacpp".into();
        d.model_path = Some("C:\\models\\model.gguf".into());
        d.mmproj_path = Some("C:\\models\\mmproj.gguf".into());
        d.ctx_size = Some(32768);
        d.n_cpu_moe = Some(24);
        d.threads = Some(12);
        d.batch_size = Some(512);
        d.ubatch_size = Some(128);
        d.extra_args = vec!["--no-mmap".into(), "--verbose".into()];
        let args = build_llamacpp_args(&d);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "-m" && w[1] == "C:\\models\\model.gguf"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--mmproj" && w[1] == "C:\\models\\mmproj.gguf"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--n-cpu-moe" && w[1] == "24"));
        assert!(args.windows(2).any(|w| w[0] == "-fa" && w[1] == "on"));
        assert!(args.ends_with(&["--no-mmap".into(), "--verbose".into()]));
    }

    #[test]
    fn llamacpp_args_omit_unset_optional_values() {
        let mut d = def("model.gguf", "instruct", 8123, "gguf", None);
        d.backend = "llamacpp".into();
        d.flash_attn = false;
        d.jinja = false;
        d.metrics = false;
        d.parallel = 0;
        d.cache_type_k.clear();
        d.cache_type_v.clear();
        let args = build_llamacpp_args(&d);
        assert!(!args.contains(&"-fa".into()));
        assert!(!args.contains(&"--jinja".into()));
        assert!(!args.contains(&"--metrics".into()));
        assert!(!args.contains(&"-np".into()));
    }

    #[test]
    fn llamacpp_args_follow_installed_help_capabilities() {
        let mut d = def("model.gguf", "instruct", 8123, "gguf", None);
        d.backend = "llamacpp".into();
        d.n_cpu_moe = Some(24);
        let args = filter_llamacpp_args(
            build_llamacpp_args(&d),
            "  -m FNAME  --host HOST  --port PORT  -ngl N  -c N\n  -fa on\n  --metrics\n  --jinja\n",
        );
        assert!(!args.contains(&"--n-cpu-moe".into()));
        assert!(args.contains(&"-fa".into()));
        assert!(args.contains(&"--metrics".into()));
    }

    #[test]
    fn llamacpp_fit_is_capability_gated_and_targeted() {
        let mut d = def("qwen35moe-agentworld.gguf", "instruct", 8123, "gguf", None);
        d.backend = "llamacpp".into();
        d.n_gpu_layers = None;
        d.n_cpu_moe = Some(24);
        d.fit_target = Some(1536);
        d.extra_args = vec![
            "--override-kv".into(),
            "qwen35moe.block_count=int:41,custom.foo=bool:true".into(),
        ];
        let args = build_llamacpp_args_with_help(&d, "--fit on --fit-target MiB --device");
        assert!(!args.contains(&"-ngl".into()));
        assert!(args.windows(2).any(|w| w == ["--fit", "on"]));
        assert!(args.windows(2).any(|w| w == ["--fit-target", "1536"]));
        let override_value = args
            .windows(2)
            .find(|w| w[0] == "--override-kv")
            .map(|w| w[1].clone())
            .unwrap();
        assert!(override_value.contains("qwen35moe.block_count=int:41"));
        assert!(!override_value.contains("int:40"));
        assert_eq!(override_value.matches("qwen35moe.block_count=").count(), 1);

        let no_fit = build_llamacpp_args_with_help(&d, "--device --metrics");
        assert!(!no_fit.contains(&"--fit".into()));
        assert!(no_fit.windows(2).any(|w| w == ["--n-cpu-moe", "24"]));
    }

    #[test]
    fn startup_failure_hint_explains_common_cuda_failure() {
        let hint = startup_failure_hint(
            "llamacpp",
            "exit code: 1",
            "W common_fit_params: failed to fit params to free device memory\n\
             ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n\
             E alloc_tensor_range: failed to allocate Vulkan1 buffer\n\
             E llama_model_load: error loading model: unable to allocate Vulkan1 buffer",
        );
        assert!(hint.contains("automatic fit"));
        assert!(hint.contains("n-cpu-moe"));
        assert!(hint.contains("llamacpp process exited"));
    }

    #[test]
    fn startup_failure_hint_flashinfer_nvcc() {
        let hint = startup_failure_hint(
            "vllm",
            "exit code: 1",
            "ERROR flashinfer/sampling.py ... get_sampling_module().top_k_mask_logits\n\
             ERROR flashinfer/jit/cpp_ext.py, line 61, in get_cuda_path\n\
             RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
        );
        assert!(hint.contains("FlashInfer"));
        assert!(hint.contains("nvcc") || hint.contains("CUDA toolkit"));
        assert!(hint.contains("VLLM_USE_FLASHINFER_SAMPLER=0"));
    }

    #[test]
    fn startup_failure_hint_c_compiler_missing() {
        let hint = startup_failure_hint(
            "vllm",
            "exit code: 1",
            "error: failed to find C compiler\n\
             note: the msvc targets depend on the msvc linker",
        );
        assert!(hint.contains("gcc"));
    }

    #[test]
    fn startup_failure_hint_python_h_missing() {
        let hint = startup_failure_hint(
            "vllm",
            "exit code: 1",
            "fatal error: Python.h: No such file or directory\n\
             #include <Python.h>",
        );
        assert!(hint.contains("python3.12-dev"));
    }

    #[test]
    fn startup_failure_hint_ninja_missing() {
        let hint = startup_failure_hint(
            "vllm",
            "exit code: 1",
            "No such file or directory: 'ninja'\n\
             ninja: command not found",
        );
        assert!(hint.contains("ninja-build"));
    }

    #[test]
    fn llamacpp_metrics_sample_is_parsed() {
        let sample = r#"
# HELP llama_tokens_predicted_total Tokens generated
llama_tokens_predicted_total 42
llama_prompt_tokens 18
llama_requests_completed_total 2
llama_requests_processing 1
"#;
        let metrics = parse_metrics(sample).expect("llama metrics should be recognized");
        assert_eq!(metrics.total_generation_tokens, 42);
        assert_eq!(metrics.total_prompt_tokens, 18);
        assert_eq!(metrics.requests, 2);
        assert_eq!(metrics.running, 1);
    }

    #[test]
    fn launch_script_embed_with_quant_and_served() {
        let script = launch_script(
            "~/llm-lp/.venv",
            &def(
                "BAAI/bge-small-en-v1.5",
                "embed",
                8020,
                "fp8",
                Some("embedder"),
            ),
            "hf_secret_123",
            &crate::state::AdvancedSettings::default(),
            &BTreeMap::new(),
        );
        assert!(script.contains("--runner pooling"));
        assert!(script.contains("--quantization fp8"));
        assert!(script.contains("--served-model-name 'embedder'"));
        assert!(script.contains("--max-model-len 2048"));
        assert!(script.contains("export HF_TOKEN='hf_secret_123'"));
    }

    #[test]
    fn test_launch_script_with_advanced_settings() {
        let adv = crate::state::AdvancedSettings {
            hf_home: Some("/mnt/d/ai/hf".into()),
            hf_offline: true,
            host: "0.0.0.0".into(),
            api_key: Some("sk-test-123".into()),
            gateway_enabled: true,
            gateway_port: 11434,
            kv_cache_dtype: "fp8".into(),
            enable_prefix_caching: true,
            enable_chunked_prefill: true,
            max_num_seqs: Some(64),
            disable_custom_all_reduce: true,
            log_level: "DEBUG".into(),
            extra_vllm_args: Some("--tensor-parallel-size 2".into()),
            custom_env_vars: Some("CUDA_VISIBLE_DEVICES=0,1\n# comment\nNCCL_DEBUG=INFO".into()),
        };
        let script = launch_script(
            "~/llm-lp/.venv",
            &def(
                "meta-llama/Llama-3-8B-Instruct",
                "instruct",
                8000,
                "fp16",
                None,
            ),
            "hf_token_xyz",
            &adv,
            &BTreeMap::new(),
        );
        assert!(script.contains("export HF_HOME='/mnt/d/ai/hf'"));
        assert!(script.contains("mkdir -p '/mnt/d/ai/hf'"));
        assert!(script.contains("export HF_HUB_OFFLINE='1'"));
        assert!(script.contains("export VLLM_LOGGING_LEVEL='DEBUG'"));
        assert!(script.contains("export CUDA_VISIBLE_DEVICES='0,1'"));
        assert!(script.contains("export NCCL_DEBUG='INFO'"));
        assert!(script.contains("--host 0.0.0.0"));
        assert!(script.contains("--api-key 'sk-test-123'"));
        assert!(script.contains("--kv-cache-dtype fp8"));
        assert!(script.contains("--enable-prefix-caching"));
        assert!(script.contains("--enable-chunked-prefill"));
        assert!(script.contains("--max-num-seqs 64"));
        assert!(script.contains("--disable-custom-all-reduce"));
        assert!(script.contains("--tensor-parallel-size 2"));
    }

    #[test]
    fn test_build_start_command_swap_and_cpu_offload() {
        let mut d = def("Qwen/Qwen2.5-7B-Instruct", "instruct", 8010, "fp16", None);
        d.swap_space_gb = Some(8);
        d.cpu_offload_gb = Some(4);
        let cmd = build_start_command(&d, "");
        assert!(
            cmd.contains("--kv-offloading-size 8"),
            "command must include --kv-offloading-size 8: {cmd}"
        );
        assert!(
            cmd.contains("--cpu-offload-gb 4"),
            "command must include --cpu-offload-gb 4: {cmd}"
        );
        assert!(
            cmd.contains("export VLLM_WSL2_ENABLE_PIN_MEMORY='1'"),
            "command must include export VLLM_WSL2_ENABLE_PIN_MEMORY='1': {cmd}"
        );
        assert!(cmd.contains("export VLLM_WORKER_MULTIPROC_METHOD='spawn'"));
    }

    #[test]
    fn test_launch_script_prequantized_awq_omits_quantization_flag() {
        let script = launch_script(
            "~/llm-lp/.venv",
            &def(
                "TelperionAI/Huihui-Qwen3.8-27B-abliterated-INT4-AWQ-GPTQ",
                "instruct",
                8000,
                "awq",
                None,
            ),
            "",
            &crate::state::AdvancedSettings::default(),
            &BTreeMap::new(),
        );
        assert!(!script.contains("--quantization"));
    }

    #[test]
    fn test_launch_script_prequantized_fp8_omits_quantization_flag() {
        let script = launch_script(
            "~/llm-lp/.venv",
            &def(
                "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
                "instruct",
                8000,
                "fp8",
                None,
            ),
            "",
            &crate::state::AdvancedSettings::default(),
            &BTreeMap::new(),
        );
        assert!(!script.contains("--quantization"));
    }

    #[test]
    fn test_build_start_command_swap_and_cpu_offload_omitted_when_zero_or_none() {
        let mut d = def("Qwen/Qwen2.5-7B-Instruct", "instruct", 8010, "fp16", None);
        d.swap_space_gb = Some(0);
        d.cpu_offload_gb = Some(0);
        let cmd_zero = build_start_command(&d, "");
        assert!(
            !cmd_zero.contains("--kv-offloading-size") && !cmd_zero.contains("--swap-space"),
            "command must not include swap flags: {cmd_zero}"
        );
        assert!(
            !cmd_zero.contains("--cpu-offload-gb"),
            "command must not include --cpu-offload-gb: {cmd_zero}"
        );

        d.swap_space_gb = None;
        d.cpu_offload_gb = None;
        let cmd_none = build_start_command(&d, "");
        assert!(
            !cmd_none.contains("--kv-offloading-size") && !cmd_none.contains("--swap-space"),
            "command must not include swap flags: {cmd_none}"
        );
        assert!(
            !cmd_none.contains("--cpu-offload-gb"),
            "command must not include --cpu-offload-gb: {cmd_none}"
        );
    }

    #[test]
    fn test_build_start_command_max_model_len_zero_falls_back_to_4096() {
        let mut d = def("Qwen/Qwen2.5-7B-Instruct", "instruct", 8010, "fp16", None);
        d.max_model_len = Some(0);
        let cmd = build_start_command(&d, "");
        assert!(
            cmd.contains("--max-model-len 4096"),
            "max-model-len 0 must fall back to 4096: {cmd}"
        );
        assert!(!cmd.contains("--max-model-len 0"));
    }

    #[test]
    fn metrics_parse_both_generations() {
        let text = "# HELP vllm:generation_tokens_total Total generation tokens\nvllm:generation_tokens_total 1234\nvllm:prompt_tokens_total 567\nvllm:num_requests_running 2\nvllm:num_requests_waiting 3\n";
        let m = parse_metrics(text).unwrap();
        assert_eq!(m.total_generation_tokens, 1234);
        assert_eq!(m.total_prompt_tokens, 567);
        assert_eq!(m.running, 2);
        assert_eq!(m.waiting, 3);

        let v1 = "vllm:generation_tokens_succeeded_total 10\nvllm:generation_tokens_failed_total 4\nvllm:num_requests_running 1\n";
        let m1 = parse_metrics(v1).unwrap();
        assert_eq!(m1.total_generation_tokens, 14);
    }

    #[test]
    fn metrics_parse_ignores_garbage() {
        assert!(parse_metrics("hello world\nnot a metric").is_none());
    }

    #[test]
    fn test_parse_sse_token() {
        use super::parse_sse_token;
        assert_eq!(
            parse_sse_token(r#"data: {"choices":[{"delta":{"content":"Hello world"}}]}"#),
            Some("Hello world".to_string())
        );
        assert_eq!(
            parse_sse_token(r#"data: {"choices":[{"text":"Alternative"}]}"#),
            Some("Alternative".to_string())
        );
        assert_eq!(parse_sse_token("data: [DONE]"), None);
        assert_eq!(parse_sse_token("data:   "), None);
        assert_eq!(parse_sse_token(": ping"), None);
        assert_eq!(parse_sse_token(""), None);
        assert_eq!(parse_sse_token("random text"), None);
    }

    #[test]
    fn test_standardized_benchmark_prompts() {
        use super::standardized_benchmark_prompts;
        let prompts = standardized_benchmark_prompts();
        assert_eq!(prompts.len(), 3);
        assert!(prompts[0].len() < prompts[1].len());
        assert!(prompts[1].len() < prompts[2].len());
    }

    #[test]
    fn test_apply_vllm_auth_header() {
        use super::apply_vllm_auth;
        use reqwest::Client;
        let client = Client::new();

        let req = client.post("http://127.0.0.1:8000/v1/chat/completions");
        let req_with_auth = apply_vllm_auth(req, Some("sk-secret-123")).build().unwrap();
        assert_eq!(
            req_with_auth
                .headers()
                .get("Authorization")
                .unwrap()
                .to_str()
                .unwrap(),
            "Bearer sk-secret-123"
        );

        let req_blank = client.post("http://127.0.0.1:8000/v1/chat/completions");
        let req_no_auth = apply_vllm_auth(req_blank, None).build().unwrap();
        assert!(req_no_auth.headers().get("Authorization").is_none());
    }

    #[test]
    fn test_validate_env_name() {
        assert!(validate_env_name("FOO").is_ok());
        assert!(validate_env_name("FOO_BAR").is_ok());
        assert!(validate_env_name("_FOO").is_ok());
        assert!(validate_env_name("FOO123").is_ok());
        assert!(validate_env_name("VLLM_USE_FLASHINFER_SAMPLER").is_ok());
        assert!(validate_env_name("").is_err());
        assert!(validate_env_name("123FOO").is_err());
        assert!(validate_env_name("FOO-BAR").is_err());
        assert!(validate_env_name("FOO BAR").is_err());
        assert!(validate_env_name("FOO.BAR").is_err());
    }

    #[test]
    fn test_shell_quote() {
        // Basic cases
        assert_eq!(shell_quote("simple"), "'simple'");
        assert_eq!(shell_quote("hello world"), "'hello world'");
        assert_eq!(shell_quote("don't"), "'don'\\''t'");
        // Single quote character: shell representation is '\\''\\'' (wrapped: '\\''\\''\\'')
        // Actually: ' + replace(','\\'') + ' = ' + '\\'' + ' = ''\\'''
        assert_eq!(shell_quote("'"), "''\\'''");
        assert_eq!(shell_quote("a'b'c"), "'a'\\''b'\\''c'");
        // Empty string
        assert_eq!(shell_quote(""), "''");
        // Dollar sign (should be preserved, not expanded)
        assert_eq!(shell_quote("$HOME"), "'$HOME'");
        // Backticks (should be preserved)
        assert_eq!(shell_quote("`id`"), "'`id`'");
        // Backslashes
        assert_eq!(shell_quote(r"C:\path"), "'C:\\path'");
        // Stray leading quote (the bug we're fixing)
        // shell_quote("'0") = ' + replace("'0", "'\\''") + ' = ' + '\\''0 + ' = ''\\''0'
        assert_eq!(shell_quote("'0"), "''\\''0'");
        // Unicode
        assert_eq!(shell_quote("café"), "'café'");
        assert_eq!(shell_quote("测试"), "'测试'");
        // Hostile value: should not inject shell commands
        let hostile = "'; rm -rf ~ #";
        let escaped = shell_quote(hostile);
        assert!(escaped.starts_with("'") && escaped.ends_with("'"));
        // The escaped value should contain the escape sequence for single quotes
        assert!(escaped.contains("'\\''"));
        // Verify the value can be recovered by shell (tested via integration test)
    }

    #[test]
    fn test_sanitize_env_value() {
        // Normal values unchanged
        assert_eq!(sanitize_env_value("0"), "0");
        assert_eq!(sanitize_env_value("1"), "1");
        assert_eq!(sanitize_env_value("hello"), "hello");
        assert_eq!(sanitize_env_value("hello world"), "hello world");
        // Strip surrounding single quotes
        assert_eq!(sanitize_env_value("'0'"), "0");
        assert_eq!(sanitize_env_value("'hello'"), "hello");
        // Strip surrounding double quotes
        assert_eq!(sanitize_env_value("\"0\""), "0");
        assert_eq!(sanitize_env_value("\"hello\""), "hello");
        // Trim whitespace
        assert_eq!(sanitize_env_value("  0  "), "0");
        assert_eq!(sanitize_env_value("  '0'  "), "0");
        // No strip if not matching pair
        assert_eq!(sanitize_env_value("'0"), "'0");
        assert_eq!(sanitize_env_value("0'"), "0'");
        assert_eq!(sanitize_env_value("\"0'"), "\"0'");
    }

    #[test]
    fn test_launch_script_merges_global_and_per_server_env() {
        use std::collections::BTreeMap;
        let mut global_env = BTreeMap::new();
        global_env.insert("VLLM_USE_FLASHINFER_SAMPLER".to_string(), "0".to_string());
        global_env.insert("GLOBAL_VAR".to_string(), "global_value".to_string());

        let mut def = def(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "instruct",
            8010,
            "fp16",
            None,
        );
        def.env.insert("PER_SERVER_VAR".to_string(), "per_server_value".to_string());
        // Per-server should override global
        def.env.insert("VLLM_USE_FLASHINFER_SAMPLER".to_string(), "1".to_string());

        let script = launch_script(
            "~/llm-lp/.venv",
            &def,
            "",
            &crate::state::AdvancedSettings::default(),
            &global_env,
        );

        // Global default should be present
        assert!(script.contains("export GLOBAL_VAR='global_value'"));
        // Per-server var should be present
        assert!(script.contains("export PER_SERVER_VAR='per_server_value'"));
        // Per-server should override global default
        assert!(script.contains("export VLLM_USE_FLASHINFER_SAMPLER='1'"));
        assert!(!script.contains("export VLLM_USE_FLASHINFER_SAMPLER='0'"));
    }

    #[test]
    fn test_launch_script_invalid_env_name_skipped() {
        use std::collections::BTreeMap;
        let mut global_env = BTreeMap::new();
        global_env.insert("VALID_VAR".to_string(), "valid".to_string());
        global_env.insert("INVALID-VAR".to_string(), "invalid".to_string()); // hyphen not allowed

        let def = def(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "instruct",
            8010,
            "fp16",
            None,
        );

        let script = launch_script(
            "~/llm-lp/.venv",
            &def,
            "",
            &crate::state::AdvancedSettings::default(),
            &global_env,
        );

        // Valid var should be present
        assert!(script.contains("export VALID_VAR='valid'"));
        // Invalid var should be skipped (not crash)
        assert!(!script.contains("INVALID-VAR"));
    }

#[test]
    fn test_launch_script_env_export_lines_exact_format() {
        use std::collections::BTreeMap;
        let mut global_env = BTreeMap::new();
        global_env.insert("VLLM_USE_FLASHINFER_SAMPLER".to_string(), "0".to_string());
        global_env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let def = def(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "instruct",
            8010,
            "fp16",
            None,
        );

        let script = launch_script(
            "~/llm-lp/.venv",
            &def,
            "",
            &crate::state::AdvancedSettings::default(),
            &global_env,
        );

        // Check that export lines are properly formatted with shell_quote
        assert!(script.contains("export VLLM_USE_FLASHINFER_SAMPLER='0'"));
        assert!(script.contains("export TEST_VAR='test_value'"));
        
        // Check that the debug log lines are present
        assert!(script.contains("printf '[LocalLLmPanel] env %s=<%s>\\n' 'VLLM_USE_FLASHINFER_SAMPLER' '0'"));
        assert!(script.contains("printf '[LocalLLmPanel] env %s=<%s>\\n' 'TEST_VAR' 'test_value'"));
        
        // Check that the starting log line is present
        assert!(script.contains("echo '[LocalLLmPanel] Starting server"));
    }

    /// Integration test: launch a dummy command in WSL and verify env vars are passed correctly.
    /// Only runs when WSL is available and LLM_TEST_WSL=1 is set.
    #[test]
    #[ignore]
    fn test_wsl_env_var_passing() {
        use std::env;
        if env::var("LLM_TEST_WSL").is_err() {
            eprintln!("Skipping WSL integration test: set LLM_TEST_WSL=1 to run");
            return;
        }
        let distro = crate::wsl::detect_default_distro().expect("WSL distro not found");
        
        // Test values including hostile ones
        let test_cases = vec![
            ("SIMPLE", "0"),
            ("WITH_SPACE", "hello world"),
            ("WITH_QUOTE", "don't"),
            ("STRAY_QUOTE", "'0"),
            ("DOLLAR", "$HOME"),
            ("BACKTICK", "`id`"),
            ("BACKSLASH", r"C:\path"),
            ("UNICODE", "café测试"),
            ("HOSTILE", "'; rm -rf ~ #"),
            ("EMPTY", ""),
        ];

        for (name, value) in test_cases {
            // Build a script that just prints the env var
            let script = format!(
                "export {}={} && printf 'RESULT=%s\\n' \"${}\"",
                name,
                shell_quote(value),
                name
            );
            
            // Write script to file and execute
            let script_path = format!("~/.local/share/local-llm-panel/test-{}.sh", name);
            let mkdir_cmd = format!("mkdir -p ~/.local/share/local-llm-panel");
            let write_cmd = format!("cat > {}", crate::wsl::shell_quote_wsl(&script_path));
            let chmod_cmd = format!("chmod +x {}", crate::wsl::shell_quote_wsl(&script_path));
            let exec_cmd = format!("bash -l {}", crate::wsl::shell_quote_wsl(&script_path));
            let full_cmd = format!("{} && {} && {} && {}", mkdir_cmd, write_cmd, chmod_cmd, exec_cmd);

            let mut cmd = crate::wsl::wsl_command();
            cmd.env("WSL_UTF8", "1");
            cmd.args(["-d", &distro, "--exec", "bash", "-lc", &full_cmd]);
            cmd.stdin(std::process::Stdio::piped());
            
            let mut child = cmd.spawn().expect("failed to spawn wsl");
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                stdin.write_all(script.as_bytes()).ok();
                stdin.flush().ok();
            }
            let output = child.wait_with_output().expect("failed to wait");
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            // Extract the RESULT line
            let result_line = stdout.lines().find(|l| l.starts_with("RESULT="));
            let actual = result_line.map(|l| l["RESULT=".len()..].to_string()).unwrap_or_default();
            
            assert_eq!(actual, value, "Env var {}: expected {:?}, got {:?}", name, value, actual);
        }
    }
}
