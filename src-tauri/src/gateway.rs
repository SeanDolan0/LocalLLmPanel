//! Unified OpenAI-compatible reverse proxy gateway.
//!
//! Listens on a single local port (default 11434) and routes chat
//! completion requests to whichever running vLLM instance actually serves
//! the requested model. External tools (Cursor, Continue.dev, LibreChat,
//! any OpenAI-compatible client) can point at one stay-put endpoint
//! instead of juggling per-server ports.
//!
//! The HTTP layer is intentionally small (Tokio `TcpListener` + hand-rolled
//! HTTP/1.1 framing) to honor the project's "no unnecessary external
//! dependencies" constraint. Only read framing, two routes, and a streaming
//! reverse proxy are needed.

use crate::server::apply_vllm_auth;
use crate::state::{AppState, ServerDef, ServerStatus};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use tauri::Manager;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Default gateway port (the one Ollama-style tools expect).
pub const DEFAULT_GATEWAY_PORT: u16 = 11434;
const MAX_HEAD_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, PartialEq)]
pub struct RequestHead {
    pub method: String,
    pub target: String,
    pub headers: HashMap<String, String>,
    pub content_length: usize,
    pub chunked: bool,
}

pub fn check_gateway_auth(
    headers: &HashMap<String, String>,
    global_key: Option<&str>,
) -> Result<(), (u16, String)> {
    let Some(raw_key) = global_key.map(str::trim).filter(|key| !key.is_empty()) else {
        return Err((
            503,
            "Gateway API key is not configured; set one in Settings before using /v1 routes"
                .to_string(),
        ));
    };
    let decrypted_storage;
    let expected_key = if raw_key.starts_with("dpapi:") {
        decrypted_storage =
            crate::security::decrypt_token(raw_key).unwrap_or_else(|_| raw_key.to_string());
        decrypted_storage.trim()
    } else {
        raw_key
    };
    if let Some(h) = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"))
    {
        if h.starts_with("Bearer ") {
            let provided = h["Bearer ".len()..].trim();
            if !provided.is_empty() && constant_time_eq(provided, expected_key) {
                return Ok(());
            } else {
                return Err((401, "Invalid API key".to_string()));
            }
        } else {
            return Err((
                401,
                "Authorization header must use Bearer scheme".to_string(),
            ));
        }
    }
    if let Some(key) = headers.get("api-key").or_else(|| headers.get("x-api-key")) {
        if constant_time_eq(key.trim(), expected_key) {
            return Ok(());
        } else {
            return Err((401, "Invalid API key".to_string()));
        }
    }
    Err((401, "Missing Authorization header".to_string()))
}

pub fn check_gateway_auth_api(
    headers: &HashMap<String, String>,
    global_key: Option<&str>,
) -> Result<(), ApiError> {
    match check_gateway_auth(headers, global_key) {
        Ok(()) => Ok(()),
        Err((503, msg)) => Err(ApiError::service_unavailable(msg)),
        Err((401, msg)) => Err(ApiError::unauthorized(msg)),
        Err((status, msg)) => Err(ApiError::new(status, msg, "auth_error")),
    }
}

pub fn parse_head_result(bytes: &[u8]) -> Result<RequestHead, ApiError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        ApiError::bad_request(
            "invalid utf-8 in request headers",
            None,
            Some("malformed_headers".into()),
        )
    })?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next().ok_or_else(|| {
        ApiError::bad_request("empty request", None, Some("malformed_request_line".into()))
    })?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| {
            ApiError::bad_request(
                "missing method in request line",
                None,
                Some("malformed_request_line".into()),
            )
        })?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| {
            ApiError::bad_request(
                "missing target in request line",
                None,
                Some("malformed_request_line".into()),
            )
        })?
        .to_string();
    if method.is_empty() || target.is_empty() {
        return Err(ApiError::bad_request(
            "empty method or target",
            None,
            Some("malformed_request_line".into()),
        ));
    }
    let mut headers = HashMap::new();
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let mut it = line.splitn(2, ':');
        let name = match it.next() {
            Some(n) => n.trim().to_ascii_lowercase(),
            None => continue,
        };
        let value = it.next().unwrap_or("").trim().to_string();
        if name == "content-length" {
            let parsed: usize = value.parse().map_err(|_| {
                ApiError::bad_request(
                    "invalid Content-Length header",
                    Some("content-length".into()),
                    Some("invalid_header".into()),
                )
            })?;
            if content_length.replace(parsed).is_some() {
                return Err(ApiError::bad_request(
                    "duplicate Content-Length header",
                    Some("content-length".into()),
                    Some("duplicate_header".into()),
                ));
            }
        } else if name == "transfer-encoding" {
            let encodings: Vec<_> = value.split(',').map(str::trim).collect();
            if encodings.len() != 1 || !encodings[0].eq_ignore_ascii_case("chunked") {
                return Err(ApiError::bad_request(
                    "unsupported Transfer-Encoding (only chunked supported)",
                    Some("transfer-encoding".into()),
                    Some("unsupported_encoding".into()),
                ));
            }
            chunked = true;
        }
        headers.insert(name, value);
    }
    if chunked && content_length.is_some() {
        return Err(ApiError::bad_request(
            "conflicting Transfer-Encoding and Content-Length headers",
            None,
            Some("conflicting_framing".into()),
        ));
    }
    Ok(RequestHead {
        method,
        target,
        headers,
        content_length: content_length.unwrap_or(0),
        chunked,
    })
}

pub fn parse_head(bytes: &[u8]) -> Option<RequestHead> {
    parse_head_result(bytes).ok()
}

/// All live servers of `task` status that are currently `Running`.
fn running_servers<'a>(
    servers: &'a BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<&'a crate::state::LiveServer> {
    servers
        .values()
        .filter(|ls| ls.status == ServerStatus::Running && ls.def.task == task)
        .collect()
}

/// Route a requested model name to the port of the running server that
/// serves it with the given task. Exact match on the served model name or
/// HF id wins first, then a fuzzy suffix match against both (so
/// `Qwen2.5-7B-Instruct` finds `Qwen/Qwen2.5-7B-Instruct`).
/// Strips `:latest` tags, recognizes wildcard aliases (`default`, `local`, `*`),
/// and falls back to the sole running server if only one is active.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelRouteError {
    EmptyModel,
    NotFound {
        requested: String,
        running: Vec<String>,
    },
    Ambiguous {
        requested: String,
        candidates: Vec<String>,
    },
}

impl std::fmt::Display for ModelRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "model name cannot be empty"),
            Self::NotFound { requested, running } => {
                write!(
                    f,
                    "no running server for model '{requested}'. Running: [{}]",
                    running.join(", ")
                )
            }
            Self::Ambiguous {
                requested,
                candidates,
            } => {
                write!(
                    f,
                    "model '{requested}' is ambiguous; matches multiple running models: [{}]. Please specify the exact model name.",
                    candidates.join(", ")
                )
            }
        }
    }
}

pub fn route_server_for(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
    task: &str,
) -> Result<u16, ModelRouteError> {
    let running: Vec<&crate::state::LiveServer> = servers
        .values()
        .filter(|ls| {
            (ls.status == ServerStatus::Running || ls.status == ServerStatus::Starting)
                && ls.def.task == task
        })
        .collect();
    let running_names: Vec<String> = running
        .iter()
        .map(|ls| {
            ls.def.served_model_name.clone().unwrap_or_else(|| {
                if !ls.def.model_id.is_empty() {
                    ls.def.model_id.clone()
                } else {
                    ls.def.name.clone()
                }
            })
        })
        .collect();
    let m = model.trim();
    let m = m.strip_suffix(":latest").unwrap_or(m);
    if m.is_empty() {
        return Err(ModelRouteError::EmptyModel);
    }
    if running.is_empty() {
        return Err(ModelRouteError::NotFound {
            requested: m.to_string(),
            running: running_names,
        });
    }

    // If wildcard/default aliases, and exactly one server is running, route to it.
    let is_wildcard = m.eq_ignore_ascii_case("default")
        || m.eq_ignore_ascii_case("local")
        || m.eq_ignore_ascii_case("auto")
        || m == "*";
    if is_wildcard {
        return if running.len() == 1 {
            Ok(running[0].def.port)
        } else {
            Err(ModelRouteError::Ambiguous {
                requested: m.to_string(),
                candidates: running_names,
            })
        };
    }

    // Pass 1: Exact matches. Exact match always wins over fuzzy.
    let mut exact_matches = Vec::new();
    for ls in &running {
        if ls.def.effective_model_name().eq_ignore_ascii_case(m)
            || ls.def.model_id.eq_ignore_ascii_case(m)
            || ls.def.name.eq_ignore_ascii_case(m)
            || ls
                .def
                .served_model_name
                .as_deref()
                .map(|s| s.eq_ignore_ascii_case(m))
                .unwrap_or(false)
        {
            exact_matches.push(ls);
        }
    }
    if !exact_matches.is_empty() {
        return Ok(exact_matches[0].def.port);
    }

    // Pass 2: Fuzzy matches (suffix, prefix, substring).
    let lower = m.to_lowercase();
    let mut fuzzy_matches = Vec::new();
    for ls in &running {
        let hf = ls.def.model_id.to_lowercase();
        let name = ls.def.effective_model_name().to_lowercase();
        let sname = ls.def.name.to_lowercase();
        let matched = hf.ends_with(&lower)
            || name.ends_with(&lower)
            || sname.ends_with(&lower)
            || lower.ends_with(&hf)
            || lower.ends_with(&name)
            || lower.ends_with(&sname)
            || hf.starts_with(&lower)
            || name.starts_with(&lower)
            || sname.starts_with(&lower)
            || lower.starts_with(&hf)
            || lower.starts_with(&name)
            || lower.starts_with(&sname)
            || (lower.len() >= 3
                && (hf.contains(&lower) || name.contains(&lower) || sname.contains(&lower)));
        if matched {
            fuzzy_matches.push(ls);
        }
    }

    if fuzzy_matches.len() == 1 {
        return Ok(fuzzy_matches[0].def.port);
    } else if fuzzy_matches.len() > 1 {
        let mut candidates = Vec::new();
        for ls in fuzzy_matches {
            let label =
                ls.def
                    .served_model_name
                    .as_deref()
                    .unwrap_or(if !ls.def.model_id.is_empty() {
                        &ls.def.model_id
                    } else {
                        &ls.def.name
                    });
            if !candidates.contains(&label.to_string()) {
                candidates.push(label.to_string());
            }
        }
        candidates.sort();
        return Err(ModelRouteError::Ambiguous {
            requested: m.to_string(),
            candidates,
        });
    }

    Err(ModelRouteError::NotFound {
        requested: m.to_string(),
        running: running_names,
    })
}

pub fn find_server_port_for(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
    task: &str,
) -> Option<u16> {
    route_server_for(servers, model, task).ok()
}

fn configured_context_tokens(def: &ServerDef) -> Option<usize> {
    match def.backend.as_str() {
        "vllm" => def.max_model_len.filter(|value| *value > 0).or_else(|| {
            flag_value(&def.extra_args, "--max-model-len")
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
        }),
        "llamacpp" => def.ctx_size.filter(|value| *value > 0).or_else(|| {
            flag_value(&def.extra_args, "-c")
                .or_else(|| flag_value(&def.extra_args, "--ctx-size"))
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
        }),
        _ => None,
    }
}

/// Effective CLI arguments for a running backend. llama.cpp keeps its flags on
/// the server definition; vLLM accepts a single global extra-args string that
/// the caller parses once for the whole `/v1/models` response.
fn effective_launch_args<'a>(def: &'a ServerDef, vllm_args: &'a [String]) -> &'a [String] {
    if def.backend == "llamacpp" {
        &def.extra_args
    } else {
        vllm_args
    }
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter()
        .any(|arg| arg == flag || arg.starts_with(&format!("{flag}=")))
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    for (index, arg) in args.iter().enumerate() {
        if arg == flag {
            return args.get(index + 1).cloned();
        }
        if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            return Some(value.to_string());
        }
    }
    None
}

fn template_is_toggle_only(template: Option<&str>, has_ladder: bool) -> bool {
    if has_ladder {
        return false;
    }
    if let Some(t) = template {
        let lower = t.to_ascii_lowercase();
        if lower.contains("enable_thinking")
            && !lower.contains("reasoning_effort")
            && !lower.contains("thinking_effort")
        {
            return true;
        }
    }
    false
}

fn model_is_deepseek(model_id: &str, model_name: &str) -> bool {
    let lower = format!("{} {}", model_id, model_name).to_ascii_lowercase();
    lower.contains("deepseek")
}

fn resolve_model_context_sync(def: &ServerDef, launch_args: &[String]) -> (usize, &'static str) {
    if let Some(ctx) = configured_context_tokens(def) {
        return (ctx, "configured");
    }
    if def.backend == "llamacpp" {
        if let Some(ctx) = flag_value(launch_args, "-c")
            .or_else(|| flag_value(launch_args, "--ctx-size"))
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            return (ctx, "configured");
        }
        let raw_model = def.model_path.as_deref().unwrap_or(&def.model_id);
        let resolved_path = def
            .model_path
            .as_deref()
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_file())
            .or_else(|| crate::server::resolve_gguf_model_path(raw_model, "", ""));
        if let Some(path) = resolved_path {
            if let Ok(meta) = crate::gguf::metadata_from_path(&path) {
                if let Some(ctx) = meta.context_length.filter(|v| *v > 0) {
                    return (ctx, "gguf_metadata");
                }
            }
        }
    } else if def.backend == "vllm" {
        if let Some(ctx) = flag_value(launch_args, "--max-model-len")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            return (ctx, "configured");
        }
        let mut cfg_val: Option<serde_json::Value> = None;
        if let Some(path_str) = &def.model_path {
            let path = std::path::Path::new(path_str);
            let cfg_path = if path.is_dir() {
                path.join("config.json")
            } else {
                path.with_file_name("config.json")
            };
            if cfg_path.is_file() {
                if let Ok(content) = std::fs::read_to_string(&cfg_path) {
                    cfg_val = serde_json::from_str::<serde_json::Value>(&content).ok();
                }
            }
        }
        if cfg_val.is_none() {
            if let Some(home) = dirs::home_dir() {
                let folder = format!("models--{}", def.model_id.replace('/', "--"));
                let hub_path = home
                    .join(".cache")
                    .join("huggingface")
                    .join("hub")
                    .join(folder)
                    .join("snapshots");
                if hub_path.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&hub_path) {
                        for entry in entries.flatten() {
                            let p = entry.path().join("config.json");
                            if p.is_file() {
                                if let Ok(content) = std::fs::read_to_string(&p) {
                                    if let Ok(val) =
                                        serde_json::from_str::<serde_json::Value>(&content)
                                    {
                                        cfg_val = Some(val);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(val) = cfg_val {
            let (ctx, _) = crate::estimate::parse_context(&val);
            if ctx > 0 {
                return (ctx, "model_config");
            }
        }
    }

    if let Some(ctx) = crate::estimate::family_fallback(&def.model_id) {
        return (ctx, "architecture");
    }
    let db = crate::llmfit_adapter::get_model_database();
    if let Some(m) = db
        .find_model(&def.model_id)
        .into_iter()
        .find(|m| m.context_length > 0)
    {
        return (m.context_length as usize, "architecture");
    }
    if def.task == "embed" {
        (512, "default")
    } else {
        (32_768, "default")
    }
}

/// How a backend/model exposes its thinking mode.
/// `Effort` is the standard OpenAI/OpenRouter `reasoning_effort` scale
/// (`none`/`low`/`medium`/`high`), accepted across all reasoning model types.
/// `AlwaysOn` models (DeepSeek-R1 / QwQ) cannot be turned off (`mandatory: true`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningMode {
    None,
    Effort,
    AlwaysOn,
}

impl ReasoningMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Effort => "effort",
            Self::AlwaysOn => "always_on",
        }
    }

    fn levels(self) -> &'static [&'static str] {
        match self {
            Self::None => &[],
            Self::Effort => &["none", "low", "medium", "high"],
            Self::AlwaysOn => &["low", "medium", "high"],
        }
    }

    fn default_level(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Effort | Self::AlwaysOn => Some("medium"),
        }
    }
}

/// Capabilities inferred from the model family and the actual launch flags.
///
/// Tool calling and reasoning output are backend features, not just model
/// features: vLLM only honors `tool_choice: "auto"` with
/// `--enable-auto-tool-choice` plus a `--tool-call-parser`, and only returns
/// `reasoning_content` with a `--reasoning-parser`. Reporting the model as
/// capable when the process is not configured would make harnesses fail at
/// request time, so `reasoning` (model can think) and `reasoning_output`
/// (backend exposes it) are separate fields.
#[derive(Debug, Clone, Copy)]
struct ModelTraits {
    tools: bool,
    vision: bool,
    reasoning: bool,
    reasoning_output: bool,
    reasoning_mode: ReasoningMode,
    reasoning_enabled: bool,
    reasoning_source: &'static str,
}

fn looks_like_reasoning_model(name: &str) -> bool {
    const MARKERS: &[&str] = &[
        "deepseek-r1",
        "deepseek-reasoner",
        "qwq",
        "qwen3",
        "thinking",
        "reasoning",
        "gpt-oss",
        "glm-4.5",
        "glm-4.6",
        "glm-5",
        "magistral",
        "kimi-k2",
        "minimax-m2",
        "nemotron",
        "granite-4",
        "exaone-deep",
        "seed-oss",
        "olmo-3-think",
        "ernie-4.5",
        "bonsai",
        "spark",
    ];
    MARKERS.iter().any(|marker| name.contains(marker))
}

fn reasoning_is_always_on(name: &str) -> bool {
    // These models always emit a reasoning trace; there is no off switch.
    [
        "deepseek-r1",
        "deepseek-reasoner",
        "qwq",
        "phi-4-reasoning",
        "olmo-3-think",
        "exaone-deep",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}

fn looks_like_vision_model(name: &str) -> bool {
    const MARKERS: &[&str] = &[
        "vision",
        "-vl-",
        "llava",
        "pixtral",
        "internvl",
        "minicpm-v",
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "gemma-3",
        "gemma-4",
        "idefics",
        "llama-4",
        "molmo",
        "ovis",
        "glm-4v",
        "glmv",
        "phi-4-multimodal",
        "mistral-small-3",
        "step-1o",
        "aria",
        "paligemma",
        "pali-gemma",
        "smolvlm",
        "cogvlm",
        "chameleon",
        "deepseek-vl",
        "omni",
    ];
    MARKERS.iter().any(|marker| name.contains(marker))
}

pub fn template_has_reasoning(template: &str) -> bool {
    let lower = template.to_ascii_lowercase();
    lower.contains("<think>")
        || lower.contains("</think>")
        || lower.contains("<thought>")
        || lower.contains("</thought>")
        || lower.contains("<|thought|>")
        || lower.contains("<|endofthought|>")
        || lower.contains("[think]")
        || lower.contains("[/think]")
        || lower.contains("[thought]")
        || lower.contains("[/thought]")
        || lower.contains("enable_thinking")
        || lower.contains("reasoning_content")
        || lower.contains("reasoning_effort")
        || lower.contains("thinking_effort")
        || lower.contains("reasoning_budget")
        || lower.contains("thinking_token_budget")
}

fn def_is_vision(def: &ServerDef) -> bool {
    if crate::model_intelligence::infer_model_family(&def.model_id, None, &[]).is_multimodal {
        return true;
    }
    if let Some(path_str) = &def.model_path {
        let path = std::path::Path::new(path_str);
        let cfg_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.with_file_name("config.json")
        };
        if cfg_path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&cfg_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if val.get("vision_config").is_some()
                        || val.get("image_token_id").is_some()
                        || val.get("image_token_index").is_some()
                        || val.get("visual").is_some()
                    {
                        return true;
                    }
                    if let Some(archs) = val.get("architectures").and_then(|a| a.as_array()) {
                        for arch in archs {
                            if let Some(s) = arch.as_str() {
                                if s.contains("VL")
                                    || s.contains("Vision")
                                    || s.contains("MultiModal")
                                    || s.contains("Llava")
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

fn model_traits(
    def: &ServerDef,
    launch_args: &[String],
    info: &crate::state::TemplateEffortInfo,
) -> ModelTraits {
    let name = format!("{} {}", def.model_id, def.effective_model_name()).to_ascii_lowercase();
    let is_instruct = def.task == "instruct";

    let family_reasoning =
        info.is_reasoning || info.ladder.is_some() || looks_like_reasoning_model(&name);
    let family_always_on = if info.is_reasoning {
        info.is_always_on
    } else {
        reasoning_is_always_on(&name)
    };

    let (reasoning, reasoning_output, configured, reasoning_enabled) = if def.backend == "llamacpp"
    {
        // llama.cpp parses think tags by default (`--reasoning-format auto`)
        // when the Jinja template emits them; `--reasoning off` disables
        // thinking entirely.
        let configured =
            has_flag(launch_args, "--reasoning-format") || has_flag(launch_args, "--reasoning");
        let output_disabled =
            flag_value(launch_args, "--reasoning-format").as_deref() == Some("none");
        let reasoning = is_instruct && (family_reasoning || configured);
        let enabled = reasoning && flag_value(launch_args, "--reasoning").as_deref() != Some("off");
        (
            reasoning,
            reasoning && !output_disabled,
            configured,
            enabled,
        )
    } else {
        // vLLM exposes reasoning_content only when a parser is configured.
        let parser = has_flag(launch_args, "--reasoning-parser");
        let reasoning = is_instruct && (family_reasoning || parser);
        (reasoning, parser, parser, reasoning)
    };

    let reasoning_mode = if !reasoning {
        ReasoningMode::None
    } else if family_always_on {
        ReasoningMode::AlwaysOn
    } else {
        ReasoningMode::Effort
    };

    let reasoning_source = if configured {
        "launch_args"
    } else if family_reasoning {
        "model_family"
    } else {
        "none"
    };

    let tools = is_instruct
        && match def.backend.as_str() {
            // llama.cpp needs --jinja for OpenAI-style tool calls.
            "llamacpp" => def.jinja,
            // vLLM rejects tool_choice "auto" unless both flags are present.
            "vllm" => {
                has_flag(launch_args, "--enable-auto-tool-choice")
                    && has_flag(launch_args, "--tool-call-parser")
            }
            _ => false,
        };

    let vision = is_instruct
        && (def
            .mmproj_path
            .as_deref()
            .is_some_and(|path| !path.trim().is_empty())
            || def_is_vision(def)
            || looks_like_vision_model(&name));

    ModelTraits {
        tools,
        vision,
        reasoning,
        reasoning_output,
        reasoning_mode,
        reasoning_enabled,
        reasoning_source,
    }
}

/// Actionable notes for fields that are false because of launch configuration
/// rather than model support. Harnesses ignore this extension; humans reading
/// `GET /v1/models` use it to fix tool/reasoning setup in one step.
fn model_hints(def: &ServerDef, traits: &ModelTraits, launch_args: &[String]) -> Vec<String> {
    let mut hints = Vec::new();
    if def.task != "instruct" {
        return hints;
    }
    if def.backend == "vllm" {
        if !traits.tools {
            hints.push(
                "Tool calling is disabled: launch vLLM with --enable-auto-tool-choice and \
                 --tool-call-parser <parser> (Advanced → extra vLLM args)."
                    .to_string(),
            );
        }
        if traits.reasoning && !traits.reasoning_output {
            hints.push(
                "This model reasons, but vLLM will not return reasoning_content without \
                 --reasoning-parser <parser> (Advanced → extra vLLM args)."
                    .to_string(),
            );
        }
        if flag_value(launch_args, "--override-generation-config")
            .is_some_and(|value| value.contains("\"max_new_tokens\""))
        {
            hints.push(
                "vLLM's --override-generation-config max_new_tokens caps every request; harness \
                 max_tokens values above that cap are clamped. Remove max_new_tokens from extra \
                 vLLM args to let requests choose their output limit."
                    .to_string(),
            );
        }
    } else if def.backend == "llamacpp" && !def.jinja {
        hints.push("Tool calling is disabled: llama.cpp needs --jinja.".to_string());
    }
    hints
}

#[allow(dead_code)]
fn model_row(
    def: &ServerDef,
    context: Option<usize>,
    context_source: &str,
    launch_args: &[String],
) -> serde_json::Value {
    model_row_with_template(
        def,
        context,
        context_source,
        launch_args,
        &crate::state::TemplateEffortInfo::default(),
    )
}

#[cfg(test)]
fn model_row_with_ladder(
    def: &ServerDef,
    context: Option<usize>,
    context_source: &str,
    launch_args: &[String],
    effort_ladder: Option<&[String]>,
    template_reasoning: bool,
) -> serde_json::Value {
    let ladder = effort_ladder.filter(|l| l.len() >= 2).map(|l| l.to_vec());
    let default_effort = ladder.as_deref().and_then(|l| {
        if l.iter().any(|level| level.eq_ignore_ascii_case("medium")) {
            Some("medium".to_string())
        } else {
            clamp_effort_to_ladder("medium", l)
        }
    });
    let info = crate::state::TemplateEffortInfo {
        ladder,
        default_effort,
        is_reasoning: template_reasoning,
        is_toggle_only: false,
        is_always_on: false,
        template: None,
    };
    model_row_with_template(def, context, context_source, launch_args, &info)
}

fn model_row_with_template(
    def: &ServerDef,
    context: Option<usize>,
    context_source: &str,
    launch_args: &[String],
    info: &crate::state::TemplateEffortInfo,
) -> serde_json::Value {
    let effort_ladder = info.ladder.as_deref();

    let (context, context_source) = match context.filter(|value| *value > 0) {
        Some(ctx) => (Some(ctx), context_source),
        None => {
            let (ctx, src) = resolve_model_context_sync(def, launch_args);
            (Some(ctx), src)
        }
    };
    let is_instruct = def.task == "instruct";
    let is_embed = def.task == "embed";

    let output_ceiling = flag_value(launch_args, "-n")
        .or_else(|| flag_value(launch_args, "--n-predict"))
        .or_else(|| flag_value(launch_args, "--max-tokens"))
        .or_else(|| flag_value(launch_args, "--max-new-tokens"))
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);

    let (output_tokens, output_tokens_source) = if !is_instruct {
        (None, "none")
    } else if let Some(ceiling) = output_ceiling {
        (Some(ceiling), "launch_args")
    } else if let Some(ctx) = context {
        (Some(ctx), "context_window")
    } else {
        (Some(8192), "default_ceiling")
    };

    let traits = model_traits(def, launch_args, info);
    let hints = model_hints(def, &traits, launch_args);
    let model_name = def.effective_model_name();

    let model_type = if is_embed {
        "embeddings"
    } else if traits.vision {
        "vlm"
    } else {
        "llm"
    };
    let modality = if is_embed {
        "text->embedding"
    } else if traits.vision {
        "text+image->text"
    } else {
        "text->text"
    };
    let input_modalities = if traits.vision {
        vec!["text", "image"]
    } else {
        vec!["text"]
    };
    let output_modalities = if is_embed {
        vec!["embedding"]
    } else {
        vec!["text"]
    };
    let publisher = def
        .model_id
        .split_once('/')
        .map(|(owner, _)| owner.to_string());
    // LM Studio-compatible compatibility marker; vLLM consumes safetensors
    // checkpoints while llama.cpp consumes GGUF files.
    let compatibility_type = if def.backend == "llamacpp" {
        "gguf"
    } else {
        "safetensors"
    };
    let supports_template_controls = def.backend != "llamacpp" || def.jinja;
    let is_toggle_only = traits.reasoning && info.is_toggle_only;
    let control = if !supports_template_controls || traits.reasoning_mode == ReasoningMode::None {
        "none"
    } else if is_toggle_only {
        "enable_thinking"
    } else {
        "reasoning_effort"
    };
    // vLLM maps reasoning_effort onto the model's thinking toggle and
    // reasoning parser; llama.cpp forwards it to the Jinja template and
    // parses reasoning tags. The gateway canonicalizes reasoning_effort
    // across all reasoning models and formats.
    let supports_effort_parameter = traits.reasoning_mode != ReasoningMode::None
        && supports_template_controls;
    let supports_thinking_toggle =
        traits.reasoning_mode == ReasoningMode::Effort && supports_template_controls;
    let budget_parameter = if traits.reasoning_mode == ReasoningMode::None {
        None
    } else if def.backend == "vllm" {
        Some("thinking_token_budget")
    } else {
        Some("reasoning_budget_tokens")
    };
    let mut reasoning_controls: Vec<&str> = Vec::new();
    if traits.reasoning_mode != ReasoningMode::None {
        if supports_effort_parameter {
            reasoning_controls.push("reasoning_effort");
        }
        if supports_thinking_toggle {
            reasoning_controls.push("chat_template_kwargs.enable_thinking");
        }
        if let Some(budget_parameter) = budget_parameter {
            reasoning_controls.push(budget_parameter);
        }
        if def.backend == "llamacpp" {
            reasoning_controls.push("reasoning_format");
        }
    }

    let mut supported_parameters: Vec<&str> = if is_embed {
        Vec::new()
    } else {
        vec![
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_format",
            "logprobs",
        ]
    };
    if traits.tools {
        supported_parameters.extend(["tools", "tool_choice", "parallel_tool_calls"]);
    }
    if traits.reasoning_mode != ReasoningMode::None {
        if !supported_parameters.contains(&"reasoning") {
            supported_parameters.push("reasoning");
        }
        if !supported_parameters.contains(&"include_reasoning") {
            supported_parameters.push("include_reasoning");
        }
        if supports_thinking_toggle {
            supported_parameters.push("chat_template_kwargs");
        }
        if supports_effort_parameter {
            supported_parameters.push("reasoning_effort");
        }
        if let Some(budget_parameter) = budget_parameter {
            supported_parameters.push(budget_parameter);
        }
    }

    let supported_efforts: Vec<&str> = match (traits.reasoning_mode, effort_ladder) {
        (ReasoningMode::None, _) => Vec::new(),
        _ if is_toggle_only => vec!["none", "high"],
        (ReasoningMode::Effort, Some(ladder)) => ladder.iter().map(String::as_str).collect(),
        (ReasoningMode::AlwaysOn, Some(ladder)) => ladder.iter().map(String::as_str).collect(),
        (ReasoningMode::Effort, None) => {
            vec!["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        }
        (ReasoningMode::AlwaysOn, None) => vec!["low", "medium", "high"],
    };
    let levels: Vec<&str> = match (traits.reasoning_mode, effort_ladder) {
        (ReasoningMode::None, _) => Vec::new(),
        _ if is_toggle_only => vec!["high"],
        (ReasoningMode::Effort, Some(ladder)) => ladder.iter().map(String::as_str).collect(),
        (ReasoningMode::AlwaysOn, Some(ladder)) => ladder.iter().map(String::as_str).collect(),
        (mode, None) => mode.levels().to_vec(),
    };
    let default_level: Option<String> = if is_toggle_only {
        Some("high".to_string())
    } else {
        match traits.reasoning_mode {
            ReasoningMode::None => None,
            mode => {
                if let Some(def_eff) = &info.default_effort {
                    Some(def_eff.clone())
                } else if let Some(ladder) = effort_ladder {
                    clamp_effort_to_ladder("medium", ladder)
                } else {
                    mode.default_level().map(str::to_string)
                }
            }
        }
    };
    let level_map_val = |level: &str| -> serde_json::Value {
        if traits.reasoning_mode == ReasoningMode::None {
            return serde_json::Value::Null;
        }
        if is_toggle_only {
            if level == "high" {
                return serde_json::json!("high");
            } else {
                return serde_json::Value::Null;
            }
        }
        let is_supported = match effort_ladder {
            Some(ladder) => ladder.iter().any(|l| l.eq_ignore_ascii_case(level)),
            None => true,
        };
        if is_supported {
            serde_json::json!(level)
        } else {
            serde_json::Value::Null
        }
    };
    let thinking_level_map = if traits.reasoning_mode == ReasoningMode::None {
        serde_json::json!({
            "off": "off",
            "none": "none",
            "minimal": serde_json::Value::Null,
            "low": serde_json::Value::Null,
            "medium": serde_json::Value::Null,
            "high": serde_json::Value::Null,
            "xhigh": serde_json::Value::Null,
            "max": serde_json::Value::Null,
        })
    } else {
        let off_val = if traits.reasoning_mode == ReasoningMode::AlwaysOn {
            serde_json::Value::Null
        } else {
            serde_json::json!("none")
        };
        serde_json::json!({
            "off": off_val.clone(),
            "none": off_val,
            "minimal": level_map_val("minimal"),
            "low": level_map_val("low"),
            "medium": level_map_val("medium"),
            "high": level_map_val("high"),
            "xhigh": level_map_val("xhigh"),
            "max": level_map_val("max"),
        })
    };

    let reasoning = serde_json::json!({
        "supported": traits.reasoning,
        // Whether the running backend actually returns reasoning_content.
        "output": traits.reasoning_output,
        "enabled_by_default": traits.reasoning_enabled,
        "default_enabled": traits.reasoning_enabled,
        "mandatory": traits.reasoning_mode == ReasoningMode::AlwaysOn,
        // Both backends let a request-level enable_thinking override the
        // launch default.  llama.cpp additionally lets a request-level
        // reasoning_format override --reasoning-format none.
        "enabled_overridable": supports_thinking_toggle,
        "output_overridable": def.backend == "llamacpp"
            && traits.reasoning_mode != ReasoningMode::None,
        "mode": traits.reasoning_mode.as_str(),
        "levels": levels,
        "default_level": default_level,
        "supported_efforts": supported_efforts,
        "default_effort": default_level.clone(),
        "thinking_level_map": thinking_level_map,
        "thinkingLevelMap": thinking_level_map.clone(),
        "control": control,
        "controls": reasoning_controls,
        // Request values are made explicit by the gateway and win over panel
        // launch flags / server defaults.  Panel settings only provide the
        // default for requests that do not control thinking themselves.
        "request_precedence": "harness",
        "parameters": {
            "reasoning_effort": supports_effort_parameter,
            "chat_template_kwargs": {
                "enable_thinking": supports_thinking_toggle,
            },
            "thinking_token_budget": budget_parameter == Some("thinking_token_budget"),
            "reasoning_budget_tokens": budget_parameter == Some("reasoning_budget_tokens"),
            "reasoning_format": def.backend == "llamacpp"
                && traits.reasoning_mode != ReasoningMode::None,
        },
        "source": traits.reasoning_source,
    });

    let mut row = serde_json::json!({
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "local-llm-panel",
        "name": def.name.clone(),
        "display_name": def.name.clone(),
        "root": def.model_id.clone(),
        "parent": serde_json::Value::Null,
        // vLLM uses max_model_len and llama.cpp calls the same value n_ctx;
        // the other names make the effective context discoverable to clients
        // that use OpenAI/LM Studio-style model metadata instead.
        "max_model_len": context,
        "context_length": context,
        "max_context_length": context,
        "max_input_tokens": context,
        "context_window": context,
        "n_ctx": context,
        "context_source": context_source,
        // Hard upper bound on generation: the prompt and completion share the
        // context window. Embedding models do not generate tokens.
        "max_output_tokens": if is_instruct { output_tokens } else { None },
        "max_completion_tokens": if is_instruct { output_tokens } else { None },
        "max_output_tokens_source": if is_instruct && output_tokens.is_some() {
            Some(output_tokens_source)
        } else {
            None
        },
        "architecture": {
            "modality": modality,
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "instruct_type": serde_json::Value::Null,
        },
        "type": model_type,
        "state": "loaded",
        "status": "loaded",
        "publisher": publisher,
        "compatibility_type": compatibility_type,
        "backend": def.backend.clone(),
        "quantization": def.quant.clone(),
        "task": def.task.clone(),
        "capabilities": {
            "chat_completions": is_instruct,
            "completions": is_instruct,
            "embeddings": is_embed,
            "streaming": is_instruct,
            "tools": traits.tools,
            "tool_calling": traits.tools,
            "parallel_tool_calls": traits.tools,
            "vision": traits.vision,
            "reasoning": traits.reasoning,
            "reasoning_output": traits.reasoning_output,
            "thinking": traits.reasoning,
            "system_role": is_instruct,
            "json_mode": is_instruct,
        },
        "supports_tools": traits.tools,
        "supports_tool_calls": traits.tools,
        "supports_vision": traits.vision,
        "supports_reasoning": traits.reasoning,
        "supports_thinking": traits.reasoning,
        "supports_embeddings": is_embed,
        // Boolean so OpenAI-compatible harnesses (including Pi's compat
        // extension) can discover that the model reasons. Detailed levels and
        // request controls live in `reasoning_config` (and its `thinking` alias).
        "reasoning": traits.reasoning,
        "reasoning_config": reasoning,
        // Alias for harnesses that look for an explicit "thinking" object.
        "thinking": reasoning,
        "supported_parameters": supported_parameters,
        "port": def.port,
    });
    if let Some(params) = def.params_b.filter(|value| *value > 0.0) {
        row["parameters_billions"] = serde_json::json!(params);
        row["parameter_count"] = serde_json::json!((params * 1_000_000_000.0).round() as u64);
    }
    // Aliases are added after the macro: the json! expansion is already close
    // to the default recursion limit, and Pi's compat extension reads
    // `max_tokens` (not max_output_tokens/max_completion_tokens) from the
    // catalog, falling back to 4096 when absent. It also discovers image
    // support only from a top-level `input` array, not
    // architecture.input_modalities.
    row["max_tokens"] = serde_json::json!(if is_instruct { output_tokens } else { None });
    row["input"] = serde_json::json!(input_modalities);
    row["permission"] = serde_json::json!([]);

    let mut advertised_reasoning_levels: Vec<&str> = Vec::new();
    if traits.reasoning_mode == ReasoningMode::Effort && !levels.iter().any(|l| *l == "none") {
        advertised_reasoning_levels.push("none");
    }
    advertised_reasoning_levels.extend(levels.iter().copied());

    let default_reasoning_level_val = if traits.reasoning_mode != ReasoningMode::None {
        default_level.clone()
    } else {
        None
    };

    let compat = serde_json::json!({
        "supportsReasoningEffort": supports_effort_parameter,
        "supportsThinking": traits.reasoning,
        "reasoningDisableMode": if traits.reasoning_mode == ReasoningMode::Effort {
            Some("none-effort")
        } else {
            None
        },
    });

    let metadata = serde_json::json!({
        "supported_reasoning_levels": advertised_reasoning_levels,
        "default_reasoning_level": default_reasoning_level_val,
        "reasoning": traits.reasoning,
    });

    row["supported_reasoning_levels"] = serde_json::json!(advertised_reasoning_levels);
    row["default_reasoning_level"] = serde_json::json!(default_reasoning_level_val);
    row["metadata"] = metadata;
    row["compat"] = compat;

    row["supportsReasoning"] = serde_json::json!(traits.reasoning);
    row["supportsReasoningEffort"] = serde_json::json!(advertised_reasoning_levels);
    row["supportsReasoningBinary"] = serde_json::json!(is_toggle_only);

    if traits.reasoning {
        row["reasoning"] = serde_json::json!({
            "mandatory": traits.reasoning_mode == ReasoningMode::AlwaysOn,
            "default_enabled": traits.reasoning_enabled,
            "default_effort": default_reasoning_level_val,
            "supported_efforts": advertised_reasoning_levels,
        });
    } else {
        row["reasoning"] = serde_json::Value::Null;
    }

    if traits.reasoning {
        row["thinkingLevelMap"] = thinking_level_map.clone();
        row["thinking_level_map"] = thinking_level_map;
    }
    if let Some(template) = &info.template {
        row["chat_template"] = serde_json::json!(template);
    }
    if !hints.is_empty() {
        row["configuration_hints"] = serde_json::json!(hints);
    }
    row
}

fn model_rows_with_launch_args(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    vllm_args: &[String],
) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    // Instruct models come first so clients that pick the first entry of
    // `/v1/models` still land on a chat model. Embedding models remain
    // discoverable for RAG/indexing clients.
    for task in ["instruct", "embed"] {
        for server in running_servers(servers, task) {
            let args = effective_launch_args(&server.def, vllm_args);
            let (context, source) = resolve_model_context_sync(&server.def, args);
            let template = read_model_chat_template(&server.def, "", "");
            let info = analyze_chat_template(template.as_deref());
            rows.push(model_row_with_template(
                &server.def,
                Some(context),
                source,
                args,
                &info,
            ));
        }
    }
    rows
}

/// OpenAI-style `data` rows for `GET /v1/models`. Prefer the served model
/// name when present (that is the name a client would pass in `"model"`).
///
/// The context/capability fields are extensions to the OpenAI schema. They are
/// deliberately derived from the same definition used to launch the backend,
/// so a harness never sees a different context or capability than the process
/// actually provides.
pub fn model_rows(servers: &BTreeMap<String, crate::state::LiveServer>) -> Vec<serde_json::Value> {
    model_rows_with_launch_args(servers, &[])
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    let parsed = value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| {
            value
                .as_str()
                .and_then(|value| value.trim().parse::<usize>().ok())
        })?;
    (parsed > 0).then_some(parsed)
}

/// Read the context token limit from vLLM's model list or llama.cpp's
/// `/props` response. Different llama.cpp releases put `n_ctx` in slightly
/// different places, hence the small recursive lookup.
fn extract_runtime_context(value: &serde_json::Value) -> Option<usize> {
    const CONTEXT_KEYS: &[&str] = &[
        "max_model_len",
        "max_context_length",
        "context_length",
        "context_window",
        "n_ctx",
    ];
    if let Some(object) = value.as_object() {
        for key in CONTEXT_KEYS {
            if let Some(value) = object.get(*key).and_then(json_usize) {
                return Some(value);
            }
        }
        for key in [
            "data",
            "models",
            "default_generation_settings",
            "model_info",
        ] {
            if let Some(child) = object.get(key) {
                if let Some(context) = extract_runtime_context(child) {
                    return Some(context);
                }
            }
        }
    } else if let Some(items) = value.as_array() {
        for item in items {
            if let Some(context) = extract_runtime_context(item) {
                return Some(context);
            }
        }
    }
    None
}

async fn fetch_runtime_context(state: &Arc<AppState>, def: &ServerDef) -> Option<usize> {
    let api_key = crate::server::server_api_key(state, def);
    let paths: &[&str] = if def.backend == "llamacpp" {
        &["/props", "/v1/models"]
    } else {
        &["/v1/models"]
    };
    for path in paths {
        let url = format!("http://127.0.0.1:{}{path}", def.port);
        let request = apply_vllm_auth(state.http.get(&url), api_key.as_deref());
        let response =
            match tokio::time::timeout(std::time::Duration::from_millis(350), request.send()).await
            {
                Ok(Ok(response)) if response.status().is_success() => response,
                _ => continue,
            };
        let value = match tokio::time::timeout(
            std::time::Duration::from_millis(350),
            response.json::<serde_json::Value>(),
        )
        .await
        {
            Ok(Ok(value)) => value,
            _ => continue,
        };
        if let Some(context) = extract_runtime_context(&value) {
            return Some(context);
        }
    }
    None
}

/// How long a template's effort vocabulary is trusted before re-reading
/// `/props`. Long enough to avoid a probe per completion, short enough that a
/// restart with a different model on the same port is picked up quickly.
const EFFORT_LADDER_TTL_MS: u64 = 30_000;
const RETRY_LADDER_TTL_MS: u64 = 3_000;

fn read_symlink_or_file(path: &std::path::Path) -> Option<String> {
    if let Ok(content) = std::fs::read_to_string(path) {
        return Some(content);
    }
    if let Ok(target) = std::fs::read_link(path) {
        let parent = path.parent()?;
        let resolved = parent.join(target);
        if let Ok(content) = std::fs::read_to_string(resolved) {
            return Some(content);
        }
    }
    None
}

fn extract_chat_template_from_json(content: &str) -> Option<String> {
    let val = serde_json::from_str::<serde_json::Value>(content).ok()?;
    if let Some(tmpl) = val.get("chat_template") {
        if let Some(s) = tmpl.as_str() {
            if !s.trim().is_empty() {
                return Some(s.to_string());
            }
        } else if let Some(arr) = tmpl.as_array() {
            for item in arr {
                if let Some(t) = item.get("template").and_then(|t| t.as_str()) {
                    if !t.trim().is_empty() {
                        return Some(t.to_string());
                    }
                }
            }
        }
    }
    None
}

fn read_vllm_chat_template(distro: &str, def: &ServerDef) -> Option<String> {
    if let Some(path) = &def.model_path {
        let p = std::path::Path::new(path);
        for file in [
            "tokenizer_config.json",
            "chat_template.json",
            "chat_template.jinja",
        ] {
            let candidate = p.join(file);
            if let Some(content) = read_symlink_or_file(&candidate) {
                if file.ends_with(".jinja") {
                    return Some(content);
                }
                if let Some(tmpl) = extract_chat_template_from_json(&content) {
                    return Some(tmpl);
                }
            }
        }
    }
    // Also check local Windows Hugging Face hub cache
    if let Some(home) = dirs::home_dir() {
        let folder = format!("models--{}", def.model_id.replace('/', "--"));
        let hub_path = home
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join(folder)
            .join("snapshots");
        if hub_path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&hub_path) {
                for entry in entries.flatten() {
                    let snapshot_dir = entry.path();
                    if snapshot_dir.is_dir() {
                        for file in [
                            "tokenizer_config.json",
                            "chat_template.json",
                            "chat_template.jinja",
                        ] {
                            let candidate = snapshot_dir.join(file);
                            if let Some(content) = read_symlink_or_file(&candidate) {
                                if file.ends_with(".jinja") {
                                    return Some(content);
                                }
                                if let Some(tmpl) = extract_chat_template_from_json(&content) {
                                    return Some(tmpl);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if !distro.is_empty() {
        let hf_model = &def.model_id;
        let script = format!(
            r#"python3 -c "import glob, json, os, sys
folder = 'models--' + '{hf_model}'.replace('/', '--')
for pat in [f'/root/.cache/huggingface/hub/{{folder}}/snapshots/*', f'/home/*/.cache/huggingface/hub/{{folder}}/snapshots/*']:
    for s in glob.glob(pat):
        for f in ['tokenizer_config.json', 'chat_template.json', 'chat_template.jinja']:
            p = os.path.join(s, f)
            if os.path.isfile(p):
                try:
                    if f.endswith('.jinja'):
                        with open(p, 'r', encoding='utf-8') as fh:
                            sys.stdout.write(fh.read())
                            sys.exit(0)
                    else:
                        with open(p, 'r', encoding='utf-8') as fh:
                            d = json.load(fh)
                            t = d.get('chat_template')
                            if isinstance(t, str) and t.strip():
                                sys.stdout.write(t)
                                sys.exit(0)
                except Exception:
                    pass
""#,
            hf_model = hf_model.replace('\'', "")
        );
        let out = crate::wsl::run_script_root(distro, &script);
        if out.ok && !out.stdout.trim().is_empty() {
            return Some(out.stdout);
        }
    }
    None
}

fn read_vllm_config_context(distro: &str, def: &ServerDef) -> Option<usize> {
    if let Some(home) = dirs::home_dir() {
        let folder = format!("models--{}", def.model_id.replace('/', "--"));
        let hub_path = home
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join(folder)
            .join("snapshots");
        if hub_path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&hub_path) {
                for entry in entries.flatten() {
                    let p = entry.path().join("config.json");
                    if p.is_file() {
                        if let Ok(content) = std::fs::read_to_string(&p) {
                            if let Ok(d) = serde_json::from_str::<serde_json::Value>(&content) {
                                for k in [
                                    "max_position_embeddings",
                                    "max_sequence_length",
                                    "model_max_length",
                                    "max_seq_len",
                                ] {
                                    if let Some(ctx) = d.get(k).and_then(json_usize) {
                                        return Some(ctx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if distro.is_empty() {
        return None;
    }
    let hf_model = &def.model_id;
    let script = format!(
        r#"python3 -c "import glob, json, os, sys
folder = 'models--' + '{hf_model}'.replace('/', '--')
for pat in [f'/root/.cache/huggingface/hub/{{folder}}/snapshots/*', f'/home/*/.cache/huggingface/hub/{{folder}}/snapshots/*']:
    for s in glob.glob(pat):
        p = os.path.join(s, 'config.json')
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                    for k in ['max_position_embeddings', 'max_sequence_length', 'model_max_length', 'max_seq_len']:
                        if k in d and isinstance(d[k], int) and d[k] > 0:
                            sys.stdout.write(str(d[k]))
                            sys.exit(0)
            except Exception:
                pass
""#,
        hf_model = hf_model.replace('\'', "")
    );
    let out = crate::wsl::run_script_root(distro, &script);
    if out.ok && !out.stdout.trim().is_empty() {
        return out.stdout.trim().parse::<usize>().ok();
    }
    None
}

async fn resolve_context_for_server_async(
    state: &Arc<AppState>,
    def: &ServerDef,
    launch_args: &[String],
) -> (usize, &'static str) {
    if let Some(ctx) = configured_context_tokens(def) {
        return (ctx, "configured");
    }
    if def.backend == "llamacpp" {
        if let Some(ctx) = flag_value(launch_args, "-c")
            .or_else(|| flag_value(launch_args, "--ctx-size"))
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            return (ctx, "configured");
        }
    } else if def.backend == "vllm" {
        if let Some(ctx) = flag_value(launch_args, "--max-model-len")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            return (ctx, "configured");
        }
    }

    if let Some(ctx) = fetch_runtime_context(state, def).await {
        return (ctx, "runtime");
    }

    if def.backend == "vllm" && !state.config().distro.is_empty() {
        if let Some(ctx) = read_vllm_config_context(&state.config().distro, def) {
            return (ctx, "model_config");
        }
    }

    resolve_model_context_sync(def, launch_args)
}

fn read_model_chat_template(def: &ServerDef, gguf_dir: &str, distro: &str) -> Option<String> {
    if def.backend == "llamacpp" {
        let raw_model = def.model_path.as_deref().unwrap_or(&def.model_id);
        if let Some(resolved) = crate::server::resolve_gguf_model_path(raw_model, gguf_dir, distro)
        {
            if let Ok(Some(template)) = crate::gguf::chat_template_from_path(&resolved) {
                if !template.trim().is_empty() {
                    return Some(template);
                }
            }
        }
    } else if def.backend == "vllm" {
        if let Some(template) = read_vllm_chat_template(distro, def) {
            if !template.trim().is_empty() {
                return Some(template);
            }
        }
    }
    None
}

async fn fetch_template_info(
    state: &Arc<AppState>,
    def: &ServerDef,
) -> crate::state::TemplateEffortInfo {
    let cfg = state.config();
    let mut template = read_model_chat_template(def, &cfg.gguf_dir, &cfg.distro);
    if template.is_none() && def.backend == "llamacpp" {
        let api_key = crate::server::server_api_key(state, def);
        let url = format!("http://127.0.0.1:{}/props", def.port);
        let request = apply_vllm_auth(state.http.get(&url), api_key.as_deref());
        let response = tokio::time::timeout(std::time::Duration::from_millis(500), request.send())
            .await
            .ok()
            .and_then(|r| r.ok());
        if let Some(resp) = response.filter(|r| r.status().is_success()) {
            let val = tokio::time::timeout(
                std::time::Duration::from_millis(500),
                resp.json::<serde_json::Value>(),
            )
            .await
            .ok()
            .and_then(|v| v.ok());
            template = val.and_then(|v| {
                v.get("chat_template")
                    .or_else(|| {
                        v.get("default_generation_settings")
                            .and_then(|d| d.get("chat_template"))
                    })
                    .or_else(|| v.get("model_info").and_then(|m| m.get("chat_template")))
                    .and_then(|t| t.as_str())
                    .map(str::to_string)
            });
        }
    }

    analyze_chat_template(template.as_deref())
}

async fn template_info_for(
    state: &Arc<AppState>,
    def: &ServerDef,
) -> crate::state::TemplateEffortInfo {
    let now = crate::state::now_ms();
    {
        let cache = state.template_effort_cache.lock().unwrap();
        if let Some((fetched_at, info)) = cache.get(&def.port) {
            let ttl = if info.template.is_some() {
                EFFORT_LADDER_TTL_MS
            } else {
                RETRY_LADDER_TTL_MS
            };
            if now.saturating_sub(*fetched_at) < ttl {
                return info.clone();
            }
        }
    }
    let info = fetch_template_info(state, def).await;
    state
        .template_effort_cache
        .lock()
        .unwrap()
        .insert(def.port, (now, info.clone()));
    info
}

async fn model_rows_for_state(state: &Arc<AppState>) -> Vec<serde_json::Value> {
    let vllm_args = state
        .config()
        .advanced_settings
        .extra_vllm_args
        .as_deref()
        .map(|extra| shlex::split(extra).unwrap_or_default())
        .unwrap_or_default();
    let defs: Vec<ServerDef> = {
        let servers = state.servers.lock().unwrap();
        let mut defs = Vec::new();
        for task in ["instruct", "embed"] {
            defs.extend(
                running_servers(&servers, task)
                    .into_iter()
                    .map(|server| server.def.clone()),
            );
        }
        defs
    };
    let mut rows = Vec::with_capacity(defs.len());
    for def in defs {
        let args = effective_launch_args(&def, &vllm_args);
        let (context, source) = resolve_context_for_server_async(state, &def, args).await;
        let info = template_info_for(state, &def).await;
        rows.push(model_row_with_template(
            &def,
            Some(context),
            source,
            args,
            &info,
        ));
    }
    rows
}

fn decode_path_component(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = *bytes.get(index + 1)?;
            let low = *bytes.get(index + 2)?;
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                b'A'..=b'F' => Some(byte - b'A' + 10),
                _ => None,
            };
            decoded.push((digit(high)? << 4) | digit(low)?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn model_id_from_path(path: &str) -> Option<String> {
    let clean = path.split('?').next()?;
    let raw = clean
        .strip_prefix("/v1/models/")
        .or_else(|| clean.strip_prefix("/models/"))?;
    if raw.is_empty() {
        None
    } else {
        // Decode first: an encoded `%3Alatest` suffix must be stripped too, so
        // a client that refreshes ids from `/v1/models` can address them.
        decode_path_component(raw).map(|id| id.strip_suffix(":latest").unwrap_or(&id).to_string())
    }
}

/// Extract the requested model id from a JSON request body.
/// Accepts `"model"` or `"name"` and strips `:latest` if present.
pub fn model_from_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let name = value
        .get("model")
        .or_else(|| value.get("name"))?
        .as_str()?
        .trim();
    if name.is_empty() {
        return None;
    }
    let stem = name.strip_suffix(":latest").unwrap_or(name);
    Some(stem.to_string())
}

fn json_bool(value: &serde_json::Value) -> Option<bool> {
    match value {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("true") => Some(true),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("on") => Some(true),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("false") => Some(false),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("off") => Some(false),
        _ => None,
    }
}

fn effort_disables_thinking(effort: &str) -> bool {
    matches!(
        effort.trim().to_ascii_lowercase().as_str(),
        "none" | "off" | "false" | "disabled" | "no"
    )
}

/// Best-effort mapping of a harness thinking level onto an on/off toggle.
/// Unknown non-empty levels are treated as "on": the harness asked for
/// thinking even if the gateway does not know that vendor's label.
fn effort_to_toggle(effort: &str) -> Option<bool> {
    let effort = effort.trim();
    if effort.is_empty()
        || effort.eq_ignore_ascii_case("auto")
        || effort.eq_ignore_ascii_case("default")
    {
        None
    } else {
        Some(!effort_disables_thinking(effort))
    }
}

fn is_toggle_label(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "on" | "true" | "enabled"
    )
}

/// Map a harness label onto the vocabulary the target backend accepts.
/// vLLM's request schema accepts the standard levels; llama.cpp forwards
/// levels to the chat template. Toggle-only labels (`on`, `true`, `auto`, ...)
/// return `None` because they are represented by `chat_template_kwargs.enable_thinking`.
fn normalize_effort_label(effort: &str, backend: &str) -> Option<String> {
    let value = effort.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "on" | "true" | "enabled" | "auto" | "default" => None,
        "none" | "off" | "false" | "disabled" | "no" => Some("none".to_string()),
        "minimal" => {
            if backend == "vllm" {
                Some("low".to_string())
            } else {
                Some("minimal".to_string())
            }
        }
        "low" | "medium" | "high" => Some(value),
        "xhigh" | "max" => {
            if backend == "vllm" {
                Some("high".to_string())
            } else {
                Some(value)
            }
        }
        _ if backend == "llamacpp" => Some(value),
        _ => None,
    }
}

/// Standard effort ladder in ascending order.
const EFFORT_LADDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn effort_rank(effort: &str) -> Option<usize> {
    EFFORT_LADDER
        .iter()
        .position(|level| effort.trim().eq_ignore_ascii_case(level))
}

/// Round a standard effort label onto the subset a chat template accepts.
/// Higher labels fall forward (Qwen3.8's official template aliases `high` to
/// `xhigh`) while lower labels fall back to the lowest accepted level.
fn clamp_effort_to_ladder(effort: &str, ladder: &[String]) -> Option<String> {
    let requested = effort_rank(effort)?;
    if requested == 0 {
        return Some("none".to_string());
    }
    let mut ranks: Vec<(usize, &str)> = ladder
        .iter()
        .filter_map(|level| effort_rank(level).map(|rank| (rank, level.as_str())))
        .collect();
    ranks.sort_by_key(|(rank, _)| *rank);
    ranks
        .iter()
        .find(|(rank, _)| *rank >= requested)
        .or_else(|| ranks.last())
        .map(|(_, level)| level.to_string())
}

/// Some chat templates validate `reasoning_effort` and raise a Jinja error for
/// anything outside a small set (Qwen3.8: `xhigh`, `medium`, `low`). Extract
/// that set from the template source so the gateway can round a standard
/// harness label (`high`) instead of forwarding a value the template rejects.
fn template_effort_ladder(template: &str) -> Option<Vec<String>> {
    static CHECK_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
        regex_lite::Regex::new(
            r"(?i)(?:[a-z0-9_]*reasoning|[a-z0-9_]*thinking|[a-z0-9_]*effort)[a-z0-9_]*[^\r\n]{0,80}?(?:not\s+in|in)\s*[\(\[]([^\(\)\]]{0,240})[\)\]]",
        )
        .expect("valid reasoning effort validation regex")
    });
    static MSG_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
        regex_lite::Regex::new(r"(?i)supported\s+(?:types|levels|efforts|values)\s+(?:are|include|:)\s+([^\.\n]{2,100})")
            .expect("valid reasoning effort message regex")
    });

    let mut levels: Vec<String> = Vec::new();
    if let Some(caps) = CHECK_RE.captures(template) {
        if let Some(matched) = caps.get(1) {
            let labels = matched.as_str();
            for (index, part) in labels.split(['\'', '"']).enumerate() {
                if index % 2 == 0 {
                    continue;
                }
                let value = part.trim().to_ascii_lowercase();
                if value == "none"
                    || effort_rank(&value).is_none()
                    || levels.iter().any(|level| level == &value)
                {
                    continue;
                }
                levels.push(value);
            }
        }
    }

    if levels.len() < 2 {
        if let Some(caps) = MSG_RE.captures(template) {
            if let Some(matched) = caps.get(1) {
                for token in matched.as_str().split(|c: char| !c.is_alphabetic()) {
                    let value = token.trim().to_ascii_lowercase();
                    if value.is_empty()
                        || value == "and"
                        || value == "or"
                        || value == "none"
                        || effort_rank(&value).is_none()
                        || levels.iter().any(|level| level == &value)
                    {
                        continue;
                    }
                    levels.push(value);
                }
            }
        }
    }

    levels.sort_by_key(|level| effort_rank(level).unwrap_or(usize::MAX));
    (levels.len() >= 2).then_some(levels)
}

fn template_default_effort(template: &str, ladder: Option<&[String]>) -> Option<String> {
    static DEFAULT_FILTER_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
        regex_lite::Regex::new(
            r#"(?i)(?:reasoning|thinking)(?:_effort|_level)?\s*\|\s*default\(\s*['"]([a-z0-9_-]+)['"]\s*\)"#,
        )
        .expect("valid default filter regex")
    });
    static DEFAULT_MSG_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
        regex_lite::Regex::new(r#"(?i)\b([a-z0-9_-]+)\s*\(\s*default\s*\)"#)
            .expect("valid default message regex")
    });
    static DEFAULT_SET_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
        regex_lite::Regex::new(r#"(?i)set\s+(?:[a-z0-9_]*reasoning|[a-z0-9_]*thinking)[a-z0-9_]*\s*=\s*['"]([a-z0-9_-]+)['"]"#)
            .expect("valid default set regex")
    });

    if let Some(caps) = DEFAULT_FILTER_RE.captures(template) {
        if let Some(matched) = caps.get(1) {
            let val = matched.as_str().trim().to_ascii_lowercase();
            if effort_rank(&val).is_some() {
                if let Some(ladder) = ladder {
                    if ladder.iter().any(|l| l.eq_ignore_ascii_case(&val)) {
                        return Some(val);
                    }
                } else {
                    return Some(val);
                }
            }
        }
    }

    if let Some(caps) = DEFAULT_MSG_RE.captures(template) {
        if let Some(matched) = caps.get(1) {
            let val = matched.as_str().trim().to_ascii_lowercase();
            if effort_rank(&val).is_some() {
                if let Some(ladder) = ladder {
                    if ladder.iter().any(|l| l.eq_ignore_ascii_case(&val)) {
                        return Some(val);
                    }
                } else {
                    return Some(val);
                }
            }
        }
    }

    if let Some(caps) = DEFAULT_SET_RE.captures(template) {
        if let Some(matched) = caps.get(1) {
            let val = matched.as_str().trim().to_ascii_lowercase();
            if effort_rank(&val).is_some() {
                if let Some(ladder) = ladder {
                    if ladder.iter().any(|l| l.eq_ignore_ascii_case(&val)) {
                        return Some(val);
                    }
                } else {
                    return Some(val);
                }
            }
        }
    }

    if let Some(ladder) = ladder {
        if ladder.iter().any(|l| l.eq_ignore_ascii_case("medium")) {
            Some("medium".to_string())
        } else {
            clamp_effort_to_ladder("medium", ladder)
        }
    } else {
        None
    }
}

fn analyze_chat_template(template: Option<&str>) -> crate::state::TemplateEffortInfo {
    let Some(template) = template.filter(|t| !t.trim().is_empty()) else {
        return crate::state::TemplateEffortInfo::default();
    };
    let is_reasoning = template_has_reasoning(template);
    let ladder = template_effort_ladder(template);
    let default_effort = template_default_effort(template, ladder.as_deref());
    let is_toggle_only = template_is_toggle_only(Some(template), ladder.is_some());
    let lower = template.to_ascii_lowercase();
    let is_always_on = is_reasoning
        && !is_toggle_only
        && ladder.is_none()
        && (!lower.contains("enable_thinking")
            && !lower.contains("reasoning_effort")
            && !lower.contains("thinking_effort"));
    crate::state::TemplateEffortInfo {
        ladder,
        default_effort,
        is_reasoning,
        is_toggle_only,
        is_always_on,
        template: Some(template.to_string()),
    }
}

/// Normalize an effort label for the target backend, then round it onto the
/// template's accepted vocabulary when one is known.
fn normalize_effort_for_ladder(
    effort: &str,
    backend: &str,
    ladder: Option<&[String]>,
) -> Option<String> {
    let canonical = normalize_effort_label(effort, backend)?;
    match ladder {
        Some(ladder) if canonical != "none" => {
            Some(clamp_effort_to_ladder(&canonical, ladder).unwrap_or(canonical))
        }
        _ => Some(canonical),
    }
}

fn think_toggle(value: &serde_json::Value) -> Option<bool> {
    match value {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("true") => Some(true),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("on") => Some(true),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("false") => Some(false),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("off") => Some(false),
        _ => None,
    }
}

fn think_level(value: &serde_json::Value) -> Option<&str> {
    let value = value.as_str()?.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("off")
    {
        None
    } else {
        Some(value)
    }
}

/// Read the harness thinking level from every shape we support:
/// OpenAI `reasoning_effort`, OpenRouter `reasoning.effort`, Anthropic
/// `thinking.level`, Ollama `think: "high"`, and vLLM/llama.cpp
/// `chat_template_kwargs.reasoning_effort`.
fn harness_effort(object: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let nested = |key: &str| object.get(key).and_then(serde_json::Value::as_object);
    let value = object
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            object
                .get("thinking_effort")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("reasoning")
                .and_then(|value| value.get("effort"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("reasoning")
                .and_then(|value| value.get("level"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("thinking")
                .and_then(|value| value.get("level"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("thinking")
                .and_then(|value| value.get("effort"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| object.get("think").and_then(think_level))
        .or_else(|| {
            nested("chat_template_kwargs")
                .and_then(|value| value.get("reasoning_effort"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("extra_body")
                .and_then(|value| value.get("reasoning_effort"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            nested("extra_body")
                .and_then(|value| value.get("thinking_effort"))
                .and_then(serde_json::Value::as_str)
        })?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// Explicit request toggles, in order of specificity. `chat_template_kwargs`
/// wins because both backends treat it as the most specific request control.
fn harness_toggle(object: &serde_json::Map<String, serde_json::Value>) -> Option<bool> {
    if let Some(kwargs) = object
        .get("chat_template_kwargs")
        .and_then(serde_json::Value::as_object)
    {
        if let Some(value) = kwargs.get("enable_thinking").and_then(json_bool) {
            return Some(value);
        }
        if let Some(value) = kwargs.get("thinking").and_then(json_bool) {
            return Some(value);
        }
    }
    if let Some(thinking) = object
        .get("thinking")
        .and_then(serde_json::Value::as_object)
    {
        if let Some(kind) = thinking.get("type").and_then(serde_json::Value::as_str) {
            match kind.trim().to_ascii_lowercase().as_str() {
                "enabled" | "on" => return Some(true),
                "disabled" | "off" | "none" => return Some(false),
                _ => {}
            }
        }
        if let Some(value) = thinking.get("enabled").and_then(json_bool) {
            return Some(value);
        }
    }
    if let Some(value) = object.get("enable_thinking").and_then(json_bool) {
        return Some(value);
    }
    if let Some(value) = object.get("think").and_then(think_toggle) {
        return Some(value);
    }
    object
        .get("reasoning")
        .and_then(serde_json::Value::as_object)
        .and_then(|value| value.get("enabled"))
        .and_then(json_bool)
        .or_else(|| object.get("thinking").and_then(json_bool))
}

/// Every budget spelling seen from harnesses. Zero is meaningful in both
/// backends and is preserved.
fn harness_budget(object: &serde_json::Map<String, serde_json::Value>) -> Option<u64> {
    for key in [
        "reasoning_budget_tokens",
        "thinking_budget_tokens",
        "thinking_token_budget",
        "max_thinking_tokens",
    ] {
        if let Some(value) = object.get(key).and_then(serde_json::Value::as_u64) {
            return Some(value);
        }
    }
    object
        .get("thinking")
        .and_then(serde_json::Value::as_object)
        .and_then(|value| value.get("budget_tokens"))
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            object
                .get("reasoning")
                .and_then(serde_json::Value::as_object)
                .and_then(|value| value.get("max_tokens"))
                .and_then(serde_json::Value::as_u64)
        })
}

/// Make harness-controlled fields explicit so they take precedence over any
/// panel/server defaults (launch flags, `--chat-template-kwargs`,
/// `--default-chat-template-kwargs`, environment variables, ...).
///
/// Both backends already let a request-level `chat_template_kwargs` override
/// server defaults, but not every harness sends that shape: some send only
/// `reasoning_effort`, and llama.cpp does not turn that into
/// `enable_thinking=true` on toggle-style templates.  Canonicalizing here
/// closes that gap.  Nothing is injected when the harness does not ask for
/// thinking, so panel settings remain the default for other clients.
#[allow(dead_code)]
fn apply_harness_precedence_with_ladder(
    object: &mut serde_json::Map<String, serde_json::Value>,
    backend: &str,
    context_tokens: Option<usize>,
    effort_ladder: Option<&[String]>,
) {
    let info = effort_ladder.map(|l| crate::state::TemplateEffortInfo {
        ladder: Some(l.to_vec()),
        default_effort: None,
        is_reasoning: true,
        is_toggle_only: false,
        is_always_on: false,
        template: None,
    });
    apply_harness_precedence_with_info(object, backend, context_tokens, info.as_ref());
}

fn apply_harness_precedence_with_info(
    object: &mut serde_json::Map<String, serde_json::Value>,
    backend: &str,
    context_tokens: Option<usize>,
    template_info: Option<&crate::state::TemplateEffortInfo>,
) {
    let model_name = object
        .get("model")
        .or_else(|| object.get("name"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();

    let effort_ladder = template_info.and_then(|info| info.ladder.as_deref());
    let is_toggle_only = template_info
        .map(|info| info.is_toggle_only)
        .unwrap_or_else(|| model_is_deepseek(&model_name, &model_name));

    let raw_effort = harness_effort(object);
    let explicit_toggle = harness_toggle(object);
    let budget = harness_budget(object);

    if is_toggle_only {
        if raw_effort.is_some() || explicit_toggle.is_some() || budget.is_some() {
            let toggle = explicit_toggle
                .or_else(|| raw_effort.as_deref().and_then(effort_to_toggle))
                .or_else(|| budget.filter(|value| *value > 0).map(|_| true))
                .unwrap_or(true);

            if object.get("thinking").is_none() {
                object.insert(
                    "thinking".to_string(),
                    serde_json::json!({
                        "type": if toggle { "enabled" } else { "disabled" }
                    }),
                );
            }

            let mut kwargs = object
                .get("chat_template_kwargs")
                .and_then(serde_json::Value::as_object)
                .cloned()
                .unwrap_or_default();
            kwargs.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(toggle),
            );
            kwargs.remove("reasoning_effort");
            object.insert(
                "chat_template_kwargs".to_string(),
                serde_json::Value::Object(kwargs),
            );

            object.remove("reasoning_effort");
        }
    } else {
        let effort = raw_effort
            .as_deref()
            .and_then(|effort| normalize_effort_for_ladder(effort, backend, effort_ladder));

        // Canonicalize an effort the harness already sent.  vLLM rejects
        // toggle-only labels such as "on"/"auto" in its reasoning_effort enum, so
        // those are represented by the explicit toggle instead.
        if let Some(current) = object
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        {
            match normalize_effort_for_ladder(&current, backend, effort_ladder) {
                Some(canonical) if canonical != current => {
                    object.insert(
                        "reasoning_effort".to_string(),
                        serde_json::Value::String(canonical),
                    );
                }
                None if backend == "vllm" || is_toggle_label(&current) => {
                    object.remove("reasoning_effort");
                }
                _ => {}
            }
        }
        // Some harnesses put the level inside `chat_template_kwargs`, which both
        // backends forward to the template. Round that copy too so it cannot
        // reintroduce a label the template rejects.
        if let Some(kwargs) = object
            .get_mut("chat_template_kwargs")
            .and_then(serde_json::Value::as_object_mut)
        {
            if let Some(current) = kwargs
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
            {
                if let Some(canonical) =
                    normalize_effort_for_ladder(&current, backend, effort_ladder)
                {
                    if canonical != current {
                        kwargs.insert(
                            "reasoning_effort".to_string(),
                            serde_json::Value::String(canonical),
                        );
                    }
                }
            }
        }
        if object
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            if let Some(effort) = &effort {
                object.insert(
                    "reasoning_effort".to_string(),
                    serde_json::Value::String(effort.clone()),
                );
            }
        }

        // A positive thinking budget is itself a request to think.
        let toggle = explicit_toggle
            .or_else(|| raw_effort.as_deref().and_then(effort_to_toggle))
            .or_else(|| budget.filter(|value| *value > 0).map(|_| true));

        if let Some(toggle) = toggle {
            // `chat_template_kwargs.enable_thinking` is the most specific control.
            // llama.cpp parses `reasoning_effort: "none"` *after* it, so drop that
            // contradictory value when the harness explicitly enabled thinking.
            if toggle
                && explicit_toggle == Some(true)
                && object
                    .get("reasoning_effort")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(effort_disables_thinking)
            {
                object.remove("reasoning_effort");
            }

            let mut kwargs = object
                .get("chat_template_kwargs")
                .and_then(serde_json::Value::as_object)
                .cloned()
                .unwrap_or_default();
            let current = kwargs.get("enable_thinking");
            let needs_insert = current.and_then(json_bool) != Some(toggle)
                || current.is_some_and(|value| !value.is_boolean());
            if needs_insert {
                kwargs.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(toggle),
                );
                object.insert(
                    "chat_template_kwargs".to_string(),
                    serde_json::Value::Object(kwargs),
                );
            }

            if backend == "llamacpp" {
                // A panel `--reasoning-format none` must not hide the reasoning
                // trace the harness explicitly asked for.  A request-level
                // `reasoning_format` always overrides the launch default.
                if toggle && object.get("reasoning_format").is_none() {
                    object.insert(
                        "reasoning_format".to_string(),
                        serde_json::Value::String("auto".to_string()),
                    );
                }
                // llama.cpp cannot express "unlimited" per request when the panel
                // set `--reasoning-budget 0` (-1 falls back to the server value),
                // so an explicit "thinking on" request without a budget gets the
                // full context window as its budget.
                if toggle && budget.is_none() && object.get("reasoning_budget_tokens").is_none() {
                    let budget = context_tokens.filter(|value| *value > 0).unwrap_or(32_768);
                    object.insert(
                        "reasoning_budget_tokens".to_string(),
                        serde_json::Value::Number((budget as u64).into()),
                    );
                }
            }
        }
    }

    if let Some(budget) = budget {
        let target = if backend == "vllm" {
            "thinking_token_budget"
        } else {
            "reasoning_budget_tokens"
        };
        if object.get(target).is_none() {
            object.insert(target.to_string(), serde_json::Value::Number(budget.into()));
        }
    }

    // OpenAI's newer alias (max_completion_tokens) and the Responses-style spelling
    // (max_output_tokens) take precedence over older max_tokens defaults, and are
    // mapped onto max_tokens for backends that only accept the canonical name.
    if let Some(value) = object
        .get("max_completion_tokens")
        .or_else(|| object.get("max_output_tokens"))
        .cloned()
    {
        object.insert("max_tokens".to_string(), value);
    }

    // Normalize Ollama-style `"format": "json"` into standard `"response_format"`
    if object.get("response_format").is_none() {
        if let Some(format) = object.get("format").and_then(serde_json::Value::as_str) {
            if format.eq_ignore_ascii_case("json") {
                object.insert(
                    "response_format".to_string(),
                    serde_json::json!({"type": "json_object"}),
                );
            }
        }
    }

    // Normalize legacy OpenAI 0.x functions -> tools
    if object.get("tools").is_none() {
        if let Some(functions) = object
            .get("functions")
            .and_then(serde_json::Value::as_array)
        {
            let tools: Vec<serde_json::Value> = functions
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "type": "function",
                        "function": f.clone()
                    })
                })
                .collect();
            object.insert("tools".to_string(), serde_json::Value::Array(tools));
        }
    }
    // Normalize legacy OpenAI 0.x function_call -> tool_choice
    if object.get("tool_choice").is_none() {
        if let Some(fc) = object.get("function_call") {
            match fc {
                serde_json::Value::String(s) => {
                    object.insert(
                        "tool_choice".to_string(),
                        serde_json::Value::String(s.clone()),
                    );
                }
                serde_json::Value::Object(map) if map.contains_key("name") => {
                    object.insert(
                        "tool_choice".to_string(),
                        serde_json::json!({
                            "type": "function",
                            "function": map.clone()
                        }),
                    );
                }
                _ => {}
            }
        }
    }

    // Normalize OpenAI 'developer' role to 'system' for templates that only support system messages
    if let Some(messages) = object
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    {
        for msg in messages {
            if let Some(msg_obj) = msg.as_object_mut() {
                if let Some(role) = msg_obj.get("role").and_then(serde_json::Value::as_str) {
                    if role == "developer" {
                        msg_obj.insert(
                            "role".to_string(),
                            serde_json::Value::String("system".to_string()),
                        );
                    }
                }
            }
        }
    }
}

/// Normalize a chat request for the target backend.
///
/// The original bytes are returned verbatim when the harness did not ask for
/// any thinking control or OpenAI token-limit alias, so panel settings remain
/// the default for clients that do not care.  When the harness did ask, the
/// returned body contains explicit request-level controls that both vLLM and
/// llama.cpp rank above server defaults.
#[cfg(test)]
fn normalize_chat_request(body: &[u8], backend: &str, context_tokens: Option<usize>) -> Vec<u8> {
    normalize_chat_request_with_ladder(body, backend, context_tokens, None)
}

fn normalize_chat_request_with_info(
    body: &[u8],
    backend: &str,
    context_tokens: Option<usize>,
    template_info: Option<&crate::state::TemplateEffortInfo>,
) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return body.to_vec();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_vec();
    };
    let original = object.clone();
    apply_harness_precedence_with_info(object, backend, context_tokens, template_info);
    if *object == original {
        return body.to_vec();
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
}

#[cfg(test)]
fn normalize_chat_request_with_ladder(
    body: &[u8],
    backend: &str,
    context_tokens: Option<usize>,
    effort_ladder: Option<&[String]>,
) -> Vec<u8> {
    let info = effort_ladder.map(|l| crate::state::TemplateEffortInfo {
        ladder: Some(l.to_vec()),
        default_effort: None,
        is_reasoning: true,
        is_toggle_only: false,
        is_always_on: false,
        template: None,
    });
    normalize_chat_request_with_info(body, backend, context_tokens, info.as_ref())
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn generate_request_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let count = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("req_{now:x}{count:x}")
}

pub fn is_valid_request_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 512 && id.is_ascii() && !id.bytes().any(|b| b.is_ascii_control())
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
    pub retry_after: Option<String>,
    pub request_id: Option<String>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            error_type: error_type.into(),
            param: None,
            code: None,
            retry_after: None,
            request_id: None,
        }
    }

    pub fn bad_request(
        message: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error".into(),
            param,
            code,
            retry_after: None,
            request_id: None,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: 401,
            message: message.into(),
            error_type: "authentication_error".into(),
            param: None,
            code: None,
            retry_after: None,
            request_id: None,
        }
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: 403,
            message: message.into(),
            error_type: "permission_error".into(),
            param: None,
            code: None,
            retry_after: None,
            request_id: None,
        }
    }

    pub fn not_found(
        message: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            status: 404,
            message: message.into(),
            error_type: "invalid_request_error".into(),
            param,
            code,
            retry_after: None,
            request_id: None,
        }
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: 413,
            message: message.into(),
            error_type: "invalid_request_error".into(),
            param: None,
            code: Some("request_too_large".into()),
            retry_after: None,
            request_id: None,
        }
    }

    pub fn rate_limited(message: impl Into<String>, retry_after: Option<String>) -> Self {
        Self {
            status: 429,
            message: message.into(),
            error_type: "rate_limit_error".into(),
            param: None,
            code: Some("rate_limit_exceeded".into()),
            retry_after,
            request_id: None,
        }
    }

    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: 502,
            message: message.into(),
            error_type: "api_error".into(),
            param: None,
            code: Some("bad_gateway".into()),
            retry_after: None,
            request_id: None,
        }
    }

    pub fn method_not_allowed(message: impl Into<String>) -> Self {
        Self {
            status: 405,
            message: message.into(),
            error_type: "invalid_request_error".into(),
            param: None,
            code: Some("method_not_allowed".into()),
            retry_after: None,
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, req_id: impl Into<String>) -> Self {
        self.request_id = Some(req_id.into());
        self
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            message: message.into(),
            error_type: "api_error".into(),
            param: None,
            code: Some("service_unavailable".into()),
            retry_after: None,
            request_id: None,
        }
    }

    pub fn to_json_value(&self) -> serde_json::Value {
        let mut err = serde_json::Map::new();
        err.insert(
            "message".to_string(),
            serde_json::Value::String(self.message.clone()),
        );
        err.insert(
            "type".to_string(),
            serde_json::Value::String(self.error_type.clone()),
        );
        err.insert(
            "param".to_string(),
            match &self.param {
                Some(p) => serde_json::Value::String(p.clone()),
                None => serde_json::Value::Null,
            },
        );
        err.insert(
            "code".to_string(),
            match &self.code {
                Some(c) => serde_json::Value::String(c.clone()),
                None => serde_json::Value::Null,
            },
        );
        serde_json::json!({ "error": serde_json::Value::Object(err) })
    }
}

pub fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };

        result.push(ALPHABET[(b0 >> 2) as usize] as char);
        result.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            result.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

pub fn floats_to_base64(floats: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64_encode(&bytes)
}

pub fn validate_json_content_type(headers: &HashMap<String, String>) -> Result<(), ApiError> {
    let ct = headers.get("content-type").map(|s| s.trim()).unwrap_or("");
    if ct.to_ascii_lowercase().starts_with("application/json") {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "Content-Type must be application/json",
            Some("Content-Type".into()),
            Some("invalid_content_type".into()),
        ))
    }
}

pub fn parse_json_body(body: &[u8]) -> Result<serde_json::Value, ApiError> {
    if body.is_empty() {
        return Err(ApiError::bad_request(
            "Request body must not be empty",
            None,
            Some("missing_body".into()),
        ));
    }
    serde_json::from_slice(body).map_err(|e| {
        ApiError::bad_request(
            format!("Invalid JSON body: {e}"),
            None,
            Some("invalid_json".into()),
        )
    })
}

pub fn model_from_json_body(val: &serde_json::Value) -> Result<String, ApiError> {
    val.get("model")
        .or_else(|| val.get("name"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ApiError::bad_request(
                "Missing required parameter: 'model'",
                Some("model".into()),
                Some("missing_required_parameter".into()),
            )
        })
}

pub fn validate_embedding_input(input_val: &serde_json::Value) -> Result<(), ApiError> {
    match input_val {
        serde_json::Value::String(s) => {
            if s.is_empty() {
                return Err(ApiError::bad_request(
                    "Input string must not be empty",
                    Some("input".into()),
                    Some("invalid_input".into()),
                ));
            }
            Ok(())
        }
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                return Err(ApiError::bad_request(
                    "Input array must not be empty",
                    Some("input".into()),
                    Some("invalid_input".into()),
                ));
            }
            let first = &arr[0];
            if first.is_string() {
                for item in arr {
                    if !item.is_string() {
                        return Err(ApiError::bad_request(
                            "All elements in string input array must be strings",
                            Some("input".into()),
                            Some("invalid_input".into()),
                        ));
                    }
                }
                Ok(())
            } else if first.is_number() {
                for item in arr {
                    if !item.is_number() {
                        return Err(ApiError::bad_request(
                            "All elements in token array must be integers",
                            Some("input".into()),
                            Some("invalid_input".into()),
                        ));
                    }
                }
                Ok(())
            } else if first.is_array() {
                for sub in arr {
                    if let Some(sub_arr) = sub.as_array() {
                        if sub_arr.is_empty() {
                            return Err(ApiError::bad_request(
                                "Token sub-arrays must not be empty",
                                Some("input".into()),
                                Some("invalid_input".into()),
                            ));
                        }
                        for item in sub_arr {
                            if !item.is_number() {
                                return Err(ApiError::bad_request(
                                    "All elements in token array must be integers",
                                    Some("input".into()),
                                    Some("invalid_input".into()),
                                ));
                            }
                        }
                    } else {
                        return Err(ApiError::bad_request(
                            "All elements in batch token input must be arrays",
                            Some("input".into()),
                            Some("invalid_input".into()),
                        ));
                    }
                }
                Ok(())
            } else {
                Err(ApiError::bad_request(
                    "Input array elements must be strings, integers, or arrays of integers",
                    Some("input".into()),
                    Some("invalid_input".into()),
                ))
            }
        }
        _ => Err(ApiError::bad_request(
            "Input must be a string, array of strings, or array of integers",
            Some("input".into()),
            Some("invalid_input".into()),
        )),
    }
}

pub fn normalize_embedding_response(
    body_bytes: &[u8],
    encoding_format: Option<&str>,
    dimensions: Option<usize>,
) -> Result<Vec<u8>, ApiError> {
    let mut val: serde_json::Value = serde_json::from_slice(body_bytes).map_err(|e| {
        ApiError::bad_gateway(format!("Invalid upstream embeddings response JSON: {e}"))
    })?;

    if let Some(data) = val.get_mut("data").and_then(|d| d.as_array_mut()) {
        for (idx, item) in data.iter_mut().enumerate() {
            if let Some(obj) = item.as_object_mut() {
                // Ensure correct index order
                obj.insert("index".to_string(), serde_json::json!(idx));
                obj.insert("object".to_string(), serde_json::json!("embedding"));

                if let Some(emb_val) = obj.get("embedding") {
                    if let Some(arr) = emb_val.as_array() {
                        let mut floats = Vec::with_capacity(arr.len());
                        for v in arr {
                            if let Some(f) = v.as_f64() {
                                floats.push(f as f32);
                            }
                        }
                        if let Some(dim) = dimensions {
                            if dim < floats.len() {
                                floats.truncate(dim);
                            }
                        }
                        if encoding_format == Some("base64") {
                            let b64 = floats_to_base64(&floats);
                            obj.insert("embedding".to_string(), serde_json::Value::String(b64));
                        } else if dimensions.is_some() {
                            let new_arr: Vec<serde_json::Value> =
                                floats.iter().map(|f| serde_json::json!(*f)).collect();
                            obj.insert("embedding".to_string(), serde_json::Value::Array(new_arr));
                        }
                    }
                }
            }
        }
    }

    if val.get("usage").is_none() {
        if let Some(obj) = val.as_object_mut() {
            obj.insert(
                "usage".to_string(),
                serde_json::json!({
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }),
            );
        }
    }

    serde_json::to_vec(&val).map_err(|e| {
        ApiError::bad_gateway(format!(
            "Failed to serialize normalized embeddings response: {e}"
        ))
    })
}

pub fn responses_request_to_chat_completions(
    req_val: &serde_json::Value,
) -> Result<(String, serde_json::Value, bool), ApiError> {
    let model = model_from_json_body(req_val)?;

    if let Some(prev) = req_val.get("previous_response_id") {
        if !prev.is_null() {
            return Err(ApiError::bad_request(
                "previous_response_id is not supported: conversation persistence is disabled on this local gateway",
                Some("previous_response_id".into()),
                Some("unsupported_parameter".into()),
            ));
        }
    }

    if let Some(store) = req_val.get("store").and_then(|s| s.as_bool()) {
        if store {
            return Err(ApiError::bad_request(
                "store=true is not supported: response storage is disabled on this local gateway",
                Some("store".into()),
                Some("unsupported_parameter".into()),
            ));
        }
    }

    let input_val = req_val.get("input").ok_or_else(|| {
        ApiError::bad_request(
            "Missing required parameter: 'input'",
            Some("input".into()),
            Some("missing_required_parameter".into()),
        )
    })?;

    let mut messages = Vec::new();

    if let Some(inst) = req_val.get("instructions").and_then(|i| i.as_str()) {
        if !inst.trim().is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": inst.trim()
            }));
        }
    }

    match input_val {
        serde_json::Value::String(s) => {
            messages.push(serde_json::json!({
                "role": "user",
                "content": s
            }));
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                match item {
                    serde_json::Value::String(s) => {
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": s
                        }));
                    }
                    serde_json::Value::Object(obj) => {
                        let item_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if item_type == "function_call_output" {
                            let call_id = obj.get("call_id").and_then(|c| c.as_str()).unwrap_or("");
                            let output = obj.get("output").and_then(|o| o.as_str()).unwrap_or("");
                            messages.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": output
                            }));
                        } else {
                            let role = obj.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                            let role = if role == "developer" { "system" } else { role };
                            let content = obj
                                .get("content")
                                .cloned()
                                .unwrap_or(serde_json::Value::String(String::new()));
                            messages.push(serde_json::json!({
                                "role": role,
                                "content": content
                            }));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {
            return Err(ApiError::bad_request(
                "Invalid input: must be a string or array of messages/items",
                Some("input".into()),
                Some("invalid_input".into()),
            ));
        }
    }

    let stream = req_val
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let mut chat_req = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });

    if let Some(obj) = chat_req.as_object_mut() {
        if let Some(max_tokens) = req_val
            .get("max_output_tokens")
            .or_else(|| req_val.get("max_tokens"))
        {
            obj.insert("max_tokens".to_string(), max_tokens.clone());
        }
        if let Some(temp) = req_val.get("temperature") {
            obj.insert("temperature".to_string(), temp.clone());
        }
        if let Some(top_p) = req_val.get("top_p") {
            obj.insert("top_p".to_string(), top_p.clone());
        }
        if let Some(tools) = req_val.get("tools") {
            obj.insert("tools".to_string(), tools.clone());
        }
        if let Some(tc) = req_val.get("tool_choice") {
            obj.insert("tool_choice".to_string(), tc.clone());
        }
        if stream {
            obj.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }
    }

    Ok((model, chat_req, stream))
}

pub fn chat_completion_to_responses(
    chat_resp: &serde_json::Value,
    req_model: &str,
    created_at: u64,
) -> serde_json::Value {
    let id = chat_resp
        .get("id")
        .and_then(|v| v.as_str())
        .map(|id| {
            if id.starts_with("chatcmpl-") {
                format!("resp_{}", &id["chatcmpl-".len()..])
            } else if id.starts_with("resp_") {
                id.to_string()
            } else {
                format!("resp_{id}")
            }
        })
        .unwrap_or_else(|| format!("resp_{:016x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed)));

    let model = chat_resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(req_model);

    let choice = chat_resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("stop");

    let (status, incomplete_details) = match finish_reason {
        "length" => (
            "incomplete",
            serde_json::json!({ "reason": "max_output_tokens" }),
        ),
        "content_filter" => (
            "incomplete",
            serde_json::json!({ "reason": "content_filter" }),
        ),
        _ => ("completed", serde_json::Value::Null),
    };

    let mut output_items = Vec::new();
    let mut output_text = String::new();

    if let Some(c) = choice {
        if let Some(msg) = c.get("message") {
            if let Some(refusal) = msg
                .get("refusal")
                .and_then(|r| r.as_str())
                .filter(|r| !r.is_empty())
            {
                let item = serde_json::json!({
                    "id": format!("item_{:016x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed)),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": refusal
                        }
                    ]
                });
                output_items.push(item);
            }
            if let Some(content) = msg.get("content").and_then(|ct| ct.as_str()) {
                output_text = content.to_string();
                let item = serde_json::json!({
                    "id": format!("item_{:016x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed)),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content
                        }
                    ]
                });
                output_items.push(item);
            }
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for tc in tool_calls {
                    let call_id = tc
                        .get("id")
                        .and_then(|id| id.as_str())
                        .unwrap_or("call_unknown");
                    let fn_obj = tc.get("function");
                    let name = fn_obj
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let args = fn_obj
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}");
                    let item = serde_json::json!({
                        "id": call_id,
                        "type": "function_call",
                        "status": "completed",
                        "name": name,
                        "call_id": call_id,
                        "arguments": args
                    });
                    output_items.push(item);
                }
            }
        }
    }

    let usage = chat_resp
        .get("usage")
        .map(|u| {
            serde_json::json!({
                "input_tokens": u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                "output_tokens": u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                "total_tokens": u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            })
        })
        .unwrap_or_else(|| {
            serde_json::json!({
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })
        });

    serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "usage": usage,
        "incomplete_details": incomplete_details
    })
}

/// Non-origin CORS headers shared by all allowed responses.  Kept public for
/// callers that need to inspect the policy without trusting a request origin.
pub const CORS_HEADERS: &str = "Access-Control-Allow-Methods: GET, POST, OPTIONS, PUT, DELETE\r\nAccess-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With, api-key, x-api-key, X-Stainless-Lang, X-Stainless-Package-Version, X-Stainless-OS, X-Stainless-Arch, X-Stainless-Runtime, X-Stainless-Runtime-Version, X-Client-Request-Id, x-client-request-id, X-Request-ID, x-request-id\r\nAccess-Control-Expose-Headers: x-request-id, x-client-request-id, retry-after, x-ratelimit-limit-requests, x-ratelimit-limit-tokens, x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens, x-ratelimit-reset-requests, x-ratelimit-reset-tokens\r\nAccess-Control-Max-Age: 86400\r\n";

/// Return CORS headers only for explicitly trusted local origins.
///
/// The gateway is bound to loopback, but a browser page on any origin can
/// still attempt a localhost request.  Echoing an arbitrary `Origin` (or `*`)
/// would turn the gateway into a cross-site credential/data exfiltration point.
/// The Tauri origins and loopback development origins are the only origins
/// accepted by the local panel.
pub fn cors_headers(origin: Option<&str>) -> String {
    let Some(origin) = origin.filter(|value| is_allowed_origin(value)) else {
        return String::new();
    };
    format!("Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n{CORS_HEADERS}")
}

fn is_allowed_origin(origin: &str) -> bool {
    if origin == "tauri://localhost"
        || origin == "https://tauri.localhost"
        || origin == "http://localhost"
        || origin == "http://127.0.0.1"
        || origin == "https://localhost"
        || origin == "https://127.0.0.1"
    {
        return true;
    }
    [
        "http://127.0.0.1:",
        "http://localhost:",
        "https://127.0.0.1:",
        "https://localhost:",
    ]
    .iter()
    .any(|prefix| {
        origin.strip_prefix(prefix).is_some_and(|port| {
            !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|number| number > 0)
        })
    })
}

pub fn is_head_terminated(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub fn running_model_names(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<String> {
    running_servers(servers, task)
        .into_iter()
        .map(|ls| ls.def.effective_model_name())
        .collect()
}

#[allow(dead_code)]
fn not_found_payload_with_hint(model: &str, kind: &str, running: &[String]) -> String {
    serde_json::json!({
        "error": {
            "message": format!(
                "no running {kind} server for model '{model}'. Running: [{}]",
                running.join(", ")
            ),
            "type": "invalid_request_error",
        },
        "running": running,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Connection serving
// ---------------------------------------------------------------------------

async fn serve_connection(stream: &mut TcpStream, state: &Arc<AppState>) -> Result<(), ApiError> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 8192];
    loop {
        let n = tokio::time::timeout(std::time::Duration::from_secs(30), stream.read(&mut tmp))
            .await
            .map_err(|_| {
                ApiError::bad_request(
                    "request header read timed out",
                    None,
                    Some("read_timeout".into()),
                )
            })?
            .map_err(|e| {
                ApiError::bad_request(
                    format!("read request: {e}"),
                    None,
                    Some("read_error".into()),
                )
            })?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(sep) = is_head_terminated(&buf) {
            let head = parse_head_result(&buf[..sep])?;
            if head.content_length > MAX_BODY_BYTES {
                return Err(ApiError::payload_too_large(
                    "Request body exceeds 32MB limit",
                ));
            }

            // Expect: 100-continue handling
            if head
                .headers
                .get("expect")
                .map(|v| v.eq_ignore_ascii_case("100-continue"))
                .unwrap_or(false)
            {
                let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await;
                let _ = stream.flush().await;
            }

            let total = sep + 4;
            let body = if head.chunked {
                read_chunked_body(stream, &mut buf, total).await?
            } else {
                while buf.len() < total + head.content_length {
                    let m = tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        stream.read(&mut tmp),
                    )
                    .await
                    .map_err(|_| {
                        ApiError::bad_request(
                            "request body read timed out",
                            None,
                            Some("read_timeout".into()),
                        )
                    })?
                    .map_err(|e| {
                        ApiError::bad_request(
                            format!("read body: {e}"),
                            None,
                            Some("read_error".into()),
                        )
                    })?;
                    if m == 0 {
                        return Err(ApiError::bad_request(
                            "connection closed during body",
                            None,
                            Some("connection_closed".into()),
                        ));
                    }
                    buf.extend_from_slice(&tmp[..m]);
                }
                buf[total..total + head.content_length].to_vec()
            };
            return route(stream, state, &head, &body).await;
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err(ApiError::payload_too_large(
                "Request header exceeds 32KB limit",
            ));
        }
    }

    async fn read_chunked_body(
        stream: &mut TcpStream,
        buf: &mut Vec<u8>,
        mut cursor: usize,
    ) -> Result<Vec<u8>, ApiError> {
        let mut body = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            while buf[cursor..]
                .windows(2)
                .position(|w| w == b"\r\n")
                .is_none()
            {
                let n =
                    tokio::time::timeout(std::time::Duration::from_secs(30), stream.read(&mut tmp))
                        .await
                        .map_err(|_| {
                            ApiError::bad_request(
                                "chunked body read timed out",
                                None,
                                Some("read_timeout".into()),
                            )
                        })?
                        .map_err(|e| {
                            ApiError::bad_request(
                                format!("read chunk size: {e}"),
                                None,
                                Some("read_error".into()),
                            )
                        })?;
                if n == 0 {
                    return Err(ApiError::bad_request(
                        "connection closed during chunked body",
                        None,
                        Some("connection_closed".into()),
                    ));
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let relative = buf[cursor..]
                .windows(2)
                .position(|w| w == b"\r\n")
                .expect("chunk delimiter exists");
            let line_end = cursor + relative;
            let size_text = std::str::from_utf8(&buf[cursor..line_end]).map_err(|_| {
                ApiError::bad_request(
                    "invalid chunk size",
                    None,
                    Some("invalid_chunk_size".into()),
                )
            })?;
            let size_text = size_text.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                ApiError::bad_request(
                    "invalid chunk size",
                    None,
                    Some("invalid_chunk_size".into()),
                )
            })?;
            cursor = line_end + 2;
            if size == 0 {
                // Read optional trailer lines until \r\n\r\n or leading \r\n
                loop {
                    if cursor + 2 <= buf.len() && &buf[cursor..cursor + 2] == b"\r\n" {
                        return Ok(body);
                    }
                    if let Some(_pos) = buf[cursor..].windows(4).position(|w| w == b"\r\n\r\n") {
                        return Ok(body);
                    }
                    let n = tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        stream.read(&mut tmp),
                    )
                    .await
                    .map_err(|_| {
                        ApiError::bad_request(
                            "chunk trailers read timed out",
                            None,
                            Some("read_timeout".into()),
                        )
                    })?
                    .map_err(|e| {
                        ApiError::bad_request(
                            format!("read chunk trailers: {e}"),
                            None,
                            Some("read_error".into()),
                        )
                    })?;
                    if n == 0 {
                        return Ok(body);
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
            }
            if body.len().saturating_add(size) > MAX_BODY_BYTES {
                return Err(ApiError::payload_too_large("request body too large"));
            }
            while buf.len() < cursor + size + 2 {
                let n =
                    tokio::time::timeout(std::time::Duration::from_secs(30), stream.read(&mut tmp))
                        .await
                        .map_err(|_| {
                            ApiError::bad_request(
                                "chunked body read timed out",
                                None,
                                Some("read_timeout".into()),
                            )
                        })?
                        .map_err(|e| {
                            ApiError::bad_request(
                                format!("read chunk: {e}"),
                                None,
                                Some("read_error".into()),
                            )
                        })?;
                if n == 0 {
                    return Err(ApiError::bad_request(
                        "connection closed during chunked body",
                        None,
                        Some("connection_closed".into()),
                    ));
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            body.extend_from_slice(&buf[cursor..cursor + size]);
            if &buf[cursor + size..cursor + size + 2] != b"\r\n" {
                return Err(ApiError::bad_request(
                    "invalid chunk terminator",
                    None,
                    Some("invalid_chunk".into()),
                ));
            }
            cursor += size + 2;
        }
    }
}

pub fn extract_or_generate_request_id(head: &RequestHead) -> String {
    head.headers
        .get("x-request-id")
        .or_else(|| head.headers.get("x-client-request-id"))
        .filter(|id| is_valid_request_id(id))
        .cloned()
        .unwrap_or_else(generate_request_id)
}

async fn route(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    head: &RequestHead,
    body: &[u8],
) -> Result<(), ApiError> {
    if let Some(client_req_id) = head.headers.get("x-client-request-id") {
        if !is_valid_request_id(client_req_id) {
            return Err(ApiError::bad_request(
                "Invalid X-Client-Request-Id: must be ASCII and at most 512 characters",
                Some("X-Client-Request-Id".into()),
                Some("invalid_request_id".into()),
            ));
        }
    }
    if let Some(req_id_hdr) = head.headers.get("x-request-id") {
        if !is_valid_request_id(req_id_hdr) {
            return Err(ApiError::bad_request(
                "Invalid x-request-id: must be ASCII and at most 512 characters",
                Some("x-request-id".into()),
                Some("invalid_request_id".into()),
            ));
        }
    }
    let request_id = extract_or_generate_request_id(head);

    if head.method.as_str() == "OPTIONS" {
        if let Some(origin) = head.headers.get("origin") {
            if !is_allowed_origin(origin) {
                return Err(ApiError::forbidden("origin not allowed"));
            }
        }
        let extra = [("x-request-id", request_id.as_str())];
        let _ = write_response_with_origin_and_headers(
            stream,
            204,
            "text/plain",
            b"",
            head.headers.get("origin").map(String::as_str),
            &extra,
        )
        .await;
        return Ok(());
    }

    eprintln!("[gateway] [{request_id}] {} {}", head.method, head.target);

    // Auth check for all /v1/*, /models/*, /models, and /api/* routes except /health, /v1/health, /, and /api/version
    let is_protected = (head.target.starts_with("/v1/")
        || head.target.starts_with("/api/")
        || head.target.starts_with("/models"))
        && head.target != "/health"
        && head.target != "/v1/health"
        && head.target != "/"
        && head.target != "/api/version";
    if is_protected {
        let cfg = state.config();
        let global_key = cfg.advanced_settings.api_key.as_deref();
        if let Err(mut auth_err) = check_gateway_auth_api(&head.headers, global_key) {
            auth_err.request_id = Some(request_id);
            return Err(auth_err);
        }
    }

    let path = head.target.split('?').next().unwrap_or("");
    match (head.method.as_str(), path) {
        ("GET", "/v1/models") | ("GET", "/models") => {
            let text = serde_json::json!({
                "object": "list",
                "data": model_rows_for_state(state).await,
            })
            .to_string();
            write_request_response(stream, head, 200, "application/json", text.as_bytes())
                .await
                .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("GET", candidate_path)
            if candidate_path.starts_with("/v1/models/")
                || candidate_path.starts_with("/models/") =>
        {
            let Some(model_id) = model_id_from_path(candidate_path) else {
                return Err(ApiError::bad_request(
                    "Model id is required",
                    Some("model".into()),
                    Some("missing_model".into()),
                ));
            };
            let rows = model_rows_for_state(state).await;
            let Some(row) = rows.into_iter().find(|row| {
                row["id"].as_str() == Some(model_id.as_str())
                    || row["root"].as_str() == Some(model_id.as_str())
            }) else {
                return Err(ApiError::not_found(
                    format!("Model '{model_id}' was not found"),
                    Some("model".into()),
                    Some("model_not_found".into()),
                ));
            };
            write_request_response(
                stream,
                head,
                200,
                "application/json",
                row.to_string().as_bytes(),
            )
            .await
            .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("GET", "/health") | ("GET", "/v1/health") => {
            write_response_with_origin_and_headers(
                stream,
                200,
                "application/json",
                b"{\"status\":\"ok\",\"service\":\"llm-panel-gateway\"}",
                head.headers.get("origin").map(String::as_str),
                &[("X-Gateway-Service", "llm-panel-gateway")],
            )
            .await
            .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("GET", "/") => {
            write_request_response(stream, head, 200, "text/plain", b"Ollama is running")
                .await
                .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("GET", "/api/version") => {
            write_request_response(
                stream,
                head,
                200,
                "application/json",
                format!(r#"{{"version":"{}"}}"#, env!("CARGO_PKG_VERSION")).as_bytes(),
            )
            .await
            .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("GET", "/api/tags") | ("GET", "/api/ps") => {
            let payload = ollama_tags_payload(state).await.to_string();
            write_request_response(stream, head, 200, "application/json", payload.as_bytes())
                .await
                .map_err(|e| ApiError::new(500, e, "internal_error"))
        }
        ("POST", "/api/show") => {
            let requested_name = model_from_body(body).unwrap_or_default();
            let rows = model_rows_for_state(state).await;
            let found = rows.into_iter().find(|row| {
                row["id"].as_str() == Some(requested_name.as_str())
                    || row["name"].as_str() == Some(requested_name.as_str())
                    || row["display_name"].as_str() == Some(requested_name.as_str())
                    || row["root"].as_str() == Some(requested_name.as_str())
            });
            if let Some(row) = found {
                let id = row["id"].as_str().unwrap_or("model");
                let quant = row["quantization"].as_str().unwrap_or("unknown");
                let format = row["compatibility_type"].as_str().unwrap_or("gguf");
                let params = row
                    .get("parameters_billions")
                    .and_then(serde_json::Value::as_f64);
                let param_size = params.map(|p| format!("{:.0}B", p)).unwrap_or_default();
                let family_info = crate::model_intelligence::infer_model_family(id, None, &[]);
                let family_name = if !family_info.family_name.is_empty() {
                    family_info.family_name
                } else {
                    id.split(['-', '/', '_'])
                        .next()
                        .unwrap_or("model")
                        .to_string()
                };
                let template_str = row
                    .get("chat_template")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let ctx = row
                    .get("context_length")
                    .or_else(|| row.get("max_model_len"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768);
                let parameters_str =
                    format!("num_ctx {ctx}\nstop \"<|im_end|>\"\nstop \"<|endoftext|>\"");
                let is_embed = row["task"].as_str() == Some("embed");
                let mut capabilities = vec!["completion", "chat"];
                if is_embed {
                    capabilities = vec!["embedding"];
                }
                if row["capabilities"]["tools"].as_bool().unwrap_or(false) {
                    capabilities.push("tools");
                }
                if row["capabilities"]["thinking"].as_bool().unwrap_or(false)
                    || row["capabilities"]["reasoning"].as_bool().unwrap_or(false)
                    || row["supportsReasoning"].as_bool().unwrap_or(false)
                    || row["reasoning"].as_bool().unwrap_or(false)
                    || row["reasoning"].is_object()
                {
                    capabilities.push("thinking");
                }
                if row["capabilities"]["vision"].as_bool().unwrap_or(false) {
                    capabilities.push("vision");
                }
                let payload = serde_json::json!({
                    "modelfile": format!("FROM {id}"),
                    "parameters": parameters_str,
                    "template": template_str,
                    "capabilities": capabilities,
                    "details": {
                        "parent_model": "",
                        "format": format,
                        "family": family_name,
                        "families": [family_name],
                        "parameter_size": param_size,
                        "quantization_level": quant,
                    },
                    "thinking_level_map": row.get("thinking_level_map").cloned().unwrap_or(serde_json::Value::Null),
                    "thinkingLevelMap": row.get("thinkingLevelMap").cloned().unwrap_or(serde_json::Value::Null),
                    "model_info": row,
                })
                .to_string();
                write_request_response(stream, head, 200, "application/json", payload.as_bytes())
                    .await
                    .map_err(|e| ApiError::new(500, e, "internal_error"))
            } else {
                Err(ApiError::not_found(
                    format!("model '{requested_name}' not found"),
                    Some("model".into()),
                    Some("model_not_found".into()),
                ))
            }
        }
        ("POST", "/v1/chat/completions") | ("POST", "/api/chat") => {
            validate_json_content_type(&head.headers)?;
            let val = parse_json_body(body)?;
            let model = model_from_json_body(&val)?;
            let route_res = {
                let servers = state.servers.lock().unwrap();
                route_server_for(&servers, &model, "instruct")
            };
            let port = match route_res {
                Ok(port) => port,
                Err(ModelRouteError::Ambiguous { requested, candidates }) => {
                    return Err(ApiError::bad_request(
                        format!(
                            "model '{requested}' is ambiguous; matches multiple running models: [{}]. Please specify the exact model name.",
                            candidates.join(", ")
                        ),
                        Some("model".into()),
                        Some("ambiguous_model".into()),
                    ));
                }
                Err(ModelRouteError::NotFound { requested, running }) => {
                    return Err(ApiError::not_found(
                        format!(
                            "no running instruction server for model '{requested}'. Running: [{}]",
                            running.join(", ")
                        ),
                        Some("model".into()),
                        Some("model_not_found".into()),
                    ));
                }
                Err(ModelRouteError::EmptyModel) => {
                    return Err(ApiError::bad_request(
                        "model name cannot be empty",
                        Some("model".into()),
                        Some("missing_model".into()),
                    ));
                }
            };
            let _permit = acquire_server_permit(state, port, &request_id)?;
            proxy_chat(
                stream,
                state,
                port,
                body,
                head.headers.get("origin").map(String::as_str),
                &request_id,
            )
            .await
        }
        ("POST", "/v1/completions") | ("POST", "/api/generate") => {
            validate_json_content_type(&head.headers)?;
            let val = parse_json_body(body)?;
            if path == "/v1/completions" && val.get("messages").is_some() {
                return Err(ApiError::bad_request(
                    "The 'messages' parameter is not supported on /v1/completions. Please use /v1/chat/completions instead.",
                    Some("messages".into()),
                    Some("unsupported_parameter".into()),
                ));
            }
            let model = model_from_json_body(&val)?;
            let route_res = {
                let servers = state.servers.lock().unwrap();
                route_server_for(&servers, &model, "instruct")
            };
            let port = match route_res {
                Ok(port) => port,
                Err(ModelRouteError::Ambiguous { requested, candidates }) => {
                    return Err(ApiError::bad_request(
                        format!(
                            "model '{requested}' is ambiguous; matches multiple running models: [{}]. Please specify the exact model name.",
                            candidates.join(", ")
                        ),
                        Some("model".into()),
                        Some("ambiguous_model".into()),
                    ));
                }
                Err(ModelRouteError::NotFound { requested, running }) => {
                    return Err(ApiError::not_found(
                        format!(
                            "no running instruction server for model '{requested}'. Running: [{}]",
                            running.join(", ")
                        ),
                        Some("model".into()),
                        Some("model_not_found".into()),
                    ));
                }
                Err(ModelRouteError::EmptyModel) => {
                    return Err(ApiError::bad_request(
                        "model name cannot be empty",
                        Some("model".into()),
                        Some("missing_model".into()),
                    ));
                }
            };
            let _permit = acquire_server_permit(state, port, &request_id)?;
            proxy_post(
                stream,
                state,
                port,
                "/v1/completions",
                body,
                head.headers.get("origin").map(String::as_str),
                &request_id,
            )
            .await
        }
        ("POST", "/v1/embeddings") | ("POST", "/api/embeddings") | ("POST", "/api/embed") => {
            validate_json_content_type(&head.headers)?;
            let val = parse_json_body(body)?;
            let model = model_from_json_body(&val)?;
            let input_val = val.get("input").ok_or_else(|| {
                ApiError::bad_request(
                    "Missing required parameter: 'input'",
                    Some("input".into()),
                    Some("missing_required_parameter".into()),
                )
            })?;
            validate_embedding_input(input_val)?;

            let encoding_format = val.get("encoding_format").and_then(|e| e.as_str());
            if let Some(ef) = encoding_format {
                if ef != "float" && ef != "base64" {
                    return Err(ApiError::bad_request(
                        "Invalid encoding_format: must be 'float' or 'base64'",
                        Some("encoding_format".into()),
                        Some("invalid_parameter".into()),
                    ));
                }
            }

            let dimensions = val.get("dimensions").and_then(|d| d.as_u64()).map(|d| d as usize);
            if let Some(dim) = dimensions {
                if dim == 0 {
                    return Err(ApiError::bad_request(
                        "dimensions must be greater than 0",
                        Some("dimensions".into()),
                        Some("invalid_parameter".into()),
                    ));
                }
            }

            let route_res = {
                let servers = state.servers.lock().unwrap();
                route_server_for(&servers, &model, "embed")
            };
            let port = match route_res {
                Ok(port) => port,
                Err(ModelRouteError::Ambiguous { requested, candidates }) => {
                    return Err(ApiError::bad_request(
                        format!(
                            "model '{requested}' is ambiguous; matches multiple running models: [{}]. Please specify the exact model name.",
                            candidates.join(", ")
                        ),
                        Some("model".into()),
                        Some("ambiguous_model".into()),
                    ));
                }
                Err(ModelRouteError::NotFound { requested, running }) => {
                    return Err(ApiError::not_found(
                        format!(
                            "no running embedding server for model '{requested}'. Running: [{}]",
                            running.join(", ")
                        ),
                        Some("model".into()),
                        Some("model_not_found".into()),
                    ));
                }
                Err(ModelRouteError::EmptyModel) => {
                    return Err(ApiError::bad_request(
                        "model name cannot be empty",
                        Some("model".into()),
                        Some("missing_model".into()),
                    ));
                }
            };
            let _permit = acquire_server_permit(state, port, &request_id)?;
            proxy_embeddings(
                stream,
                state,
                port,
                body,
                head.headers.get("origin").map(String::as_str),
                &request_id,
                encoding_format,
                dimensions,
            )
            .await
        }
        ("POST", "/v1/responses") => {
            validate_json_content_type(&head.headers)?;
            let req_val = parse_json_body(body)?;
            let (model, chat_req_val, stream_mode) =
                responses_request_to_chat_completions(&req_val)?;
            let route_res = {
                let servers = state.servers.lock().unwrap();
                route_server_for(&servers, &model, "instruct")
            };
            let port = match route_res {
                Ok(port) => port,
                Err(ModelRouteError::Ambiguous { requested, candidates }) => {
                    return Err(ApiError::bad_request(
                        format!(
                            "model '{requested}' is ambiguous; matches multiple running models: [{}]. Please specify the exact model name.",
                            candidates.join(", ")
                        ),
                        Some("model".into()),
                        Some("ambiguous_model".into()),
                    ));
                }
                Err(ModelRouteError::NotFound { requested, running }) => {
                    return Err(ApiError::not_found(
                        format!(
                            "no running instruction server for model '{requested}'. Running: [{}]",
                            running.join(", ")
                        ),
                        Some("model".into()),
                        Some("model_not_found".into()),
                    ));
                }
                Err(ModelRouteError::EmptyModel) => {
                    return Err(ApiError::bad_request(
                        "model name cannot be empty",
                        Some("model".into()),
                        Some("missing_model".into()),
                    ));
                }
            };
            let _permit = acquire_server_permit(state, port, &request_id)?;
            let chat_req_bytes = serde_json::to_vec(&chat_req_val).unwrap_or_default();
            let origin = head.headers.get("origin").map(String::as_str);
            if stream_mode {
                proxy_responses_stream(
                    stream,
                    state,
                    port,
                    &chat_req_bytes,
                    origin,
                    &request_id,
                    &model,
                )
                .await
            } else {
                proxy_responses(
                    stream,
                    state,
                    port,
                    &chat_req_bytes,
                    origin,
                    &request_id,
                    &model,
                )
                .await
            }
        }
        ("GET", p) if p == "/v1/responses" || p.starts_with("/v1/responses/") => {
            Err(ApiError::not_found(
                "Response retrieval is not supported: response storage is disabled on this local gateway",
                None,
                Some("not_supported".into()),
            ))
        }
        ("DELETE", p) if p == "/v1/responses" || p.starts_with("/v1/responses/") => {
            Err(ApiError::not_found(
                "Response deletion is not supported: response storage is disabled on this local gateway",
                None,
                Some("not_supported".into()),
            ))
        }
        ("POST", p) if p.starts_with("/v1/responses/") && p.ends_with("/cancel") => {
            Err(ApiError::bad_request(
                "Response cancellation is not supported on this local gateway",
                None,
                Some("not_supported".into()),
            ))
        }
        (_, p) if p.starts_with("/v1/conversations") => {
            Err(ApiError::not_found(
                "Conversations API is not supported: conversation persistence is disabled on this local gateway",
                None,
                Some("not_supported".into()),
            ))
        }
        (method, p) if (p == "/v1/models" || p.starts_with("/v1/models/")) && method != "GET" => {
            Err(ApiError::new(
                405,
                format!("Method '{method}' not allowed on {p}"),
                "invalid_request_error",
            ))
        }
        _ => Err(ApiError::not_found(
            format!("Path '{path}' not found"),
            None,
            Some("not_found".into()),
        )),
    }
}

async fn write_response_with_origin_and_headers(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    origin: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let cors = cors_headers(origin);
    let mut extra = String::new();
    for (k, v) in extra_headers {
        extra.push_str(k);
        extra.push_str(": ");
        extra.push_str(v);
        extra.push_str("\r\n");
    }
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{cors}{extra}Connection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(body).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())
}

async fn write_request_response(
    stream: &mut TcpStream,
    head: &RequestHead,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let req_id = extract_or_generate_request_id(head);
    let extra = [("x-request-id", req_id.as_str())];
    write_response_with_origin_and_headers(
        stream,
        status,
        content_type,
        body,
        head.headers.get("origin").map(String::as_str),
        &extra,
    )
    .await
}

async fn write_api_error_response(
    stream: &mut TcpStream,
    err: &ApiError,
    origin: Option<&str>,
) -> Result<(), String> {
    let json_bytes = serde_json::to_vec(&err.to_json_value()).unwrap_or_default();
    let mut extra = Vec::new();
    let req_id_str;
    if let Some(ref rid) = err.request_id {
        extra.push(("x-request-id", rid.as_str()));
    } else {
        req_id_str = generate_request_id();
        extra.push(("x-request-id", req_id_str.as_str()));
    }
    if let Some(ref ra) = err.retry_after {
        extra.push(("Retry-After", ra.as_str()));
    }
    write_response_with_origin_and_headers(
        stream,
        err.status,
        "application/json",
        &json_bytes,
        origin,
        &extra,
    )
    .await
}

fn hash_to_hex(input: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let h1 = hasher.finish();
    format!("{h1:016x}{h1:016x}{h1:016x}{h1:016x}")
}

async fn ollama_tags_payload(state: &Arc<AppState>) -> serde_json::Value {
    let rows = model_rows_for_state(state).await;
    let models: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|row| {
            let id = row["id"].as_str().unwrap_or("model");
            let quant = row["quantization"].as_str().unwrap_or("unknown");
            let format = row["compatibility_type"].as_str().unwrap_or("gguf");
            let params = row
                .get("parameters_billions")
                .and_then(serde_json::Value::as_f64);
            let param_size = params.map(|p| format!("{:.0}B", p)).unwrap_or_default();
            let size = row
                .get("parameter_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let family_info = crate::model_intelligence::infer_model_family(id, None, &[]);
            let family_name = if !family_info.family_name.is_empty() {
                family_info.family_name
            } else {
                id.split(['-', '/', '_'])
                    .next()
                    .unwrap_or("model")
                    .to_string()
            };
            serde_json::json!({
                "name": format!("{id}:latest"),
                "model": format!("{id}:latest"),
                "modified_at": chrono::Utc::now().to_rfc3339(),
                "size": size,
                "digest": hash_to_hex(id),
                "details": {
                    "parent_model": "",
                    "format": format,
                    "family": family_name,
                    "families": [family_name],
                    "parameter_size": param_size,
                    "quantization_level": quant,
                }
            })
        })
        .collect();
    serde_json::json!({ "models": models })
}

pub fn normalize_chat_response_json(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    if let Some(choices) = obj
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
    {
        for choice in choices {
            let Some(choice_obj) = choice.as_object_mut() else {
                continue;
            };
            if let Some(delta) = choice_obj
                .get_mut("delta")
                .and_then(serde_json::Value::as_object_mut)
            {
                normalize_reasoning_message(delta);
            }
            if let Some(msg) = choice_obj
                .get_mut("message")
                .and_then(serde_json::Value::as_object_mut)
            {
                normalize_reasoning_message(msg);
            }
        }
    }
    if let Some(usage) = obj
        .get_mut("usage")
        .and_then(serde_json::Value::as_object_mut)
    {
        normalize_usage_reasoning(usage);
    }
}

fn normalize_reasoning_message(map: &mut serde_json::Map<String, serde_json::Value>) {
    let rc = map.get("reasoning_content").cloned();
    let r = map.get("reasoning").cloned();
    if let Some(rc_val) = rc {
        if !map.contains_key("reasoning") {
            map.insert("reasoning".to_string(), rc_val);
        }
    } else if let Some(r_val) = r {
        if !map.contains_key("reasoning_content") {
            map.insert("reasoning_content".to_string(), r_val);
        }
    }
}

fn normalize_usage_reasoning(usage: &mut serde_json::Map<String, serde_json::Value>) {
    let mut reasoning_tokens = usage
        .get("reasoning_tokens")
        .and_then(serde_json::Value::as_u64);
    if reasoning_tokens.is_none() {
        reasoning_tokens = usage
            .get("completion_tokens_details")
            .and_then(serde_json::Value::as_object)
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(serde_json::Value::as_u64);
    }
    if let Some(tokens) = reasoning_tokens {
        if !usage.contains_key("reasoning_tokens") {
            usage.insert("reasoning_tokens".to_string(), serde_json::json!(tokens));
        }
        let details = usage
            .entry("completion_tokens_details".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(details_obj) = details.as_object_mut() {
            if !details_obj.contains_key("reasoning_tokens") {
                details_obj.insert("reasoning_tokens".to_string(), serde_json::json!(tokens));
            }
        }
    }
}

pub fn normalize_sse_line(line: &str) -> String {
    let crlf = line.ends_with("\r\n");
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let Some(data) = trimmed.strip_prefix("data: ") else {
        return line.to_string();
    };
    if data == "[DONE]" {
        return line.to_string();
    }
    let Ok(mut val) = serde_json::from_str::<serde_json::Value>(data) else {
        return line.to_string();
    };
    normalize_chat_response_json(&mut val);
    let serialized = serde_json::to_string(&val).unwrap_or_else(|_| data.to_string());
    if crlf {
        format!("data: {serialized}\r\n")
    } else {
        format!("data: {serialized}\n")
    }
}

fn extract_safe_upstream_headers(resp: &reqwest::Response) -> Vec<(String, String)> {
    let mut safe = Vec::new();
    for (k, v) in resp.headers() {
        let key = k.as_str().to_ascii_lowercase();
        if key == "retry-after" || key.starts_with("x-ratelimit-") || key == "x-request-id" {
            if let Ok(val) = v.to_str() {
                safe.push((key, val.to_string()));
            }
        }
    }
    safe
}

async fn proxy_chat(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    body: &[u8],
    origin: Option<&str>,
    request_id: &str,
) -> Result<(), ApiError> {
    let server_def = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .map(|server| server.def.clone())
    };
    let backend = server_def
        .as_ref()
        .map(|def| def.backend.clone())
        .unwrap_or_else(|| "vllm".to_string());
    let vllm_args = state
        .config()
        .advanced_settings
        .extra_vllm_args
        .as_deref()
        .map(|extra| shlex::split(extra).unwrap_or_default())
        .unwrap_or_default();
    let launch_args = server_def
        .as_ref()
        .map(|def| effective_launch_args(def, &vllm_args))
        .unwrap_or_default();
    let context = match server_def.as_ref() {
        Some(def) => {
            let (ctx, _) = resolve_context_for_server_async(state, def, launch_args).await;
            Some(ctx)
        }
        None => None,
    };
    // llama.cpp forwards the effort label to the model's Jinja template, and
    // a template may only accept a subset of the standard ladder. Round the
    // harness value onto that subset instead of letting the template raise.
    let template_info = match server_def.as_ref() {
        Some(def) => Some(template_info_for(state, def).await),
        None => None,
    };
    let body = normalize_chat_request_with_info(body, &backend, context, template_info.as_ref());
    proxy_post(
        stream,
        state,
        port,
        "/v1/chat/completions",
        &body,
        origin,
        request_id,
    )
    .await
}

/// Forward a chat/completions request to the target vLLM port, preserving
/// streaming when the caller asked for `"stream": true`.
async fn proxy_post(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    upstream_path: &str,
    body: &[u8],
    origin: Option<&str>,
    request_id: &str,
) -> Result<(), ApiError> {
    let url = format!("http://127.0.0.1:{port}{upstream_path}");
    let streaming = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    let api_key = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .and_then(|server| crate::server::server_api_key(state, &server.def))
    }
    .or_else(|| {
        state
            .config()
            .servers
            .iter()
            .find(|server| server.port == port)
            .and_then(|server| crate::server::server_api_key(state, server))
    });
    let req = state
        .chat_http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec());
    let resp = match apply_vllm_auth(req, api_key.as_deref()).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_connect() || e.is_timeout() {
                let mut err = ApiError::service_unavailable(format!(
                    "Upstream server on port {port} is temporarily unreachable: {e}"
                ));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            } else {
                let mut err = ApiError::bad_gateway(format!("upstream POST {url}: {e}"));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            }
        }
    };

    let status = resp.status().as_u16();
    let safe_headers = extract_safe_upstream_headers(&resp);

    if status == 429 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let mut err = ApiError::rate_limited("upstream rate limit exceeded", retry_after);
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }
    if status >= 500 {
        let body_bytes = resp.bytes().await.unwrap_or_default();
        let msg = String::from_utf8_lossy(&body_bytes);
        let mut err = ApiError::bad_gateway(format!("upstream error ({status}): {msg}"));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }

    let reason = resp.status().canonical_reason().unwrap_or("OK");

    if streaming {
        let mut safe_header_str = String::new();
        for (k, v) in &safe_headers {
            safe_header_str.push_str(k);
            safe_header_str.push_str(": ");
            safe_header_str.push_str(v);
            safe_header_str.push_str("\r\n");
        }
        let head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nx-request-id: {request_id}\r\n{safe_header_str}{}Connection: close\r\n\r\n",
            cors_headers(origin)
        );
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
        let mut upstream = resp;
        let mut line_buf = Vec::new();
        // Completion aggregator: observes canonical SSE events without
        // altering forwarded bytes. Diagnostics never include prompts,
        // completions, or authorization material.
        let requested_model = model_from_body(body).unwrap_or_default();
        let mut aggregator =
            crate::completion::StreamAggregator::new("openai-compatible", &requested_model);
        let mut decoder = crate::completion::SseDecoder::new();
        let mut termination = crate::completion::StreamTermination::CleanEof;
        let mut saw_done_line = false;
        loop {
            match upstream.chunk().await {
                Ok(Some(chunk)) => {
                    for event in decoder.feed(&chunk) {
                        aggregator.apply(&event);
                    }
                    line_buf.extend_from_slice(&chunk);
                    while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = line_buf.drain(..=pos).collect();
                        if let Ok(line_str) = std::str::from_utf8(&line_bytes) {
                            if line_str.trim().trim_end_matches(['\r', '\n']) == "data: [DONE]" {
                                saw_done_line = true;
                            }
                            let normalized = normalize_sse_line(line_str);
                            let size = format!("{:x}\r\n", normalized.len());
                            stream
                                .write_all(size.as_bytes())
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                            stream
                                .write_all(normalized.as_bytes())
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                            stream
                                .write_all(b"\r\n")
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                        } else {
                            let size = format!("{:x}\r\n", line_bytes.len());
                            stream
                                .write_all(size.as_bytes())
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                            stream
                                .write_all(&line_bytes)
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                            stream
                                .write_all(b"\r\n")
                                .await
                                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    // Headers are already on the wire, so the status cannot be
                    // changed. Log the real cause instead of silently turning a
                    // broken upstream stream into a clean EOF.
                    eprintln!("[gateway] upstream stream error on port {port}: {e}");
                    termination = crate::completion::StreamTermination::Disconnect;
                    break;
                }
            }
        }
        if let Some(final_event) = decoder.finish() {
            aggregator.apply(&final_event);
        }
        if !line_buf.is_empty() {
            // Forward a final event that did not end in a newline as a proper
            // SSE line, otherwise the client never parses it.
            if !line_buf.ends_with(b"\n") {
                line_buf.push(b'\n');
            }
            let size = format!("{:x}\r\n", line_buf.len());
            stream
                .write_all(size.as_bytes())
                .await
                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
            stream
                .write_all(&line_buf)
                .await
                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
            stream
                .write_all(b"\r\n")
                .await
                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
        }
        if termination == crate::completion::StreamTermination::CleanEof {
            termination = if aggregator.saw_explicit_terminal() {
                crate::completion::StreamTermination::ExplicitTerminal
            } else if saw_done_line {
                crate::completion::StreamTermination::DoneSentinel
            } else {
                crate::completion::StreamTermination::CleanEof
            };
        }
        let normalized = aggregator.finish(termination);
        let diagnostics = crate::completion::CompletionDiagnostics::from(&normalized);
        eprintln!("{}", diagnostics.log_line());
        // Only an explicit terminal + [DONE] yields a clean chunked EOF.
        // Otherwise shut down the client socket so OpenAI SDKs observe an
        // interrupted stream (missing terminal/[DONE]) rather than a
        // silently successful completion.
        if normalized.explicit_terminal {
            if !saw_done_line && !normalized.content.is_empty() {
                let terminal = match serde_json::to_string(&serde_json::json!({
                    "choices": [{
                        "delta": {},
                        "finish_reason": normalized.finish_reason.to_openai_chat().unwrap_or("stop"),
                        "index": 0,
                    }],
                    "created": 0,
                    "id": normalized.response_id.clone().unwrap_or_else(|| "chatcmpl-local".to_string()),
                    "model": normalized.model,
                    "object": "chat.completion.chunk",
                })) {
                    Ok(payload) => format!("data: {payload}\n\n"),
                    Err(_) => String::new(),
                };
                if !terminal.is_empty() {
                    let size = format!("{:x}\r\n", terminal.len());
                    let _ = stream.write_all(size.as_bytes()).await;
                    let _ = stream.write_all(terminal.as_bytes()).await;
                    let _ = stream.write_all(b"\r\n").await;
                    let done = "data: [DONE]\n\n";
                    let size = format!("{:x}\r\n", done.len());
                    let _ = stream.write_all(size.as_bytes()).await;
                    let _ = stream.write_all(done.as_bytes()).await;
                    let _ = stream.write_all(b"\r\n").await;
                }
            }
            stream
                .write_all(b"0\r\n\r\n")
                .await
                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;
            stream
                .flush()
                .await
                .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))
        } else {
            // Incomplete stream: flush forwarded bytes then abort without
            // the chunked terminator so the client treats it as truncated.
            let _ = stream.flush().await;
            let _ = stream.shutdown().await;
            Ok(())
        }
    } else {
        let body_bytes = resp
            .bytes()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("upstream body read: {e}")))?;
        let mut final_bytes = body_bytes.to_vec();
        if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            normalize_chat_response_json(&mut val);
            // Truthful completion extension: preserves the provider's
            // `finish_reason` and `usage` verbatim while making the
            // completion contract explicit for diagnostics. A body without
            // `finish_reason` is `unknown`, never promoted to `stop`..
            let model_fallback = model_from_body(body).unwrap_or_default();
            let normalized = crate::completion::normalize_non_streaming(
                "openai-compatible",
                &model_fallback,
                &val,
                None,
            );
            let diagnostics = crate::completion::CompletionDiagnostics::from(&normalized);
            eprintln!("{}", diagnostics.log_line());
            if let Some(obj) = val.as_object_mut() {
                obj.insert(
                    "x_local_panel_completion".to_string(),
                    crate::completion::completion_extension(&normalized),
                );
            }
            if let Ok(s) = serde_json::to_vec(&val) {
                final_bytes = s;
            }
        }
        let mut extra: Vec<(&str, &str)> = safe_headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        extra.push(("x-request-id", request_id));
        write_response_with_origin_and_headers(
            stream,
            status,
            "application/json",
            &final_bytes,
            origin,
            &extra,
        )
        .await
        .map_err(|e| ApiError::new(500, e, "internal_error"))
    }
}

async fn proxy_embeddings(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    body: &[u8],
    origin: Option<&str>,
    request_id: &str,
    encoding_format: Option<&str>,
    dimensions: Option<usize>,
) -> Result<(), ApiError> {
    let url = format!("http://127.0.0.1:{port}/v1/embeddings");
    let api_key = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .and_then(|server| crate::server::server_api_key(state, &server.def))
    }
    .or_else(|| {
        state
            .config()
            .servers
            .iter()
            .find(|server| server.port == port)
            .and_then(|server| crate::server::server_api_key(state, server))
    });

    let req = state
        .chat_http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec());

    let resp = match apply_vllm_auth(req, api_key.as_deref()).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_connect() || e.is_timeout() {
                let mut err = ApiError::service_unavailable(format!(
                    "Upstream server on port {port} is temporarily unreachable: {e}"
                ));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            } else {
                let mut err = ApiError::bad_gateway(format!("upstream POST {url}: {e}"));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            }
        }
    };

    let status = resp.status().as_u16();
    let safe_headers = extract_safe_upstream_headers(&resp);

    if status == 429 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let mut err = ApiError::rate_limited("upstream rate limit exceeded", retry_after);
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }
    if status >= 500 {
        let body_bytes = resp.bytes().await.unwrap_or_default();
        let msg = String::from_utf8_lossy(&body_bytes);
        let mut err = ApiError::bad_gateway(format!("upstream error ({status}): {msg}"));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }

    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| ApiError::bad_gateway(format!("upstream body read: {e}")))?;

    let normalized_bytes = normalize_embedding_response(&body_bytes, encoding_format, dimensions)?;

    let mut extra: Vec<(&str, &str)> = safe_headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    extra.push(("x-request-id", request_id));

    write_response_with_origin_and_headers(
        stream,
        status,
        "application/json",
        &normalized_bytes,
        origin,
        &extra,
    )
    .await
    .map_err(|e| ApiError::new(500, e, "internal_error"))
}

async fn proxy_responses(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    chat_body: &[u8],
    origin: Option<&str>,
    request_id: &str,
    req_model: &str,
) -> Result<(), ApiError> {
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let api_key = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .and_then(|server| crate::server::server_api_key(state, &server.def))
    }
    .or_else(|| {
        state
            .config()
            .servers
            .iter()
            .find(|server| server.port == port)
            .and_then(|server| crate::server::server_api_key(state, server))
    });

    let req = state
        .chat_http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(chat_body.to_vec());

    let resp = match apply_vllm_auth(req, api_key.as_deref()).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_connect() || e.is_timeout() {
                let mut err = ApiError::service_unavailable(format!(
                    "Upstream server on port {port} is temporarily unreachable: {e}"
                ));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            } else {
                let mut err = ApiError::bad_gateway(format!("upstream POST {url}: {e}"));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            }
        }
    };

    let status = resp.status().as_u16();
    let safe_headers = extract_safe_upstream_headers(&resp);

    if status == 429 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let mut err = ApiError::rate_limited("upstream rate limit exceeded", retry_after);
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }
    if status >= 500 {
        let body_bytes = resp.bytes().await.unwrap_or_default();
        let msg = String::from_utf8_lossy(&body_bytes);
        let mut err = ApiError::bad_gateway(format!("upstream error ({status}): {msg}"));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }

    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| ApiError::bad_gateway(format!("upstream body read: {e}")))?;

    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let val = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .map_err(|e| ApiError::bad_gateway(format!("invalid upstream JSON response: {e}")))?;

    let resp_obj = chat_completion_to_responses(&val, req_model, created_at);
    let final_bytes = serde_json::to_vec(&resp_obj)
        .map_err(|e| ApiError::bad_gateway(format!("serialization error: {e}")))?;

    let mut extra: Vec<(&str, &str)> = safe_headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    extra.push(("x-request-id", request_id));

    write_response_with_origin_and_headers(
        stream,
        status,
        "application/json",
        &final_bytes,
        origin,
        &extra,
    )
    .await
    .map_err(|e| ApiError::new(500, e, "internal_error"))
}

async fn proxy_responses_stream(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    chat_body: &[u8],
    origin: Option<&str>,
    request_id: &str,
    req_model: &str,
) -> Result<(), ApiError> {
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let api_key = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .and_then(|server| crate::server::server_api_key(state, &server.def))
    }
    .or_else(|| {
        state
            .config()
            .servers
            .iter()
            .find(|server| server.port == port)
            .and_then(|server| crate::server::server_api_key(state, server))
    });

    let req = state
        .chat_http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(chat_body.to_vec());

    let mut upstream = match apply_vllm_auth(req, api_key.as_deref()).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_connect() || e.is_timeout() {
                let mut err = ApiError::service_unavailable(format!(
                    "Upstream server on port {port} is temporarily unreachable: {e}"
                ));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            } else {
                let mut err = ApiError::bad_gateway(format!("upstream POST {url}: {e}"));
                err.request_id = Some(request_id.to_string());
                return Err(err);
            }
        }
    };

    let status = upstream.status().as_u16();
    if status == 429 {
        let retry_after = upstream
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let mut err = ApiError::rate_limited("upstream rate limit exceeded", retry_after);
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }
    if status >= 500 {
        let body_bytes = upstream.bytes().await.unwrap_or_default();
        let msg = String::from_utf8_lossy(&body_bytes);
        let mut err = ApiError::bad_gateway(format!("upstream error ({status}): {msg}"));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }

    let reason = upstream.status().canonical_reason().unwrap_or("OK");
    let safe_headers = extract_safe_upstream_headers(&upstream);
    let mut safe_header_str = String::new();
    for (k, v) in &safe_headers {
        safe_header_str.push_str(k);
        safe_header_str.push_str(": ");
        safe_header_str.push_str(v);
        safe_header_str.push_str("\r\n");
    }

    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nx-request-id: {request_id}\r\n{safe_header_str}{}Connection: close\r\n\r\n",
        cors_headers(origin)
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| ApiError::new(500, e.to_string(), "internal_error"))?;

    let resp_id = format!("resp_{:016x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let item_id = format!("item_{:016x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    async fn send_sse(
        stream: &mut TcpStream,
        event: &str,
        data: &serde_json::Value,
    ) -> Result<(), String> {
        let payload = format!(
            "event: {event}\ndata: {}\n\n",
            serde_json::to_string(data).unwrap_or_default()
        );
        let size = format!("{:x}\r\n", payload.len());
        stream
            .write_all(size.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stream
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stream.write_all(b"\r\n").await.map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())
    }

    let initial_resp = serde_json::json!({
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": req_model,
            "output": [],
            "output_text": "",
            "usage": serde_json::Value::Null,
            "incomplete_details": serde_json::Value::Null,
        }
    });
    let _ = send_sse(stream, "response.created", &initial_resp).await;
    let _ = send_sse(stream, "response.in_progress", &initial_resp).await;

    #[derive(Default)]
    struct ActiveToolItem {
        output_index: usize,
        call_id: String,
        name: String,
        arguments: String,
        added_emitted: bool,
        done_emitted: bool,
    }

    let mut decoder = crate::completion::SseDecoder::new();
    let mut aggregator = crate::completion::StreamAggregator::new("openai-compatible", req_model);
    let mut next_output_index: usize = 0;
    let mut message_output_index: Option<usize> = None;
    let mut item_added = false;
    let mut full_output_text = String::new();
    let mut active_tools: BTreeMap<usize, ActiveToolItem> = BTreeMap::new();

    loop {
        match upstream.chunk().await {
            Ok(Some(chunk)) => {
                for event in decoder.feed(&chunk) {
                    aggregator.apply(&event);
                    if !event.is_comment {
                        let line = event.joined_data();
                        let trimmed = line.trim();
                        if trimmed == "[DONE]" {
                            continue;
                        }
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                            if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                                if let Some(first) = choices.first() {
                                    if let Some(delta) = first.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|ct| ct.as_str())
                                        {
                                            if !content.is_empty() {
                                                if !item_added {
                                                    let out_idx = next_output_index;
                                                    next_output_index += 1;
                                                    message_output_index = Some(out_idx);
                                                    let added_payload = serde_json::json!({
                                                        "output_index": out_idx,
                                                        "item": {
                                                            "id": &item_id,
                                                            "type": "message",
                                                            "status": "in_progress",
                                                            "role": "assistant",
                                                            "content": []
                                                        }
                                                    });
                                                    let _ = send_sse(
                                                        stream,
                                                        "response.output_item.added",
                                                        &added_payload,
                                                    )
                                                    .await;
                                                    item_added = true;
                                                }
                                                let out_idx = message_output_index.unwrap_or(0);
                                                let delta_payload = serde_json::json!({
                                                    "item_id": &item_id,
                                                    "output_index": out_idx,
                                                    "content_index": 0,
                                                    "delta": content
                                                });
                                                let _ = send_sse(
                                                    stream,
                                                    "response.output_text.delta",
                                                    &delta_payload,
                                                )
                                                .await;
                                                full_output_text.push_str(content);
                                            }
                                        }

                                        if let Some(tool_calls) =
                                            delta.get("tool_calls").and_then(|tc| tc.as_array())
                                        {
                                            for tc in tool_calls {
                                                let tc_idx = tc
                                                    .get("index")
                                                    .and_then(|i| i.as_u64())
                                                    .map(|i| i as usize)
                                                    .unwrap_or(0);
                                                let tool = active_tools
                                                    .entry(tc_idx)
                                                    .or_insert_with(|| {
                                                        let out_idx = next_output_index;
                                                        next_output_index += 1;
                                                        ActiveToolItem {
                                                            output_index: out_idx,
                                                            call_id: format!(
                                                                "call_{:016x}",
                                                                REQ_COUNTER.fetch_add(
                                                                    1,
                                                                    Ordering::Relaxed
                                                                )
                                                            ),
                                                            name: String::new(),
                                                            arguments: String::new(),
                                                            added_emitted: false,
                                                            done_emitted: false,
                                                        }
                                                    });

                                                if let Some(id) = tc
                                                    .get("id")
                                                    .and_then(|id| id.as_str())
                                                    .filter(|s| !s.is_empty())
                                                {
                                                    tool.call_id = id.to_string();
                                                }
                                                if let Some(fn_obj) = tc.get("function") {
                                                    if let Some(name) = fn_obj
                                                        .get("name")
                                                        .and_then(|n| n.as_str())
                                                        .filter(|s| !s.is_empty())
                                                    {
                                                        tool.name = name.to_string();
                                                    }
                                                }

                                                if !tool.added_emitted {
                                                    let added_payload = serde_json::json!({
                                                        "output_index": tool.output_index,
                                                        "item": {
                                                            "id": &tool.call_id,
                                                            "type": "function_call",
                                                            "status": "in_progress",
                                                            "name": &tool.name,
                                                            "call_id": &tool.call_id,
                                                            "arguments": ""
                                                        }
                                                    });
                                                    let _ = send_sse(
                                                        stream,
                                                        "response.output_item.added",
                                                        &added_payload,
                                                    )
                                                    .await;
                                                    tool.added_emitted = true;
                                                }

                                                if let Some(fn_obj) = tc.get("function") {
                                                    if let Some(arg_delta) = fn_obj
                                                        .get("arguments")
                                                        .and_then(|a| a.as_str())
                                                    {
                                                        if !arg_delta.is_empty() {
                                                            let delta_payload = serde_json::json!({
                                                                "item_id": &tool.call_id,
                                                                "output_index": tool.output_index,
                                                                "call_id": &tool.call_id,
                                                                "delta": arg_delta
                                                            });
                                                            let _ = send_sse(
                                                                stream,
                                                                "response.function_call_arguments.delta",
                                                                &delta_payload,
                                                            )
                                                            .await;
                                                            tool.arguments.push_str(arg_delta);
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
            Ok(None) => break,
            Err(e) => {
                eprintln!("[gateway] upstream stream error during responses: {e}");
                let err_payload = serde_json::json!({
                    "error": {
                        "message": format!("Upstream stream error: {e}"),
                        "type": "api_error",
                        "param": serde_json::Value::Null,
                        "code": "upstream_error"
                    }
                });
                let _ = send_sse(stream, "error", &err_payload).await;
                let _ = stream.write_all(b"0\r\n\r\n").await;
                let _ = stream.flush().await;
                return Ok(());
            }
        }
    }

    if let Some(final_event) = decoder.finish() {
        aggregator.apply(&final_event);
    }

    let normalized = aggregator.finish(crate::completion::StreamTermination::DoneSentinel);

    if item_added {
        let out_idx = message_output_index.unwrap_or(0);
        let item_done_payload = serde_json::json!({
            "output_index": out_idx,
            "item": {
                "id": &item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": &full_output_text
                    }
                ]
            }
        });
        let _ = send_sse(stream, "response.output_item.done", &item_done_payload).await;
    }

    for (_, tool) in active_tools.iter_mut() {
        if !tool.done_emitted {
            let args_done_payload = serde_json::json!({
                "item_id": &tool.call_id,
                "output_index": tool.output_index,
                "call_id": &tool.call_id,
                "arguments": &tool.arguments
            });
            let _ = send_sse(
                stream,
                "response.function_call_arguments.done",
                &args_done_payload,
            )
            .await;

            let item_done_payload = serde_json::json!({
                "output_index": tool.output_index,
                "item": {
                    "id": &tool.call_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": &tool.name,
                    "call_id": &tool.call_id,
                    "arguments": &tool.arguments
                }
            });
            let _ = send_sse(stream, "response.output_item.done", &item_done_payload).await;
            tool.done_emitted = true;
        }
    }

    let is_length = normalized.finish_reason == crate::completion::NormalizedFinishReason::Length;
    let (event_name, status_str, incomplete_details) = if is_length {
        (
            "response.incomplete",
            "incomplete",
            serde_json::json!({ "reason": "max_output_tokens" }),
        )
    } else {
        ("response.completed", "completed", serde_json::Value::Null)
    };

    let mut output_items: Vec<(usize, serde_json::Value)> = Vec::new();
    if item_added {
        let out_idx = message_output_index.unwrap_or(0);
        output_items.push((
            out_idx,
            serde_json::json!({
                "id": &item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": &full_output_text
                    }
                ]
            }),
        ));
    }
    for tool in active_tools.values() {
        output_items.push((
            tool.output_index,
            serde_json::json!({
                "id": &tool.call_id,
                "type": "function_call",
                "status": "completed",
                "name": &tool.name,
                "call_id": &tool.call_id,
                "arguments": &tool.arguments
            }),
        ));
    }
    output_items.sort_by_key(|(idx, _)| *idx);
    let output_values: Vec<serde_json::Value> =
        output_items.into_iter().map(|(_, val)| val).collect();

    let usage_val = normalized.usage.to_responses_usage();
    let final_resp = serde_json::json!({
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": status_str,
            "model": req_model,
            "output": output_values,
            "output_text": full_output_text,
            "usage": usage_val,
            "incomplete_details": incomplete_details,
        }
    });

    let _ = send_sse(stream, event_name, &final_resp).await;

    let done_payload = "data: [DONE]\n\n";
    let size = format!("{:x}\r\n", done_payload.len());
    let _ = stream.write_all(size.as_bytes()).await;
    let _ = stream.write_all(done_payload.as_bytes()).await;
    let _ = stream.write_all(b"\r\n").await;

    let _ = stream.write_all(b"0\r\n\r\n").await;
    let _ = stream.flush().await;

    Ok(())
}

async fn handle_connection(mut stream: TcpStream, state: Arc<AppState>) {
    let result = serve_connection(&mut stream, &state).await;
    if let Err(e) = result {
        let _ = write_api_error_response(&mut stream, &e, None).await;
    }
    let _ = stream.shutdown().await;
}

/// Bind the gateway on `port` and serve until aborted. Returns a join handle
/// so the supervisor can stop it by aborting.
pub const DEFAULT_SERVER_CONCURRENCY: usize = 16;

#[derive(Debug, PartialEq, Eq)]
pub enum ConcurrencyError {
    LimitExceeded {
        port: u16,
        current: usize,
        limit: usize,
    },
}

#[derive(Default, Debug)]
pub struct ServerConcurrencyManager {
    inflight: std::sync::Mutex<HashMap<u16, usize>>,
}

#[derive(Debug)]
pub struct ServerPermit {
    port: u16,
    manager: Arc<ServerConcurrencyManager>,
}

impl Drop for ServerPermit {
    fn drop(&mut self) {
        let mut map = self.manager.inflight.lock().unwrap();
        if let Some(count) = map.get_mut(&self.port) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
}

impl ServerConcurrencyManager {
    pub fn new() -> Self {
        Self {
            inflight: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn current_inflight(&self, port: u16) -> usize {
        *self.inflight.lock().unwrap().get(&port).unwrap_or(&0)
    }

    pub fn try_acquire(
        self: &Arc<Self>,
        port: u16,
        limit: usize,
    ) -> Result<ServerPermit, ConcurrencyError> {
        let mut map = self.inflight.lock().unwrap();
        let current = map.entry(port).or_insert(0);
        if *current >= limit {
            return Err(ConcurrencyError::LimitExceeded {
                port,
                current: *current,
                limit,
            });
        }
        *current += 1;
        Ok(ServerPermit {
            port,
            manager: Arc::clone(self),
        })
    }
}

pub static CONCURRENCY_MANAGER: LazyLock<Arc<ServerConcurrencyManager>> =
    LazyLock::new(|| Arc::new(ServerConcurrencyManager::new()));

pub fn acquire_server_permit(
    state: &Arc<AppState>,
    port: u16,
    request_id: &str,
) -> Result<ServerPermit, ApiError> {
    let (status, parallel) = {
        let servers = state.servers.lock().unwrap();
        let server = servers.values().find(|s| s.def.port == port);
        match server {
            Some(s) => (s.status, s.def.parallel),
            None => (ServerStatus::Stopped, 0),
        }
    };

    if status == ServerStatus::Starting {
        let mut err = ApiError::service_unavailable(format!(
            "Server on port {port} is still starting. Please retry in a few seconds."
        ));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }
    if status != ServerStatus::Running {
        let mut err = ApiError::service_unavailable(format!(
            "Server on port {port} is not running (status: {status:?})."
        ));
        err.request_id = Some(request_id.to_string());
        return Err(err);
    }

    let limit = if parallel > 0 {
        parallel
    } else {
        DEFAULT_SERVER_CONCURRENCY
    };

    CONCURRENCY_MANAGER.try_acquire(port, limit).map_err(|e| match e {
        ConcurrencyError::LimitExceeded { port, current, limit } => {
            let mut err = ApiError::rate_limited(
                format!("Server on port {port} concurrency limit reached ({current}/{limit} active requests). Please retry after a moment."),
                Some("1".to_string()),
            );
            err.request_id = Some(request_id.to_string());
            err
        }
    })
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", content = "details")]
pub enum SupervisorState {
    Stopped,
    Running {
        port: u16,
    },
    PortConflict {
        port: u16,
        error: String,
    },
    TransientError {
        port: u16,
        error: String,
        retry_in_secs: u64,
    },
}

static SUPERVISOR_STATE: LazyLock<std::sync::Mutex<SupervisorState>> =
    LazyLock::new(|| std::sync::Mutex::new(SupervisorState::Stopped));

pub fn get_supervisor_state() -> SupervisorState {
    SUPERVISOR_STATE.lock().unwrap().clone()
}

pub fn set_supervisor_state(s: SupervisorState) {
    *SUPERVISOR_STATE.lock().unwrap() = s;
}

pub fn is_port_conflict(port: u16) -> bool {
    matches!(*SUPERVISOR_STATE.lock().unwrap(), SupervisorState::PortConflict { port: p, .. } if p == port)
}

pub fn verify_gateway_health(port: u16) -> bool {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) =
        std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(300)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(300)));
    let req =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if std::io::Write::write_all(&mut stream, req.as_bytes()).is_err() {
        return false;
    }
    let mut resp = Vec::new();
    let mut buf = [0u8; 1024];
    while let Ok(n) = std::io::Read::read(&mut stream, &mut buf) {
        if n == 0 {
            break;
        }
        resp.extend_from_slice(&buf[..n]);
        if resp.len() >= 2048 {
            break;
        }
    }
    let resp_str = String::from_utf8_lossy(&resp);
    resp_str.contains("200 OK")
        && (resp_str.contains("llm-panel-gateway")
            || resp_str.contains("x-gateway-service: llm-panel-gateway"))
}

/// Bind the gateway on `port` and serve until aborted. Returns a join handle
/// so the supervisor can stop it by aborting.
pub fn start_gateway(state: Arc<AppState>, port: u16) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = match TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[gateway] failed to bind {port}: {e}");
                return;
            }
        };
        eprintln!("[gateway] OpenAI-compatible gateway listening on 127.0.0.1:{port}");
        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let st = Arc::clone(&state);
                    tokio::spawn(handle_connection(stream, st));
                }
                Err(e) => {
                    eprintln!("[gateway] accept error: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                }
            }
        }
    })
}

/// Poll the persisted config once a second and keep the gateway listening on
/// the requested port & enable state, restarting it whenever those change.
pub fn spawn_supervisor(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut running: Option<(u16, tokio::task::JoinHandle<()>)> = None;
        let mut last_attempted_port: Option<u16> = None;
        let mut transient_backoff_secs: u64 = 1;
        let mut next_retry_at: Option<std::time::Instant> = None;

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let state: tauri::State<Arc<AppState>> = app.state();
            let cfg = state.config();
            let wanted = (cfg.advanced_settings.gateway_enabled
                && cfg
                    .advanced_settings
                    .api_key
                    .as_deref()
                    .map(str::trim)
                    .is_some_and(|key| !key.is_empty()))
            .then_some(cfg.advanced_settings.gateway_port);

            if let Some(port) = wanted {
                // If config changed to a different port, reset conflict and backoff
                if last_attempted_port != Some(port) {
                    last_attempted_port = Some(port);
                    transient_backoff_secs = 1;
                    next_retry_at = None;
                    if let Some((_, h)) = running.take() {
                        h.abort();
                    }
                    set_supervisor_state(SupervisorState::Stopped);
                }

                // If currently running, check if task finished unexpectedly or port changed
                let is_running_stale =
                    matches!(&running, Some((p, h)) if *p != port || h.is_finished());
                if is_running_stale {
                    if let Some((_, h)) = running.take() {
                        h.abort();
                    }
                    set_supervisor_state(SupervisorState::Stopped);
                }

                if running.is_none() {
                    // Check if in PortConflict: do not retry until config change or explicit reset
                    if is_port_conflict(port) {
                        continue;
                    }

                    // Check if in transient backoff delay
                    if let Some(retry_time) = next_retry_at {
                        if std::time::Instant::now() < retry_time {
                            continue;
                        }
                    }

                    // Attempt bind
                    match TcpListener::bind(("127.0.0.1", port)).await {
                        Ok(listener) => {
                            eprintln!(
                                "[gateway] OpenAI-compatible gateway listening on 127.0.0.1:{port}"
                            );
                            set_supervisor_state(SupervisorState::Running { port });
                            transient_backoff_secs = 1;
                            next_retry_at = None;

                            let st = Arc::clone(&state);
                            let handle = tokio::spawn(async move {
                                loop {
                                    match listener.accept().await {
                                        Ok((stream, _peer)) => {
                                            let st_conn = Arc::clone(&st);
                                            tokio::spawn(handle_connection(stream, st_conn));
                                        }
                                        Err(e) => {
                                            eprintln!("[gateway] accept error: {e}");
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                200,
                                            ))
                                            .await;
                                        }
                                    }
                                }
                            });
                            running = Some((port, handle));
                        }
                        Err(e) => {
                            let is_conflict = e.kind() == std::io::ErrorKind::AddrInUse
                                || e.raw_os_error() == Some(10048)
                                || e.raw_os_error() == Some(98);

                            if is_conflict {
                                eprintln!("[gateway] port conflict on {port}: {e}. Halting retries until port or config changes.");
                                set_supervisor_state(SupervisorState::PortConflict {
                                    port,
                                    error: e.to_string(),
                                });
                                next_retry_at = None;
                            } else {
                                eprintln!("[gateway] failed to bind {port}: {e}. Backing off for {transient_backoff_secs}s");
                                set_supervisor_state(SupervisorState::TransientError {
                                    port,
                                    error: e.to_string(),
                                    retry_in_secs: transient_backoff_secs,
                                });
                                next_retry_at = Some(
                                    std::time::Instant::now()
                                        + std::time::Duration::from_secs(transient_backoff_secs),
                                );
                                transient_backoff_secs = (transient_backoff_secs * 2).min(30);
                            }
                        }
                    }
                }
            } else {
                last_attempted_port = None;
                transient_backoff_secs = 1;
                next_retry_at = None;
                if let Some((_, h)) = running.take() {
                    h.abort();
                }
                set_supervisor_state(SupervisorState::Stopped);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{LiveServer, ServerDef, ServerStatus, VecDequeLog};
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    fn live_server(
        id: &str,
        model_id: &str,
        served: Option<&str>,
        port: u16,
        task: &str,
        status: ServerStatus,
    ) -> LiveServer {
        LiveServer {
            def: ServerDef {
                backend: "vllm".into(),
                id: id.to_string(),
                name: id.to_string(),
                model_id: model_id.to_string(),
                task: task.to_string(),
                port,
                gpu_mem_util: 0.92,
                max_model_len: None,
                quant: "fp16".to_string(),
                served_model_name: served.map(|s| s.to_string()),
                kv_cache_dtype: None,
                llamacpp_channel: "upstream".to_string(),
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
                speculative_model: None,
                speculative_model_path: None,
                speculative_num_tokens: None,
                speculative_enabled: false,
                ..ServerDef::default()
            },
            status,
            error: None,
            wsl_child: None,
            native_child: None,
            wsl_pid: None,
            log_ring: Mutex::new(VecDequeLog::new()),
            last_metrics: None,
            stopping: false,
            crash_retry_count: 0,
        }
    }

    fn running_map() -> BTreeMap<String, LiveServer> {
        let mut servers = BTreeMap::new();
        servers.insert(
            "srv-a".to_string(),
            live_server(
                "srv-a",
                "Qwen/Qwen2.5-7B-Instruct",
                Some("qwen-7b"),
                8001,
                "instruct",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-b".to_string(),
            live_server(
                "srv-b",
                "meta-llama/Llama-3.1-8B-Instruct",
                None,
                8002,
                "instruct",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-embed".to_string(),
            live_server(
                "srv-embed",
                "BAAI/bge-small-en",
                None,
                8003,
                "embed",
                ServerStatus::Running,
            ),
        );
        servers.insert(
            "srv-stopped".to_string(),
            live_server(
                "srv-stopped",
                "mistralai/Mistral-7B-Instruct-v0.3",
                None,
                8004,
                "instruct",
                ServerStatus::Stopped,
            ),
        );
        servers
    }

    #[test]
    fn test_route_exact_served_model_name() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "qwen-7b", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "Qwen-7B", "instruct"),
            Some(8001)
        );
    }

    #[test]
    fn test_route_exact_hf_id() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "Qwen/Qwen2.5-7B-Instruct", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "meta-llama/Llama-3.1-8B-Instruct", "instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_fuzzy_suffix_match() {
        let servers = running_map();
        // Client asks for just the tail of the HF id
        assert_eq!(
            find_server_port_for(&servers, "Qwen2.5-7B-Instruct", "instruct"),
            Some(8001)
        );
        assert_eq!(
            find_server_port_for(&servers, "Llama-3.1-8B-Instruct", "instruct"),
            Some(8002)
        );
    }

    #[test]
    fn test_route_ignores_embed_and_stopped_servers() {
        let servers = running_map();
        // Embed server must not be routed for chat
        assert_eq!(
            find_server_port_for(&servers, "BAAI/bge-small-en", "instruct"),
            None
        );
        // Stopped instruct server must not be routed
        assert_eq!(
            find_server_port_for(&servers, "mistralai/Mistral-7B-Instruct-v0.3", "instruct"),
            None
        );
    }

    #[test]
    fn test_route_unknown_or_empty_model() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "nope/model", "instruct"),
            None
        );
        assert_eq!(find_server_port_for(&servers, "", "instruct"), None);
        assert_eq!(find_server_port_for(&servers, "   ", "instruct"), None);
    }

    #[test]
    fn test_route_embed_server_exact_and_fuzzy() {
        let servers = running_map();
        // BAAI/bge-small-en is task=embed, port 8003 in running_map()
        assert_eq!(
            find_server_port_for(&servers, "BAAI/bge-small-en", "embed"),
            Some(8003)
        );
        assert_eq!(
            find_server_port_for(&servers, "bge-small-en", "embed"),
            Some(8003)
        );
    }

    #[test]
    fn test_route_embed_ignores_instruct_servers() {
        let servers = running_map();
        assert_eq!(
            find_server_port_for(&servers, "Qwen/Qwen2.5-7B-Instruct", "embed"),
            None
        );
        assert_eq!(find_server_port_for(&servers, "", "embed"), None);
    }

    #[test]
    fn test_not_found_payload_includes_running_hint() {
        let payload = not_found_payload_with_hint(
            "nope/model",
            "instruction",
            &["qwen-7b".to_string(), "llama-8b".to_string()],
        );
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nope/model"));
        assert_eq!(v["running"][0], "qwen-7b");
    }

    #[test]
    fn test_model_rows_lists_all_running_models_and_excludes_stopped() {
        let servers = running_map();
        let rows = model_rows(&servers);
        // Instruct models come first, then the embedding model; the stopped
        // server must never appear.
        let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "qwen-7b",
                "meta-llama/Llama-3.1-8B-Instruct",
                "BAAI/bge-small-en"
            ]
        );
        assert!(rows
            .iter()
            .all(|r| r["object"] == serde_json::json!("model")));
    }

    #[test]
    fn test_model_rows_include_effective_context_metadata() {
        let mut servers = running_map();
        servers.get_mut("srv-a").unwrap().def.max_model_len = Some(32_768);
        let rows = model_rows(&servers);
        let row = rows.iter().find(|row| row["id"] == "qwen-7b").unwrap();
        for key in [
            "max_model_len",
            "context_length",
            "max_context_length",
            "max_input_tokens",
            "context_window",
            "n_ctx",
            "max_output_tokens",
            "max_completion_tokens",
            "max_tokens",
        ] {
            assert_eq!(row[key], serde_json::json!(32_768));
        }
        // Pi's compat extension reads the top-level `input` array, not
        // architecture.input_modalities, when discovering image support.
        assert_eq!(row["input"], serde_json::json!(["text"]));
        assert_eq!(row["architecture"]["input_modalities"], row["input"]);
        assert_eq!(row["context_source"], "configured");
        assert_eq!(row["max_output_tokens_source"], "context_window");
        assert_eq!(row["backend"], "vllm");
        assert_eq!(row["capabilities"]["chat_completions"], true);
        assert_eq!(row["capabilities"]["system_role"], true);
    }

    #[test]
    fn test_model_rows_report_thinking_capability_and_levels() {
        let mut servers = running_map();
        {
            let def = &mut servers.get_mut("srv-a").unwrap().def;
            def.model_id = "Qwen/Qwen3-8B".into();
            def.served_model_name = Some("qwen3-8b".into());
            def.max_model_len = Some(65_536);
        }
        let rows = model_rows(&servers);
        let row = rows.iter().find(|row| row["id"] == "qwen3-8b").unwrap();
        assert_eq!(row["name"], "srv-a");
        assert_eq!(row["publisher"], "Qwen");
        assert_eq!(row["capabilities"]["reasoning"], true);
        assert_eq!(row["capabilities"]["thinking"], true);
        // OpenRouter object contract and root camelCase compatibility
        assert!(row["reasoning"].is_object());
        assert_eq!(row["supportsReasoning"], true);
        assert_eq!(row["supports_reasoning"], true);
        assert_eq!(row["thinking"]["mode"], "effort");
        assert_eq!(row["reasoning_config"]["supported"], true);
        assert_eq!(row["reasoning_config"]["mode"], "effort");
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "medium");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
        assert_eq!(row["reasoning_config"]["mandatory"], false);
        assert_eq!(row["reasoning_config"]["default_effort"], "medium");
        assert_eq!(row["reasoning_config"]["request_precedence"], "harness");
        assert!(row["reasoning_config"]["controls"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("reasoning_effort")));
        assert!(row["reasoning_config"]["controls"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("thinking_token_budget")));
        // No --reasoning-parser on this vLLM launch, so reasoning_content is
        // not returned even though the model can reason.
        assert_eq!(row["reasoning_config"]["output"], false);
        assert_eq!(row["capabilities"]["reasoning_output"], false);
        assert_eq!(row["capabilities"]["tools"], false);
        assert!(row["configuration_hints"]
            .as_array()
            .is_some_and(|hints| hints.iter().any(|hint| hint
                .as_str()
                .unwrap_or_default()
                .contains("--reasoning-parser"))));
        assert_eq!(row["max_output_tokens"], serde_json::json!(65_536));
    }

    #[test]
    fn test_vllm_launch_args_enable_tools_and_reasoning_output() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = "Qwen/Qwen3-8B".into();
        def.max_model_len = Some(32_768);
        let args: Vec<String> = [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
        ]
        .iter()
        .map(|arg| arg.to_string())
        .collect();
        let row = model_row(&def, Some(32_768), "configured", &args);
        assert_eq!(row["capabilities"]["tools"], true);
        assert_eq!(row["capabilities"]["parallel_tool_calls"], true);
        assert_eq!(row["reasoning_config"]["output"], true);
        assert_eq!(row["reasoning_config"]["source"], "launch_args");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
        assert_eq!(
            row["reasoning_config"]["parameters"]["reasoning_effort"],
            true
        );
        assert_eq!(
            row["reasoning_config"]["parameters"]["chat_template_kwargs"]["enable_thinking"],
            true
        );
        assert!(row["supported_parameters"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("tools")));
        assert!(row.get("configuration_hints").is_none());
    }

    #[test]
    fn test_vllm_max_token_cap_is_reported_as_a_hint() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = "Qwen/Qwen3-8B".into();
        let args: Vec<String> = [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
            "--override-generation-config",
            r#"{"max_new_tokens":1024}"#,
        ]
        .iter()
        .map(|arg| arg.to_string())
        .collect();
        let row = model_row(&def, Some(32_768), "configured", &args);
        assert!(row["configuration_hints"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hint| hint.as_str().unwrap_or_default().contains("max_new_tokens")));
    }

    #[test]
    fn test_gpt_oss_reports_effort_levels() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = "openai/gpt-oss-120b".into();
        def.max_model_len = Some(131_072);
        let args = vec!["--reasoning-parser".to_string(), "openai".to_string()];
        let row = model_row(&def, Some(131_072), "configured", &args);
        assert_eq!(row["reasoning_config"]["mode"], "effort");
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "medium");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
    }

    #[test]
    fn test_bonsai_reports_standard_thinking_metadata() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        let path = r"D:\models\ternary\Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string();
        def.model_path = Some(path.clone());
        def.model_id = path;
        def.served_model_name = None;
        def.jinja = true;
        let row = model_row(&def, Some(32_768), "configured", &[]);
        // llama.cpp exposes the GGUF stem, not the filesystem path.
        assert_eq!(row["id"], "Ternary-Bonsai-2-27B-PQ2_0");
        // OpenRouter object contract and root camelCase compatibility
        assert!(row["reasoning"].is_object());
        assert_eq!(row["supportsReasoning"], true);
        assert_eq!(row["supports_reasoning"], true);
        assert_eq!(row["reasoning_config"]["supported"], true);
        assert_eq!(row["reasoning_config"]["mode"], "effort");
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "medium");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
        assert_eq!(row["reasoning_config"]["mandatory"], false);
    }

    #[test]
    fn test_bonsai_and_openai_models_report_advertised_reasoning_for_harnesses() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        let path = r"D:\models\ternary\Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string();
        def.model_path = Some(path.clone());
        def.model_id = path;
        def.served_model_name = None;
        def.jinja = true;
        let row = model_row(&def, Some(32_768), "configured", &[]);

        // Oh My Pi discovery reads supported_reasoning_levels and default_reasoning_level
        assert_eq!(
            row["supported_reasoning_levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["default_reasoning_level"], "medium");

        // Oh My Pi also accepts them under metadata
        assert_eq!(
            row["metadata"]["supported_reasoning_levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["metadata"]["default_reasoning_level"], "medium");

        // Harness compatibility block for OpenAI/Pi reasoning effort
        assert_eq!(row["compat"]["supportsReasoningEffort"], true);
        assert_eq!(row["compat"]["supportsThinking"], true);
        assert_eq!(row["compat"]["reasoningDisableMode"], "none-effort");

        // Always-on reasoning model (e.g. DeepSeek-R1) does NOT advertise "none"
        let mut r1_def = running_map().remove("srv-a").unwrap().def;
        r1_def.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".into();
        let r1_args = vec!["--reasoning-parser".to_string(), "deepseek_r1".to_string()];
        let r1_row = model_row(&r1_def, None, "unknown", &r1_args);
        assert_eq!(
            r1_row["supported_reasoning_levels"],
            serde_json::json!(["low", "medium", "high"])
        );
        assert_eq!(r1_row["default_reasoning_level"], "medium");
        assert_eq!(r1_row["compat"]["supportsReasoningEffort"], true);
        assert_eq!(r1_row["compat"]["reasoningDisableMode"], serde_json::Value::Null);

        // Non-reasoning model reports empty levels and false reasoning
        let mut base_def = running_map().remove("srv-a").unwrap().def;
        base_def.model_id = "meta-llama/Llama-3.2-3B-Instruct".into();
        base_def.served_model_name = Some("llama-3.2".into());
        let base_row = model_row(&base_def, Some(8192), "configured", &[]);
        assert_eq!(base_row["supported_reasoning_levels"], serde_json::json!([]));
        assert_eq!(base_row["default_reasoning_level"], serde_json::Value::Null);
        assert_eq!(base_row["compat"]["supportsReasoningEffort"], false);
        assert_eq!(base_row["metadata"]["reasoning"], false);

        // Verify harness request shapes: thinking_effort, extra_body, top-level enable_thinking
        let req_extra = r#"{"model":"m","messages":[],"extra_body":{"reasoning_effort":"high"}}"#;
        let norm_extra = normalized(req_extra, "llamacpp", Some(8_192));
        assert_eq!(norm_extra["reasoning_effort"], "high");
        assert_eq!(norm_extra["chat_template_kwargs"]["enable_thinking"], true);

        let req_top_toggle = r#"{"model":"m","messages":[],"enable_thinking":true}"#;
        let norm_toggle = normalized(req_top_toggle, "llamacpp", Some(8_192));
        assert_eq!(norm_toggle["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn test_normalize_chat_request_generic_effort_labels() {
        // vLLM normalizes extended effort labels onto the standard low/medium/high scale
        for (input, expected) in [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ] {
            let body = format!(r#"{{"model":"m","messages":[],"reasoning_effort":"{input}"}}"#);
            let value = normalized(&body, "vllm", Some(8_192));
            assert_eq!(value["reasoning_effort"], expected, "vllm input {input}");
        }

        // llama.cpp forwards extended effort labels to Jinja templates
        for (input, expected) in [
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ] {
            let body = format!(r#"{{"model":"m","messages":[],"reasoning_effort":"{input}"}}"#);
            let value = normalized(&body, "llamacpp", Some(8_192));
            assert_eq!(
                value["reasoning_effort"], expected,
                "llamacpp input {input}"
            );
            assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        }
    }

    #[test]
    fn test_always_on_reasoning_model_has_no_off_level() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".into();
        let args = vec!["--reasoning-parser".to_string(), "deepseek_r1".to_string()];
        let row = model_row(&def, None, "unknown", &args);
        assert_eq!(row["reasoning_config"]["mode"], "always_on");
        assert_eq!(row["reasoning_config"]["mandatory"], true);
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["low", "medium", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "medium");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
    }

    #[test]
    fn test_llamacpp_reports_vision_and_jinja_tools() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.model_id = "Qwen/Qwen3-VL-8B-Thinking-GGUF".into();
        def.max_model_len = None;
        def.ctx_size = Some(131_072);
        def.jinja = true;
        def.mmproj_path = Some("/models/mmproj.gguf".into());
        let row = model_row(&def, Some(131_072), "configured", &[]);
        assert_eq!(row["capabilities"]["vision"], true);
        assert_eq!(
            row["architecture"]["input_modalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(row["capabilities"]["tools"], true);
        assert_eq!(row["reasoning_config"]["supported"], true);
        // llama.cpp parses think tags by default when the template emits them.
        assert_eq!(row["reasoning_config"]["output"], true);
        assert_eq!(row["reasoning_config"]["source"], "model_family");
        assert_eq!(row["reasoning_config"]["request_precedence"], "harness");
        assert_eq!(
            row["reasoning_config"]["parameters"]["reasoning_budget_tokens"],
            true
        );
        assert!(row["reasoning_config"]["controls"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("reasoning_budget_tokens")));
        assert!(row["reasoning_config"]["controls"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("reasoning_format")));
        assert_eq!(row["type"], "vlm");
        assert_eq!(row["compatibility_type"], "gguf");
        assert_eq!(row["context_length"], serde_json::json!(131_072));
    }

    #[test]
    fn test_llamacpp_without_jinja_reports_tools_disabled() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.model_id = "Qwen/Qwen2.5-7B-Instruct-GGUF".into();
        def.jinja = false;
        let row = model_row(&def, None, "unknown", &[]);
        assert_eq!(row["capabilities"]["tools"], false);
        assert!(row["configuration_hints"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hint| hint.as_str().unwrap_or_default().contains("--jinja")));
    }

    #[test]
    fn test_llamacpp_without_jinja_reports_thinking_as_not_overridable() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.model_id = "Qwen/Qwen3-8B-GGUF".into();
        def.jinja = false;
        let row = model_row(&def, None, "unknown", &[]);
        assert_eq!(row["reasoning_config"]["supported"], true);
        assert_eq!(row["reasoning_config"]["enabled_overridable"], false);
        assert_eq!(row["reasoning_config"]["control"], "none");
        assert_eq!(
            row["reasoning_config"]["parameters"]["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert!(!row["reasoning_config"]["controls"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("reasoning_effort")));
    }

    #[test]
    fn test_llamacpp_panel_reasoning_locks_are_request_overridable() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.model_id = "Qwen/Qwen3-8B-GGUF".into();
        def.extra_args = vec![
            "--reasoning".into(),
            "off".into(),
            "--reasoning-format".into(),
            "none".into(),
            "--reasoning-budget".into(),
            "0".into(),
        ];
        let row = model_row(&def, Some(16_384), "configured", &def.extra_args);
        assert_eq!(row["reasoning_config"]["enabled_by_default"], false);
        assert_eq!(row["reasoning_config"]["output"], false);
        assert_eq!(row["reasoning_config"]["enabled_overridable"], true);
        assert_eq!(row["reasoning_config"]["output_overridable"], true);

        // The harness request becomes explicit about every thinking control,
        // so the panel's launch defaults cannot keep thinking off.
        let body = br#"{"model":"qwen3","messages":[],"reasoning_effort":"high"}"#;
        let normalized = normalize_chat_request(body, "llamacpp", Some(16_384));
        let value: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(value["reasoning_format"], "auto");
        assert_eq!(value["reasoning_budget_tokens"], 16_384);
    }

    #[test]
    fn test_embedding_model_reports_embedding_modalities() {
        let servers = running_map();
        let rows = model_rows(&servers);
        let row = rows
            .iter()
            .find(|row| row["id"] == "BAAI/bge-small-en")
            .unwrap();
        assert_eq!(row["type"], "embeddings");
        assert_eq!(row["capabilities"]["embeddings"], true);
        assert_eq!(row["capabilities"]["tools"], false);
        assert_eq!(row["capabilities"]["reasoning"], false);
        assert_eq!(row["supports_embeddings"], true);
        assert_eq!(row["max_output_tokens"], serde_json::Value::Null);
        assert_eq!(
            row["architecture"]["output_modalities"],
            serde_json::json!(["embedding"])
        );
        assert_eq!(row["architecture"]["modality"], "text->embedding");
    }

    #[test]
    fn test_llama_model_rows_use_configured_context_size() {
        let mut servers = running_map();
        let def = &mut servers.get_mut("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.max_model_len = None;
        def.ctx_size = Some(262_144);
        let rows = model_rows(&servers);
        let row = rows.iter().find(|row| row["id"] == "qwen-7b").unwrap();
        assert_eq!(row["max_model_len"], serde_json::json!(262_144));
        assert_eq!(row["context_length"], serde_json::json!(262_144));
        assert_eq!(row["context_source"], "configured");
    }

    #[test]
    fn test_runtime_context_parser_reads_llama_props() {
        let value = serde_json::json!({
            "default_generation_settings": { "n_ctx": "262144" }
        });
        assert_eq!(extract_runtime_context(&value), Some(262_144));
    }

    #[test]
    fn test_model_path_decodes_url_encoded_ids() {
        assert_eq!(
            model_id_from_path("/v1/models/Qwen%2FQwen2.5-7B-Instruct"),
            Some("Qwen/Qwen2.5-7B-Instruct".to_string())
        );
    }

    #[test]
    fn test_model_path_strips_latest_tag() {
        assert_eq!(
            model_id_from_path("/v1/models/llama3.2:latest"),
            Some("llama3.2".to_string())
        );
        assert_eq!(
            model_id_from_path("/v1/models/llama3.2%3Alatest?x=1"),
            Some("llama3.2".to_string())
        );
        // A name that merely contains the tag text must be preserved.
        assert_eq!(
            model_id_from_path("/v1/models/latest-model"),
            Some("latest-model".to_string())
        );
    }

    #[test]
    fn test_model_extracted_from_body() {
        let body = br#"
            {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]}
        "#;
        assert_eq!(model_from_body(body).as_deref(), Some("qwen-7b"));
        assert_eq!(model_from_body(br#"{"msg":"no model"}"#), None);
        assert_eq!(model_from_body(b"not json at all"), None);
    }

    #[test]
    fn test_model_from_body_preserves_fim_suffix() {
        let body = br#"{"model":"qwen-7b","prompt":"def f(","suffix":"):\n pass","stream":false}"#;
        assert_eq!(model_from_body(body).as_deref(), Some("qwen-7b"));
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(v["suffix"], "):\n pass");
    }

    #[test]
    fn test_parse_request_head_extracts_content_length() {
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 1024\r\n\r\n";
        let parsed = parse_head(head).expect("valid head parses");
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.target, "/v1/chat/completions");
        assert_eq!(parsed.content_length, 1024);
        assert_eq!(
            parsed.headers.get("content-type").map(|s| s.as_str()),
            Some("application/json")
        );
    }

    #[test]
    fn test_parse_request_head_accepts_chunked_bodies() {
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        let parsed = parse_head(head).expect("valid chunked head parses");
        assert!(parsed.chunked);
        assert_eq!(parsed.content_length, 0);
    }

    #[test]
    fn test_parse_request_head_rejects_conflicting_framing() {
        let both = b"POST / HTTP/1.1\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert!(parse_head(both).is_none());
        let duplicate = b"POST / HTTP/1.1\r\nContent-Length: 4\r\nContent-Length: 4\r\n\r\n";
        assert!(parse_head(duplicate).is_none());
    }

    #[test]
    fn test_parse_request_head_garbage() {
        assert!(parse_head(b"").is_none());
        assert!(parse_head(b"\r\n\r\n").is_none());
        assert!(parse_head(b"noSpacesAtAll\r\n\r\n").is_none());
    }

    #[test]
    fn test_head_terminator_found() {
        let buf = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\nbody-here";
        assert_eq!(is_head_terminated(buf), Some(32));
        assert_eq!(is_head_terminated(b"GET /v1/models HTTP/1.1\r\n"), None);
    }

    #[test]
    fn test_cors_headers_only_allow_trusted_local_origins() {
        let trusted = cors_headers(Some("http://127.0.0.1:1420"));
        assert!(trusted.contains("Access-Control-Allow-Origin: http://127.0.0.1:1420"));
        assert!(trusted.contains("Vary: Origin"));
        assert!(trusted.contains("Access-Control-Allow-Methods"));
        assert!(trusted.contains("Authorization"));
        assert_eq!(cors_headers(Some("https://evil.example")), "");
        assert_eq!(cors_headers(None), "");
    }

    #[test]
    fn test_check_gateway_auth_no_key_configured() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer anything".to_string());
        let err = check_gateway_auth(&headers, None).unwrap_err();
        assert_eq!(err.0, 503);
        assert!(err.1.contains("not configured"));
        assert_eq!(check_gateway_auth(&headers, Some("")).unwrap_err().0, 503);
    }

    #[test]
    fn test_check_gateway_auth_valid_key() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer secret123".to_string());
        assert!(check_gateway_auth(&headers, Some("secret123")).is_ok());
    }

    #[test]
    fn test_check_gateway_auth_invalid_key() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer wrong".to_string());
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Invalid API key"));
    }

    #[test]
    fn test_check_gateway_auth_missing_header() {
        let headers = HashMap::new();
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Missing Authorization header"));
    }

    #[test]
    fn test_check_gateway_auth_wrong_scheme() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Basic secret123".to_string());
        let err = check_gateway_auth(&headers, Some("secret123")).unwrap_err();
        assert_eq!(err.0, 401);
        assert!(err.1.contains("Bearer scheme"));
    }

    #[test]
    fn test_check_gateway_auth_case_insensitive_header_name() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer secret123".to_string());
        assert!(check_gateway_auth(&headers, Some("secret123")).is_ok());
    }

    fn normalized(body: &str, backend: &str, context: Option<usize>) -> serde_json::Value {
        let bytes = normalize_chat_request(body.as_bytes(), backend, context);
        serde_json::from_slice(&bytes).unwrap()
    }

    fn normalized_with_ladder(
        body: &str,
        backend: &str,
        context: Option<usize>,
        ladder: &[String],
    ) -> serde_json::Value {
        let bytes =
            normalize_chat_request_with_ladder(body.as_bytes(), backend, context, Some(ladder));
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn test_normalize_chat_request_untouched_without_harness_controls() {
        let body = r#"{
            "model": "qwen-7b",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 256,
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "f"}}]
        }"#;
        for backend in ["vllm", "llamacpp"] {
            assert_eq!(
                normalize_chat_request(body.as_bytes(), backend, Some(8_192)),
                body.as_bytes()
            );
        }
        assert_eq!(
            normalize_chat_request(b"not json", "vllm", None),
            b"not json".to_vec()
        );
        assert_eq!(
            normalize_chat_request(br#"["not","an","object"]"#, "vllm", None),
            br#"["not","an","object"]"#.to_vec()
        );
    }

    #[test]
    fn test_normalize_chat_request_reasoning_effort_becomes_explicit_toggle() {
        for backend in ["vllm", "llamacpp"] {
            let value = normalized(
                r#"{"model":"qwen3","messages":[],"reasoning_effort":"high"}"#,
                backend,
                Some(8_192),
            );
            assert_eq!(value["reasoning_effort"], "high");
            assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        }

        let value = normalized(
            r#"{"model":"qwen3","messages":[],"reasoning_effort":"none"}"#,
            "vllm",
            Some(8_192),
        );
        assert_eq!(value["reasoning_effort"], "none");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn test_normalize_chat_request_explicit_kwargs_win_over_effort() {
        let value = normalized(
            r#"{
                "model": "qwen3",
                "messages": [],
                "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": false, "custom": "keep"}
            }"#,
            "llamacpp",
            Some(8_192),
        );
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(value["chat_template_kwargs"]["custom"], "keep");

        // llama.cpp parses reasoning_effort=none after the kwargs, so the
        // gateway drops the contradictory effort when kwargs explicitly
        // enable thinking.
        let value = normalized(
            r#"{
                "model": "qwen3",
                "messages": [],
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": true}
            }"#,
            "llamacpp",
            Some(8_192),
        );
        assert!(value.get("reasoning_effort").is_none());
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn test_normalize_chat_request_accepts_common_harness_shapes() {
        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning":{"effort":"low"}}"#,
            "vllm",
            None,
        );
        assert_eq!(value["reasoning_effort"], "low");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let value = normalized(
            r#"{"model":"m","messages":[],"think":"high"}"#,
            "vllm",
            None,
        );
        assert_eq!(value["reasoning_effort"], "high");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let value = normalized(
            r#"{"model":"m","messages":[],"think":false}"#,
            "llamacpp",
            None,
        );
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);

        let value = normalized(
            r#"{"model":"m","messages":[],"thinking":{"type":"disabled"}}"#,
            "llamacpp",
            None,
        );
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);

        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning":{"enabled":true}}"#,
            "vllm",
            None,
        );
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let value = normalized(
            r#"{"model":"m","messages":[],"chat_template_kwargs":{"enable_thinking":"true"}}"#,
            "vllm",
            None,
        );
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn test_normalize_chat_request_canonicalizes_effort_labels() {
        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"off"}"#,
            "vllm",
            None,
        );
        assert_eq!(value["reasoning_effort"], "none");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);

        let value = normalized(
            r#"{"model":"m","messages":[],"thinking":{"level":"disabled"}}"#,
            "vllm",
            None,
        );
        assert_eq!(value["reasoning_effort"], "none");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);

        // vLLM's reasoning_effort enum does not accept toggle-only labels;
        // the explicit toggle carries the signal instead.
        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"on"}"#,
            "vllm",
            None,
        );
        assert!(value.get("reasoning_effort").is_none());
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"on"}"#,
            "llamacpp",
            None,
        );
        assert!(value.get("reasoning_effort").is_none());
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"auto"}"#,
            "vllm",
            None,
        );
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn test_template_effort_ladder_reads_validation_set() {
        let qwen38 = r#"{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
{%- endif %}"#;
        assert_eq!(
            template_effort_ladder(qwen38),
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "xhigh".to_string()
            ])
        );

        assert_eq!(
            template_effort_ladder("{% if reasoning_effort not in ['low', 'medium', 'high'] %}"),
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string()
            ])
        );

        // Templates that do not validate the label keep the full ladder.
        assert_eq!(
            template_effort_ladder("{% if reasoning_effort == 'high' %}"),
            None
        );
    }

    #[test]
    fn test_llamacpp_clamps_effort_to_template_ladder() {
        let ladder = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];

        // `high` is the standard OpenAI default and the value Pi/Claude Code
        // send; it must never reach a Qwen3.8-style template that rejects it.
        let value = normalized_with_ladder(
            r#"{"model":"m","messages":[],"reasoning_effort":"high"}"#,
            "llamacpp",
            Some(131_072),
            &ladder,
        );
        assert_eq!(value["reasoning_effort"], "xhigh");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        // The copy inside chat_template_kwargs is rounded too.
        let value = normalized_with_ladder(
            r#"{"model":"m","messages":[],"chat_template_kwargs":{"reasoning_effort":"high","custom":true}}"#,
            "llamacpp",
            Some(131_072),
            &ladder,
        );
        assert_eq!(value["chat_template_kwargs"]["reasoning_effort"], "xhigh");
        assert_eq!(value["chat_template_kwargs"]["custom"], true);

        // Levels below the accepted set round up to the lowest one.
        let value = normalized_with_ladder(
            r#"{"model":"m","messages":[],"reasoning_effort":"minimal"}"#,
            "llamacpp",
            Some(131_072),
            &ladder,
        );
        assert_eq!(value["reasoning_effort"], "low");

        // vLLM also clamps to ladder when known.
        let value = normalized_with_ladder(
            r#"{"model":"m","messages":[],"reasoning_effort":"high"}"#,
            "vllm",
            Some(131_072),
            &ladder,
        );
        assert_eq!(value["reasoning_effort"], "xhigh");
    }

    #[test]
    fn test_model_row_reports_template_effort_vocabulary() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = "Ternary-Bonsai-2-27B".into();
        def.backend = "llamacpp".into();
        def.jinja = true;
        def.ctx_size = Some(131_072);
        let ladder = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let row =
            model_row_with_ladder(&def, Some(131_072), "configured", &[], Some(&ladder), true);
        assert_eq!(
            row["reasoning_config"]["supported_efforts"],
            serde_json::json!(["low", "medium", "xhigh"])
        );
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["low", "medium", "xhigh"])
        );
        assert_eq!(
            row["reasoning_config"]["thinking_level_map"]["high"],
            serde_json::Value::Null
        );
        assert_eq!(
            row["reasoning_config"]["thinking_level_map"]["xhigh"],
            "xhigh"
        );
    }

    #[test]
    fn test_normalize_chat_request_maps_budgets_per_backend() {
        let llama = normalized(
            r#"{"model":"m","messages":[],"thinking":{"type":"enabled","budget_tokens":2048}}"#,
            "llamacpp",
            Some(16_384),
        );
        assert_eq!(llama["reasoning_budget_tokens"], 2048);
        assert_eq!(llama["reasoning_format"], "auto");
        assert_eq!(llama["chat_template_kwargs"]["enable_thinking"], true);

        let vllm = normalized(
            r#"{"model":"m","messages":[],"thinking":{"type":"enabled","budget_tokens":2048}}"#,
            "vllm",
            Some(16_384),
        );
        assert_eq!(vllm["thinking_token_budget"], 2048);
        assert!(vllm.get("reasoning_budget_tokens").is_none());
        assert!(vllm.get("reasoning_format").is_none());

        // llama.cpp names, mapped onto vLLM's field.
        let vllm = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"low","reasoning_budget_tokens":512}"#,
            "vllm",
            None,
        );
        assert_eq!(vllm["thinking_token_budget"], 512);

        // vLLM name, mapped onto llama.cpp's field.
        let llama = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"low","thinking_token_budget":512}"#,
            "llamacpp",
            None,
        );
        assert_eq!(llama["reasoning_budget_tokens"], 512);

        // An explicit harness budget is never replaced by the context-window
        // fallback that protects against a panel `--reasoning-budget 0`.
        let llama = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"high","reasoning_budget_tokens":77}"#,
            "llamacpp",
            Some(16_384),
        );
        assert_eq!(llama["reasoning_budget_tokens"], 77);

        // No harness budget: enabling thinking grants the full context so a
        // panel budget cannot cut the requested thinking short.
        let llama = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"high"}"#,
            "llamacpp",
            Some(16_384),
        );
        assert_eq!(llama["reasoning_budget_tokens"], 16_384);
    }

    #[test]
    fn test_normalize_chat_request_preserves_explicit_reasoning_format() {
        let value = normalized(
            r#"{"model":"m","messages":[],"reasoning_effort":"high","reasoning_format":"none"}"#,
            "llamacpp",
            None,
        );
        assert_eq!(value["reasoning_format"], "none");
    }

    #[test]
    fn test_normalize_chat_request_maps_max_token_aliases() {
        for backend in ["vllm", "llamacpp"] {
            let value = normalized(
                r#"{"model":"m","messages":[],"max_completion_tokens":123}"#,
                backend,
                None,
            );
            assert_eq!(value["max_tokens"], 123);

            let value = normalized(
                r#"{"model":"m","messages":[],"max_output_tokens":64}"#,
                backend,
                None,
            );
            assert_eq!(value["max_tokens"], 64);

            // max_completion_tokens takes precedence over legacy max_tokens default
            let value = normalized(
                r#"{"model":"m","messages":[],"max_tokens":10,"max_completion_tokens":123}"#,
                backend,
                None,
            );
            assert_eq!(value["max_tokens"], 123);
        }
    }

    #[test]
    fn test_normalize_chat_request_format_json_to_response_format() {
        let value = normalized(
            r#"{"model":"m","messages":[],"format":"json"}"#,
            "vllm",
            None,
        );
        assert_eq!(value["response_format"]["type"], "json_object");
    }

    #[test]
    fn test_normalize_chat_request_legacy_functions_to_tools() {
        let body = r#"{
            "model": "m",
            "messages": [],
            "functions": [{"name": "get_weather", "description": "weather"}],
            "function_call": "auto"
        }"#;
        let value = normalized(body, "llamacpp", None);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(value["tool_choice"], "auto");

        let body = r#"{
            "model": "m",
            "messages": [],
            "functions": [{"name": "get_weather"}],
            "function_call": {"name": "get_weather"}
        }"#;
        let value = normalized(body, "vllm", None);
        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_normalize_chat_response_mirrors_reasoning_and_usage() {
        let mut resp = serde_json::json!({
            "id": "chat-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning_content": "think"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "completion_tokens_details": {
                    "reasoning_tokens": 15
                }
            }
        });
        normalize_chat_response_json(&mut resp);
        assert_eq!(resp["choices"][0]["message"]["reasoning"], "think");
        assert_eq!(resp["choices"][0]["message"]["reasoning_content"], "think");
        assert_eq!(resp["usage"]["reasoning_tokens"], 15);
        assert_eq!(
            resp["usage"]["completion_tokens_details"]["reasoning_tokens"],
            15
        );

        // Reverse mirror: reasoning -> reasoning_content
        let mut delta_resp = serde_json::json!({
            "id": "chat-2",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning": "thought-stream"
                }
            }],
            "usage": {
                "reasoning_tokens": 7
            }
        });
        normalize_chat_response_json(&mut delta_resp);
        assert_eq!(
            delta_resp["choices"][0]["delta"]["reasoning"],
            "thought-stream"
        );
        assert_eq!(
            delta_resp["choices"][0]["delta"]["reasoning_content"],
            "thought-stream"
        );
        assert_eq!(delta_resp["usage"]["reasoning_tokens"], 7);
        assert_eq!(
            delta_resp["usage"]["completion_tokens_details"]["reasoning_tokens"],
            7
        );
    }

    #[test]
    fn test_normalize_sse_line_mirrors_reasoning() {
        let line = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n";
        let out = normalize_sse_line(line);
        let parsed: serde_json::Value =
            serde_json::from_str(out.strip_prefix("data: ").unwrap().trim()).unwrap();
        assert_eq!(
            parsed["choices"][0]["delta"]["reasoning_content"],
            "thinking..."
        );
        assert_eq!(parsed["choices"][0]["delta"]["reasoning"], "thinking...");

        let done = "data: [DONE]\n\n";
        assert_eq!(normalize_sse_line(done), done);
    }

    #[test]
    fn test_route_single_server_fallback_and_wildcards() {
        let mut servers = BTreeMap::new();
        servers.insert(
            "sole-server".to_string(),
            live_server(
                "sole-server",
                "Qwen/Qwen3-8B",
                Some("qwen3-8b"),
                8000,
                "instruct",
                ServerStatus::Running,
            ),
        );
        assert_eq!(
            find_server_port_for(&servers, "default", "instruct"),
            Some(8000)
        );
        assert_eq!(
            find_server_port_for(&servers, "local", "instruct"),
            Some(8000)
        );
        assert_eq!(
            find_server_port_for(&servers, "auto", "instruct"),
            Some(8000)
        );
        assert_eq!(find_server_port_for(&servers, "*", "instruct"), Some(8000));
        assert_eq!(find_server_port_for(&servers, "", "instruct"), None);
        assert_eq!(
            find_server_port_for(&servers, "qwen3-8b:latest", "instruct"),
            Some(8000)
        );
    }

    #[test]
    fn test_looks_like_vision_model_expanded() {
        assert!(looks_like_vision_model("paligemma-3b"));
        assert!(looks_like_vision_model("smolvlm-500m"));
        assert!(looks_like_vision_model("internvl2-8b"));
        assert!(!looks_like_vision_model("llama-3-8b"));
    }

    #[test]
    fn test_normalize_chat_request_preserves_all_other_harness_fields() {
        let body = r#"{
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 20,
            "seed": 7,
            "stop": ["\n\n"],
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "response_format": {"type": "json_object"}
        }"#;
        let value = normalized(body, "vllm", Some(8_192));
        assert_eq!(value["model"], "qwen3");
        assert_eq!(value["temperature"], 0.3);
        assert_eq!(value["top_p"], 0.9);
        assert_eq!(value["top_k"], 20);
        assert_eq!(value["seed"], 7);
        assert_eq!(value["stream"], true);
        assert_eq!(value["tool_choice"], "auto");
        assert_eq!(value["parallel_tool_calls"], true);
        assert_eq!(value["response_format"]["type"], "json_object");
        assert_eq!(value["messages"][0]["content"], "hi");
        assert_eq!(value["tools"][0]["function"]["name"], "f");
        assert_eq!(value["stop"][0], "\n\n");
    }

    #[test]
    fn test_read_vllm_chat_template_spark() {
        let distro = "local-llm-panel-ubuntu";
        let model_id = "XHToken/Spark-X2.6-4B-INT4";
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.model_id = model_id.into();
        def.backend = "vllm".into();
        let tmpl = match read_vllm_chat_template(distro, &def) {
            Some(t) => t,
            None => {
                // WSL or the test model is not present in the environment (e.g. CI runner)
                return;
            }
        };
        let ladder = template_effort_ladder(&tmpl);
        assert_eq!(ladder, Some(vec!["low".to_string(), "high".to_string()]));
        let row = model_row_with_ladder(
            &def,
            Some(36126),
            "configured",
            &[],
            ladder.as_deref(),
            true,
        );
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["low", "high"])
        );
        assert_eq!(
            row["reasoning_config"]["supported_efforts"],
            serde_json::json!(["low", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "high");
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
        assert_eq!(row["capabilities"]["reasoning"], true);
        assert_eq!(row["max_model_len"], serde_json::json!(36126));
        assert_eq!(row["context_length"], serde_json::json!(36126));
        assert_eq!(row["max_context_length"], serde_json::json!(36126));
        assert_eq!(row["max_input_tokens"], serde_json::json!(36126));
        assert_eq!(row["max_tokens"], serde_json::json!(36126));
        assert_eq!(row["thinkingLevelMap"]["off"], "none");
        assert_eq!(row["thinkingLevelMap"]["low"], "low");
        assert_eq!(row["thinkingLevelMap"]["high"], "high");
        assert_eq!(row["thinkingLevelMap"]["minimal"], serde_json::Value::Null);
        assert_eq!(row["thinkingLevelMap"]["medium"], serde_json::Value::Null);
        assert_eq!(row["thinkingLevelMap"]["xhigh"], serde_json::Value::Null);
        assert_eq!(row["thinkingLevelMap"]["max"], serde_json::Value::Null);
    }

    #[test]
    fn test_spark_effort_medium_clamped_to_high() {
        let ladder = vec!["low".to_string(), "high".to_string()];
        let req =
            r#"{"model":"XHToken/Spark-X2.6-4B-INT4","messages":[],"reasoning_effort":"medium"}"#;
        let value = normalized_with_ladder(req, "vllm", Some(36126), &ladder);
        assert_eq!(value["reasoning_effort"], "high");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);

        let req_with_kwargs = r#"{"model":"Spark-X","messages":[],"chat_template_kwargs":{"reasoning_effort":"medium"}}"#;
        let value2 = normalized_with_ladder(req_with_kwargs, "llamacpp", Some(36126), &ladder);
        assert_eq!(value2["chat_template_kwargs"]["reasoning_effort"], "high");
        assert_eq!(value2["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn test_spark_chat_template_analysis() {
        let template = r#"{%- set enable_thinking = enable_thinking | default(true) %}{%- set reasoning_instructions = '' %}{%- if enable_thinking is true %}{%- set resolved_reasoning_effort = reasoning_effort|default('high') %}{%- if resolved_reasoning_effort not in ('high', 'low') %}{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are high (default) and low.') }}{%- endif %}{%- endif %}"#;
        let info = analyze_chat_template(Some(template));
        assert_eq!(
            info.ladder,
            Some(vec!["low".to_string(), "high".to_string()])
        );
        assert_eq!(info.default_effort, Some("high".to_string()));
        assert!(info.is_reasoning);
        assert!(!info.is_toggle_only);

        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Spark-X2.6-4B-Q4_K_M".into();
        def.model_id = "XHToken/Spark-X2.6-4B-GGUF".into();
        def.backend = "llamacpp".into();
        let row = model_row_with_template(&def, Some(64000), "configured", &[], &info);
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["low", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "high");
        assert_eq!(
            row["reasoning_config"]["supported_efforts"],
            serde_json::json!(["low", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_effort"], "high");
        assert_eq!(row["thinkingLevelMap"]["low"], "low");
        assert_eq!(row["thinkingLevelMap"]["high"], "high");
        assert_eq!(row["thinkingLevelMap"]["medium"], serde_json::Value::Null);
    }

    #[test]
    fn test_deepseek_request_normalizes_to_thinking_enabled_and_strips_effort() {
        let req = r#"{"model":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B","messages":[],"reasoning_effort":"medium"}"#;
        let value = normalized(req, "vllm", Some(32768));
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        assert!(value.get("reasoning_effort").is_none());
        assert!(value["chat_template_kwargs"]
            .get("reasoning_effort")
            .is_none());

        let req_off = r#"{"model":"deepseek-reasoner","messages":[],"reasoning_effort":"none"}"#;
        let value_off = normalized(req_off, "vllm", Some(32768));
        assert_eq!(value_off["thinking"]["type"], "disabled");
        assert_eq!(value_off["chat_template_kwargs"]["enable_thinking"], false);
        assert!(value_off.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_toggle_only_request_normalizes_to_enable_thinking_and_strips_effort() {
        let req = r#"{"model":"my-toggle-model","messages":[],"reasoning_effort":"high"}"#;
        let info = crate::state::TemplateEffortInfo {
            ladder: None,
            default_effort: None,
            is_reasoning: true,
            is_toggle_only: true,
            is_always_on: false,
            template: Some("{% if enable_thinking %}...{% endif %}".into()),
        };
        let bytes =
            normalize_chat_request_with_info(req.as_bytes(), "vllm", Some(32768), Some(&info));
        let obj: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(obj["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(obj["thinking"]["type"], "enabled");
        assert!(obj.get("reasoning_effort").is_none());
        assert!(obj["chat_template_kwargs"]
            .get("reasoning_effort")
            .is_none());
    }

    #[test]
    fn test_extra_args_context_resolution() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.backend = "llamacpp".into();
        def.extra_args = vec!["-c".into(), "16384".into()];
        assert_eq!(configured_context_tokens(&def), Some(16384));
        let (ctx, src) = resolve_model_context_sync(&def, &def.extra_args);
        assert_eq!(ctx, 16384);
        assert_eq!(src, "configured");

        let mut vllm_def = running_map().remove("srv-a").unwrap().def;
        vllm_def.backend = "vllm".into();
        vllm_def.extra_args = vec!["--max-model-len".into(), "8192".into()];
        assert_eq!(configured_context_tokens(&vllm_def), Some(8192));
        let (vllm_ctx, vllm_src) = resolve_model_context_sync(&vllm_def, &vllm_def.extra_args);
        assert_eq!(vllm_ctx, 8192);
        assert_eq!(vllm_src, "configured");
    }

    #[test]
    fn test_family_fallback_context() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.max_model_len = None;
        def.ctx_size = None;
        def.extra_args.clear();

        def.model_id = "XHToken/Spark-X2.6-4B-INT4".into();
        let (ctx_spark, src_spark) = resolve_model_context_sync(&def, &[]);
        assert_eq!(ctx_spark, 36126);
        assert_eq!(src_spark, "architecture");

        def.model_id = "meta-llama/Llama-3.1-8B-Instruct".into();
        let (ctx_llama, src_llama) = resolve_model_context_sync(&def, &[]);
        assert_eq!(ctx_llama, 131072);
        assert_eq!(src_llama, "architecture");
    }

    #[test]
    fn test_normalize_developer_role_to_system() {
        let req = r#"{"model":"qwen3","messages":[{"role":"developer","content":"Be helpful"},{"role":"user","content":"hi"}]}"#;
        let value = normalized(req, "vllm", Some(8192));
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["role"], "user");
    }

    #[test]
    fn test_check_gateway_auth_supports_api_key_and_x_api_key() {
        let mut headers = HashMap::new();
        headers.insert("api-key".to_string(), "my-secret".to_string());
        assert!(check_gateway_auth(&headers, Some("my-secret")).is_ok());

        let mut headers2 = HashMap::new();
        headers2.insert("x-api-key".to_string(), "my-secret".to_string());
        assert!(check_gateway_auth(&headers2, Some("my-secret")).is_ok());

        let mut headers3 = HashMap::new();
        headers3.insert("x-api-key".to_string(), "wrong-secret".to_string());
        assert_eq!(
            check_gateway_auth(&headers3, Some("my-secret"))
                .unwrap_err()
                .0,
            401
        );
    }

    #[test]
    fn test_model_row_includes_permission_field() {
        let servers = running_map();
        let rows = model_rows(&servers);
        let row = rows.iter().find(|r| r["id"] == "qwen-7b").unwrap();
        assert_eq!(row["permission"], serde_json::json!([]));
    }

    #[test]
    fn test_cors_headers_support_sdk_and_auth_headers() {
        let trusted = cors_headers(Some("http://127.0.0.1:3000"));
        assert!(trusted.contains("api-key"));
        assert!(trusted.contains("x-api-key"));
        assert!(trusted.contains("X-Stainless-Lang"));
        assert!(trusted.contains("Access-Control-Max-Age: 86400"));
        assert!(trusted.contains("PUT, DELETE"));
    }

    #[test]
    fn test_allowed_origins_include_portless_localhost() {
        assert!(is_allowed_origin("http://localhost"));
        assert!(is_allowed_origin("http://127.0.0.1"));
        assert!(is_allowed_origin("https://localhost"));
        assert!(is_allowed_origin("https://127.0.0.1"));
        assert!(!is_allowed_origin("http://evil.com"));
    }

    #[test]
    fn test_dynamic_always_on_reasoning_from_template() {
        let template = "{% if message.role == 'assistant' %}<think>\n{{ message.thought }}</think>{{ message.content }}{% endif %}";
        let info = analyze_chat_template(Some(template));
        assert!(info.is_reasoning);
        assert!(info.is_always_on);
        assert!(!info.is_toggle_only);

        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Custom-Thinker-7B".into();
        def.model_id = "custom-org/Custom-Thinker-7B".into();
        let row = model_row_with_template(&def, Some(16384), "configured", &[], &info);
        assert_eq!(row["reasoning_config"]["mode"], "always_on");
        assert_eq!(row["reasoning_config"]["mandatory"], true);
        assert_eq!(row["reasoning_config"]["control"], "reasoning_effort");
        assert_eq!(row["chat_template"], template);
    }

    #[test]
    fn test_dynamic_toggle_only_from_template() {
        let template = "{% if enable_thinking %}<think>{{ message.thought }}</think>{% endif %}{{ message.content }}";
        let info = analyze_chat_template(Some(template));
        assert!(info.is_reasoning);
        assert!(!info.is_always_on);
        assert!(info.is_toggle_only);

        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Custom-Toggle-7B".into();
        def.model_id = "custom-org/Custom-Toggle-7B".into();
        let row = model_row_with_template(&def, Some(16384), "configured", &[], &info);
        assert_eq!(row["reasoning_config"]["control"], "enable_thinking");
        assert_eq!(
            row["reasoning_config"]["parameters"]["reasoning_effort"],
            true
        );
        assert_eq!(
            row["reasoning_config"]["parameters"]["chat_template_kwargs"]["enable_thinking"],
            true
        );
        assert!(row["reasoning"].is_object());
        assert_eq!(row["supportsReasoning"], true);
        assert_eq!(row["supportsReasoningBinary"], true);
        assert_eq!(
            row["supportsReasoningEffort"],
            serde_json::json!(["none", "high"])
        );
        assert_eq!(row["thinkingLevelMap"]["off"], "none");
        assert_eq!(row["thinkingLevelMap"]["high"], "high");
    }

    #[test]
    fn test_template_has_reasoning_markers() {
        assert!(template_has_reasoning("<think>"));
        assert!(template_has_reasoning("</think>"));
        assert!(template_has_reasoning("<thought>"));
        assert!(template_has_reasoning("</thought>"));
        assert!(template_has_reasoning("<|thought|>"));
        assert!(template_has_reasoning("<|endofthought|>"));
        assert!(template_has_reasoning("[think]"));
        assert!(template_has_reasoning("[/think]"));
        assert!(template_has_reasoning("[thought]"));
        assert!(template_has_reasoning("[/thought]"));
        assert!(template_has_reasoning("enable_thinking"));
        assert!(template_has_reasoning("reasoning_content"));
        assert!(template_has_reasoning("reasoning_effort"));
        assert!(template_has_reasoning("thinking_effort"));
        assert!(template_has_reasoning("reasoning_budget"));
        assert!(template_has_reasoning("thinking_token_budget"));
        assert!(!template_has_reasoning("hello standard model"));
    }

    #[test]
    fn test_route_matches_user_set_server_name() {
        let mut servers = BTreeMap::new();
        let mut ls = live_server(
            "s1",
            "C:\\models\\model.gguf",
            None,
            8100,
            "instruct",
            ServerStatus::Running,
        );
        ls.def.name = "My Favorite Coder".into();
        servers.insert("s1".into(), ls);

        assert_eq!(
            find_server_port_for(&servers, "My Favorite Coder", "instruct"),
            Some(8100)
        );
        assert_eq!(
            find_server_port_for(&servers, "my favorite coder", "instruct"),
            Some(8100)
        );
    }

    #[test]
    fn test_api_error_json_structure() {
        let err = ApiError::bad_request(
            "invalid input",
            Some("model".into()),
            Some("missing_required_field".into()),
        )
        .with_request_id("req_123");
        assert_eq!(err.status, 400);
        let val = err.to_json_value();
        assert_eq!(val["error"]["message"], "invalid input");
        assert_eq!(val["error"]["type"], "invalid_request_error");
        assert_eq!(val["error"]["param"], "model");
        assert_eq!(val["error"]["code"], "missing_required_field");

        let auth_err = ApiError::unauthorized("bad key");
        assert_eq!(auth_err.status, 401);
        let auth_val = auth_err.to_json_value();
        assert_eq!(auth_val["error"]["type"], "authentication_error");

        let forbidden_err = ApiError::forbidden("forbidden origin");
        assert_eq!(forbidden_err.status, 403);
        assert_eq!(
            forbidden_err.to_json_value()["error"]["type"],
            "permission_error"
        );

        let not_found_err = ApiError::not_found("not found", None, None);
        assert_eq!(not_found_err.status, 404);
        assert_eq!(
            not_found_err.to_json_value()["error"]["type"],
            "invalid_request_error"
        );

        let method_not_allowed = ApiError::method_not_allowed("GET only");
        assert_eq!(method_not_allowed.status, 405);
        assert_eq!(
            method_not_allowed.to_json_value()["error"]["code"],
            "method_not_allowed"
        );

        let payload_err = ApiError::payload_too_large("exceeds limit");
        assert_eq!(payload_err.status, 413);
        assert_eq!(
            payload_err.to_json_value()["error"]["code"],
            "request_too_large"
        );

        let rate_err = ApiError::rate_limited("slow down", Some("60".into()));
        assert_eq!(rate_err.status, 429);
        assert_eq!(rate_err.retry_after.as_deref(), Some("60"));
        assert_eq!(
            rate_err.to_json_value()["error"]["type"],
            "rate_limit_error"
        );

        let bg_err = ApiError::bad_gateway("backend down");
        assert_eq!(bg_err.status, 502);
        assert_eq!(bg_err.to_json_value()["error"]["type"], "api_error");

        let su_err = ApiError::service_unavailable("gateway not configured");
        assert_eq!(su_err.status, 503);
        assert_eq!(su_err.to_json_value()["error"]["type"], "api_error");
    }

    #[test]
    fn test_request_id_validation_and_generation() {
        assert!(is_valid_request_id("req_abc123"));
        assert!(is_valid_request_id("my-client-req-99"));
        assert!(!is_valid_request_id(""));
        assert!(!is_valid_request_id(&"a".repeat(513)));
        assert!(!is_valid_request_id("non_ascii_test_latin_\u{00FF}"));
        assert!(!is_valid_request_id("has newline\n"));

        let gen = generate_request_id();
        assert!(gen.starts_with("req_"));
        assert!(is_valid_request_id(&gen));

        let mut head = RequestHead {
            method: "POST".into(),
            target: "/v1/chat/completions".into(),
            headers: HashMap::new(),
            content_length: 0,
            chunked: false,
        };
        let req_id1 = extract_or_generate_request_id(&head);
        assert!(req_id1.starts_with("req_"));

        head.headers
            .insert("x-client-request-id".into(), "client-abc".into());
        assert_eq!(extract_or_generate_request_id(&head), "client-abc");

        head.headers
            .insert("x-request-id".into(), "req-server-xyz".into());
        assert_eq!(extract_or_generate_request_id(&head), "req-server-xyz");
    }

    #[test]
    fn test_responses_api_request_translation() {
        let input_text = serde_json::json!({
            "model": "qwen2.5-coder",
            "input": "Write a rust function",
            "instructions": "You are a coding assistant",
            "max_output_tokens": 100,
            "temperature": 0.7,
            "stream": false
        });
        let (model, chat_req, stream) = responses_request_to_chat_completions(&input_text).unwrap();
        assert_eq!(model, "qwen2.5-coder");
        assert!(!stream);
        assert_eq!(chat_req["max_tokens"], 100);
        assert_eq!(chat_req["temperature"], 0.7);
        let msgs = chat_req["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a coding assistant");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Write a rust function");

        let input_msgs = serde_json::json!({
            "model": "qwen2.5-coder",
            "input": [
                { "role": "user", "content": "hello" }
            ],
            "stream": true
        });
        let (_, chat_req_stream, stream_flag) =
            responses_request_to_chat_completions(&input_msgs).unwrap();
        assert!(stream_flag);
        assert_eq!(chat_req_stream["stream"], true);
        assert_eq!(chat_req_stream["stream_options"]["include_usage"], true);

        let prev_resp = serde_json::json!({
            "model": "qwen2.5-coder",
            "input": "test",
            "previous_response_id": "resp_123"
        });
        assert_eq!(
            responses_request_to_chat_completions(&prev_resp)
                .unwrap_err()
                .status,
            400
        );

        let store_true = serde_json::json!({
            "model": "qwen2.5-coder",
            "input": "test",
            "store": true
        });
        assert_eq!(
            responses_request_to_chat_completions(&store_true)
                .unwrap_err()
                .status,
            400
        );
    }

    #[test]
    fn test_chat_completion_to_responses_mapping() {
        let chat_resp = serde_json::json!({
            "id": "chatcmpl-test456",
            "model": "qwen2.5-coder",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello world!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17
            }
        });

        let resp = chat_completion_to_responses(&chat_resp, "qwen2.5-coder", 1700000000);
        assert_eq!(resp["id"], "resp_test456");
        assert_eq!(resp["object"], "response");
        assert_eq!(resp["created_at"], 1700000000);
        assert_eq!(resp["status"], "completed");
        assert_eq!(resp["model"], "qwen2.5-coder");
        assert_eq!(resp["output_text"], "Hello world!");
        assert_eq!(resp["usage"]["input_tokens"], 12);
        assert_eq!(resp["usage"]["output_tokens"], 5);
        assert_eq!(resp["usage"]["total_tokens"], 17);
        assert_eq!(resp["incomplete_details"], serde_json::Value::Null);

        let output_arr = resp["output"].as_array().unwrap();
        assert_eq!(output_arr.len(), 1);
        assert_eq!(output_arr[0]["type"], "message");
        assert_eq!(output_arr[0]["role"], "assistant");
        assert_eq!(output_arr[0]["content"][0]["type"], "output_text");
        assert_eq!(output_arr[0]["content"][0]["text"], "Hello world!");

        let refusal_resp = serde_json::json!({
            "id": "chatcmpl-refusal",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "refusal": "I cannot do that."
                    },
                    "finish_reason": "content_filter"
                }
            ]
        });
        let resp_refusal = chat_completion_to_responses(&refusal_resp, "qwen", 1700000000);
        assert_eq!(resp_refusal["status"], "incomplete");
        assert_eq!(
            resp_refusal["incomplete_details"]["reason"],
            "content_filter"
        );
        assert_eq!(resp_refusal["output"][0]["content"][0]["type"], "refusal");
        assert_eq!(
            resp_refusal["output"][0]["content"][0]["refusal"],
            "I cannot do that."
        );

        let tool_resp = serde_json::json!({
            "id": "chatcmpl-tools",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":\"Boston\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        });
        let resp_tools = chat_completion_to_responses(&tool_resp, "qwen", 1700000000);
        assert_eq!(resp_tools["status"], "completed");
        assert_eq!(resp_tools["output"][0]["type"], "function_call");
        assert_eq!(resp_tools["output"][0]["name"], "get_weather");
        assert_eq!(resp_tools["output"][0]["call_id"], "call_abc");
    }

    #[test]
    fn test_embeddings_input_validation_and_normalization() {
        assert!(validate_embedding_input(&serde_json::json!("hello")).is_ok());
        assert!(validate_embedding_input(&serde_json::json!(["hello", "world"])).is_ok());
        assert!(validate_embedding_input(&serde_json::json!([1, 2, 3])).is_ok());
        assert!(validate_embedding_input(&serde_json::json!([[1, 2], [3, 4]])).is_ok());

        assert!(validate_embedding_input(&serde_json::json!(123)).is_err());
        assert!(validate_embedding_input(&serde_json::json!([])).is_err());
        assert!(validate_embedding_input(&serde_json::json!([true])).is_err());

        let raw_upstream = serde_json::json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [1.0, 2.0, 3.0, 4.0]
                }
            ],
            "model": "bge-small",
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 4
            }
        });

        let upstream_bytes = serde_json::to_vec(&raw_upstream).unwrap();
        let norm_dim_bytes = normalize_embedding_response(&upstream_bytes, None, Some(2)).unwrap();
        let norm_dim: serde_json::Value = serde_json::from_slice(&norm_dim_bytes).unwrap();
        let emb_dim = norm_dim["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb_dim.len(), 2);
        assert_eq!(emb_dim[0], 1.0);
        assert_eq!(emb_dim[1], 2.0);

        let norm_b64_bytes =
            normalize_embedding_response(&upstream_bytes, Some("base64"), None).unwrap();
        let norm_b64: serde_json::Value = serde_json::from_slice(&norm_b64_bytes).unwrap();
        let b64_str = norm_b64["data"][0]["embedding"].as_str().unwrap();
        assert!(!b64_str.is_empty());
        assert_eq!(b64_str.len(), 24);
    }

    #[test]
    fn test_http_head_parsing_and_limits() {
        let valid = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 42\r\n\r\n";
        let parsed = parse_head_result(valid).unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.target, "/v1/chat/completions");
        assert_eq!(parsed.content_length, 42);
        assert!(!parsed.chunked);

        let dup = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\nContent-Length: 10\r\n\r\n";
        let err_dup = parse_head_result(dup).unwrap_err();
        assert_eq!(err_dup.status, 400);
        assert_eq!(err_dup.code.as_deref(), Some("duplicate_header"));

        let conflict = b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 10\r\n\r\n";
        let err_conflict = parse_head_result(conflict).unwrap_err();
        assert_eq!(err_conflict.status, 400);
        assert_eq!(err_conflict.code.as_deref(), Some("conflicting_framing"));

        let bad_te = b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n";
        let err_bad_te = parse_head_result(bad_te).unwrap_err();
        assert_eq!(err_bad_te.status, 400);

        let bad_line = b"GARBAGE_NO_SPACES\r\n\r\n";
        assert_eq!(parse_head_result(bad_line).unwrap_err().status, 400);
    }

    #[tokio::test]
    async fn test_mock_upstream_gateway_integration() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mock_upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_port = mock_upstream.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = mock_upstream.accept().await {
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    let resp_body = serde_json::json!({
                        "id": "chatcmpl-mock1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "mock-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "mock response from local upstream"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 6,
                            "total_tokens": 14
                        }
                    })
                    .to_string();
                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = socket.write_all(http_resp.as_bytes()).await;
                    let _ = socket.flush().await;
                }
            }
        });

        let mut servers = BTreeMap::new();
        servers.insert(
            "mock-srv".into(),
            live_server(
                "mock-srv",
                "mock-model",
                None,
                upstream_port,
                "instruct",
                ServerStatus::Running,
            ),
        );

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key-123".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let (stream, _) = gateway_listener.accept().await.unwrap();
            handle_connection(stream, state_clone).await;
        });

        let req_payload = serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}]
        })
        .to_string();
        let req_data = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key-123\r\nContent-Type: application/json\r\nX-Client-Request-Id: test-client-req-01\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_payload.len(),
            req_payload
        );

        let mut client_stream = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client_stream.write_all(req_data.as_bytes()).await.unwrap();

        let mut resp_buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = client_stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            resp_buf.extend_from_slice(&tmp[..n]);
        }
        let resp_str = String::from_utf8_lossy(&resp_buf);
        eprintln!("GATEWAY RESPONSE:\n{resp_str}");
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("x-request-id: test-client-req-01"));
        assert!(resp_str.contains("mock response from local upstream"));
    }

    #[test]
    fn test_route_server_exact_match_wins_over_fuzzy() {
        let mut servers = BTreeMap::new();
        let mut s1 = live_server(
            "srv1",
            "qwen2.5-7b",
            Some("qwen"),
            8001,
            "instruct",
            ServerStatus::Running,
        );
        s1.def.max_model_len = Some(4096);
        let mut s2 = live_server(
            "srv2",
            "qwen-7b",
            None,
            8002,
            "instruct",
            ServerStatus::Running,
        );
        s2.def.max_model_len = Some(4096);
        servers.insert("srv1".into(), s1);
        servers.insert("srv2".into(), s2);

        // "qwen" is an exact match for srv1 (served_model_name) and a prefix/fuzzy match for srv2 ("qwen-7b").
        // Exact match MUST win over fuzzy.
        let port = route_server_for(&servers, "qwen", "instruct").unwrap();
        assert_eq!(port, 8001);
    }

    #[test]
    fn test_route_server_multiple_fuzzy_matches_returns_ambiguous_error() {
        let mut servers = BTreeMap::new();
        let mut s1 = live_server(
            "srv1",
            "Qwen/Qwen2.5-7B-Instruct",
            Some("qwen2.5-7b"),
            8001,
            "instruct",
            ServerStatus::Running,
        );
        s1.def.max_model_len = Some(4096);
        let mut s2 = live_server(
            "srv2",
            "Qwen/Qwen2.5-14B-Instruct",
            Some("qwen2.5-14b"),
            8002,
            "instruct",
            ServerStatus::Running,
        );
        s2.def.max_model_len = Some(4096);
        servers.insert("srv1".into(), s1);
        servers.insert("srv2".into(), s2);

        // "qwen" matches both srv1 and srv2 fuzzily.
        let err = route_server_for(&servers, "qwen", "instruct").unwrap_err();
        match err {
            ModelRouteError::Ambiguous {
                requested,
                candidates,
            } => {
                assert_eq!(requested, "qwen");
                assert_eq!(candidates, vec!["qwen2.5-14b", "qwen2.5-7b"]);
            }
            _ => panic!("Expected Ambiguous error, got {err:?}"),
        }

        // Wildcard with multiple servers must also return Ambiguous
        let wildcard_err = route_server_for(&servers, "*", "instruct").unwrap_err();
        assert!(matches!(wildcard_err, ModelRouteError::Ambiguous { .. }));
    }

    #[test]
    fn test_concurrency_policy_limit_enforced() {
        let manager = Arc::new(ServerConcurrencyManager::new());
        let port = 9050;
        assert_eq!(manager.current_inflight(port), 0);

        let permit1 = manager
            .try_acquire(port, 2)
            .expect("first acquire succeeds");
        assert_eq!(manager.current_inflight(port), 1);

        let permit2 = manager
            .try_acquire(port, 2)
            .expect("second acquire succeeds");
        assert_eq!(manager.current_inflight(port), 2);

        // Third acquire exceeds limit 2
        let err = manager.try_acquire(port, 2);
        assert!(matches!(
            err,
            Err(ConcurrencyError::LimitExceeded {
                port: 9050,
                current: 2,
                limit: 2
            })
        ));

        // Dropping one permit frees a slot
        drop(permit1);
        assert_eq!(manager.current_inflight(port), 1);

        let permit3 = manager
            .try_acquire(port, 2)
            .expect("third acquire succeeds after drop");
        assert_eq!(manager.current_inflight(port), 2);
        drop(permit2);
        drop(permit3);
        assert_eq!(manager.current_inflight(port), 0);
    }

    #[test]
    fn test_port_conflict_state_and_identity_verification() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // 1. Verify foreign service fails identity check (on dedicated thread)
        let foreign_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let foreign_port = foreign_listener.local_addr().unwrap().port();
        let handle1 = std::thread::spawn(move || {
            if let Ok((mut socket, _)) = foreign_listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf);
                let resp = "HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
                let _ = socket.write_all(resp.as_bytes());
            }
        });
        assert!(!verify_gateway_health(foreign_port));
        let _ = handle1.join();

        // 2. Verify authentic gateway passes identity check
        let gw_mock_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let gw_mock_port = gw_mock_listener.local_addr().unwrap().port();
        let handle2 = std::thread::spawn(move || {
            if let Ok((mut socket, _)) = gw_mock_listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf);
                let body = r#"{"status":"ok","service":"llm-panel-gateway"}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Gateway-Service: llm-panel-gateway\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(resp.as_bytes());
            }
        });
        assert!(verify_gateway_health(gw_mock_port));
        let _ = handle2.join();

        // 3. Verify supervisor PortConflict state tracking
        set_supervisor_state(SupervisorState::PortConflict {
            port: 11434,
            error: "Address already in use".into(),
        });
        assert!(is_port_conflict(11434));
        assert!(!is_port_conflict(11435));
        set_supervisor_state(SupervisorState::Stopped);
        assert!(!is_port_conflict(11434));
    }

    #[tokio::test]
    async fn test_conformance_streaming_chat() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mock_upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_port = mock_upstream.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = mock_upstream.accept().await {
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    let chunk1 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n";
                    let chunk2 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n";
                    let chunk3 = "data: [DONE]\n\n";
                    let sse_stream = format!("{chunk1}{chunk2}{chunk3}");
                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                        sse_stream.len(),
                        sse_stream
                    );
                    let _ = socket.write_all(http_resp.as_bytes()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                }
            }
        });

        let mut servers = BTreeMap::new();
        let mut srv = live_server(
            "stream-srv",
            "stream-model",
            None,
            upstream_port,
            "instruct",
            ServerStatus::Running,
        );
        srv.def.max_model_len = Some(4096);
        servers.insert("stream-srv".into(), srv);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let (stream, _) = gateway_listener.accept().await.unwrap();
            handle_connection(stream, state_clone).await;
        });

        let req_body = serde_json::json!({
            "model": "stream-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })
        .to_string();

        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_body.len(),
            req_body
        );

        let mut client = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client.write_all(req.as_bytes()).await.unwrap();

        let mut resp_buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = client.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            resp_buf.extend_from_slice(&tmp[..n]);
        }
        let resp_str = String::from_utf8_lossy(&resp_buf);
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("text/event-stream"));
        assert!(resp_str.contains("Hello"));
        assert!(resp_str.contains("world"));
        assert!(resp_str.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_conformance_tool_calling_and_structured_output() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mock_upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_port = mock_upstream.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = mock_upstream.accept().await {
                    let mut buf = [0u8; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    let _req_text = String::from_utf8_lossy(&buf[..n]);

                    let resp_body = serde_json::json!({
                        "id": "chatcmpl-tools",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "tool-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [
                                        {
                                            "id": "call_weather123",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": "{\"location\": \"Boston, MA\"}"
                                            }
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls"
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
                    })
                    .to_string();

                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = socket.write_all(http_resp.as_bytes()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                }
            }
        });

        let mut servers = BTreeMap::new();
        let mut srv = live_server(
            "tool-srv",
            "tool-model",
            None,
            upstream_port,
            "instruct",
            ServerStatus::Running,
        );
        srv.def.max_model_len = Some(4096);
        servers.insert("tool-srv".into(), srv);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let (stream, _) = gateway_listener.accept().await.unwrap();
            handle_connection(stream, state_clone).await;
        });

        let req_body = serde_json::json!({
            "model": "tool-model",
            "messages": [{"role": "user", "content": "what is the weather in Boston?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "response_format": {"type": "json_object"}
        })
        .to_string();

        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_body.len(),
            req_body
        );

        let mut client = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client.write_all(req.as_bytes()).await.unwrap();

        let mut resp_buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = client.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            resp_buf.extend_from_slice(&tmp[..n]);
        }
        let resp_str = String::from_utf8_lossy(&resp_buf);
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("call_weather123"));
        assert!(resp_str.contains("get_weather"));
        assert!(resp_str.contains("Boston, MA"));
    }

    #[tokio::test]
    async fn test_conformance_system_prompt_normalization() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mock_upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_port = mock_upstream.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = mock_upstream.accept().await {
                    let mut buf = [0u8; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    let req_text = String::from_utf8_lossy(&buf[..n]);
                    // Verify developer role was normalized to system
                    assert!(req_text.contains(r#"role":"system"#));
                    assert!(!req_text.contains(r#"role":"developer"#));

                    let resp_body = serde_json::json!({
                        "id": "chatcmpl-sys",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "sys-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "I am system-instructed."},
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
                    })
                    .to_string();

                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = socket.write_all(http_resp.as_bytes()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                }
            }
        });

        let mut servers = BTreeMap::new();
        let mut srv = live_server(
            "sys-srv",
            "sys-model",
            None,
            upstream_port,
            "instruct",
            ServerStatus::Running,
        );
        srv.def.max_model_len = Some(4096);
        servers.insert("sys-srv".into(), srv);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let (stream, _) = gateway_listener.accept().await.unwrap();
            handle_connection(stream, state_clone).await;
        });

        let req_body = serde_json::json!({
            "model": "sys-model",
            "messages": [
                {"role": "developer", "content": "Follow system directives precisely."},
                {"role": "user", "content": "Hello"}
            ]
        })
        .to_string();

        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_body.len(),
            req_body
        );

        let mut client = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client.write_all(req.as_bytes()).await.unwrap();

        let mut resp_buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = client.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            resp_buf.extend_from_slice(&tmp[..n]);
        }
        let resp_str = String::from_utf8_lossy(&resp_buf);
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"));
        assert!(resp_str.contains("I am system-instructed."));
    }

    #[tokio::test]
    async fn test_conformance_responses_tool_call_streaming_lifecycle() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mock_upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_port = mock_upstream.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = mock_upstream.accept().await {
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;

                    // Stream tool call chunks from upstream chat completions
                    let c1 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_calc_99\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"\"}}]}}]}\n\n";
                    let c2 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"expr\\\":\\\"40 + 2\\\"}\"}}]}}]}\n\n";
                    let c3 = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";
                    let c4 = "data: [DONE]\n\n";

                    let sse_stream = format!("{c1}{c2}{c3}{c4}");
                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                        sse_stream.len(),
                        sse_stream
                    );
                    let _ = socket.write_all(http_resp.as_bytes()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                }
            }
        });

        let mut servers = BTreeMap::new();
        let mut srv = live_server(
            "resp-srv",
            "resp-model",
            None,
            upstream_port,
            "instruct",
            ServerStatus::Running,
        );
        srv.def.max_model_len = Some(4096);
        servers.insert("resp-srv".into(), srv);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let (stream, _) = gateway_listener.accept().await.unwrap();
            handle_connection(stream, state_clone).await;
        });

        let req_body = serde_json::json!({
            "model": "resp-model",
            "input": "compute 40 + 2",
            "tools": [{"type": "function", "name": "calculate", "description": "calc"}],
            "stream": true
        })
        .to_string();

        let req = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_body.len(),
            req_body
        );

        let mut client = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client.write_all(req.as_bytes()).await.unwrap();

        let mut resp_buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = client.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            resp_buf.extend_from_slice(&tmp[..n]);
        }
        let resp_str = String::from_utf8_lossy(&resp_buf);

        // Verify Responses API tool call streaming event lifecycle:
        assert!(resp_str.contains("event: response.created"));
        assert!(resp_str.contains("event: response.output_item.added"));
        assert!(resp_str.contains(r#""type":"function_call""#));
        assert!(resp_str.contains("call_calc_99"));
        assert!(resp_str.contains("calculate"));
        assert!(resp_str.contains("event: response.function_call_arguments.delta"));
        assert!(resp_str.contains("event: response.function_call_arguments.done"));
        assert!(resp_str.contains("event: response.output_item.done"));
        assert!(resp_str.contains("event: response.completed"));
        assert!(resp_str.contains("40 + 2"));
        assert!(resp_str.contains("expr"));
        assert!(resp_str.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_conformance_auth_rejection() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        state.config.lock().unwrap().advanced_settings.api_key = Some("correct-secret".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = gateway_listener.accept().await {
                    let st = Arc::clone(&state_clone);
                    tokio::spawn(handle_connection(stream, st));
                }
            }
        });

        // 1. Missing Authorization header
        let req1 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let mut client1 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client1.write_all(req1.as_bytes()).await.unwrap();
        let mut resp1 = String::new();
        let mut buf = [0u8; 1024];
        while let Ok(n) = client1.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp1.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp1.starts_with("HTTP/1.1 401 Unauthorized"));
        assert!(resp1.contains("Missing Authorization header"));

        // 2. Invalid API key
        let req2 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer wrong-key\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let mut client2 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client2.write_all(req2.as_bytes()).await.unwrap();
        let mut resp2 = String::new();
        while let Ok(n) = client2.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp2.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp2.starts_with("HTTP/1.1 401 Unauthorized"));
        assert!(resp2.contains("Invalid API key"));

        // 3. Wrong authorization scheme (Basic instead of Bearer)
        let req3 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Basic dXNlcjpwYXNz\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let mut client3 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client3.write_all(req3.as_bytes()).await.unwrap();
        let mut resp3 = String::new();
        while let Ok(n) = client3.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp3.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp3.starts_with("HTTP/1.1 401 Unauthorized"));
        assert!(resp3.contains("Authorization header must use Bearer scheme"));

        // 4. Gateway key not configured at all in settings -> 503
        state.config.lock().unwrap().advanced_settings.api_key = None;
        let req4 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer some-key\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let mut client4 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client4.write_all(req4.as_bytes()).await.unwrap();
        let mut resp4 = String::new();
        while let Ok(n) = client4.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp4.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp4.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(resp4.contains("Gateway API key is not configured"));
    }

    #[tokio::test]
    async fn test_conformance_cors_handling() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = gateway_listener.accept().await {
                    let st = Arc::clone(&state_clone);
                    tokio::spawn(handle_connection(stream, st));
                }
            }
        });

        // 1. OPTIONS preflight with allowed local origin
        let preflight = format!(
            "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nOrigin: http://localhost:3000\r\nConnection: close\r\n\r\n"
        );
        let mut client1 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client1.write_all(preflight.as_bytes()).await.unwrap();
        let mut resp1 = String::new();
        let mut buf = [0u8; 1024];
        while let Ok(n) = client1.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp1.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp1.starts_with("HTTP/1.1 204 No Content"));
        assert!(resp1
            .to_lowercase()
            .contains("access-control-allow-origin: http://localhost:3000"));
        assert!(resp1
            .to_lowercase()
            .contains("access-control-allow-methods: get, post, options"));
        assert!(resp1.to_lowercase().contains("authorization"));

        // 2. OPTIONS preflight with untrusted external origin
        let untrusted = format!(
            "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nOrigin: https://evil.attacker.com\r\nConnection: close\r\n\r\n"
        );
        let mut client2 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client2.write_all(untrusted.as_bytes()).await.unwrap();
        let mut resp2 = String::new();
        while let Ok(n) = client2.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp2.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp2.starts_with("HTTP/1.1 403 Forbidden"));
        assert!(resp2.contains("origin not allowed"));
        assert!(!resp2.to_lowercase().contains("access-control-allow-origin"));
    }

    #[tokio::test]
    async fn test_conformance_model_routing_and_concurrency_http_errors() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mut servers = BTreeMap::new();
        let mut s_a = live_server(
            "srv-a",
            "Qwen/Qwen2.5-7B",
            Some("qwen2.5-7b"),
            8010,
            "instruct",
            ServerStatus::Running,
        );
        s_a.def.max_model_len = Some(4096);
        let mut s_b = live_server(
            "srv-b",
            "Qwen/Qwen2.5-14B",
            Some("qwen2.5-14b"),
            8011,
            "instruct",
            ServerStatus::Running,
        );
        s_b.def.max_model_len = Some(4096);
        let mut s_starting = live_server(
            "srv-starting",
            "meta-llama/Llama-3-8B",
            Some("llama-3-8b"),
            8012,
            "instruct",
            ServerStatus::Starting,
        );
        s_starting.def.max_model_len = Some(4096);

        servers.insert("srv-a".into(), s_a);
        servers.insert("srv-b".into(), s_b);
        servers.insert("srv-starting".into(), s_starting);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state.config.lock().unwrap().advanced_settings.api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = gateway_listener.accept().await {
                    let st = Arc::clone(&state_clone);
                    tokio::spawn(handle_connection(stream, st));
                }
            }
        });

        // 1. Ambiguous fuzzy model request returns HTTP 400 Bad Request with candidate list
        let req_ambiguous = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}]
        })
        .to_string();
        let req1 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_ambiguous.len(),
            req_ambiguous
        );
        let mut client1 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client1.write_all(req1.as_bytes()).await.unwrap();
        let mut resp1 = String::new();
        let mut buf = [0u8; 1024];
        while let Ok(n) = client1.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp1.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp1.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(resp1.contains("ambiguous_model"));
        assert!(resp1.contains("qwen2.5-7b"));
        assert!(resp1.contains("qwen2.5-14b"));

        // 2. Starting server returns HTTP 503 Service Unavailable
        let req_starting = serde_json::json!({
            "model": "llama-3-8b",
            "messages": [{"role": "user", "content": "hi"}]
        })
        .to_string();
        let req2 = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            req_starting.len(),
            req_starting
        );
        let mut client2 = TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client2.write_all(req2.as_bytes()).await.unwrap();
        let mut resp2 = String::new();
        while let Ok(n) = client2.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp2.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp2.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(resp2.contains("is still starting"));
    }

    #[test]
    fn test_root_level_multi_harness_compatibility_schema() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Spark-X2.6-4B-Q4_K_M".into();
        def.model_id = "XHToken/Spark-X2.6-4B-GGUF".into();
        def.backend = "llamacpp".into();
        def.jinja = true;

        let ladder = vec!["low".to_string(), "high".to_string()];
        let info = crate::state::TemplateEffortInfo {
            ladder: Some(ladder.clone()),
            default_effort: Some("high".to_string()),
            is_reasoning: true,
            is_toggle_only: false,
            is_always_on: false,
            template: None,
        };

        let row = model_row_with_template(&def, Some(32768), "configured", &[], &info);

        // 1. Roo Code / Cline root-level camelCase contracts
        assert_eq!(row["supportsReasoning"], serde_json::json!(true));
        assert_eq!(
            row["supportsReasoningEffort"],
            serde_json::json!(["none", "low", "high"])
        );

        // 2. OpenRouter object contract at row["reasoning"]
        assert!(row["reasoning"].is_object());
        assert_eq!(
            row["reasoning"]["supported_efforts"],
            serde_json::json!(["none", "low", "high"])
        );
        assert_eq!(row["reasoning"]["default_effort"], "high");

        // 3. Backward compatibility preservation
        assert_eq!(row["capabilities"]["reasoning"], true);
        assert_eq!(row["compat"]["supportsReasoningEffort"], true);
    }

    #[test]
    fn test_toggle_only_model_truthful_advertising() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Qwen2.5-Coder-32B-Instruct".into();
        def.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF".into();
        def.backend = "llamacpp".into();
        def.jinja = true;

        let info = crate::state::TemplateEffortInfo {
            ladder: None,
            default_effort: None,
            is_reasoning: true,
            is_toggle_only: true,
            is_always_on: false,
            template: Some("{% if enable_thinking %}...{% endif %}".into()),
        };

        let row = model_row_with_template(&def, Some(32768), "configured", &[], &info);

        // 1. Binary indicator
        assert_eq!(row["supportsReasoningBinary"], serde_json::json!(true));

        // 2. Truthful Off vs On effort tiers
        assert_eq!(
            row["supportsReasoningEffort"],
            serde_json::json!(["none", "high"])
        );
        assert_eq!(
            row["reasoning"]["supported_efforts"],
            serde_json::json!(["none", "high"])
        );

        // 3. Parameters include reasoning, include_reasoning, and reasoning_effort
        let params: Vec<&str> = row["supported_parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        assert!(params.contains(&"reasoning"));
        assert!(params.contains(&"include_reasoning"));
        assert!(params.contains(&"reasoning_effort"));
    }

    #[test]
    fn test_sparse_ladder_and_pi_normalization() {
        let mut def = running_map().remove("srv-a").unwrap().def;
        def.name = "Spark-X2.6-4B-Q4_K_M".into();
        def.model_id = "XHToken/Spark-X2.6-4B-GGUF".into();
        def.backend = "llamacpp".into();
        def.jinja = true;

        let ladder = vec!["low".to_string(), "high".to_string()];
        let info = crate::state::TemplateEffortInfo {
            ladder: Some(ladder),
            default_effort: Some("high".to_string()),
            is_reasoning: true,
            is_toggle_only: false,
            is_always_on: false,
            template: None,
        };

        let row = model_row_with_template(&def, Some(32768), "configured", &[], &info);

        let map = &row["thinkingLevelMap"];
        // 1. Both off and none keys populated
        assert_eq!(map["off"], "none");
        assert_eq!(map["none"], "none");

        // 2. Exact supported tiers present
        assert_eq!(map["low"], "low");
        assert_eq!(map["high"], "high");

        // 3. Truthful nulls for unsupported tiers
        assert_eq!(map["minimal"], serde_json::Value::Null);
        assert_eq!(map["medium"], serde_json::Value::Null);
        assert_eq!(map["xhigh"], serde_json::Value::Null);
        assert_eq!(map["max"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn test_models_route_alias() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let mut servers = BTreeMap::new();
        let mut s_a = live_server(
            "srv-a",
            "Qwen/Qwen2.5-7B",
            Some("qwen2.5-7b"),
            8010,
            "instruct",
            ServerStatus::Running,
        );
        s_a.def.max_model_len = Some(4096);
        servers.insert("srv-a".into(), s_a);

        let gateway_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway_port = gateway_listener.local_addr().unwrap().port();

        let state = Arc::new(AppState::new());
        *state.servers.lock().unwrap() = servers;
        state
            .config
            .lock()
            .unwrap()
            .advanced_settings
            .api_key = Some("test-key".into());

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = gateway_listener.accept().await {
                    let st = Arc::clone(&state_clone);
                    tokio::spawn(handle_connection(stream, st));
                }
            }
        });

        // 1. GET /models returns HTTP 200 with object: "list"
        let req_models = format!(
            "GET /models HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nConnection: close\r\n\r\n"
        );
        let mut client1 = tokio::net::TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client1.write_all(req_models.as_bytes()).await.unwrap();
        let mut resp1 = String::new();
        let mut buf = [0u8; 1024];
        while let Ok(n) = client1.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp1.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp1.starts_with("HTTP/1.1 200 OK"), "resp1 was: {resp1}");
        assert!(resp1.contains(r#""object":"list""#));
        assert!(resp1.contains("qwen2.5-7b"));

        // 2. GET /models/qwen2.5-7b returns HTTP 200 with model object
        let req_single = format!(
            "GET /models/qwen2.5-7b HTTP/1.1\r\nHost: 127.0.0.1:{gateway_port}\r\nAuthorization: Bearer test-key\r\nConnection: close\r\n\r\n"
        );
        let mut client2 = tokio::net::TcpStream::connect(format!("127.0.0.1:{gateway_port}"))
            .await
            .unwrap();
        client2.write_all(req_single.as_bytes()).await.unwrap();
        let mut resp2 = String::new();
        while let Ok(n) = client2.read(&mut buf).await {
            if n == 0 {
                break;
            }
            resp2.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        assert!(resp2.starts_with("HTTP/1.1 200 OK"));
        assert!(resp2.contains(r#""id":"qwen2.5-7b""#));
    }

    #[tokio::test]
    async fn test_resilient_template_cache_retry() {
        let state = Arc::new(AppState::new());
        let def = ServerDef {
            port: 9999,
            backend: "llamacpp".into(),
            task: "instruct".into(),
            model_id: "test/uninitialized-model".into(),
            ..ServerDef::default()
        };

        let now = crate::state::now_ms();
        // Insert an empty template result cached 4 seconds ago
        let empty_info = crate::state::TemplateEffortInfo::default();
        state
            .template_effort_cache
            .lock()
            .unwrap()
            .insert(def.port, (now - 4_000, empty_info));

        let _ = template_info_for(&state, &def).await;

        let updated_time = state
            .template_effort_cache
            .lock()
            .unwrap()
            .get(&def.port)
            .unwrap()
            .0;

        assert!(
            updated_time >= now,
            "Cache should have re-fetched and updated timestamp because 4s > 3s retry TTL"
        );
    }
}
