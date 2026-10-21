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
use std::sync::Arc;
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
pub fn find_server_port_for(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    model: &str,
    task: &str,
) -> Option<u16> {
    let running = running_servers(servers, task);
    if running.is_empty() {
        return None;
    }
    let m = model.trim();
    let m = m.strip_suffix(":latest").unwrap_or(m);
    if m.is_empty() {
        return None;
    }

    // If wildcard/default aliases, and exactly one server is running, route to it.
    let is_wildcard = m.eq_ignore_ascii_case("default")
        || m.eq_ignore_ascii_case("local")
        || m.eq_ignore_ascii_case("auto")
        || m == "*";
    if is_wildcard {
        return if running.len() == 1 {
            Some(running[0].def.port)
        } else {
            None
        };
    }
    for ls in &running {
        if ls.def.effective_model_name().eq_ignore_ascii_case(m)
            || ls.def.model_id.eq_ignore_ascii_case(m)
        {
            return Some(ls.def.port);
        }
    }
    let lower = m.to_lowercase();
    for ls in &running {
        let hf = ls.def.model_id.to_lowercase();
        let name = ls.def.effective_model_name().to_lowercase();
        if hf.ends_with(&lower) || name.ends_with(&lower) || lower.ends_with(&hf) {
            return Some(ls.def.port);
        }
    }
    None
}

fn configured_context_tokens(def: &ServerDef) -> Option<usize> {
    match def.backend.as_str() {
        "vllm" => def.max_model_len.filter(|value| *value > 0),
        "llamacpp" => def.ctx_size.filter(|value| *value > 0),
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

fn model_traits(def: &ServerDef, launch_args: &[String]) -> ModelTraits {
    let name = format!("{} {}", def.model_id, def.effective_model_name()).to_ascii_lowercase();
    let is_instruct = def.task == "instruct";

    let family_reasoning = looks_like_reasoning_model(&name);
    let family_always_on = reasoning_is_always_on(&name);

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
        && (def.mmproj_path
            .as_deref()
            .is_some_and(|path| !path.trim().is_empty())
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

fn model_row(
    def: &ServerDef,
    context: Option<usize>,
    context_source: &str,
    launch_args: &[String],
) -> serde_json::Value {
    let context = context.filter(|value| *value > 0);
    let is_instruct = def.task == "instruct";
    let is_embed = def.task == "embed";
    let traits = model_traits(def, launch_args);
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
    let control = if !supports_template_controls || traits.reasoning_mode == ReasoningMode::None {
        "none"
    } else {
        "reasoning_effort"
    };
    // vLLM maps reasoning_effort onto the model's thinking toggle and
    // reasoning parser; llama.cpp forwards it to the Jinja template and
    // parses reasoning tags. The gateway canonicalizes reasoning_effort
    // across all reasoning models and formats.
    let supports_effort_parameter =
        traits.reasoning_mode != ReasoningMode::None && supports_template_controls;
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

    let supported_efforts: &[&str] = match traits.reasoning_mode {
        ReasoningMode::None => &[],
        ReasoningMode::Effort => &["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        ReasoningMode::AlwaysOn => &["low", "medium", "high"],
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
        "levels": traits.reasoning_mode.levels(),
        "default_level": traits.reasoning_mode.default_level(),
        "supported_efforts": supported_efforts,
        "default_effort": traits.reasoning_mode.default_level(),
        "thinking_level_map": serde_json::json!({
            "off": "none",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
        }),
        "thinkingLevelMap": serde_json::json!({
            "off": "none",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
        }),
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
        "context_window": context,
        "n_ctx": context,
        "context_source": context_source,
        // Hard upper bound on generation: the prompt and completion share the
        // context window. Embedding models do not generate tokens.
        "max_output_tokens": if is_instruct { context } else { None },
        "max_completion_tokens": if is_instruct { context } else { None },
        "max_output_tokens_source": if is_instruct && context.is_some() {
            Some("context_window")
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
    row["max_tokens"] = serde_json::json!(if is_instruct { context } else { None });
    row["input"] = serde_json::json!(input_modalities);
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
            let context = configured_context_tokens(&server.def);
            let source = if context.is_some() {
                "configured"
            } else {
                "unknown"
            };
            let args = effective_launch_args(&server.def, vllm_args);
            rows.push(model_row(&server.def, context, source, args));
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
        let configured = configured_context_tokens(&def);
        let (context, source) = if let Some(context) = configured {
            (Some(context), "configured")
        } else if let Some(context) = fetch_runtime_context(state, &def).await {
            (Some(context), "runtime")
        } else {
            (None, "unknown")
        };
        let args = effective_launch_args(&def, &vllm_args);
        rows.push(model_row(&def, context, source, args));
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
    let raw = path.split('?').next()?.strip_prefix("/v1/models/")?;
    if raw.is_empty() {
        None
    } else {
        // Decode first: an encoded `%3Alatest` suffix must be stripped too, so
        // a client that refreshes ids from `/v1/models` can address them.
        decode_path_component(raw).map(|id| {
            id.strip_suffix(":latest").unwrap_or(&id).to_string()
        })
    }
}

/// Extract the requested model id from a JSON request body.
/// Accepts `"model"` or `"name"` and strips `:latest` if present.
pub fn model_from_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let name = value.get("model").or_else(|| value.get("name"))?.as_str()?.trim();
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
        .or_else(|| object.get("think").and_then(think_level))
        .or_else(|| {
            nested("chat_template_kwargs")
                .and_then(|value| value.get("reasoning_effort"))
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
    }
    if let Some(value) = object.get("think").and_then(think_toggle) {
        return Some(value);
    }
    object
        .get("reasoning")
        .and_then(serde_json::Value::as_object)
        .and_then(|value| value.get("enabled"))
        .and_then(json_bool)
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
fn apply_harness_precedence(
    object: &mut serde_json::Map<String, serde_json::Value>,
    backend: &str,
    context_tokens: Option<usize>,
) {
    let raw_effort = harness_effort(object);
    let effort = raw_effort
        .as_deref()
        .and_then(|effort| normalize_effort_label(effort, backend));

    // Canonicalize an effort the harness already sent.  vLLM rejects
    // toggle-only labels such as "on"/"auto" in its reasoning_effort enum, so
    // those are represented by the explicit toggle instead.
    if let Some(current) = object
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
    {
        match normalize_effort_label(&current, backend) {
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

    let explicit_toggle = harness_toggle(object);
    let budget = harness_budget(object);
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
        if let Some(functions) = object.get("functions").and_then(serde_json::Value::as_array) {
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
                    object.insert("tool_choice".to_string(), serde_json::Value::String(s.clone()));
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
}

/// Normalize a chat request for the target backend.
///
/// The original bytes are returned verbatim when the harness did not ask for
/// any thinking control or OpenAI token-limit alias, so panel settings remain
/// the default for clients that do not care.  When the harness did ask, the
/// returned body contains explicit request-level controls that both vLLM and
/// llama.cpp rank above server defaults.
fn normalize_chat_request(
    body: &[u8],
    backend: &str,
    context_tokens: Option<usize>,
) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return body.to_vec();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_vec();
    };
    let original = object.clone();
    apply_harness_precedence(object, backend, context_tokens);
    if *object == original {
        return body.to_vec();
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
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

/// Validate the Authorization header against the configured gateway API key.
///
/// A gateway without a key is a misconfiguration, not an open server.  Return
/// a configuration error instead of silently exposing every routed model to
/// any local process (or browser page) that can reach localhost.
pub fn check_gateway_auth(
    headers: &HashMap<String, String>,
    global_key: Option<&str>,
) -> Result<(), (u16, String)> {
    let Some(expected_key) = global_key.map(str::trim).filter(|key| !key.is_empty()) else {
        return Err((
            503,
            "Gateway API key is not configured; set one in Settings before using /v1 routes"
                .to_string(),
        ));
    };
    let auth_header = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"));
    match auth_header {
        Some(h) if h.starts_with("Bearer ") => {
            let provided = h["Bearer ".len()..].trim();
            if !provided.is_empty() && constant_time_eq(provided, expected_key) {
                Ok(())
            } else {
                Err((401, "Invalid API key".to_string()))
            }
        }
        Some(_) => Err((
            401,
            "Authorization header must use Bearer scheme".to_string(),
        )),
        None => Err((401, "Missing Authorization header".to_string())),
    }
}

/// Is the chunk of bytes between the request line and the blank line a
/// valid `Content-Length: N` header we must read before routing?
pub fn parse_head(bytes: &[u8]) -> Option<RequestHead> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    if method.is_empty() || target.is_empty() {
        return None;
    }
    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let mut it = line.splitn(2, ':');
        let name = it.next()?.trim().to_ascii_lowercase();
        let value = it.next().unwrap_or("").trim().to_string();
        if name == "content-length" {
            content_length = value.parse().unwrap_or(0);
        }
        headers.insert(name, value);
    }
    Some(RequestHead {
        method,
        target,
        headers,
        content_length,
    })
}

/// Non-origin CORS headers shared by all allowed responses.  Kept public for
/// callers that need to inspect the policy without trusting a request origin.
pub const CORS_HEADERS: &str = "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n";

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
    format!(
        "Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n{CORS_HEADERS}"
    )
}

fn is_allowed_origin(origin: &str) -> bool {
    if origin == "tauri://localhost" || origin == "https://tauri.localhost" {
        return true;
    }
    ["http://127.0.0.1:", "http://localhost:"]
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

fn running_model_names(
    servers: &BTreeMap<String, crate::state::LiveServer>,
    task: &str,
) -> Vec<String> {
    running_servers(servers, task)
        .into_iter()
        .map(|ls| ls.def.effective_model_name())
        .collect()
}

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

async fn serve_connection(stream: &mut TcpStream, state: &Arc<AppState>) -> Result<(), String> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 8192];
    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|e| format!("read request: {e}"))?;
        if n == 0 {
            return Err("connection closed before request completed".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(sep) = is_head_terminated(&buf) {
            let head =
                parse_head(&buf[..sep]).ok_or_else(|| "malformed request head".to_string())?;
            if head.content_length > MAX_BODY_BYTES {
                return Err("request body too large".into());
            }
            let total = sep + 4;
            while buf.len() < total + head.content_length {
                let m = stream
                    .read(&mut tmp)
                    .await
                    .map_err(|e| format!("read body: {e}"))?;
                if m == 0 {
                    return Err("connection closed during body".into());
                }
                buf.extend_from_slice(&tmp[..m]);
            }
            let body = buf[total..total + head.content_length].to_vec();
            return route(stream, state, &head, &body).await;
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err("request head too large".into());
        }
    }
}

async fn route(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    head: &RequestHead,
    body: &[u8],
) -> Result<(), String> {
    if head.method.as_str() == "OPTIONS" {
        if let Some(origin) = head.headers.get("origin") {
            if !is_allowed_origin(origin) {
                return write_request_response(stream, head, 403, "text/plain", b"origin not allowed").await;
            }
        }
        return write_request_response(stream, head, 204, "text/plain", b"").await;
    }

    // Auth check for all /v1/* and /api/* routes except /health and /api/version
    let is_protected = (head.target.starts_with("/v1/") || head.target.starts_with("/api/"))
        && head.target != "/health"
        && head.target != "/api/version";
    if is_protected {
        let cfg = state.config();
        let global_key = cfg.advanced_settings.api_key.as_deref();
        if let Err((status, msg)) = check_gateway_auth(&head.headers, global_key) {
            return write_request_response(
                stream,
                head,
                status,
                "application/json",
                serde_json::json!({ "error": { "message": msg, "type": "authentication_error" } })
                    .to_string()
                    .as_bytes(),
            )
            .await;
        }
    }

    let path = head.target.split('?').next().unwrap_or("");
    match (head.method.as_str(), path) {
        ("GET", "/v1/models") => {
            let text = serde_json::json!({
                "object": "list",
                "data": model_rows_for_state(state).await,
            })
            .to_string();
            write_request_response(stream, head, 200, "application/json", text.as_bytes()).await
        }
        (candidate_path, _)
            if head.method == "GET" && candidate_path.starts_with("/v1/models/") =>
        {
            let Some(model_id) = model_id_from_path(candidate_path) else {
                return write_request_response(
                    stream,
                    head,
                    400,
                    "application/json",
                    serde_json::json!({
                        "error": {
                            "message": "Model id is required",
                            "type": "invalid_request_error",
                        }
                    })
                    .to_string()
                    .as_bytes(),
                )
                .await;
            };
            let rows = model_rows_for_state(state).await;
            let Some(row) = rows.into_iter().find(|row| {
                row["id"].as_str() == Some(model_id.as_str())
                    || row["root"].as_str() == Some(model_id.as_str())
            }) else {
                let payload = serde_json::json!({
                    "error": {
                        "code": "model_not_found",
                        "message": format!("Model '{model_id}' was not found"),
                        "type": "invalid_request_error",
                    }
                })
                .to_string();
                return write_request_response(stream, head, 404, "application/json", payload.as_bytes()).await;
            };
            write_request_response(stream, head, 200, "application/json", row.to_string().as_bytes()).await
        }
        ("GET", "/health") => {
            write_request_response(stream, head, 200, "application/json", b"{\"status\":\"ok\"}").await
        }
        ("GET", "/api/version") => {
            write_request_response(
                stream,
                head,
                200,
                "application/json",
                b"{\"version\":\"0.5.4\"}",
            )
            .await
        }
        ("GET", "/api/tags") => {
            let payload = ollama_tags_payload(state).await.to_string();
            write_request_response(stream, head, 200, "application/json", payload.as_bytes()).await
        }
        ("POST", "/api/show") => {
            let requested_name = model_from_body(body).unwrap_or_default();
            let rows = model_rows_for_state(state).await;
            let found = rows.into_iter().find(|row| {
                row["id"].as_str() == Some(requested_name.as_str())
                    || row["root"].as_str() == Some(requested_name.as_str())
            });
            if let Some(row) = found {
                let id = row["id"].as_str().unwrap_or("model");
                let quant = row["quantization"].as_str().unwrap_or("unknown");
                let format = row["compatibility_type"].as_str().unwrap_or("gguf");
                let params = row.get("parameters_billions").and_then(serde_json::Value::as_f64);
                let param_size = params.map(|p| format!("{:.0}B", p)).unwrap_or_default();
                let payload = serde_json::json!({
                    "modelfile": format!("FROM {id}"),
                    "parameters": "",
                    "template": "",
                    "details": {
                        "parent_model": "",
                        "format": format,
                        "family": id.split(['-', '/', '_']).next().unwrap_or("model"),
                        "families": [id.split(['-', '/', '_']).next().unwrap_or("model")],
                        "parameter_size": param_size,
                        "quantization_level": quant,
                    },
                    "model_info": row,
                })
                .to_string();
                write_request_response(stream, head, 200, "application/json", payload.as_bytes()).await
            } else {
                let payload = serde_json::json!({
                    "error": format!("model '{requested_name}' not found")
                })
                .to_string();
                write_request_response(stream, head, 404, "application/json", payload.as_bytes()).await
            }
        }
        ("POST", "/v1/chat/completions") | ("POST", "/api/chat") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model or name field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "instruct")
            };
            match port {
                Some(port) => proxy_chat(stream, state, port, body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "instruct");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "instruction", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        ("POST", "/v1/completions") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "instruct")
            };
            match port {
                Some(port) => proxy_post(stream, state, port, "/v1/completions", body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "instruct");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "instruction", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        ("POST", "/v1/embeddings") => {
            let model = model_from_body(body)
                .ok_or_else(|| "request body must be JSON with a model field".to_string())?;
            let port = {
                let servers = state.servers.lock().unwrap();
                find_server_port_for(&servers, &model, "embed")
            };
            match port {
                Some(port) => proxy_post(stream, state, port, "/v1/embeddings", body, head.headers.get("origin").map(String::as_str)).await,
                None => {
                    let running = running_model_names(&state.servers.lock().unwrap(), "embed");
                    write_request_response(
                        stream,
                        head,
                        404,
                        "application/json",
                        not_found_payload_with_hint(&model, "embedding", &running).as_bytes(),
                    )
                    .await
                }
            }
        }
        (method, "/v1/models") if method != "GET" => {
            write_request_response(stream, head, 405, "text/plain", b"method not allowed").await
        }
        _ => write_request_response(stream, head, 404, "text/plain", b"not found").await,
    }
}

async fn write_request_response(
    stream: &mut TcpStream,
    head: &RequestHead,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    write_response_with_origin(
        stream,
        status,
        content_type,
        body,
        head.headers.get("origin").map(String::as_str),
    )
    .await
}

async fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    write_response_with_origin(stream, status, content_type, body, None).await
}

async fn write_response_with_origin(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    origin: Option<&str>,
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let cors = cors_headers(origin);
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{cors}Connection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(body).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())
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
            let params = row.get("parameters_billions").and_then(serde_json::Value::as_f64);
            let param_size = params.map(|p| format!("{:.0}B", p)).unwrap_or_default();
            let size = row.get("parameter_count").and_then(serde_json::Value::as_u64).unwrap_or(0);
            serde_json::json!({
                "name": format!("{id}:latest"),
                "model": format!("{id}:latest"),
                "modified_at": "2026-10-21T00:00:00Z",
                "size": size,
                "digest": hash_to_hex(id),
                "details": {
                    "parent_model": "",
                    "format": format,
                    "family": id.split(['-', '/', '_']).next().unwrap_or("model"),
                    "families": [id.split(['-', '/', '_']).next().unwrap_or("model")],
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
    if let Some(choices) = obj.get_mut("choices").and_then(serde_json::Value::as_array_mut) {
        for choice in choices {
            let Some(choice_obj) = choice.as_object_mut() else {
                continue;
            };
            if let Some(delta) = choice_obj.get_mut("delta").and_then(serde_json::Value::as_object_mut) {
                normalize_reasoning_message(delta);
            }
            if let Some(msg) = choice_obj.get_mut("message").and_then(serde_json::Value::as_object_mut) {
                normalize_reasoning_message(msg);
            }
        }
    }
    if let Some(usage) = obj.get_mut("usage").and_then(serde_json::Value::as_object_mut) {
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
    let mut reasoning_tokens = usage.get("reasoning_tokens").and_then(serde_json::Value::as_u64);
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

async fn proxy_chat(
    stream: &mut TcpStream,
    state: &Arc<AppState>,
    port: u16,
    body: &[u8],
    origin: Option<&str>,
) -> Result<(), String> {
    let (backend, context) = {
        let servers = state.servers.lock().unwrap();
        servers
            .values()
            .find(|server| server.def.port == port)
            .map(|server| {
                (
                    server.def.backend.clone(),
                    configured_context_tokens(&server.def),
                )
            })
            .unwrap_or_else(|| ("vllm".to_string(), None))
    };
    let body = normalize_chat_request(body, &backend, context);
    proxy_post(stream, state, port, "/v1/chat/completions", &body, origin).await
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
) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}{upstream_path}");
    let streaming = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    let api_key = state
        .config()
        .servers
        .iter()
        .find(|server| server.port == port)
        .and_then(|server| crate::server::server_api_key(state, server));
    let req = state
        .http
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec());
    let resp = apply_vllm_auth(req, api_key.as_deref())
        .send()
        .await
        .map_err(|e| format!("upstream POST {url}: {e}"))?;

    let status = resp.status().as_u16();
    let reason = resp.status().canonical_reason().unwrap_or("OK");

    if streaming {
        let head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{}Connection: close\r\n\r\n",
            cors_headers(origin)
        );
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        let mut upstream = resp;
        let mut line_buf = Vec::new();
        while let Ok(Some(chunk)) = upstream.chunk().await {
            line_buf.extend_from_slice(&chunk);
            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = line_buf.drain(..=pos).collect();
                if let Ok(line_str) = std::str::from_utf8(&line_bytes) {
                    let normalized = normalize_sse_line(line_str);
                    let size = format!("{:x}\r\n", normalized.len());
                    stream
                        .write_all(size.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                    stream.write_all(normalized.as_bytes()).await.map_err(|e| e.to_string())?;
                    stream.write_all(b"\r\n").await.map_err(|e| e.to_string())?;
                } else {
                    let size = format!("{:x}\r\n", line_bytes.len());
                    stream
                        .write_all(size.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                    stream.write_all(&line_bytes).await.map_err(|e| e.to_string())?;
                    stream.write_all(b"\r\n").await.map_err(|e| e.to_string())?;
                }
            }
        }
        if !line_buf.is_empty() {
            let size = format!("{:x}\r\n", line_buf.len());
            stream
                .write_all(size.as_bytes())
                .await
                .map_err(|e| e.to_string())?;
            stream.write_all(&line_buf).await.map_err(|e| e.to_string())?;
            stream.write_all(b"\r\n").await.map_err(|e| e.to_string())?;
        }
        stream
            .write_all(b"0\r\n\r\n")
            .await
            .map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())
    } else {
        let body_bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("upstream body read: {e}"))?;
        let mut final_bytes = body_bytes.to_vec();
        if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            normalize_chat_response_json(&mut val);
            if let Ok(s) = serde_json::to_vec(&val) {
                final_bytes = s;
            }
        }
        write_response_with_origin(stream, status, "application/json", &final_bytes, origin).await
    }
}

async fn handle_connection(mut stream: TcpStream, state: Arc<AppState>) {
    let result = serve_connection(&mut stream, &state).await;
    if let Err(e) = result {
        let _ = write_response(
            &mut stream,
            500,
            "text/plain",
            format!("gateway error: {e}").as_bytes(),
        )
        .await;
    }
    let _ = stream.shutdown().await;
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
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
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
                let stale = matches!(&running, Some((p, h)) if *p != port || h.is_finished());
                if stale {
                    if let Some((_, h)) = running.take() {
                        h.abort();
                    }
                }
                if running.is_none() {
                    running = Some((port, start_gateway(Arc::clone(&state), port)));
                }
            } else if let Some((_, h)) = running.take() {
                h.abort();
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
                llamacpp_channel: crate::state::LlamaCppChannel::Upstream,
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
        // Boolean top-level field so OpenAI-compatible harnesses detect a
        // reasoning model; details stay in reasoning_config/thinking.
        assert_eq!(row["reasoning"], true);
        assert_eq!(row["supports_reasoning"], true);
        assert_eq!(row["thinking"]["mode"], "effort");
        assert_eq!(row["reasoning_config"]["supported"], true);
        assert_eq!(row["reasoning_config"]["mode"], "effort");
        assert_eq!(
            row["reasoning_config"]["levels"],
            serde_json::json!(["none", "low", "medium", "high"])
        );
        assert_eq!(row["reasoning_config"]["default_level"], "medium");
        assert_eq!(
            row["reasoning_config"]["control"],
            "reasoning_effort"
        );
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
        assert_eq!(
            row["reasoning_config"]["control"],
            "reasoning_effort"
        );
        assert_eq!(row["reasoning_config"]["parameters"]["reasoning_effort"], true);
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
            .any(|hint| hint.as_str().unwrap_or_default().contains("max_new_tokens")));    }

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
        // Pi's compat extension reads boolean `reasoning`.
        assert_eq!(row["reasoning"], true);
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
            let body = format!(
                r#"{{"model":"m","messages":[],"reasoning_effort":"{input}"}}"#
            );
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
            let body = format!(
                r#"{{"model":"m","messages":[],"reasoning_effort":"{input}"}}"#
            );
            let value = normalized(&body, "llamacpp", Some(8_192));
            assert_eq!(value["reasoning_effort"], expected, "llamacpp input {input}");
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
        assert_eq!(resp["usage"]["completion_tokens_details"]["reasoning_tokens"], 15);

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
        assert_eq!(delta_resp["choices"][0]["delta"]["reasoning"], "thought-stream");
        assert_eq!(delta_resp["choices"][0]["delta"]["reasoning_content"], "thought-stream");
        assert_eq!(delta_resp["usage"]["reasoning_tokens"], 7);
        assert_eq!(delta_resp["usage"]["completion_tokens_details"]["reasoning_tokens"], 7);
    }

    #[test]
    fn test_normalize_sse_line_mirrors_reasoning() {
        let line = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n";
        let out = normalize_sse_line(line);
        let parsed: serde_json::Value = serde_json::from_str(out.strip_prefix("data: ").unwrap().trim()).unwrap();
        assert_eq!(parsed["choices"][0]["delta"]["reasoning_content"], "thinking...");
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
        assert_eq!(find_server_port_for(&servers, "default", "instruct"), Some(8000));
        assert_eq!(find_server_port_for(&servers, "local", "instruct"), Some(8000));
        assert_eq!(find_server_port_for(&servers, "auto", "instruct"), Some(8000));
        assert_eq!(find_server_port_for(&servers, "*", "instruct"), Some(8000));
        assert_eq!(find_server_port_for(&servers, "", "instruct"), None);
        assert_eq!(find_server_port_for(&servers, "qwen3-8b:latest", "instruct"), Some(8000));
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
}
