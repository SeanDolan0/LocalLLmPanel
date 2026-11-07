//! Normalized response-completion contract for the OpenAI-compatible gateway.
//!
//! The gateway is a transport proxy, not the model. An HTTP 200 from an
//! upstream vLLM / llama.cpp server therefore does **not** imply a completed
//! generation: the provider may stop for the token limit, a stop sequence, a
//! tool call, a policy refusal, or it may omit termination metadata entirely.
//! This module models those distinctions explicitly so neither the gateway nor
//! downstream coding harnesses have to guess.
//!
//! ## Provider behavior (verified against official docs)
//!
//! * **OpenAI Chat Completions** (`/v1/chat/completions`): every streaming
//!   `chat.completion.chunk` carries `choices[].finish_reason` (`null` for
//!   intermediate deltas; the terminal chunk carries `stop`, `length`,
//!   `tool_calls`, `content_filter`, or `function_call`). With
//!   `stream_options.include_usage: true` the provider emits a final chunk
//!   with `choices: []` and a non-null `usage` before `data: [DONE]`. All
//!   other chunks carry `usage: null`. The `[DONE]` sentinel is only a
//!   transport terminator — it is not itself a completion signal.
//! * **OpenAI Responses** (`/v1/responses`): `response.status` is `completed`,
//!   `incomplete`, `failed`, or `queued`-style; `incomplete_details.reason`
//!   distinguishes `max_output_tokens`, `content_filter`, or `steered`.
//!   Streaming surfaces `response.completed` / `response.incomplete` events.
//! * **Anthropic Messages**: `Message.stop_reason` is one of `end_turn`
//!   (natural stop), `max_tokens` (output limit), `stop_sequence` (custom
//!   stop hit), `tool_use` (tool call), `pause_turn` (resumable pause),
//!   `refusal` (streaming policy intervention), or
//!   `model_context_window_exceeded`. In streaming it arrives on the final
//!   `message_delta` event.
//! * **Google Gemini** (`generateContent`): `candidates[].finishReason` is
//!   `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`,
//!   `SPII`, `MALFORMED_FUNCTION_CALL`, `TOOL_CALL`, `LANGUAGE`, or `OTHER`.
//!   Prompt/output billing lives in `usageMetadata`
//!   (`promptTokenCount` / `candidatesTokenCount` / `totalTokenCount`).
//! * **vLLM / llama.cpp (OpenAI-compatible local servers)**: both expose
//!   Chat Completions `finish_reason` in non-streaming and streaming modes
//!   and honor `stream_options.include_usage` (llama.cpp recently added the
//!   trailing usage chunk; older builds omit usage entirely). The gateway
//!   preserves provider metadata verbatim and never invents a finish reason
//!   or a usage count the backend did not report.
//!
//! ## Rules enforced here
//!
//! * Never fabricate `stop` / `completed` when no authoritative terminal
//!   signal was received. Clean EOF without terminal metadata is
//!   `Unknown`/`incomplete`, not success.
//! * Never overwrite an authoritative usage block with another chunk's
//!   usage; never default missing usage to `1`.
//! * Reasoning/thinking tokens stay separate from visible output tokens.
//! * `[DONE]` is emitted only after an explicit terminal event has been
//!   forwarded to the client.

use serde::Serialize;

/// Normalized completion status. Transport success (HTTP 200 + clean EOF) is
/// deliberately *not* a status: see [`StreamTermination`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    /// Model produced a complete, usable response.
    Completed,
    /// Model stopped early for a known reason (length, stop sequence,
    /// refusal, pause-turn, context exhaustion).
    Incomplete,
    /// Provider reported a failure (error, filter block, malformed payload).
    Failed,
    /// Client disconnected or the stream was cancelled.
    Cancelled,
    /// No authoritative termination signal was received; the app contract
    /// treats this as *not* completed.
    Unknown,
}

impl CompletionStatus {
    pub fn is_success(self) -> bool {
        matches!(self, Self::Completed)
    }
}

impl std::fmt::Display for CompletionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// Normalized finish reason, provider-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizedFinishReason {
    /// Natural stop (OpenAI `stop`, Anthropic `end_turn`, Gemini `STOP`,
    /// Responses `completed` with no incomplete reason).
    Stop,
    /// Output-token limit hit (OpenAI `length`, Anthropic `max_tokens`,
    /// Gemini `MAX_TOKENS`, Responses `max_output_tokens`).
    Length,
    /// A stop sequence terminated generation (Anthropic `stop_sequence`).
    StopSequence,
    /// Turn ended to invoke tools (OpenAI `tool_calls`/`function_call`,
    /// Anthropic `tool_use`, Gemini `TOOL_CALL`).
    ToolCalls,
    /// Policy/safety refusal or block (OpenAI `content_filter`, Anthropic
    /// `refusal`, Gemini `SAFETY`/`RECITATION`/etc., Responses
    /// `content_filter`).
    Refusal,
    /// Anthropic `pause_turn`: resumable pause.
    Paused,
    /// Context window exhausted (Anthropic
    /// `model_context_window_exceeded`, Gemini `OTHER` with context note).
    ContextExhausted,
    /// No finish/stop reason was exposed by the provider.
    Unknown,
    /// Provider reported an error instead of a finish reason.
    Failed,
}

impl NormalizedFinishReason {
    pub fn from_openai_chat(raw: Option<&str>) -> Self {
        match raw {
            Some("stop") => Self::Stop,
            Some("length") => Self::Length,
            Some("tool_calls") | Some("function_call") => Self::ToolCalls,
            Some("content_filter") => Self::Refusal,
            Some("refusal") => Self::Refusal,
            Some("error") => Self::Failed,
            _ => Self::Unknown,
        }
    }

    pub fn from_responses_status(raw: Option<&str>, incomplete_reason: Option<&str>) -> Self {
        match raw {
            Some("completed") => match incomplete_reason {
                Some("max_output_tokens") => Self::Length,
                Some("content_filter") => Self::Refusal,
                _ => Self::Stop,
            },
            Some("incomplete") => match incomplete_reason {
                Some("max_output_tokens") => Self::Length,
                Some("content_filter") => Self::Refusal,
                _ => Self::Unknown,
            },
            Some("failed") => Self::Failed,
            Some("cancelled") => Self::Paused,
            _ => Self::Unknown,
        }
    }

    pub fn from_anthropic(raw: Option<&str>) -> Self {
        match raw {
            Some("end_turn") => Self::Stop,
            Some("max_tokens") => Self::Length,
            Some("stop_sequence") => Self::StopSequence,
            Some("tool_use") => Self::ToolCalls,
            Some("refusal") => Self::Refusal,
            Some("pause_turn") => Self::Paused,
            Some("model_context_window_exceeded") => Self::ContextExhausted,
            Some("error") => Self::Failed,
            _ => Self::Unknown,
        }
    }

    pub fn from_gemini(raw: Option<&str>) -> Self {
        match raw {
            Some("STOP") => Self::Stop,
            Some("MAX_TOKENS") => Self::Length,
            Some("SAFETY")
            | Some("RECITATION")
            | Some("BLOCKLIST")
            | Some("PROHIBITED_CONTENT")
            | Some("SPII")
            | Some("MALFORMED_FUNCTION_CALL") => Self::Refusal,
            Some("TOOL_CALL") => Self::ToolCalls,
            Some("OTHER") => Self::ContextExhausted,
            _ => Self::Unknown,
        }
    }

    /// Map back onto an OpenAI-compatible `finish_reason` for gateway output.
    /// `Unknown` and `Failed` map to `null` so downstream SDKs cannot mistake
    /// them for a normal stop.
    pub fn to_openai_chat(self) -> Option<&'static str> {
        match self {
            Self::Stop => Some("stop"),
            Self::Length => Some("length"),
            Self::StopSequence => Some("stop"),
            Self::ToolCalls => Some("tool_calls"),
            Self::Refusal => Some("content_filter"),
            Self::Paused | Self::ContextExhausted => Some("length"),
            Self::Unknown | Self::Failed => None,
        }
    }
}

/// Where token counts came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageSource {
    /// Provider reported counts explicitly.
    Authoritative,
    /// Gateway estimated from text/tool payloads (never presented as
    /// authoritative).
    Estimated,
    /// Provider omitted usage entirely.
    #[default]
    Unavailable,
}

impl UsageSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Authoritative => "authoritative",
            Self::Estimated => "estimated",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Token usage split into visible output vs reasoning/thinking.
/// `None` means "not reported", never a guessed `1`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub visible_output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub source: UsageSource,
}

impl TokenUsage {
    pub fn unavailable() -> Self {
        Self {
            input_tokens: None,
            visible_output_tokens: None,
            reasoning_tokens: None,
            cached_input_tokens: None,
            total_tokens: None,
            source: UsageSource::Unavailable,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.input_tokens.is_none()
            && self.visible_output_tokens.is_none()
            && self.reasoning_tokens.is_none()
            && self.total_tokens.is_none()
    }

    pub fn to_responses_usage(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        if let Some(i) = self.input_tokens {
            obj.insert("input_tokens".to_string(), serde_json::json!(i));
        }
        if let Some(o) = self.visible_output_tokens {
            obj.insert("output_tokens".to_string(), serde_json::json!(o));
        }
        if let Some(t) = self.total_tokens {
            obj.insert("total_tokens".to_string(), serde_json::json!(t));
        }
        if let Some(r) = self.reasoning_tokens {
            obj.insert(
                "output_tokens_details".to_string(),
                serde_json::json!({ "reasoning_tokens": r }),
            );
        }
        serde_json::Value::Object(obj)
    }
}

/// How the byte stream ended, independent of generation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamTermination {
    /// Explicit provider terminal event (`finish_reason`, Responses
    /// `response.completed/incomplete`, Anthropic `message_stop`, Gemini
    /// `finishReason`) was observed.
    ExplicitTerminal,
    /// `[DONE]` sentinel was observed (transport terminator, not by itself
    /// a completion signal).
    DoneSentinel,
    /// TCP/FIN clean EOF without any terminal metadata.
    CleanEof,
    /// Abrupt disconnect, read error, or timeout.
    Disconnect,
    /// Malformed SSE frame or undecodable UTF-8.
    Malformed,
    /// Client cancelled the request.
    Cancelled,
    /// Stream is still open / not yet finalized.
    Pending,
}

impl StreamTermination {
    pub fn has_explicit_terminal(self) -> bool {
        matches!(self, Self::ExplicitTerminal)
    }
    pub fn ended_cleanly(self) -> bool {
        matches!(
            self,
            Self::ExplicitTerminal | Self::DoneSentinel | Self::CleanEof
        )
    }
}

/// Normalized view of a completed generation, for diagnostics and for the
/// `x_local_panel_completion` extension emitted on gateway responses.
#[derive(Debug, Clone, Serialize)]
pub struct NormalizedResponse {
    pub provider: String,
    pub model: String,
    pub request_id: Option<String>,
    pub response_id: Option<String>,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<serde_json::Value>,
    pub status: CompletionStatus,
    pub finish_reason: NormalizedFinishReason,
    pub provider_finish_reason: Option<String>,
    pub incomplete_reason: Option<String>,
    pub usage: TokenUsage,
    pub explicit_terminal: bool,
    pub clean_eof: bool,
    pub termination: StreamTermination,
    pub chunk_count: u64,
    pub event_count: u64,
    pub malformed_events: u64,
    pub raw_provider: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// One parsed SSE event (per SSE spec: blank line terminates an event;
/// `data:` lines join with `\n`; `:` lines are comments/keepalives).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub data: Vec<String>,
    pub event_type: Option<String>,
    pub is_comment: bool,
}

impl SseEvent {
    pub fn joined_data(&self) -> String {
        self.data.join("\n")
    }
}

/// Incremental SSE decoder. Feed raw bytes; drain complete events as they
/// become available. Partial trailing events stay buffered, so feed bytes
/// in any chunking and call [`SseDecoder::finish`] at EOF.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    current: Vec<String>,
    current_type: Option<String>,
    saw_comment: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut consumed = 0usize;
        // Scan for \n boundaries; lines may be \r\n or \n.
        while let Some(pos) = self.buffer[consumed..].iter().position(|&b| b == b'\n') {
            let line_end = consumed + pos; // index of \n
            let mut line = &self.buffer[consumed..line_end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            let line = line.to_vec();
            consumed = line_end + 1;
            if let Some(event) = self.consume_line(&line) {
                events.push(event);
            }
        }
        self.buffer.drain(..consumed);
        events
    }

    fn consume_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            // Blank line terminates the event.
            if self.current.is_empty() && self.current_type.is_none() {
                let comment = self.saw_comment;
                self.saw_comment = false;
                if comment {
                    return Some(SseEvent {
                        data: Vec::new(),
                        event_type: None,
                        is_comment: true,
                    });
                }
                return None;
            }
            let event = SseEvent {
                data: std::mem::take(&mut self.current),
                event_type: std::mem::take(&mut self.current_type),
                is_comment: false,
            };
            return Some(event);
        }
        if line[0] == b':' {
            self.saw_comment = true;
            return None;
        }
        // Field: value. Tolerate `data:{...}` and `data: {...}`.
        let colon = match line.iter().position(|&b| b == b':') {
            Some(idx) => idx,
            None => {
                // Field-less line: per spec treat whole line as data.
                self.current
                    .push(String::from_utf8_lossy(line).into_owned());
                return None;
            }
        };
        let field = String::from_utf8_lossy(&line[..colon]).into_owned();
        let mut value = &line[colon + 1..];
        if value.first() == Some(&b' ') {
            value = &value[1..];
        }
        let value = String::from_utf8_lossy(value).into_owned();
        match field.as_str() {
            "data" => self.current.push(value),
            "event" => self.current_type = Some(value),
            // `id:` and `retry:` are ignored for completion purposes.
            _ => {}
        }
        None
    }

    /// Flush any final buffered event at EOF. Returns `None` when the buffer
    /// held no complete event.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = self.buffer.clone();
            self.buffer.clear();
            // Treat the leftover as one final line.
            if let Some(event) = self.consume_line(&line) {
                return Some(event);
            }
        }
        if self.current.is_empty() && self.current_type.is_none() {
            return None;
        }
        Some(SseEvent {
            data: std::mem::take(&mut self.current),
            event_type: std::mem::take(&mut self.current_type),
            is_comment: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Chunk interpretation
// ---------------------------------------------------------------------------

/// What a single parsed SSE `data:` payload means for completion state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkSignal {
    /// Chat delta with text/reasoning/tool-call fragments.
    Delta,
    /// Chunk with a non-null finish/stop reason — explicit terminal.
    Terminal,
    /// Chunk that only carries usage (OpenAI `include_usage` trailing chunk).
    UsageOnly,
    /// `data: [DONE]` sentinel.
    Done,
    /// Non-JSON payload that is not `[DONE]` (malformed for JSON SSE).
    Malformed,
    /// Keepalive comment or provider-specific non-data event.
    Ignore,
}

/// Classify one SSE event payload without consuming it.
pub fn classify_event(event: &SseEvent) -> ChunkSignal {
    if event.is_comment {
        return ChunkSignal::Ignore;
    }
    let joined = event.joined_data();
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        return ChunkSignal::Ignore;
    }
    if trimmed == "[DONE]" {
        return ChunkSignal::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return ChunkSignal::Malformed;
    };
    // Anthropic / Responses-style typed events.
    if let Some(event_type) = event.event_type.as_deref() {
        match event_type {
            "message_stop" | "response.completed" => return ChunkSignal::Terminal,
            "response.incomplete" => return ChunkSignal::Terminal,
            "response.output_text.delta"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.created"
            | "response.in_progress" => return ChunkSignal::Delta,
            "message_delta" => {
                if value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                    .is_some_and(|s| !s.is_empty())
                {
                    return ChunkSignal::Terminal;
                }
                return ChunkSignal::Delta;
            }
            _ => {}
        }
    }
    if let Some(type_field) = value.get("type").and_then(|t| t.as_str()) {
        match type_field {
            "message_stop" | "response.completed" | "response.incomplete" => {
                return ChunkSignal::Terminal
            }
            "response.output_text.delta"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.created"
            | "response.in_progress" => return ChunkSignal::Delta,
            "message_delta" => {
                if value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                    .is_some_and(|s| !s.is_empty())
                {
                    return ChunkSignal::Terminal;
                }
                return ChunkSignal::Delta;
            }
            _ => {}
        }
    }
    // OpenAI Chat Completions chunk.
    if value.get("choices").is_some() {
        let has_usage = value.get("usage").is_some_and(|u| !u.is_null());
        let finish = value["choices"]
            .as_array()
            .and_then(|choices| choices.iter().find_map(|c| c.get("finish_reason")))
            .and_then(|f| f.as_str());
        if finish.is_some() {
            return ChunkSignal::Terminal;
        }
        let choices = value["choices"].as_array();
        let is_usage_only = has_usage && choices.is_some_and(|choices| choices.is_empty());
        if is_usage_only {
            return ChunkSignal::UsageOnly;
        }
        return ChunkSignal::Delta;
    }
    // Gemini streaming uses candidates[].finishReason.
    if value.get("candidates").is_some() {
        let finish = value["candidates"]
            .as_array()
            .and_then(|candidates| candidates.iter().find_map(|c| c.get("finishReason")))
            .and_then(|f| f.as_str());
        if finish.is_some() {
            return ChunkSignal::Terminal;
        }
        return ChunkSignal::Delta;
    }
    ChunkSignal::Malformed
}

// ---------------------------------------------------------------------------
// Usage extraction
// ---------------------------------------------------------------------------

/// Extract normalized usage from an OpenAI Chat Completions `usage` object,
/// preserving the visible-vs-reasoning split. Returns `None` when no usage
/// object is present — the caller must not substitute a default of `1`.
pub fn usage_from_openai(value: &serde_json::Value) -> Option<TokenUsage> {
    let usage = value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    Some(parse_openai_usage_object(usage))
}

fn parse_openai_usage_object(usage: &serde_json::Value) -> TokenUsage {
    let get = |key: &str| usage.get(key).and_then(|v| v.as_u64());
    let details = usage.get("completion_tokens_details");
    let details_get = |key: &str| details.and_then(|d| d.get(key)).and_then(|v| v.as_u64());
    let prompt_details = usage.get("prompt_tokens_details");
    let cached = prompt_details
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64());
    let reasoning = get("reasoning_tokens").or_else(|| details_get("reasoning_tokens"));
    let visible = get("completion_tokens").or_else(|| get("output_tokens"));
    // Some providers report completion_tokens excluding reasoning; when both
    // are present we keep them separate and report the total as-is.
    TokenUsage {
        input_tokens: get("prompt_tokens").or_else(|| get("input_tokens")),
        visible_output_tokens: visible,
        reasoning_tokens: reasoning,
        cached_input_tokens: cached,
        total_tokens: get("total_tokens"),
        source: UsageSource::Authoritative,
    }
}

/// Extract normalized usage from a Gemini `usageMetadata` object.
pub fn usage_from_gemini(value: &serde_json::Value) -> Option<TokenUsage> {
    let meta = value.get("usageMetadata")?;
    let get = |key: &str| meta.get(key).and_then(|v| v.as_u64());
    Some(TokenUsage {
        input_tokens: get("promptTokenCount"),
        visible_output_tokens: get("candidatesTokenCount"),
        reasoning_tokens: get("thoughtsTokenCount"),
        cached_input_tokens: get("cachedContentTokenCount"),
        total_tokens: get("totalTokenCount"),
        source: UsageSource::Authoritative,
    })
}

/// Extract normalized usage from an Anthropic `usage` object.
pub fn usage_from_anthropic(value: &serde_json::Value) -> Option<TokenUsage> {
    let usage = value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    let get = |key: &str| usage.get(key).and_then(|v| v.as_u64());
    Some(TokenUsage {
        input_tokens: get("input_tokens"),
        visible_output_tokens: get("output_tokens"),
        reasoning_tokens: None,
        cached_input_tokens: get("cache_read_input_tokens"),
        total_tokens: None,
        source: UsageSource::Authoritative,
    })
}

/// Extract normalized usage from a Responses API `usage` object.
pub fn usage_from_responses(value: &serde_json::Value) -> Option<TokenUsage> {
    let usage = if value.get("usage").is_some() {
        value.get("usage")?
    } else {
        value
    };
    if usage.is_null() || !usage.is_object() {
        return None;
    }
    let get = |key: &str| usage.get(key).and_then(|v| v.as_u64());
    let details = usage.get("output_tokens_details");
    let reasoning = details
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64());
    let input = get("input_tokens");
    let output = get("output_tokens");
    let total = get("total_tokens");
    if input.is_none() && output.is_none() && total.is_none() {
        return None;
    }
    Some(TokenUsage {
        input_tokens: input,
        visible_output_tokens: output,
        reasoning_tokens: reasoning,
        cached_input_tokens: None,
        total_tokens: total,
        source: UsageSource::Authoritative,
    })
}

/// Merge two usage snapshots, keeping the most recent authoritative values.
/// Never overwrites a non-null authoritative count with `None`.
pub fn merge_usage(current: &mut TokenUsage, next: TokenUsage) {
    if next.source == UsageSource::Unavailable {
        return;
    }
    if next.input_tokens.is_some() {
        current.input_tokens = next.input_tokens;
    }
    if next.visible_output_tokens.is_some() {
        current.visible_output_tokens = next.visible_output_tokens;
    }
    // Reasoning/visible split only changes when the provider reports it.
    if next.reasoning_tokens.is_some() {
        current.reasoning_tokens = next.reasoning_tokens;
    }
    if next.cached_input_tokens.is_some() {
        current.cached_input_tokens = next.cached_input_tokens;
    }
    if next.total_tokens.is_some() {
        current.total_tokens = next.total_tokens;
    }
    if next.source == UsageSource::Authoritative {
        current.source = UsageSource::Authoritative;
    }
}

// ---------------------------------------------------------------------------
// Aggregator (streaming state machine)
// ---------------------------------------------------------------------------

/// Streaming aggregator: observes raw SSE events, accumulates content in
/// order, tracks terminal metadata, and finalizes into a
/// [`NormalizedResponse`]. Feed events via [`StreamAggregator::apply`];
/// call [`StreamAggregator::finish`] with the observed [`StreamTermination`].
#[derive(Debug)]
pub struct StreamAggregator {
    pub provider: String,
    pub model: String,
    pub request_id: Option<String>,
    pub response_id: Option<String>,
    content: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    usage: TokenUsage,
    provider_finish_reason: Option<String>,
    normalized_finish: Option<NormalizedFinishReason>,
    incomplete_reason: Option<String>,
    explicit_terminal: bool,
    saw_done: bool,
    saw_terminal_done: bool,
    chunk_count: u64,
    event_count: u64,
    malformed_events: u64,
    raw_last: Option<serde_json::Value>,
    assistant_role_seen: bool,
}

impl StreamAggregator {
    pub fn new(provider: &str, model: &str) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            request_id: None,
            response_id: None,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            usage: TokenUsage::unavailable(),
            provider_finish_reason: None,
            normalized_finish: None,
            incomplete_reason: None,
            explicit_terminal: false,
            saw_done: false,
            saw_terminal_done: false,
            chunk_count: 0,
            event_count: 0,
            malformed_events: 0,
            raw_last: None,
            assistant_role_seen: false,
        }
    }

    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    /// Apply one SSE event. Returns the signal so callers can decide when a
    /// terminal event has been observed (terminal metadata may arrive before
    /// or after the final text delta — both orders are accepted because the
    /// aggregator keeps accumulating until [`StreamAggregator::finish`]).
    pub fn apply(&mut self, event: &SseEvent) -> ChunkSignal {
        self.event_count += 1;
        let signal = classify_event(event);
        match signal {
            ChunkSignal::Ignore => return signal,
            ChunkSignal::Malformed => {
                self.malformed_events += 1;
                return signal;
            }
            ChunkSignal::Done => {
                self.saw_done = true;
                if self.explicit_terminal {
                    self.saw_terminal_done = true;
                }
                // [DONE] is transport-only: it does not set a finish reason.
                return signal;
            }
            ChunkSignal::Delta | ChunkSignal::Terminal | ChunkSignal::UsageOnly => {}
        }
        let joined = event.joined_data();
        let trimmed = joined.trim();
        let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            self.malformed_events += 1;
            return ChunkSignal::Malformed;
        };
        self.chunk_count += 1;
        self.raw_last = Some(value.clone());

        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            if self.response_id.is_none() {
                self.response_id = Some(id.to_string());
            }
        }
        if let Some(model) = value.get("model").and_then(|v| v.as_str()) {
            if self.model.is_empty() {
                self.model = model.to_string();
            }
        }

        // Usage: merge authoritative counts; null/partial usage never erases.
        if let Some(usage_value) = value.get("usage") {
            if !usage_value.is_null() {
                let parsed = parse_openai_usage_object(usage_value);
                merge_usage(&mut self.usage, parsed);
            }
        }
        if let Some(usage) = usage_from_gemini(&value) {
            merge_usage(&mut self.usage, usage);
        }
        if let Some(usage) = usage_from_anthropic(&value) {
            // Anthropic usage lives on message_start / message_delta; treat
            // it as authoritative too, but only when it looks Anthropic.
            if value.get("type").and_then(|t| t.as_str()) == Some("message_delta")
                || value.get("type").and_then(|t| t.as_str()) == Some("message_start")
            {
                merge_usage(&mut self.usage, usage);
            }
        }

        // OpenAI / vLLM / llama.cpp chat chunks.
        if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    if let Some(role) = delta.get("role").and_then(|v| v.as_str()) {
                        if role == "assistant" {
                            self.assistant_role_seen = true;
                        }
                    }
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        self.content.push_str(content);
                    }
                    if let Some(reasoning) = delta
                        .get("reasoning_content")
                        .or_else(|| delta.get("reasoning"))
                        .and_then(|v| v.as_str())
                    {
                        self.reasoning.push_str(reasoning);
                    }
                    if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for call in calls {
                            self.merge_tool_call(call);
                        }
                    }
                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                        self.content.push_str(text);
                    }
                }
                if let Some(message) = choice.get("message") {
                    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                        if self.content.is_empty() {
                            self.content.push_str(content);
                        }
                    }
                    if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                        for call in calls {
                            self.merge_tool_call(call);
                        }
                    }
                }
                if let Some(text) = choice.get("text").and_then(|v| v.as_str()) {
                    self.content.push_str(text);
                }
                if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    self.set_terminal(
                        finish,
                        NormalizedFinishReason::from_openai_chat(Some(finish)),
                    );
                }
            }
        }

        // Anthropic streaming.
        if let Some(delta) = value.get("delta") {
            if let Some(stop_reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                self.set_terminal(
                    stop_reason,
                    NormalizedFinishReason::from_anthropic(Some(stop_reason)),
                );
            }
            if let Some(typ) = delta.get("type").and_then(|v| v.as_str()) {
                if typ == "text_delta" {
                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                        self.content.push_str(text);
                    }
                }
            }
        }
        let _ = self.assistant_role_seen;

        // Anthropic message_stop / Responses completed|incomplete events.
        if let Some(type_field) = value.get("type").and_then(|v| v.as_str()) {
            match type_field {
                "response.output_text.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                        self.content.push_str(delta);
                    }
                }
                "message_stop" => {
                    if self.normalized_finish.is_none() {
                        // message_stop without a reason is a terminal event
                        // but not evidence of *why* it stopped.
                        self.explicit_terminal = true;
                        self.normalized_finish = Some(NormalizedFinishReason::Unknown);
                    }
                }
                "response.completed" => {
                    self.explicit_terminal = true;
                    if self.normalized_finish.is_none() {
                        self.normalized_finish = Some(NormalizedFinishReason::Stop);
                    }
                    if let Some(resp) = value.get("response") {
                        if let Some(usage) = resp.get("usage") {
                            if let Some(u) = usage_from_responses(usage) {
                                merge_usage(&mut self.usage, u);
                            }
                        }
                    }
                }
                "response.incomplete" => {
                    self.explicit_terminal = true;
                    let reason = value
                        .get("response")
                        .and_then(|r| r.get("incomplete_details"))
                        .and_then(|d| d.get("reason"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    self.incomplete_reason.clone_from(&reason);
                    if self.normalized_finish.is_none() {
                        self.normalized_finish =
                            Some(NormalizedFinishReason::from_responses_status(
                                Some("incomplete"),
                                reason.as_deref(),
                            ));
                    }
                    if let Some(resp) = value.get("response") {
                        if let Some(usage) = resp.get("usage") {
                            if let Some(u) = usage_from_responses(usage) {
                                merge_usage(&mut self.usage, u);
                            }
                        }
                    }
                }
                "response.failed" => {
                    self.explicit_terminal = true;
                    self.normalized_finish = Some(NormalizedFinishReason::Failed);
                }
                _ => {}
            }
        }

        // Gemini candidates.
        if let Some(candidates) = value.get("candidates").and_then(|c| c.as_array()) {
            for candidate in candidates {
                if let Some(parts) = candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|p| p.as_array())
                {
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            // Avoid double-appending when the same payload is
                            // also a regular chunk; Gemini streaming is
                            // cumulative-safe because each chunk is incremental.
                            self.content.push_str(text);
                        }
                    }
                }
                if let Some(finish) = candidate.get("finishReason").and_then(|v| v.as_str()) {
                    self.set_terminal(finish, NormalizedFinishReason::from_gemini(Some(finish)));
                }
            }
        }

        signal
    }

    fn set_terminal(&mut self, raw: &str, normalized: NormalizedFinishReason) {
        // First terminal wins; duplicates are idempotent (never overwrite a
        // truthful reason with a later one).
        if self.explicit_terminal {
            return;
        }
        self.explicit_terminal = true;
        self.provider_finish_reason = Some(raw.to_string());
        self.normalized_finish = Some(normalized);
    }

    fn merge_tool_call(&mut self, call: &serde_json::Value) {
        let index = call
            .get("index")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.tool_calls.len() as u64) as usize;
        if index >= self.tool_calls.len() {
            self.tool_calls.resize_with(index + 1, || {
                serde_json::json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            });
        }
        let entry = &mut self.tool_calls[index];
        if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
            if !id.is_empty() {
                entry["id"] = serde_json::json!(id);
            }
        }
        if let Some(name) = call
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
        {
            if !name.is_empty() {
                entry["function"]["name"] = serde_json::json!(name);
            }
        }
        if let Some(args) = call
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
        {
            let existing = entry["function"]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            entry["function"]["arguments"] = serde_json::json!(format!("{existing}{args}"));
        }
    }

    /// Finalize with the observed transport termination.
    ///
    /// * `Disconnect`/`Malformed`/`Cancelled` → `failed`/`cancelled`/`incomplete`,
    ///   never `completed`.
    /// * Clean EOF without an explicit terminal signal → `Unknown` (the app
    ///   treats this as not-completed unless the provider exposed a terminal
    ///   reason). It is *not* promoted to `completed`.
    /// * Explicit terminal + stop-like reason → `completed` (except refusal /
    ///   length / context / paused which stay `incomplete` so harnesses can
    ///   distinguish a truncated response from a finished one).
    pub fn finish(self, termination: StreamTermination) -> NormalizedResponse {
        let normalized = self
            .normalized_finish
            .unwrap_or(NormalizedFinishReason::Unknown);
        let status = match termination {
            StreamTermination::Cancelled => CompletionStatus::Cancelled,
            StreamTermination::Disconnect | StreamTermination::Malformed => {
                if self.explicit_terminal && normalized == NormalizedFinishReason::Stop {
                    // A transport failure after a real stop still means the
                    // chunked stream was incomplete; callers must not treat
                    // it as a clean full response.
                    CompletionStatus::Failed
                } else {
                    CompletionStatus::Failed
                }
            }
            StreamTermination::ExplicitTerminal | StreamTermination::DoneSentinel => {
                match normalized {
                    NormalizedFinishReason::Stop | NormalizedFinishReason::ToolCalls => {
                        CompletionStatus::Completed
                    }
                    NormalizedFinishReason::Failed => CompletionStatus::Failed,
                    NormalizedFinishReason::Unknown => CompletionStatus::Unknown,
                    _ => CompletionStatus::Incomplete,
                }
            }
            StreamTermination::CleanEof => {
                // Clean EOF alone is not completion evidence.
                if self.explicit_terminal {
                    match normalized {
                        NormalizedFinishReason::Stop | NormalizedFinishReason::ToolCalls => {
                            CompletionStatus::Completed
                        }
                        NormalizedFinishReason::Failed => CompletionStatus::Failed,
                        NormalizedFinishReason::Unknown => CompletionStatus::Unknown,
                        _ => CompletionStatus::Incomplete,
                    }
                } else {
                    CompletionStatus::Unknown
                }
            }
            StreamTermination::Pending => CompletionStatus::Unknown,
        };
        let clean_eof = termination.ended_cleanly();
        let explicit_terminal = self.explicit_terminal;
        // If we saw [DONE] after a terminal event, the transport ended
        // CleanEof-style; otherwise keep the reported termination.
        let termination = if self.saw_terminal_done && termination == StreamTermination::Pending {
            StreamTermination::ExplicitTerminal
        } else {
            termination
        };
        NormalizedResponse {
            provider: self.provider.clone(),
            model: self.model.clone(),
            request_id: self.request_id.clone(),
            response_id: self.response_id.clone(),
            content: self.content.clone(),
            reasoning: self.reasoning.clone(),
            tool_calls: self.tool_calls.clone(),
            status,
            finish_reason: normalized,
            provider_finish_reason: self.provider_finish_reason.clone(),
            incomplete_reason: self.incomplete_reason.clone(),
            usage: self.usage.clone(),
            explicit_terminal,
            clean_eof,
            termination,
            chunk_count: self.chunk_count,
            event_count: self.event_count,
            malformed_events: self.malformed_events,
            raw_provider: self.raw_last.clone(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }
    pub fn saw_explicit_terminal(&self) -> bool {
        self.explicit_terminal
    }
    pub fn saw_done(&self) -> bool {
        self.saw_done
    }
}

/// Normalize a non-streaming Chat Completions JSON body into a
/// [`NormalizedResponse`]. Preserves provider metadata, extracts the truthful
/// `finish_reason`, and separates visible output from reasoning tokens.
pub fn normalize_non_streaming(
    provider: &str,
    model_fallback: &str,
    value: &serde_json::Value,
    request_id: Option<String>,
) -> NormalizedResponse {
    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(model_fallback)
        .to_string();
    let response_id = value.get("id").and_then(|v| v.as_str()).map(str::to_string);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut provider_finish_reason: Option<String> = None;
    let mut normalized = NormalizedFinishReason::Unknown;

    if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(message) = choice.get("message") {
                if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
                    content.push_str(text);
                }
                if let Some(reasoning_text) = message
                    .get("reasoning_content")
                    .or_else(|| message.get("reasoning"))
                    .and_then(|v| v.as_str())
                {
                    reasoning.push_str(reasoning_text);
                }
                if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                    tool_calls = calls.clone();
                }
            }
            if let Some(text) = choice.get("text").and_then(|v| v.as_str()) {
                content.push_str(text);
            }
            if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                provider_finish_reason = Some(finish.to_string());
                normalized = NormalizedFinishReason::from_openai_chat(Some(finish));
            }
        }
    }

    let explicit_terminal = provider_finish_reason.is_some();
    let usage = match usage_from_openai(value) {
        Some(usage) => usage,
        None => match usage_from_gemini(value).or_else(|| usage_from_anthropic(value)) {
            Some(usage) => usage,
            None => TokenUsage::unavailable(),
        },
    };

    let status = match normalized {
        NormalizedFinishReason::Stop | NormalizedFinishReason::ToolCalls => {
            if explicit_terminal {
                CompletionStatus::Completed
            } else {
                // HTTP 200 / JSON body without a finish reason is not
                // completion evidence.
                CompletionStatus::Unknown
            }
        }
        NormalizedFinishReason::Failed => CompletionStatus::Failed,
        NormalizedFinishReason::Unknown => CompletionStatus::Unknown,
        _ => CompletionStatus::Incomplete,
    };

    NormalizedResponse {
        provider: provider.to_string(),
        model,
        request_id,
        response_id,
        content,
        reasoning,
        tool_calls,
        status,
        finish_reason: normalized,
        provider_finish_reason,
        incomplete_reason: None,
        usage,
        explicit_terminal,
        clean_eof: true,
        termination: if explicit_terminal {
            StreamTermination::ExplicitTerminal
        } else {
            StreamTermination::CleanEof
        },
        chunk_count: 1,
        event_count: 1,
        malformed_events: 0,
        raw_provider: Some(value.clone()),
    }
}

/// Structured, secret-free diagnostics. Log this — never prompts, completions,
/// or authorization headers.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionDiagnostics {
    pub request_id: Option<String>,
    pub provider: String,
    pub model: String,
    pub chunks: u64,
    pub events: u64,
    pub malformed: u64,
    pub content_len: usize,
    pub status: String,
    pub finish_reason: String,
    pub provider_finish_reason: Option<String>,
    pub usage_source: String,
    pub explicit_terminal: bool,
    pub termination: StreamTermination,
}

impl From<&NormalizedResponse> for CompletionDiagnostics {
    fn from(response: &NormalizedResponse) -> Self {
        Self {
            request_id: response.request_id.clone(),
            provider: response.provider.clone(),
            model: response.model.clone(),
            chunks: response.chunk_count,
            events: response.event_count,
            malformed: response.malformed_events,
            content_len: response.content.len(),
            status: response.status.to_string(),
            finish_reason: format!("{:?}", response.finish_reason),
            provider_finish_reason: response.provider_finish_reason.clone(),
            usage_source: response.usage.source.as_str().to_string(),
            explicit_terminal: response.explicit_terminal,
            termination: response.termination,
        }
    }
}

impl CompletionDiagnostics {
    pub fn log_line(&self) -> String {
        format!(
            "[completion] request_id={} provider={} model={} chunks={} events={} malformed={} content_len={} status={} finish={} provider_finish={} usage_source={} explicit_terminal={} termination={:?}",
            self.request_id.as_deref().unwrap_or("-"),
            self.provider,
            self.model,
            self.chunks,
            self.events,
            self.malformed,
            self.content_len,
            self.status,
            self.finish_reason,
            self.provider_finish_reason.as_deref().unwrap_or("-"),
            self.usage_source,
            self.explicit_terminal,
            self.termination,
        )
    }
}

/// Build the `x_local_panel_completion` extension object emitted on gateway
/// responses so downstream tools can distinguish completion from transport
/// success without inventing a finish reason.
pub fn completion_extension(response: &NormalizedResponse) -> serde_json::Value {
    serde_json::json!({
        "status": response.status.to_string(),
        "finish_reason": response.finish_reason.to_openai_chat().map(|v| serde_json::json!(v)).unwrap_or(serde_json::Value::Null),
        "normalized_finish_reason": format!("{:?}", response.finish_reason).to_lowercase(),
        "provider_finish_reason": response.provider_finish_reason.clone(),
        "incomplete_reason": response.incomplete_reason.clone(),
        "usage_source": response.usage.source.as_str(),
        "explicit_terminal": response.explicit_terminal,
        "clean_eof": response.clean_eof,
        "termination": format!("{:?}", response.termination).to_lowercase(),
        "chunk_count": response.chunk_count,
        "event_count": response.event_count,
        "malformed_events": response.malformed_events,
    })
}

/// Build the terminal SSE chunk that must be emitted after the provider's own
/// terminal chunk (or after aggregation) when wrapping a stream for an
/// OpenAI-compatible client. Returns `None` when no authoritative terminal
/// exists, so callers emit nothing rather than fabricating `stop`.
pub fn terminal_chunk_json(response: &NormalizedResponse) -> Option<serde_json::Value> {
    if !response.explicit_terminal {
        return None;
    }
    let finish = response.finish_reason.to_openai_chat()?;
    Some(serde_json::json!({
        "id": response.response_id.clone().unwrap_or_else(|| "chatcmpl-local".to_string()),
        "object": "chat.completion.chunk",
        "created": 0,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish,
        }],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(data: &str) -> SseEvent {
        SseEvent {
            data: vec![data.to_string()],
            event_type: None,
            is_comment: false,
        }
    }

    fn decode(feed: &str) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::new();
        let mut events = decoder.feed(feed.as_bytes());
        if let Some(last) = decoder.finish() {
            events.push(last);
        }
        events
    }

    #[test]
    fn non_streaming_completion_stop() {
        let value = serde_json::json!({
            "id": "c1", "model": "m",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        });
        let normalized = normalize_non_streaming("openai", "m", &value, None);
        assert_eq!(normalized.status, CompletionStatus::Completed);
        assert_eq!(normalized.finish_reason, NormalizedFinishReason::Stop);
        assert!(normalized.explicit_terminal);
        assert_eq!(normalized.usage.input_tokens, Some(5));
        assert_eq!(normalized.usage.visible_output_tokens, Some(3));
        assert_eq!(normalized.usage.source, UsageSource::Authoritative);
        // External view: finish_reason is truthful stop.
        assert_eq!(normalized.finish_reason.to_openai_chat(), Some("stop"));
        let ext = completion_extension(&normalized);
        assert_eq!(ext["status"], "completed");
    }

    #[test]
    fn streaming_completion_with_done() {
        let mut agg = StreamAggregator::new("openai", "m");
        let events = decode(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
             data: [DONE]\n\n",
        );
        for e in &events {
            agg.apply(e);
        }
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.content, "Hello");
        assert_eq!(out.status, CompletionStatus::Completed);
        assert!(out.explicit_terminal);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Stop);
    }

    #[test]
    fn streaming_completion_usage_in_final_chunk() {
        let mut agg = StreamAggregator::new("openai", "m");
        let events = decode(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n\
             data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n\
             data: [DONE]\n\n",
        );
        for e in &events {
            agg.apply(e);
        }
        // Terminal stop arrived on an earlier chunk in this synthetic case;
        // synthesize it so the test checks usage merging + terminal path.
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
        ));
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.usage.visible_output_tokens, Some(2));
        assert_eq!(out.usage.total_tokens, Some(11));
        assert_eq!(out.status, CompletionStatus::Completed);
    }

    #[test]
    fn streaming_completion_usage_absent_stays_unknown() {
        let mut agg = StreamAggregator::new("openai", "m");
        let events = decode(
            "data: {\"choices\":[{\"delta\":{\"content\":\"abc\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
             data: [DONE]\n\n",
        );
        for e in &events {
            agg.apply(e);
        }
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.usage.source, UsageSource::Unavailable);
        assert!(out.usage.is_empty());
        // Completion still recognized via finish_reason; usage is not invented.
        assert_eq!(out.status, CompletionStatus::Completed);
        assert!(terminal_chunk_json(&out).is_some());
    }

    #[test]
    fn finish_reason_stop_vs_length() {
        let mut length_agg = StreamAggregator::new("openai", "m");
        length_agg.apply(&event("{\"choices\":[{\"delta\":{\"content\":\"cut\"}}]}"));
        length_agg.apply(&event(
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}",
        ));
        let length_out = length_agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(length_out.finish_reason, NormalizedFinishReason::Length);
        assert_eq!(length_out.status, CompletionStatus::Incomplete);
        assert_eq!(length_out.finish_reason.to_openai_chat(), Some("length"));

        let stop_value = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        });
        let stop_out = normalize_non_streaming("openai", "m", &stop_value, None);
        assert_eq!(stop_out.status, CompletionStatus::Completed);
    }

    #[test]
    fn tool_call_completion() {
        let mut agg = StreamAggregator::new("openai", "m");
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \"}}]}}]}",
        ));
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"NYC\\\"}\"}}]}}]}",
        ));
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
        ));
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.finish_reason, NormalizedFinishReason::ToolCalls);
        assert_eq!(out.status, CompletionStatus::Completed);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_stop_reasons() {
        assert_eq!(
            NormalizedFinishReason::from_anthropic(Some("end_turn")),
            NormalizedFinishReason::Stop
        );
        assert_eq!(
            NormalizedFinishReason::from_anthropic(Some("max_tokens")),
            NormalizedFinishReason::Length
        );
        assert_eq!(
            NormalizedFinishReason::from_anthropic(Some("refusal")),
            NormalizedFinishReason::Refusal
        );
        let mut agg = StreamAggregator::new("anthropic", "m");
        agg.apply(&event(JSON_ANTHROPIC_END_TURN));
        let out = agg.finish(StreamTermination::ExplicitTerminal);
        assert_eq!(out.status, CompletionStatus::Completed);
        assert!(out.explicit_terminal);
    }

    const JSON_ANTHROPIC_END_TURN: &str =
        "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}";

    #[test]
    fn anthropic_max_tokens_is_incomplete() {
        let mut agg = StreamAggregator::new("anthropic", "m");
        agg.apply(&event(
            "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}",
        ));
        let out = agg.finish(StreamTermination::ExplicitTerminal);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Length);
        assert_eq!(out.status, CompletionStatus::Incomplete);
    }

    #[test]
    fn gemini_finish_reasons() {
        assert_eq!(
            NormalizedFinishReason::from_gemini(Some("STOP")),
            NormalizedFinishReason::Stop
        );
        assert_eq!(
            NormalizedFinishReason::from_gemini(Some("MAX_TOKENS")),
            NormalizedFinishReason::Length
        );
        assert_eq!(
            NormalizedFinishReason::from_gemini(Some("SAFETY")),
            NormalizedFinishReason::Refusal
        );
        let value = serde_json::json!({
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 5, "totalTokenCount": 9},
        });
        let usage = usage_from_gemini(&value).unwrap();
        assert_eq!(usage.input_tokens, Some(4));
        assert_eq!(usage.visible_output_tokens, Some(5));
    }

    #[test]
    fn premature_clean_eof_without_terminal_is_unknown() {
        let mut agg = StreamAggregator::new("openai", "m");
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}",
        ));
        let out = agg.finish(StreamTermination::CleanEof);
        assert_eq!(out.status, CompletionStatus::Unknown);
        assert!(!out.explicit_terminal);
        assert!(!out.status.is_success());
        // No terminal chunk may be fabricated.
        assert!(terminal_chunk_json(&out).is_none());
        let ext = completion_extension(&out);
        assert_eq!(ext["status"], "unknown");
    }

    #[test]
    fn abrupt_disconnect_is_failed() {
        let mut agg = StreamAggregator::new("openai", "m");
        agg.apply(&event("{\"choices\":[{\"delta\":{\"content\":\"x\"}}]}"));
        let out = agg.finish(StreamTermination::Disconnect);
        assert_eq!(out.status, CompletionStatus::Failed);
    }

    #[test]
    fn malformed_sse_event_is_counted() {
        let mut agg = StreamAggregator::new("openai", "m");
        let bad = decode("data: {not json}\n\n");
        for e in &bad {
            assert_eq!(agg.apply(e), ChunkSignal::Malformed);
        }
        let out = agg.finish(StreamTermination::CleanEof);
        assert_eq!(out.malformed_events, 1);
        assert_eq!(out.status, CompletionStatus::Unknown);
    }

    #[test]
    fn duplicate_terminal_events_first_wins() {
        let mut agg = StreamAggregator::new("openai", "m");
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}",
        ));
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"length\"}]}",
        ));
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Stop);
        assert_eq!(out.content, "ab");
    }

    #[test]
    fn multi_line_sse_data_lines_join() {
        let events = decode("data: {\"choices\":\ndata: [{\"delta\":{\"content\":\"x\"}}]}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].joined_data(),
            "{\"choices\":\n[{\"delta\":{\"content\":\"x\"}}]}"
        );
        // Keepalive comments are safely ignored.
        let keepalive = decode(": ping\n\ndata: [DONE]\n\n");
        assert!(keepalive.iter().any(|e| e.is_comment));
    }

    #[test]
    fn genuinely_complete_single_token_response() {
        let mut agg = StreamAggregator::new("openai", "m");
        agg.apply(&event("{\"choices\":[{\"delta\":{\"content\":\"!\"}}]}"));
        agg.apply(&event(
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
        ));
        let out = agg.finish(StreamTermination::DoneSentinel);
        assert_eq!(out.content, "!");
        assert_eq!(out.status, CompletionStatus::Completed);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Stop);
    }

    #[test]
    fn reported_one_token_with_longer_text_is_preserved() {
        // Provider usage says 1 but text is longer (e.g. reasoning counted
        // separately): neither value is rewritten.
        let value = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{"message": {"content": "clearly more than one token of visible text"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        });
        let out = normalize_non_streaming("openai", "m", &value, None);
        assert_eq!(out.usage.visible_output_tokens, Some(1));
        assert!(out.content.len() > 1);
        assert_eq!(out.status, CompletionStatus::Completed);
    }

    #[test]
    fn missing_usage_never_defaults_to_one() {
        let value = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        });
        let out = normalize_non_streaming("openai", "m", &value, None);
        assert_eq!(out.usage.source, UsageSource::Unavailable);
        assert!(out.usage.visible_output_tokens.is_none());
        assert!(out.usage.total_tokens.is_none());
    }

    #[test]
    fn reasoning_tokens_separated_from_visible() {
        let value = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{"message": {"content": "answer", "reasoning_content": "thinking"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 15},
                "total_tokens": 30,
            },
        });
        let out = normalize_non_streaming("openai", "m", &value, None);
        assert_eq!(out.reasoning, "thinking");
        assert_eq!(out.content, "answer");
        assert_eq!(out.usage.reasoning_tokens, Some(15));
        assert_eq!(out.usage.visible_output_tokens, Some(20));
    }

    #[test]
    fn regression_truncation_after_one_delta_without_terminal() {
        // Reproduces "Response was truncated before completion / one token":
        // the upstream delivered exactly one content delta and then clean EOF
        // without any finish_reason, without [DONE] after a terminal, and
        // without usage. The normalized contract must surface this as
        // Unknown/not-completed with no fabricated terminal chunk, so the
        // harness sees a truthful truncation instead of a fake success.
        let mut agg = StreamAggregator::new("llamacpp", "m");
        agg.apply(&event("{\"choices\":[{\"delta\":{\"content\":\"X\"}}]}"));
        let out = agg.finish(StreamTermination::CleanEof);
        assert_eq!(out.content, "X");
        assert_eq!(out.status, CompletionStatus::Unknown);
        assert!(!out.status.is_success());
        assert!(!out.explicit_terminal);
        assert!(terminal_chunk_json(&out).is_none());
        let diag = CompletionDiagnostics::from(&out);
        assert_eq!(diag.status, "unknown");
        // External OpenAI view: finish_reason is null (via extension), so a
        // client checking finish_reason cannot mistake this for "stop".
        let ext = completion_extension(&out);
        assert!(ext["finish_reason"].is_null());
    }

    #[test]
    fn non_streaming_missing_finish_reason_is_unknown_not_stop() {
        // HTTP 200 + JSON body with no finish_reason must not be treated as
        // completed generation. This is the non-streaming twin of the
        // regression above.
        let value = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{"message": {"content": "only one token delivered"}}],
        });
        let out = normalize_non_streaming("llamacpp", "m", &value, None);
        assert_eq!(out.status, CompletionStatus::Unknown);
        assert!(out.finish_reason.to_openai_chat().is_none());
    }

    #[test]
    fn responses_api_incomplete_max_tokens() {
        let mut agg = StreamAggregator::new("openai-responses", "m");
        agg.apply(&event(
            "{\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}",
        ));
        let out = agg.finish(StreamTermination::ExplicitTerminal);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Length);
        assert_eq!(out.status, CompletionStatus::Incomplete);
    }

    #[test]
    fn responses_api_streaming_text_deltas_and_usage() {
        let mut agg = StreamAggregator::new("openai-responses", "m");
        agg.apply(&event(
            "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}",
        ));
        agg.apply(&event(
            "{\"type\":\"response.in_progress\",\"response_id\":\"resp_1\"}",
        ));
        agg.apply(&event(
            "{\"type\":\"response.output_text.delta\",\"delta\":\"Hello \"}",
        ));
        agg.apply(&event(
            "{\"type\":\"response.output_text.delta\",\"delta\":\"world!\"}",
        ));
        agg.apply(&event("{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}"));
        let out = agg.finish(StreamTermination::ExplicitTerminal);
        assert_eq!(out.content, "Hello world!");
        assert_eq!(out.status, CompletionStatus::Completed);
        assert_eq!(out.finish_reason, NormalizedFinishReason::Stop);
        assert_eq!(out.usage.input_tokens, Some(5));
        assert_eq!(out.usage.visible_output_tokens, Some(2));
        assert_eq!(out.usage.reasoning_tokens, Some(1));
        assert_eq!(out.usage.total_tokens, Some(7));

        let res_usage = out.usage.to_responses_usage();
        assert_eq!(res_usage["input_tokens"], 5);
        assert_eq!(res_usage["output_tokens"], 2);
        assert_eq!(res_usage["total_tokens"], 7);
        assert_eq!(res_usage["output_tokens_details"]["reasoning_tokens"], 1);
    }
}
