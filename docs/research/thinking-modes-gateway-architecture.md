# Research Report: Thinking Mode Discovery, Lifecycle, and Advertising Flow in LocalLLmPanel Unified Gateway

## Executive Summary

When developers connect coding harnesses (such as **Pi**, **Roo Code**, **Cline**, **Continue**, or **OpenRouter-compatible extensions**) to the LocalLLmPanel unified gateway, reasoning-capable models (e.g. Qwen 2.5/3.x, DeepSeek R1/Distill, Spark, GLM, etc.) frequently report having **no available thinking modes / reasoning effort levels** in the client UI.

This investigation analyzes the complete end-to-end lifecycle of how LocalLLmPanel:
1. Detects and extracts reasoning metadata from local weights and running backends (llama.cpp and vLLM).
2. Generates the catalog schema across `/v1/models`, `/api/tags`, and `/api/show`.
3. Normalizes incoming inference requests across various harness dialects.
4. Identifies the exact architectural and contractual breakdown causing coding harnesses to report zero available thinking levels.

---

## 1. End-to-End Gateway Architecture Flow

```
+--------------------------------------------------------------------------------------------------+
| 1. Model Discovery & Template Analysis                                                            |
|    - ServerDef (backend: llamacpp | vllm, task: instruct, port, args, jinja)                     |
|    - read_model_chat_template: GGUF metadata / disk tokenizer_config / http://127.0.0.1:{p}/props    |
|    - analyze_chat_template -> TemplateEffortInfo (ladder, default_effort, is_toggle_only, etc.)  |
|    - Cached in AppState::template_effort_cache (TTL: 300,000ms = 5 minutes)                       |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
| 2. Capability Ingestion (model_traits & model_row_with_template)                                 |
|    - looks_like_reasoning_model(name) || info.is_reasoning || configured flags                   |
|    - Computes ReasoningMode (None | Effort | AlwaysOn)                                           |
|    - Calculates supported_efforts & thinking_level_map                                           |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
| 3. Advertising Layer (GET /v1/models, /v1/models/{id}, /api/tags, /api/show)                     |
|    - Emits OpenAI-style JSON list                                                                |
|    - Current: sets row["reasoning"] = boolean (traits.reasoning)                                 |
|    - Current: puts "supportsReasoningEffort" inside row["compat"] as boolean                     |
|    - Current: sets thinkingLevelMap with keys ["off", "minimal", "low", "medium", "high", ...]   |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
| 4. Inbound Request Normalization (POST /v1/chat/completions)                                     |
|    - apply_harness_precedence_with_info                                                          |
|    - harness_effort extracts: reasoning_effort, thinking_effort, reasoning.effort, think, etc.   |
|    - harness_toggle extracts: chat_template_kwargs.enable_thinking, thinking.type, think, etc.   |
|    - Canonicalizes request to match backend requirements (vLLM parser vs llama.cpp Jinja)        |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Root Cause Analysis: Why Coding Harnesses See No Thinking Modes

Detailed investigation against harness schemas and first-party provider documentation reveals **5 primary failure vectors**:

### Vector 1: Roo Code / Cline Schema Incompatibility
* **Primary Source**: `packages/types/src/model.ts` in Roo-Code/Cline (`@roo-code/types`).
* **Harness Expectation**:
  ```typescript
  export const modelInfoSchema = z.object({
    supportsReasoning: z.boolean().optional(),
    supportsReasoningEffort: z.array(z.string()).optional(), // <--- ARRAY of strings, not boolean
    reasoningEffort: z.string().optional(),
    ...
  });
  ```
  Roo Code inspects `model.supportsReasoningEffort` at the top level of the model object. If present as an array (e.g. `["low", "medium", "high"]`), it renders the effort dropdown. If boolean `true`, some components fall back, but Zod schema parsing rejects or ignores non-array values.
* **Gateway Flaw**:
  In `LocalLLmPanel/src-tauri/src/gateway.rs`:
  - `supportsReasoningEffort` is **omitted from the top level** of `row`.
  - Inside `row["compat"]`, `supportsReasoningEffort` is serialized as a **boolean** (`supports_effort_parameter`), rather than an array of supported effort levels (`["low", "medium", "high"]`).
  - Result: Roo Code sees `supportsReasoningEffort: undefined` at the root, completely hiding the thinking modes.

### Vector 2: OpenRouter & OpenAI Proxy Catalog Contract Violation
* **Primary Source**: OpenRouter Model Catalog Specification (`https://openrouter.ai/api/v1/models`).
* **Harness Expectation**:
  In OpenRouter-compatible harnesses, `model.reasoning` is an **object**:
  ```json
  "reasoning": {
    "mandatory": false,
    "default_enabled": true,
    "default_effort": "medium",
    "supported_efforts": ["low", "medium", "high"]
  }
  ```
  Harnesses test `model.reasoning?.supported_efforts?.length > 0`.
* **Gateway Flaw**:
  In `src-tauri/src/gateway.rs` (lines 1150-1175):
  The gateway constructs a comprehensive `reasoning` object, but then overrides `row["reasoning"]` with a raw boolean:
  ```rust
  "reasoning": traits.reasoning, // boolean: true
  "reasoning_config": reasoning,
  "thinking": reasoning,
  ```
  When harnesses execute `if (model.reasoning?.supported_efforts)`, evaluating `true.supported_efforts` yields `undefined`. Consequently, harnesses conclude the model has no configurable thinking levels.

### Vector 3: Suppression of Reasoning Effort on Toggle-Only Models
* **Primary Source**: `src-tauri/src/gateway.rs` line 1004 & line 1150.
* **Gateway Code**:
  ```rust
  let supports_effort_parameter = traits.reasoning_mode != ReasoningMode::None
      && supports_template_controls
      && !is_toggle_only; // <--- SUPPRESSION
  ```
* **Flaw**:
  For models using toggle-only templates (e.g. DeepSeek R1 Distill, Qwen 2.5 with `enable_thinking`), `is_toggle_only` is true. The gateway explicitly sets `supports_effort_parameter = false`, removes `reasoning_effort` from `supported_parameters`, and omits it from advertised controls.
  However, in `apply_harness_precedence_with_info` (lines 2280-2315), the gateway **already contains logic** to accept `reasoning_effort: "low"` or `"high"` from harnesses and automatically map it to `enable_thinking: true/false`.
  Because the gateway refuses to advertise `reasoning_effort`, the client never presents the option, despite the gateway being fully capable of translating it.

### Vector 4: Pi Coding Agent (`@earendil-works/pi`) Thinking Level Map Drop
* **Primary Source**: `earendil-works/pi` (`packages/coding-agent/docs/models.md`).
* **Harness Expectation**:
  Pi maps internal levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) through `thinkingLevelMap`.
  - If a level is mapped to `null`, Pi hides that level.
  - Pi's default startup level is `"medium"`.
* **Gateway Flaw**:
  When a model's Jinja template exposes an effort ladder of `["low", "high"]` (e.g., Spark or Qwen 3.8):
  `level_map_val("medium")` returns `serde_json::Value::Null`.
  Because `"medium"` is `null`, Pi considers the default level invalid or disabled and does not surface the thinking tier.
  Furthermore, `thinking_level_map["off"]` maps to `"none"`, but if a harness uses `"none"` as key or expects boolean/null semantics, it encounters missing keys.

### Vector 5: GGUF & WSL Template Resolution Fragility
* **Primary Source**: `fetch_template_info` (lines 1690-1725).
* **Flaw**:
  When llama-server is booting up, `read_model_chat_template` falls back to `GET http://127.0.0.1:{port}/props` with a **strict 500ms timeout**.
  If the server is still loading model weights or under initial I/O pressure:
  1. The request times out.
  2. `fetch_template_info` returns `TemplateEffortInfo::default()` (all reasoning flags `false`).
  3. `template_info_for` caches this empty default in `state.template_effort_cache` for **5 full minutes** (`EFFORT_LADDER_TTL_MS = 300_000`).
  4. Subsequent `/v1/models` requests during that window report `reasoning: false` and omit `thinkingLevelMap` entirely.

---

## 3. Comparative Matrix of Harness Expectations

| Harness / Specification | Discovery Field Checked | Format Expected | Gateway Current Behavior | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Roo Code / Cline** | `supportsReasoningEffort` | `string[]` (e.g. `["low", "medium", "high"]`) at model root | Inside `compat` as a boolean (`true`/`false`) | **Broken** |
| **Roo Code / Cline** | `supportsReasoning` | `boolean` at model root | `row["supports_reasoning"]` (snake_case) | **Broken** (case mismatch) |
| **OpenRouter / LiteLLM** | `reasoning` | Object with `supported_efforts: string[]` | Overwritten as boolean `true` | **Broken** |
| **OpenRouter / LiteLLM** | `supported_parameters` | Array containing `"reasoning"`, `"reasoning_effort"` | Only contains `"reasoning_effort"` (or neither if toggle) | **Partial** |
| **Pi (`pi-ai`)** | `thinkingLevelMap` | `Record<string, string \| null>` at model root | Present, but `medium` is `null` when ladder is `["low", "high"]` | **Degraded** |
| **Generic OpenAI Probes**| `GET /models` | List of models | Route does not exist (only `/v1/models`) | **404 Error** |
| **Ollama Harnesses** | `GET /api/tags` | Capabilities in model details | No capabilities or reasoning metadata included | **Silent Drop** |

---

## 4. Production-Ready Redesign Principles

To establish a bulletproof advertising architecture:
1. **Multi-Dialect Compatibility at Model Root**:
   Expose reasoning capabilities simultaneously in:
   - Roo Code format: `supportsReasoning: true`, `supportsReasoningEffort: ["low", "medium", "high", ...]`
   - OpenRouter format: `reasoning: { supported_efforts: [...], default_effort: "...", mandatory: ... }`
   - Pi format: `thinkingLevelMap` with fallback clamping for missing intermediate tiers (e.g. medium -> high)
   - OpenAI/LM Studio format: `capabilities: { reasoning: true, thinking: true }`
2. **Virtual Effort Synthesis for Toggle Models**:
   When a model supports binary thinking (`enable_thinking`), advertise virtual effort levels `["none", "low", "medium", "high"]` with a clear mapping. The inbound normalizer already translates effort to toggle; advertising must match execution capabilities.
3. **Route Aliasing**:
   Mirror `GET /models` to `GET /v1/models` and `GET /models/{id}` to `GET /v1/models/{id}`.
4. **Resilient Template Caching**:
   Do NOT cache failed template lookups for 5 minutes. If a template fetch fails or times out during boot, cache with a 2-second retry TTL until successfully retrieved.
