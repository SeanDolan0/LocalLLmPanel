# 0003: Consolidate Model Discovery, Metadata Resolution, and Inventory into ModelCatalog

Model discovery, HuggingFace metadata resolution, local filesystem library inventory, and chat template extraction were previously fragmented across `src-tauri/src/hf.rs`, `src-tauri/src/hf_discovery.rs`, `src-tauri/src/model_registry.rs`, `src-tauri/src/model_intelligence.rs`, `src-tauri/src/recommendation.rs`, `src-tauri/src/gateway.rs`, and six disconnected cache maps in `AppState`. We decided to consolidate these responsibilities into an encapsulated `ModelCatalog` deep module, providing unified multi-tier caching, two-phase GGUF hydration, authoritative 3-tier context resolution, mtime-invalidated local inventory, and model-centric template analysis.

## Status

Accepted

## Context

The model management and discovery subsystems suffered from significant architectural complexity:
1. **Cache Proliferation & Invalidation Blindspots**: Six separate cache stores (`enrichment_cache`, `quant_cache`, `rec_cache`, `discovery_cache`, `template_effort_cache`, `model_registry`) operated without cross-invalidation. Port-based keying (`port: u16`) for chat templates caused cache poisoning across server restarts.
2. **Eager Hydration & Rate Limiting**: Eager GGUF tree resolution triggered dozens of parallel network requests during search, risking HuggingFace HTTP 429 rate limit errors for unauthenticated users.
3. **Conflicting Context Lengths**: Architectural context capacity (`config.json`), hardware recommended context, and live runtime context were conflated across multiple endpoints.
4. **Scattered Intelligence & Duplicated Parsing**: Parameter regex extraction, base model inference, and speculative draft pairings were implemented redundantly across four modules.
5. **Slow Library Scanning**: Library inventory scans triggered expensive WSL bash and Python scripts on every page view regardless of whether underlying directories changed.

## Decision

1. **Encapsulated Deep Module (`ModelCatalog`)**: Create `src-tauri/src/model_catalog/` as the single authoritative authority for online discovery, metadata resolution, local artifact inventory, and chat template capabilities.
2. **Unified Multi-Tier Cache**: Consolidate disconnected in-memory and disk caches into a unified cache layer with model-centric keys (`model_id` / `artifact_path`) and centralized invalidation on library mutations or user refresh.
3. **Two-Phase GGUF Hydration**: Discovery feeds and search return normalized candidates immediately using cached metadata and embedded tables; deep repo tree file verification and multi-shard grouping are hydrated on demand when viewing details or starting a download.
4. **Authoritative 3-Tier Context Separation**: `CatalogModelDetails` explicitly distinguishes `native_max_context` (model architectural capability), `recommended_context` (hardware ceiling from `HardwareSizer`), and `runtime_active_context` (live server configuration).
5. **Cached Local Inventory with Mtime Invalidation**: Local models across WSL (`~/.cache/huggingface/hub`) and Windows native GGUF storage are cached in memory and re-scanned only when directory timestamps change or when the user initiates an explicit refresh.
6. **First-Class Speculative Draft Association**: Catalog models maintain indexed mappings to compatible low-latency draft models based on architecture family and tokenizer vocabulary compatibility.
7. **Model-Centric Chat Template Keying**: Chat templates and reasoning effort capabilities are keyed by `model_id` or artifact path, reading directly from tokenizer config or GGUF metadata before server launch and eliminating port-keyed transient caching.

## Consequences

- **Positive**: Eliminates 6 fragmented cache maps and port-keyed poisoning; eliminates HuggingFace API rate-limiting spikes; establishes clear context semantics; speeds up library listing; provides seamless speculative decoding draft suggestions.
- **Negative / Risks**: Requires maintaining backward compatibility with existing Tauri IPC command signatures (`discovery_feed`, `search_models_with_fit`, `model_stats`, `library_list`) via the strangler pattern.
