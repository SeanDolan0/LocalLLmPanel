# 0002: Consolidate Hardware Detection and Memory Estimation into HardwareSizer

Hardware topology detection, VRAM reservations, model context sizing, and memory spill calculations were previously fragmented across `src-tauri/crates/llmfit-core/src/hardware.rs`, `src-tauri/src/system_info.rs`, `src-tauri/src/estimate.rs`, `src-tauri/src/fit.rs`, `src-tauri/src/llmfit_adapter.rs`, and duplicated in TypeScript on the frontend (`src/utils/weightEstimation.ts`). We decided to consolidate these responsibilities into an encapsulated `HardwareSizer` deep module, providing unified hardware discovery, authoritative dual-pool memory analysis (host vs. WSL2), coalesced VRAM reservations, and server allocation reporting.

## Status

Accepted

## Context

Hardware detection and sizing logic suffered from several architectural defects:
1. **Fragmented Multi-Path Probing**: Windows host GPU/RAM and WSL2 virtual machine memory were probed independently through divergent execution paths, resulting in conflicting capacity reports and race conditions.
2. **Competing Fit Engines**: Model fit was evaluated concurrently by `src-tauri/src/fit.rs` and `llmfit-core::fit`, creating inconsistent scoring between HuggingFace discovery search and offline model catalogs.
3. **Leaked Spill Arithmetic**: Frontend components (`Dashboard.tsx`, `Servers.tsx`) duplicated weight calculations and hardcoded VRAM overhead rules (e.g. 1.0 GB constants), bypassing user-configured `MemorySettings`.
4. **Phantom Free Memory**: In-flight server startups were tracked out-of-band via static maps, so live `nvidia-smi` telemetry showed uncommitted VRAM as free, leading to concurrent oversubscription.

## Decision

1. **Encapsulated Deep Module (`HardwareSizer`)**: Create `src-tauri/src/hardware_sizer/` as the single authoritative engine for hardware detection, memory topology, and model sizing.
2. **Two-Tier Hardware Topology Caching**: Physical specs (GPU device name, 64-bit Registry/SMBIOS VRAM, physical DIMM capacities, CPU cores) are discovered and cached once on application startup. Dynamic telemetry (free VRAM, current RAM utilization) is queried on the polling cycle.
3. **Dual Execution Environment Memory Pools**: `HardwareSizer` explicitly models both Windows Host RAM (for native llama.cpp) and WSL2 VM RAM (for vLLM), scoring model fit against the address space of the target runtime engine.
4. **Coalesced Effective Available VRAM**: `HardwareSizer::effective_free_vram_mb()` automatically computes `vram_free_mb.saturating_sub(reserved_vram_mb)`, reconciling live telemetry with in-flight server startups to prevent race conditions.
5. **Unified Canonical Fit Engine**: Consolidate `src-tauri/src/fit.rs` and `estimate.rs` into `HardwareSizer`, retiring duplicated scoring logic.
6. **Backend Allocation Reporting & Frontend Thinning**: The backend computes and emits `AllocationReport` directly in `env_status` and server snapshots. Frontend `src/utils/weightEstimation.ts` and inline alert math are deprecated and replaced by binding directly to the backend payload.

## Consequences

- **Positive**: Single source of truth for all memory math; zero drift between frontend UI alerts and backend process admission; eliminates startup VRAM race conditions; eliminates expensive PowerShell registry queries on recurring poll loops.
- **Negative / Risks**: Requires careful migration of existing frontend state consumers in `Dashboard.tsx` and `Servers.tsx` to read the unified `AllocationReport` without breaking UI reactivity.
