# RAM Context Window Overflow & Hardware-Aware Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow local LLM context windows to overflow beyond GPU VRAM into WSL2 system RAM using vLLM's PagedAttention swapping (`--swap-space`) and weight offloading (`--cpu-offload-gb`), evaluated via an `llmfit`-inspired multi-tier scoring model with granular user controls in Settings.

**Architecture:** Pure Rust estimation in `estimate.rs` and `fit.rs` calculates dual context windows (pure VRAM vs. RAM-extended), RunModes (`Gpu`, `GpuRamSwap`, `CpuOffload`), and roofline speed penalties. WSL2 RAM is detected dynamically from `/proc/meminfo` and balanced with a user-configurable safety reserve and manual overrides in `state.rs`. The calculated swap and offload buffers are passed directly to `vllm` CLI flags in `server.rs`, and surfaced across Settings, Search, Library, and Servers UI in the React frontend.

**Tech Stack:** Rust (Tauri 2 backend), TypeScript, React 18, Tailwind CSS, vLLM (WSL2 Ubuntu).

**Spec:** [`docs/superpowers/specs/2026-09-17-ram-context-overflow-design.md`](file:///c:/Users/sedol/Documents/LocalLLmPanel/docs/superpowers/specs/2026-09-17-ram-context-overflow-design.md)

## Global Constraints

- Never break existing pure VRAM inference for models that fit comfortably.
- WSL2 memory probing must never panic if WSL is stopped; fallback gracefully to sensible defaults (16GB RAM).
- Pinned memory (`export VLLM_WSL2_ENABLE_PIN_MEMORY=1`) must always remain enabled when starting vLLM servers.
- All estimator math in `estimate.rs` and `fit.rs` must remain pure functions without I/O and covered by unit tests.

---

### Task 1: Multi-Tier Context & Swap Estimator (`estimate.rs`)

**Files:**
- Modify: `src-tauri/src/estimate.rs`
- Test: `src-tauri/src/estimate.rs` (inline unit tests)

**Interfaces:**
- Produces:
  - `pub struct TieredContextFit { pub vram_context: usize, pub extended_context: usize, pub swap_space_gb: usize, pub cpu_offload_gb: usize }`
  - `pub fn context_fit_tiered(vram_total_mb: f64, gpu_util: f64, ram_usable_mb: f64, weight_gb: f64, kv_bpt: f64, overhead_mb: f64, max_context: usize, allow_weight_offload: bool) -> TieredContextFit`
  - `pub fn tokens_per_sec_tiered(gpu_bandwidth_gbs: f64, ram_bandwidth_gbs: f64, params_b: f64, quant: &str, swap_space_gb: usize, cpu_offload_gb: usize, weight_gb: f64) -> f64`

- [ ] **Step 1: Write failing unit tests for tiered context and swap calculation**

Add tests to `src-tauri/src/estimate.rs` in `mod tests`:

```rust
#[test]
fn test_context_fit_tiered_vram_and_swap() {
    let bpt = kv_bytes_per_token(32, 8, 128); // 7B model KV cache rate (~131072 bytes/tok)
    // 12GB GPU (util 0.92 = 11048MB), 4.4GB weights, 2500MB overhead -> ~4048MB VRAM for KV (~32k tokens)
    // Model max context: 131,072. Usable RAM: 16,384 MB (16 GB)
    let res = context_fit_tiered(12000.0, 0.92, 16384.0, 4.4, bpt, 2500.0, 131072, true);
    assert!(res.vram_context > 30000 && res.vram_context < 35000);
    assert!(res.extended_context > res.vram_context);
    assert!(res.extended_context <= 131072);
    assert!(res.swap_space_gb > 0);
    assert_eq!(res.cpu_offload_gb, 0);
}

#[test]
fn test_context_fit_tiered_weight_offload() {
    let bpt = kv_bytes_per_token(40, 8, 128);
    // 14B model: weight 15GB on a 12GB GPU (usable VRAM ~11GB - 2.5GB overhead = 8.5GB budget) -> weights do not fit in VRAM
    let res = context_fit_tiered(12000.0, 0.92, 32768.0, 15.0, bpt, 2500.0, 32768, true);
    assert_eq!(res.vram_context, 0);
    assert!(res.cpu_offload_gb >= 6);
    assert!(res.extended_context >= 512);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src-tauri && cargo test test_context_fit_tiered`  
Expected: FAIL with `cannot find function context_fit_tiered`

- [ ] **Step 3: Implement `TieredContextFit` and `context_fit_tiered`**

In `src-tauri/src/estimate.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TieredContextFit {
    pub vram_context: usize,
    pub extended_context: usize,
    pub swap_space_gb: usize,
    pub cpu_offload_gb: usize,
}

pub fn context_fit_tiered(
    vram_total_mb: f64,
    gpu_util: f64,
    ram_usable_mb: f64,
    weight_gb: f64,
    kv_bpt: f64,
    overhead_mb: f64,
    max_context: usize,
    allow_weight_offload: bool,
) -> TieredContextFit {
    if kv_bpt <= 0.0 || max_context == 0 {
        return TieredContextFit { vram_context: 0, extended_context: 0, swap_space_gb: 0, cpu_offload_gb: 0 };
    }

    let usable_vram_mb = vram_total_mb * gpu_util;
    let weights_mb = weight_gb * 1024.0;
    let kv_vram_mb = usable_vram_mb - weights_mb - overhead_mb;

    if kv_vram_mb > 0.0 {
        let vram_ctx_raw = (kv_vram_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
        let vram_context = vram_ctx_raw.min(max_context);

        if vram_context >= max_context || ram_usable_mb <= 0.0 {
            return TieredContextFit {
                vram_context,
                extended_context: vram_context,
                swap_space_gb: 0,
                cpu_offload_gb: 0,
            };
        }

        let deficit_tokens = max_context - vram_context;
        let ram_tokens_possible = (ram_usable_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
        let extended_tokens = deficit_tokens.min(ram_tokens_possible);
        let extended_context = vram_context + extended_tokens;
        let swap_space_gb = ((extended_tokens as f64 * kv_bpt) / (1024.0 * 1024.0 * 1024.0)).ceil() as usize;

        TieredContextFit {
            vram_context,
            extended_context,
            swap_space_gb: swap_space_gb.max(1),
            cpu_offload_gb: 0,
        }
    } else if allow_weight_offload && ram_usable_mb > 0.0 {
        let shortfall_mb = (weights_mb + overhead_mb) - usable_vram_mb;
        let cpu_offload_gb = (shortfall_mb / 1024.0).ceil() as usize;
        let offload_mb = (cpu_offload_gb * 1024) as f64;

        if ram_usable_mb > offload_mb {
            let remaining_ram_mb = ram_usable_mb - offload_mb;
            let ram_tokens = (remaining_ram_mb * 1024.0 * 1024.0 / kv_bpt) as usize;
            let extended_context = ram_tokens.min(max_context).max(512);
            let swap_space_gb = ((extended_context as f64 * kv_bpt) / (1024.0 * 1024.0 * 1024.0)).ceil() as usize;

            TieredContextFit {
                vram_context: 0,
                extended_context,
                swap_space_gb: swap_space_gb.max(1),
                cpu_offload_gb,
            }
        } else {
            TieredContextFit { vram_context: 0, extended_context: 0, swap_space_gb: 0, cpu_offload_gb }
        }
    } else {
        TieredContextFit { vram_context: 0, extended_context: 0, swap_space_gb: 0, cpu_offload_gb: 0 }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src-tauri && cargo test test_context_fit_tiered`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/estimate.rs
git commit -m "feat(estimate): add tiered context fit and swap space calculations"
```

---

### Task 2: Multi-Tier Fit Engine & RunMode Classification (`fit.rs`)

**Files:**
- Modify: `src-tauri/src/fit.rs`
- Test: `src-tauri/src/fit.rs` (inline unit tests)

**Interfaces:**
- Consumes: `estimate::context_fit_tiered`, `estimate::TieredContextFit`
- Produces:
  - `pub enum RunMode { Gpu, GpuRamSwap, CpuOffload, DoesNotFit }`
  - `pub struct HardwareProfile` (expanded with RAM fields)
  - `pub struct FitResult` (expanded with `run_mode`, `vram_context`, `extended_context`, `swap_space_gb`, `cpu_offload_gb`, `ram_pct`)

- [ ] **Step 1: Write failing unit test for RunMode and tiered scoring**

In `src-tauri/src/fit.rs` `mod tests`:

```rust
#[test]
fn test_run_mode_gpu_ram_swap() {
    let hw = HardwareProfile {
        gpu_name: "RTX 5070 Ti".into(),
        vram_total_mb: 12227,
        bandwidth_gbs: 672.0,
        bandwidth_known: true,
        ram_total_mb: 24576,
        ram_usable_mb: 20480,
        ram_bandwidth_gbs: 65.0,
    };
    let v = variant("awq", false);
    let arch = arch_7b(); // 32k native context
    let r = score_variant(&hw, &v, &arch, None, 0.92, 2500.0, true, None);
    assert_eq!(r.run_mode, RunMode::GpuRamSwap);
    assert!(r.extended_context > r.vram_context);
    assert!(r.swap_space_gb > 0);
    assert_eq!(r.cpu_offload_gb, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src-tauri && cargo test test_run_mode_gpu_ram_swap`  
Expected: FAIL with compilation errors on fields and arguments

- [ ] **Step 3: Update `HardwareProfile`, `RunMode`, `FitResult`, and `score_variant`**

In `src-tauri/src/fit.rs`:

```rust
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum RunMode {
    Gpu,
    GpuRamSwap,
    CpuOffload,
    DoesNotFit,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub bandwidth_gbs: f64,
    pub bandwidth_known: bool,
    pub ram_total_mb: u64,
    pub ram_usable_mb: u64,
    pub ram_bandwidth_gbs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub verdict: FitVerdict,
    pub run_mode: RunMode,
    pub score: u8,
    pub weight_gb: f64,
    pub vram_context: usize,
    pub extended_context: usize,
    pub native_context: usize,
    pub swap_space_gb: usize,
    pub cpu_offload_gb: usize,
    pub est_tok_s: Option<f64>,
    pub measured_tok_s: Option<f64>,
    pub vram_pct: u8,
    pub ram_pct: u8,
    pub format_support: FormatSupport,
    pub reason: String,
}
```

Update `score_variant` to compute `tiered`:

```rust
pub fn score_variant(
    hw: &HardwareProfile,
    variant: &VariantInput,
    arch: &ModelArchInfo,
    measured_tok_s: Option<f64>,
    gpu_util: f64,
    vram_overhead_mb: f64,
    allow_weight_offload: bool,
    max_context_cap: Option<usize>,
) -> FitResult {
    // ... compute weight_gb and kv_bpt ...
    let target_context = max_context_cap.unwrap_or(arch.context).min(arch.context);
    let tiered = estimate::context_fit_tiered(
        hw.vram_total_mb as f64,
        gpu_util,
        hw.ram_usable_mb as f64,
        weight_gb,
        kv_bpt,
        vram_overhead_mb,
        target_context,
        allow_weight_offload,
    );

    let run_mode = if tiered.extended_context == 0 {
        RunMode::DoesNotFit
    } else if tiered.cpu_offload_gb > 0 {
        RunMode::CpuOffload
    } else if tiered.swap_space_gb > 0 {
        RunMode::GpuRamSwap
    } else {
        RunMode::Gpu
    };
    // ... score composite & reason text formatting ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src-tauri && cargo test`  
Expected: PASS (all tests in `fit.rs` updated and passing)

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/fit.rs
git commit -m "feat(fit): implement RunMode and tiered hardware fit scoring"
```

---

### Task 3: Memory Settings & Server Definitions (`state.rs`)

**Files:**
- Modify: `src-tauri/src/state.rs`
- Test: `src-tauri/src/state.rs` (config round-trip tests)

**Interfaces:**
- Produces:
  - `pub struct MemorySettings`
  - `AppConfig.memory_settings: MemorySettings`
  - `ServerDef.swap_space_gb: Option<usize>`
  - `ServerDef.cpu_offload_gb: Option<usize>`

- [ ] **Step 1: Write test for serializing and deserializing `MemorySettings`**

In `src-tauri/src/state.rs` `mod tests`:

```rust
#[test]
fn test_memory_settings_default_and_roundtrip() {
    let cfg = AppConfig::default();
    assert_eq!(cfg.memory_settings.default_gpu_mem_util, 0.92);
    assert_eq!(cfg.memory_settings.vram_overhead_mb, 2500.0);
    assert!(cfg.memory_settings.enable_ram_overflow);
    assert_eq!(cfg.memory_settings.safety_reserve_mb, 4096);
    let json = serde_json::to_string(&cfg).unwrap();
    let parsed: AppConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.memory_settings.default_gpu_mem_util, 0.92);
}
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd src-tauri && cargo test test_memory_settings`  
Expected: FAIL with `no field memory_settings`

- [ ] **Step 3: Define `MemorySettings` and update `AppConfig` and `ServerDef`**

In `src-tauri/src/state.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySettings {
    #[serde(default = "default_gpu_mem_util")]
    pub default_gpu_mem_util: f64,
    #[serde(default = "default_vram_overhead")]
    pub vram_overhead_mb: f64,
    #[serde(default = "default_true")]
    pub enable_ram_overflow: bool,
    pub manual_ram_limit_mb: Option<u64>,
    #[serde(default = "default_safety_reserve")]
    pub safety_reserve_mb: u64,
    #[serde(default = "default_true")]
    pub offload_weights_allowed: bool,
    pub max_context_cap: Option<usize>,
}

fn default_gpu_mem_util() -> f64 { 0.92 }
fn default_vram_overhead() -> f64 { 2500.0 }
fn default_safety_reserve() -> u64 { 4096 }

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            default_gpu_mem_util: 0.92,
            vram_overhead_mb: 2500.0,
            enable_ram_overflow: true,
            manual_ram_limit_mb: None,
            safety_reserve_mb: 4096,
            offload_weights_allowed: true,
            max_context_cap: None,
        }
    }
}
```

Update `ServerDef`:

```rust
pub struct ServerDef {
    // ... existing fields ...
    pub swap_space_gb: Option<usize>,
    pub cpu_offload_gb: Option<usize>,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src-tauri && cargo test test_memory_settings`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/state.rs
git commit -m "feat(state): add MemorySettings and swap/offload fields to ServerDef"
```

---

### Task 4: WSL2 RAM Probing & Tauri Commands (`wsl.rs` & `commands.rs`)

**Files:**
- Modify: `src-tauri/src/wsl.rs`
- Modify: `src-tauri/src/commands.rs`
- Test: `src-tauri/src/commands.rs`

**Interfaces:**
- Produces:
  - `wsl::detect_wsl_memory(distro: &str) -> (u64, u64)` (total_mb, available_mb)
  - Tauri commands: `get_memory_settings`, `update_memory_settings`, `get_system_memory`

- [ ] **Step 1: Write test for parsing `/proc/meminfo`**

In `src-tauri/src/wsl.rs`:

```rust
#[test]
fn test_parse_proc_meminfo() {
    let sample = "MemTotal:       24576000 kB\nMemFree:         4000000 kB\nMemAvailable:   18432000 kB\n";
    let (total_mb, avail_mb) = parse_meminfo(sample);
    assert_eq!(total_mb, 24000);
    assert_eq!(avail_mb, 18000);
}
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd src-tauri && cargo test test_parse_proc_meminfo`  
Expected: FAIL with `cannot find function parse_meminfo`

- [ ] **Step 3: Implement `parse_meminfo` and `detect_wsl_memory`**

In `src-tauri/src/wsl.rs`:

```rust
pub fn parse_meminfo(content: &str) -> (u64, u64) {
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        } else if line.starts_with("MemAvailable:") {
            avail_kb = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        }
    }
    (total_kb / 1024, avail_kb / 1024)
}

pub fn detect_wsl_memory(distro: &str) -> (u64, u64) {
    let out = std::process::Command::new("wsl.exe")
        .args(["-d", distro, "--", "cat", "/proc/meminfo"])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout);
            return parse_meminfo(&s);
        }
    }
    (16384, 12288) // Safe fallback: 16GB total / 12GB avail
}
```

- [ ] **Step 4: Update `hardware_profile` and add Tauri commands in `commands.rs`**

In `src-tauri/src/commands.rs`:
- Update `hardware_profile(state: &AppState)` to populate `ram_total_mb`, `ram_usable_mb`, and `ram_bandwidth_gbs` using `detect_wsl_memory`, the user's `safety_reserve_mb`, and `manual_ram_limit_mb`.
- Add Tauri commands:
  - `#[tauri::command] pub fn get_memory_settings(state: State<'_, AppState>) -> MemorySettings`
  - `#[tauri::command] pub fn update_memory_settings(state: State<'_, AppState>, settings: MemorySettings) -> Result<(), String>`
  - `#[tauri::command] pub fn get_system_memory(state: State<'_, AppState>) -> SystemMemoryInfo`

- [ ] **Step 5: Run tests and verify compile**

Run: `cd src-tauri && cargo test`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src-tauri/src/wsl.rs src-tauri/src/commands.rs
git commit -m "feat(commands): add WSL2 memory probing and memory settings commands"
```

---

### Task 5: Server Execution with Swap & Offload Flags (`server.rs`)

**Files:**
- Modify: `src-tauri/src/server.rs`
- Test: `src-tauri/src/server.rs` (inline test for command line builder)

**Interfaces:**
- Consumes: `ServerDef.swap_space_gb`, `ServerDef.cpu_offload_gb`
- Produces: Correct vLLM command line containing `--swap-space` and `--cpu-offload-gb`

- [ ] **Step 1: Write unit test for `build_start_command` with swap and cpu offload**

In `src-tauri/src/server.rs` `mod tests`:

```rust
#[test]
fn test_build_start_command_swap_and_cpu_offload() {
    let mut def = sample_server_def("test-server");
    def.swap_space_gb = Some(8);
    def.cpu_offload_gb = Some(4);
    let cmd = build_start_command(&def, "");
    assert!(cmd.contains("--swap-space 8"), "command must include --swap-space 8: {cmd}");
    assert!(cmd.contains("--cpu-offload-gb 4"), "command must include --cpu-offload-gb 4: {cmd}");
    assert!(cmd.contains("export VLLM_WSL2_ENABLE_PIN_MEMORY=1"));
}
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd src-tauri && cargo test test_build_start_command_swap_and_cpu_offload`  
Expected: FAIL (flags missing from generated command)

- [ ] **Step 3: Update `build_start_command` in `server.rs`**

In `src-tauri/src/server.rs`:

```rust
    if let Some(swap) = def.swap_space_gb {
        if swap > 0 {
            args.push("--swap-space".into());
            args.push(swap.to_string());
        }
    }
    if let Some(offload) = def.cpu_offload_gb {
        if offload > 0 {
            args.push("--cpu-offload-gb".into());
            args.push(offload.to_string());
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src-tauri && cargo test test_build_start_command_swap_and_cpu_offload`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/server.rs
git commit -m "feat(server): pass --swap-space and --cpu-offload-gb to vLLM"
```

---

### Task 6: Frontend Settings & API Client (`src/api.ts`, `src/types.ts`, `src/pages/Settings.tsx`)

**Files:**
- Modify: `src/types.ts`
- Modify: `src/api.ts`
- Modify: `src/pages/Settings.tsx`
- Test: Run TypeScript build verification (`npm run build`)

**Interfaces:**
- Produces:
  - `MemorySettings` interface in `types.ts`
  - `getMemorySettings()`, `updateMemorySettings(settings)` in `api.ts`
  - "Hardware & Memory Tuning" card in `Settings.tsx`

- [ ] **Step 1: Add types in `src/types.ts`**

```typescript
export interface MemorySettings {
  default_gpu_mem_util: number;
  vram_overhead_mb: number;
  enable_ram_overflow: boolean;
  manual_ram_limit_mb: number | null;
  safety_reserve_mb: number;
  offload_weights_allowed: boolean;
  max_context_cap: number | null;
}

export type RunMode = 'Gpu' | 'GpuRamSwap' | 'CpuOffload' | 'DoesNotFit';
```

- [ ] **Step 2: Add API wrappers in `src/api.ts`**

```typescript
export async function getMemorySettings(): Promise<MemorySettings> {
  return invoke('get_memory_settings');
}

export async function updateMemorySettings(settings: MemorySettings): Promise<void> {
  return invoke('update_memory_settings', { settings });
}
```

- [ ] **Step 3: Implement "Hardware & Memory Tuning" card in `src/pages/Settings.tsx`**

Add controls for:
- GPU VRAM Utilization slider (50% – 98%)
- VRAM Overhead Buffer (MB)
- Enable RAM Context Overflow toggle
- Manual RAM Budget input / Auto mode
- RAM Safety Reserve (MB)
- Weight Offload Allowed toggle
- Global Context Cap input

- [ ] **Step 4: Verify build succeeds**

Run: `npm run build`  
Expected: TypeScript check and Vite build succeed without errors.

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/api.ts src/pages/Settings.tsx
git commit -m "feat(ui): add Hardware & Memory Tuning settings panel"
```

---

### Task 7: Frontend Model Discovery & Server Creation (`src/pages/Search.tsx`, `src/pages/Library.tsx`, `src/pages/Servers.tsx`)

**Files:**
- Modify: `src/pages/Search.tsx`
- Modify: `src/pages/Library.tsx`
- Modify: `src/pages/Servers.tsx`
- Test: Run TypeScript build verification (`npm run build`)

**Interfaces:**
- Surfacing:
  - Dual context badges (`8k VRAM ➔ 64k RAM`)
  - RunMode badges (`[GPU]`, `[GPU + RAM Swap]`, `[CPU Offload]`, `[Does Not Fit]`)
  - Context slider with green/blue tier indicator and auto-filled swap space in `Servers.tsx`

- [ ] **Step 1: Update Model Cards and Variant List in `Search.tsx` & `Library.tsx`**

Render the dual context and RunMode badge:
```tsx
{variant.fit.run_mode === 'GpuRamSwap' ? (
  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs bg-cyan-950/80 text-cyan-300 border border-cyan-800">
    <span className="font-semibold">{Math.round(variant.fit.vram_context / 1000)}k VRAM</span>
    <span>➔</span>
    <span className="font-bold text-cyan-200">{Math.round(variant.fit.extended_context / 1000)}k RAM</span>
  </span>
) : (
  <span className="text-xs text-neutral-300">
    {Math.round(variant.fit.vram_context / 1000)}k VRAM
  </span>
)}
```

- [ ] **Step 2: Update New Server Drawer in `Servers.tsx`**

- When selecting context length, show marker where pure VRAM ends and RAM swap begins.
- Automatically populate `swap_space_gb` and `cpu_offload_gb` fields from the variant's `fit` calculation.

- [ ] **Step 3: Verify build succeeds**

Run: `npm run build`  
Expected: Build passes with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/pages/Search.tsx src/pages/Library.tsx src/pages/Servers.tsx
git commit -m "feat(ui): display dual context badges, RunModes, and swap controls"
```

---

## Plan Review & Verification Checklist

1. **Rust Test Suite**:
   ```bash
   cd src-tauri
   cargo test
   ```
   Must verify:
   - Tiered context math correctly allocates VRAM first, then RAM swap.
   - RunMode transitions correctly between `Gpu`, `GpuRamSwap`, `CpuOffload`, and `DoesNotFit`.
   - `/proc/meminfo` parser correctly extracts memory in MB.
   - `build_start_command` produces valid vLLM arguments.

2. **Frontend Build**:
   ```bash
   npm run build
   ```
   Must compile cleanly with no TypeScript diagnostics.
