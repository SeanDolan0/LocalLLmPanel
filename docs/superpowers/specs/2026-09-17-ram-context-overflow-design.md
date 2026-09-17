# Using RAM to Overflow Context Window with Hardware-Aware Fit Scoring

## Problem

Currently, LocalLLmPanel restricts context window estimation and model scoring exclusively to available GPU VRAM (`usable_vram_mb - weight_gb * 1024 - overhead_mb`). Models requiring large context windows (such as 32k, 64k, or 128k) or models with weights that slightly exceed VRAM are marked as `DoesNotFit` or capped at very low token limits. 

However, modern systems often have ample system RAM (e.g., 24–64 GB inside WSL2). Runtimes like vLLM natively support PagedAttention CPU block swapping (`--swap-space`) and CPU model weight offloading (`--cpu-offload-gb`), allowing users to dramatically expand their usable context window into RAM without running out of VRAM.

## Goal

1. Automatically detect WSL2 available memory and host system RAM, allowing user-configured budgets in Settings.
2. Extend the heuristic estimators (`estimate.rs` and `fit.rs`) with a multi-tier memory model inspired by `AlexsJones/llmfit`:
   - Dual context computation: pure GPU VRAM context vs. extended context with RAM overflow.
   - RunMode classification: `GPU`, `GpuRamSwap`, `CpuOffload`, `DoesNotFit`.
   - Speed estimation with PCIe and DDR bandwidth degradation modeling.
3. Automatically configure `--swap-space` and `--cpu-offload-gb` flags when launching vLLM servers.
4. Provide clear visual cues in the frontend (dual-context badges `8k VRAM ➔ 64k RAM`, `llmfit`-style RunMode tags, and memory tooltips).

---

## 1. Hardware Profiling & Memory Hierarchy

### 1.1 WSL2 & Host RAM Discovery

When probing system hardware (in `wsl.rs` and `commands.rs`), the system executes `cat /proc/meminfo` (or `free -b`) inside the WSL2 distro:
- `MemTotal`: Total memory configured for the WSL2 VM (governed by `.wslconfig`).
- `MemAvailable`: Instantaneous unreserved memory safe to allocate without triggering the Linux OOM killer.
- Host Windows RAM is also checked via Win32 API / `sysinfo` for diagnostic awareness.

### 1.2 Configuration & Settings (`state.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    /// Whether RAM overflow is allowed for extending context beyond VRAM. Default: true.
    pub enable_ram_overflow: bool,
    /// Manual override in MB for usable RAM budget. If None, auto-detects from WSL2.
    pub manual_ram_limit_mb: Option<u64>,
    /// Safety headroom in MB to reserve for OS and WSL daemons. Default: 4096 (4 GB).
    pub safety_reserve_mb: u64,
    /// Allow offloading model weights to CPU RAM when model > VRAM. Default: true.
    pub offload_weights_allowed: bool,
}
```

### 1.3 Expanded `HardwareProfile` (`fit.rs`)

```rust
#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub bandwidth_gbs: f64,
    pub bandwidth_known: bool,
    pub ram_total_mb: u64,
    pub ram_usable_mb: u64,
    pub ram_bandwidth_gbs: f64, // Default ~45.0 GB/s for DDR4 / ~65.0 GB/s for DDR5
}
```

---

## 2. Mathematical Estimator & Scoring Model

### 2.1 Multi-Tier Context Calculation (`estimate.rs`)

For a model with parameter count $P$ (in billions), weight $W$ (GB), and fp16 KV cache rate $\text{kv\_bpt}$ (bytes/token):

1. **VRAM Context**:
   $$\text{kv\_vram\_mb} = (\text{vram\_total\_mb} \times \text{gpu\_util}) - (W \times 1024) - \text{overhead\_mb}$$
   $$\text{vram\_context} = \begin{cases} 
     \min\left(\text{max\_context},\; \frac{\text{kv\_vram\_mb} \times 1024^2}{\text{kv\_bpt}}\right) & \text{if } \text{kv\_vram\_mb} > 0 \\
     0 & \text{otherwise}
   \end{cases}$$

2. **RAM Context Overflow (`--swap-space`)**:
   When $\text{vram\_context} < \text{max\_context}$ and $\text{ram\_usable\_mb} > 0$:
   $$\text{deficit\_tokens} = \text{max\_context} - \text{vram\_context}$$
   $$\text{ram\_tokens\_possible} = \frac{\text{ram\_usable\_mb} \times 1024^2}{\text{kv\_bpt}}$$
   $$\text{extended\_tokens} = \min(\text{deficit\_tokens},\; \text{ram\_tokens\_possible})$$
   $$\text{extended\_context} = \text{vram\_context} + \text{extended\_tokens}$$
   $$\text{swap\_space\_gb} = \left\lceil \frac{\text{extended\_tokens} \times \text{kv\_bpt}}{1024^3} \right\rceil$$

3. **Weight Offloading (`--cpu-offload-gb`)**:
   When $W \times 1024 + \text{overhead\_mb} > \text{vram\_total\_mb} \times \text{gpu\_util}$:
   $$\text{shortfall\_mb} = (W \times 1024 + \text{overhead\_mb}) - (\text{vram\_total\_mb} \times \text{gpu\_util})$$
   $$\text{cpu\_offload\_gb} = \left\lceil \frac{\text{shortfall\_mb}}{1024} \right\rceil$$
   If $\text{ram\_usable\_mb} > \text{cpu\_offload\_gb} \times 1024$, the model fits via hybrid execution; remaining RAM is used for context swap.

### 2.2 RunMode & Fit Verdicts (`fit.rs`)

```rust
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum RunMode {
    Gpu,          // Weights + context fit 100% in VRAM
    GpuRamSwap,   // Weights in VRAM; context overflows to RAM swap
    CpuOffload,   // Weights partially offloaded to RAM
    DoesNotFit,   // Exceeds combined VRAM + RAM
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

### 2.3 Speed & Roofline Modeling

- **Pure GPU**: $\text{tok/s} = \frac{\text{gpu\_bandwidth} \times 10^9 \times 0.5}{P \times 10^9 \times \text{bytes\_per\_param}}$
- **GpuRamSwap**: Base GPU tok/s discounted by swap penalty:
  $$\text{tok/s}_{\text{swap}} = \text{tok/s}_{\text{gpu}} \times \left(1.0 - 0.20 \times \frac{\text{swap\_space\_gb}}{W + \text{swap\_space\_gb}}\right)$$
- **CpuOffload**: Bandwidth bottlenecked by PCIe 4.0/5.0 bus (~20 GB/s):
  $$\text{tok/s}_{\text{offload}} = \frac{20.0 \times 10^9 \times 0.5}{P \times 10^9 \times \text{bytes\_per\_param}} \quad (\sim 4\text{--}12\text{ tok/s})$$

---

## 3. Server Execution Integration (`server.rs`)

In `server.rs::build_start_command`:
1. Injects `--swap-space <swap_space_gb>` when `swap_space_gb > 0`.
2. Injects `--cpu-offload-gb <cpu_offload_gb>` when `cpu_offload_gb > 0`.
3. Retains `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` to guarantee fast host-to-device DMA memory transfers during PagedAttention swapping.

---

## 4. Frontend Presentation Layer (`src/`)

1. **Settings (`Settings.tsx`)**:
   - Card displaying detected WSL2 memory ceiling and host physical RAM.
   - Toggle to enable/disable RAM context overflow.
   - Input for manual RAM budget override.
   - Toggle for weight offloading.
2. **Model Cards & Discovery (`Search.tsx`, `Library.tsx`, `api.ts`)**:
   - **Context Badge**: `8k VRAM ➔ 64k RAM` (or `32k VRAM` if non-overflowed).
   - **RunMode Badge**:
     - `[GPU]` (emerald)
     - `[GPU + RAM Swap]` (cyan/blue)
     - `[CPU Offload]` (amber)
     - `[Does Not Fit]` (rose)
   - Tooltip displaying VRAM breakdown, required RAM swap, and speed impact.
3. **Server Modal (`Servers.tsx`)**:
   - Context slider displaying VRAM zone (green) and RAM overflow zone (blue).
   - Advanced options pre-populated with calculated `swap_space_gb` and `cpu_offload_gb`.

---

## 5. Verification Plan

### 5.1 Unit Tests (Pure Rust)
- `cargo test`:
  - Verify context expansion math for 0.5B, 7B, 14B, 27B models on 8GB, 12GB, 16GB, and 24GB GPUs with varying RAM pools.
  - Verify `RunMode` transitions: `Gpu` $\to$ `GpuRamSwap` $\to$ `CpuOffload` $\to$ `DoesNotFit`.
  - Verify swap space rounding and roofline speed estimation penalties.

### 5.2 End-to-End Server Validation (WSL2 Integration)
- Verify `build_start_command` produces correct flags:
  - e.g., `--max-model-len 32768 --swap-space 8`.
- Verify vLLM starts up and serves requests with context lengths exceeding GPU VRAM capacity without OOM crashes.
