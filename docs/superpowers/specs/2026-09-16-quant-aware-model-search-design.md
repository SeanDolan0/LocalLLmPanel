# Dynamic Quant-Aware Model Search with System-Fit Recommendations

## Problem

LocalLLmPanel's search page seeds with a hardcoded 13-model preset list (`src/data/recommended.ts`) and shows a single-quant estimate per model. There is no quant variant discovery — the detail modal merely suggests repo name suffixes like `-AWQ`. Users can't see which quantized versions exist on HF, how each fits their GPU, or which quant+model combination gives the best experience for their hardware.

## Goal

Replace the static preset list with dynamic, hardware-aware discovery: for any searched or recommended model, automatically find every servable quant variant on Hugging Face, score each against the user's detected GPU, and surface an honest ranked fit — per variant, not just per model.

## Architecture Decision: Hybrid Scoring

Backend `fit.rs` computes raw fit metrics (usable context, weight GB, tok/s estimate, verdict) and returns them per quant variant. Frontend uses those pre-computed values for its existing UI scoring/sorting without re-deriving the estimate math. The frontend's `computeLlmfitScore` / 4-pillar display stays for presentation-layer concerns.

---

## 1. Quant Discovery (`hf.rs` additions)

### 1.1 Cross-Repo Pattern (AWQ/GPTQ/FP8/BNB)

For a base model like `Qwen/Qwen2.5-7B-Instruct`, search HF API for sibling repos:

**Suffix convention** — search for:
- `{org}/{model_name}-AWQ`
- `{org}/{model_name}-GPTQ`, `{org}/{model_name}-GPTQ-Int4`, `{org}/{model_name}-GPTQ-Int8`
- `{org}/{model_name}-FP8`, `{org}/{model_name}-FP8-dynamic`
- `{org}/{model_name}-bnb-4bit`

**Known quantizer publishers** — search for:
- `neuralmagic/{base_name}*`
- `hugging-quants/{base_name}*`
- `ISTA-DASLab/{base_name}*`
- `TheBloke/{base_name}*`

**Verification**: For each candidate, confirm it's a quant of the target model by checking:
1. `config.json` → `base_model` or `_name_or_path` field matches the base model ID
2. Name substring overlap (org-agnostic model name match)

Reject false positives silently.

### 1.2 Same-Repo GGUF Pattern

For repos like `unsloth/Qwen2.5-7B-Instruct-GGUF` or `bartowski/Qwen2.5-7B-Instruct-GGUF`:

1. GET `https://huggingface.co/api/models/{repo_id}` — response includes `siblings` (file list with `rfilename` and `size`)
2. Filter to files ending `.gguf`
3. Parse quant label from filename against pattern table:
   - `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`
   - `Q4_0`, `Q4_K_S`, `Q4_K_M`
   - `Q5_0`, `Q5_K_S`, `Q5_K_M`
   - `Q6_K`, `Q8_0`
   - `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ4_XS`, `IQ4_NL`
   - Unsloth dynamic: `UD-Q4_K_XL`, `UD-IQ1_S`, etc.
4. Use each file's `size` field as exact weight bytes (no dtype guessing needed)
5. Also search for GGUF repos by pattern: `{publisher}/{base_name}-GGUF` for publishers `unsloth`, `bartowski`, `TheBloke`

### 1.3 Fallback

If neither pattern finds any variants, the model still appears with its base FP16 entry. The UI allows the user to paste a specific quant repo ID manually (existing behavior preserved via the repo ID input in the detail modal).

### 1.4 Data Structure

```rust
#[derive(Debug, Clone, Serialize)]
pub enum QuantFormat {
    FP16,
    FP8,
    AWQ,
    GPTQ,
    BNB,
    GGUF { quant_label: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantVariant {
    pub repo_id: String,
    pub format: QuantFormat,
    pub label: String,
    pub weight_bytes: Option<u64>,
    pub params_b: Option<f64>,
    pub gguf_file: Option<String>,
    pub vllm_native: bool,
}
```

### 1.5 Concurrency

- **Enrichment**: Fix existing sequential enrichment loop → `JoinSet` + `Semaphore(10)`.
- **Quant discovery**: Separate `Semaphore(6)` for discovery HTTP calls.
- Discovery is best-effort: timeouts or failures for individual quant searches don't fail the whole search.

---

## 2. Fit Scoring (`fit.rs`)

New module: `src-tauri/src/fit.rs`. Pure functions, no I/O, unit-testable. Composes `estimate.rs`.

### 2.1 Types

```rust
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FitVerdict { Comfortable, Constrained, DoesNotFit }

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FormatSupport { Native, Experimental }

pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub bandwidth_gbs: f64,
    pub bandwidth_known: bool,
}

pub struct ModelArchInfo {
    pub params_b: Option<f64>,
    pub context: usize,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
}

pub struct FitResult {
    pub verdict: FitVerdict,
    pub score: u8,
    pub weight_gb: f64,
    pub usable_context: usize,
    pub native_context: usize,
    pub est_tok_s: Option<f64>,
    pub measured_tok_s: Option<f64>,
    pub vram_pct: u8,
    pub format_support: FormatSupport,
    pub reason: String,
}
```

### 2.2 Verdict Thresholds

```rust
pub const COMFORTABLE_MAX_RATIO: f64 = 0.60;
pub const CONSTRAINED_MAX_RATIO: f64 = 0.95;
pub const OVERHEAD_MB: f64 = 2500.0;
pub const GPU_UTIL_DEFAULT: f64 = 0.92;
```

### 2.3 Core Functions

- `score_variant(hw, variant, arch, measured) -> FitResult`
- `rank_variants(results: &mut [(QuantVariant, FitResult)])` — best score first, native > GGUF at equal score
- `best_variant(results, preferred_format) -> usize`

### 2.4 Score Computation

1. **Weight GB**: GGUF → `weight_bytes / 1e9`; others → `estimate::weight_gb(params_b, quant)`
2. **VRAM ratio**: `(weight_gb * 1024 + OVERHEAD_MB) / (vram_total_mb * GPU_UTIL)`
3. **Verdict**: ratio vs thresholds
4. **Usable context**: `estimate::context_fit(...)`, clamped to native context
5. **Tok/s**: `estimate::tokens_per_sec(...)`, measured wins if available
6. **Composite score**: fit (40%) + speed (30%) + context utilization (30%), crushed to ≤25 if DoesNotFit

---

## 3. Data Sourcing Improvements

### 3.1 Params — Primary Path
`GET /api/models/{id}?expand[]=safetensors` — use `safetensors.parameters.total`.

### 3.2 Params — Fallback
Existing `safetensors.index.json` → `metadata.total_size` / dtype path.

### 3.3 Context
- Add `model_config` nesting search
- Apply `rope_scaling.factor` when type ∈ {`yarn`, `linear`, `dynamic`}

### 3.4 Architecture Dims
Consistent nesting fallback for `num_hidden_layers`, `num_key_value_heads`, `head_dim` across `text_config` / `config` / `model_config`.

### 3.5 Enrichment Caching
`HashMap<String, CachedEnrichment>` in `AppState`, TTL 1 hour.

---

## 4. New Tauri Commands

### 4.1 `search_models_with_fit`
Search → enrich → discover quants → score → rank → return `Vec<ModelWithFit>`.

### 4.2 `recommended_models`
Fetch HF trending text-generation models (limit=30) → same pipeline → return top ~20 ranked. Cache 10 minutes.

### 4.3 Return Shape

```rust
pub struct ModelWithFit {
    pub id: String,
    pub downloads: i64,
    pub likes: i64,
    pub trending_score: f64,
    pub pipeline_tag: Option<String>,
    pub params_b: Option<f64>,
    pub context: Option<usize>,
    pub context_source: Option<&'static str>,
    pub context_estimated: bool,
    pub head_dim: Option<usize>,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub variants: Vec<QuantVariantWithFit>,
    pub best_variant_idx: usize,
}

pub struct QuantVariantWithFit {
    pub variant: QuantVariant,
    pub fit: FitResult,
}
```

### 4.4 Backward Compatibility
Existing `search_models` and `model_stats` stay untouched.

---

## 5. Frontend Changes

### 5.1 Search.tsx
- Remove `RECOMMENDED_MODELS` import; delete `src/data/recommended.ts`
- Empty query → `api.recommendedModels()` (dynamic)
- Typed query → debounced `api.searchModelsWithFit(query)`
- GGUF variants get `⚠ Experimental` badge

### 5.2 Auto-Search
- 400ms debounce, min 2 chars, seq counter for stale rejection
- Enter bypasses debounce
- Remove explicit "Search" button
- Spinner in search input during loading

### 5.3 Types & API
- Add `ModelWithFit`, `QuantVariantWithFit`, `FitResult`, `QuantVariant` to `types.ts`
- Add `searchModelsWithFit`, `recommendedModels` to `api.ts`

---

## 6. `default_quant` stays global fallback
Used as `preferred_format` in `best_variant()`, fallback when discovery returns nothing, default for `servers_create`.

## 7. GGUF Experimental Flagging
`FormatSupport::Experimental` + amber badge with tooltip. Ranked below native at equal score.

## 8. Documentation
Update `docs/ARCHITECTURE.md`: add `fit.rs`, update `hf.rs`, add new commands.

## 9. Testing
- `fit.rs`: verdict thresholds, score computation, ranking, GGUF exact weight, measured stats
- GGUF filename parsing: quant label extraction patterns
- Data sourcing: RoPE scaling, nested config dims, safetensors expand params
