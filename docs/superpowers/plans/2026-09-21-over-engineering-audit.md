# Over-Engineering Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete ~10,000 lines of dead code, 700 MB of tracked binaries, and ~5 unused dependencies surfaced by the 2026-09-21 ponytail audit, without changing behavior.

**Architecture:** Deletion-first refactor. Each task deletes a verified-dead module, wrapper, duplicate, or dependency, then re-runs the crate/app test suite to prove nothing broke. Groups are ordered so the biggest, safest cuts go first and each group leaves the tree compiling and green.

**Tech Stack:** Rust (Tauri 2 app + llmfit-core workspace crate), React 18 + Vite + TSX frontend, Node build scripts.

**Spec:** This plan implements findings from the 2026-09-21 ponytail audit of this repo (four parallel subagent scans of the Rust app crate, the llmfit-core crate, the React frontend, and repo-root bloat). The audit's findings list is the spec.

**Root note — uncommitted work:** The working tree already has unrelated edits in `src-tauri/src/gateway.rs`, `src-tauri/src/hf.rs`, `src-tauri/src/wsl.rs`, `src/pages/Search.tsx`, `src/types.ts`, and an untracked `.github/workflows/`. Do NOT revert, stash, or commit them. All deletions below apply on top of the current working tree; conflict-resolution edits (the gateway/search dedups) must preserve the user's new lines.

## Global Constraints

- Every task ends with the verification command(s) in its last step; a task is done only when they pass.
- Rust gates: `cargo build` (workspace, debug), `cargo test` (workspace) must pass at the end of each llmfit-core / src-tauri task. The app also needs a frontend `npm run build` (`tsc --noEmit && vite build`) at the end of each frontend task.
- llmfit-core verification must run from `src-tauri` (`cargo test -p llmfit-core` and `cargo test -p llm-panel`) because the app depends on the crate.
- Do not delete a `pub use` in `llmfit-core/src/lib.rs` until every external reference (app crate, tests, integration tests) is gone. Grep `llmfit_core::` in `src-tauri/src/` after each crate-side deletion.
- Do not delete a data file while a build.rs or `include_str!` still reads it: `build.rs` embeds `data/community/` and `data/hardware/`; `benchmarks.rs:19` embeds the community aggregate; `hwprofile.rs:31` embeds the hardware aggregate.
- No new dependencies. Removing deps must be verified by `cargo build` (unused deps are not errors) and, for `yaml_serde`, by the fact that `quality.rs` is gone.
- Never touch correctness, security, or benchmark-scoring behavior. If a deletion changes a number a user sees, stop and restore.
- Keep `requirements`: the `dirs`, `reqwest blocking`, `reqwest stream`, `sysinfo`, `which`, `objc2-metal`, and all live `data/*.json` files verified used by the audit are OFF-LIMITS.
- `npm run build` runs `tsc --noEmit`; TSX edits must be type-clean.

## Review Focus

- A deleted llmfit-core module that turns out to be live: observable as a compile error in either crate or an integration test. The per-task build/test gate catches it; restore the file and the `pub use`, then re-run.
- A `pub use` left pointing at a deleted symbol: compile error in the app crate. Grep for the re-export in `lib.rs` after each module deletion.
- A removed data file still referenced by `include_str!` or build.rs: compile error with a path-not-found message. Verify before `git rm`.
- A frontend dedup that changes copy/labels (Settings simple vs advanced views have one-character differences): preserve the exact strings; the diff must be byte-identical behavior.
- Deleting a duplicated helper that the user's uncommitted edit newly uses (gateway.rs, hf.rs, wsl.rs, Search.tsx): compile/type errors surface it; keep the user's added code compiling.
- share.rs/bench.rs/quality.rs reference each other (`share.rs:22` imports `bench::BenchResult`, `quality.rs:14` imports `bench::{ChatCompletionResponse,...}`, `share.rs:463` calls `bench::is_plausible_tps`). Order matters: trim share+quality (Task 1) before deleting bench.rs (Task 2), or compile breaks.

---
### Task 1: llmfit-core — trim `share.rs` to its local-store block

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/share.rs` (keep only lines ~278-491 `LocalBenchIndex` block; delete `build_submission`/`ShareOptions`/`store_local` at 47-277 and the orchestration/OAuth/PR-upload half at 493-end of file), `src-tauri/crates/llmfit-core/src/lib.rs` (drop the `share` re-export if it names deleted symbols — check what lib.rs re-exports from share)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `share::{LocalBenchIndex, LoadResult …}` (whatever `analysis.rs:294` uses) still available; `BenchResult`/`BenchSummary`/`tag_matches_model` references in the deleted halves gone.

- [ ] **Step 1: Read the file boundaries and lib.rs re-exports**

Read `share.rs` and `lib.rs`. Note the exact line span of the local-store block (`LocalBenchIndex::load` is the live entry via `analysis.rs:294`) and what `lib.rs` re-exports from `share`.

- [ ] **Step 2: Delete the two dead halves**

Delete lines 47-277 and everything from the "Orchestration" section start (~line 493) to the end of the file, keeping the local-store block contiguous. Fix imports: drop now-unused `crate::bench::*` and provider imports. Remove any `pub use share::…` from `lib.rs` that names deleted symbols (keep anything still exported).

- [ ] **Step 3: Build/test**

Run (from `src-tauri`): `cargo build && cargo test -p llmfit-core`
Expected: PASS. If `share.rs` still imports `bench::` (line 22, 463) delete those now-dead references too.

- [ ] **Step 4: Commit** — `git commit -am "refactor(llmfit): trim share.rs to local benchmark store"`

---
### Task 2: llmfit-core — delete `quality.rs`, `plan.rs`, `claim.rs`, `concurrency.rs`, `storage.rs`, `doctor.rs` + their data files

**Files:**
- Delete: `src-tauri/crates/llmfit-core/src/quality.rs` (imports `bench::` types — must precede Task 3), `plan.rs`, `claim.rs`, `concurrency.rs`, `storage.rs`, `doctor.rs`
- Delete: `src-tauri/crates/llmfit-core/data/benchmarks.yaml`, `src-tauri/crates/llmfit-core/data/baselines.json`
- Modify: `src-tauri/crates/llmfit-core/src/lib.rs` (remove `pub mod quality; pub mod plan; pub mod claim; pub mod concurrency; pub mod storage; pub mod doctor;` and their `pub use` lines), `src-tauri/crates/llmfit-core/Cargo.toml:21` (remove `yaml_serde = "0.10.4"`)

**Interfaces:**
- Consumes: nothing at runtime (all six verified zero-caller).
- Produces: lib.rs without six `pub mod`/`pub use`; Cargo.toml without `yaml_serde`.

- [ ] **Step 1: Confirm dead + find lib.rs lines**

Run: `rg -n "estimate_model_plan|resolve_model_selector|normalize_quant|collect_diagnostics|estimate_storage|estimate_concurrency|claim_name|render_json" src-tauri/src/`
Expected: no matches outside llmfit-core. Read `lib.rs` and note the exact `pub mod` / `pub use` lines for each deleted module.

- [ ] **Step 2: git rm the six modules and two data files**

`git rm src-tauri/crates/llmfit-core/src/quality.rs src-tauri/crates/llmfit-core/src/plan.rs src-tauri/crates/llmfit-core/src/claim.rs src-tauri/crates/llmfit-core/src/concurrency.rs src-tauri/crates/llmfit-core/src/storage.rs src-tauri/crates/llmfit-core/src/doctor.rs src-tauri/crates/llmfit-core/data/benchmarks.yaml src-tauri/crates/llmfit-core/data/baselines.json`

- [ ] **Step 3: Edit lib.rs**

Remove the `pub mod` lines (and any `pub use` blocks referencing the deleted modules) for quality/plan/claim/concurrency/storage/doctor.

- [ ] **Step 4: Remove yaml_serde**

Edit `Cargo.toml` line 21: delete `yaml_serde = "0.10.4"`.

- [ ] **Step 5: Build/test**

Run (from `src-tauri`): `cargo build && cargo test -p llmfit-core && cargo test`
Expected: PASS. If a compile error names a deleted file/symbol, restore and re-add to plan.

- [ ] **Step 6: Commit** — `git commit -am "refactor(llmfit): delete dead quality/plan/claim/concurrency/storage/doctor modules"`

---
### Task 3: llmfit-core — delete `bench.rs`, move `is_plausible_tps` into `benchmarks.rs`

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/benchmarks.rs` (add `is_plausible_tps`; fix call site at `benchmarks.rs:585` which references `crate::bench::is_plausible_tps`)
- Delete: `src-tauri/crates/llmfit-core/src/bench.rs`
- Modify: `llmfit-core/src/lib.rs` (remove `pub mod bench;` and any bench re-exports)

**Interfaces:**
- Consumes: `bench::is_plausible_tps` (5 lines, from bench.rs) still called by `benchmarks.rs:585`.
- Produces: `benchmarks::is_plausible_tps`.

- [ ] **Step 1: Read `is_plausible_tps` in bench.rs**

Read bench.rs around lines 145-160 to get the exact function body and its imports.

- [ ] **Step 2: Move it into benchmarks.rs**

Copy the function into `benchmarks.rs`, update `sign benchmarks.rs:585` to call the local `is_plausible_tps`, and delete `bench.rs` (`git rm`). Remove `pub mod bench;` from `lib.rs`; grep `pub use bench` and remove any such re-exports.

- [ ] **Step 3: Verify no `crate::bench::` refs remain**

Run: `rg -n "crate::bench::|pub use bench" src-tauri/`
Expected: no matches (quality.rs and share.rs were cleaned in Tasks 1-2).

- [ ] **Step 4: Build/test**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "refactor(llmfit): delete bench.rs, keep is_plausible_tps in benchmarks"`

---
### Task 4: llmfit-core — trim `update.rs` to the live cache helpers

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/update.rs` (keep ~60 live lines: `cache_dir`/`cache_file`/`load_cache` per models.rs:1976; delete `update_model_cache`, `UpdateOptions`, `save_cache`, `clear_cache`, and the whole `HfApiModel`/scraper stack, lines ~107-832), `lib.rs` (drop the `update::` re-export block to only what remains)

**Interfaces:**
- Consumes: `load_cache` called by models.rs:1976.
- Produces: same `cache_dir`/`cache_file`/`load_cache` signatures; nothing else exported.

- [ ] **Step 1: Read update.rs + find live callers**

Read `update.rs`. Confirm `cache_dir`/`cache_file`/`load_cache` signatures and callers (models.rs:1976).

- [ ] **Step 2: Delete the scraper half**

Delete everything except the cache-path/load helpers. Fix `lib.rs:34-36` re-exports. If `hf_models.json` (11 MB) is only embedded by the scraper stack, delete that file too only if the `include_str!` is in the deleted code; else keep it.

- [ ] **Step 3: Build/test**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor(llmfit): trim update.rs to cache helpers"`

---
### Task 5: llmfit-core — delete `hwprofile.rs` + hardware data + its integration test

**Files:**
- Delete: `src-tauri/crates/llmfit-core/src/hwprofile.rs`, `src-tauri/crates/llmfit-core/tests/hardware_profiles.rs`, `src-tauri/crates/llmfit-core/data/hardware/` (3 json + README + schema.json)
- Modify: `lib.rs` (remove `pub use hwprofile::HardwareProfile;` + `pub mod hwprofile;`), `build.rs` (remove `embed_hardware_profiles()` if no other reader of `hardware_profiles.json` remains)

**Interfaces:**
- Consumes: nothing (hwprofile referenced only by its own test).
- Produces: no `hwprofile` module; build.rs no longer embeds hardware.

- [ ] **Step 1: Confirm only reference is the test**

Run: `rg -n "hwprofile|HardwareProfile" src-tauri/`
Expected: matches only in hwprofile.rs, tests/hardware_profiles.rs, lib.rs, build.rs.

- [ ] **Step 2: Delete files**

`git rm src-tauri/crates/llmfit-core/src/hwprofile.rs src-tauri/crates/llmfit-core/tests/hardware_profiles.rs`
`git rm -r src-tauri/crates/llmfit-core/data/hardware`
Edit `lib.rs` and `build.rs` to drop the module + `embed_hardware_profiles` + its `rerun-if-changed=data/hardware` line.

- [ ] **Step 3: Build/test**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor(llmfit): delete unused hwprofile module and hardware data"`

---
### Task 6: llmfit-core — delete benchmarks.rs network half + dedupe ureq agent builders

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/benchmarks.rs` (delete `get_json`, `fetch_benchmarks`, `fetch_benchmarks_for_model`, `fetch_leaderboard`, `fetch_leaderboard_for_preset`, `hw_query_params`, `hw_leaderboard_params`, `lookup_mem_tier`, `urlencoded`, dead accessors `cached_preset_benchmark_count`/`cache_timestamp`, ~line 652-915), `src-tauri/crates/llmfit-core/src/providers.rs` (replace the repeated `ureq::Agent::config_builder()…timeout_global(10s)` blocks at providers.rs:166 with one `fn ureq_agent()` helper)

**Interfaces:**
- Consumes: nothing at runtime.
- Produces: embedded-only `benchmarks.rs`.

- [ ] **Step 1: Read benchmarks.rs network half + confirm dead**

Read `benchmarks.rs:1-120` and the network section. Confirm none of the network fns are called outside benchmarks.rs (grep `fetch_benchmarks|fetch_leaderboard|lookup_mem_tier|urlencoded|hw_query_params`).

- [ ] **Step 2: Delete the network half**

Delete the dead fns; keep embedded caches (`cached_*_for_preset`, `MeasuredTpsIndex`, `CommunityBenchIndex`) and `community_benchmarks.json` include. Remove now-unused imports.

- [ ] **Step 3: Dedupe the ureq agent builder**

In providers.rs, read the agent-builder block (~line 166) and extract `fn ureq_agent() -> ureq::Agent`; replace the other copies (benchmarks.rs which no longer exists, plus any remaining in share/providers) with calls. Skip if providers.rs already has one.

- [ ] **Step 4: Build/test**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "refactor(llmfit): delete offline benchmark fetch path, dedupe ureq agent"`

---
### Task 7: llmfit-core — delete providers.rs pull/download API (runtime-dead)

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/providers.rs` (delete `start_pull`, `PullHandle`/`PullEvent`, `download_gguf`, `poll_lmstudio_download_status`, `lmstudio_pull_tag`, `vllm_pull_tag`, and any other pull/download fn with zero runtime callers; keep `is_available`, `installed_models_counted`, `is_model_installed_*`, `tag_matches_model`, path/identity helpers)

**Interfaces:**
- Consumes: the app only uses `is_available` (commands.rs:131-136); analysis.rs uses `installed_models_counted`/`is_model_installed_*`; benchmarks.rs uses `tag_matches_model`.
- Produces: providers module with pull/install/download removed.

- [ ] **Step 1: Map live entry points**

Grep app + analysis/benchmarks for `providers::` usage. Determine exactly which provider items are live. (adapter/commands use `LlamaCppProvider`, `OllamaProvider`, `is_available`, `ModelProvider`.)

- [ ] **Step 2: Delete the dead pull fns per provider**

For each of the 5 providers (LlamaCpp, LmStudio, Mlx, Ollama, Vllm), delete the pull/download/poll methods that are unreachable, and their `PullEvent`/`PullHandle` types. Use `rg "fn start_pull|fn download_gguf|fn poll_|pull_tag"` to find them all. This is a large mechanical deletion; re-run `cargo build` iteratively after each provider.

- [ ] **Step 3: Verify app compiles**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS. The command handler `commands.rs` `pull_model`/`pull_status`/`pull_cancel`/`download_gguf` must NOT depend on the deleted fns — verify by compiling; if they do, this task is scoped wrongly and must instead keep the live subset (report back, don't delete a live path).

- [ ] **Step 4: Commit** — `git commit -am "refactor(llmfit): delete runtime-dead provider pull/download API"`

---
### Task 8: llmfit-core — models.rs quant-table consolidation

**Files:**
- Modify: `src-tauri/crates/llmfit-core/src/models.rs` (replace five parallel match fns `quant_bpp` lines 20-60, `quant_bytes_per_param` 62-86, `quant_speed_multiplier` 88-118, `quant_quality_penalty` 119-167, `quant_is_recognized` 168-~200 with one `const QUANT_SPECS` table + small accessors)

**Interfaces:**
- Consumes: callers of all five `quant_*` fns (same names return same values).
- Produces: same five fn exports, same values.

- [ ] **Step 1: Read the five quant_* fns**

Read models.rs lines 1-210. Verify the five match lists are over identical UD/Q/MLX/AWQ labels.

- [ ] **Step 2: Write the QUANT_SPECS table**

One `struct QuantSpec { bpp: f64, bytes_per_param: f64, speed: f64, penalty: f64 }` + one match to resolve from quant name, and five one-line accessor fns (or have callers read the struct). Keep fn signatures identical so callers don't change.

- [ ] **Step 3: Build/test**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS — all existing quant tests still green (they pin the values).

- [ ] **Step 4: Commit** — `git commit -am "refactor(llmfit): consolidate five quant match tables into one"`

---
### Task 9: app crate — pick one model scorer (app fit.rs vs llmfit-core fit.rs)

**Files:**
- Modify: `src-tauri/src/fit.rs` (delete; pull any app-unique logic into llmfit-core if needed), `src-tauri/src/llmfit_adapter.rs` (delete if no longer needed), `src-tauri/src/commands.rs` (point all scoring at llmfit-core), `src-tauri/src/lib.rs` (remove `pub mod fit;` / `pub mod llmfit_adapter;`)

**Interfaces:**
- Consumes: `commands.rs:584` `fit::score_variant` and `commands.rs:847` `llmfit_core::fit::ModelFit::analyze`.
- Produces: both call sites use `llmfit_core::fit` only.

**Warning — highest-risk task.** Two live scorers exist; `llmfit_adapter.rs` exists only to convert llmfit-core `FitLevel`/`RunMode` into app `FitVerdict`/`RunMode`. Verify the two produce identical frontend output for the same model+spec before deleting either; if `score_variant` (app fit.rs) has behavior llmfit-core lacks, port it.

- [ ] **Step 1: Diff the two scoring paths**

Read `src-tauri/src/fit.rs` and `llmfit_adapter.rs`; compare `score_variant` vs `llmfit-core::fit::ModelFit::analyze` behavior. Identify what the app's `score_variant` provides that llmfit-core doesn't (variant ranking, quant comparison).

- [ ] **Step 2: Port missing behavior into llmfit-core (if any)**

If `score_variant` has unique logic, move it into llmfit-core `fit.rs` as a small fn; update `commands.rs:584` to call llmfit-core directly.

- [ ] **Step 3: Delete app fit.rs + llmfit_adapter.rs**

`git rm src-tauri/src/fit.rs src-tauri/src/llmfit_adapter.rs`; update `lib.rs` and all `crate::fit::`/`llmfit_adapter::` callers.

- [ ] **Step 4: Build/test both crates + frontend**

Run (from `src-tauri`): `cargo build && cargo test` and (repo root) `npm run build`
Expected: PASS. Grep for any remaining `crate::fit::` or `llmfit_adapter::` references.

- [ ] **Step 5: Commit** — `git commit -am "refactor(app): collapse duplicate fit scorer into llmfit-core"`

---
### Task 10: app crate — dead wrappers and duplicates

**Files:**
- Modify: `src-tauri/src/hf.rs` (delete `GgufFile`, `GgufShardGroup`, `group_gguf_files`, `gguf_fit_indicator` test-only machinery), `src-tauri/src/commands.rs` (merge `CreateServerInput`/`UpdateServerInput`), `src-tauri/src/gateway.rs` (merge `find_server_port_for_model`/`find_server_port_for_embed` and `running_instruct_servers`/`running_model_names`), `src-tauri/src/state.rs` (delete `validate_env_name`/`shell_escape_single_quoted` dup), `src-tauri/src/server.rs` (delete test-only `build_start_command`, `build_llamacpp_args`, `llamacpp_moe_tokens_per_sec` in estimate.rs), `src-tauri/src/estimate.rs` (delete `llamacpp_moe_tokens_per_sec`, merge `as_usize`→`usize_of`), `src-tauri/src/fit.rs` (delete `rank_variants`) — note fit.rs is deleted by Task 9, so fold that into Task 9 if tasks run in order

**Interfaces:**
- Consumes: nothing.
- Produces: fewer public fns; behavior unchanged.

- [ ] **Step 1: hf.rs dead gguf machinery**

Delete the 3 test-only types/fns and their tests in hf.rs (lines ~16-110 per audit). `cargo test` to confirm.

- [ ] **Step 2: Merge input structs**

In commands.rs, collapse `CreateServerInput` + `UpdateServerInput` into one `ServerInput` all-Option struct; update the two handlers' field access (identical names, so mostly mechanical). `cargo build`.

- [ ] **Step 3: Merge gateway helpers**

In gateway.rs, merge the two byte-identical `find_server_port_for_*` into `fn find_server_port_for(servers, model, key)` with the task keyword as a param, and the two running-servers filters into one helper. Preserve the user's uncommitted gateway.rs changes — read the file first and keep their new lines. `cargo build && cargo test`.

- [ ] **Step 4: Delete dup env helpers + test-only wrappers**

Delete `validate_env_name`/`shell_escape_single_quoted` in state.rs (tests import from `crate::server`); delete `build_start_command` in server.rs (tests call `launch_script`); delete `build_llamacpp_args`; delete `llamacpp_moe_tokens_per_sec` in estimate.rs; merge `as_usize`/`usize_of`; delete `rank_variants` (covered by Task 9 if fit.rs is gone). `cargo test`.

- [ ] **Step 5: Remove `AppState::save_config` + `_prerelease`**

Delete `save_config` (no callers) and the unused `Release._prerelease` field + its instances. `cargo build && cargo test`.

- [ ] **Step 6: Commit** — `git commit -am "refactor(app): delete dead wrappers and duplicate helpers"`

---
### Task 11: frontend — delete dead API surface

**Files:**
- Modify: `src/api.ts` (delete the whole TS llmfit reimplementation `UseCase`,`FitAssessment`,`computeLlmfitScore`,`evaluateSystemFit`,`recommendBestQuant`,`SUPPORTED_QUANTS`,`QuantFitSummary`, ScoreComponents dup, lines 306-631; dead wrappers `searchModels`, `modelStats`, `serversUpdateEnv`, non-stream `serversChat`, `events.llamacppInstallProgress`, standalone `getMemorySettings`/`updateMemorySettings`/`getSystemMemory`), `src/types.ts` (delete `ConfigExportPackage`)

**Interfaces:**
- Consumes: pages import only live API names (`searchModelsWithFit`, `serversChatStream`, `api.getMemorySettings`).
- Produces: slimmer `api.ts` with same live exports.

- [ ] **Step 1: Grep API usage per page**

Run: `rg -n "computeLlmfitScore|evaluateSystemFit|recommendBestQuant|searchModels\b|modelStats|serversUpdateEnv|serversChat\b|llamacppInstallProgress|getMemorySettings|getSystemMemory|ConfigExportPackage" src/`
Expected: only `api.ts`/`types.ts` definitions match, no page usage. (TypeScript build will confirm.)

- [ ] **Step 2: Delete the dead exports**

Delete the listed blocks from api.ts and types.ts. Ensure `src/types.ts` re-export line for `api` (line ~38-46 per audit) still compiles.

- [ ] **Step 3: Type-check**

Run (repo root): `npm run build`
Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor(frontend): delete dead API wrappers and TS llmfit reimplementation"`

---
### Task 12: frontend — dedupe Search.tsx repeated blocks

**Files:**
- Modify: `src/pages/Search.tsx` (one `PullStateButton` from the 3 near-verbatim pull-state ternaries at 852-906/1285-1327/1689-1741; one `HfLink` component from the 4 copies at 742-759/1087-1104/1388-1406/1594-1611; one `filterFit(cat, onlyRecommended)` from the two filter memos; one `ErrorBanner` from searchErr/recsErr banners)

**Interfaces:**
- Consumes: existing props as current blocks use them.
- Produces: same rendered output.

- [ ] **Step 1: Extract `HfLink`**

Read the 4 "Open on Hugging Face" anchors, extract `function HfLink({url, text})`, replace all 4. Preserve the user's uncommitted Search.tsx edits — verify against current file state.

- [ ] **Step 2: Extract `PullStateButton`**

Read the 3 pull-state blocks, extract one component (with props for state/labels/onCancel/onRetry/onPull/inLibrary), replace.

- [ ] **Step 3: Extract `filterFit` + `ErrorBanner`**

Merge the two filter memos into one helper; merge the two error banners into one component.

- [ ] **Step 4: Type-check**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "refactor(frontend): dedupe Search pull-state, HF link, filters, error banner"`

---
### Task 13: frontend — Settings simple/advanced shared fields + Dashboard FlashInfer map

**Files:**
- Modify: `src/pages/Settings.tsx` (hoist the WSL-distro Field, HF-token block, VRAM range slider, RAM-overflow toggle shared between "simple" (lines 428-583) and "advanced" (619-773) into consts/components), `src/pages/Dashboard.tsx:285-312` (4 near-identical nvcc/gcc/ninja/python_dev rows → one `.map()`)

**Interfaces:**
- Consumes: current settings state setters.
- Produces: same rendered output.

- [ ] **Step 1: Extract shared settings fields**

Read the two view blocks; extract each shared field into a `const SimpleOrAdvancedField` (or prop-driven component) preserving exact labels/ids (one-character differences like datalist-id vs button-label). `npm run build`.

- [ ] **Step 2: FlashInfer rows → map**

Replace the 4 status rows in Dashboard.tsx with `[["nvcc",…],["gcc",…],["ninja",…],["python3.12-dev",…]].map(...)`. `npm run build`.

- [ ] **Step 3: Commit** — `git commit -am "refactor(frontend): dedupe Settings fields and Dashboard status rows"`

---
### Task 14: frontend — Servers.tsx globalDefaults plumbing + dead subscriber + Library banner

**Files:**
- Modify: `src/pages/Servers.tsx` (delete `globalDefaults`/`initialEnv={{}}` frozen plumbing and the unreachable "Global Defaults" read-only block at 712-715/1230-1244; delete no-op `events.serverMetrics` subscriber at 193-195; `|| true` constant → literal), `src/pages/Library.tsx:186-205` (reuse Search's `DownloadProgress` component for the Active Downloads banner)

**Interfaces:**
- Consumes: nothing (globalDefaults never set).
- Produces: same UI minus the dead globals panel.

- [ ] **Step 1: Servers.tsx trims**

Read the file; delete the `globalDefaults`/`initialEnv` state, the `flashInferDisabled` `|| true`, the read-only globals block, and the identity subscriber. `npm run build`.

- [ ] **Step 2: Library banner reuse**

Export `DownloadProgress` from Search.tsx (or move to ui.tsx), import in Library.tsx, replace the inline banner. `npm run build`.

- [ ] **Step 3: Commit** — `git commit -am "refactor(frontend): remove Servers dead plumbing, reuse DownloadProgress"`

---
### Task 15: repo bloat — dist-release binaries, build.mjs alias, data caches

**Files:**
- Delete: `dist-release/` (32 tracked binaries, 699.6 MB), `src-tauri/crates/llmfit-core/data/benchmark_cache.json` (3.4 MB)
- Modify: `scripts/build.mjs:150-153` (drop the alias copy that stages two identical binaries), `.gitignore` (already ignores `dist-release/` — keep line)

**Interfaces:**
- Consumes: nothing; `gh release create` is the real publish path (skills/github-deployment).
- Produces: no committed binaries.

- [ ] **Step 1: git rm dist-release and the cache**

`git rm -r src-tauri/crates/llmfit-core/dist-release` (path is repo-root `dist-release/`): `git rm -r dist-release src-tauri/crates/llmfit-core/data/benchmark_cache.json`
Read `benchmarks.rs:14` first — only remove `benchmark_cache.json` if the `include_str!` for it is among the deleted network half (Task 6), else keep it (audit flagged it as a committee decision, not a hard delete — if still embedded, re-add the file and note it).

- [ ] **Step 2: Drop the build.mjs alias**

Read `scripts/build.mjs:140-160`; remove the second staged name / alias copy; keep one `.exe` + one `-standalone.exe`. `npm run check` (if quick) or `node scripts/build.mjs --check`.

- [ ] **Step 3: Commit** — `git commit -am "chore: stop tracking release binaries and committed benchmark cache"`

---
### Task 16: repo bloat — stale docs and patch

**Files:**
- Delete: `docs/PLAN.md`, `docs/ROADMAP.md`, `docs/superpowers/plans/2026-09-19-roadmap-completion.md`, `docs/superpowers/plans/2026-09-17-ram-context-overflow.md`, `docs/superpowers/specs/2026-09-20-p0-gateway-design.md`, `docs/superpowers/plans/2026-09-20-p0-gateway.md`, `docs/superpowers/specs/2026-09-16-quant-aware-model-search-design.md`, `docs/superpowers/specs/2026-09-17-ram-context-overflow-design.md`, `server-signature.patch`
- Keep: `docs/superpowers/plans/2026-09-20-audit-improvements.md` (in-flight)

**Interfaces:** none.

- [ ] **Step 1: Verify each is stale**

Spot-check ROADMAP (header claims v1.0.0; phases all ✅) and the roadmap-completion plan (single done task). Confirm gateway + quant-variant + ram-context are all implemented (gateway.rs, hf.rs, fit.rs exist and are live).

- [ ] **Step 2: git rm the listed files**

`git rm docs/PLAN.md docs/ROADMAP.md docs/superpowers/plans/2026-09-19-roadmap-completion.md docs/superpowers/plans/2026-09-17-ram-context-overflow.md docs/superpowers/specs/2026-09-20-p0-gateway-design.md docs/superpowers/plans/2026-09-20-p0-gateway.md docs/superpowers/specs/2026-09-16-quant-aware-model-search-design.md docs/superpowers/specs/2026-09-17-ram-context-overflow-design.md server-signature.patch`

- [ ] **Step 3: Commit** — `git commit -am "docs: remove stale plans, specs, and patch"`

---
### Task 17: final full verification

**Files:** none.

**Interfaces:** all tasks complete.

- [ ] **Step 1: Full Rust gates**

Run (from `src-tauri`): `cargo build && cargo test`
Expected: PASS.

- [ ] **Step 2: Full frontend gate**

Run (repo root): `npm run build`
Expected: PASS.

- [ ] **Step 3: Audit leftover dead patterns**

Run: `rg -n "estimate_model_plan|bench_ollama|update_model_cache|computeLlmfitScore|globalDefaults|GgufShardGroup|yaml_serde|_prerelease" src/ src-tauri/`
Expected: no matches (or only intentional comments).

- [ ] **Step 4: Report**

Summarize net lines/deps removed per area; leave the branch uncommitted state clean except the pre-existing edits and `.github/workflows/` (do not touch those if user hasn't asked).