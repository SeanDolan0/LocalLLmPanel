// Types mirroring the Rust side (commands.rs / state.rs / server.rs / hf.rs).

export interface GpuSnapshot {
  name: string;
  vram_total_mb: number;
  vram_free_mb: number;
  util_percent: number;
}

export interface ProvisionReport {
  phases_completed: string[];
  distro: string;
  vllm_version: string | null;
  torch_version: string | null;
  cuda_available: boolean;
  gpu_name: string | null;
  vram_mb: number | null;
  bf16_supported: boolean;
}

export interface EnvStatus {
  wsl_ok: boolean;
  distro: string;
  apt_based: boolean;
  provisioned: boolean;
  report: ProvisionReport | null;
  gpu: GpuSnapshot | null;
  servers_running: number;
  running_weight_gb: number;
  gpu_bandwidth_gbs: number;
  gpu_bw_known: boolean;
  cpu_name?: string | null;
  cpu_cores?: number | null;
  total_ram_gb?: number | null;
  available_ram_gb?: number | null;
  ram_bandwidth_gbps?: number | null;
  providers_detected?: string[];
  llamacpp_installed?: boolean;
  llamacpp_tag?: string | null;
  llamacpp_version?: string | null;
  llamacpp_executable?: string | null;
}

export interface ModelWithStats {
  id: string;
  downloads: number;
  likes: number;
  trending_score: number;
  pipeline_tag: string | null;
  params_b: number | null;
  context: number | null;
  context_source: string | null;
  context_estimated: boolean;
  head_dim: number | null;
  max_tok_s: number | null;
  quant_assumed: string;
}

export interface ModelStats {
  model_id: string;
  params_b: number | null;
  context: number | null;
  context_source: string | null;
  context_estimated: boolean;
  head_dim: number | null;
  n_layers: number | null;
  n_kv_heads: number | null;
  kv_bytes_per_token: number | null;
  weight_gb: number | null;
  context_fit: number | null;
  max_tok_s_fp16: number | null;
  max_tok_s_fp8: number | null;
  max_tok_s_int4: number | null;
  measured: MeasuredStats | null;
  vram_total_mb: number | null;
  gpu_name: string | null;
  score?: number | null;
  score_components?: ScoreComponents | null;
  usable_context?: number | null;
  runtime?: string | null;
  fit_level?: string | null;
  run_mode?: string | null;
  notes?: string[];
}

export interface MeasuredStats {
  tokens_per_sec: number | null;
  prompt_tokens_per_sec: number | null;
  total_prompt_tokens: number;
  total_generation_tokens: number;
  requests: number;
  measured_at_ms: number | null;
}

export interface ServerDef {
  backend: "vllm" | "llamacpp";
  id: string;
  name: string;
  model_id: string;
  task: string; // "instruct" | "embed"
  port: number;
  gpu_mem_util: number;
  max_model_len: number | null;
  quant: string;
  served_model_name: string | null;
  enforce_eager?: boolean;
  params_b: number | null;
  swap_space_gb?: number | null;
  cpu_offload_gb?: number | null;
  was_running?: boolean;
  model_path?: string | null;
  mmproj_path?: string | null;
  ctx_size?: number | null;
  n_gpu_layers: number;
  n_cpu_moe?: number | null;
  flash_attn: boolean;
  cache_type_k: string;
  cache_type_v: string;
  threads?: number | null;
  batch_size?: number | null;
  ubatch_size?: number | null;
  parallel: number;
  jinja: boolean;
  no_kv_offload: boolean;
  metrics: boolean;
  extra_args: string[];
}

export function effectiveModelName(def: ServerDef): string {
  return def.served_model_name ?? def.model_id;
}

export interface MetricsSnapshot {
  running: number;
  waiting: number;
  total_prompt_tokens: number;
  total_generation_tokens: number;
  requests: number;
  measured: MeasuredStats | null;
}

export interface ServerListRow {
  def: ServerDef;
  status: string; // stopped | starting | running | error
  error: string | null;
  metrics: MetricsSnapshot | null;
}

export interface AdvancedSettings {
  hf_home?: string | null;
  hf_offline: boolean;
  host: string;
  api_key?: string | null;
  gateway_enabled: boolean;
  gateway_port: number;
  kv_cache_dtype: string;
  enable_prefix_caching: boolean;
  enable_chunked_prefill: boolean;
  max_num_seqs?: number | null;
  disable_custom_all_reduce: boolean;
  log_level: string;
  extra_vllm_args?: string | null;
  custom_env_vars?: string | null;
}

export interface GatewayStatus {
  enabled: boolean;
  port: number;
  running: boolean;
}

export interface Settings {
  distro: string;
  llm_dir: string;
  venv_dir: string;
  llamacpp_dir: string;
  gguf_dir: string;
  llamacpp_executable?: string | null;
  llamacpp_installed_tag?: string | null;
  llamacpp_version?: string | null;
  hf_token: string;
  default_quant: string;
  servers: ServerDef[];
  measured: Record<string, MeasuredStats>;
  advanced_settings: AdvancedSettings;
  minimize_to_tray?: boolean;
  resume_servers_on_launch?: boolean;
  auto_restart_crashed?: boolean;
  launch_at_login?: boolean;
}

export interface PullStatus {
  model: string;
  state: string; // downloading | complete | failed
  file?: string | null;
  percent?: number | null;
  speed_bps?: number | null;
}

export interface WslLogEvent {
  phase: string;
  line: string;
}

export interface ServerStatusEvent {
  id: string;
  status: string;
  error: string | null;
}

export interface ServerLogEvent {
  id: string;
  line: string;
}

export interface CreateServerInput {
  backend?: "vllm" | "llamacpp";
  name: string;
  model_id: string;
  task?: string;
  port?: number;
  gpu_mem_util?: number;
  max_model_len?: number;
  quant?: string;
  served_model_name?: string;
  enforce_eager?: boolean;
  swap_space_gb?: number | null;
  cpu_offload_gb?: number | null;
  model_path?: string;
  mmproj_path?: string;
  ctx_size?: number;
  n_gpu_layers?: number;
  n_cpu_moe?: number;
  flash_attn?: boolean;
  cache_type_k?: string;
  cache_type_v?: string;
  threads?: number;
  batch_size?: number;
  ubatch_size?: number;
  parallel?: number;
  jinja?: boolean;
  no_kv_offload?: boolean;
  metrics?: boolean;
  extra_args?: string[];
}

export interface WslConfigInfo {
  path: string | null;
  content: string | null;
}

export interface LibraryEntry {
  model_id: string;
  size_mb: number;
  files: number;
  quant?: string | null;
  params_b?: number | null;
  installed?: boolean;
  in_use?: boolean;
  in_use_server?: string | null;
  task?: string | null;
  is_local?: boolean;
}

export type FitVerdict = "Comfortable" | "Constrained" | "DoesNotFit";
export type FormatSupport = "Native" | "Experimental";
export type QuantFormatTag = "FP16" | "FP8" | "AWQ" | "GPTQ" | "BNB" | "GGUF";

export interface QuantVariant {
  repo_id: string;
  format: QuantFormatTag;
  label: string;
  weight_bytes: number | null;
  params_b: number | null;
  gguf_file: string | null;
  vllm_native: boolean;
}

export interface MemorySettings {
  default_gpu_mem_util: number;
  vram_overhead_mb: number;
  enable_ram_overflow: boolean;
  manual_ram_limit_mb: number | null;
  safety_reserve_mb: number;
  offload_weights_allowed: boolean;
  max_context_cap: number | null;
}

export interface SystemMemoryInfo {
  wsl_total_mb: number;
  wsl_available_mb: number;
  usable_budget_mb: number;
  safety_reserve_mb: number;
  manual_override_mb: number | null;
}

export type RunMode = "Gpu" | "GpuRamSwap" | "CpuOffload" | "DoesNotFit";

export interface ScoreComponents {
  quality: number;
  speed: number;
  fit: number;
  context: number;
}

export interface GgufSource {
  provider: string;
  repo: string;
}

export interface FitResultBackend {
  verdict: FitVerdict;
  run_mode: RunMode;
  score: number;
  weight_gb: number;
  vram_context: number;
  extended_context: number;
  native_context: number;
  usable_context?: number;
  swap_space_gb: number;
  cpu_offload_gb: number;
  est_tok_s: number | null;
  measured_tok_s: number | null;
  vram_pct: number;
  ram_pct: number;
  format_support: FormatSupport;
  reason: string;
  score_components?: ScoreComponents | null;
  runtime?: string | null;
  notes?: string[];
}

export interface QuantVariantWithFit {
  variant: QuantVariant;
  fit: FitResultBackend;
}

export interface ModelWithFit {
  id: string;
  provider?: string | null;
  downloads: number;
  likes: number;
  trending_score: number;
  pipeline_tag: string | null;
  params_b: number | null;
  parameter_count?: string | null;
  context: number | null;
  context_source: string | null;
  context_estimated: boolean;
  head_dim: number | null;
  n_layers: number | null;
  n_kv_heads: number | null;
  use_case?: string | null;
  category?: string | null;
  release_date?: string | null;
  variants: QuantVariantWithFit[];
  best_variant_idx: number;
  score?: number;
  score_components?: ScoreComponents | null;
  best_quant?: string | null;
  runtime?: string | null;
  fit_level?: string | null;
  run_mode?: string | null;
  usable_context?: number | null;
  effective_context_length?: number | null;
  estimated_tps?: number | null;
  memory_required_gb?: number | null;
  memory_available_gb?: number | null;
  utilization_pct?: number | null;
  notes?: string[];
  capabilities?: string[];
  gguf_sources?: GgufSource[];
  installed?: boolean;
}

export interface ChatMessage {
  role: string; // "system" | "user" | "assistant"
  content: string;
  images?: string[]; // base64 data URLs for vision-capable (VLM) models
}

export interface Conversation {
  id: string;
  server_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: ChatMessage[];
}

export interface ChatTokenPayload {
  request_id: string;
  server_id: string;
  token: string;
}

export interface ChatDonePayload {
  request_id: string;
  server_id: string;
}

export interface ChatCancelPayload {
  request_id: string;
  server_id: string;
}

export interface ChatErrorPayload {
  request_id: string;
  server_id: string;
  error: string;
}

export interface BenchmarkRun {
  id: string;
  server_id: string;
  model_id: string;
  quant: string | null;
  timestamp: number;
  prompt_tok_s: number;
  gen_tok_s: number;
  latency_ms: number;
  prompt_count: number;
}

export interface BenchmarkStepPayload {
  server_id: string;
  step: number;
  total_steps: number;
  prompt_tok_s: number;
  gen_tok_s: number;
  latency_ms: number;
}

export interface BenchmarkCancelPayload {
  server_id: string;
}

export interface BenchmarkErrorPayload {
  server_id: string;
  error: string;
}

export interface SystemMetricPoint {
  timestamp: number;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_free_mb: number;
  gpu_util_pct: number;
}

export interface ServerMetricPoint {
  timestamp: number;
  tok_s: number;
  prompt_tok_s: number;
  requests_running: number;
  requests_waiting: number;
}

export interface ServerRecipe {
  schema?: string;
  model_id: string;
  task: string;
  port: number;
  gpu_mem_util: number;
  quant: string;
  max_model_len?: number | null;
  served_model_name?: string | null;
  enforce_eager: boolean;
  swap_space_gb?: number | null;
  cpu_offload_gb?: number | null;
}

export interface ConfigExportPackage {
  schema: string;
  exported_at: string;
  distro: string;
  llm_dir: string;
  venv_dir: string;
  default_quant: string;
  servers: ServerDef[];
  memory_settings: MemorySettings;
  advanced_settings: AdvancedSettings;
  minimize_to_tray: boolean;
  resume_servers_on_launch: boolean;
  auto_restart_crashed: boolean;
  launch_at_login: boolean;
}