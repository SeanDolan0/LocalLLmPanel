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

export interface Settings {
  distro: string;
  llm_dir: string;
  venv_dir: string;
  hf_token: string;
  default_quant: string;
  servers: ServerDef[];
  measured: Record<string, MeasuredStats>;
}

export interface PullStatus {
  model: string;
  state: string; // downloading | complete | failed
  file: string | null;
  percent: number | null;
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
  name: string;
  model_id: string;
  task?: string;
  port?: number;
  gpu_mem_util?: number;
  max_model_len?: number;
  quant?: string;
  served_model_name?: string;
  enforce_eager?: boolean;
}

export interface WslConfigInfo {
  path: string | null;
  content: string | null;
}

export interface LibraryEntry {
  model_id: string;
  size_mb: number;
  files: number;
}