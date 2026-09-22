import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  CreateServerInput,
  EnvStatus,
  FitResultBackend,
  MemorySettings,
  MetricsSnapshot,
  ModelWithFit,
  ProvisionReport,
  PullStatus,
  QuantVariant,
  QuantVariantWithFit,
  RunMode,
  ServerDef,
  ServerListRow,
  ServerLogEvent,
  ServerStatusEvent,
  Settings,
  SystemMemoryInfo,
  WslConfigInfo,
  WslLogEvent,
  ChatMessage,
  Conversation,
  ChatTokenPayload,
  ChatDonePayload,
  ChatCancelPayload,
  ChatErrorPayload,
  BenchmarkRun,
  BenchmarkStepPayload,
  BenchmarkCancelPayload,
  BenchmarkErrorPayload,
  LlamacppInstallStatus,
  GgufRepoFile,
} from "./types";

export type {
  FitResultBackend,
  MemorySettings,
  ModelWithFit,
  QuantVariant,
  QuantVariantWithFit,
  RunMode,
  SystemMemoryInfo,
};

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

export const api = {
  envStatus: () => invoke<EnvStatus>("env_status"),
  provision: () => invoke<ProvisionReport>("provision"),
  installLlamacpp: () => invoke<LlamacppInstallStatus>("install_llamacpp"),
  llamacppStatus: () => invoke<LlamacppInstallStatus>("llamacpp_status"),
  searchModelsWithFit: (query: string) =>
    invoke<ModelWithFit[]>("search_models_with_fit", { query }),
  recommendedModels: () =>
    invoke<ModelWithFit[]>("recommended_models"),
  pullModel: (modelId: string) => invoke<void>("pull_model", { modelId }),
  pullStatus: () => invoke<{ pulling: string[] }>("pull_status"),
  pullCancel: (modelId: string) => invoke<void>("pull_cancel", { modelId }),
  ggufFiles: (repoId: string) => invoke<GgufRepoFile[]>("gguf_files", { repoId }),
  downloadGguf: (repoId: string, files: string[]) =>
    invoke<void>("download_gguf", { repoId, files }),
  serversList: () => invoke<ServerListRow[]>("servers_list"),
  serversCreate: (input: CreateServerInput) => invoke<ServerDef>("servers_create", { input }),
  serversDelete: (id: string) => invoke<void>("servers_delete", { id }),
  serversStart: (id: string) => invoke<void>("servers_start", { id }),
  serversStop: (id: string) => invoke<void>("servers_stop", { id }),
  serversRestart: (id: string) => invoke<void>("servers_restart", { id }),
  serversLogs: (id: string, since: number) => invoke<string>("servers_logs", { id, since }),
  serversUpdate: (id: string, input: Partial<CreateServerInput> & { restart?: boolean }) =>
    invoke<ServerDef>("servers_update", { input: { id, ...input } }),
  serversTestToolCall: (id: string, maxTokens = 2048, disableThinking = true) =>
    invoke<{
      passed: boolean;
      response: Record<string, unknown>;
      hint: string;
      tool_call?: Record<string, unknown>;
      reasoning_content?: string;
      max_tokens: number;
      disable_thinking: boolean;
    }>("servers_test_tool_call", {
      id,
      maxTokens,
      disableThinking,
    }),
  serversChatStream: (
    requestId: string,
    serverId: string,
    messages: ChatMessage[],
    temperature?: number
  ) =>
    invoke<void>("servers_chat_stream", {
      requestId,
      serverId,
      messages,
      temperature: temperature ?? null,
    }),
  serversChatCancel: (requestId: string) =>
    invoke<void>("servers_chat_cancel", { requestId }),
  conversationsList: () =>
    invoke<Conversation[]>("conversations_list"),
  conversationsSave: (conversation: Conversation) =>
    invoke<void>("conversations_save", { conversation }),
  conversationsDelete: (id: string) =>
    invoke<void>("conversations_delete", { id }),
  benchmarksRun: (serverId: string) =>
    invoke<void>("benchmarks_run", { serverId }),
  benchmarksCancel: (serverId: string) =>
    invoke<void>("benchmarks_cancel", { serverId }),
  benchmarksHistory: (serverId?: string) =>
    invoke<BenchmarkRun[]>("benchmarks_history", { serverId: serverId ?? null }),
  libraryList: () => invoke<import("./types").LibraryEntry[]>("library_list"),
  libraryImportLocal: (path: string) =>
    invoke<import("./types").LibraryEntry>("library_import_local", { path }),
  libraryRemove: (modelId: string) => invoke<void>("library_remove", { modelId }),
  libraryDiskUsage: () => invoke<number>("library_disk_usage"),
  settingsGet: () => invoke<Settings>("settings_get"),
  gatewayStatus: () => invoke<import("./types").GatewayStatus>("gateway_status"),
  settingsSet: (
    patch: Partial<
      Pick<
        Settings,
        | "distro"
        | "llm_dir"
        | "venv_dir"
        | "llamacpp_dir"
        | "gguf_dir"
        | "llamacpp_executable"
        | "hf_token"
        | "github_token"
        | "default_quant"
        | "advanced_settings"
        | "minimize_to_tray"
        | "auto_restart_crashed"
        | "launch_at_login"
      >
    >
  ) => invoke<Settings>("settings_set", { patch }),
  githubAccess: () => invoke<import("./types").GithubAccess>("github_access"),
  clearGithubToken: () => invoke<Settings>("clear_github_token"),
  autostartGet: () => invoke<boolean>("autostart_get"),
  autostartSet: (enabled: boolean) => invoke<void>("autostart_set", { enabled }),
  wslconfigGet: () => invoke<WslConfigInfo>("wslconfig_get"),
  gpuStatus: () => invoke<import("./types").GpuSnapshot | null>("gpu_status"),
  systemMetricsSeries: () =>
    invoke<import("./types").SystemMetricPoint[]>("system_metrics_series"),
  serverMetricsSeries: (serverId: string) =>
    invoke<import("./types").ServerMetricPoint[]>("server_metrics_series", { serverId }),
  getMemorySettings: () => invoke<MemorySettings>("get_memory_settings"),
  updateMemorySettings: (settings: MemorySettings) =>
    invoke<void>("update_memory_settings", { settings }),
  getSystemMemory: () => invoke<SystemMemoryInfo>("get_system_memory"),
  openUrl: (url: string) => invoke<void>("open_url", { url }),
  wslDistros: () => invoke<string[]>("wsl_distros"),
  configExport: () => invoke<string>("config_export"),
  configImport: (json: string) => invoke<Settings>("config_import", { json }),
  serverRecipeExport: (serverId: string) => invoke<string>("server_recipe_export", { serverId }),
  serverRecipeParse: (json: string) =>
    invoke<import("./types").ServerRecipe>("server_recipe_parse", { json }),
  checkFlashInferReady: () => invoke<import("./types").FlashInferReady>("check_flashinfer_ready"),
  installCudaBuildTools: () => invoke<import("./types").ProvisionReport>("install_cuda_build_tools"),
};

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export const events = {
  wslLog: (cb: (e: WslLogEvent) => void) => listen<WslLogEvent>("wsl-log", (e) => cb(e.payload)),
  serverStatus: (cb: (e: ServerStatusEvent) => void) =>
    listen<ServerStatusEvent>("server-status", (e) => cb(e.payload)),
  serverLog: (cb: (e: ServerLogEvent) => void) => listen<ServerLogEvent>("server-log", (e) => cb(e.payload)),
  serverMetrics: (cb: (e: MetricsSnapshot & { id?: string }) => void) =>
    listen("server-metrics", (e) => cb(e.payload as MetricsSnapshot & { id?: string })),
  pullProgress: (cb: (e: PullStatus) => void) => listen<PullStatus>("pull-progress", (e) => cb(e.payload)),
  chatToken: (cb: (e: ChatTokenPayload) => void) =>
    listen<ChatTokenPayload>("chat-token", (e) => cb(e.payload)),
  chatDone: (cb: (e: ChatDonePayload) => void) =>
    listen<ChatDonePayload>("chat-done", (e) => cb(e.payload)),
  chatCancel: (cb: (e: ChatCancelPayload) => void) =>
    listen<ChatCancelPayload>("chat-cancel", (e) => cb(e.payload)),
  chatError: (cb: (e: ChatErrorPayload) => void) =>
    listen<ChatErrorPayload>("chat-error", (e) => cb(e.payload)),
  benchmarkStep: (cb: (e: BenchmarkStepPayload) => void) =>
    listen<BenchmarkStepPayload>("benchmark-step", (e) => cb(e.payload)),
  benchmarkDone: (cb: (e: BenchmarkRun) => void) =>
    listen<BenchmarkRun>("benchmark-done", (e) => cb(e.payload)),
  benchmarkCancel: (cb: (e: BenchmarkCancelPayload) => void) =>
    listen<BenchmarkCancelPayload>("benchmark-cancel", (e) => cb(e.payload)),
  benchmarkError: (cb: (e: BenchmarkErrorPayload) => void) =>
    listen<BenchmarkErrorPayload>("benchmark-error", (e) => cb(e.payload)),
  llamacppInstallProgress: (cb: (e: { file: string; done: number; total?: number }) => void) =>
    listen("llamacpp-install-progress", (e) => cb(e.payload as { file: string; done: number; total?: number })),
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

export function fmtNum(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function fmtTransferRate(bytesPerSecond: number | null | undefined): string {
  if (bytesPerSecond === null || bytesPerSecond === undefined || !Number.isFinite(bytesPerSecond) || bytesPerSecond < 0) {
    return "Calculating speed…";
  }
  const units = ["B/s", "KB/s", "MB/s", "GB/s"];
  let value = bytesPerSecond;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function fmtTokPerSec(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(2)}k tok/s`;
  return `${n.toFixed(1)} tok/s`;
}

export function fmtGB(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toFixed(2)} GB`;
}

export function fmtContext(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${n}`;
}

export function statusColor(status: string): string {
  switch (status) {
    case "running":
      return "text-emerald-400";
    case "starting":
      return "text-amber-400";
    case "error":
      return "text-red-400";
    default:
      return "text-slate-500";
  }
}

export function quantLabel(q: string): string {
  switch (q.toLowerCase()) {
    case "fp16":
      return "FP16";
    case "fp8":
      return "FP8";
    case "awq":
      return "AWQ (4-bit)";
    case "gptq":
      return "GPTQ (4-bit)";
    default:
      return q;
  }
}