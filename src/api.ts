import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  CreateServerInput,
  EnvStatus,
  FitResultBackend,
  MemorySettings,
  MetricsSnapshot,
  ModelStats,
  ModelWithFit,
  ModelWithStats,
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
  searchModels: (query: string, quant?: string) =>
    invoke<ModelWithStats[]>("search_models", { query, quant: quant ?? null }),
  searchModelsWithFit: (query: string) =>
    invoke<ModelWithFit[]>("search_models_with_fit", { query }),
  recommendedModels: () =>
    invoke<ModelWithFit[]>("recommended_models"),
  modelStats: (modelId: string, quant?: string) =>
    invoke<ModelStats>("model_stats", { modelId, quant: quant ?? null }),
  pullModel: (modelId: string) => invoke<void>("pull_model", { modelId }),
  pullStatus: () => invoke<{ pulling: string[] }>("pull_status"),
  serversList: () => invoke<ServerListRow[]>("servers_list"),
  serversCreate: (input: CreateServerInput) => invoke<ServerDef>("servers_create", { input }),
  serversDelete: (id: string) => invoke<void>("servers_delete", { id }),
  serversStart: (id: string) => invoke<void>("servers_start", { id }),
  serversStop: (id: string) => invoke<void>("servers_stop", { id }),
  serversRestart: (id: string) => invoke<void>("servers_restart", { id }),
  serversLogs: (id: string, since: number) => invoke<string>("servers_logs", { id, since }),
  serversMetrics: (id: string) => invoke<MetricsSnapshot | null>("servers_metrics", { id }),
  serversChat: (id: string, messages: { role: string; content: string }[]) =>
    invoke<Record<string, unknown>>("servers_chat", { id, messages }),
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
  libraryRemove: (modelId: string) => invoke<void>("library_remove", { modelId }),
  libraryDiskUsage: () => invoke<number>("library_disk_usage"),
  settingsGet: () => invoke<Settings>("settings_get"),
  settingsSet: (patch: Partial<Pick<Settings, "distro" | "llm_dir" | "venv_dir" | "hf_token" | "default_quant" | "advanced_settings">>) =>
    invoke<Settings>("settings_set", { patch }),
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
};

export async function getMemorySettings(): Promise<MemorySettings> {
  return invoke<MemorySettings>("get_memory_settings");
}

export async function updateMemorySettings(settings: MemorySettings): Promise<void> {
  return invoke<void>("update_memory_settings", { settings });
}

export async function getSystemMemory(): Promise<SystemMemoryInfo> {
  return invoke<SystemMemoryInfo>("get_system_memory");
}

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
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

export function fmtNum(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
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

export function bytesPerParam(quant: string): number {
  switch (quant.toLowerCase()) {
    case "fp8":
    case "int8":
      return 1.0;
    case "awq":
    case "gptq":
    case "int4":
      return 1.1;
    default:
      return 2.0; // fp16 / bf16
  }
}

export function estimateWeightGb(paramsB: number, quant: string): number {
  return paramsB * bytesPerParam(quant);
}

export type UseCase = "general" | "chat" | "coding" | "reasoning" | "embedding";

export interface ScoreComponents {
  quality: number; // 0-100
  speed: number;   // 0-100
  fit: number;     // 0-100
  context: number; // 0-100
}

export type FitLevel = "perfect" | "good" | "marginal" | "too_tight" | "unknown";
export type FitRating = "optimal" | "tight" | "heavy" | "exceeds" | "unknown";

export interface FitAssessment {
  score: number; // 0-100 llmfit composite score
  fitLevel: FitLevel;
  components: ScoreComponents;
  rating: FitRating; // backwards compatible
  label: string;
  badgeColor: "emerald" | "cyan" | "amber" | "red" | "slate";
  estWeightGb: number | null;
  vramPct: number | null;
  estTokS: number | null;
  reason: string;
}

// Weights per use-case matching AlexsJones/llmfit
const USE_CASE_WEIGHTS: Record<UseCase, [number, number, number, number]> = {
  // [Quality, Speed, Fit, Context]
  chat: [0.40, 0.35, 0.15, 0.10],
  coding: [0.50, 0.20, 0.15, 0.15],
  reasoning: [0.55, 0.15, 0.15, 0.15],
  embedding: [0.30, 0.40, 0.20, 0.10],
  general: [0.45, 0.30, 0.15, 0.10],
};

// Quantization quality penalties matching llmfit
function quantizationPenalty(quant: string): number {
  switch (quant.toLowerCase()) {
    case "fp16":
    case "bf16":
      return 1.0;
    case "fp8":
    case "int8":
      return 0.95;
    case "awq":
    case "gptq":
      return 0.88;
    case "int4":
      return 0.85;
    default:
      return 0.92;
  }
}

/**
 * Calculates a 0-100 multi-dimensional fit score inspired by AlexsJones/llmfit.
 */
export function computeLlmfitScore(
  paramsB: number,
  quant: string,
  totalVramMb: number,
  bandwidthGbs: number,
  contextTokens: number,
  useCase: UseCase = "general",
  taskQualityPrior?: number
): { score: number; components: ScoreComponents; fitLevel: FitLevel; estTokS: number; memoryNeededGb: number; vramRatio: number } {
  const estWeightGb = estimateWeightGb(paramsB, quant);
  const overheadGb = 2.0; // CUDA + vLLM context buffer
  const memoryNeededGb = estWeightGb + overheadGb;
  const totalVramGb = totalVramMb / 1024;
  const vramRatio = memoryNeededGb / totalVramGb;

  // 1. Fit Level categorization (matching llmfit FIT_*_MAX_RATIO)
  let fitLevel: FitLevel;
  if (vramRatio <= 0.60) {
    fitLevel = "perfect";
  } else if (vramRatio <= 0.85) {
    fitLevel = "good";
  } else if (vramRatio <= 0.98) {
    fitLevel = "marginal";
  } else {
    fitLevel = "too_tight";
  }

  // 2. Pillar: Quality (0 - 100)
  // Scaled by log10(params) mapped against modern standard frontier (1B -> ~50, 7B -> ~75, 70B -> ~95)
  // Adjusted by quantization retention and use-case task rating
  const baseParamQuality = Math.min(100, Math.max(25, 45 + 30 * Math.log10(Math.max(paramsB, 0.5))));
  const quantMult = quantizationPenalty(quant);
  const taskPrior = taskQualityPrior ? taskQualityPrior / 100 : 1.0;
  const quality = Math.min(100, Math.max(0, Math.round(baseParamQuality * quantMult * taskPrior)));

  // 3. Pillar: Speed (0 - 100)
  // Bandwidth Roofline model: tok/s = (bandwidth_GB_s / est_model_size_GB) * efficiency (0.55)
  const eff = 0.55;
  const estTokS = estWeightGb > 0 ? (bandwidthGbs / estWeightGb) * eff : 0;
  // Speed score: 80 tok/s maps to 100 pts, 30 tok/s maps to ~65 pts
  const speed = Math.min(100, Math.max(0, Math.round((estTokS / 80.0) * 100)));

  // 4. Pillar: Fit (0 - 100)
  // llmfit rewards 40-75% sweet spot. Under 30% has slight underutilization penalty. Over 85% drops sharply.
  let fitScore = 0;
  if (vramRatio <= 0.60) {
    // 0.40 -> 100, 0.10 -> 80
    fitScore = Math.round(75 + 25 * (vramRatio / 0.60));
  } else if (vramRatio <= 0.85) {
    // 60-85% is prime for large context without OOM
    fitScore = Math.round(100 - ((vramRatio - 0.60) / 0.25) * 15);
  } else if (vramRatio <= 0.98) {
    // Marginal: 85 - 40
    fitScore = Math.round(85 - ((vramRatio - 0.85) / 0.13) * 45);
  } else {
    // Too tight / OOM danger
    fitScore = Math.max(0, Math.round(30 - (vramRatio - 0.98) * 100));
  }

  // 5. Pillar: Context (0 - 100)
  // Normalized against 32k benchmark target
  const context = Math.min(100, Math.max(15, Math.round((contextTokens / 32768) * 85)));

  // Weighted composite score
  const [wQ, wS, wF, wC] = USE_CASE_WEIGHTS[useCase] || USE_CASE_WEIGHTS.general;
  let composite = Math.round(quality * wQ + speed * wS + fitScore * wF + context * wC);

  // Severe penalty if it exceeds VRAM (llmfit: TooTight models have crushed scores)
  if (fitLevel === "too_tight") {
    composite = Math.min(composite, 25);
  }

  return {
    score: composite,
    components: {
      quality,
      speed,
      fit: fitScore,
      context,
    },
    fitLevel,
    estTokS: Math.round(estTokS * 10) / 10,
    memoryNeededGb: Math.round(memoryNeededGb * 10) / 10,
    vramRatio,
  };
}

export function evaluateSystemFit(
  paramsB: number | null | undefined,
  quant: string,
  totalVramMb: number | null | undefined,
  bandwidthGbs: number = 300,
  contextTokens: number = 32768,
  useCase: UseCase = "general",
  taskQualityPrior?: number
): FitAssessment {
  if (paramsB == null || paramsB <= 0) {
    return {
      score: 0,
      fitLevel: "unknown",
      components: { quality: 0, speed: 0, fit: 0, context: 0 },
      rating: "unknown",
      label: "Unknown Fit",
      badgeColor: "slate",
      estWeightGb: null,
      vramPct: null,
      estTokS: null,
      reason: "Model parameters count is missing or unindexed",
    };
  }

  const estWeightGb = estimateWeightGb(paramsB, quant);
  if (!totalVramMb || totalVramMb <= 0) {
    return {
      score: 50,
      fitLevel: "unknown",
      components: { quality: 50, speed: 50, fit: 50, context: 50 },
      rating: "unknown",
      label: "Fits ~" + estWeightGb.toFixed(1) + " GB",
      badgeColor: "slate",
      estWeightGb,
      vramPct: null,
      estTokS: null,
      reason: "GPU VRAM could not be verified",
    };
  }

  const llmfit = computeLlmfitScore(
    paramsB,
    quant,
    totalVramMb,
    bandwidthGbs,
    contextTokens,
    useCase,
    taskQualityPrior
  );

  const vramPct = Math.round(llmfit.vramRatio * 100);

  switch (llmfit.fitLevel) {
    case "perfect":
      return {
        score: llmfit.score,
        fitLevel: "perfect",
        components: llmfit.components,
        rating: "optimal",
        label: `${llmfit.score}/100 · Perfect`,
        badgeColor: "emerald",
        estWeightGb,
        vramPct,
        estTokS: llmfit.estTokS,
        reason: `Perfect fit (${vramPct}% VRAM). Generous room for 32k+ KV cache and peak generation throughput.`,
      };
    case "good":
      return {
        score: llmfit.score,
        fitLevel: "good",
        components: llmfit.components,
        rating: "optimal",
        label: `${llmfit.score}/100 · Good`,
        badgeColor: "cyan",
        estWeightGb,
        vramPct,
        estTokS: llmfit.estTokS,
        reason: `Good fit (${vramPct}% VRAM). Balanced parameter density with reliable runtime memory headroom.`,
      };
    case "marginal":
      return {
        score: llmfit.score,
        fitLevel: "marginal",
        components: llmfit.components,
        rating: "tight",
        label: `${llmfit.score}/100 · Marginal`,
        badgeColor: "amber",
        estWeightGb,
        vramPct,
        estTokS: llmfit.estTokS,
        reason: `Marginal fit (${vramPct}% VRAM). Tight KV cache space; large prompts or high batch concurrency may OOM.`,
      };
    case "too_tight":
    default:
      return {
        score: llmfit.score,
        fitLevel: "too_tight",
        components: llmfit.components,
        rating: "exceeds",
        label: `${llmfit.score}/100 · Too Tight`,
        badgeColor: "red",
        estWeightGb,
        vramPct,
        estTokS: llmfit.estTokS,
        reason: `Exceeds safe VRAM limit (~${llmfit.memoryNeededGb} GB required vs ${(totalVramMb / 1024).toFixed(1)} GB GPU). OOM likely.`,
      };
  }
}

export const SUPPORTED_QUANTS = ["fp16", "fp8", "awq", "gptq"] as const;
export type SupportedQuant = (typeof SUPPORTED_QUANTS)[number];

export interface QuantFitSummary {
  quant: SupportedQuant;
  label: string;
  fit: FitAssessment;
}

export interface RecommendedQuantResult {
  bestQuant: SupportedQuant;
  fit: FitAssessment;
  allQuants: QuantFitSummary[];
}

export function recommendBestQuant(
  paramsB: number | null | undefined,
  totalVramMb: number | null | undefined,
  bandwidthGbs: number = 300,
  contextTokens: number = 32768,
  useCase: UseCase = "general",
  taskQualityPrior?: number
): RecommendedQuantResult {
  const quants: SupportedQuant[] = ["fp16", "fp8", "awq", "gptq"];
  const allQuants: QuantFitSummary[] = quants.map((q) => ({
    quant: q,
    label: quantLabel(q),
    fit: evaluateSystemFit(
      paramsB,
      q,
      totalVramMb,
      bandwidthGbs,
      contextTokens,
      useCase,
      taskQualityPrior
    ),
  }));

  if (paramsB == null || paramsB <= 0 || !totalVramMb || totalVramMb <= 0) {
    return {
      bestQuant: "fp16",
      fit: allQuants[0].fit,
      allQuants,
    };
  }

  // Preference hierarchy:
  // 1. If FP16 fits with "perfect", recommend FP16 (maximum quality retention)
  // 2. If FP8 fits with "perfect", recommend FP8
  // 3. If AWQ fits with "perfect", recommend AWQ
  // 4. Same check for "good" (FP16 -> FP8 -> AWQ)
  // 5. Otherwise, pick the quant with the highest composite score
  for (const q of ["fp16", "fp8", "awq"] as const) {
    const found = allQuants.find((item) => item.quant === q && item.fit.fitLevel === "perfect");
    if (found) {
      return { bestQuant: found.quant, fit: found.fit, allQuants };
    }
  }

  for (const q of ["fp16", "fp8", "awq"] as const) {
    const found = allQuants.find((item) => item.quant === q && item.fit.fitLevel === "good");
    if (found) {
      return { bestQuant: found.quant, fit: found.fit, allQuants };
    }
  }

  const sorted = [...allQuants].sort((a, b) => b.fit.score - a.fit.score);
  return {
    bestQuant: sorted[0].quant,
    fit: sorted[0].fit,
    allQuants,
  };
}