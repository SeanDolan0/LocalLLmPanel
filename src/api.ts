import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  CreateServerInput,
  EnvStatus,
  MetricsSnapshot,
  ModelStats,
  ModelWithStats,
  ProvisionReport,
  PullStatus,
  ServerDef,
  ServerListRow,
  ServerLogEvent,
  ServerStatusEvent,
  Settings,
  WslConfigInfo,
  WslLogEvent,
} from "./types";

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

export const api = {
  envStatus: () => invoke<EnvStatus>("env_status"),
  provision: () => invoke<ProvisionReport>("provision"),
  searchModels: (query: string, quant?: string) =>
    invoke<ModelWithStats[]>("search_models", { query, quant: quant ?? null }),
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
  libraryList: () => invoke<import("./types").LibraryEntry[]>("library_list"),
  settingsGet: () => invoke<Settings>("settings_get"),
  settingsSet: (patch: Partial<Pick<Settings, "distro" | "llm_dir" | "venv_dir" | "hf_token" | "default_quant">>) =>
    invoke<Settings>("settings_set", { patch }),
  wslconfigGet: () => invoke<WslConfigInfo>("wslconfig_get"),
  gpuStatus: () => invoke<import("./types").GpuSnapshot | null>("gpu_status"),
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

export function fmtBytesToMB(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return `${(n / (1024 * 1024)).toFixed(0)} MB`;
}

export function fmtContext(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${n}`;
}

export function timeAgo(ms: number | null | undefined): string {
  if (!ms) return "never";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
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