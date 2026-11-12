import { ServerCompatibilityModal } from "../components/ServerCompatibilityModal";
import { ErrorAlert } from "../components/ErrorAlert";
import { StoragePlanner } from "../components/StoragePlanner";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { api, events, fmtNum, fmtTokPerSec, quantLabel, statusColor } from "../api";
import { Badge, Button, Card, CardTitle, Collapsible, Field, HelpBubble, Menu, inputCls, Spinner } from "../ui";
import { Sparkline } from "../components/Sparkline";
import { ServerMetricsPanel } from "../components/ServerMetricsPanel";
import { effectiveModelName } from "../types";
import type { ContextFitReport, ServerListRow, ServerDef, ChatMessage, Conversation, BenchmarkRun, ServerMetricPoint, GpuSnapshot, LlamaDevice, LibraryEntry } from "../types";
import { estimateServerWeights } from "../utils/weightEstimation";

const SECRET_PLACEHOLDER = "__LLM_PANEL_SECRET_REDACTED__";

export default function Servers() {
  const location = useLocation();
  const [rows, setRows] = useState<ServerListRow[]>([]);
  const [serverSeries, setServerSeries] = useState<Record<string, ServerMetricPoint[]>>({});
  const [gpu, setGpu] = useState<GpuSnapshot | null>(null);
  const [llamaDevices, setLlamaDevices] = useState<LlamaDevice[]>([]);
  const [showNew, setShowNew] = useState(false);
  const [prefillModel, setPrefillModel] = useState<string | null>(null);
  const [prefillBackend, setPrefillBackend] = useState<"vllm" | "llamacpp" | undefined>(undefined);
  const [prefillModelPath, setPrefillModelPath] = useState<string | undefined>(undefined);
  const [prefillQuant, setPrefillQuant] = useState<string | null>(null);
  const [prefillSwapSpace, setPrefillSwapSpace] = useState<number | undefined>(undefined);
  const [prefillCpuOffload, setPrefillCpuOffload] = useState<number | undefined>(undefined);
  const [prefillMaxLen, setPrefillMaxLen] = useState<number | undefined>(undefined);
  const [prefillVramContext, setPrefillVramContext] = useState<number | undefined>(undefined);
  const [prefillGpuUtil, setPrefillGpuUtil] = useState<number | undefined>(undefined);
  const [prefillServed, setPrefillServed] = useState<string | undefined>(undefined);
  const [prefillKvDtype, setPrefillKvDtype] = useState<string | undefined>(undefined);
  const [prefillTask, setPrefillTask] = useState<"instruct" | "embed" | undefined>(undefined);
  const [prefillChannel, setPrefillChannel] = useState<string | undefined>(undefined);
  const [prefillSpeculativeEnabled, setPrefillSpeculativeEnabled] = useState<boolean | undefined>(undefined);
  const [prefillSpeculativeModel, setPrefillSpeculativeModel] = useState<string | undefined>(undefined);
  const [prefillSpeculativeModelPath, setPrefillSpeculativeModelPath] = useState<string | undefined>(undefined);
  const [prefillSpeculativeNumTokens, setPrefillSpeculativeNumTokens] = useState<number | undefined>(undefined);

  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [copiedUrlId, setCopiedUrlId] = useState<string | null>(null);
  const [showImportRecipe, setShowImportRecipe] = useState(false);
    const [compatTestServer, setCompatTestServer] = useState<ServerListRow | null>(null);
  const [recipeInput, setRecipeInput] = useState("");
  const [recipeErr, setRecipeErr] = useState<string | null>(null);
  const [toolResults, setToolResults] = useState<Record<string, string>>({});
  const [toolMaxTokens, setToolMaxTokens] = useState(2048);
  const [toolDisableThinking, setToolDisableThinking] = useState(true);
  // Advanced per-row tooling, opened from the overflow menu instead of inline
  const [toolsOpen, setToolsOpen] = useState<Record<string, boolean>>({});
  const [benchFor, setBenchFor] = useState<string | null>(null);

  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // server id being start/stop/delete
  const [editingServer, setEditingServer] = useState<ServerDef | null>(null);
  
  // log buffers per server (event-driven + hydrated)
  const [logs, setLogs] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const logEndRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const clearPrefills = () => {
    setPrefillModel(null);
    setPrefillBackend(undefined);
    setPrefillModelPath(undefined);
    setPrefillQuant(null);
    setPrefillSwapSpace(undefined);
    setPrefillCpuOffload(undefined);
    setPrefillMaxLen(undefined);
    setPrefillVramContext(undefined);
    setPrefillGpuUtil(undefined);
    setPrefillServed(undefined);
    setPrefillKvDtype(undefined);
    setPrefillTask(undefined);
    setPrefillChannel(undefined);
    setPrefillSpeculativeEnabled(undefined);
    setPrefillSpeculativeModel(undefined);
    setPrefillSpeculativeModelPath(undefined);
    setPrefillSpeculativeNumTokens(undefined);
    setEditingServer(null);
  };

  const copyRecipe = async (def: ServerDef) => {
    try {
      const jsonStr = await api.serverRecipeExport(def.id);
      await navigator.clipboard.writeText(jsonStr);
      setCopiedId(def.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (e) {
      setErr(`Failed to copy recipe: ${e}`);
    }
  };

  const applyRecipe = async () => {
    setRecipeErr(null);
    if (!recipeInput.trim()) return;
    try {
      const r = await api.serverRecipeParse(recipeInput.trim());
      setPrefillModel(r.model_id);
      setPrefillQuant(r.quant);
      setPrefillSwapSpace(r.swap_space_gb ?? undefined);
      setPrefillCpuOffload(r.cpu_offload_gb ?? undefined);
      setPrefillMaxLen(r.max_model_len && r.max_model_len > 0 ? r.max_model_len : undefined);
      setPrefillGpuUtil(r.gpu_mem_util);
      setPrefillServed(r.served_model_name ?? undefined);
      setPrefillKvDtype(r.kv_cache_dtype ?? undefined);
      setPrefillTask(r.task as "instruct" | "embed");
      setPrefillSpeculativeEnabled(r.speculative_enabled ?? false);
      setPrefillSpeculativeModel(r.speculative_model ?? undefined);
      setPrefillSpeculativeModelPath(r.speculative_model_path ?? undefined);
      setPrefillSpeculativeNumTokens(r.speculative_num_tokens ?? undefined);
      setShowImportRecipe(false);
      setRecipeInput("");
      setShowNew(true);
    } catch (e) {
      setRecipeErr(`Invalid recipe JSON: ${e}`);
    }
  };

  // Check navigation state for prefill model, quant, swap, offload, and context (e.g. from Search or Library Deploy)
  useEffect(() => {
    const state = location.state as {
      prefillModel?: string;
      prefillBackend?: "vllm" | "llamacpp";
      prefillModelPath?: string;
      prefillQuant?: string;
      prefillSwapSpace?: number;
      prefillCpuOffload?: number;
      prefillMaxLen?: number;
      prefillVramContext?: number;
      prefillGpuUtil?: number;
      prefillServed?: string;
      prefillKvDtype?: string;
      prefillTask?: "instruct" | "embed";
      prefillChannel?: string;
      prefillSpeculativeEnabled?: boolean;
      prefillSpeculativeModel?: string;
      prefillSpeculativeModelPath?: string;
      prefillSpeculativeNumTokens?: number;
    } | null;
    if (state?.prefillModel) {
      setPrefillModel(state.prefillModel);
      if (state.prefillBackend) {
        setPrefillBackend(state.prefillBackend);
      }
      if (state.prefillModelPath) {
        setPrefillModelPath(state.prefillModelPath);
      }
      if (state.prefillQuant) {
        setPrefillQuant(state.prefillQuant);
      }
      if (state.prefillSwapSpace !== undefined) {
        setPrefillSwapSpace(state.prefillSwapSpace);
      }
      if (state.prefillCpuOffload !== undefined) {
        setPrefillCpuOffload(state.prefillCpuOffload);
      }
      if (state.prefillMaxLen !== undefined && state.prefillMaxLen > 0) {
        setPrefillMaxLen(state.prefillMaxLen);
      }
      if (state.prefillVramContext !== undefined) {
        setPrefillVramContext(state.prefillVramContext);
      }
      if (state.prefillGpuUtil !== undefined) {
        setPrefillGpuUtil(state.prefillGpuUtil);
      }
      if (state.prefillSpeculativeEnabled !== undefined) {
        setPrefillSpeculativeEnabled(state.prefillSpeculativeEnabled);
      }
      if (state.prefillSpeculativeModel !== undefined) {
        setPrefillSpeculativeModel(state.prefillSpeculativeModel);
      }
      if (state.prefillSpeculativeModelPath !== undefined) {
        setPrefillSpeculativeModelPath(state.prefillSpeculativeModelPath);
      }
      if (state.prefillSpeculativeNumTokens !== undefined) {
        setPrefillSpeculativeNumTokens(state.prefillSpeculativeNumTokens);
      }
      if (state.prefillServed !== undefined) {
        setPrefillServed(state.prefillServed);
      }
      if (state.prefillKvDtype !== undefined) {
        setPrefillKvDtype(state.prefillKvDtype);
      }
      if (state.prefillTask !== undefined) {
        setPrefillTask(state.prefillTask);
      }
      if (state.prefillChannel !== undefined) {
        setPrefillChannel(state.prefillChannel);
      }
      setShowNew(true);
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  // Populate form when editing a server
  useEffect(() => {
    if (editingServer) {
      setPrefillModel(editingServer.model_id);
      setPrefillBackend(editingServer.backend as "vllm" | "llamacpp");
      setPrefillModelPath(editingServer.model_path ?? undefined);
      setPrefillQuant(editingServer.quant);
      setPrefillSwapSpace(editingServer.swap_space_gb ?? undefined);
      setPrefillCpuOffload(editingServer.cpu_offload_gb ?? undefined);
      setPrefillMaxLen(editingServer.max_model_len ?? undefined);
      setPrefillGpuUtil(editingServer.gpu_mem_util);
      setPrefillServed(editingServer.served_model_name ?? undefined);
      setPrefillKvDtype(editingServer.kv_cache_dtype ?? undefined);
      setPrefillTask(editingServer.task as "instruct" | "embed");
      setPrefillChannel(editingServer.llamacpp_channel);
      setShowNew(true);
    }
  }, [editingServer]);

  const refresh = useCallback(() => {
    api
      .serversList()
      .then(async (newRows) => {
        setRows(newRows);
        const running = newRows.filter((r) => r.status === "running");
        if (running.length > 0) {
          const seriesMap: Record<string, ServerMetricPoint[]> = {};
          await Promise.all(
            running.map(async (r) => {
              try {
                const pts = await api.serverMetricsSeries(r.def.id);
                seriesMap[r.def.id] = pts;
              } catch (e) {
                setErr(`Metrics unavailable for ${r.def.id}: ${e}`);
              }
            })
          );
          setServerSeries((prev) => ({ ...prev, ...seriesMap }));
        }
      })
      .catch((e) => setErr(`Failed to refresh servers: ${e}`));
    api.gpuStatus().then(setGpu).catch((e) => setErr(`GPU status unavailable: ${e}`));
  }, []);

  useEffect(() => {
    api.llamacppStatus().then((status) => setLlamaDevices(status.devices ?? [])).catch(() => {});
    refresh();
    const t = setInterval(refresh, 4000);
    const unsubs = [
      events.serverLog((e) => {
        setLogs((prev) => ({ ...prev, [e.id]: (prev[e.id] ?? "") + e.line + "\n" }));
      }),
      events.serverStatus((e) => {
        setRows((prev) =>
          prev.map((r) =>
            r.def.id === e.id ? { ...r, status: e.status, error: e.error } : r
          )
        );
        if (e.status === "stopped") setLogs((prev) => ({ ...prev, [e.id]: "" }));
      }),
    ];
    return () => {
      clearInterval(t);
      unsubs.forEach((u) => u.then((f) => f()));
    };
  }, [refresh]);

  // Hydrate logs when selecting a server
  useEffect(() => {
    if (selected) {
      api
        .serversLogs(selected, 0)
        .then((historical) => {
          if (historical) {
            setLogs((prev) => ({
              ...prev,
              [selected]: historical,
            }));
          }
        })
        .catch(() => {});
    }
  }, [selected]);

  // Auto-scroll selected log
  useEffect(() => {
    if (selected) {
      const el = logEndRefs.current[selected];
      if (el) el.scrollIntoView({ block: "end" });
    }
  }, [logs, selected]);

  const act = async (id: string, fn: () => Promise<unknown>) => {
    setBusy(id);
    setErr(null);
    try {
      await fn();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
      refresh();
    }
  };

  const runToolTest = async (def: ServerDef) => {
    try {
      const result = await api.serversTestToolCall(def.id, toolMaxTokens, toolDisableThinking);
      const timing = result.response.timings as { prompt_per_second?: number; predicted_per_second?: number } | undefined;
      const timingText = timing
        ? `\nTiming: prompt ${timing.prompt_per_second ?? "?"} tok/s, predicted ${timing.predicted_per_second ?? "?"} tok/s. The first request after loading is slower while weights are paged in.`
        : "";
      const callText = result.tool_call
        ? `\nParsed call: ${JSON.stringify(result.tool_call)}`
        : "";
      const reasoningText = result.reasoning_content
        ? `\nReasoning content:\n${result.reasoning_content}`
        : "";
      setToolResults((prev) => ({
        ...prev,
        [def.id]: result.passed
          ? `Tool call passed${callText}${timingText}`
          : `Failed: ${result.hint}${reasoningText}${timingText}\nRaw response:\n${JSON.stringify(result.response, null, 2)}`,
      }));
    } catch (e) {
      setToolResults((prev) => ({ ...prev, [def.id]: `Tool call error: ${String(e)}` }));
    }
  };

  const selectedRow = rows.find((r) => r.def.id === selected) ?? null;
  const selectedLog = selected ? logs[selected] ?? "" : "";
  const freeGb = gpu ? gpu.vram_free_mb / 1024 : null;
  const totalGb = gpu ? gpu.vram_total_mb / 1024 : 0;

  const runningWeights = useMemo(() => {
    let mainWeightGb = 0;
    let specWeightGb = 0;
    let explicitOffloadGb = 0;
    for (const r of rows) {
      if (r.status === "running") {
        const est = estimateServerWeights(r.def);
        mainWeightGb += est.mainWeightGb;
        specWeightGb += est.specWeightGb;
        if (r.def.cpu_offload_gb && r.def.cpu_offload_gb > 0) {
          explicitOffloadGb += r.def.cpu_offload_gb;
        } else if (r.def.backend === "llamacpp" && r.def.n_gpu_layers === 0) {
          explicitOffloadGb += est.totalWeightGb;
        }
      }
    }
    return {
      mainWeightGb,
      specWeightGb,
      totalWeightGb: mainWeightGb + specWeightGb,
      explicitOffloadGb,
    };
  }, [rows]);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      {/* VRAM Pressure Alert */}
      {gpu && freeGb !== null && (() => {
        const overheadGb = 1.0;
        const usableCapacityGb = Math.max(0, totalGb - overheadGb);
        const reqGpuWeightGb = Math.max(0, runningWeights.totalWeightGb - runningWeights.explicitOffloadGb);
        const capacitySpillGb = totalGb > 0 ? Math.max(0, reqGpuWeightGb - usableCapacityGb) : 0;
        const loadedIntoRamGb = runningWeights.explicitOffloadGb + capacitySpillGb;
        const isLoadedIntoRam = runningWeights.totalWeightGb > 0 && loadedIntoRamGb > 0.01;
        const isLowHeadroom = freeGb < 1.0;

        if (!isLoadedIntoRam && !isLowHeadroom) return null;

        return (
          <div className="rounded-xl border border-amber-500/50 bg-amber-500/10 p-3.5 text-xs text-amber-200 shadow-sm flex items-center gap-2.5">
            <span className="text-base">⚠️</span>
            <div>
              <strong>Low VRAM Warning:</strong>{" "}
              {isLoadedIntoRam ? (
                <>
                  Active model weights ({runningWeights.totalWeightGb.toFixed(2)} GB
                  {runningWeights.specWeightGb > 0 ? ` [model: ${runningWeights.mainWeightGb.toFixed(2)} GB, speculative draft: ${runningWeights.specWeightGb.toFixed(2)} GB]` : ""}
                  ) exceed GPU VRAM capacity ({totalGb.toFixed(2)} GB total, ~{usableCapacityGb.toFixed(2)} GB usable for weights).{" "}
                  <strong>Your model is getting loaded into system RAM by ~{loadedIntoRamGb.toFixed(2)} GB.</strong>{" "}
                  (Counts model and speculative decoding weights; context is excluded). Running weights from system RAM will reduce generation speed.
                </>
              ) : (
                <>
                  Less than 1.0 GB GPU VRAM headroom remaining ({freeGb.toFixed(2)} GB free). Current model weights fit in VRAM (0 GB loaded into system RAM).
                </>
              )}
            </div>
          </div>
        );
      })()}

      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-100">Servers</h1>
        <div className="flex items-center gap-2">
          <Button variant="subtle" onClick={() => setShowImportRecipe(true)}>
            Import Recipe
          </Button>
          <Button onClick={() => setShowNew((s) => !s)}>{showNew ? "Cancel" : "+ New server"}</Button>
        </div>
      </div>

      {showImportRecipe && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-xl border border-edge bg-surface-1 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-slate-100">Import Server Recipe</h2>
              <button
                onClick={() => {
                  setShowImportRecipe(false);
                  setRecipeErr(null);
                }}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Paste a portable server recipe JSON snippet to prefill all configuration fields for a new server.
            </p>
            {recipeErr && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2.5 text-xs text-red-300">
                {recipeErr}
              </div>
            )}
            <textarea
              className="w-full h-44 rounded-lg border border-edge bg-surface-2 p-3 font-mono text-xs text-slate-200 focus:border-accent focus:outline-none"
              placeholder={'{\n  "schema": "local-llm-panel/server-recipe/v1",\n  "model_id": "meta-llama/Llama-3.1-8B-Instruct",\n  "task": "instruct",\n  "port": 8000,\n  "gpu_mem_util": 0.9,\n  "quant": "awq"\n}'}
              value={recipeInput}
              onChange={(e) => setRecipeInput(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setShowImportRecipe(false)}>
                Cancel
              </Button>
              <Button onClick={applyRecipe} disabled={!recipeInput.trim()}>
                Apply Recipe
              </Button>
            </div>
          </div>
        </div>
      )}

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{err}</div>}

      {showNew && (
        <NewServerForm
          freeGb={freeGb}
          totalGb={totalGb}
          devices={llamaDevices}
          initialModelId={prefillModel ?? ""}
          initialBackend={prefillBackend}
          initialModelPath={prefillModelPath}
          initialQuant={prefillQuant ?? undefined}
          initialSwapSpace={prefillSwapSpace}
          initialCpuOffload={prefillCpuOffload}
          initialMaxLen={prefillMaxLen}
          initialVramContext={prefillVramContext}
          initialGpuUtil={prefillGpuUtil}
          initialServed={prefillServed}
          initialKvDtype={prefillKvDtype}
          initialTask={prefillTask}
          initialChannel={prefillChannel}
          initialEnv={editingServer?.env ?? {}}
          initialSpeculativeEnabled={prefillSpeculativeEnabled}
          initialSpeculativeModel={prefillSpeculativeModel}
          initialSpeculativeModelPath={prefillSpeculativeModelPath}
          initialSpeculativeNumTokens={prefillSpeculativeNumTokens}
          initialServer={editingServer}
          onDone={(s) => {
            setShowNew(false);
            clearPrefills();
            setSelected(s.id);
            refresh();
          }}
          onCancel={() => {
            setShowNew(false);
            clearPrefills();
          }}
          onErr={setErr}
        />
      )}

      {rows.length === 0 && !showNew ? (
        <Card>
          <CardTitle>No servers yet</CardTitle>
          <div className="text-sm text-slate-500">
            Define servers pointing at any HF model id (e.g. <code>Qwen/Qwen2.5-0.5B-Instruct</code> for instruct or{" "}
            <code>BAAI/bge-small-en-v1.5</code> for embeddings). One vLLM process per server.
          </div>
        </Card>
      ) : (
        <div className="space-y-2.5">
          {rows.map((r) => (
            <Card key={r.def.id} className={`p-4 rounded-2xl border border-edge/80 bg-surface-2 transition-all hover:border-accent/40 ${selected === r.def.id ? "ring-1 ring-accent/50 bg-surface-2/95" : ""}`}>
              <div className="flex flex-wrap items-center gap-3">
                <button className="min-w-0 flex-1 text-left group" onClick={() => setSelected(selected === r.def.id ? null : r.def.id)}>
                  <div className="flex items-center gap-2">
                    <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${
                      r.status === "running" ? "bg-emerald-400 animate-pulse" :
                      r.status === "starting" ? "bg-amber-400 animate-pulse" :
                      r.status === "error" ? "bg-red-400" : "bg-slate-600"
                    }`} />
                    <span className="font-medium text-slate-200">{cleanModelName(r.def.name)}</span>
                    <Badge color={r.status === "running" ? "emerald" : r.status === "starting" ? "amber" : r.status === "error" ? "red" : "slate"}>
                      {r.status}
                    </Badge>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-slate-500 group-hover:text-slate-400">
                    <span>{r.def.model_id}</span>
                    <span className="text-accent">{r.def.backend === "llamacpp" ? "llama.cpp · Windows" : "vLLM · WSL2"}</span>
                    <span>·</span>
                    <span>port {r.def.port}</span>
                    <span>·</span>
                    <span>{r.def.task}</span>
                    <span>·</span>
                    <span>{quantLabel(r.def.quant)}</span>
                    {r.def.backend === "vllm" && r.def.max_model_len ? <span>· ctx {fmtNum(r.def.max_model_len)}</span> : null}
                    {r.def.backend === "llamacpp" && r.def.ctx_size ? <span>· ctx {fmtNum(r.def.ctx_size)}</span> : null}
                    {r.def.params_b ? <span>· ~{r.def.params_b.toFixed(2)}B</span> : null}
                    {r.def.backend === "vllm" && r.def.swap_space_gb != null && r.def.swap_space_gb > 0 && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                        swap {r.def.swap_space_gb}GB
                      </span>
                    )}
                    {r.def.backend === "vllm" && r.def.cpu_offload_gb != null && r.def.cpu_offload_gb > 0 && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-950/80 text-amber-300 border border-amber-800">
                        offload {r.def.cpu_offload_gb}GB
                      </span>
                    )}
                    {r.def.speculative_enabled && (
                      <span
                        className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-950/80 text-purple-300 border border-purple-800"
                        title={`Speculative decoding enabled: ${r.def.speculative_type || "draft-simple"} (${r.def.speculative_model || r.def.speculative_model_path || "configured"}, max ${r.def.speculative_num_tokens ?? 5}t${r.def.speculative_draft_ngl ? `, ngl: ${r.def.speculative_draft_ngl}` : ""}${r.def.speculative_draft_p_min != null ? `, p-min: ${r.def.speculative_draft_p_min}` : ""})`}
                      >
                        spec {r.def.speculative_type ? `${r.def.speculative_type.replace("draft-", "")} ` : ""}{r.def.speculative_num_tokens ?? 5}t
                      </span>
                    )}
                  </div>
                </button>
                <div className="flex items-center gap-2">
                  {r.status === "running" && r.metrics?.measured?.tokens_per_sec != null && (
                    <span className="font-mono text-xs text-cyan-300">
                      {fmtTokPerSec(r.metrics.measured.tokens_per_sec)}
                    </span>
                  )}
                  {r.status !== "running" && r.status !== "starting" && (
                    <Button variant="primary" disabled={busy === r.def.id} onClick={() => act(r.def.id, () => api.serversStart(r.def.id))}>
                      Start
                    </Button>
                  )}
                  {(r.status === "running" || r.status === "starting") && (
                    <Button variant="danger" disabled={busy === r.def.id} onClick={() => act(r.def.id, () => api.serversStop(r.def.id))}>
                      Stop
                    </Button>
                  )}
                  {r.status === "running" && r.def.task === "instruct" && (
                    <ChatButton serverId={r.def.id} port={r.def.port} model={effectiveModelName(r.def)} />
                  )}
                  {r.status === "running" && (
                    <Button
                      variant="subtle"
                      onClick={() => setCompatTestServer(r)}
                      title="Run live compatibility test battery against this server"
                    >
                      🧪 Test
                    </Button>
                  )}
                  <Menu
                    items={[
                      ...(r.status === "running"
                        ? [
                            {
                              label: copiedUrlId === r.def.id ? "URL copied ✓" : "Copy API URL",
                              onClick: async () => {
                                await navigator.clipboard.writeText(`http://127.0.0.1:${r.def.port}/v1`);
                                setCopiedUrlId(r.def.id);
                                setTimeout(() => setCopiedUrlId(null), 2000);
                              },
                            },
                            {
                              label: "Restart server",
                              disabled: busy === r.def.id,
                              onClick: () => act(r.def.id, () => api.serversRestart(r.def.id)),
                            },
                            ...(r.def.task === "instruct"
                              ? [
                                  {
                                    label: toolsOpen[r.def.id] ? "Hide tool test" : "Tool-calling test",
                                    onClick: () => setToolsOpen((s) => ({ ...s, [r.def.id]: !s[r.def.id] })),
                                  },
                                  {
                                    label: "Benchmark",
                                    onClick: () => setBenchFor(r.def.id),
                                  },
                                ]
                              : []),
                          ]
                        : []),
                      ...(r.def.backend === "vllm"
                        ? [
                            {
                              label: copiedId === r.def.id ? "Recipe copied ✓" : "Copy recipe (JSON)",
                              onClick: () => copyRecipe(r.def),
                            },
                          ]
                        : []),
                      {
                        label: "Edit server",
                        onClick: () => {
                          setEditingServer(r.def);
                          setShowNew(true);
                        },
                      },
                      {
                        label: "Delete server",
                        danger: true,
                        onClick: () => {
                          if (confirm(`Delete server "${cleanModelName(r.def.name)}"?`)) act(r.def.id, () => api.serversDelete(r.def.id));
                        },
                      },
                    ]}
                  />
                </div>
              </div>
              {r.error && (
                <ErrorAlert
                  source="server"
                  error={r.error}
                  title={`Server Engine Error (${r.def.name})`}
                  className="mt-2"
                />
              )}
              {toolsOpen[r.def.id] && r.status === "running" && r.def.task === "instruct" && (
                <Collapsible
                  compact
                  open
                  onOpenChange={(o) => {
                    if (!o) setToolsOpen((s) => ({ ...s, [r.def.id]: false }));
                  }}
                  title="Tool-calling test"
                  subtitle="Asks the model to call a built-in tool and checks the parsed call"
                  className="mt-2.5"
                >
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
                      <label className="flex items-center gap-1">
                        max tokens
                        <input
                          className="w-16 rounded border border-edge bg-surface-2 px-1.5 py-1 text-right text-slate-200"
                          type="number"
                          min={1}
                          step={256}
                          value={toolMaxTokens}
                          onChange={(e) => setToolMaxTokens(Math.max(1, Number(e.target.value) || 1))}
                        />
                      </label>
                      <label className="flex items-center gap-1">
                        <input
                          type="checkbox"
                          checked={toolDisableThinking}
                          onChange={(e) => setToolDisableThinking(e.target.checked)}
                        />
                        disable thinking
                      </label>
                      <Button variant="primary" disabled={busy === r.def.id} onClick={() => runToolTest(r.def)}>
                        Test tool calling
                      </Button>
                    </div>
                    {toolResults[r.def.id] && (
                      <pre className="whitespace-pre-wrap rounded-md border border-edge bg-surface p-2 text-[11px] text-slate-300">{toolResults[r.def.id]}</pre>
                    )}
                  </div>
                </Collapsible>
              )}
              {benchFor === r.def.id && (
                <div className="mt-2.5">
                  <BenchmarkDrawer serverId={r.def.id} model={effectiveModelName(r.def)} onClose={() => setBenchFor(null)} />
                </div>
              )}
              {r.status === "running" && serverSeries[r.def.id] && serverSeries[r.def.id].length > 0 && (
                <div className="mt-2.5 pt-2 border-t border-edge/40">
                  <Sparkline
                    label="Throughput (last 5 min)"
                    data={serverSeries[r.def.id].map((p) => p.tok_s)}
                    height={28}
                    min={0}
                    color="#22d3ee"
                    unit="tok/s"
                    currentValue={r.metrics?.measured?.tokens_per_sec != null ? fmtTokPerSec(r.metrics.measured.tokens_per_sec) : undefined}
                    showMinMax={false}
                  />
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {selectedRow && (
        <div className="space-y-4">
          <Card>
            <CardTitle
              right={
                <div className="flex gap-2">
                  <span className={`${statusColor(selectedRow.status)} text-sm font-medium`}>{selectedRow.status}</span>
                  <Button
                    variant="ghost"
                    onClick={async () => {
                      await navigator.clipboard.writeText(selectedLog);
                      setCopiedId(selectedRow.def.id);
                      setTimeout(() => setCopiedId(null), 1500);
                    }}
                  >
                    {copiedId === selectedRow.def.id ? "Log copied" : "Copy log"}
                  </Button>
                  {selectedRow.status === "running" && selectedRow.def.task === "instruct" && (
                    <>
                      <ChatButton serverId={selectedRow.def.id} port={selectedRow.def.port} model={effectiveModelName(selectedRow.def)} />
                      <BenchmarkButton serverId={selectedRow.def.id} model={effectiveModelName(selectedRow.def)} />
                    </>
                  )}
                </div>
              }
            >
              {cleanModelName(selectedRow.def.name)} · logs
            </CardTitle>
            <div className="h-80 overflow-y-auto rounded-md bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-slate-300">
              {selectedLog ? (
                <>
                  <pre className="whitespace-pre-wrap">
                    {selectedLog.split("\n").map((line, index) => {
                      const trimmed = line.trimStart();
                      const isError = /^E(?:\s|:)/.test(trimmed) || /\b(?:error|failed|unable to allocate)\b/i.test(trimmed);
                      const isWarning = /^W(?:\s|:)/.test(trimmed);
                      const isCors = /cors is set to allow all origins|no api key is set/i.test(line);
                      return (
                        <span
                          key={index}
                          className={isError && !isCors ? "text-red-300" : isCors || isWarning ? "text-amber-300/80" : /^I(?:\s|:)/.test(trimmed) ? "text-slate-500" : "text-slate-300"}
                          title={isCors ? "Informational: this only matters if the localhost port is reachable from another machine." : undefined}
                        >
                          {line}
                          {"\n"}
                        </span>
                      );
                    })}
                  </pre>
                  <div ref={(el) => { if (selected) logEndRefs.current[selected] = el; }} />
                </>
              ) : (
                <span className="text-slate-600">{selectedRow.status === "running" ? "streaming logs…" : "server stopped — start it to see logs."}</span>
              )}
            </div>
          </Card>
          <Card>
            <CardTitle
              right={
                <span className="text-[11px] text-slate-500">
                  {selectedRow.def.metrics ? "live · 5s samples" : "disabled"}
                </span>
              }
            >
              Metrics
            </CardTitle>
            <ServerMetricsPanel
              metrics={selectedRow.metrics}
              series={serverSeries[selectedRow.def.id] ?? []}
              status={selectedRow.status}
              metricsEnabled={selectedRow.def.metrics}
            />
          </Card>
        </div>
      )}

      {compatTestServer && (
        <ServerCompatibilityModal
          server={compatTestServer}
          isOpen={true}
          onClose={() => setCompatTestServer(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function inferVllmQuant(modelId: string): string {
  const id = modelId.toUpperCase();
  if (id.includes("GPTQ")) return "gptq";
  if (id.includes("AWQ")) return "awq";
  if (id.includes("FP8")) return "fp8";
  return "auto";
}

export function cleanModelName(raw: string): string {
  if (!raw) return "";
  const trimmed = raw.trim();
  const withoutPath = trimmed.split(/[\\/]/).pop() || trimmed;
  const stripped = withoutPath.replace(/\.(gguf|bin|safetensors)$/i, "");
  return stripped.trim() || trimmed;
}

function inferGgufQuantLabel(modelPath: string): string {
  const file = modelPath.split(/[\\/]/).pop() || "";
  const isMtp = /(?:[-_.]mtp)(?:[-_.]\d+-of-\d+)?\.gguf$/i.test(file);
  const clean = file.replace(/(?:[-_.]mtp)?(?:[-_.]\d+-of-\d+)?\.gguf$/i, ".gguf");
  const match = clean.match(/(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?=\.gguf$)/i);
  const base = match?.[1]?.toUpperCase() || "GGUF";
  return isMtp && base !== "GGUF" ? `${base} (MTP)` : isMtp ? "GGUF (MTP)" : base;
}

function formatMemoryMb(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value >= 1024 ? `${(value / 1024).toFixed(2)} GiB` : `${Math.round(value)} MiB`;
}

function ContextFitResultPanel({
  result,
  error,
  loading,
  onApply,
}: {
  result: ContextFitReport | null;
  error: string | null;
  loading: boolean;
  onApply: () => void;
}) {
  if (!loading && !error && !result) return null;
  if (loading) {
    return (
      <div className="mt-4 rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-4 text-sm text-cyan-200">
        <Spinner label="Analyzing weights, KV cache, VRAM, and RAM…" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/5 p-4 text-sm text-rose-200">
        Context fit test failed: {error}
      </div>
    );
  }
  if (!result) return null;

  const tone =
    result.status === "fits"
      ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-200"
      : result.status === "fits_with_cpu"
        ? "border-cyan-500/30 bg-cyan-500/5 text-cyan-200"
        : result.status === "tight"
          ? "border-amber-500/30 bg-amber-500/5 text-amber-200"
          : result.status === "does_not_fit"
            ? "border-rose-500/30 bg-rose-500/5 text-rose-200"
            : "border-slate-500/30 bg-slate-500/5 text-slate-300";
  const title =
    result.status === "fits"
      ? "Context fits"
      : result.status === "fits_with_cpu"
        ? "Fits with CPU offload"
        : result.status === "tight"
          ? "Fits, but VRAM is tight"
          : result.status === "does_not_fit"
            ? result.backend === "llamacpp" && result.warnings.some((warning) => warning.includes("--fit"))
              ? "GPU-only placement does not fit"
              : "Context does not fit"
            : "More metadata is required";
  const hasRecommendation =
    result.recommended_context != null ||
    result.recommended_kv_cache_dtype != null ||
    result.recommended_cache_type_k != null;

  return (
    <div className={`mt-4 rounded-xl border p-4 ${tone}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          <div className="mt-1 text-xs opacity-80">
            {fmtNum(result.requested_context)} tokens · {result.kv_cache_label} · {result.weight_source.replaceAll("_", " ")}
          </div>
        </div>
        {hasRecommendation && (
          <Button variant="subtle" className="shrink-0 text-xs" onClick={onApply}>
            Apply recommendation
          </Button>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Weights</div>
          <div className="mt-0.5 font-mono">{formatMemoryMb(result.weight_gib == null ? null : result.weight_gib * 1024)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">KV cache</div>
          <div className="mt-0.5 font-mono">{formatMemoryMb(result.required_kv_mb)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">VRAM headroom</div>
          <div className="mt-0.5 font-mono">{formatMemoryMb(result.vram_headroom_mb)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Max without overflow</div>
          <div className="mt-0.5 font-mono">{result.max_context_no_overflow == null ? "—" : fmtNum(result.max_context_no_overflow)}</div>
          <div className="mt-0.5 text-[10px] opacity-60">GPU only · current free VRAM</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Max with overflow</div>
          <div className="mt-0.5 font-mono">{result.max_context_with_overflow == null ? "—" : fmtNum(result.max_context_with_overflow)}</div>
          <div className="mt-0.5 text-[10px] opacity-60">
            {result.overflow_enabled
              ? `${result.overflow_label} enabled`
              : result.overflow_required_gb
                ? `potential · needs ${result.overflow_required_gb} GB ${result.overflow_label}`
                : `potential · ${result.overflow_label} not enabled`}
          </div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Free VRAM now</div>
          <div className="mt-0.5 font-mono">{formatMemoryMb(result.vram_free_mb)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Available RAM</div>
          <div className="mt-0.5 font-mono">{formatMemoryMb(result.ram_available_mb)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Native context</div>
          <div className="mt-0.5 font-mono">{result.native_context == null ? "unknown" : fmtNum(result.native_context)}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-black/10 p-2">
          <div className="opacity-65">Weight offload needed</div>
          <div className="mt-0.5 font-mono">{result.required_cpu_offload_gb == null ? "—" : `${result.required_cpu_offload_gb} GiB`}</div>
        </div>
      </div>

      {result.fits_with_current_free === false && result.status !== "does_not_fit" && (
        <p className="mt-3 text-xs text-amber-200">
          The no-overflow limit uses current free VRAM. Close GPU-heavy apps to raise it, or enable CPU KV overflow for a larger context.
        </p>
      )}
      <p className="mt-3 text-xs leading-relaxed">{result.recommendation}</p>
      {result.warnings.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[11px] opacity-80">
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </div>
  );
}

function NewServerForm({
  freeGb,
  totalGb,
  devices = [],
  initialModelId = "",
  initialBackend,
  initialModelPath,
  initialQuant,
  initialSwapSpace,
  initialCpuOffload,
  initialMaxLen,
  initialVramContext,
  initialGpuUtil,
  initialServed,
  initialKvDtype,
  initialTask,
  initialChannel,
  initialEnv,
  initialSpeculativeEnabled,
  initialSpeculativeModel,
  initialSpeculativeModelPath,
  initialSpeculativeNumTokens,
  initialServer, // When provided, enables edit mode
  onDone,
  onCancel,
  onErr,
}: {
  freeGb?: number | null;
  totalGb?: number | null;
  devices?: LlamaDevice[];
  initialModelId?: string;
  initialBackend?: "vllm" | "llamacpp";
  initialModelPath?: string;
  initialQuant?: string;
  initialSwapSpace?: number;
  initialCpuOffload?: number;
  initialMaxLen?: number;
  initialVramContext?: number;
  initialGpuUtil?: number;
  initialServed?: string;
  initialKvDtype?: string;
  initialTask?: "instruct" | "embed";
  initialChannel?: string;
  initialEnv?: Record<string, string>;
  initialSpeculativeEnabled?: boolean;
  initialSpeculativeModel?: string;
  initialSpeculativeModelPath?: string;
  initialSpeculativeNumTokens?: number;
  initialServer?: ServerDef | null; // For editing existing server
  onDone: (s: { id: string }) => void;
  onCancel: () => void;
  onErr: (e: string) => void;
}) {
  const isEditing = !!initialServer;
  const isInitialGguf =
    initialBackend === "llamacpp" ||
    initialQuant?.toUpperCase() === "GGUF" ||
    (initialModelId
      ? initialModelId.toUpperCase().includes("GGUF") || initialModelId.toLowerCase().endsWith(".gguf")
      : false) ||
    Boolean(initialModelPath);

  const [modelId, setModelId] = useState(initialModelId);
  const [name, setName] = useState(initialModelId ? cleanModelName(initialModelId) : "");
  const [task, setTask] = useState<"instruct" | "embed">(initialTask || "instruct");
  const [backend, setBackend] = useState<"vllm" | "llamacpp">(
    initialBackend || (isInitialGguf ? "llamacpp" : "vllm")
  );
  const [quant, setQuant] = useState(
    initialQuant || (isInitialGguf ? "GGUF" : inferVllmQuant(initialModelId))
  );
  const [libraryModels, setLibraryModels] = useState<LibraryEntry[]>([]);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [selectedLibraryModel, setSelectedLibraryModel] = useState<string>("");

  useEffect(() => {
    let active = true;
    setLibraryLoading(true);
    api.libraryList()
      .then((entries) => {
        if (active && Array.isArray(entries)) {
          setLibraryModels(entries);
        }
      })
      .catch((err) => {
        console.warn("Failed to load library models:", err);
      })
      .finally(() => {
        if (active) setLibraryLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleSelectLibraryModel = (targetKey: string) => {
    setSelectedLibraryModel(targetKey);
    if (!targetKey) return;

    const entry = libraryModels.find((m) => (m.model_path || m.model_id) === targetKey);
    if (!entry) return;

    const isGguf =
      entry.quant?.toUpperCase() === "GGUF" ||
      entry.model_id.toUpperCase().includes("GGUF") ||
      Boolean(entry.model_path && entry.model_path.toLowerCase().endsWith(".gguf")) ||
      Boolean(entry.is_local && entry.model_path && entry.model_path.toLowerCase().endsWith(".gguf"));

    if (isGguf) {
      setBackend("llamacpp");
      const path = entry.model_path || entry.model_id;
      setModelPath(path);
      setModelId(entry.model_id);
      const inferredQuant = entry.quant || inferGgufQuantLabel(path);
      setQuant(inferredQuant);
      if (inferredQuant.includes("(MTP)") || /(?:[-_.]mtp)(?:[-_.]\d+-of-\d+)?\.gguf$/i.test(path)) {
        setSpeculativeEnabled(true);
        setSpeculativeType("draft-mtp");
        if (!speculativeNumTokens || speculativeNumTokens === "5") {
          setSpeculativeNumTokens("2");
        }
      }
    } else {
      setBackend("vllm");
      setModelId(entry.model_id);
      setModelPath(entry.model_path || "");
      setQuant(entry.quant || inferVllmQuant(entry.model_id));
    }

    if (entry.task === "embed") {
      setTask("embed");
    } else {
      setTask("instruct");
    }

    setName(cleanModelName(entry.model_id));
  };

  const handleBrowseFileExplorer = async () => {
    try {
      const selected = await open({
        title: "Select model file from computer",
        multiple: false,
        filters: [
          { name: "Model Files (*.gguf, *.safetensors, *.bin)", extensions: ["gguf", "safetensors", "bin"] },
          { name: "GGUF Models (*.gguf)", extensions: ["gguf"] },
          { name: "All Files (*.*)", extensions: ["*"] },
        ],
      });
      if (selected && typeof selected === "string") {
        const isGguf = selected.toLowerCase().endsWith(".gguf");
        if (isGguf) {
          setBackend("llamacpp");
          setModelPath(selected);
          setModelId(selected);
          const inferred = inferGgufQuantLabel(selected);
          setQuant(inferred);
          if (inferred.includes("(MTP)") || /(?:[-_.]mtp)(?:[-_.]\d+-of-\d+)?\.gguf$/i.test(selected)) {
            setSpeculativeEnabled(true);
            setSpeculativeType("draft-mtp");
            if (!speculativeNumTokens || speculativeNumTokens === "5") {
              setSpeculativeNumTokens("2");
            }
          }
        } else {
          setModelPath(selected);
          setModelId(selected);
          if (!initialServer) setQuant(inferVllmQuant(selected));
        }
        setName(cleanModelName(selected));
        setSelectedLibraryModel("");
      }
    } catch (err) {
      console.error("Failed to open file dialog", err);
    }
  };
  const [gpuUtil, setGpuUtil] = useState(initialGpuUtil !== undefined ? String(initialGpuUtil) : "0.85");
  const [maxLen, setMaxLen] = useState(initialMaxLen ? String(initialMaxLen) : "");
  const [swapSpaceGb, setSwapSpaceGb] = useState<string>(
    initialSwapSpace !== undefined ? String(initialSwapSpace) : ""
  );
  const [cpuOffloadGb, setCpuOffloadGb] = useState<string>(
    initialCpuOffload !== undefined ? String(initialCpuOffload) : ""
  );
  const [vramContextLimit, setVramContextLimit] = useState<number | undefined>(initialVramContext);
  const [served, setServed] = useState(initialServed || "");
  const [enforceEager, setEnforceEager] = useState(true);
  const [creating, setCreating] = useState(false);
  const [modelPath, setModelPath] = useState(initialModelPath || "");
  const [mmprojPath, setMmprojPath] = useState("");
  const [ctxSize, setCtxSize] = useState("");
  const [nGpuLayers, setNGpuLayers] = useState("");
  const [nCpuMoe, setNCpuMoe] = useState("");
  const [fit, setFit] = useState(true);
  const [fitTarget, setFitTarget] = useState("");
  const [device, setDevice] = useState(
    () => devices.find((candidate) => candidate.backend.toLowerCase() === "cuda")?.id ?? ""
  );
  const [apiKey, setApiKey] = useState("");
  const [apiKeyDirty, setApiKeyDirty] = useState(false);
  const [apiKeyConfigured, setApiKeyConfigured] = useState(false);
  const [logVerbosity, setLogVerbosity] = useState("");
  const [flashAttn, setFlashAttn] = useState(true);
  const [jinja, setJinja] = useState(true);
  const [noKvOffload, setNoKvOffload] = useState(false);
  const [metrics, setMetrics] = useState(true);
  const [threads, setThreads] = useState("");
  const [batchSize, setBatchSize] = useState("");
  const [ubatchSize, setUbatchSize] = useState("");
  const [parallel, setParallel] = useState("0");
  const [extraArgs, setExtraArgs] = useState("");
  const [env, setEnv] = useState<Record<string, string>>(initialEnv ?? {});
  const [llamaCppChannel, setLlamaCppChannel] = useState<string>(initialChannel || "upstream");
  const [llamaCppForks, setLlamaCppForks] = useState<import("../types").LlamaCppFork[]>([
    { id: "upstream", name: "Upstream (ggml-org)", repo: "ggml-org/llama.cpp", asset_regex: "", runtime_regex: "", main_prefix: "llama-", builtin: true },
  ]);
  useEffect(() => {
    api.llamacppForks()
      .then((forks) => { if (Array.isArray(forks) && forks.length) setLlamaCppForks(forks); })
      .catch(() => {});
  }, []);
  const [cacheTypeK, setCacheTypeK] = useState("q8_0");
  const [cacheTypeV, setCacheTypeV] = useState("q8_0");
  const [kvCacheDtype, setKvCacheDtype] = useState(initialKvDtype || "");
  const [speculativeEnabled, setSpeculativeEnabled] = useState(
    initialServer?.speculative_enabled ?? initialSpeculativeEnabled ?? false
  );
  const [speculativeModel, setSpeculativeModel] = useState(
    initialServer?.speculative_model ?? initialSpeculativeModel ?? ""
  );
  const [speculativeModelPath, setSpeculativeModelPath] = useState(
    initialServer?.speculative_model_path ?? initialSpeculativeModelPath ?? ""
  );
  const [speculativeNumTokens, setSpeculativeNumTokens] = useState(
    initialServer?.speculative_num_tokens !== undefined && initialServer?.speculative_num_tokens !== null
      ? String(initialServer.speculative_num_tokens)
      : initialSpeculativeNumTokens !== undefined && initialSpeculativeNumTokens !== null
        ? String(initialSpeculativeNumTokens)
        : "5"
  );
  const [speculativeDraftNgl, setSpeculativeDraftNgl] = useState(
    initialServer?.speculative_draft_ngl ?? ""
  );
  const [speculativeDraftNMin, setSpeculativeDraftNMin] = useState(
    initialServer?.speculative_draft_n_min != null ? String(initialServer.speculative_draft_n_min) : ""
  );
  const [speculativeDraftPMin, setSpeculativeDraftPMin] = useState(
    initialServer?.speculative_draft_p_min != null ? String(initialServer.speculative_draft_p_min) : ""
  );
  const [speculativeDraftPSplit, setSpeculativeDraftPSplit] = useState(
    initialServer?.speculative_draft_p_split != null ? String(initialServer.speculative_draft_p_split) : ""
  );
  const [speculativeType, setSpeculativeType] = useState(
    initialServer?.speculative_type ?? ""
  );
  const [speculativeDraftDevice, setSpeculativeDraftDevice] = useState(
    initialServer?.speculative_draft_device ?? ""
  );
  const [speculativeDraftSampling, setSpeculativeDraftSampling] = useState(
    initialServer?.speculative_draft_sampling ?? ""
  );
  const [speculativeDraftBackendSampling, setSpeculativeDraftBackendSampling] = useState<string>(
    initialServer?.speculative_draft_backend_sampling === true
      ? "true"
      : initialServer?.speculative_draft_backend_sampling === false
      ? "false"
      : ""
  );
  const [speculativeDraftCacheTypeK, setSpeculativeDraftCacheTypeK] = useState(
    initialServer?.speculative_draft_cache_type_k?.toLowerCase().replaceAll("-", "_") ?? ""
  );
  const [speculativeDraftCacheTypeV, setSpeculativeDraftCacheTypeV] = useState(
    initialServer?.speculative_draft_cache_type_v?.toLowerCase().replaceAll("-", "_") ?? ""
  );
  const [speculativeDraftNUbatch, setSpeculativeDraftNUbatch] = useState(
    initialServer?.speculative_draft_n_ubatch != null ? String(initialServer.speculative_draft_n_ubatch) : ""
  );
  const [speculativeDraftThreads, setSpeculativeDraftThreads] = useState(
    initialServer?.speculative_draft_threads != null ? String(initialServer.speculative_draft_threads) : ""
  );
  const [speculativeDraftThreadsBatch, setSpeculativeDraftThreadsBatch] = useState(
    initialServer?.speculative_draft_threads_batch != null ? String(initialServer.speculative_draft_threads_batch) : ""
  );
  const [speculativeDraftCpuMoe, setSpeculativeDraftCpuMoe] = useState<boolean>(
    initialServer?.speculative_draft_cpu_moe ?? false
  );
  const [speculativeDraftNCpuMoe, setSpeculativeDraftNCpuMoe] = useState(
    initialServer?.speculative_draft_n_cpu_moe != null ? String(initialServer.speculative_draft_n_cpu_moe) : ""
  );
  const [speculativeDraftPrio, setSpeculativeDraftPrio] = useState(
    initialServer?.speculative_draft_prio != null ? String(initialServer.speculative_draft_prio) : ""
  );
  const [speculativeDraftPoll, setSpeculativeDraftPoll] = useState(
    initialServer?.speculative_draft_poll != null ? String(initialServer.speculative_draft_poll) : ""
  );
  const [contextFit, setContextFit] = useState<ContextFitReport | null>(null);
  const [contextFitError, setContextFitError] = useState<string | null>(null);
  const [contextFitLoading, setContextFitLoading] = useState(false);

  useEffect(() => {
    if (initialModelId) {
      setModelId(initialModelId);
      setName(cleanModelName(initialModelId));
    }
    if (initialBackend) {
      setBackend(initialBackend);
    } else if (isInitialGguf) {
      setBackend("llamacpp");
    }
    if (initialModelPath) {
      setModelPath(initialModelPath);
    }
    if (initialQuant) {
      setQuant(initialQuant);
    }
    if (
      !initialServer &&
      (initialQuant?.includes("(MTP)") ||
        (initialModelPath && /(?:[-_.]mtp)(?:[-_.]\d+-of-\d+)?\.gguf$/i.test(initialModelPath)))
    ) {
      setSpeculativeEnabled(true);
      setSpeculativeType("draft-mtp");
      if (!initialSpeculativeNumTokens) {
        setSpeculativeNumTokens("2");
      }
    }
    if (initialSwapSpace !== undefined) {
      setSwapSpaceGb(String(initialSwapSpace));
    }
    if (initialCpuOffload !== undefined) {
      setCpuOffloadGb(String(initialCpuOffload));
    }
    if (initialMaxLen !== undefined) {
      setMaxLen(String(initialMaxLen));
    }
    if (initialVramContext !== undefined) {
      setVramContextLimit(initialVramContext);
    }
    if (initialGpuUtil !== undefined) {
      setGpuUtil(String(initialGpuUtil));
    }
    if (initialServed !== undefined) {
      setServed(initialServed);
    }
    if (initialKvDtype !== undefined) {
      setKvCacheDtype(initialKvDtype);
    }
    if (initialTask !== undefined) {
      setTask(initialTask);
    }
    if (initialChannel !== undefined) {
      setLlamaCppChannel(initialChannel);
    }
    if (initialSpeculativeEnabled !== undefined) {
      setSpeculativeEnabled(initialSpeculativeEnabled);
    }
    if (initialSpeculativeModel !== undefined) {
      setSpeculativeModel(initialSpeculativeModel);
    }
    if (initialSpeculativeModelPath !== undefined) {
      setSpeculativeModelPath(initialSpeculativeModelPath);
    }
    if (initialSpeculativeNumTokens !== undefined) {
      setSpeculativeNumTokens(String(initialSpeculativeNumTokens));
    }
    if (initialServer) {
      if (initialServer.name) setName(cleanModelName(initialServer.name));
      setEnforceEager(initialServer.enforce_eager ?? true);
      setLlamaCppChannel(initialServer.llamacpp_channel || "upstream");
      setCacheTypeK(initialServer.cache_type_k || "q8_0");
      setCacheTypeV(initialServer.cache_type_v || "q8_0");
      setKvCacheDtype(initialServer.kv_cache_dtype || "");
      setCtxSize(initialServer.ctx_size ? String(initialServer.ctx_size) : "");
      setNGpuLayers(initialServer.n_gpu_layers != null ? String(initialServer.n_gpu_layers) : "");
      setNCpuMoe(initialServer.n_cpu_moe != null ? String(initialServer.n_cpu_moe) : "");
      setFit(initialServer.fit ?? true);
      setFitTarget(initialServer.fit_target != null ? String(initialServer.fit_target) : "");
      setDevice(initialServer.device || "");
      const initialApiKey = initialServer.api_key || "";
      setApiKey(initialApiKey === SECRET_PLACEHOLDER ? "" : initialApiKey);
      setApiKeyConfigured(!!initialApiKey);
      setApiKeyDirty(false);
      setLogVerbosity(initialServer.log_verbosity != null ? String(initialServer.log_verbosity) : "");
      setFlashAttn(initialServer.flash_attn);
      setJinja(initialServer.jinja);
      setNoKvOffload(initialServer.no_kv_offload);
      setMetrics(initialServer.metrics);
      setThreads(initialServer.threads != null ? String(initialServer.threads) : "");
      setBatchSize(initialServer.batch_size != null ? String(initialServer.batch_size) : "");
      setUbatchSize(initialServer.ubatch_size != null ? String(initialServer.ubatch_size) : "");
      setParallel(String(initialServer.parallel ?? 0));
      setExtraArgs((initialServer.extra_args || []).join("\n"));
      setEnv(initialServer.env ?? {});
      setMmprojPath(initialServer.mmproj_path || "");
      setSpeculativeEnabled(initialServer.speculative_enabled ?? false);
      setSpeculativeModel(initialServer.speculative_model || "");
      setSpeculativeModelPath(initialServer.speculative_model_path || "");
      setSpeculativeNumTokens(
        initialServer.speculative_num_tokens != null
          ? String(initialServer.speculative_num_tokens)
          : "5"
      );
      setSpeculativeDraftNgl(initialServer.speculative_draft_ngl || "");
      setSpeculativeDraftNMin(initialServer.speculative_draft_n_min != null ? String(initialServer.speculative_draft_n_min) : "");
      setSpeculativeDraftPMin(initialServer.speculative_draft_p_min != null ? String(initialServer.speculative_draft_p_min) : "");
      setSpeculativeDraftPSplit(initialServer.speculative_draft_p_split != null ? String(initialServer.speculative_draft_p_split) : "");
      setSpeculativeType(initialServer.speculative_type || "");
      setSpeculativeDraftDevice(initialServer.speculative_draft_device || "");
      setSpeculativeDraftSampling(initialServer.speculative_draft_sampling || "");
      setSpeculativeDraftBackendSampling(
        initialServer.speculative_draft_backend_sampling === true
          ? "true"
          : initialServer.speculative_draft_backend_sampling === false
          ? "false"
          : ""
      );
      setSpeculativeDraftCacheTypeK(initialServer.speculative_draft_cache_type_k?.toLowerCase().replaceAll("-", "_") || "");
      setSpeculativeDraftCacheTypeV(initialServer.speculative_draft_cache_type_v?.toLowerCase().replaceAll("-", "_") || "");
      setSpeculativeDraftNUbatch(initialServer.speculative_draft_n_ubatch != null ? String(initialServer.speculative_draft_n_ubatch) : "");
      setSpeculativeDraftThreads(initialServer.speculative_draft_threads != null ? String(initialServer.speculative_draft_threads) : "");
      setSpeculativeDraftThreadsBatch(initialServer.speculative_draft_threads_batch != null ? String(initialServer.speculative_draft_threads_batch) : "");
      setSpeculativeDraftCpuMoe(initialServer.speculative_draft_cpu_moe ?? false);
      setSpeculativeDraftNCpuMoe(initialServer.speculative_draft_n_cpu_moe != null ? String(initialServer.speculative_draft_n_cpu_moe) : "");
      setSpeculativeDraftPrio(initialServer.speculative_draft_prio != null ? String(initialServer.speculative_draft_prio) : "");
      setSpeculativeDraftPoll(initialServer.speculative_draft_poll != null ? String(initialServer.speculative_draft_poll) : "");
    }
  }, [
    initialModelId,
    initialBackend,
    initialModelPath,
    isInitialGguf,
    initialQuant,
    initialSwapSpace,
    initialCpuOffload,
    initialMaxLen,
    initialVramContext,
    initialGpuUtil,
    initialServed,
    initialKvDtype,
    initialTask,
    initialChannel,
    initialSpeculativeEnabled,
    initialSpeculativeModel,
    initialSpeculativeModelPath,
    initialSpeculativeNumTokens,
    initialServer,
  ]);

  const changeBackend = (nextBackend: "vllm" | "llamacpp") => {
    setBackend(nextBackend);
    setContextFit(null);
    setContextFitError(null);
    setSpeculativeEnabled(false);
    setSpeculativeModel("");
    setSpeculativeModelPath("");
    setSpeculativeNumTokens("5");
    setSpeculativeDraftNgl("");
    setSpeculativeDraftNMin("");
    setSpeculativeDraftPMin("");
    setSpeculativeDraftPSplit("");
    setSpeculativeType("");
    setSpeculativeDraftDevice("");
    setSpeculativeDraftSampling("");
    setSpeculativeDraftBackendSampling("");
    setSpeculativeDraftCacheTypeK("");
    setSpeculativeDraftCacheTypeV("");
    setSpeculativeDraftNUbatch("");
    setSpeculativeDraftThreads("");
    setSpeculativeDraftThreadsBatch("");
    setSpeculativeDraftCpuMoe(false);
    setSpeculativeDraftNCpuMoe("");
    setSpeculativeDraftPrio("");
    setSpeculativeDraftPoll("");
    if (nextBackend === "vllm") {
      if (/[\\/]|\.gguf$/i.test(modelId)) setModelId("");
      setModelPath("");
      setMmprojPath("");
      setCtxSize("");
      setNGpuLayers("");
      setNCpuMoe("");
      setFit(false);
      setFitTarget("");
      setDevice("");
      setLogVerbosity("");
      setFlashAttn(false);
      setCacheTypeK("");
      setCacheTypeV("");
      setKvCacheDtype("");
      setThreads("");
      setBatchSize("");
      setUbatchSize("");
      setParallel("0");
      setNoKvOffload(false);
      setJinja(false);
      setExtraArgs("");
      setLlamaCppChannel("upstream");
      setQuant(inferVllmQuant(modelId));
    } else {
      setGpuUtil("0.85");
      setMaxLen("");
      setSwapSpaceGb("");
      setCpuOffloadGb("");
      setEnforceEager(false);
      setEnv({});
      setModelId("");
      setTask("instruct");
      setQuant("GGUF");
      setFit(true);
      setFlashAttn(true);
      setJinja(true);
      setCacheTypeK("q8_0");
      setCacheTypeV("q8_0");
    }
  };

  const analyzeContext = async () => {
    const chosenModelId = modelId.trim();
    const chosenModelPath = modelPath.trim();
    if (backend === "vllm" && !chosenModelId) {
      setContextFitError("Enter a model ID before testing context fit.");
      return;
    }
    if (backend === "llamacpp" && !chosenModelPath) {
      setContextFitError("Select or enter a GGUF file before testing context fit.");
      return;
    }
    setContextFitLoading(true);
    setContextFitError(null);
    setContextFit(null);
    try {
      const result = await api.analyzeContextFit({
        backend,
        model_id: backend === "llamacpp" ? (chosenModelPath || chosenModelId) : chosenModelId,
        model_path: backend === "llamacpp" ? (chosenModelPath || chosenModelId) : null,
        context_tokens:
          backend === "vllm"
            ? maxLen
              ? parseInt(maxLen, 10)
              : null
            : ctxSize
              ? parseInt(ctxSize, 10)
              : null,
        quant: backend === "vllm" ? quant : null,
        kv_cache_dtype: backend === "vllm" ? (kvCacheDtype || null) : null,
        gpu_mem_util: backend === "vllm" ? parseFloat(gpuUtil) : null,
        cpu_offload_gb:
          backend === "vllm" && cpuOffloadGb ? parseInt(cpuOffloadGb, 10) : 0,
        kv_offload_gb:
          backend === "vllm" && swapSpaceGb ? parseInt(swapSpaceGb, 10) : 0,
        cache_type_k: backend === "llamacpp" ? cacheTypeK : null,
        cache_type_v: backend === "llamacpp" ? cacheTypeV : null,
        flash_attn: backend === "llamacpp" ? flashAttn : null,
        n_gpu_layers:
          backend === "llamacpp" && nGpuLayers ? parseInt(nGpuLayers, 10) : null,
        n_cpu_moe:
          backend === "llamacpp" && nCpuMoe ? parseInt(nCpuMoe, 10) : null,
        fit: backend === "llamacpp" ? fit : null,
        fit_target:
          backend === "llamacpp" && fitTarget ? parseInt(fitTarget, 10) : 1024,
        no_kv_offload: backend === "llamacpp" ? noKvOffload : null,
        speculative_enabled: task !== "embed" ? speculativeEnabled : false,
        speculative_model:
          backend === "vllm"
            ? (speculativeModel.trim() || null)
            : (speculativeModelPath.trim() || speculativeModel.trim() || null),
        speculative_model_path:
          backend === "llamacpp"
            ? (speculativeModelPath.trim() || speculativeModel.trim() || null)
            : null,
        speculative_num_tokens:
          speculativeEnabled && speculativeNumTokens ? parseInt(speculativeNumTokens, 10) : 5,
        speculative_draft_ngl: backend === "llamacpp" && speculativeEnabled && speculativeDraftNgl.trim() ? speculativeDraftNgl.trim() : null,
        speculative_draft_n_min: backend === "llamacpp" && speculativeEnabled && speculativeDraftNMin.trim() ? parseInt(speculativeDraftNMin, 10) : null,
        speculative_draft_p_min: backend === "llamacpp" && speculativeEnabled && speculativeDraftPMin.trim() ? parseFloat(speculativeDraftPMin) : null,
        speculative_draft_p_split: backend === "llamacpp" && speculativeEnabled && speculativeDraftPSplit.trim() ? parseFloat(speculativeDraftPSplit) : null,
        speculative_type: backend === "llamacpp" && speculativeEnabled && speculativeType.trim() ? speculativeType.trim() : null,
        speculative_draft_device: backend === "llamacpp" && speculativeEnabled && speculativeDraftDevice.trim() ? speculativeDraftDevice.trim() : null,
        speculative_draft_sampling: backend === "llamacpp" && speculativeEnabled && speculativeDraftSampling.trim() ? speculativeDraftSampling.trim() : null,
        speculative_draft_backend_sampling:
          backend === "llamacpp" && speculativeEnabled
            ? speculativeDraftBackendSampling === "true"
              ? true
              : speculativeDraftBackendSampling === "false"
              ? false
              : null
            : null,
        speculative_draft_cache_type_k: backend === "llamacpp" && speculativeEnabled && speculativeDraftCacheTypeK.trim() ? speculativeDraftCacheTypeK.trim().toLowerCase().replaceAll("-", "_") : null,
        speculative_draft_cache_type_v: backend === "llamacpp" && speculativeEnabled && speculativeDraftCacheTypeV.trim() ? speculativeDraftCacheTypeV.trim().toLowerCase().replaceAll("-", "_") : null,
        speculative_draft_n_ubatch: backend === "llamacpp" && speculativeEnabled && speculativeDraftNUbatch.trim() ? parseInt(speculativeDraftNUbatch, 10) : null,
        speculative_draft_threads: backend === "llamacpp" && speculativeEnabled && speculativeDraftThreads.trim() ? parseInt(speculativeDraftThreads, 10) : null,
        speculative_draft_threads_batch: backend === "llamacpp" && speculativeEnabled && speculativeDraftThreadsBatch.trim() ? parseInt(speculativeDraftThreadsBatch, 10) : null,
        speculative_draft_cpu_moe: backend === "llamacpp" && speculativeEnabled ? speculativeDraftCpuMoe : null,
        speculative_draft_n_cpu_moe: backend === "llamacpp" && speculativeEnabled && speculativeDraftNCpuMoe.trim() ? parseInt(speculativeDraftNCpuMoe, 10) : null,
        speculative_draft_prio: backend === "llamacpp" && speculativeEnabled && speculativeDraftPrio.trim() ? parseInt(speculativeDraftPrio, 10) : null,
        speculative_draft_poll: backend === "llamacpp" && speculativeEnabled && speculativeDraftPoll.trim() ? parseInt(speculativeDraftPoll, 10) : null,
      });
      setContextFit(result);
    } catch (error) {
      setContextFitError(String(error));
    } finally {
      setContextFitLoading(false);
    }
  };

  const applyContextRecommendation = () => {
    if (!contextFit) return;
    if (contextFit.recommended_context && contextFit.recommended_context > 0) {
      if (backend === "vllm") {
        setMaxLen(String(contextFit.recommended_context));
      } else {
        setCtxSize(String(contextFit.recommended_context));
      }
    }
    if (contextFit.recommended_kv_cache_dtype) {
      setKvCacheDtype(contextFit.recommended_kv_cache_dtype);
    }
    if (contextFit.recommended_cache_type_k) {
      setCacheTypeK(contextFit.recommended_cache_type_k);
    }
    if (contextFit.recommended_cache_type_v) {
      setCacheTypeV(contextFit.recommended_cache_type_v);
    }
  };

  const submit = async () => {
    const chosenModelId = modelId.trim();
    const chosenModelPath = modelPath.trim();
    if (backend === "llamacpp") {
      if (!chosenModelPath && !chosenModelId.toLowerCase().endsWith(".gguf")) {
        onErr("Please select or provide a path to a .gguf model file for llama.cpp.");
        return;
      }
    } else {
      if (!chosenModelId) return;
    }
    setCreating(true);
    try {
      const parsedSwap = swapSpaceGb !== "" ? parseInt(swapSpaceGb, 10) : null;
      const parsedOffload = cpuOffloadGb !== "" ? parseInt(cpuOffloadGb, 10) : null;
      const parsedSpecTokens = parseInt(speculativeNumTokens, 10);
      const isMtpSpec = speculativeType.trim().toLowerCase().includes("mtp");
      const isNgramSpec = speculativeType.trim().toLowerCase().includes("ngram");
      const defaultTokens = isMtpSpec ? 2 : 5;
      const validSpecTokens = !isNaN(parsedSpecTokens) && parsedSpecTokens > 0 ? parsedSpecTokens : defaultTokens;
      const isSpecActive = speculativeEnabled && task !== "embed";
      const specDraftVllm = isSpecActive && speculativeModel.trim() ? speculativeModel.trim() : null;
      const specDraftLlama = isSpecActive && (speculativeModelPath.trim() || speculativeModel.trim())
        ? (speculativeModelPath.trim() || speculativeModel.trim())
        : null;
      const isLlamaSpecActive = isSpecActive && (Boolean(specDraftLlama) || isNgramSpec || isMtpSpec);

      const parsedDraftNMin = speculativeDraftNMin.trim() ? parseInt(speculativeDraftNMin, 10) : null;
      const parsedDraftPMin = speculativeDraftPMin.trim() ? parseFloat(speculativeDraftPMin) : null;
      const parsedDraftPSplit = speculativeDraftPSplit.trim() ? parseFloat(speculativeDraftPSplit) : null;
      const parsedDraftNUbatch = speculativeDraftNUbatch.trim() ? parseInt(speculativeDraftNUbatch, 10) : null;
      const parsedDraftThreads = speculativeDraftThreads.trim() ? parseInt(speculativeDraftThreads, 10) : null;
      const parsedDraftThreadsBatch = speculativeDraftThreadsBatch.trim() ? parseInt(speculativeDraftThreadsBatch, 10) : null;
      const parsedDraftNCpuMoe = speculativeDraftNCpuMoe.trim() ? parseInt(speculativeDraftNCpuMoe, 10) : null;
      const parsedDraftPrio = speculativeDraftPrio.trim() ? parseInt(speculativeDraftPrio, 10) : null;
      const parsedDraftPoll = speculativeDraftPoll.trim() ? parseInt(speculativeDraftPoll, 10) : null;
      const draftBackendSamplingBool =
        speculativeDraftBackendSampling === "true"
          ? true
          : speculativeDraftBackendSampling === "false"
          ? false
          : null;

      const serverData = {
        backend,
        model_id: backend === "llamacpp" ? (chosenModelPath || chosenModelId) : chosenModelId,
        name:
          cleanModelName(
            name.trim() ||
            (chosenModelId || chosenModelPath).split(/[\\/]/).pop()?.replace(/\.(gguf|bin|safetensors)$/i, "") ||
            "server"
          ),
        task: backend === "vllm" ? task : "instruct",
        quant,
        gpu_mem_util: backend === "vllm" ? parseFloat(gpuUtil) : undefined,
        max_model_len: backend === "vllm" && maxLen ? parseInt(maxLen, 10) : null,
        swap_space_gb:
          backend === "vllm" && parsedSwap !== null && !isNaN(parsedSwap) ? parsedSwap : null,
        cpu_offload_gb:
          backend === "vllm" && parsedOffload !== null && !isNaN(parsedOffload) ? parsedOffload : null,
        enforce_eager: backend === "vllm" ? enforceEager : undefined,
        served_model_name: served.trim() || null,
        kv_cache_dtype: backend === "vllm" ? (kvCacheDtype || null) : null,
        ...(apiKeyDirty || !isEditing
          ? { api_key: apiKey.trim() || null }
          : {}),
        ...(isEditing && apiKeyDirty && !apiKey.trim() ? { clear_api_key: true } : {}),
        metrics,
        model_path: backend === "llamacpp" ? (chosenModelPath || chosenModelId) : null,
        mmproj_path: backend === "llamacpp" ? (mmprojPath.trim() || null) : null,
        ctx_size: backend === "llamacpp" && ctxSize ? parseInt(ctxSize, 10) : null,
        n_gpu_layers:
          backend === "llamacpp" && nGpuLayers.trim() ? parseInt(nGpuLayers, 10) : null,
        n_cpu_moe: backend === "llamacpp" && nCpuMoe.trim() ? parseInt(nCpuMoe, 10) : null,
        fit: backend === "llamacpp" ? fit : undefined,
        fit_target: backend === "llamacpp" && fitTarget.trim() ? parseInt(fitTarget, 10) : null,
        device: backend === "llamacpp" ? (device.trim() || null) : null,
        log_verbosity:
          backend === "llamacpp" && logVerbosity.trim() ? parseInt(logVerbosity, 10) : null,
        flash_attn: backend === "llamacpp" ? flashAttn : undefined,
        jinja: backend === "llamacpp" ? jinja : undefined,
        no_kv_offload: backend === "llamacpp" ? noKvOffload : undefined,
        llamacpp_channel: backend === "llamacpp" ? llamaCppChannel : undefined,
        cache_type_k: backend === "llamacpp" ? cacheTypeK : null,
        cache_type_v: backend === "llamacpp" ? cacheTypeV : null,
        threads: backend === "llamacpp" && threads.trim() ? parseInt(threads, 10) : null,
        batch_size:
          backend === "llamacpp" && batchSize.trim() ? parseInt(batchSize, 10) : null,
        ubatch_size:
          backend === "llamacpp" && ubatchSize.trim() ? parseInt(ubatchSize, 10) : null,
        parallel: backend === "llamacpp" ? parseInt(parallel, 10) || 0 : 0,
        extra_args:
          backend === "llamacpp"
            ? extraArgs
                .split(/\r?\n/)
                .map((arg) => arg.trim())
                .filter(Boolean)
            : null,
        env: backend === "vllm" ? env : {},
        speculative_enabled: backend === "vllm" ? Boolean(specDraftVllm) : isLlamaSpecActive,
        speculative_model: backend === "vllm" ? specDraftVllm : specDraftLlama,
        speculative_model_path: backend === "llamacpp" ? specDraftLlama : null,
        speculative_num_tokens:
          (backend === "vllm" ? Boolean(specDraftVllm) : isLlamaSpecActive)
            ? validSpecTokens
            : null,
        speculative_draft_ngl:
          backend === "llamacpp" && isLlamaSpecActive && speculativeDraftNgl.trim()
            ? speculativeDraftNgl.trim()
            : null,
        speculative_draft_n_min:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftNMin !== null && !isNaN(parsedDraftNMin)
            ? parsedDraftNMin
            : null,
        speculative_draft_p_min:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftPMin !== null && !isNaN(parsedDraftPMin)
            ? parsedDraftPMin
            : null,
        speculative_draft_p_split:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftPSplit !== null && !isNaN(parsedDraftPSplit)
            ? parsedDraftPSplit
            : null,
        speculative_type:
          backend === "llamacpp" && isLlamaSpecActive && speculativeType.trim()
            ? speculativeType.trim()
            : null,
        speculative_draft_device:
          backend === "llamacpp" && isLlamaSpecActive && speculativeDraftDevice.trim()
            ? speculativeDraftDevice.trim()
            : null,
        speculative_draft_sampling:
          backend === "llamacpp" && isLlamaSpecActive && speculativeDraftSampling.trim()
            ? speculativeDraftSampling.trim()
            : null,
        speculative_draft_backend_sampling:
          backend === "llamacpp" && isLlamaSpecActive ? draftBackendSamplingBool : null,
        speculative_draft_cache_type_k:
          backend === "llamacpp" && isLlamaSpecActive && speculativeDraftCacheTypeK.trim()
            ? speculativeDraftCacheTypeK.trim().toLowerCase().replaceAll("-", "_")
            : null,
        speculative_draft_cache_type_v:
          backend === "llamacpp" && isLlamaSpecActive && speculativeDraftCacheTypeV.trim()
            ? speculativeDraftCacheTypeV.trim().toLowerCase().replaceAll("-", "_")
            : null,
        speculative_draft_n_ubatch:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftNUbatch !== null && !isNaN(parsedDraftNUbatch)
            ? parsedDraftNUbatch
            : null,
        speculative_draft_threads:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftThreads !== null && !isNaN(parsedDraftThreads)
            ? parsedDraftThreads
            : null,
        speculative_draft_threads_batch:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftThreadsBatch !== null && !isNaN(parsedDraftThreadsBatch)
            ? parsedDraftThreadsBatch
            : null,
        speculative_draft_cpu_moe:
          backend === "llamacpp" && isLlamaSpecActive ? speculativeDraftCpuMoe : null,
        speculative_draft_n_cpu_moe:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftNCpuMoe !== null && !isNaN(parsedDraftNCpuMoe)
            ? parsedDraftNCpuMoe
            : null,
        speculative_draft_prio:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftPrio !== null && !isNaN(parsedDraftPrio)
            ? parsedDraftPrio
            : null,
        speculative_draft_poll:
          backend === "llamacpp" && isLlamaSpecActive && parsedDraftPoll !== null && !isNaN(parsedDraftPoll)
            ? parsedDraftPoll
            : null,
      };
      let s;
      if (isEditing && initialServer) {
        s = await api.serversUpdate(initialServer.id, serverData);
      } else {
        s = await api.serversCreate(serverData);
      }
      onDone(s);
    } catch (e) {
      onErr(String(e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <Card className="border-accent/30">
      <CardTitle>{isEditing ? "Edit server" : "Define new server"}</CardTitle>
      {freeGb !== null && freeGb !== undefined && (() => {
        const est = estimateServerWeights(
          {
            model_id: modelId,
            model_path: modelPath,
            params_b: initialServer?.params_b ?? null,
            quant,
            backend,
            speculative_enabled: speculativeEnabled,
            speculative_model: speculativeModel,
            speculative_model_path: speculativeModelPath,
          },
          libraryModels
        );
        // Compute effective available VRAM (accounting for currently editing server restarting)
        const currentServerWeight = initialServer ? estimateServerWeights(initialServer).totalWeightGb : 0;
        const effectiveFreeGb = (freeGb ?? 0) + (isEditing ? currentServerWeight : 0);
        const totalVramGb = totalGb ?? 0;
        const overheadGb = 1.0;
        const usableTotalGb = Math.max(0, totalVramGb - overheadGb);

        const cpuOffload = cpuOffloadGb ? parseFloat(cpuOffloadGb) || 0 : 0;
        const isLlamaCpu = backend === "llamacpp" && nGpuLayers === "0";
        const explicitRamWeightGb = isLlamaCpu ? est.totalWeightGb : cpuOffload;
        const reqGpuWeightGb = Math.max(0, est.totalWeightGb - explicitRamWeightGb);

        const capacitySpillGb = totalVramGb > 0 ? Math.max(0, reqGpuWeightGb - usableTotalGb) : 0;
        const headroomShortfallGb = Math.max(0, reqGpuWeightGb - effectiveFreeGb);
        const loadedRamGb = explicitRamWeightGb + (backend === "llamacpp" ? Math.max(capacitySpillGb, headroomShortfallGb) : capacitySpillGb);
        const isLoadedIntoRam = est.totalWeightGb > 0 && loadedRamGb > 0.01;
        const willOomVllm = backend === "vllm" && cpuOffload === 0 && headroomShortfallGb > 0.01;
        const isLowHeadroom = (freeGb ?? 0) < 1.0;

        const shouldWarn = isLowHeadroom || isLoadedIntoRam || willOomVllm;
        if (!shouldWarn) return null;

        return (
          <div className="mb-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-200">
            ⚠️ <strong>Low VRAM Warning:</strong>{" "}
            {isLoadedIntoRam ? (
              <>
                Model weights ({est.totalWeightGb.toFixed(2)} GB
                {speculativeEnabled && est.specWeightGb > 0
                  ? ` [model: ${est.mainWeightGb.toFixed(2)} GB, speculative draft: ${est.specWeightGb.toFixed(2)} GB]`
                  : ""}
                ) will have <strong>~{loadedRamGb.toFixed(2)} GB loaded into system RAM</strong>{" "}
                {explicitRamWeightGb > 0 ? "(via configured CPU offload)" : "(exceeds GPU VRAM capacity)"}.{" "}
                (Counts model weights and speculative decoding weights; context is excluded). Running weights from system RAM will reduce generation speed.
              </>
            ) : willOomVllm ? (
              <>
                Model weights ({est.totalWeightGb.toFixed(2)} GB) exceed currently available GPU VRAM headroom ({effectiveFreeGb.toFixed(2)} GB available).{" "}
                vLLM does not automatically offload weights to RAM without CPU Offload; starting this server may cause a CUDA out-of-memory error. Consider enabling CPU Weight Offload.
              </>
            ) : (
              <>
                Current free VRAM is only {freeGb.toFixed(2)} GB. Configured model weights ({est.totalWeightGb > 0 ? `${est.totalWeightGb.toFixed(2)} GB` : "estimated"}) fit in VRAM (0 GB loaded into system RAM), but available headroom is low.
              </>
            )}
          </div>
        );
      })()}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="sm:col-span-2 rounded-lg border border-edge/60 bg-surface/50 p-3 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-1">
            <span className="text-xs font-semibold text-slate-200">Model Selection</span>
            <span className="text-[11px] text-slate-400">Choose from library or browse computer for external model</span>
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <div className="flex-1 min-w-0">
              <select
                className={inputCls}
                value={selectedLibraryModel}
                onChange={(e) => handleSelectLibraryModel(e.target.value)}
              >
                <option value="">
                  {libraryLoading
                    ? "Loading models from library…"
                    : libraryModels.length === 0
                    ? "No models found in library"
                    : "Select a model from Library…"}
                </option>
                {libraryModels.map((entry) => {
                  const key = entry.model_path || entry.model_id;
                  const sizeGb = (entry.size_mb / 1024).toFixed(1);
                  const isGguf =
                    entry.quant?.toUpperCase() === "GGUF" ||
                    entry.model_id.toUpperCase().includes("GGUF") ||
                    Boolean(entry.model_path && entry.model_path.toLowerCase().endsWith(".gguf")) ||
                    Boolean(entry.is_local && entry.model_path);
                  return (
                    <option key={key} value={key}>
                      [{isGguf ? "GGUF" : "HF"}] {entry.model_id} ({sizeGb} GB{entry.quant && entry.quant !== "GGUF" ? ` · ${entry.quant}` : ""}{entry.is_local ? " · local" : ""})
                    </option>
                  );
                })}
              </select>
            </div>
            <Button
              variant="subtle"
              className="shrink-0 flex items-center gap-1.5"
              onClick={handleBrowseFileExplorer}
              title="Open File Explorer window to choose a model not in the library"
            >
              <span>📂</span>
              <span>Browse File Explorer…</span>
            </Button>
          </div>
        </div>

        <Field label="Backend" help="vLLM (WSL2) for Hugging Face checkpoints; llama.cpp (Windows) for GGUF models. Backends use RAM and VRAM differently — vLLM manages GPU memory automatically, llama.cpp exposes the offload flags below.">
          <select className={inputCls} value={backend} onChange={(e) => changeBackend(e.target.value as "vllm" | "llamacpp")}>
            <option value="vllm">vLLM (WSL2)</option>
            <option value="llamacpp">llama.cpp (Windows)</option>
          </select>
        </Field>
        <Field label={backend === "llamacpp" ? "GGUF model path" : "Model id (HF)"} help={backend === "llamacpp" ? "Path to the GGUF weights file. Model size in GB decides how much VRAM/RAM the server needs — bigger quants and more layers use more VRAM." : "Hugging Face repo id (or local path). Quant/precision choice is the biggest driver of VRAM and system RAM use."}>
          {backend === "llamacpp" ? (
            <input
              className={inputCls}
              placeholder="C:\models\model-Q4_K_M.gguf"
              value={modelPath}
              onChange={(e) => {
                const nextPath = e.target.value;
                setModelPath(nextPath);
                const inferred = inferGgufQuantLabel(nextPath);
                setQuant(inferred);
                if (inferred.includes("(MTP)") || /(?:[-_.]mtp)(?:[-_.]\d+-of-\d+)?\.gguf$/i.test(nextPath)) {
                  setSpeculativeEnabled(true);
                  setSpeculativeType("draft-mtp");
                  if (!speculativeNumTokens || speculativeNumTokens === "5") {
                    setSpeculativeNumTokens("2");
                  }
                }
                if (!name) {
                  setName(cleanModelName(nextPath));
                }
              }}
            />
          ) : (
            <input
              className={inputCls}
              placeholder="Qwen/Qwen2.5-0.5B-Instruct"
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value);
                if (!initialServer) setQuant(inferVllmQuant(e.target.value));
                if (!name) {
                  setName(cleanModelName(e.target.value));
                }
              }}
            />
          )}
        </Field>
        <Field label="Name (optional)" help="Display name for this server. Defaults to the model name (no paths or extensions).">
          <input className={inputCls} placeholder="coder-0.5b" value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        {backend === "vllm" && <Field label="Task" help="Instruct serves chat/completions; embed serves the embeddings endpoint. No effect on model speed or RAM — it only changes which routes are exposed.">
          <select className={inputCls} value={task} onChange={(e) => setTask(e.target.value as "instruct" | "embed")}>
            <option value="instruct">instruct (chat)</option>
            <option value="embed">embed (embeddings)</option>
          </select>
        </Field>}
      </div>

      <Collapsible
        title="Context & memory"
        subtitle="Context length, KV cache precision, and swap / offload"
        storageKey="llm_panel_server_form_context"
        className="mt-3"
        defaultOpen={isEditing || initialMaxLen !== undefined || initialSwapSpace !== undefined || initialCpuOffload !== undefined}
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {backend === "vllm" && (
            <>
              <Field
                label="Max model len (blank = auto)"
                help="Context length ceiling. Longer contexts use much more VRAM for the KV cache (and spill to RAM swap once VRAM is full) — the #1 driver of total memory use. Blank lets vLLM pick a size that fits your VRAM."
                hint={
                  (vramContextLimit ?? 0) > 0
                    ? `Pure VRAM: ≤ ${fmtNum(vramContextLimit!)} tokens | RAM Swap: > ${fmtNum(vramContextLimit!)} tokens`
                    : "Context ceiling in tokens. If exceeding VRAM, RAM swap space will be used."
                }
              >
                <input
                  className={inputCls}
                  placeholder={
                    (vramContextLimit ?? 0) > 0
                      ? initialMaxLen && initialMaxLen > vramContextLimit!
                        ? `auto (~${fmtNum(vramContextLimit!)} VRAM ➔ ~${fmtNum(initialMaxLen)} RAM)`
                        : `auto (~${fmtNum(vramContextLimit!)} VRAM)`
                      : "auto (context ∩ VRAM fit)"
                  }
                  value={maxLen}
                  onChange={(e) => setMaxLen(e.target.value)}
                />
                {(vramContextLimit ?? 0) > 0 && (
                  <div className="mt-1.5 flex items-center gap-2 text-[11px]">
                    <span className="inline-flex items-center gap-1 text-emerald-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      VRAM: ≤{fmtNum(vramContextLimit!)}
                    </span>
                    <span className="text-slate-600">➔</span>
                    <span className="inline-flex items-center gap-1 text-cyan-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                      RAM Swap: {initialMaxLen && initialMaxLen > vramContextLimit! ? `up to ~${fmtNum(initialMaxLen)} tokens` : `>${fmtNum(vramContextLimit!)}`
                      }
                    </span>
                    {maxLen && parseInt(maxLen, 10) > vramContextLimit! && (
                      <span className="text-[11px] font-semibold text-cyan-300 ml-auto">(Uses RAM swap)</span>
                    )}
                  </div>
                )}
                {(vramContextLimit ?? 0) === 0 && (initialMaxLen ?? 0) > 0 && (
                  <div className="mt-1.5 flex items-center gap-2 text-[11px]">
                    <span className="inline-flex items-center gap-1 text-amber-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                      RAM Context: up to ~{fmtNum(initialMaxLen!)} tokens (CPU Offload)
                    </span>
                  </div>
                )}
              </Field>
              <Field label="KV cache precision" help="KV-cache dtype (blank = global setting). FP8 roughly halves KV-cache VRAM so much longer contexts fit; 16-bit is highest quality but uses the most VRAM." hint="Blank inherits the global vLLM setting. FP8 reduces KV-cache VRAM; verify quality on long-context tasks.">
                <select className={inputCls} value={kvCacheDtype} onChange={(e) => setKvCacheDtype(e.target.value)}>
                  <option value="">Global / Auto</option>
                  <option value="float16">FP16</option>
                  <option value="bfloat16">BF16</option>
                  <option value="fp8_e4m3">FP8 E4M3</option>
                  <option value="fp8_e5m2">FP8 E5M2</option>
                </select>
              </Field>
              <Field label="KV Cache CPU Offload (GB)" help="How much KV cache (GB) may sit in CPU RAM instead of VRAM. Lets longer contexts fit than VRAM allows, but CPU-resident KV cache is much slower. 0 = normal GPU-only caching." hint="Advanced vLLM V1 option (--kv-offloading-size). Leave at 0 for normal GPU KV caching.">
                <input type="number" step="1" min="0" className={inputCls} placeholder="0" value={swapSpaceGb} onChange={(e) => setSwapSpaceGb(e.target.value)} />
              </Field>
              <Field label="CPU Weight Offload (GB)" help="How many GB of model weights may be offloaded to CPU RAM. Required only when the model does not fit in VRAM; CPU inference is far slower than GPU. 0 = GPU only." hint="vLLM --cpu-offload-gb. Offloads model parameters to CPU RAM; use only when GPU-only placement cannot fit.">
                <input type="number" step="1" min="0" className={inputCls} placeholder="0 (e.g. 8)" value={cpuOffloadGb} onChange={(e) => setCpuOffloadGb(e.target.value)} />
              </Field>
            </>
          )}
          {backend === "llamacpp" && (
            <>
              <Field label="Context size" help="llama.cpp context length (-c). Longer contexts use much more VRAM for the KV cache — the main driver of total GPU memory use. Blank uses llama.cpp's default.">
                <input className={inputCls} value={ctxSize} onChange={(e) => setCtxSize(e.target.value)} />
              </Field>
              <Field label="Fit target (MiB)" help="Extra VRAM headroom per device kept aside when automatic memory fit is on (--fit-target). Higher is safer against out-of-memory but leaves less VRAM for the model." hint="Memory margin per device when --fit is supported.">
                <input className={inputCls} placeholder="1024" value={fitTarget} onChange={(e) => setFitTarget(e.target.value)} />
              </Field>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" checked={fit} onChange={(e) => setFit(e.target.checked)} /> Automatic memory fit (when supported) <HelpBubble text="Lets llama.cpp size GPU layers and context to fit available VRAM automatically. Strongly recommended — prevents out-of-memory crashes; disabling it requires manual layer tuning." />
              </label>
              <Field label="KV Cache Type K (--cache-type-k)" help="Data type for the K cache. Lower precision (q8_0/q4_0) saves VRAM, allowing longer contexts; f16 is highest quality but uses the most VRAM." hint="Data type for the K cache. Lower precision saves VRAM at some quality cost.">
                <select className={inputCls} value={cacheTypeK} onChange={(e) => setCacheTypeK(e.target.value)}>
                  <option value="f16">f16 (default, highest quality)</option>
                  <option value="bf16">bf16</option>
                  <option value="q8_0">q8_0 (recommended)</option>
                  <option value="q5_1">q5_1</option>
                  <option value="q5_0">q5_0</option>
                  <option value="q4_1">q4_1</option>
                  <option value="q4_0">q4_0</option>
                  <option value="iq4_nl">iq4_nl</option>
                  <option value="f32">f32</option>
                </select>
              </Field>
              <Field label="KV Cache Type V (--cache-type-v)" help="Data type for the V cache. Lower precision (q8_0/q4_0) saves the most VRAM and lets longer contexts fit; f16 is highest quality but uses the most VRAM." hint="Data type for the V cache. Can differ from K; q4_0/q5_0 saves the most VRAM.">
                <select className={inputCls} value={cacheTypeV} onChange={(e) => setCacheTypeV(e.target.value)}>
                  <option value="f16">f16 (default, highest quality)</option>
                  <option value="bf16">bf16</option>
                  <option value="q8_0">q8_0 (recommended)</option>
                  <option value="q5_1">q5_1</option>
                  <option value="q5_0">q5_0</option>
                  <option value="q4_1">q4_1</option>
                  <option value="q4_0">q4_0</option>
                  <option value="iq4_nl">iq4_nl</option>
                  <option value="f32">f32</option>
                </select>
              </Field>
            </>
          )}
        </div>
      </Collapsible>

      <Collapsible
        title={backend === "vllm" ? "Advanced vLLM options" : "Advanced llama.cpp options"}
        subtitle="API naming, startup tuning, environment, and extra arguments"
        storageKey="llm_panel_server_form_advanced"
        className="mt-3"
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field label="API model name / alias" help="Public model name clients use to reach this server. No effect on model speed or RAM." hint="Blank uses the model ID or GGUF filename. vLLM maps this to --served-model-name; llama.cpp maps it to --alias.">
            <input className={inputCls} placeholder="blank = model identity" value={served} onChange={(e) => setServed(e.target.value)} />
          </Field>
          <Field label="Per-server API key (optional)" help="Overrides the global key for this server only. No effect on model speed or RAM." hint="Overrides the global API key for this server (applies to both vLLM and llama.cpp).">
            <div className="flex gap-1.5">
              <input
                type="password"
                className={inputCls}
                value={apiKey}
                placeholder={apiKeyConfigured ? "A key is stored; enter a replacement" : "Optional"}
                onChange={(e) => { setApiKey(e.target.value); setApiKeyDirty(true); }}
              />
              <Button variant="ghost" className="shrink-0 px-2 text-xs" onClick={() => { setApiKey(crypto.randomUUID().replaceAll("-", "")); setApiKeyDirty(true); }}>Generate</Button>
              {apiKey && <Button variant="ghost" className="shrink-0 px-2 text-xs" onClick={() => navigator.clipboard.writeText(apiKey)}>Copy</Button>}
            </div>
          </Field>
          {backend === "vllm" && <Field label="Checkpoint format" help="Pre-quantized checkpoints (FP8/GPTQ/AWQ) are detected automatically; Auto is almost always right. Format choice is the biggest lever on VRAM/RAM use: FP8/AWQ/GPTQ need far less VRAM than FP16 so bigger models or contexts fit." hint="Pre-quantized GPTQ/AWQ/FP8 checkpoints are auto-detected from model config.">
            <select className={inputCls} value={quant.toLowerCase()} onChange={(e) => setQuant(e.target.value)}>
              <option value="auto">Auto (recommended)</option>
              <option value="fp16">FP16 / BF16</option>
              <option value="fp8">FP8 runtime quantization</option>
              <option value="gptq">GPTQ checkpoint</option>
              <option value="awq">AWQ checkpoint</option>
            </select>
          </Field>}
          {backend === "vllm" && (
            <>
              <Field label="Startup / performance mode" help="Eager starts fastest and uses less VRAM (no graph capture); CUDA graphs give higher steady-state throughput but add start-up compile time and extra VRAM." hint="Eager mode starts fastest and uses less VRAM. CUDA graphs improve steady-state throughput but add compile/capture time.">
                <select className={inputCls} value={enforceEager ? "eager" : "graphs"} onChange={(e) => setEnforceEager(e.target.value === "eager")}>
                  <option value="eager">Eager (recommended — faster startup, less VRAM)</option>
                  <option value="graphs">CUDA graphs (higher throughput)</option>
                </select>
              </Field>
              <div className="sm:col-span-2 rounded-lg border border-accent/20 bg-accent/5 p-3 text-xs text-slate-400">
                <span className="font-semibold text-sky-200">vLLM 0.30+ defaults:</span> attention backend Auto, prefix caching enabled, and chunked prefill managed by vLLM. KV precision can be inherited globally or overridden per server above.
              </div>
              <EnvironmentVarsEditor initialEnv={env} onChange={setEnv} />
              <Field label="GPU memory utilization" help="Fraction of GPU VRAM vLLM may reserve for weights + KV cache. Higher fits longer contexts and more concurrency but leaves less headroom (out-of-memory risk). 0.85 default." hint="Fraction of GPU memory vLLM may reserve (0–1). Default 0.85.">
                <input type="number" step="0.05" min="0.1" max="0.95" className={inputCls} value={gpuUtil} onChange={(e) => setGpuUtil(e.target.value)} />
              </Field>
            </>
          )}
          {backend === "llamacpp" && (
            <>
              <Field label="llama.cpp Fork" help="Which llama.cpp build runs this server (e.g. PrismML for ternary quants, or your local build). Different forks can use different VRAM for the same model if they offload differently — pick the one the model requires." hint="Fork required by the model (e.g. ternary PQ2_0/PTQ1_0 → PrismML). Manage forks in Settings → llama.cpp Forks.">
                <select className={inputCls} value={llamaCppChannel} onChange={(e) => setLlamaCppChannel(e.target.value)}>
                  {llamaCppForks.map((fork) => (
                    <option key={fork.id} value={fork.id}>{fork.name}</option>
                  ))}
                </select>
              </Field>
              <Field label="GGUF quantization label" help="Informational only — llama.cpp reads the quant from the GGUF itself. The real file size (Q4/Q5/Q8…) decides how much VRAM/RAM the server needs." hint="Metadata inferred from the filename; llama.cpp reads it from the GGUF itself.">
                <input className={inputCls} placeholder="e.g. PQ2_0, PTQ1_0, Q2_0_g64" value={quant} onChange={(e) => setQuant(e.target.value)} />
              </Field>
              {nGpuLayers.trim() && fit && (
                <div className="sm:col-span-2 rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                  Explicit <code>-ngl</code> disables llama.cpp automatic fitting. Clear GPU layers to let fit choose automatically.
                </div>
              )}
              <Field label="GPU layers (-ngl)" help="How many model layers stay on the GPU (more = faster, more VRAM). Blank lets automatic fit decide. Setting this manually disables automatic fitting.">
                <input className={inputCls} placeholder="auto" value={nGpuLayers} onChange={(e) => setNGpuLayers(e.target.value)} />
              </Field>
              <Field label="CPU MoE layers" help="For mixture-of-experts models: how many expert layers stay on the CPU. Offloading them saves VRAM but slows inference significantly. Blank keeps all experts on the GPU.">
                <input className={inputCls} value={nCpuMoe} onChange={(e) => setNCpuMoe(e.target.value)} />
              </Field>
              <Field label="Device(s)" help="Which GPUs to use (e.g. CUDA0,CUDA1). Spreading across GPUs splits VRAM load; a single device is fastest when the model fits. No direct RAM effect." hint="Optional llama.cpp device ids, e.g. CUDA0 or CUDA0,CUDA1.">
                <input className={inputCls} list="llamacpp-devices" placeholder="auto" value={device} onChange={(e) => setDevice(e.target.value)} />
                <datalist id="llamacpp-devices">
                  {devices.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                </datalist>
              </Field>
              <Field label="Log verbosity (0–5)" help="llama.cpp log detail. Higher levels cost CPU/disk but help debugging; no effect on model RAM." hint="llama.cpp: 0 generic, 1 error, 2 warning, 3 info, 4 trace, 5 debug.">
                <input className={inputCls} placeholder="3" value={logVerbosity} onChange={(e) => setLogVerbosity(e.target.value)} />
              </Field>
              <Field label="mmproj path (optional)" help="Vision projector weights (.mmproj) for vision models. Adds its file size to VRAM/RAM load; leave blank for text-only models.">
                <input className={inputCls} value={mmprojPath} onChange={(e) => setMmprojPath(e.target.value)} />
              </Field>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" checked={flashAttn} onChange={(e) => setFlashAttn(e.target.checked)} /> Flash attention <HelpBubble text="Fused attention kernels. Faster and uses less VRAM for attention than standard attention; recommended on for long contexts." />
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" checked={jinja} onChange={(e) => setJinja(e.target.checked)} /> Jinja tool-calling templates <HelpBubble text="Enables the model's Jinja chat template for tool calls and thinking. No effect on model speed or RAM; required for tool-calling support." />
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" checked={noKvOffload} onChange={(e) => setNoKvOffload(e.target.checked)} /> Disable KV-cache offload <HelpBubble text="Keeps the entire KV cache in VRAM instead of spilling to CPU RAM. Faster but uses much more VRAM for long contexts; leave off to allow spill." />
              </label>
              <Field label="CPU threads" help="CPU threads for llama.cpp (--threads). More threads can speed CPU offload but steal CPU from the panel/OS; blank uses auto-detection." hint="Blank uses llama.cpp auto-detection.">
                <input type="number" min="1" className={inputCls} placeholder="auto" value={threads} onChange={(e) => setThreads(e.target.value)} />
              </Field>
              <Field label="Logical batch size" help="Logical batch size (-b). Larger batches raise throughput but scale KV-cache and activation VRAM/RAM with every extra sequence.">
                <input type="number" min="1" className={inputCls} placeholder="llama.cpp default" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
              </Field>
              <Field label="Physical micro-batch size" help="Physical micro-batch size (-ub). Smaller values lower peak VRAM/RAM but can be slower; blank uses llama.cpp's default.">
                <input type="number" min="1" className={inputCls} placeholder="llama.cpp default" value={ubatchSize} onChange={(e) => setUbatchSize(e.target.value)} />
              </Field>
              <Field label="Parallel request slots" help="How many requests run in parallel (-p). More slots = higher concurrency but proportionally more KV-cache VRAM/RAM. 0 uses llama.cpp's automatic selection." hint="0 uses llama.cpp automatic selection.">
                <input type="number" min="0" className={inputCls} placeholder="auto" value={parallel} onChange={(e) => setParallel(e.target.value)} />
              </Field>
              <Field label="Extra llama.cpp arguments" help="Raw llama.cpp flags, one per line. Speed and RAM effects depend entirely on the flags; the structured settings above take precedence." hint="One argument per line. Structured settings above take precedence; harness request parameters (thinking level, temperature, max_tokens, …) override these defaults.">
                <textarea className={`${inputCls} min-h-20 font-mono text-xs`} placeholder={"--jinja\n--override-kv key=value"} value={extraArgs} onChange={(e) => setExtraArgs(e.target.value)} />
              </Field>
            </>
          )}
          <label className="flex items-center gap-2 text-sm text-slate-300 sm:col-span-2">
            <input type="checkbox" checked={metrics} onChange={(e) => setMetrics(e.target.checked)} /> Collect runtime metrics from this server <HelpBubble text="Polls the server's /metrics endpoint for the dashboard. Negligible CPU/RAM cost; no effect on model memory." />
          </label>
        </div>
      </Collapsible>

      {task !== "embed" && (
        <Collapsible
          title="Speculative decoding (optional)"
          subtitle={
            speculativeEnabled
              ? `Enabled · ${backend === "llamacpp" ? (speculativeType || "draft-simple") : "vLLM draft"} · ${
                  backend === "llamacpp"
                    ? (speculativeType.includes("mtp")
                        ? "self-drafting (MTP)"
                        : (speculativeModelPath || speculativeModel || (speculativeType.includes("ngram") ? "ngram cache" : "none")))
                    : (speculativeModel || "none")
                } (${speculativeNumTokens || (speculativeType.includes("mtp") ? "2" : "5")} tokens${speculativeDraftNgl ? `, ngl: ${speculativeDraftNgl}` : ""}${speculativeDraftPMin ? `, p-min: ${speculativeDraftPMin}` : ""})`
              : "Accelerate inference with a smaller draft model running alongside the main model"
          }
          storageKey="llm_panel_server_form_speculative"
          className="mt-3"
        >
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="sm:col-span-2 rounded-lg border border-purple-500/30 bg-purple-500/10 p-3 text-xs text-purple-200">
              <strong>Notice:</strong> Speculative decoding loads a smaller &ldquo;draft&rdquo; model in parallel to generate candidate tokens, which are verified in batches by the main model.
              The draft model <strong>consumes additional VRAM/RAM</strong> alongside the main model weights and must share the same tokenizer family/vocabulary.
            </div>

            <label className="flex items-center gap-2 text-sm text-slate-300 sm:col-span-2">
              <input
                type="checkbox"
                checked={speculativeEnabled}
                onChange={(e) => setSpeculativeEnabled(e.target.checked)}
              />
              Enable speculative decoding <HelpBubble text="Runs a draft model (or n-gram cache) that proposes tokens the main model verifies, typically raising tokens/sec. Adds the draft model's weights to VRAM/RAM use." />
            </label>

            {speculativeEnabled && (
              <>
                {backend === "llamacpp" && (
                  <Field
                    label="Speculative decoding type (--spec-type)"
                    help="Draft method to use. 'draft-simple' uses a standard draft model. 'draft-mtp' for Multi-Token Prediction. 'draft-eagle3' for EAGLE-3 speculative decoding. 'draft-dflash' / 'draft-dspark' for deep speculative models. 'ngram-mod' uses n-gram context matching without requiring a separate draft model."
                    hint="llama.cpp: --spec-type draft-simple | draft-mtp | draft-eagle3 | draft-dflash | draft-dspark | ngram-mod"
                  >
                    <select
                      className={inputCls}
                      value={speculativeType}
                      onChange={(e) => {
                        const nextType = e.target.value;
                        setSpeculativeType(nextType);
                        if (nextType === "draft-mtp" && (!speculativeNumTokens || speculativeNumTokens === "5")) {
                          setSpeculativeNumTokens("2");
                        }
                      }}
                    >
                      <option value="">Default (draft-simple)</option>
                      <option value="draft-simple">Standard Draft Model (draft-simple)</option>
                      <option value="draft-mtp">Multi-Token Prediction (draft-mtp)</option>
                      <option value="draft-eagle3">EAGLE-3 Speculative (draft-eagle3)</option>
                      <option value="draft-dflash">dFlash Speculative (draft-dflash)</option>
                      <option value="draft-dspark">dSpark Speculative (draft-dspark)</option>
                      <option value="ngram-mod">N-gram / Lookup Cache (ngram-mod)</option>
                    </select>
                  </Field>
                )}

                {backend === "llamacpp" && speculativeType === "draft-mtp" ? (
                  <div className="sm:col-span-2 rounded-lg border border-indigo-500/30 bg-indigo-950/20 p-3 text-xs text-indigo-300">
                    <div className="flex items-center gap-1.5 font-medium text-indigo-200">
                      <span className="inline-block h-2 w-2 rounded-full bg-indigo-400" />
                      Self-Drafting Active (Multi-Token Prediction)
                    </div>
                    <div className="mt-1 text-slate-400">
                      This model carries integrated Multi-Token Prediction heads directly in its weights. Llama.cpp will draft future tokens using internal MTP heads without needing a separate draft GGUF file.
                    </div>
                  </div>
                ) : (
                  <Field
                    label={backend === "llamacpp" ? "Draft GGUF model path" : "Draft model ID (HF or local)"}
                    help="Small draft model loaded alongside the main model. Its weights add to VRAM/RAM use; a faster draft means faster overall generation."
                    hint={
                      backend === "llamacpp"
                        ? speculativeType.toLowerCase().includes("ngram")
                          ? "Optional for ngram-mod. If omitted, ngram cache matches against prompt context."
                          : "Must be a GGUF file compatible with the target model (e.g. 0.5B draft for 7B target)."
                        : "Hugging Face model repository ID or local directory (e.g. Qwen/Qwen2.5-0.5B-Instruct)."
                    }
                  >
                    {backend === "llamacpp" ? (
                      <div className="flex gap-1.5">
                        <input
                          className={inputCls}
                          placeholder={
                            speculativeType.toLowerCase().includes("ngram")
                              ? "Optional: C:\\models\\draft-model.gguf (or leave blank for context ngrams)"
                              : "C:\\models\\draft-model-Q4_K_M.gguf"
                          }
                          value={speculativeModelPath || speculativeModel}
                          onChange={(e) => {
                            setSpeculativeModelPath(e.target.value);
                            setSpeculativeModel(e.target.value);
                          }}
                        />
                        <Button
                          variant="ghost"
                          className="shrink-0 px-3 text-xs"
                          onClick={async () => {
                            try {
                              const selected = await open({
                                title: "Select draft GGUF model file",
                                filters: [{ name: "GGUF models", extensions: ["gguf"] }],
                              });
                              if (selected && typeof selected === "string") {
                                setSpeculativeModelPath(selected);
                                setSpeculativeModel(selected);
                              }
                            } catch (browseErr) {
                              onErr(String(browseErr));
                            }
                          }}
                        >
                          Browse…
                        </Button>
                      </div>
                    ) : (
                      <input
                        className={inputCls}
                        placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
                        value={speculativeModel}
                        onChange={(e) => setSpeculativeModel(e.target.value)}
                      />
                    )}
                  </Field>
                )}

                <Field
                  label={backend === "llamacpp" ? "Max draft tokens (--spec-draft-n-max / --draft-max)" : "Number of speculative tokens"}
                  help="Tokens the draft proposes per forward pass. More tokens can mean higher tokens/sec but less gain if predictions are often wrong. No direct VRAM/RAM change."
                  hint={speculativeType === "draft-mtp" ? "llama.cpp MTP: --spec-draft-n-max N. Default is 2 for MTP heads." : "llama.cpp: --spec-draft-n-max N (fallback --draft-max N). Default is 5."}
                >
                  <input
                    type="number"
                    min="1"
                    max="64"
                    className={inputCls}
                    placeholder={speculativeType === "draft-mtp" ? "2" : "5"}
                    value={speculativeNumTokens}
                    onChange={(e) => setSpeculativeNumTokens(e.target.value)}
                  />
                </Field>

                {backend === "llamacpp" && (
                  <>
                    <Field
                      label="Min draft tokens (--spec-draft-n-min)"
                      help="Minimum number of draft tokens to draft before early verification. Default is 0."
                      hint="llama.cpp: --spec-draft-n-min N (fallback --draft-min N). Blank uses 0."
                    >
                      <input
                        type="number"
                        min="0"
                        max="32"
                        className={inputCls}
                        placeholder="auto (0)"
                        value={speculativeDraftNMin}
                        onChange={(e) => setSpeculativeDraftNMin(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft GPU layers (-ngld / --spec-draft-ngl)"
                      help="Number of layers to offload to GPU for the draft model. Set to '99' or 'all' to fully offload, '0' to keep draft on CPU, or a specific layer count."
                      hint="llama.cpp: -ngld N / --spec-draft-ngl N. Blank auto-offloads draft to GPU."
                    >
                      <input
                        className={inputCls}
                        placeholder="auto (e.g. 99, all, 0)"
                        value={speculativeDraftNgl}
                        onChange={(e) => setSpeculativeDraftNgl(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft hardware device (-devd / --spec-draft-device)"
                      help="Target hardware device for draft model (e.g. CUDA0, CUDA1). Setting this to a secondary GPU isolates draft memory and execution from the main model."
                      hint="llama.cpp: -devd <device> / --spec-draft-device <device>."
                    >
                      <input
                        className={inputCls}
                        list="llamacpp-draft-devices"
                        placeholder="auto (same as main model)"
                        value={speculativeDraftDevice}
                        onChange={(e) => setSpeculativeDraftDevice(e.target.value)}
                      />
                      <datalist id="llamacpp-draft-devices">
                        {devices.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.name} ({d.backend})
                          </option>
                        ))}
                      </datalist>
                    </Field>

                    <Field
                      label="Min draft probability (--spec-draft-p-min)"
                      help="Minimum acceptance probability for speculative draft tokens (0.0 to 1.0). Tokens with confidence below this threshold are pruned early to prevent wasted verification passes."
                      hint="llama.cpp: --spec-draft-p-min P (fallback --draft-p-min P). e.g. 0.8"
                    >
                      <input
                        type="number"
                        step="0.05"
                        min="0.0"
                        max="1.0"
                        className={inputCls}
                        placeholder="auto (e.g. 0.8)"
                        value={speculativeDraftPMin}
                        onChange={(e) => setSpeculativeDraftPMin(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft split probability (--spec-draft-p-split)"
                      help="Probability threshold to split speculative tree branches when using tree-based draft decoding."
                      hint="llama.cpp: --spec-draft-p-split P (fallback --draft-p-split P). e.g. 0.1"
                    >
                      <input
                        type="number"
                        step="0.05"
                        min="0.0"
                        max="1.0"
                        className={inputCls}
                        placeholder="auto (e.g. 0.1)"
                        value={speculativeDraftPSplit}
                        onChange={(e) => setSpeculativeDraftPSplit(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft sampling mode (--spec-draft-sampling)"
                      help="Token selection method used by the draft model. Greedy is standard and fastest; probabilistic matches the target sampling distribution."
                      hint="llama.cpp: --spec-draft-sampling greedy | probabilistic"
                    >
                      <select
                        className={inputCls}
                        value={speculativeDraftSampling}
                        onChange={(e) => setSpeculativeDraftSampling(e.target.value)}
                      >
                        <option value="">Default (greedy)</option>
                        <option value="greedy">Greedy (standard, fastest)</option>
                        <option value="probabilistic">Probabilistic</option>
                      </select>
                    </Field>

                    <Field
                      label="Draft backend sampling"
                      help="Performs draft sampling directly on the GPU/backend instead of host CPU."
                      hint="llama.cpp: --spec-draft-backend-sampling / --no-spec-draft-backend-sampling"
                    >
                      <select
                        className={inputCls}
                        value={speculativeDraftBackendSampling}
                        onChange={(e) => setSpeculativeDraftBackendSampling(e.target.value)}
                      >
                        <option value="">Default (auto)</option>
                        <option value="true">Enable (--spec-draft-backend-sampling)</option>
                        <option value="false">Disable (--no-spec-draft-backend-sampling)</option>
                      </select>
                    </Field>

                    <div className="sm:col-span-2 pt-2 border-t border-edge/60">
                      <span className="text-xs font-semibold text-slate-300">
                        Advanced Draft Performance &amp; Memory Flags
                      </span>
                    </div>

                    <Field
                      label="Draft KV cache type K (-ctkd / --spec-draft-type-k)"
                      help="KV cache quantization format for draft model keys. Lower precision (q8_0/q4_0) saves draft VRAM/RAM; f16 is highest quality."
                      hint="llama.cpp: -ctkd format / --spec-draft-type-k format. Default is f16."
                    >
                      <select
                        className={inputCls}
                        value={speculativeDraftCacheTypeK}
                        onChange={(e) => setSpeculativeDraftCacheTypeK(e.target.value.toLowerCase().replaceAll("-", "_"))}
                      >
                        <option value="">Auto (llama.cpp default: f16)</option>
                        <option value="q8_0">q8_0 (recommended)</option>
                        <option value="q4_0">q4_0</option>
                        <option value="q4_1">q4_1</option>
                        <option value="q5_0">q5_0</option>
                        <option value="q5_1">q5_1</option>
                        <option value="iq4_nl">iq4_nl</option>
                        <option value="f16">f16</option>
                        <option value="bf16">bf16</option>
                        <option value="f32">f32</option>
                      </select>
                    </Field>

                    <Field
                      label="Draft KV cache type V (-ctvd / --spec-draft-type-v)"
                      help="KV cache quantization format for draft model values. Lower precision (q8_0/q4_0) saves draft VRAM/RAM."
                      hint="llama.cpp: -ctvd format / --spec-draft-type-v format. Default is f16."
                    >
                      <select
                        className={inputCls}
                        value={speculativeDraftCacheTypeV}
                        onChange={(e) => setSpeculativeDraftCacheTypeV(e.target.value.toLowerCase().replaceAll("-", "_"))}
                      >
                        <option value="">Auto (llama.cpp default: f16)</option>
                        <option value="q8_0">q8_0 (recommended)</option>
                        <option value="q4_0">q4_0</option>
                        <option value="q4_1">q4_1</option>
                        <option value="q5_0">q5_0</option>
                        <option value="q5_1">q5_1</option>
                        <option value="iq4_nl">iq4_nl</option>
                        <option value="f16">f16</option>
                        <option value="bf16">bf16</option>
                        <option value="f32">f32</option>
                      </select>
                    </Field>

                    <Field
                      label="Draft physical micro-batch size (--spec-draft-n-ubatch)"
                      help="Physical micro-batch size for draft model evaluation. Lower values reduce peak memory spikes."
                      hint="llama.cpp: --spec-draft-n-ubatch N. Blank uses default."
                    >
                      <input
                        type="number"
                        min="1"
                        className={inputCls}
                        placeholder="auto"
                        value={speculativeDraftNUbatch}
                        onChange={(e) => setSpeculativeDraftNUbatch(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft CPU threads (-td / --spec-draft-threads)"
                      help="CPU threads allocated to draft model inference. Blank uses auto-detection."
                      hint="llama.cpp: -td N / --spec-draft-threads N"
                    >
                      <input
                        type="number"
                        min="1"
                        className={inputCls}
                        placeholder="auto"
                        value={speculativeDraftThreads}
                        onChange={(e) => setSpeculativeDraftThreads(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft batch threads (-tbd / --spec-draft-threads-batch)"
                      help="CPU threads allocated to draft model batch processing."
                      hint="llama.cpp: -tbd N / --spec-draft-threads-batch N"
                    >
                      <input
                        type="number"
                        min="1"
                        className={inputCls}
                        placeholder="auto"
                        value={speculativeDraftThreadsBatch}
                        onChange={(e) => setSpeculativeDraftThreadsBatch(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft CPU MoE offload (-cmoed)"
                      help="Offload draft model Mixture-of-Experts weights to system CPU memory."
                      hint="llama.cpp: -cmoed / --spec-draft-cpu-moe"
                    >
                      <label className="flex items-center gap-2 pt-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          checked={speculativeDraftCpuMoe}
                          onChange={(e) => setSpeculativeDraftCpuMoe(e.target.checked)}
                        />
                        Enable draft CPU MoE offload
                      </label>
                    </Field>

                    <Field
                      label="Draft CPU MoE layers (-ncmoed)"
                      help="Number of draft model MoE layers to offload to CPU."
                      hint="llama.cpp: -ncmoed N / --spec-draft-n-cpu-moe N"
                    >
                      <input
                        type="number"
                        min="0"
                        className={inputCls}
                        placeholder="auto"
                        value={speculativeDraftNCpuMoe}
                        onChange={(e) => setSpeculativeDraftNCpuMoe(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft process priority (--spec-draft-prio / --prio-draft)"
                      help="Execution priority for draft model compute queue (0-3)."
                      hint="llama.cpp: --spec-draft-prio N / --prio-draft N"
                    >
                      <input
                        type="number"
                        min="0"
                        max="3"
                        className={inputCls}
                        placeholder="auto (0-3)"
                        value={speculativeDraftPrio}
                        onChange={(e) => setSpeculativeDraftPrio(e.target.value)}
                      />
                    </Field>

                    <Field
                      label="Draft polling mode (--spec-draft-poll / --poll-draft)"
                      help="Polling level for draft worker completion (0-100)."
                      hint="llama.cpp: --spec-draft-poll N / --poll-draft N"
                    >
                      <input
                        type="number"
                        min="0"
                        max="100"
                        className={inputCls}
                        placeholder="auto (0-100)"
                        value={speculativeDraftPoll}
                        onChange={(e) => setSpeculativeDraftPoll(e.target.value)}
                      />
                    </Field>
                  </>
                )}
              </>
            )}
          </div>
        </Collapsible>
      )}

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-edge bg-surface-2/60 p-3">
        <div>
          <div className="text-sm font-semibold text-slate-200">Context fit test</div>
          <div className="text-xs text-slate-500">
            Checks the configured context against model weights, KV cache, current VRAM, and available RAM without starting a server.
          </div>
        </div>
        <Button variant="subtle" onClick={analyzeContext} disabled={contextFitLoading}>
          {contextFitLoading ? <Spinner label="Testing…" /> : "Analyze context fit"}
        </Button>
      </div>
      <ContextFitResultPanel
        result={contextFit}
        error={contextFitError}
        loading={contextFitLoading}
        onApply={applyContextRecommendation}
      />
      <StoragePlanner modelWeightGb={8.0} kvCacheGb={Number(swapSpaceGb) || 0} compact />
      <div className="mt-4 flex justify-end gap-2">
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
        <Button
          onClick={submit}
          disabled={creating || (backend === "vllm" ? !modelId.trim() : !modelPath.trim())}
        >
          {creating ? <Spinner label={isEditing ? "saving…" : "creating…"} /> : isEditing ? "Save" : "Create"}
        </Button>
      </div>
    </Card>
  );
}


// Environment Variables Editor for vLLM servers
interface EnvironmentVarsEditorProps {
  initialEnv?: Record<string, string>;
  onChange: (env: Record<string, string>) => void;
}

/// Sanitize a user-supplied environment variable value:
/// - Trim whitespace
/// - Strip a matching pair of surrounding single or double quotes
/// This allows users to enter '0' or "0" and get 0.
function sanitizeEnvValue(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === "'" && last === "'") || (first === '"' && last === '"')) {
      return trimmed.slice(1, -1);
    }
  }
  return trimmed;
}

/// Check if an env var name is a known boolean vLLM var (VLLM_USE_* or similar)
/// and validate that its value is "0" or "1".
function validateBooleanEnvVar(key: string, value: string): string | null {
  if (value === SECRET_PLACEHOLDER) return null;
  const knownBooleanPrefixes = ["VLLM_USE_"];
  const isKnownBoolean = knownBooleanPrefixes.some((prefix) => key.startsWith(prefix));
  if (isKnownBoolean) {
    const sanitized = sanitizeEnvValue(value);
    if (sanitized !== "0" && sanitized !== "1") {
      return `Warning: ${key} is a boolean flag (expected "0" or "1"), got "${sanitized}"`;
    }
  }
  return null;
}

function EnvironmentVarsEditor({ initialEnv, onChange }: EnvironmentVarsEditorProps) {
  const [localEnv, setLocalEnv] = useState<Record<string, string>>(initialEnv ?? {});
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [warnings, setWarnings] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      Object.entries(initialEnv ?? {})
        .map(([key, value]) => [key, validateBooleanEnvVar(key, value)] as const)
        .filter((entry): entry is [string, string] => Boolean(entry[1]))
    )
  );

  const handleAdd = () => {
    const key = newKey.trim();
    const rawValue = newValue;
    if (!key) return;
    // Validate env name: ^[A-Za-z_][A-Za-z0-9_]*$
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      alert("Invalid environment variable name. Use only letters, numbers, and underscores, starting with a letter or underscore.");
      return;
    }
    const value = sanitizeEnvValue(rawValue);
    const warn = validateBooleanEnvVar(key, rawValue);
    setLocalEnv((prev) => ({ ...prev, [key]: value }));
    setWarnings((prev) => {
      const next = { ...prev };
      if (warn) next[key] = warn; else delete next[key];
      return next;
    });
    setNewKey("");
    setNewValue("");
  };

  const handleRemove = (key: string) => {
    setLocalEnv((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setWarnings((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const handleChange = (key: string, rawValue: string) => {
    const value = sanitizeEnvValue(rawValue);
    const warn = validateBooleanEnvVar(key, rawValue);
    setLocalEnv((prev) => ({ ...prev, [key]: value }));
    setWarnings((prev) => {
      const next = { ...prev };
      if (warn) next[key] = warn; else delete next[key];
      return next;
    });
  };

  useEffect(() => {
    onChange(localEnv);
  }, [localEnv, onChange]);

  return (
    <Field label="Environment Variables" help="Per-server overrides for vLLM. Variables can change performance or stability (e.g. CUDA/NCCL/Triton); none change model RAM directly. Bad values can prevent startup." hint="Per-server vLLM overrides. Safe global defaults from Settings are merged first; these values win.">
      <div className="space-y-2">
        {/* Per-server Overrides */}
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-[11px] text-slate-500 font-medium uppercase tracking-wider">
            Per-Server Overrides
            <Button variant="ghost" className="px-2 py-0.5 text-[11px]" onClick={handleAdd} disabled={!newKey.trim()}>
              Add
            </Button>
          </div>
          <div className="flex gap-1.5">
            <input
              className={inputCls}
              placeholder="VAR_NAME"
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              style={{ width: "140px" }}
            />
            <input
              className={inputCls}
              placeholder="value"
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              style={{ flex: 1 }}
            />
          </div>
          {Object.keys(localEnv).length === 0 && (
            <div className="text-xs text-slate-500 italic">No per-server overrides. Add variables above or configure global defaults in Settings.</div>
          )}
          {Object.entries(localEnv).map(([key, value]) => {
            const warn = warnings[key];
            return (
              <div key={key} className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-slate-400 w-48 truncate">{key}</span>
                  <input
                    className={inputCls}
                    value={value === SECRET_PLACEHOLDER ? "" : value}
                     placeholder={value === SECRET_PLACEHOLDER ? "Stored securely; leave blank to keep" : "value"}
                    onChange={(e) => handleChange(key, e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <Button variant="ghost" className="px-2 py-0.5 text-[11px]" onClick={() => handleRemove(key)}>
                    ✕
                  </Button>
                </div>
                {warn && (
                  <div className="ml-[7.5rem] flex items-center gap-1.5 text-[10px] text-amber-300">
                    <span>⚠</span>
                    <span>{warn}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </Field>
  );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

function MarkdownContent({ content }: { content: string }) {
  const parts = content.split(/(```[\s\S]*?```)/g);
  return (
    <div className="space-y-2 text-sm leading-relaxed break-words">
      {parts.map((part, i) => {
        if (part.startsWith("```") && part.endsWith("```")) {
          const inner = part.slice(3, -3);
          const firstLineBreak = inner.indexOf("\n");
          let lang = "";
          let code = inner;
          if (firstLineBreak !== -1) {
            lang = inner.slice(0, firstLineBreak).trim();
            code = inner.slice(firstLineBreak + 1);
          }
          return <CodeBlock key={i} lang={lang} code={code} />;
        }
        return <FormattedText key={i} text={part} />;
      })}
    </div>
  );
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="my-2 overflow-hidden rounded-lg border border-edge bg-surface-1 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-edge bg-surface-3/60 px-3 py-1 text-slate-400">
        <span>{lang || "code"}</span>
        <button
          onClick={copy}
          className="rounded px-1.5 py-0.5 text-[11px] text-slate-400 hover:bg-surface-3 hover:text-slate-200"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-slate-200">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function FormattedText({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <>
      {lines.map((line, idx) => (
        <span key={idx}>
          {idx > 0 && <br />}
          {renderInlineFormatting(line)}
        </span>
      ))}
    </>
  );
}

function renderInlineFormatting(line: string) {
  const segments = line.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  return segments.map((seg, i) => {
    if (seg.startsWith("`") && seg.endsWith("`") && seg.length >= 2) {
      return (
        <code key={i} className="rounded bg-surface-3 px-1 py-0.5 font-mono text-xs text-amber-300">
          {seg.slice(1, -1)}
        </code>
      );
    }
    if (seg.startsWith("**") && seg.endsWith("**") && seg.length >= 4) {
      return <strong key={i} className="font-semibold text-slate-100">{seg.slice(2, -2)}</strong>;
    }
    return seg;
  });
}

function ChatButton({ serverId, port, model }: { serverId: string; port: number; model: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="ghost" onClick={() => setOpen(true)}>💬 Chat</Button>
      {open && (
        <ChatDrawer serverId={serverId} port={port} model={model} onClose={() => setOpen(false)} />
      )}
    </>
  );
}

function ChatDrawer({
  serverId,
  port,
  model,
  onClose,
}: {
  serverId: string;
  port: number;
  model: string;
  onClose: () => void;
}) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.");
  const [temperature, setTemperature] = useState(0.7);
  const [showSettings, setShowSettings] = useState(false);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [editingTitleText, setEditingTitleText] = useState("");
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeRequestIdRef = useRef<string | null>(null);
  activeRequestIdRef.current = activeRequestId;

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const conversationsRef = useRef<Conversation[]>([]);
  conversationsRef.current = conversations;

  const activeConv = conversations.find((c) => c.id === activeId);

  // Load conversations on mount
  useEffect(() => {
    let unmounted = false;
    api.conversationsList().then((all) => {
      if (unmounted) return;
      const filtered = all.filter((c) => c.server_id === serverId);
      setConversations(filtered);
      if (filtered.length > 0) {
        setActiveId(filtered[0].id);
        const sysMsg = filtered[0].messages.find((m) => m.role === "system");
        if (sysMsg) setSystemPrompt(sysMsg.content);
      } else {
        // Create initial conversation
        const initial: Conversation = {
          id: `conv_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          server_id: serverId,
          title: "New Conversation",
          created_at: Date.now(),
          updated_at: Date.now(),
          messages: [{ role: "system", content: "You are a helpful assistant." }],
        };
        setConversations([initial]);
        setActiveId(initial.id);
        api.conversationsSave(initial);
      }
    }).catch(console.error);

    return () => {
      unmounted = true;
    };
  }, [serverId]);

  // Auto-scroll on messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConv?.messages, streaming]);

  // Listen to streaming events
  useEffect(() => {
    const unlistenTokenPromise = events.chatToken((payload) => {
      if (payload.request_id !== activeRequestIdRef.current) return;
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const msgs = [...c.messages];
          if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
            const last = msgs[msgs.length - 1];
            msgs[msgs.length - 1] = { ...last, content: last.content + payload.token };
          }
          return { ...c, messages: msgs, updated_at: Date.now() };
        })
      );
    });

    const unlistenDonePromise = events.chatDone((payload) => {
      if (payload.request_id !== activeRequestIdRef.current) return;
      setStreaming(false);
      setActiveRequestId(null);
      const current = conversationsRef.current.find((c) => c.id === activeId);
      if (current) {
        api.conversationsSave(current).catch(console.error);
      }
    });

    const unlistenCancelPromise = events.chatCancel((payload) => {
      if (payload.request_id !== activeRequestIdRef.current) return;
      setStreaming(false);
      setActiveRequestId(null);
      const current = conversationsRef.current.find((c) => c.id === activeId);
      if (current) {
        api.conversationsSave(current).catch(console.error);
      }
    });

    const unlistenErrorPromise = events.chatError((payload) => {
      if (payload.request_id !== activeRequestIdRef.current) return;
      setStreaming(false);
      setActiveRequestId(null);
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const msgs = [...c.messages];
          if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
            const last = msgs[msgs.length - 1];
            msgs[msgs.length - 1] = {
              ...last,
              content: last.content ? `${last.content}\n\n⚠ Error: ${payload.error}` : `⚠ Error: ${payload.error}`,
            };
          }
          const updated = { ...c, messages: msgs, updated_at: Date.now() };
          api.conversationsSave(updated).catch(console.error);
          return updated;
        })
      );
    });

    return () => {
      unlistenTokenPromise.then((fn) => fn());
      unlistenDonePromise.then((fn) => fn());
      unlistenCancelPromise.then((fn) => fn());
      unlistenErrorPromise.then((fn) => fn());
    };
  }, [activeId]);

  const handleNewChat = () => {
    const newConv: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      server_id: serverId,
      title: `Chat ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
      created_at: Date.now(),
      updated_at: Date.now(),
      messages: [{ role: "system", content: systemPrompt }],
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveId(newConv.id);
    api.conversationsSave(newConv).catch(console.error);
  };

  const handleDeleteConv = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    api.conversationsDelete(id).catch(console.error);
    const next = conversations.filter((c) => c.id !== id);
    setConversations(next);
    if (activeId === id) {
      if (next.length > 0) {
        setActiveId(next[0].id);
      } else {
        const fallback: Conversation = {
          id: `conv_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          server_id: serverId,
          title: "New Conversation",
          created_at: Date.now(),
          updated_at: Date.now(),
          messages: [{ role: "system", content: systemPrompt }],
        };
        setConversations([fallback]);
        setActiveId(fallback.id);
        api.conversationsSave(fallback).catch(console.error);
      }
    }
  };

  const startRename = (e: React.MouseEvent, conv: Conversation) => {
    e.stopPropagation();
    setEditingTitleId(conv.id);
    setEditingTitleText(conv.title);
  };

  const saveRename = (id: string) => {
    if (!editingTitleText.trim()) {
      setEditingTitleId(null);
      return;
    }
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id === id) {
          const updated = { ...c, title: editingTitleText.trim(), updated_at: Date.now() };
          api.conversationsSave(updated).catch(console.error);
          return updated;
        }
        return c;
      })
    );
    setEditingTitleId(null);
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const readers = Array.from(files).map(
      (f) =>
        new Promise<string>((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(String(r.result));
          r.onerror = () => reject(new Error(`Failed to read ${f.name}`));
          r.readAsDataURL(f);
        })
    );
    Promise.all(readers)
      .then((urls) => setPendingImages((prev) => [...prev, ...urls].slice(0, 4)))
      .catch(console.error);
  };

  const handleSend = async () => {
    if ((!input.trim() && pendingImages.length === 0) || streaming || !activeConv) return;
    const userMsg: ChatMessage = {
      role: "user",
      content: input.trim() || "What do you see in this image?",
      ...(pendingImages.length > 0 ? { images: [...pendingImages] } : {}),
    };
    const assistantMsg: ChatMessage = { role: "assistant", content: "" };

    const nextMessages = [...activeConv.messages, userMsg, assistantMsg];
    const autoTitle =
      activeConv.messages.length <= 1 && activeConv.title.startsWith("Chat ")
        ? userMsg.content.slice(0, 24) + (userMsg.content.length > 24 ? "…" : "")
        : activeConv.title;

    const updatedConv: Conversation = {
      ...activeConv,
      title: autoTitle,
      messages: nextMessages,
      updated_at: Date.now(),
    };

    setConversations((prev) =>
      prev.map((c) => (c.id === activeConv.id ? updatedConv : c))
    );
    setInput("");
    setPendingImages([]);

    const reqId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setActiveRequestId(reqId);
    setStreaming(true);

    try {
      await api.serversChatStream(
        reqId,
        serverId,
        nextMessages.slice(0, -1),
        temperature
      );
    } catch (err) {
      setStreaming(false);
      setActiveRequestId(null);
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeConv.id) return c;
          const msgs = [...c.messages];
          msgs[msgs.length - 1] = {
            role: "assistant",
            content: `⚠ Failed to start stream: ${String(err)}`,
          };
          return { ...c, messages: msgs };
        })
      );
    }
  };

  const handleStop = () => {
    if (activeRequestId) {
      api.serversChatCancel(activeRequestId).catch(console.error);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 sm:p-6"
      onClick={onClose}
    >
      <div
        className="flex h-[88vh] w-full max-w-5xl rounded-2xl border border-edge bg-surface-1 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Left Sidebar: Conversations list */}
        <div className="flex w-64 flex-col border-r border-edge bg-surface-2/60">
          <div className="flex items-center justify-between border-b border-edge p-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Chats
            </span>
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1 rounded border border-edge bg-surface-3 px-2 py-1 text-xs text-slate-200 hover:bg-surface-3/80 hover:text-white"
            >
              <span>+</span> New
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {conversations.map((c) => (
              <div
                key={c.id}
                onClick={() => {
                  setActiveId(c.id);
                  const sys = c.messages.find((m) => m.role === "system");
                  if (sys) setSystemPrompt(sys.content);
                }}
                className={`group flex items-center justify-between rounded-lg px-2.5 py-2 text-xs cursor-pointer transition-colors ${
                  c.id === activeId
                    ? "bg-accent/15 text-sky-200 border border-accent/30"
                    : "text-slate-400 hover:bg-surface-3 hover:text-slate-200"
                }`}
              >
                {editingTitleId === c.id ? (
                  <input
                    autoFocus
                    className="w-full rounded bg-surface-1 px-1.5 py-0.5 text-xs text-slate-100 border border-accent outline-hidden"
                    value={editingTitleText}
                    onChange={(e) => setEditingTitleText(e.target.value)}
                    onBlur={() => saveRename(c.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") saveRename(c.id);
                      if (e.key === "Escape") setEditingTitleId(null);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span className="truncate flex-1 font-medium">{c.title}</span>
                )}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity ml-1">
                  <button
                    onClick={(e) => startRename(e, c)}
                    className="p-1 hover:text-slate-100 text-slate-400"
                    title="Rename"
                  >
                    ✏
                  </button>
                  <button
                    onClick={(e) => handleDeleteConv(e, c.id)}
                    className="p-1 hover:text-red-400 text-slate-400"
                    title="Delete"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Main Chat Area */}
        <div className="flex flex-1 flex-col bg-surface-1">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-edge bg-surface-2/40 px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-slate-200">{model}</span>
              <Badge color="slate">port {port}</Badge>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowSettings((s) => !s)}
                className={`rounded border px-2.5 py-1 text-xs transition-colors ${
                  showSettings
                    ? "border-accent/50 bg-accent/10 text-sky-200"
                    : "border-edge bg-surface-3 text-slate-400 hover:text-slate-200"
                }`}
              >
                ⚙ Options
              </button>
              <button
                onClick={onClose}
                className="text-slate-400 hover:text-slate-200 text-sm font-medium"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Options Drawer/Bar */}
          {showSettings && (
            <div className="border-b border-edge bg-surface-2/70 p-3 text-xs space-y-3">
              <div className="flex items-center gap-4">
                <span className="text-slate-400 w-24 font-medium">Temperature:</span>
                <input
                  type="range"
                  min="0.0"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-48 accent-accent"
                />
                <span className="font-mono text-slate-300 w-8">{temperature.toFixed(2)}</span>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-slate-400 w-24 font-medium pt-1">System Prompt:</span>
                <textarea
                  rows={2}
                  value={systemPrompt}
                  onChange={(e) => {
                    setSystemPrompt(e.target.value);
                    if (activeConv) {
                      const updatedMsgs = [...activeConv.messages];
                      const sysIdx = updatedMsgs.findIndex((m) => m.role === "system");
                      if (sysIdx !== -1) {
                        updatedMsgs[sysIdx] = { role: "system", content: e.target.value };
                      } else {
                        updatedMsgs.unshift({ role: "system", content: e.target.value });
                      }
                      const updated = { ...activeConv, messages: updatedMsgs };
                      setConversations((prev) =>
                        prev.map((c) => (c.id === activeConv.id ? updated : c))
                      );
                      api.conversationsSave(updated).catch(console.error);
                    }
                  }}
                  className="flex-1 rounded-md border border-edge bg-surface-1 p-2 text-xs text-slate-200 outline-hidden focus:border-accent"
                  placeholder="System instructions..."
                />
              </div>
            </div>
          )}

          {/* Message History */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {activeConv?.messages
              .filter((m) => m.role !== "system")
              .map((m, i) => (
                <div
                  key={i}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-xl px-4 py-3 text-sm shadow-xs ${
                      m.role === "user"
                        ? "bg-accent text-white rounded-br-xs"
                        : "border border-edge bg-surface-2 text-slate-200 rounded-bl-xs"
                    }`}
                  >
                    {m.images && m.images.length > 0 && (
                      <div className="mb-2 flex flex-wrap gap-2">
                        {m.images.map((src, k) => (
                          <img
                            key={k}
                            src={src}
                            alt={`attachment ${k + 1}`}
                            className="h-24 w-24 rounded-lg border border-white/20 object-cover"
                          />
                        ))}
                      </div>
                    )}
                    <MarkdownContent
                      content={
                        m.content ||
                        (streaming && i === activeConv.messages.filter((x) => x.role !== "system").length - 1
                          ? "…"
                          : "")
                      }
                    />
                  </div>
                </div>
              ))}
            {streaming && (
              <div className="flex items-center gap-2 text-xs text-accent">
                <span className="inline-block h-2 w-2 animate-ping rounded-full bg-accent" />
                <span>Streaming tokens...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Pending image previews */}
          {pendingImages.length > 0 && (
            <div className="flex flex-wrap gap-2 border-t border-edge bg-surface-2/30 px-3 pt-2">
              {pendingImages.map((src, i) => (
                <div key={i} className="relative">
                  <img
                    src={src}
                    alt={`pending ${i + 1}`}
                    className="h-14 w-14 rounded-lg border border-edge object-cover"
                  />
                  <button
                    onClick={() => setPendingImages((prev) => prev.filter((_, k) => k !== i))}
                    className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-surface-3 text-[10px] text-slate-300 hover:text-red-400 border border-edge"
                    title="Remove image"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Input Bar */}
          <div className="flex items-center gap-2 border-t border-edge bg-surface-2/30 p-3">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                handleFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="rounded border border-edge bg-surface-3 px-2.5 py-1.5 text-sm text-slate-300 hover:text-white hover:bg-surface-3/80"
              title="Attach image for vision models (Qwen2-VL, Pixtral)"
              disabled={streaming}
            >
              📎
            </button>
            <input
              className={`${inputCls} flex-1`}
              placeholder="Ask anything... (Press Enter to send)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={streaming}
            />
            {streaming ? (
              <Button variant="danger" onClick={handleStop}>
                ⏹ Stop
              </Button>
            ) : (
              <Button onClick={handleSend} disabled={!input.trim() && pendingImages.length === 0}>
                Send
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function BenchmarkButton({ serverId, model }: { serverId: string; model: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="ghost" onClick={() => setOpen(true)}>⚡ Benchmark</Button>
      {open && (
        <BenchmarkDrawer serverId={serverId} model={model} onClose={() => setOpen(false)} />
      )}
    </>
  );
}

function BenchmarkDrawer({
  serverId,
  model,
  onClose,
}: {
  serverId: string;
  model: string;
  onClose: () => void;
}) {
  const [history, setHistory] = useState<BenchmarkRun[]>([]);
  const [running, setRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState<{
    step: number;
    total_steps: number;
    prompt_tok_s: number;
    gen_tok_s: number;
    latency_ms: number;
  } | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const loadHistory = useCallback(() => {
    api.benchmarksHistory(serverId).then(setHistory).catch(console.error);
  }, [serverId]);

  useEffect(() => {
    loadHistory();

    const unsubStep = events.benchmarkStep((e) => {
      if (e.server_id !== serverId) return;
      setCurrentStep(e);
      setStatusMsg(`Completed prompt ${e.step} of ${e.total_steps}`);
    });

    const unsubDone = events.benchmarkDone((run) => {
      if (run.server_id !== serverId) return;
      setRunning(false);
      setCurrentStep(null);
      setStatusMsg(`Benchmark complete: ${run.gen_tok_s.toFixed(1)} tok/s`);
      setHistory((prev) => [run, ...prev]);
    });

    const unsubCancel = events.benchmarkCancel((e) => {
      if (e.server_id !== serverId) return;
      setRunning(false);
      setCurrentStep(null);
      setStatusMsg("Benchmark cancelled.");
    });

    const unsubError = events.benchmarkError((e) => {
      if (e.server_id !== serverId) return;
      setRunning(false);
      setCurrentStep(null);
      setStatusMsg(`Benchmark error: ${e.error}`);
    });

    return () => {
      unsubStep.then((fn) => fn());
      unsubDone.then((fn) => fn());
      unsubCancel.then((fn) => fn());
      unsubError.then((fn) => fn());
    };
  }, [serverId, loadHistory]);

  const handleRun = async () => {
    setRunning(true);
    setCurrentStep(null);
    setStatusMsg("Starting 3-prompt standardized benchmark suite (small, medium, large context)...");
    try {
      await api.benchmarksRun(serverId);
    } catch (err) {
      setRunning(false);
      setStatusMsg(`Failed to start benchmark: ${String(err)}`);
    }
  };

  const handleStop = async () => {
    try {
      await api.benchmarksCancel(serverId);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 sm:p-6"
      onClick={onClose}
    >
      <div
        className="flex h-[80vh] w-full max-w-4xl flex-col rounded-2xl border border-edge bg-surface-1 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-edge bg-surface-2/40 px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <span className="text-base font-bold text-slate-100">Benchmark Suite</span>
            <span className="text-xs text-slate-400">· {model}</span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-sm font-medium"
          >
            ✕
          </button>
        </div>

        {/* Live Runner Card */}
        <div className="border-b border-edge bg-surface-2/60 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-semibold text-slate-200">Standardized Speed & Latency Benchmark</h4>
              <p className="text-xs text-slate-400 mt-0.5">
                Executes 3 repeatable prompts (small: 50 tok, medium: 250 tok, large: 1000 tok) to measure prompt prefill & generation tok/s.
              </p>
            </div>
            {running ? (
              <Button variant="danger" onClick={handleStop}>
                ⏹ Stop Benchmark
              </Button>
            ) : (
              <Button onClick={handleRun}>
                ⚡ Run Benchmark
              </Button>
            )}
          </div>

          {/* Progress / Status */}
          {statusMsg && (
            <div className="rounded-lg border border-accent/30 bg-accent/10 p-3 text-xs text-sky-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                {running && <Spinner />}
                <span>{statusMsg}</span>
              </div>
              {currentStep && (
                <div className="flex items-center gap-4 font-mono">
                  <span>Gen: <strong className="text-emerald-300">{fmtTokPerSec(currentStep.gen_tok_s)}</strong></span>
                  <span>Prompt: <strong className="text-cyan-300">{fmtTokPerSec(currentStep.prompt_tok_s)}</strong></span>
                  <span>Latency: <strong className="text-slate-200">{currentStep.latency_ms.toFixed(0)} ms</strong></span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* History Table */}
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Past Benchmark Runs</h4>
            <span className="text-xs text-slate-500">{history.length} runs</span>
          </div>
          {history.length === 0 ? (
            <div className="rounded-xl border border-edge bg-surface-2/30 p-8 text-center text-xs text-slate-400">
              No benchmark runs recorded yet. Click "Run Benchmark" above to measure actual tokens/second.
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-edge">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-edge bg-surface-2 text-[11px] font-semibold text-slate-400">
                  <tr>
                    <th className="px-3.5 py-2.5">Date & Time</th>
                    <th className="px-3.5 py-2.5">Quant</th>
                    <th className="px-3.5 py-2.5 text-right">Generation Speed</th>
                    <th className="px-3.5 py-2.5 text-right">Prompt Speed</th>
                    <th className="px-3.5 py-2.5 text-right">Latency</th>
                    <th className="px-3.5 py-2.5 text-right">Prompts</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge/60 bg-surface-1">
                  {history.map((h) => (
                    <tr key={h.id} className="hover:bg-surface-2/40 transition-colors">
                      <td className="px-3.5 py-2.5 text-slate-300">
                        {new Date(h.timestamp * 1000).toLocaleString()}
                      </td>
                      <td className="px-3.5 py-2.5">
                        <span className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] text-accent">
                          {h.quant || "native"}
                        </span>
                      </td>
                      <td className="px-3.5 py-2.5 text-right font-mono font-bold text-emerald-400">
                        {fmtTokPerSec(h.gen_tok_s)}
                      </td>
                      <td className="px-3.5 py-2.5 text-right font-mono text-cyan-300">
                        {fmtTokPerSec(h.prompt_tok_s)}
                      </td>
                      <td className="px-3.5 py-2.5 text-right font-mono text-slate-400">
                        {h.latency_ms.toFixed(0)} ms
                      </td>
                      <td className="px-3.5 py-2.5 text-right text-slate-400">
                        {h.prompt_count}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      

</div>
    </div>
  );
}