import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { api, events, fmtNum, fmtTokPerSec, quantLabel, statusColor } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls, Spinner } from "../ui";
import { Sparkline } from "../components/Sparkline";
import { effectiveModelName } from "../types";
import type { ServerListRow, ServerDef, ChatMessage, Conversation, BenchmarkRun, ServerMetricPoint, GpuSnapshot } from "../types";

export default function Servers() {
  const location = useLocation();
  const [rows, setRows] = useState<ServerListRow[]>([]);
  const [serverSeries, setServerSeries] = useState<Record<string, ServerMetricPoint[]>>({});
  const [gpu, setGpu] = useState<GpuSnapshot | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [prefillModel, setPrefillModel] = useState<string | null>(null);
  const [prefillQuant, setPrefillQuant] = useState<string | null>(null);
  const [prefillSwapSpace, setPrefillSwapSpace] = useState<number | undefined>(undefined);
  const [prefillCpuOffload, setPrefillCpuOffload] = useState<number | undefined>(undefined);
  const [prefillMaxLen, setPrefillMaxLen] = useState<number | undefined>(undefined);
  const [prefillVramContext, setPrefillVramContext] = useState<number | undefined>(undefined);
  const [prefillGpuUtil, setPrefillGpuUtil] = useState<number | undefined>(undefined);
  const [prefillServed, setPrefillServed] = useState<string | undefined>(undefined);
  const [prefillTask, setPrefillTask] = useState<"instruct" | "embed" | undefined>(undefined);

  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [copiedUrlId, setCopiedUrlId] = useState<string | null>(null);
  const [showImportRecipe, setShowImportRecipe] = useState(false);
  const [recipeInput, setRecipeInput] = useState("");
  const [recipeErr, setRecipeErr] = useState<string | null>(null);
  const [toolResults, setToolResults] = useState<Record<string, string>>({});

  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // server id being start/stop/delete
  // log buffers per server (event-driven + hydrated)
  const [logs, setLogs] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const logEndRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const clearPrefills = () => {
    setPrefillModel(null);
    setPrefillQuant(null);
    setPrefillSwapSpace(undefined);
    setPrefillCpuOffload(undefined);
    setPrefillMaxLen(undefined);
    setPrefillVramContext(undefined);
    setPrefillGpuUtil(undefined);
    setPrefillServed(undefined);
    setPrefillTask(undefined);
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
      setPrefillTask(r.task as "instruct" | "embed");
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
      prefillQuant?: string;
      prefillSwapSpace?: number;
      prefillCpuOffload?: number;
      prefillMaxLen?: number;
      prefillVramContext?: number;
    } | null;
    if (state?.prefillModel) {
      setPrefillModel(state.prefillModel);
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
      setShowNew(true);
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

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
              } catch {
                // ignore
              }
            })
          );
          setServerSeries((prev) => ({ ...prev, ...seriesMap }));
        }
      })
      .catch(() => {});
    api.gpuStatus().then(setGpu).catch(() => {});
  }, []);

  useEffect(() => {
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
      events.serverMetrics((e) => {
        setRows((prev) => prev.map((r) => (r.def.id === (e as { id?: string }).id ? r : r)));
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

  const selectedRow = rows.find((r) => r.def.id === selected) ?? null;
  const selectedLog = selected ? logs[selected] ?? "" : "";
  const freeGb = gpu ? gpu.vram_free_mb / 1024 : null;

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      {/* VRAM Pressure Alert */}
      {gpu && freeGb !== null && freeGb < 1.0 && (
        <div className="rounded-xl border border-amber-500/50 bg-amber-500/10 p-3.5 text-xs text-amber-200 shadow-sm flex items-center gap-2.5">
          <span className="text-base">⚠️</span>
          <div>
            <strong>VRAM Pressure Warning:</strong> Less than 1.0 GB GPU VRAM headroom remaining ({freeGb.toFixed(1)} GB free). Starting another server or expanding context length may cause CUDA out-of-memory errors.
          </div>
        </div>
      )}

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
              className="w-full h-44 rounded-lg border border-edge bg-surface-2 p-3 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
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
          initialModelId={prefillModel ?? ""}
          initialQuant={prefillQuant ?? undefined}
          initialSwapSpace={prefillSwapSpace}
          initialCpuOffload={prefillCpuOffload}
          initialMaxLen={prefillMaxLen}
          initialVramContext={prefillVramContext}
          initialGpuUtil={prefillGpuUtil}
          initialServed={prefillServed}
          initialTask={prefillTask}
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
            <Card key={r.def.id} className={`p-3.5 ${selected === r.def.id ? "ring-1 ring-indigo-500/50" : ""}`}>
              <div className="flex flex-wrap items-center gap-3">
                <button className="min-w-0 flex-1 text-left group" onClick={() => setSelected(selected === r.def.id ? null : r.def.id)}>
                  <div className="flex items-center gap-2">
                    <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${
                      r.status === "running" ? "bg-emerald-400 animate-pulse" :
                      r.status === "starting" ? "bg-amber-400 animate-pulse" :
                      r.status === "error" ? "bg-red-400" : "bg-slate-600"
                    }`} />
                    <span className="font-medium text-slate-200">{r.def.name}</span>
                    <Badge color={r.status === "running" ? "emerald" : r.status === "starting" ? "amber" : r.status === "error" ? "red" : "slate"}>
                      {r.status}
                    </Badge>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-slate-500 group-hover:text-slate-400">
                    <span>{r.def.model_id}</span>
                    <span className="text-indigo-300">{r.def.backend === "llamacpp" ? "llama.cpp · Windows" : "vLLM · WSL2"}</span>
                    <span>·</span>
                    <span>port {r.def.port}</span>
                    <span>·</span>
                    <span>{r.def.task}</span>
                    <span>·</span>
                    <span>{quantLabel(r.def.quant)}</span>
                    {r.def.max_model_len ? <span>· ctx {fmtNum(r.def.max_model_len)}</span> : null}
                    {r.def.params_b ? <span>· ~{r.def.params_b.toFixed(2)}B</span> : null}
                    {r.def.swap_space_gb != null && r.def.swap_space_gb > 0 && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                        swap {r.def.swap_space_gb}GB
                      </span>
                    )}
                    {r.def.cpu_offload_gb != null && r.def.cpu_offload_gb > 0 && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-950/80 text-amber-300 border border-amber-800">
                        offload {r.def.cpu_offload_gb}GB
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
                    <>
                      <ChatButton serverId={r.def.id} port={r.def.port} model={effectiveModelName(r.def)} />
                      <Button variant="subtle" onClick={async () => {
                        try {
                          const result = await api.serversTestToolCall(r.def.id);
                          setToolResults((prev) => ({ ...prev, [r.def.id]: result.passed ? "Tool call passed" : `Failed: ${result.hint}\n${JSON.stringify(result.response)}` }));
                        } catch (e) {
                          setToolResults((prev) => ({ ...prev, [r.def.id]: `Tool call error: ${String(e)}` }));
                        }
                      }}>
                        Test tool calling
                      </Button>
                      <BenchmarkButton serverId={r.def.id} model={effectiveModelName(r.def)} />
                    </>
                  )}
                  {r.status === "running" && (
                    <Button variant="ghost" onClick={async () => {
                      await navigator.clipboard.writeText(`http://127.0.0.1:${r.def.port}/v1`);
                      setCopiedUrlId(r.def.id);
                      setTimeout(() => setCopiedUrlId(null), 2000);
                    }}>
                      {copiedUrlId === r.def.id ? "URL copied" : "Copy API URL"}
                    </Button>
                  )}
                  {r.status === "running" && (
                    <Button variant="ghost" disabled={busy === r.def.id} onClick={() => act(r.def.id, () => api.serversRestart(r.def.id))}>
                      Restart
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    title="Copy portable server recipe JSON to clipboard"
                    onClick={() => {
                      copyRecipe(r.def);
                    }}
                  >
                    {copiedId === r.def.id ? "Copied!" : "Recipe"}
                  </Button>
                  <Button variant="subtle" disabled={busy === r.def.id} onClick={() => { if (confirm(`Delete server "${r.def.name}"?`)) act(r.def.id, () => api.serversDelete(r.def.id)); }}>
                    ✕
                  </Button>
                </div>
              </div>
              {r.error && <div className="mt-2 text-xs text-red-300">{r.error}</div>}
              {toolResults[r.def.id] && (
                <pre className="mt-2 whitespace-pre-wrap rounded-md border border-edge bg-surface p-2 text-[11px] text-slate-300">{toolResults[r.def.id]}</pre>
              )}
              {r.metrics && r.status === "running" && (
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-500 sm:grid-cols-4">
                  <div>gen tok: <span className="text-slate-300">{fmtNum(r.metrics.total_generation_tokens)}</span></div>
                  <div>prompt tok: <span className="text-slate-300">{fmtNum(r.metrics.total_prompt_tokens)}</span></div>
                  <div>running: <span className="text-slate-300">{r.metrics.running}</span> · waiting <span className="text-slate-300">{r.metrics.waiting}</span></div>
                  <div>requests: <span className="text-slate-300">{fmtNum(r.metrics.requests)}</span></div>
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
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card>
            <CardTitle
              right={
                <div className="flex gap-2">
                  <span className={`${statusColor(selectedRow.status)} text-sm font-medium`}>{selectedRow.status}</span>
                  {selectedRow.status === "running" && selectedRow.def.task === "instruct" && (
                    <>
                      <ChatButton serverId={selectedRow.def.id} port={selectedRow.def.port} model={effectiveModelName(selectedRow.def)} />
                      <BenchmarkButton serverId={selectedRow.def.id} model={effectiveModelName(selectedRow.def)} />
                    </>
                  )}
                </div>
              }
            >
              {selectedRow.def.name} · logs
            </CardTitle>
            <div className="h-80 overflow-y-auto rounded-md bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-slate-300">
              {selectedLog ? (
                <>
                  <pre className="whitespace-pre-wrap">{selectedLog}</pre>
                  <div ref={(el) => { if (selected) logEndRefs.current[selected] = el; }} />
                </>
              ) : (
                <span className="text-slate-600">{selectedRow.status === "running" ? "streaming logs…" : "server stopped — start it to see logs."}</span>
              )}
            </div>
          </Card>
          <Card>
            <CardTitle>Metrics</CardTitle>
            {selectedRow.metrics ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <Stat label="Generation tok/s" value={selectedRow.metrics.measured?.tokens_per_sec != null ? fmtTokPerSec(selectedRow.metrics.measured.tokens_per_sec) : "– (needs traffic)"} />
                  <Stat label="Prompt tok/s" value={selectedRow.metrics.measured?.prompt_tokens_per_sec != null ? fmtTokPerSec(selectedRow.metrics.measured.prompt_tokens_per_sec) : "–"} />
                  <Stat label="Total gen tokens" value={fmtNum(selectedRow.metrics.total_generation_tokens)} />
                  <Stat label="Total prompt tokens" value={fmtNum(selectedRow.metrics.total_prompt_tokens)} />
                  <Stat label="In-flight / waiting" value={`${selectedRow.metrics.running} / ${selectedRow.metrics.waiting}`} />
                  <Stat label="Requests" value={fmtNum(selectedRow.metrics.requests)} />
                </div>
                {selectedRow.status === "running" && serverSeries[selectedRow.def.id] && serverSeries[selectedRow.def.id].length > 0 && (
                  <div className="pt-2 border-t border-edge/40">
                    <Sparkline
                      label="Throughput History (last 5 min)"
                      data={serverSeries[selectedRow.def.id].map((p) => p.tok_s)}
                      height={44}
                      min={0}
                      color="#38bdf8"
                      unit="tok/s"
                      currentValue={selectedRow.metrics.measured?.tokens_per_sec != null ? fmtTokPerSec(selectedRow.metrics.measured.tokens_per_sec) : undefined}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-slate-500">No metrics yet — the monitor samples /metrics every 5s once running.</div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-edge bg-surface p-2.5">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="mt-0.5 font-mono text-slate-200">{value}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function NewServerForm({
  freeGb,
  initialModelId = "",
  initialQuant,
  initialSwapSpace,
  initialCpuOffload,
  initialMaxLen,
  initialVramContext,
  initialGpuUtil,
  initialServed,
  initialTask,
  onDone,
  onCancel,
  onErr,
}: {
  freeGb?: number | null;
  initialModelId?: string;
  initialQuant?: string;
  initialSwapSpace?: number;
  initialCpuOffload?: number;
  initialMaxLen?: number;
  initialVramContext?: number;
  initialGpuUtil?: number;
  initialServed?: string;
  initialTask?: "instruct" | "embed";
  onDone: (s: { id: string }) => void;
  onCancel: () => void;
  onErr: (e: string) => void;
}) {
  const [modelId, setModelId] = useState(initialModelId);
  const [name, setName] = useState(initialModelId ? initialModelId.split("/").pop() || "" : "");
  const [task, setTask] = useState<"instruct" | "embed">(initialTask || "instruct");
  const [backend, setBackend] = useState<"vllm" | "llamacpp">("vllm");
  const [quant, setQuant] = useState(initialQuant || "fp16");
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
  const [creating, setCreating] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [mmprojPath, setMmprojPath] = useState("");
  const [ctxSize, setCtxSize] = useState("");
  const [nGpuLayers, setNGpuLayers] = useState("99");
  const [nCpuMoe, setNCpuMoe] = useState("");
  const [flashAttn, setFlashAttn] = useState(true);
  const [jinja, setJinja] = useState(true);
  const [moePresetApplied, setMoePresetApplied] = useState(false);

  useEffect(() => {
    if (initialModelId) {
      setModelId(initialModelId);
      setName(initialModelId.split("/").pop() || "");
    }
    if (initialQuant) {
      setQuant(initialQuant);
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
    if (initialTask !== undefined) {
      setTask(initialTask);
    }
  }, [
    initialModelId,
    initialQuant,
    initialSwapSpace,
    initialCpuOffload,
    initialMaxLen,
    initialVramContext,
    initialGpuUtil,
    initialServed,
    initialTask,
  ]);

  const submit = async () => {
    if (!modelId.trim()) return;
    setCreating(true);
    try {
      const parsedSwap = swapSpaceGb !== "" ? parseInt(swapSpaceGb, 10) : null;
      const parsedOffload = cpuOffloadGb !== "" ? parseInt(cpuOffloadGb, 10) : null;
      const s = await api.serversCreate({
        backend,
        model_id: modelId.trim(),
        name: name.trim() || modelId.split("/").pop() || "server",
        task,
        quant,
        gpu_mem_util: parseFloat(gpuUtil) || 0.85,
        max_model_len: maxLen ? parseInt(maxLen, 10) : undefined,
        swap_space_gb: parsedSwap !== null && !isNaN(parsedSwap) ? parsedSwap : undefined,
        cpu_offload_gb: parsedOffload !== null && !isNaN(parsedOffload) ? parsedOffload : undefined,
        served_model_name: served.trim() || undefined,
        model_path: backend === "llamacpp" ? (modelPath.trim() || modelId.trim()) : undefined,
        mmproj_path: backend === "llamacpp" ? (mmprojPath.trim() || undefined) : undefined,
        ctx_size: backend === "llamacpp" ? parseInt(ctxSize, 10) || undefined : undefined,
        n_gpu_layers: backend === "llamacpp" ? parseInt(nGpuLayers, 10) || 99 : undefined,
        n_cpu_moe: backend === "llamacpp" ? parseInt(nCpuMoe, 10) || undefined : undefined,
        flash_attn: backend === "llamacpp" ? flashAttn : undefined,
        jinja: backend === "llamacpp" ? jinja : undefined,
      });
      onDone(s);
    } catch (e) {
      onErr(String(e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <Card className="border-indigo-500/30">
      <CardTitle>Define new server</CardTitle>
      {freeGb !== null && freeGb !== undefined && freeGb < 1.0 && (
        <div className="mb-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-200">
          ⚠️ <strong>VRAM Pressure:</strong> Current free VRAM is only {freeGb.toFixed(1)} GB. Consider configuring Swap Space or CPU Offload below to avoid OOM errors.
        </div>
      )}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Backend">
          <select className={inputCls} value={backend} onChange={(e) => setBackend(e.target.value as "vllm" | "llamacpp")}>
            <option value="vllm">vLLM (WSL2)</option>
            <option value="llamacpp">llama.cpp (Windows)</option>
          </select>
        </Field>
        <Field label={backend === "llamacpp" ? "GGUF model path" : "Model id (HF)"}>
          <input className={inputCls} placeholder={backend === "llamacpp" ? "C:\\models\\model-Q4_K_M.gguf" : "Qwen/Qwen2.5-0.5B-Instruct"} value={backend === "llamacpp" ? modelPath : modelId} onChange={(e) => {
            if (backend === "llamacpp") {
              setModelPath(e.target.value);
              setModelId(e.target.value);
            } else {
              setModelId(e.target.value);
            }
          }} />
        </Field>
        <Field label="Name (optional)">
          <input className={inputCls} placeholder="coder-0.5b" value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        {backend === "vllm" && <Field label="Task">
          <select className={inputCls} value={task} onChange={(e) => setTask(e.target.value as "instruct" | "embed")}>
            <option value="instruct">instruct (chat)</option>
            <option value="embed">embed (embeddings)</option>
          </select>
        </Field>}
        {backend === "vllm" && <Field label="Quantization">
          <select className={inputCls} value={quant} onChange={(e) => setQuant(e.target.value)}>
            <option value="fp16">FP16</option>
            <option value="fp8">FP8</option>
            <option value="awq">AWQ</option>
            <option value="gptq">GPTQ</option>
          </select>
        </Field>}
        {backend === "vllm" && <Field label="GPU memory utilization (0–1)">
          <input className={inputCls} value={gpuUtil} onChange={(e) => setGpuUtil(e.target.value)} />
        </Field>}
        {backend === "llamacpp" && (
          <>
            {!moePresetApplied && (
              <div className="sm:col-span-2 flex items-center justify-between rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
                <div>
                  <div className="text-sm font-medium text-indigo-200">MoE with CPU expert offload</div>
                  <div className="text-xs text-slate-400">Starting point for large GGUF MoE models on limited VRAM.</div>
                </div>
                <Button
                  variant="subtle"
                  onClick={() => {
                    setNGpuLayers("99");
                    setNCpuMoe("24");
                    setCtxSize("32768");
                    setFlashAttn(true);
                    setJinja(true);
                    setMoePresetApplied(true);
                  }}
                >
                  Apply preset
                </Button>
              </div>
            )}
            <Field label="Context size">
              <input className={inputCls} value={ctxSize} onChange={(e) => setCtxSize(e.target.value)} />
            </Field>
            <Field label="GPU layers (-ngl)">
              <input className={inputCls} value={nGpuLayers} onChange={(e) => setNGpuLayers(e.target.value)} />
            </Field>
            <Field label="CPU MoE layers">
              <input className={inputCls} value={nCpuMoe} onChange={(e) => setNCpuMoe(e.target.value)} />
            </Field>
            <Field label="mmproj path (optional)">
              <input className={inputCls} value={mmprojPath} onChange={(e) => setMmprojPath(e.target.value)} />
            </Field>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={flashAttn} onChange={(e) => setFlashAttn(e.target.checked)} /> Flash attention
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={jinja} onChange={(e) => setJinja(e.target.checked)} /> Jinja tool-calling templates
            </label>
          </>
        )}
        <Field
          label="Max model len (blank = auto)"
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
                ? (initialMaxLen && initialMaxLen > vramContextLimit!
                    ? `auto (~${fmtNum(vramContextLimit!)} VRAM ➔ ~${fmtNum(initialMaxLen)} RAM)`
                    : `auto (~${fmtNum(vramContextLimit!)} VRAM)`)
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
                RAM Swap: {initialMaxLen && initialMaxLen > vramContextLimit! ? `up to ~${fmtNum(initialMaxLen)} tokens` : `>${fmtNum(vramContextLimit!)}`}
              </span>
              {maxLen && parseInt(maxLen, 10) > vramContextLimit! && (
                <span className="text-[11px] font-semibold text-cyan-300 ml-auto">
                  (Uses RAM swap)
                </span>
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
        <Field
          label="RAM Swap Space (GB)"
          hint="vLLM RAM offload (--kv-offloading-size). Allocates system RAM for spilled KV cache blocks."
        >
          <input
            type="number"
            step="1"
            min="0"
            className={inputCls}
            placeholder="0 (e.g. 16, 32)"
            value={swapSpaceGb}
            onChange={(e) => setSwapSpaceGb(e.target.value)}
          />
        </Field>
        <Field
          label="CPU Weight Offload (GB)"
          hint="vLLM --cpu-offload-gb. Offloads model parameter weights to CPU RAM."
        >
          <input
            type="number"
            step="1"
            min="0"
            className={inputCls}
            placeholder="0 (e.g. 8)"
            value={cpuOffloadGb}
            onChange={(e) => setCpuOffloadGb(e.target.value)}
          />
        </Field>
        <Field label="Served model name (optional)">
          <input className={inputCls} placeholder="blank = model id" value={served} onChange={(e) => setServed(e.target.value)} />
        </Field>
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
        <Button onClick={submit} disabled={creating || !modelId.trim()}>
          {creating ? <Spinner label="creating…" /> : "Create"}
        </Button>
      </div>
    </Card>
  );
}

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
                    ? "bg-indigo-500/15 text-indigo-200 border border-indigo-500/30"
                    : "text-slate-400 hover:bg-surface-3 hover:text-slate-200"
                }`}
              >
                {editingTitleId === c.id ? (
                  <input
                    autoFocus
                    className="w-full rounded bg-surface-1 px-1.5 py-0.5 text-xs text-slate-100 border border-indigo-500 outline-hidden"
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
                    ? "border-indigo-500/50 bg-indigo-500/10 text-indigo-200"
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
                  className="w-48 accent-indigo-500"
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
                  className="flex-1 rounded-md border border-edge bg-surface-1 p-2 text-xs text-slate-200 outline-hidden focus:border-indigo-500"
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
                        ? "bg-indigo-600 text-white rounded-br-xs"
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
              <div className="flex items-center gap-2 text-xs text-indigo-300">
                <span className="inline-block h-2 w-2 animate-ping rounded-full bg-indigo-400" />
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
            <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-3 text-xs text-indigo-200 flex items-center justify-between">
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
                        <span className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
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