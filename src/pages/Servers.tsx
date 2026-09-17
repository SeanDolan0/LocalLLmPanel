import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { api, events, fmtNum, fmtTokPerSec, quantLabel, statusColor } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls, Spinner } from "../ui";
import { effectiveModelName } from "../types";
import type { CreateServerInput, ServerListRow } from "../types";

export default function Servers() {
  const location = useLocation();
  const [rows, setRows] = useState<ServerListRow[]>([]);
  const [showNew, setShowNew] = useState(false);
  const [prefillModel, setPrefillModel] = useState<string | null>(null);
  const [prefillQuant, setPrefillQuant] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // server id being start/stop/delete
  // log buffers per server (event-driven + hydrated)
  const [logs, setLogs] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const logEndRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Check navigation state for prefill model and quant (e.g. from Search or Library Deploy)
  useEffect(() => {
    const state = location.state as { prefillModel?: string; prefillQuant?: string } | null;
    if (state?.prefillModel) {
      setPrefillModel(state.prefillModel);
      if (state.prefillQuant) {
        setPrefillQuant(state.prefillQuant);
      }
      setShowNew(true);
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const refresh = useCallback(() => {
    api.serversList().then(setRows).catch(() => {});
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

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-100">Servers</h1>
        <Button onClick={() => setShowNew((s) => !s)}>{showNew ? "Cancel" : "+ New server"}</Button>
      </div>

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{err}</div>}

      {showNew && (
        <NewServerForm
          initialModelId={prefillModel ?? ""}
          initialQuant={prefillQuant ?? undefined}
          onDone={(s) => {
            setShowNew(false);
            setPrefillModel(null);
            setPrefillQuant(null);
            setSelected(s.id);
            refresh();
          }}
          onCancel={() => {
            setShowNew(false);
            setPrefillModel(null);
            setPrefillQuant(null);
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
                  <div className="mt-1 truncate text-xs text-slate-500 group-hover:text-slate-400">
                    {r.def.model_id} · port {r.def.port} · {r.def.task} · {quantLabel(r.def.quant)}
                    {r.def.max_model_len ? ` · ctx ${fmtNum(r.def.max_model_len)}` : ""}
                    {r.def.params_b ? ` · ~${r.def.params_b.toFixed(2)}B` : ""}
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
                  {r.status === "running" && (
                    <Button variant="ghost" disabled={busy === r.def.id} onClick={() => act(r.def.id, () => api.serversRestart(r.def.id))}>
                      Restart
                    </Button>
                  )}
                  <Button variant="subtle" disabled={busy === r.def.id} onClick={() => { if (confirm(`Delete server "${r.def.name}"?`)) act(r.def.id, () => api.serversDelete(r.def.id)); }}>
                    ✕
                  </Button>
                </div>
              </div>
              {r.error && <div className="mt-2 text-xs text-red-300">{r.error}</div>}
              {r.metrics && r.status === "running" && (
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-500 sm:grid-cols-4">
                  <div>gen tok: <span className="text-slate-300">{fmtNum(r.metrics.total_generation_tokens)}</span></div>
                  <div>prompt tok: <span className="text-slate-300">{fmtNum(r.metrics.total_prompt_tokens)}</span></div>
                  <div>running: <span className="text-slate-300">{r.metrics.running}</span> · waiting <span className="text-slate-300">{r.metrics.waiting}</span></div>
                  <div>requests: <span className="text-slate-300">{fmtNum(r.metrics.requests)}</span></div>
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
                    <ChatButton serverId={selectedRow.def.id} port={selectedRow.def.port} model={effectiveModelName(selectedRow.def)} />
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
              <div className="grid grid-cols-2 gap-3 text-sm">
                <Stat label="Generation tok/s" value={selectedRow.metrics.measured?.tokens_per_sec != null ? fmtTokPerSec(selectedRow.metrics.measured.tokens_per_sec) : "– (needs traffic)"} />
                <Stat label="Prompt tok/s" value={selectedRow.metrics.measured?.prompt_tokens_per_sec != null ? fmtTokPerSec(selectedRow.metrics.measured.prompt_tokens_per_sec) : "–"} />
                <Stat label="Total gen tokens" value={fmtNum(selectedRow.metrics.total_generation_tokens)} />
                <Stat label="Total prompt tokens" value={fmtNum(selectedRow.metrics.total_prompt_tokens)} />
                <Stat label="In-flight / waiting" value={`${selectedRow.metrics.running} / ${selectedRow.metrics.waiting}`} />
                <Stat label="Requests" value={fmtNum(selectedRow.metrics.requests)} />
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
  initialModelId = "",
  initialQuant,
  onDone,
  onCancel,
  onErr,
}: {
  initialModelId?: string;
  initialQuant?: string;
  onDone: (s: { id: string }) => void;
  onCancel: () => void;
  onErr: (e: string) => void;
}) {
  const [modelId, setModelId] = useState(initialModelId);
  const [name, setName] = useState(initialModelId ? initialModelId.split("/").pop() || "" : "");
  const [task, setTask] = useState<"instruct" | "embed">("instruct");
  const [quant, setQuant] = useState(initialQuant || "fp16");
  const [gpuUtil, setGpuUtil] = useState("0.92");
  const [maxLen, setMaxLen] = useState("");
  const [served, setServed] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (initialModelId) {
      setModelId(initialModelId);
      setName(initialModelId.split("/").pop() || "");
    }
    if (initialQuant) {
      setQuant(initialQuant);
    }
  }, [initialModelId, initialQuant]);

  const submit = async () => {
    if (!modelId.trim()) return;
    setCreating(true);
    try {
      const input: CreateServerInput = {
        name: name.trim() || modelId.split("/").pop() || "server",
        model_id: modelId.trim(),
        task,
        quant,
        gpu_mem_util: parseFloat(gpuUtil) || 0.92,
        max_model_len: maxLen ? parseInt(maxLen) || undefined : undefined,
        served_model_name: served.trim() || undefined,
      };
      const def = await api.serversCreate(input);
      onDone(def);
    } catch (e) {
      onErr(String(e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <Card className="border-indigo-500/30">
      <CardTitle>Define new server</CardTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Model id (HF)">
          <input className={inputCls} placeholder="Qwen/Qwen2.5-0.5B-Instruct" value={modelId} onChange={(e) => setModelId(e.target.value)} />
        </Field>
        <Field label="Name (optional)">
          <input className={inputCls} placeholder="coder-0.5b" value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Task">
          <select className={inputCls} value={task} onChange={(e) => setTask(e.target.value as "instruct" | "embed")}>
            <option value="instruct">instruct (chat)</option>
            <option value="embed">embed (embeddings)</option>
          </select>
        </Field>
        <Field label="Quantization">
          <select className={inputCls} value={quant} onChange={(e) => setQuant(e.target.value)}>
            <option value="fp16">FP16</option>
            <option value="fp8">FP8</option>
            <option value="awq">AWQ</option>
            <option value="gptq">GPTQ</option>
          </select>
        </Field>
        <Field label="GPU memory utilization (0–1)">
          <input className={inputCls} value={gpuUtil} onChange={(e) => setGpuUtil(e.target.value)} />
        </Field>
        <Field label="Max model len (blank = auto)">
          <input className={inputCls} placeholder="auto (context ∩ VRAM fit)" value={maxLen} onChange={(e) => setMaxLen(e.target.value)} />
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

function ChatButton({ serverId, port, model }: { serverId: string; port: number; model: string }) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<{ role: string; content: string }[]>([
    { role: "system", content: "You are a helpful assistant." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [msgs, open]);

  const send = async () => {
    if (!input.trim() || busy) return;
    const next: { role: string; content: string }[] = [...msgs, { role: "user", content: input }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    try {
      const resp = await api.serversChat(serverId, next);
      const content = (resp.choices as { message?: { content?: string } }[])?.[0]?.message?.content ?? "";
      setMsgs((m) => [...m, { role: "assistant", content }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: `⚠ ${String(e)}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button variant="ghost" onClick={() => setOpen((o) => !o)}>💬 Chat</Button>
      {open && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-6" onClick={() => setOpen(false)}>
          <div
            className="flex h-[70vh] w-full max-w-2xl flex-col rounded-xl border border-edge bg-surface-2 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-edge px-4 py-2.5">
              <div className="text-sm font-medium text-slate-200">
                Playground · {model} <span className="text-slate-500">(port {port})</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="rounded border border-edge bg-surface-3 px-2 py-0.5 text-xs text-slate-400 hover:text-slate-200"
                  onClick={() => setMsgs([{ role: "system", content: "You are a helpful assistant." }])}
                  title="Reset conversation"
                >
                  Clear chat
                </button>
                <button className="text-slate-500 hover:text-slate-300" onClick={() => setOpen(false)}>✕</button>
              </div>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              {msgs.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                      m.role === "user" ? "bg-indigo-500/20 text-indigo-100" : m.role === "system" ? "bg-slate-800/60 text-slate-400 italic" : "bg-surface-3 text-slate-200"
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
              {busy && <Spinner label="generating…" />}
              <div ref={endRef} />
            </div>
            <div className="flex gap-2 border-t border-edge p-3">
              <input
                className={inputCls}
                placeholder="Message… (Enter to send)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                disabled={busy}
              />
              <Button onClick={send} disabled={busy || !input.trim()}>Send</Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}