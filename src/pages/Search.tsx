import { useEffect, useRef, useState } from "react";
import { api, events, fmtContext, fmtNum, fmtTokPerSec, quantLabel } from "../api";
import { Badge, Button, inputCls, Spinner } from "../ui";
import type { ModelWithStats, PullStatus } from "../types";

export default function Search() {
  const [query, setQuery] = useState("Qwen2.5-0.5B");
  const [quant, setQuant] = useState("fp16");
  const [results, setResults] = useState<ModelWithStats[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [env, setEnv] = useState<{ bandwidth: number; known: boolean } | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    api.envStatus().then((e) => setEnv({ bandwidth: e.gpu_bandwidth_gbs, known: e.gpu_bw_known }));
    const unsub = events.pullProgress((p) => {
      setPulls((prev) => ({ ...prev, [p.model]: p }));
    });
    return () => {
      unsub.then((f) => f());
    };
  }, []);

  const doSearch = async () => {
    const q = query.trim();
    if (!q) return;
    const id = ++seq.current;
    setSearching(true);
    setErr(null);
    try {
      const res = await api.searchModels(q, quant);
      if (id === seq.current) setResults(res);
    } catch (e) {
      if (id === seq.current) setErr(String(e));
    } finally {
      if (id === seq.current) setSearching(false);
    }
  };

  const pull = (modelId: string) => {
    api.pullModel(modelId).catch((e) => setErr(String(e)));
  };

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      <h1 className="text-xl font-bold text-slate-100">Model Search</h1>

      <div className="flex gap-2">
        <input
          className={inputCls}
          placeholder="e.g. Qwen2.5-0.5B, llama-3, bge-small…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && doSearch()}
        />
        <select className={`${inputCls} w-36`} value={quant} onChange={(e) => setQuant(e.target.value)}>
          <option value="fp16">FP16</option>
          <option value="fp8">FP8</option>
          <option value="awq">AWQ</option>
          <option value="gptq">GPTQ</option>
        </select>
        <Button onClick={doSearch} disabled={searching || !query.trim()}>
          {searching ? <Spinner label="searching…" /> : "Search"}
        </Button>
      </div>

      {env && (
        <div className="text-xs text-slate-500">
          Estimates for your GPU assume <span className="text-slate-300">{env.bandwidth} GB/s</span>
          {!env.known && " (default — GPU unknown)"} and {quantLabel(quant)} weights. Measured tok/s from a live server wins.
        </div>
      )}

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{err}</div>}

      {results === null && !searching ? (
        <div className="rounded-xl border border-dashed border-edge p-16 text-center text-sm text-slate-600">
          Search Hugging Face for models. The table shows {quantLabel(quant)}-quantized estimates for your GPU:
          params, context (config.json or family default), and estimated max decode tok/s.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-edge">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-edge bg-surface-2 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2.5">Model</th>
                <th className="px-3 py-2.5 text-right">Params</th>
                <th className="px-3 py-2.5 text-right">Context</th>
                <th className="px-3 py-2.5 text-right">Est. tok/s</th>
                <th className="px-3 py-2.5 text-right">Downloads</th>
                <th className="px-3 py-2.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {results?.map((m) => {
                const pullState = pulls[m.id];
                return (
                  <tr key={m.id} className="border-b border-edge/60 last:border-0 hover:bg-surface-2/60">
                    <td className="px-3 py-2.5">
                      <div className="font-medium text-slate-200">{m.id}</div>
                      <div className="flex items-center gap-2 text-[11px] text-slate-500">
                        {m.pipeline_tag && <span>{m.pipeline_tag}</span>}
                        {m.params_b == null && <Badge color="amber">estimate missing</Badge>}
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right text-slate-300">
                      {m.params_b != null ? `${m.params_b.toFixed(2)}B` : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <span className="text-slate-300">{fmtContext(m.context)}</span>
                      {m.context_estimated && (
                        <div className="text-[10px] text-slate-600">~ {m.context_source}</div>
                      )}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      {m.max_tok_s != null ? (
                        <span className="font-mono text-cyan-300">{fmtTokPerSec(m.max_tok_s)}</span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="px-3 py-2.5 text-right text-slate-400">{fmtNum(m.downloads)}</td>
                    <td className="px-3 py-2.5 text-right">
                      {pullState ? (
                        <Badge color={pullState.state === "complete" ? "emerald" : pullState.state === "failed" ? "red" : "indigo"}>
                          {pullState.state}
                        </Badge>
                      ) : (
                        <Button variant="ghost" onClick={() => pull(m.id)}>
                          Pull
                        </Button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {Object.entries(pulls).filter(([, p]) => p.state === "downloading").length > 0 && (
          <div className="space-y-1 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3 text-xs text-slate-400">
            {Object.entries(pulls)
              .filter(([, p]) => p.state === "downloading")
              .map(([m, p]) => (
                <div key={m} className="truncate">
                  <span className="text-indigo-300">{m}</span>: {p.file}
                </div>
              ))}
          </div>
        )}
    </div>
  );
}