import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  evaluateSystemFit,
  events,
  fmtContext,
  fmtNum,
  fmtTokPerSec,
  quantLabel,
} from "../api";
import { Badge, Button, inputCls, Spinner } from "../ui";
import type { EnvStatus, ModelWithStats, PullStatus } from "../types";
import { RECOMMENDED_MODELS, type RecommendedModel } from "../data/recommended";

export default function Search() {
  const [query, setQuery] = useState("");
  const [quant, setQuant] = useState("fp16");
  const [results, setResults] = useState<ModelWithStats[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [onlyRecommended, setOnlyRecommended] = useState<boolean>(false);
  const seq = useRef(0);

  useEffect(() => {
    api.envStatus().then(setEnv).catch(() => {});
    const unsub = events.pullProgress((p) => {
      setPulls((prev) => ({ ...prev, [p.model]: p }));
    });
    return () => {
      unsub.then((f) => f());
    };
  }, []);

  const totalVramMb = env?.gpu?.vram_total_mb ?? null;
  const bandwidth = env?.gpu_bandwidth_gbs ?? 700;

  const doSearch = async (searchQuery?: string) => {
    const q = (searchQuery !== undefined ? searchQuery : query).trim();
    if (!q) {
      setResults(null);
      return;
    }
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

  const selectRecommended = (m: RecommendedModel) => {
    setQuery(m.id);
    doSearch(m.id);
  };

  const pull = (modelId: string) => {
    api.pullModel(modelId).catch((e) => setErr(String(e)));
  };

  // Filter curated models according to active category
  const filteredRecommendations = useMemo(() => {
    return RECOMMENDED_MODELS.filter((m) => {
      if (activeCategory !== "all" && m.category !== activeCategory) return false;
      return true;
    });
  }, [activeCategory]);

  // Compute stats and fit for recommendations
  const recommendationRows = useMemo(() => {
    return filteredRecommendations.map((m) => {
      const fit = evaluateSystemFit(m.params_b, quant, totalVramMb);
      // Rough tokens/sec calculation for recommendations: bw * 1e9 * 0.5 / (params * 1e9 * bpp)
      const bpp = quant === "fp8" || quant === "int8" ? 1.0 : quant === "awq" || quant === "gptq" ? 1.1 : 2.0;
      const estTokS = m.params_b > 0 ? (bandwidth * 1e9 * 0.5) / (m.params_b * 1e9 * bpp) : null;
      return {
        ...m,
        fit,
        estTokS,
      };
    });
  }, [filteredRecommendations, quant, totalVramMb, bandwidth]);

  // Process search results with fit evaluation
  const evaluatedResults = useMemo(() => {
    if (!results) return null;
    const list = results.map((m) => {
      const fit = evaluateSystemFit(m.params_b, quant, totalVramMb);
      return { ...m, fit };
    });
    if (onlyRecommended) {
      return list.filter((m) => m.fit.rating === "optimal" || m.fit.rating === "tight");
    }
    return list;
  }, [results, quant, totalVramMb, onlyRecommended]);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Model Search & Discovery</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Discover and pull AI models hardware-tailored for vLLM & your GPU.
          </p>
        </div>
        {env?.gpu && (
          <div className="flex items-center gap-2 rounded-lg border border-edge bg-surface-2 px-3 py-1.5 text-xs text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold">{env.gpu.name}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{(env.gpu.vram_total_mb / 1024).toFixed(1)} GB VRAM</span>
            <span className="text-slate-500">|</span>
            <span className="text-cyan-300 font-mono">{env.gpu_bandwidth_gbs} GB/s</span>
          </div>
        )}
      </div>

      {/* Search Input Bar */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            className={`${inputCls} pl-9`}
            placeholder="Search Hugging Face (e.g. Qwen2.5, Llama-3.1, DeepSeek-R1, bge)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
          />
          <svg
            className="absolute left-3 top-2.5 h-4 w-4 text-slate-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          {query && (
            <button
              onClick={() => {
                setQuery("");
                setResults(null);
              }}
              className="absolute right-3 top-2.5 text-xs text-slate-500 hover:text-slate-300"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>
        <select
          className={`${inputCls} w-36`}
          value={quant}
          onChange={(e) => setQuant(e.target.value)}
          title="Assumed precision / quantization"
        >
          <option value="fp16">FP16 / BF16</option>
          <option value="fp8">FP8 (1-byte)</option>
          <option value="awq">AWQ (4-bit)</option>
          <option value="gptq">GPTQ (4-bit)</option>
        </select>
        <Button onClick={() => doSearch()} disabled={searching || !query.trim()}>
          {searching ? <Spinner label="searching…" /> : "Search"}
        </Button>
      </div>

      {/* Quick category filters */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-slate-500 font-medium mr-1">Categories:</span>
          {(
            [
              ["all", "All"],
              ["chat", "Chat & Instruct"],
              ["coding", "Coding"],
              ["reasoning", "Reasoning (R1)"],
              ["embedding", "Embeddings"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setActiveCategory(key)}
              className={`rounded-full px-2.5 py-1 transition-colors ${
                activeCategory === key
                  ? "bg-indigo-600 text-white font-medium"
                  : "bg-surface-2 border border-edge text-slate-400 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {results !== null && (
          <label className="flex items-center gap-2 text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={onlyRecommended}
              onChange={(e) => setOnlyRecommended(e.target.checked)}
              className="rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0"
            />
            <span>Show only hardware-compatible models</span>
          </label>
        )}
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
          {err}
        </div>
      )}

      {/* Recommended Models Section (Visible when no active search query or user is browsing) */}
      {results === null && !searching && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-200">Recommended for Your System</h2>
              <p className="text-xs text-slate-500">
                Models curated for vLLM compatibility, scored against your GPU's VRAM ({totalVramMb ? `${(totalVramMb / 1024).toFixed(1)} GB` : "detected"}) and bandwidth at {quantLabel(quant)}.
              </p>
            </div>
            <span className="text-xs text-slate-500">
              {recommendationRows.length} models
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {recommendationRows.map((m) => {
              const pullState = pulls[m.id];
              return (
                <div
                  key={m.id}
                  className="flex flex-col justify-between rounded-xl border border-edge bg-surface-2 p-4 transition-all hover:border-slate-600 hover:bg-surface-2/90"
                >
                  <div className="space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="font-semibold text-slate-200 text-sm">{m.name}</div>
                        <div className="text-[11px] font-mono text-slate-500 truncate max-w-[210px]" title={m.id}>
                          {m.id}
                        </div>
                      </div>
                      <Badge color={m.fit.badgeColor}>{m.fit.label}</Badge>
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed min-h-[36px]">
                      {m.description}
                    </p>

                    <div className="grid grid-cols-3 gap-2 rounded-lg bg-surface-3/50 p-2 text-center text-xs">
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Params</div>
                        <div className="font-medium text-slate-300">{m.params_b}B</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Context</div>
                        <div className="font-medium text-slate-300">{fmtContext(m.context)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Est. Speed</div>
                        <div className="font-mono text-cyan-300">{fmtTokPerSec(m.estTokS)}</div>
                      </div>
                    </div>

                    {m.fit.reason && (
                      <div className="text-[11px] text-slate-500 line-clamp-2">
                        {m.fit.reason}
                      </div>
                    )}
                  </div>

                  <div className="mt-3.5 pt-3 border-t border-edge/60 flex items-center justify-between gap-2">
                    <button
                      onClick={() => selectRecommended(m)}
                      className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-2 hover:underline"
                    >
                      Inspect HF Details
                    </button>
                    {pullState ? (
                      <Badge color={pullState.state === "complete" ? "emerald" : pullState.state === "failed" ? "red" : "indigo"}>
                        {pullState.state}
                      </Badge>
                    ) : (
                      <Button variant="ghost" onClick={() => pull(m.id)}>
                        Pull Model
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Search Results Table */}
      {results !== null && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300">
              Search Results ({evaluatedResults?.length ?? 0} found)
            </h2>
            <button
              onClick={() => {
                setQuery("");
                setResults(null);
              }}
              className="text-xs text-indigo-400 hover:underline"
            >
              ← Back to Recommended Models
            </button>
          </div>

          <div className="overflow-x-auto rounded-xl border border-edge">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-edge bg-surface-2 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2.5">Model</th>
                  <th className="px-3 py-2.5 text-center">System Fit</th>
                  <th className="px-3 py-2.5 text-right">Params</th>
                  <th className="px-3 py-2.5 text-right">Context</th>
                  <th className="px-3 py-2.5 text-right">Est. Speed</th>
                  <th className="px-3 py-2.5 text-right">Downloads</th>
                  <th className="px-3 py-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {evaluatedResults?.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-8 text-center text-sm text-slate-500">
                      No models found matching your criteria. Try loosening filters or searching another keyword.
                    </td>
                  </tr>
                ) : (
                  evaluatedResults?.map((m) => {
                    const pullState = pulls[m.id];
                    return (
                      <tr
                        key={m.id}
                        className="border-b border-edge/60 last:border-0 hover:bg-surface-2/60 transition-colors"
                      >
                        <td className="px-3 py-2.5 max-w-[280px]">
                          <div className="font-medium text-slate-200 truncate" title={m.id}>
                            {m.id}
                          </div>
                          <div className="flex items-center gap-2 text-[11px] text-slate-500">
                            {m.pipeline_tag && <span>{m.pipeline_tag}</span>}
                            {m.params_b == null && <Badge color="amber">params missing</Badge>}
                          </div>
                        </td>
                        <td className="px-3 py-2.5 text-center">
                          <div className="inline-flex flex-col items-center">
                            <Badge color={m.fit.badgeColor}>{m.fit.label}</Badge>
                            {m.fit.vramPct !== null && (
                              <span className="text-[10px] text-slate-500 mt-0.5">
                                ~{m.fit.vramPct}% VRAM
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-3 py-2.5 text-right text-slate-300">
                          {m.params_b != null ? `${m.params_b.toFixed(2)}B` : "—"}
                          {m.fit.estWeightGb !== null && (
                            <div className="text-[10px] text-slate-500">
                              ~{m.fit.estWeightGb.toFixed(1)} GB
                            </div>
                          )}
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
                            <Badge
                              color={
                                pullState.state === "complete"
                                  ? "emerald"
                                  : pullState.state === "failed"
                                  ? "red"
                                  : "indigo"
                              }
                            >
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
                  })
                )}
              </tbody>
            </table>
          </div>
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