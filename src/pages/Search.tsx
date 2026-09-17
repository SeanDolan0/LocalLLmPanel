import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  events,
  fmtContext,
  fmtNum,
  fmtTokPerSec,
} from "../api";
import { Badge, Button, inputCls, Spinner } from "../ui";
import type {
  EnvStatus,
  FitResultBackend,
  FitVerdict,
  ModelWithFit,
  PullStatus,
  QuantVariantWithFit,
  RunMode,
} from "../types";

// ---------------------------------------------------------------------------
// Inline debounce hook (no new external dependencies)
// ---------------------------------------------------------------------------
function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

// ---------------------------------------------------------------------------
// Verdict Badge Component
// ---------------------------------------------------------------------------
export function FitVerdictBadge({ verdict, score }: { verdict: FitVerdict; score?: number }) {
  let cls = "";
  let label = "";
  switch (verdict) {
    case "Comfortable":
      cls = "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
      label = "Comfortable";
      break;
    case "Constrained":
      cls = "border-amber-500/30 bg-amber-500/10 text-amber-300";
      label = "Constrained";
      break;
    case "DoesNotFit":
      cls = "border-rose-500/30 bg-rose-500/10 text-rose-300";
      label = "Does Not Fit";
      break;
  }
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {score !== undefined ? `${score}/100 · ${label}` : label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// RunMode Badge Component
// ---------------------------------------------------------------------------
export function RunModeBadge({ mode }: { mode: RunMode }) {
  let cls = "";
  let label = "";
  switch (mode) {
    case "Gpu":
      cls = "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
      label = "GPU";
      break;
    case "GpuRamSwap":
      cls = "border-cyan-500/30 bg-cyan-500/10 text-cyan-300";
      label = "GPU + RAM Swap";
      break;
    case "CpuOffload":
      cls = "border-amber-500/30 bg-amber-500/10 text-amber-300";
      label = "CPU Offload";
      break;
    case "DoesNotFit":
      cls = "border-rose-500/30 bg-rose-500/10 text-rose-300";
      label = "Does Not Fit";
      break;
  }
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Dual Context Badge Component
// ---------------------------------------------------------------------------
export function ContextBadge({ fit }: { fit: FitResultBackend }) {
  if (fit.vram_context === 0 && fit.extended_context === 0) {
    return <span className="text-xs text-neutral-500">—</span>;
  }
  if (fit.run_mode === "GpuRamSwap") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs bg-cyan-950/80 text-cyan-300 border border-cyan-800">
        <span className="font-semibold">{Math.round(fit.vram_context / 1000)}k VRAM</span>
        <span>➔</span>
        <span className="font-bold text-cyan-200">{Math.round(fit.extended_context / 1000)}k RAM</span>
      </span>
    );
  }
  return (
    <span className="text-xs text-neutral-300">
      {Math.round((fit.vram_context || fit.extended_context) / 1000)}k {fit.run_mode === "CpuOffload" ? "RAM" : "VRAM"}
    </span>
  );
}

// Helper to get best variant safely
function getBestVariant(m: ModelWithFit): QuantVariantWithFit | undefined {
  if (!m.variants || m.variants.length === 0) return undefined;
  if (m.best_variant_idx >= 0 && m.best_variant_idx < m.variants.length) {
    return m.variants[m.best_variant_idx];
  }
  return m.variants[0];
}

const CATEGORIES = [
  ["all", "All"],
  ["chat", "Chat & Instruct"],
  ["coding", "Coding"],
  ["reasoning", "Reasoning (R1)"],
  ["embedding", "Embeddings"],
] as const;

function matchesCategory(m: ModelWithFit, cat: string): boolean {
  if (cat === "all") return true;
  const idLower = m.id.toLowerCase();
  if (cat === "embedding") {
    return (
      m.pipeline_tag === "feature-extraction" ||
      idLower.includes("embed") ||
      idLower.includes("bge") ||
      idLower.includes("gte")
    );
  }
  if (cat === "coding") {
    return (
      idLower.includes("coder") ||
      idLower.includes("code") ||
      idLower.includes("starcoder") ||
      idLower.includes("deepseek-coder")
    );
  }
  if (cat === "reasoning") {
    return (
      idLower.includes("r1") ||
      idLower.includes("reason") ||
      idLower.includes("qwq") ||
      idLower.includes("thinking")
    );
  }
  if (cat === "chat") {
    return (
      idLower.includes("instruct") ||
      idLower.includes("chat") ||
      (!idLower.includes("coder") && !idLower.includes("r1") && m.pipeline_tag !== "feature-extraction")
    );
  }
  return true;
}

export default function Search() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 500);

  const [results, setResults] = useState<ModelWithFit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchErr, setSearchErr] = useState<string | null>(null);

  const [recommended, setRecommended] = useState<ModelWithFit[]>([]);
  const [loadingRecs, setLoadingRecs] = useState(true);
  const [recsErr, setRecsErr] = useState<string | null>(null);

  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [onlyRecommended, setOnlyRecommended] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [selectedModel, setSelectedModel] = useState<ModelWithFit | null>(null);

  const seq = useRef(0);
  const lastSearchedQuery = useRef<string>("");

  useEffect(() => {
    api.envStatus().then(setEnv).catch(() => {});
    const unsub = events.pullProgress((p) => {
      setPulls((prev) => ({ ...prev, [p.model]: p }));
    });
    return () => {
      unsub.then((f) => f());
    };
  }, []);

  // Fetch dynamic recommendations on component mount
  useEffect(() => {
    let active = true;
    setLoadingRecs(true);
    setRecsErr(null);
    api
      .recommendedModels()
      .then((data) => {
        if (active) {
          setRecommended(data);
          setLoadingRecs(false);
        }
      })
      .catch((e) => {
        if (active) {
          setRecsErr(String(e));
          setLoadingRecs(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  // Auto-search logic
  const doSearch = async (searchTerm: string) => {
    const q = searchTerm.trim();
    if (q.length < 2) {
      setResults(null);
      setSearching(false);
      lastSearchedQuery.current = "";
      return;
    }
    lastSearchedQuery.current = q;
    const id = ++seq.current;
    setSearching(true);
    setSearchErr(null);
    try {
      const res = await api.searchModelsWithFit(q);
      if (id === seq.current) {
        setResults(res);
      }
    } catch (e) {
      if (id === seq.current) {
        setSearchErr(String(e));
      }
    } finally {
      if (id === seq.current) {
        setSearching(false);
      }
    }
  };

  // Debounced query effect
  useEffect(() => {
    const trimmed = debouncedQuery.trim();
    if (trimmed.length < 2) {
      if (results !== null) {
        setResults(null);
      }
      lastSearchedQuery.current = "";
    } else if (trimmed !== lastSearchedQuery.current) {
      doSearch(trimmed);
    }
  }, [debouncedQuery]);

  const pull = (repoId: string) => {
    api.pullModel(repoId).catch((e) => setSearchErr(String(e)));
  };

  const deploy = (repoId: string, quant: string, fit?: FitResultBackend) => {
    navigate("/servers", {
      state: {
        prefillModel: repoId,
        prefillQuant: quant,
        prefillSwapSpace: fit?.swap_space_gb,
        prefillCpuOffload: fit?.cpu_offload_gb,
        prefillMaxLen: fit?.extended_context,
        prefillVramContext: fit?.vram_context,
      },
    });
  };

  const selectCategory = (key: string) => {
    setActiveCategory(key);
  };

  const clearSearch = () => {
    setQuery("");
    setResults(null);
    lastSearchedQuery.current = "";
    setSearchErr(null);
  };

  // Filter recommendations
  const filteredRecommendations = useMemo(() => {
    return recommended.filter((m) => {
      if (!matchesCategory(m, activeCategory)) return false;
      if (onlyRecommended) {
        const best = getBestVariant(m);
        if (!best || best.fit.verdict === "DoesNotFit") return false;
      }
      return true;
    });
  }, [recommended, activeCategory, onlyRecommended]);

  // Filter search results
  const filteredResults = useMemo(() => {
    if (!results) return null;
    return results.filter((m) => {
      if (!matchesCategory(m, activeCategory)) return false;
      if (onlyRecommended) {
        const best = getBestVariant(m);
        if (!best || best.fit.verdict === "DoesNotFit") return false;
      }
      return true;
    });
  }, [results, activeCategory, onlyRecommended]);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Model Search & Discovery</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Discover models and automatically find the optimal quantization tailored for your hardware.
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

      {/* Streamlined Search Input Bar with debounced auto-search & instant Enter */}
      <div className="relative">
        <input
          className={`${inputCls} pl-9 pr-16`}
          placeholder="Search Hugging Face models (e.g. Qwen2.5, Llama-3.1, DeepSeek-R1, bge)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const q = query.trim();
              if (q.length >= 2) {
                doSearch(q);
              } else {
                setResults(null);
                lastSearchedQuery.current = "";
              }
            }
          }}
        />
        <svg
          className="absolute left-3 top-2.5 h-4 w-4 text-slate-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <div className="absolute right-3 top-2 flex items-center gap-2">
          {searching && <Spinner label="" />}
          {query && (
            <button
              onClick={clearSearch}
              className="text-xs text-slate-500 hover:text-slate-300 p-0.5 rounded"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Quick category filters & compatibility toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-slate-500 font-medium mr-1">Categories:</span>
          {CATEGORIES.map(([key, label]) => (
            <button
              key={key}
              onClick={() => selectCategory(key)}
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

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={onlyRecommended}
              onChange={(e) => setOnlyRecommended(e.target.checked)}
              className="rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0"
            />
            <span>Show only compatible (Comfortable / Constrained)</span>
          </label>

          {results !== null && (
            <div className="flex items-center rounded-lg border border-edge bg-surface-2 p-0.5">
              <button
                onClick={() => setViewMode("grid")}
                className={`rounded px-2 py-0.5 text-xs transition-colors ${
                  viewMode === "grid" ? "bg-indigo-600 text-white font-medium" : "text-slate-400 hover:text-slate-200"
                }`}
                title="Grid card view"
              >
                Cards
              </button>
              <button
                onClick={() => setViewMode("table")}
                className={`rounded px-2 py-0.5 text-xs transition-colors ${
                  viewMode === "table" ? "bg-indigo-600 text-white font-medium" : "text-slate-400 hover:text-slate-200"
                }`}
                title="Table list view"
              >
                Table
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Error Banners */}
      {searchErr && (
        <div className="flex items-center justify-between rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
          <span>{searchErr}</span>
          {(searchErr.includes("429") || searchErr.toLowerCase().includes("rate limit")) && (
            <button
              onClick={() => navigate("/settings")}
              className="ml-3 shrink-0 rounded bg-red-500/20 px-2.5 py-1 text-xs font-medium text-red-200 hover:bg-red-500/30"
            >
              Open Settings
            </button>
          )}
        </div>
      )}
      {recsErr && results === null && (
        <div className="flex items-center justify-between rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-300">
          <span>Could not load recommendations: {recsErr}</span>
          {(recsErr.includes("429") || recsErr.toLowerCase().includes("rate limit")) && (
            <button
              onClick={() => navigate("/settings")}
              className="ml-3 shrink-0 rounded bg-amber-500/20 px-2.5 py-1 text-xs font-medium text-amber-200 hover:bg-amber-500/30"
            >
              Open Settings
            </button>
          )}
        </div>
      )}

      {/* Recommended Models Section (Visible when no active search results) */}
      {results === null && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-200">Recommended for Your System</h2>
              <p className="text-xs text-slate-500">
                Trending models scored in real time against your GPU architecture. Click any card to inspect all quant variants.
              </p>
            </div>
            {!loadingRecs && (
              <span className="text-xs text-slate-500">
                {filteredRecommendations.length} models
              </span>
            )}
          </div>

          {loadingRecs ? (
            <div className="flex flex-col items-center justify-center p-16 space-y-3 rounded-xl border border-edge bg-surface-2">
              <Spinner label="Scoring recommendations for your GPU…" />
              <p className="text-xs text-slate-400">
                Discovering models and calculating VRAM fit, context ceiling, and bandwidth throughput…
              </p>
            </div>
          ) : filteredRecommendations.length === 0 ? (
            <div className="rounded-xl border border-edge bg-surface-2 p-8 text-center text-sm text-slate-400">
              No recommended models found for this category or filter.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {filteredRecommendations.map((m) => (
                <ModelCard
                  key={m.id}
                  model={m}
                  onSelect={() => setSelectedModel(m)}
                  onDeploy={deploy}
                  onPull={pull}
                  pullState={pulls[getBestVariant(m)?.variant.repo_id || m.id] || pulls[m.id]}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Search Results Section */}
      {results !== null && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-300">
                Search Results ({filteredResults?.length ?? 0} found)
              </h2>
              <p className="text-xs text-slate-500">
                Models scored for your hardware with discovered quant variants. Click any card or row for variant breakdown.
              </p>
            </div>
            <button
              onClick={clearSearch}
              className="text-xs text-indigo-400 hover:underline"
            >
              ← Back to Recommended Models
            </button>
          </div>

          {filteredResults?.length === 0 ? (
            <div className="rounded-xl border border-edge bg-surface-2 p-12 text-center text-sm text-slate-400">
              No models found matching your search. Try adjusting terms or loosening the compatibility filter.
            </div>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {filteredResults?.map((m) => (
                <ModelCard
                  key={m.id}
                  model={m}
                  onSelect={() => setSelectedModel(m)}
                  onDeploy={deploy}
                  onPull={pull}
                  pullState={pulls[getBestVariant(m)?.variant.repo_id || m.id] || pulls[m.id]}
                />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-edge">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-edge bg-surface-2 text-left text-xs uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2.5">Model</th>
                    <th className="px-3 py-2.5 text-center">System Fit</th>
                    <th className="px-3 py-2.5 text-right">Params</th>
                    <th className="px-3 py-2.5 text-right">Usable Context</th>
                    <th className="px-3 py-2.5 text-right">Est. Speed</th>
                    <th className="px-3 py-2.5 text-right">Downloads</th>
                    <th className="px-3 py-2.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResults?.map((m) => {
                    const best = getBestVariant(m);
                    const fit = best?.fit;
                    const variant = best?.variant;
                    const pullState = pulls[variant?.repo_id || m.id] || pulls[m.id];
                    const tokS = fit?.measured_tok_s ?? fit?.est_tok_s;
                    return (
                      <tr
                        key={m.id}
                        onClick={() => setSelectedModel(m)}
                        className="border-b border-edge/60 last:border-0 hover:bg-surface-2/70 transition-colors cursor-pointer"
                      >
                        <td className="px-3 py-2.5 max-w-[280px]">
                          <div className="font-medium text-slate-200 truncate hover:text-indigo-300" title={m.id}>
                            {m.id}
                          </div>
                          <div className="flex items-center gap-2 text-[11px] text-slate-500">
                            {m.pipeline_tag && <span>{m.pipeline_tag}</span>}
                            {m.params_b == null && <Badge color="amber">params missing</Badge>}
                          </div>
                        </td>
                        <td className="px-3 py-2.5 text-center">
                          {fit ? (
                            <div className="inline-flex flex-col items-center gap-1">
                              <div className="flex items-center gap-1 flex-wrap justify-center">
                                <FitVerdictBadge verdict={fit.verdict} score={fit.score} />
                                <RunModeBadge mode={fit.run_mode} />
                              </div>
                              <div className="flex items-center gap-1">
                                <span className="text-[10px] px-1.5 py-0.2 rounded font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                                  Rec: {variant?.label || variant?.format || "FP16"}
                                </span>
                                {(variant?.format === "GGUF" || fit?.format_support === "Experimental") && (
                                  <span className="text-[9px] px-1 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30" title="vLLM support for GGUF is experimental">
                                    ⚠️ GGUF
                                  </span>
                                )}
                              </div>
                              <span className="text-[10px] text-slate-500">
                                ~{fit.vram_pct}% VRAM
                                {fit.swap_space_gb > 0 ? ` · ${fit.swap_space_gb}GB swap` : ""}
                                {fit.cpu_offload_gb > 0 ? ` · ${fit.cpu_offload_gb}GB offload` : ""}
                              </span>
                            </div>
                          ) : (
                            <span className="text-slate-500 text-xs">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right text-slate-300">
                          {m.params_b != null ? `${m.params_b.toFixed(2)}B` : "—"}
                          {fit && (
                            <div className="text-[10px] text-slate-500">
                              ~{fit.weight_gb.toFixed(1)} GB
                            </div>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          {fit ? (
                            <div className="flex flex-col items-end gap-0.5">
                              <ContextBadge fit={fit} />
                              {m.context && fit.extended_context < m.context && (
                                <div className="text-[10px] text-amber-400/80">
                                  max {fmtContext(m.context)}
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="text-slate-300">{fmtContext(m.context)}</span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          {tokS != null ? (
                            <div>
                              <span className="font-mono text-cyan-300">{fmtTokPerSec(tokS)}</span>
                              {fit?.measured_tok_s != null && (
                                <div className="text-[9px] text-emerald-400">measured</div>
                              )}
                            </div>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right text-slate-400">{fmtNum(m.downloads)}</td>
                        <td className="px-3 py-2.5 text-right" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-1.5">
                            <Button
                              variant="ghost"
                              className="text-xs px-2.5 py-1"
                              onClick={() => setSelectedModel(m)}
                            >
                              Inspect
                            </Button>
                            <Button
                              variant="primary"
                              className="text-xs px-2.5 py-1"
                              onClick={() => deploy(variant?.repo_id || m.id, (variant?.format || "fp16").toLowerCase(), fit)}
                            >
                              Deploy
                            </Button>
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
                              <Button
                                variant="ghost"
                                className="text-xs px-2.5 py-1"
                                onClick={() => pull(variant?.repo_id || m.id)}
                              >
                                Pull
                              </Button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Live Pulls Indicator Banner */}
      {Object.entries(pulls).filter(([, p]) => p.state === "downloading").length > 0 && (
        <div className="space-y-1 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3 text-xs text-slate-400">
          {Object.entries(pulls)
            .filter(([, p]) => p.state === "downloading")
            .map(([m, p]) => (
              <div key={m} className="truncate">
                <span className="text-indigo-300 font-medium">{m}</span>: {p.file}
              </div>
            ))}
        </div>
      )}

      {/* Model Details & Discovered Quant Variants Modal */}
      {selectedModel && (
        <ModelDetailModal
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
          pulls={pulls}
          onPull={pull}
          onDeploy={deploy}
          totalVramMb={env?.gpu?.vram_total_mb ?? null}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model Card Component (Used for Recommended Models and Grid Search Results)
// ---------------------------------------------------------------------------
function ModelCard({
  model,
  onSelect,
  onDeploy,
  onPull,
  pullState,
}: {
  model: ModelWithFit;
  onSelect: () => void;
  onDeploy: (repoId: string, quant: string, fit?: FitResultBackend) => void;
  onPull: (repoId: string) => void;
  pullState?: PullStatus;
}) {
  const best = getBestVariant(model);
  const fit = best?.fit;
  const variant = best?.variant;
  const isGguf = variant?.format === "GGUF" || fit?.format_support === "Experimental";
  const tokS = fit?.measured_tok_s ?? fit?.est_tok_s;

  return (
    <div
      onClick={onSelect}
      className="group flex flex-col justify-between rounded-xl border border-edge bg-surface-2 p-4 transition-all hover:border-indigo-500/60 hover:bg-surface-2/95 cursor-pointer shadow-sm hover:shadow-md"
    >
      <div className="space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-slate-200 text-sm group-hover:text-indigo-300 transition-colors truncate">
                {model.id.split("/").pop()}
              </span>
              {model.pipeline_tag && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-3 text-slate-400 shrink-0">
                  {model.pipeline_tag}
                </span>
              )}
            </div>
            <div className="text-[11px] font-mono text-slate-500 truncate max-w-[210px]" title={model.id}>
              {model.id}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            {fit && (
              <div className="flex items-center gap-1 flex-wrap justify-end">
                <FitVerdictBadge verdict={fit.verdict} score={fit.score} />
                <RunModeBadge mode={fit.run_mode} />
              </div>
            )}
            <div className="flex items-center gap-1">
              <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                Rec: {variant?.label || variant?.format || "FP16"}
              </span>
              {isGguf && (
                <span
                  className="inline-flex items-center gap-0.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-medium text-amber-300"
                  title="vLLM support for GGUF is experimental"
                >
                  ⚠️ GGUF
                </span>
              )}
            </div>
          </div>
        </div>

        {/* 3-column stats */}
        <div className="grid grid-cols-3 gap-2 rounded-lg bg-surface-3/30 p-2 text-center text-xs">
          <div>
            <div className="text-[10px] text-slate-500 uppercase">Params</div>
            <div className="font-medium text-slate-300">
              {model.params_b != null ? `${model.params_b.toFixed(1)}B` : "—"}
            </div>
          </div>
          <div className="flex flex-col items-center justify-center">
            <div className="text-[10px] text-slate-500 uppercase">Usable Ctx</div>
            <div className="font-medium text-slate-300 mt-0.5">
              {fit ? <ContextBadge fit={fit} /> : fmtContext(model.context)}
            </div>
            {fit && model.context && fit.extended_context < model.context && (
              <div className="text-[9px] text-amber-400/80 truncate">
                max {fmtContext(model.context)}
              </div>
            )}
          </div>
          <div>
            <div className="text-[10px] text-slate-500 uppercase">
              {fit?.measured_tok_s != null ? "Meas. Speed" : "Est. Speed"}
            </div>
            <div className="font-mono text-cyan-300">
              {fmtTokPerSec(tokS)}
            </div>
          </div>
        </div>

        {/* Memory and reasoning */}
        {fit && (
          <div className="text-[11px] text-slate-400 space-y-1">
            <div className="flex items-center justify-between text-slate-500 text-[10px]">
              <span>VRAM Footprint:</span>
              <span className={fit.vram_pct > 90 ? "text-amber-400 font-medium" : "text-slate-300"}>
                ~{fit.vram_pct}% ({fit.weight_gb.toFixed(1)} GB)
              </span>
            </div>
            {fit.swap_space_gb > 0 && (
              <div className="flex items-center justify-between text-slate-500 text-[10px]">
                <span>RAM Swap:</span>
                <span className="text-cyan-300 font-medium">
                  {fit.swap_space_gb} GB
                </span>
              </div>
            )}
            {fit.cpu_offload_gb > 0 && (
              <div className="flex items-center justify-between text-slate-500 text-[10px]">
                <span>CPU Offload:</span>
                <span className="text-amber-300 font-medium">
                  {fit.cpu_offload_gb} GB
                </span>
              </div>
            )}
            {fit.reason && (
              <div className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                {fit.reason}
              </div>
            )}
          </div>
        )}

        {model.variants && model.variants.length > 1 && (
          <div className="text-[10px] text-indigo-400/80">
            {model.variants.length} quant variants discovered
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div
        className="mt-3.5 pt-3 border-t border-edge/60 flex items-center justify-between gap-2"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onSelect}
          className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-2 hover:underline flex items-center gap-1"
        >
          <span>View Details & Quants</span>
          <span>→</span>
        </button>
        <div className="flex items-center gap-1.5">
          <Button
            variant="primary"
            className="text-xs px-2.5 py-1"
            onClick={() => onDeploy(variant?.repo_id || model.id, (variant?.format || "fp16").toLowerCase(), fit)}
          >
            Deploy
          </Button>
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
            <Button
              variant="ghost"
              className="text-xs px-2.5 py-1"
              onClick={() => onPull(variant?.repo_id || model.id)}
            >
              Pull
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model Details Modal (All Discovered Quant Variants with GGUF Flagging)
// ---------------------------------------------------------------------------
function ModelDetailModal({
  model,
  onClose,
  pulls,
  onPull,
  onDeploy,
  totalVramMb,
}: {
  model: ModelWithFit;
  onClose: () => void;
  pulls: Record<string, PullStatus>;
  onPull: (repoId: string) => void;
  onDeploy: (repoId: string, quant: string, fit?: FitResultBackend) => void;
  totalVramMb: number | null;
}) {
  const sortedVariants = useMemo(() => {
    return [...model.variants].sort((a, b) => b.fit.score - a.fit.score);
  }, [model.variants]);

  const hasGguf = sortedVariants.some(
    (vwf) => vwf.variant.format === "GGUF" || vwf.fit.format_support === "Experimental"
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-2xl border border-edge bg-surface-2 p-6 shadow-2xl space-y-5 text-slate-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-start justify-between gap-4 border-b border-edge/80 pb-4">
          <div className="space-y-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-bold text-slate-100 truncate">
                {model.id.split("/").pop()}
              </h2>
              {model.pipeline_tag && (
                <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                  {model.pipeline_tag}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span className="truncate">{model.id}</span>
              <a
                href={`https://huggingface.co/${model.id}`}
                target="_blank"
                rel="noreferrer"
                className="text-indigo-400 hover:text-indigo-300 hover:underline flex items-center gap-1 font-sans text-[11px]"
                title="Open on Hugging Face"
              >
                <span>Hugging Face</span>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-100 text-lg p-1 rounded-md hover:bg-surface-3 transition-colors"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Model Meta Overview */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Parameters</div>
            <div className="font-semibold text-slate-200 text-sm">
              {model.params_b != null ? `${model.params_b.toFixed(2)}B` : "—"}
            </div>
          </div>
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Native Context</div>
            <div className="font-semibold text-slate-200 text-sm">
              {fmtContext(model.context)}
            </div>
            {model.context_source && (
              <div className="text-[9px] text-slate-500 truncate" title={model.context_source}>
                src: {model.context_source}
              </div>
            )}
          </div>
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Downloads</div>
            <div className="font-semibold text-slate-200 text-sm">
              {fmtNum(model.downloads)}
            </div>
          </div>
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Likes</div>
            <div className="font-semibold text-slate-200 text-sm">
              {fmtNum(model.likes)}
            </div>
          </div>
        </div>

        {/* Discovered Quantization Variants List */}
        <div className="space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">
                Discovered Quantization Variants ({sortedVariants.length})
              </h3>
              <p className="text-xs text-slate-400">
                Sorted by hardware fit score for your GPU. Select any variant to pull or deploy.
              </p>
            </div>
            {totalVramMb && (
              <span className="text-xs text-slate-400">
                Available GPU VRAM: <strong className="text-slate-200">{(totalVramMb / 1024).toFixed(1)} GB</strong>
              </span>
            )}
          </div>

          {/* GGUF Experimental Warning Banner */}
          {hasGguf && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200 flex items-start gap-2.5">
              <span className="text-base leading-none">⚠️</span>
              <div className="space-y-0.5">
                <div className="font-semibold text-amber-300">vLLM GGUF Support is Experimental</div>
                <div className="text-amber-200/80 leading-relaxed">
                  vLLM support for GGUF quantization is experimental and may have limited kernel optimization or reduced feature support. For peak throughput and full context support, native FP8, AWQ, or GPTQ formats are recommended.
                </div>
              </div>
            </div>
          )}

          {sortedVariants.length === 0 ? (
            <div className="rounded-xl border border-edge bg-surface-3/30 p-6 text-center text-sm text-slate-400">
              No specific quantization variants discovered. Deploying will use the default baseline format.
            </div>
          ) : (
            <div className="space-y-2.5 max-h-[48vh] overflow-y-auto pr-1">
              {sortedVariants.map((vwf, idx) => {
                const { variant, fit } = vwf;
                const best = getBestVariant(model);
                const isBest = vwf === best;
                const isGguf = variant.format === "GGUF" || fit.format_support === "Experimental";
                const pullState = pulls[variant.repo_id] || (isBest ? pulls[model.id] : undefined);
                const tokS = fit.measured_tok_s ?? fit.est_tok_s;

                return (
                  <div
                    key={`${variant.repo_id}-${variant.label}-${idx}`}
                    className={`rounded-xl border p-4 transition-colors ${
                      isBest
                        ? "border-indigo-500/60 bg-indigo-500/5 ring-1 ring-indigo-500/20"
                        : "border-edge bg-surface-3/30 hover:bg-surface-3/50"
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                      {/* Left: Info */}
                      <div className="space-y-2 flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="px-2 py-0.5 rounded font-mono font-bold text-xs bg-surface-2 border border-edge text-slate-200">
                            {variant.format}
                          </span>
                          <span className="font-semibold text-sm text-slate-100">
                            {variant.label}
                          </span>
                          {isBest && (
                            <span className="px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded-full bg-emerald-500 text-black shadow">
                              Best Fit
                            </span>
                          )}
                          {isGguf && (
                            <span
                              className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300"
                              title="vLLM support for GGUF is experimental"
                            >
                              ⚠️ Experimental
                            </span>
                          )}
                        </div>

                        <div className="font-mono text-xs text-slate-400 truncate" title={variant.repo_id}>
                          {variant.repo_id}
                          {variant.gguf_file && (
                            <span className="text-amber-300/90 ml-1.5 font-sans">
                              (file: {variant.gguf_file})
                            </span>
                          )}
                        </div>

                        {/* Fit Metrics Breakdown */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-xs">
                          <div className="rounded bg-surface-2/60 p-2">
                            <div className="text-[10px] text-slate-500">Weight & VRAM</div>
                            <div className="font-semibold text-slate-200">
                              {fit.weight_gb.toFixed(1)} GB
                              <span className={`ml-1 text-[11px] ${fit.vram_pct > 90 ? "text-amber-400 font-bold" : "text-slate-400"}`}>
                                ({fit.vram_pct}%)
                              </span>
                            </div>
                            {(fit.swap_space_gb > 0 || fit.cpu_offload_gb > 0) && (
                              <div className="text-[10px] text-cyan-300 mt-0.5">
                                {fit.swap_space_gb > 0 && `Swap: ${fit.swap_space_gb}GB `}
                                {fit.cpu_offload_gb > 0 && `Offload: ${fit.cpu_offload_gb}GB`}
                              </div>
                            )}
                          </div>

                          <div className="rounded bg-surface-2/60 p-2">
                            <div className="text-[10px] text-slate-500">Usable Context</div>
                            <div className="font-semibold text-slate-200 mt-0.5">
                              <ContextBadge fit={fit} />
                              {fit.extended_context < fit.native_context && (
                                <span className="text-[10px] text-amber-400/90 ml-1">
                                  / {fmtContext(fit.native_context)}
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="rounded bg-surface-2/60 p-2">
                            <div className="text-[10px] text-slate-500">
                              {fit.measured_tok_s != null ? "Speed (Measured)" : "Speed (Estimated)"}
                            </div>
                            <div className="font-mono font-semibold text-cyan-300">
                              {fmtTokPerSec(tokS)}
                            </div>
                          </div>

                          <div className="rounded bg-surface-2/60 p-2">
                            <div className="text-[10px] text-slate-500">Support</div>
                            <div className="font-semibold text-slate-200">
                              {fit.format_support === "Native" ? (
                                <span className="text-emerald-400">Native vLLM</span>
                              ) : (
                                <span className="text-amber-400">Experimental</span>
                              )}
                            </div>
                          </div>
                        </div>

                        {fit.reason && (
                          <p className="text-xs text-slate-400 leading-relaxed pt-0.5">
                            {fit.reason}
                          </p>
                        )}
                      </div>

                      {/* Right: Badge and Action Buttons */}
                      <div className="flex flex-col items-end justify-between gap-3 shrink-0 sm:min-w-[140px]">
                        <div className="flex items-center gap-1.5 flex-wrap justify-end">
                          <FitVerdictBadge verdict={fit.verdict} score={fit.score} />
                          <RunModeBadge mode={fit.run_mode} />
                        </div>

                        <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
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
                            <Button
                              variant="ghost"
                              className="text-xs px-2.5 py-1"
                              onClick={() => onPull(variant.repo_id)}
                              title="Download to local cache"
                            >
                              Pull
                            </Button>
                          )}
                          <Button
                            variant="primary"
                            className="text-xs px-3 py-1"
                            onClick={() => onDeploy(variant.repo_id, variant.format.toLowerCase(), fit)}
                            title={`Deploy server with ${variant.label}`}
                          >
                            Deploy →
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-edge/60 flex justify-end">
          <Button variant="ghost" onClick={onClose} className="text-xs">
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}