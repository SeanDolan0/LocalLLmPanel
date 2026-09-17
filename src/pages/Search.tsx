import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  events,
  fmtContext,
  fmtNum,
  fmtTokPerSec,
  quantLabel,
  recommendBestQuant,
  SUPPORTED_QUANTS,
  type FitAssessment,
  type QuantFitSummary,
  type SupportedQuant,
  type UseCase,
} from "../api";
import { Badge, Button, inputCls, Spinner } from "../ui";
import type { EnvStatus, ModelWithStats, PullStatus } from "../types";
import { RECOMMENDED_MODELS } from "../data/recommended";

export interface DetailModel {
  id: string;
  name: string;
  category?: string;
  tag?: string;
  description?: string;
  pipeline_tag?: string | null;
  params_b: number | null;
  context: number | null;
  context_source?: string | null;
  context_estimated?: boolean;
  downloads?: number;
  likes?: number;
  trending_score?: number;
  recommendedQuant: SupportedQuant;
  fit: FitAssessment;
  allQuants: QuantFitSummary[];
}

export default function Search() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ModelWithStats[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [onlyRecommended, setOnlyRecommended] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<DetailModel | null>(null);
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
      // Backend search models (defaults to fp16 baseline; quants evaluated dynamically in UI)
      const res = await api.searchModels(q, "fp16");
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

  const deploy = (modelId: string, quant: string) => {
    navigate("/servers", { state: { prefillModel: modelId, prefillQuant: quant } });
  };

  const selectCategory = (key: string) => {
    setActiveCategory(key);
    if (results !== null || query) {
      setQuery("");
      setResults(null);
    }
  };

  // Filter curated models according to active category
  const filteredRecommendations = useMemo(() => {
    return RECOMMENDED_MODELS.filter((m) => {
      if (activeCategory !== "all" && m.category !== activeCategory) return false;
      return true;
    });
  }, [activeCategory]);

  // Compute recommended quants and stats for curated recommendations
  const recommendationRows = useMemo<DetailModel[]>(() => {
    const list: DetailModel[] = filteredRecommendations.map((m) => {
      const quantRec = recommendBestQuant(
        m.params_b,
        totalVramMb,
        bandwidth,
        m.context,
        m.category,
        m.quality_prior
      );
      return {
        id: m.id,
        name: m.name,
        category: m.category,
        tag: m.tag,
        description: m.description,
        pipeline_tag: m.pipeline_tag,
        params_b: m.params_b,
        context: m.context,
        recommendedQuant: quantRec.bestQuant,
        fit: quantRec.fit,
        allQuants: quantRec.allQuants,
      };
    });
    // Sort recommendations by recommended fit score descending
    return list.sort((a, b) => b.fit.score - a.fit.score);
  }, [filteredRecommendations, totalVramMb, bandwidth]);

  // Process search results with best quant evaluation and multi-quant data
  const evaluatedResults = useMemo<DetailModel[] | null>(() => {
    if (!results) return null;
    const list: DetailModel[] = results.map((m) => {
      const useCase = (m.pipeline_tag === "feature-extraction" ? "embedding" : "general") as UseCase;
      const quantRec = recommendBestQuant(
        m.params_b,
        totalVramMb,
        bandwidth,
        m.context ?? 32768,
        useCase
      );
      return {
        id: m.id,
        name: m.id.split("/").pop() || m.id,
        pipeline_tag: m.pipeline_tag,
        params_b: m.params_b,
        context: m.context,
        context_source: m.context_source,
        context_estimated: m.context_estimated,
        downloads: m.downloads,
        likes: m.likes,
        trending_score: m.trending_score,
        recommendedQuant: quantRec.bestQuant,
        fit: quantRec.fit,
        allQuants: quantRec.allQuants,
      };
    });
    if (onlyRecommended) {
      return list.filter((m) => m.fit.fitLevel === "perfect" || m.fit.fitLevel === "good");
    }
    return list;
  }, [results, totalVramMb, bandwidth, onlyRecommended]);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
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

      {/* Search Input Bar (No longer restricted by single quant dropdown) */}
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

        {results !== null && (
          <label className="flex items-center gap-2 text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={onlyRecommended}
              onChange={(e) => setOnlyRecommended(e.target.checked)}
              className="rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0"
            />
            <span>Show only compatible (Perfect / Good)</span>
          </label>
        )}
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
          {err}
        </div>
      )}

      {/* Recommended Models Section (Visible when no active search query) */}
      {results === null && !searching && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-200">Recommended for Your System</h2>
              <p className="text-xs text-slate-500">
                Each card shows your hardware's recommended quantization & llmfit rating. Click any card for details and quant options.
              </p>
            </div>
            <span className="text-xs text-slate-500">
              {recommendationRows.length} models
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {recommendationRows.map((m) => {
              const pullState = pulls[m.id];
              const { components } = m.fit;
              return (
                <div
                  key={m.id}
                  onClick={() => setSelectedModel(m)}
                  className="group flex flex-col justify-between rounded-xl border border-edge bg-surface-2 p-4 transition-all hover:border-indigo-500/60 hover:bg-surface-2/95 cursor-pointer shadow-sm hover:shadow-md"
                >
                  <div className="space-y-2.5">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-slate-200 text-sm group-hover:text-indigo-300 transition-colors truncate">
                            {m.name}
                          </span>
                          {m.tag && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-3 text-slate-400 shrink-0">
                              {m.tag}
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] font-mono text-slate-500 truncate max-w-[210px]" title={m.id}>
                          {m.id}
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-1 shrink-0">
                        <Badge color={m.fit.badgeColor}>{m.fit.label}</Badge>
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                          Rec: {quantLabel(m.recommendedQuant)}
                        </span>
                      </div>
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed min-h-[36px] line-clamp-2">
                      {m.description}
                    </p>

                    {/* llmfit 4-pillar mini-meters */}
                    <div className="rounded-lg bg-surface-3/50 p-2.5 space-y-1.5 text-xs">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-400 font-medium">llmfit Pillars</span>
                        <span className="text-indigo-300 font-mono font-semibold">{m.fit.score}/100</span>
                      </div>
                      <div className="grid grid-cols-4 gap-1.5 text-[10px]">
                        <div className="rounded bg-surface-2/80 p-1 text-center">
                          <div className="text-slate-500">Quality</div>
                          <div className="font-medium text-slate-300">{components.quality}</div>
                        </div>
                        <div className="rounded bg-surface-2/80 p-1 text-center">
                          <div className="text-slate-500">Speed</div>
                          <div className="font-medium text-cyan-300">{components.speed}</div>
                        </div>
                        <div className="rounded bg-surface-2/80 p-1 text-center">
                          <div className="text-slate-500">Fit</div>
                          <div className="font-medium text-emerald-300">{components.fit}</div>
                        </div>
                        <div className="rounded bg-surface-2/80 p-1 text-center">
                          <div className="text-slate-500">Context</div>
                          <div className="font-medium text-slate-300">{components.context}</div>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 rounded-lg bg-surface-3/30 p-2 text-center text-xs">
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
                        <div className="font-mono text-cyan-300">{fmtTokPerSec(m.fit.estTokS)}</div>
                      </div>
                    </div>

                    {m.fit.reason && (
                      <div className="text-[11px] text-slate-500 line-clamp-1">
                        {m.fit.reason}
                      </div>
                    )}
                  </div>

                  <div
                    className="mt-3.5 pt-3 border-t border-edge/60 flex items-center justify-between gap-2"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      onClick={() => setSelectedModel(m)}
                      className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-2 hover:underline flex items-center gap-1"
                    >
                      <span>View Details & Quants</span>
                      <span>→</span>
                    </button>
                    <div className="flex items-center gap-1.5">
                      <Button
                        variant="primary"
                        className="text-xs px-2.5 py-1"
                        onClick={() => deploy(m.id, m.recommendedQuant)}
                      >
                        Deploy
                      </Button>
                      {pullState ? (
                        <Badge color={pullState.state === "complete" ? "emerald" : pullState.state === "failed" ? "red" : "indigo"}>
                          {pullState.state}
                        </Badge>
                      ) : (
                        <Button variant="ghost" className="text-xs px-2.5 py-1" onClick={() => pull(m.id)}>
                          Pull
                        </Button>
                      )}
                    </div>
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
            <div>
              <h2 className="text-sm font-semibold text-slate-300">
                Search Results ({evaluatedResults?.length ?? 0} found)
              </h2>
              <p className="text-xs text-slate-500">
                Click any row or Inspect to compare all quantization options and download specific quants.
              </p>
            </div>
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
                          <div className="inline-flex flex-col items-center gap-0.5">
                            <Badge color={m.fit.badgeColor}>{m.fit.label}</Badge>
                            <span className="text-[10px] px-1.5 py-0.2 rounded font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                              Rec: {quantLabel(m.recommendedQuant)}
                            </span>
                            {m.fit.vramPct !== null && (
                              <span className="text-[10px] text-slate-500">
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
                          {m.fit.estTokS != null ? (
                            <span className="font-mono text-cyan-300">{fmtTokPerSec(m.fit.estTokS)}</span>
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
                              onClick={() => deploy(m.id, m.recommendedQuant)}
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
                              <Button variant="ghost" className="text-xs px-2.5 py-1" onClick={() => pull(m.id)}>
                                Pull
                              </Button>
                            )}
                          </div>
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

      {/* Model Details & Quantization Modal */}
      {selectedModel && (
        <ModelDetailModal
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
          pullState={pulls[selectedModel.id]}
          allPulls={pulls}
          onPull={pull}
          onDeploy={deploy}
          totalVramMb={totalVramMb}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model Details & Quantization Modal Component
// ---------------------------------------------------------------------------

function ModelDetailModal({
  model,
  onClose,
  pullState,
  allPulls,
  onPull,
  onDeploy,
  totalVramMb,
}: {
  model: DetailModel;
  onClose: () => void;
  pullState?: PullStatus;
  allPulls: Record<string, PullStatus>;
  onPull: (id: string) => void;
  onDeploy: (id: string, quant: string) => void;
  totalVramMb: number | null;
}) {
  const [selectedQuant, setSelectedQuant] = useState<SupportedQuant>(model.recommendedQuant);
  const [targetRepo, setTargetRepo] = useState<string>(model.id);

  // Suggested repo name based on chosen quantization
  const suggestedRepo = useMemo(() => {
    if (selectedQuant === "awq") {
      return model.id.toLowerCase().includes("awq") ? model.id : `${model.id}-AWQ`;
    }
    if (selectedQuant === "gptq") {
      return model.id.toLowerCase().includes("gptq") ? model.id : `${model.id}-GPTQ-Int4`;
    }
    return model.id;
  }, [model.id, selectedQuant]);

  // When selected quant changes, auto-suggest target repo if switching to/from base
  const handleSelectQuant = (q: SupportedQuant) => {
    setSelectedQuant(q);
    if (q === "fp16" || q === "fp8") {
      setTargetRepo(model.id);
    }
  };

  const activeQuantSummary =
    model.allQuants.find((q) => q.quant === selectedQuant) || model.allQuants[0];
  const activeFit = activeQuantSummary.fit;
  const currentPull = allPulls[targetRepo] || pullState;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-3xl max-h-[92vh] overflow-y-auto rounded-2xl border border-edge bg-surface-2 p-6 shadow-2xl space-y-5 text-slate-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 border-b border-edge/80 pb-4">
          <div className="space-y-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-bold text-slate-100 truncate">{model.name}</h2>
              {model.tag && (
                <span className="text-xs px-2 py-0.5 rounded bg-surface-3 text-slate-400">
                  {model.tag}
                </span>
              )}
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

        {/* Description & Overview metrics */}
        {model.description && (
          <p className="text-xs text-slate-300 leading-relaxed bg-surface-3/30 p-3 rounded-lg border border-edge/40">
            {model.description}
          </p>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Parameters</div>
            <div className="font-semibold text-slate-200 text-sm">
              {model.params_b != null ? `${model.params_b.toFixed(2)}B` : "—"}
            </div>
          </div>
          <div className="rounded-lg bg-surface-3/50 p-2.5">
            <div className="text-[10px] text-slate-500 uppercase">Context Window</div>
            <div className="font-semibold text-slate-200 text-sm">
              {fmtContext(model.context)}
            </div>
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

        {/* Quantization Comparison Matrix & Selector */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">
                Quantization & Hardware Fit
              </h3>
              <p className="text-xs text-slate-400">
                Select a quantization to inspect runtime footprint and download or deploy.
              </p>
            </div>
            {totalVramMb && (
              <span className="text-xs text-slate-400">
                System GPU: <strong className="text-slate-200">{(totalVramMb / 1024).toFixed(1)} GB</strong>
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
            {SUPPORTED_QUANTS.map((q) => {
              const summary = model.allQuants.find((item) => item.quant === q);
              if (!summary) return null;
              const isSelected = selectedQuant === q;
              const isRecommended = model.recommendedQuant === q;
              const { fit } = summary;

              return (
                <button
                  key={q}
                  type="button"
                  onClick={() => handleSelectQuant(q)}
                  className={`flex flex-col justify-between p-3 rounded-xl border text-left transition-all relative ${
                    isSelected
                      ? "border-indigo-500 bg-indigo-500/10 shadow-sm shadow-indigo-500/20 ring-1 ring-indigo-500/50"
                      : "border-edge bg-surface-3/40 hover:bg-surface-3/70 hover:border-slate-600"
                  }`}
                >
                  {isRecommended && (
                    <span className="absolute -top-2 right-2 px-1.5 py-0.2 text-[9px] font-bold uppercase tracking-wider rounded-full bg-emerald-500 text-black shadow">
                      Recommended
                    </span>
                  )}
                  <div className="space-y-1 w-full">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-slate-200">
                        {quantLabel(q)}
                      </span>
                      <Badge color={fit.badgeColor}>{fit.fitLevel}</Badge>
                    </div>

                    <div className="text-[11px] text-slate-400">
                      Score: <strong className="text-indigo-300">{fit.score}/100</strong>
                    </div>

                    <div className="pt-2 text-[11px] space-y-0.5 border-t border-edge/40">
                      <div className="flex justify-between text-slate-400">
                        <span>Weight:</span>
                        <span className="text-slate-200">
                          {fit.estWeightGb !== null ? `~${fit.estWeightGb.toFixed(1)} GB` : "—"}
                        </span>
                      </div>
                      <div className="flex justify-between text-slate-400">
                        <span>VRAM:</span>
                        <span className={fit.vramPct && fit.vramPct > 90 ? "text-amber-300 font-medium" : "text-slate-200"}>
                          {fit.vramPct !== null ? `~${fit.vramPct}%` : "—"}
                        </span>
                      </div>
                      <div className="flex justify-between text-slate-400">
                        <span>Est. Speed:</span>
                        <span className="font-mono text-cyan-300">
                          {fmtTokPerSec(fit.estTokS)}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Active Quant Fit Breakdown */}
          <div className="rounded-xl border border-edge bg-surface-3/30 p-3.5 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-200">
                  {quantLabel(selectedQuant)} Fit Assessment:
                </span>
                <Badge color={activeFit.badgeColor}>{activeFit.label}</Badge>
              </div>
              <span className="text-xs font-mono text-indigo-300 font-semibold">
                Score: {activeFit.score}/100
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              {activeFit.reason}
            </p>

            {/* Pillar breakdown */}
            <div className="grid grid-cols-4 gap-2 text-center text-[10px] pt-1 border-t border-edge/40">
              <div className="p-1 rounded bg-surface-2">
                <div className="text-slate-500">Quality Retention</div>
                <div className="font-semibold text-slate-200">{activeFit.components.quality}/100</div>
              </div>
              <div className="p-1 rounded bg-surface-2">
                <div className="text-slate-500">Speed Score</div>
                <div className="font-semibold text-cyan-300">{activeFit.components.speed}/100</div>
              </div>
              <div className="p-1 rounded bg-surface-2">
                <div className="text-slate-500">VRAM Fit</div>
                <div className="font-semibold text-emerald-300">{activeFit.components.fit}/100</div>
              </div>
              <div className="p-1 rounded bg-surface-2">
                <div className="text-slate-500">Context Score</div>
                <div className="font-semibold text-slate-200">{activeFit.components.context}/100</div>
              </div>
            </div>
          </div>
        </div>

        {/* Download / Pull & Deploy Section */}
        <div className="rounded-xl border border-edge/80 bg-surface-3/50 p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
            <div>
              <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wide">
                Download & Deployment Configuration
              </h4>
              <p className="text-[11px] text-slate-400">
                Pull to local cache or launch directly in vLLM with <strong>{quantLabel(selectedQuant)}</strong>.
              </p>
            </div>
            {currentPull && (
              <Badge
                color={
                  currentPull.state === "complete"
                    ? "emerald"
                    : currentPull.state === "failed"
                    ? "red"
                    : "indigo"
                }
              >
                {currentPull.state}
              </Badge>
            )}
          </div>

          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-slate-400 block">
              Hugging Face Repository ID to pull:
            </label>
            <div className="flex gap-2">
              <input
                className={`${inputCls} font-mono text-xs`}
                value={targetRepo}
                onChange={(e) => setTargetRepo(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-7B-Instruct"
              />
              {suggestedRepo !== targetRepo && (
                <Button
                  variant="ghost"
                  className="text-xs shrink-0"
                  onClick={() => setTargetRepo(suggestedRepo)}
                  title={`Switch to suggested ${quantLabel(selectedQuant)} repo`}
                >
                  Use {suggestedRepo.split("/").pop()}
                </Button>
              )}
            </div>
            {suggestedRepo !== model.id && selectedQuant !== "fp16" && (
              <p className="text-[10px] text-slate-500">
                Tip: You can download the base model and let vLLM apply <code>--quantization {selectedQuant}</code>, or pull a pre-quantized repo like <code>{suggestedRepo}</code>.
              </p>
            )}
          </div>

          {currentPull?.state === "downloading" && currentPull.file && (
            <div className="text-[11px] text-indigo-300 font-mono truncate bg-surface-2 p-2 rounded border border-indigo-500/20">
              Downloading: {currentPull.file}
            </div>
          )}

          <div className="pt-2 flex flex-wrap items-center justify-between gap-2 border-t border-edge/60">
            <Button variant="ghost" onClick={onClose} className="text-xs">
              Close
            </Button>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                className="text-xs"
                disabled={!targetRepo.trim() || currentPull?.state === "downloading"}
                onClick={() => onPull(targetRepo.trim())}
              >
                {currentPull?.state === "downloading" ? (
                  <Spinner label="Pulling…" />
                ) : currentPull?.state === "complete" ? (
                  "Re-pull Model"
                ) : (
                  `Pull (${quantLabel(selectedQuant)})`
                )}
              </Button>
              <Button
                variant="primary"
                className="text-xs"
                disabled={!targetRepo.trim()}
                onClick={() => onDeploy(targetRepo.trim(), selectedQuant)}
              >
                Deploy Server with {quantLabel(selectedQuant)} →
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}