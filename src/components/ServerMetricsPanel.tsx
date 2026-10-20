import type { ReactNode } from "react";
import { fmtNum, fmtTokPerSec } from "../api";
import type { MetricsSnapshot, ServerMetricPoint } from "../types";
import { MultiSparkline, Sparkline } from "./Sparkline";

interface ServerMetricsPanelProps {
  metrics: MetricsSnapshot | null;
  series: ServerMetricPoint[];
  status: string;
  metricsEnabled: boolean;
}

interface StatItem {
  label: string;
  value: string;
  hint?: string;
  accent?: string;
}

const fmtPct = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)}%`;

const fmtMs = (value: number | null | undefined) => {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value < 10 ? 1 : 0)} ms`;
};

const fmtRate = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)}/s`;

const cacheColor = (value: number | null | undefined) => {
  if (value == null) return "text-slate-200";
  if (value >= 90) return "text-red-300";
  if (value >= 75) return "text-amber-300";
  return "text-emerald-300";
};

function Notice({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-edge bg-surface/50 p-3">
      <div className="text-sm font-medium text-slate-300">{title}</div>
      {children && <div className="mt-1 text-xs leading-relaxed text-slate-500">{children}</div>}
    </div>
  );
}

function MetricStat({ label, value, hint, accent = "text-slate-200" }: StatItem) {
  return (
    <div className="rounded-md border border-edge bg-surface p-2.5">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className={`mt-0.5 font-mono text-sm ${accent}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[10px] text-slate-600">{hint}</div>}
    </div>
  );
}

function StatSection({ title, items }: { title: string; items: StatItem[] }) {
  if (items.length === 0) return null;
  return (
    <section>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        {title}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
        {items.map((item) => (
          <MetricStat key={item.label} {...item} />
        ))}
      </div>
    </section>
  );
}

/**
 * Live runtime metrics for one server.  The top section shows the current
 * snapshot; below it are per-series charts backed by the monitor's 5-second
 * ring buffer.  Optional metrics are omitted when a backend does not expose
 * them, so llama.cpp and vLLM both get a clean layout.
 */
export function ServerMetricsPanel({
  metrics,
  series,
  status,
  metricsEnabled,
}: ServerMetricsPanelProps) {
  if (!metricsEnabled) {
    return (
      <Notice title="Metrics collection is disabled for this server">
        Turn on <span className="text-slate-300">Collect runtime metrics</span> in the server
        settings, then restart the server. vLLM exposes metrics by default; llama.cpp needs the
        <code className="mx-1 rounded bg-black/30 px-1">--metrics</code> flag that this option adds.
      </Notice>
    );
  }

  if (status !== "running") {
    return (
      <Notice
        title={
          status === "starting"
            ? "Server is starting — metrics will appear when it is ready."
            : "Start this server to collect live metrics."
        }
      />
    );
  }

  if (!metrics) {
    return (
      <Notice title="Waiting for the first /metrics sample…">
        The monitor polls the backend every 5 seconds. Numbers appear first, followed by the
        history charts.
      </Notice>
    );
  }

  const measured = metrics.measured;
  const genTokS = measured?.tokens_per_sec ?? null;
  const promptTokS = measured?.prompt_tokens_per_sec ?? null;

  const liveStats: StatItem[] = [
    {
      label: "Decode",
      value: genTokS == null ? "—" : fmtTokPerSec(genTokS),
      hint: "generation tok/s",
      accent: "text-cyan-300",
    },
    {
      label: "Prefill",
      value: promptTokS == null ? "—" : fmtTokPerSec(promptTokS),
      hint: "prompt tok/s",
      accent: "text-violet-300",
    },
    {
      label: "Running",
      value: fmtNum(metrics.running),
      hint: "requests in flight",
      accent: metrics.running > 0 ? "text-emerald-300" : "text-slate-200",
    },
    {
      label: "Waiting",
      value: fmtNum(metrics.waiting),
      hint: "queued requests",
      accent: metrics.waiting > 0 ? "text-amber-300" : "text-slate-200",
    },
  ];
  if (metrics.requests_per_sec != null) {
    liveStats.push({
      label: "Request rate",
      value: fmtRate(metrics.requests_per_sec),
      hint: "completed per second",
      accent: "text-sky-300",
    });
  }
  if (metrics.kv_cache_pct != null) {
    liveStats.push({
      label: "KV cache",
      value: fmtPct(metrics.kv_cache_pct),
      hint: "allocated blocks",
      accent: cacheColor(metrics.kv_cache_pct),
    });
  }

  const efficiencyStats: StatItem[] = [];
  if (metrics.ttft_ms != null) {
    efficiencyStats.push({ label: "TTFT", value: fmtMs(metrics.ttft_ms), hint: "avg first token" });
  }
  if (metrics.e2e_latency_ms != null) {
    efficiencyStats.push({
      label: "E2E latency",
      value: fmtMs(metrics.e2e_latency_ms),
      hint: "avg full request",
    });
  }
  if (metrics.prefill_time_ms != null) {
    efficiencyStats.push({
      label: "Prefill time",
      value: fmtMs(metrics.prefill_time_ms),
      hint: "avg prompt phase",
    });
  }
  if (metrics.queue_time_ms != null) {
    efficiencyStats.push({
      label: "Queue time",
      value: fmtMs(metrics.queue_time_ms),
      hint: "avg wait before run",
    });
  }
  if (metrics.cache_hit_pct != null) {
    efficiencyStats.push({
      label: "Cache hit",
      value: fmtPct(metrics.cache_hit_pct),
      hint: "prefix reuse",
      accent: "text-amber-300",
    });
  }

  const totalStats: StatItem[] = [
    {
      label: "Decode tokens",
      value: fmtNum(metrics.total_generation_tokens),
      hint: "since server start",
    },
    {
      label: "Prompt tokens",
      value: fmtNum(metrics.total_prompt_tokens),
      hint: "since server start",
    },
  ];
  if (metrics.requests_available) {
    totalStats.push({
      label: "Requests",
      value: fmtNum(metrics.requests),
      hint: "completed",
    });
  }
  if (metrics.prompt_tokens_cached > 0) {
    totalStats.push({
      label: "Cached tokens",
      value: fmtNum(metrics.prompt_tokens_cached),
      hint: "reused prefill",
    });
  }
  if (metrics.preemptions > 0) {
    totalStats.push({
      label: "Preemptions",
      value: fmtNum(metrics.preemptions),
      hint: "KV pressure",
      accent: "text-amber-300",
    });
  }

  const kvSeries = series
    .map((point) => point.kv_cache_pct)
    .filter((value): value is number => value != null);
  const cacheSeries = series
    .map((point) => point.cache_hit_pct)
    .filter((value): value is number => value != null);
  const latencySeries = [
    {
      label: "TTFT",
      data: series
        .map((point) => point.ttft_ms)
        .filter((value): value is number => value != null),
      color: "#38bdf8",
    },
    {
      label: "E2E",
      data: series
        .map((point) => point.e2e_latency_ms)
        .filter((value): value is number => value != null),
      color: "#f472b6",
    },
  ].filter((line) => line.data.length > 0);

  const charts: ReactNode[] = [
    <Sparkline
      key="decode"
      label="Decode throughput"
      data={series.map((point) => point.tok_s)}
      height={52}
      min={0}
      color="#22d3ee"
      unit="tok/s"
      currentValue={genTokS == null ? undefined : fmtTokPerSec(genTokS)}
    />,
    <Sparkline
      key="prefill"
      label="Prefill throughput"
      data={series.map((point) => point.prompt_tok_s)}
      height={52}
      min={0}
      color="#a78bfa"
      unit="tok/s"
      currentValue={promptTokS == null ? undefined : fmtTokPerSec(promptTokS)}
    />,
    <MultiSparkline
      key="queue"
      label="Queue depth"
      series={[
        {
          label: "Running",
          data: series.map((point) => point.requests_running),
          color: "#34d399",
        },
        {
          label: "Waiting",
          data: series.map((point) => point.requests_waiting),
          color: "#fbbf24",
        },
      ]}
      height={52}
      min={0}
      formatValue={(value) => value.toFixed(0)}
    />,
  ];
  if (kvSeries.length > 0) {
    charts.push(
      <Sparkline
        key="kv"
        label="KV cache usage"
        data={kvSeries}
        height={52}
        min={0}
        max={100}
        color="#fb7185"
        unit="%"
        formatValue={(value) => value.toFixed(0)}
        currentValue={metrics.kv_cache_pct == null ? undefined : fmtPct(metrics.kv_cache_pct)}
      />
    );
  }
  if (latencySeries.length > 0) {
    charts.push(
      <MultiSparkline
        key="latency"
        label="Latency"
        series={latencySeries}
        height={52}
        min={0}
        formatValue={(value) =>
          value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(0)}ms`
        }
      />
    );
  }
  if (cacheSeries.length > 0) {
    charts.push(
      <Sparkline
        key="cache"
        label="Prefix cache hit rate"
        data={cacheSeries}
        height={52}
        min={0}
        max={100}
        color="#f59e0b"
        unit="%"
        formatValue={(value) => value.toFixed(0)}
        currentValue={metrics.cache_hit_pct == null ? undefined : fmtPct(metrics.cache_hit_pct)}
      />
    );
  }

  return (
    <div className="space-y-4">
      <StatSection title="Live" items={liveStats} />
      <StatSection title="Latency & efficiency" items={efficiencyStats} />

      {series.length > 0 ? (
        <div className="grid gap-3 sm:grid-cols-2">{charts}</div>
      ) : (
        <div className="text-xs text-slate-500">
          History charts appear after the next sample (about 5 seconds).
        </div>
      )}

      <StatSection title="Totals" items={totalStats} />
    </div>
  );
}
