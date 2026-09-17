import { useCallback, useEffect, useRef, useState } from "react";
import { api, events, fmtGB, fmtNum } from "../api";
import { Button, Card, CardTitle, Gauge, Spinner } from "../ui";
import type { EnvStatus, ProvisionReport, WslLogEvent } from "../types";

export default function Dashboard() {
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [logs, setLogs] = useState<WslLogEvent[]>([]);
  const [provisioning, setProvisioning] = useState(false);
  const [provisionErr, setProvisionErr] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (provisioning && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [logs, provisioning]);

  const refresh = useCallback(() => {
    api
      .envStatus()
      .then(setEnv)
      .catch(() => setEnv((e) => e ?? { wsl_ok: false, distro: "?", apt_based: false, provisioned: false, report: null, gpu: null, servers_running: 0, running_weight_gb: 0, gpu_bandwidth_gbs: 700, gpu_bw_known: false }));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    const unsub = events.wslLog((e) => {
      setLogs((l) => [...l.slice(-300), e]);
    });
    return () => {
      clearInterval(t);
      unsub.then((f) => f());
    };
  }, [refresh, refreshKey]);

  const runProvision = async () => {
    setProvisioning(true);
    setProvisionErr(null);
    setLogs([]);
    try {
      await api.provision();
    } catch (e) {
      setProvisionErr(String(e));
    } finally {
      setProvisioning(false);
      refresh();
    }
  };

  const rep: ProvisionReport | null = env?.report ?? null;
  const gpu = env?.gpu ?? null;
  const vramPct = gpu && gpu.vram_total_mb > 0 ? Math.round((1 - gpu.vram_free_mb / gpu.vram_total_mb) * 100) : 0;
  const freeGb = gpu ? gpu.vram_free_mb / 1024 : 0;
  const totalGb = gpu ? gpu.vram_total_mb / 1024 : 0;

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-100">Dashboard</h1>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={() => setRefreshKey((k) => k + 1)}>
            ↻ Refresh
          </Button>
          <Button onClick={runProvision} disabled={provisioning}>
            {provisioning ? <Spinner label="Provisioning…" /> : "⚡ Provision WSL"}
          </Button>
        </div>
      </div>

      {provisionErr && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{provisionErr}</div>
      )}

      {/* Health row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <CardTitle>WSL</CardTitle>
          {env ? (
            <div className="space-y-1.5 text-sm">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${env.wsl_ok ? "bg-emerald-400" : "bg-red-400"}`} />
                <span className="font-medium text-slate-200">{env.distro}</span>
              </div>
              <div className="text-xs text-slate-500">
                {env.wsl_ok ? "responsive" : "unreachable"} ·{" "}
                {env.apt_based ? "apt-based" : "not apt-based"}
              </div>
              <div className="text-xs text-slate-500">
                vLLM:{" "}
                {rep?.vllm_version ? (
                  <span className="text-emerald-300">{rep.vllm_version}</span>
                ) : (
                  <span className="text-amber-300">not installed</span>
                )}
              </div>
              {!env.provisioned && (
                <div className="text-xs text-amber-300">Not provisioned yet — hit “Provision WSL”.</div>
              )}
            </div>
          ) : (
            <Spinner />
          )}
        </Card>

        <Card>
          <CardTitle>GPU</CardTitle>
          {gpu ? (
            <div className="space-y-1.5 text-sm">
              <div className="font-medium text-slate-200">{gpu.name}</div>
              <div className="text-xs text-slate-500">
                {fmtNum(gpu.vram_free_mb / 1024, 1)} / {fmtNum(gpu.vram_total_mb / 1024, 0)} GB free · util{" "}
                {gpu.util_percent}%
              </div>
              <div className="text-xs text-slate-500">
                est. bandwidth{" "}
                <span className="text-slate-300">{env?.gpu_bandwidth_gbs ?? "?"} GB/s</span>
                {!env?.gpu_bw_known && <span className="text-amber-300"> (default)</span>}
              </div>
              <div className="text-xs text-slate-500">
                CUDA: {rep?.torch_version ?? "n/a"} · bf16 {rep?.bf16_supported ? "✓" : "✗"}
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">nvidia-smi not visible inside WSL.</div>
          )}
        </Card>

        <Card>
          <CardTitle>Servers</CardTitle>
          {env ? (
            <div className="space-y-1.5 text-sm">
              <div>
                <span className="font-medium text-slate-200">{env.servers_running}</span>{" "}
                <span className="text-slate-500">running</span>
              </div>
              {env.running_weight_gb > 0 && (
                <div className="text-xs text-slate-500">
                  est. weights: <span className="text-slate-300">{fmtGB(env.running_weight_gb)}</span> on {totalGb.toFixed(0)}{" "}
                  GB GPU
                </div>
              )}
              {freeGb > 0 && env.running_weight_gb > 0.05 && (
                <div className="text-xs">
                  <span className={env.running_weight_gb > freeGb ? "text-red-300" : "text-emerald-300"}>
                    {env.running_weight_gb > freeGb ? "over budget" : "fits"}
                  </span>{" "}
                  <span className="text-slate-600">(weights vs {freeGb.toFixed(1)} GB free)</span>
                </div>
              )}
            </div>
          ) : (
            <Spinner />
          )}
        </Card>
      </div>

      {/* VRAM gauge + log */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card className="flex items-center justify-center">
          <Gauge pct={vramPct} label={gpu ? `${vramPct}%` : "?"} sub={gpu ? `${fmtNum(freeGb, 1)} GB free / ${fmtNum(totalGb, 0)} GB` : "no GPU"} />
        </Card>
        <Card>
          <CardTitle
            right={
              provisioning ? <Spinner label="running…" /> : <span className="text-[11px] text-slate-600">{logs.length} lines</span>
            }
          >
            Provisioning log
          </CardTitle>
          <div className="h-64 overflow-y-auto rounded-md bg-black/30 p-3 font-mono text-[11px] leading-relaxed">
            {logs.length === 0 ? (
              <span className="text-slate-600">
                {provisioning
                  ? "waiting for output…"
                  : env?.provisioned
                    ? "Provisioned ✓ — re-run to upgrade vLLM or repair."
                    : "Run “Provision WSL” — first run installs apt basics, a uv venv (CPython 3.12) and vLLM CUDA wheels."}
              </span>
            ) : (
              logs.map((l, i) => (
                <div key={i}>
                  <span className="text-indigo-400/70">[{l.phase}]</span>{" "}
                  <span className="text-slate-300">{l.line}</span>
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </Card>
      </div>
    </div>
  );
}