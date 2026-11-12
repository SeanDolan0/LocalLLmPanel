import { useCallback, useEffect, useRef, useState } from "react";
import { FirstRunDiagnosticCard } from "../components/FirstRunDiagnosticCard";
import { SmartScreenTrustModal } from "../components/SmartScreenTrustModal";
import { ErrorAlert } from "../components/ErrorAlert";
import { api, events, fmtGB, fmtNum } from "../api";
import { Button, Card, CardTitle, Collapsible, Gauge, Spinner, InsetGroup, SettingsRow, SquircleIcon, Badge } from "../ui";
import { Sparkline } from "../components/Sparkline";
import type { EnvStatus, ProvisionReport, SystemMetricPoint, WslLogEvent, FlashInferReady } from "../types";

function RefreshIcon() {
  return (
    <svg className="h-3.5 w-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function ServerStackIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="8" x="2" y="2" rx="2" />
      <rect width="20" height="8" x="2" y="14" rx="2" />
      <line x1="6" x2="6.01" y1="6" y2="6" />
      <line x1="6" x2="6.01" y1="18" y2="18" />
    </svg>
  );
}

function GpuChipIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="16" height="16" x="4" y="4" rx="2" />
      <rect width="6" height="6" x="9" y="9" rx="1" />
      <path d="M15 2v2" /><path d="M15 20v2" /><path d="M2 15h2" /><path d="M2 9h2" />
      <path d="M20 15h2" /><path d="M20 9h2" /><path d="M9 2v2" /><path d="M9 20v2" />
    </svg>
  );
}

function TerminalIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" x2="20" y1="19" y2="19" />
    </svg>
  );
}

function CpuIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="16" height="16" x="4" y="4" rx="2" />
      <path d="M9 9h6v6H9z" />
      <path d="M9 2v2" /><path d="M9 20v2" /><path d="M20 9h2" /><path d="M2 9h2" />
      <path d="M15 2v2" /><path d="M15 20v2" /><path d="M20 15h2" /><path d="M2 15h2" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg className="h-5 w-5 text-warn shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

export default function Dashboard() {
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [systemSeries, setSystemSeries] = useState<SystemMetricPoint[]>([]);
  const [logs, setLogs] = useState<WslLogEvent[]>([]);
  const [provisioning, setProvisioning] = useState(false);
  const [provisionErr, setProvisionErr] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [installingLlamacpp, setInstallingLlamacpp] = useState(false);
  const [llamacppErr, setLlamacppErr] = useState<string | null>(null);
  const [updatingVllm, setUpdatingVllm] = useState(false);
  const [vllmUpdateMsg, setVllmUpdateMsg] = useState<string | null>(null);
  const [flashInferReady, setFlashInferReady] = useState<FlashInferReady | null>(null);
  const [checkingFlashInfer, setCheckingFlashInfer] = useState(false);
  const [showTrustModal, setShowTrustModal] = useState(false);

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
      .catch(() =>
        setEnv((e) => e ?? {
          wsl_ok: false,
          distro: "?",
          apt_based: false,
          provisioned: false,
          report: null,
          gpu: null,
          servers_running: 0,
          running_weight_gb: 0,
          gpu_bandwidth_gbs: 700,
          gpu_bw_known: false,
        })
      );

    api
      .systemMetricsSeries()
      .then(setSystemSeries)
      .catch(() => {});
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

  const checkFlashInfer = async () => {
    setCheckingFlashInfer(true);
    try {
      const ready = await api.checkFlashInferReady();
      setFlashInferReady(ready);
    } catch (e) {
      console.error("FlashInfer check failed:", e);
    } finally {
      setCheckingFlashInfer(false);
    }
  };

  const installCudaTools = async () => {
    if (!confirm("This will install CUDA build tools (gcc, python3.12-dev, ninja-build, CUDA toolkit) in WSL. It downloads several GB and may take 5-15 minutes. Continue?")) {
      return;
    }
    setCheckingFlashInfer(true);
    try {
      await api.installCudaBuildTools();
      const ready = await api.checkFlashInferReady();
      setFlashInferReady(ready);
    } catch (e) {
      alert(`CUDA build tools installation failed: ${e}`);
    } finally {
      setCheckingFlashInfer(false);
    }
  };

  const installLlamacpp = async () => {
    setInstallingLlamacpp(true);
    setLlamacppErr(null);
    try {
      await api.installLlamacpp();
      refresh();
    } catch (e) {
      setLlamacppErr(String(e));
    } finally {
      setInstallingLlamacpp(false);
    }
  };

  const updateVllm = async () => {
    setUpdatingVllm(true);
    setVllmUpdateMsg(null);
    setProvisionErr(null);
    setLogs([]);
    try {
      const res = await api.updateVllm();
      setVllmUpdateMsg(
        `vLLM ${res.previous_version ?? "not installed"} → ${res.version ?? "unknown"}`
      );
    } catch (e) {
      setVllmUpdateMsg(`vLLM update failed: ${e}`);
    } finally {
      setUpdatingVllm(false);
      refresh();
    }
  };

  const rep: ProvisionReport | null = env?.report ?? null;
  const gpu = env?.gpu ?? null;
  const vramPct = gpu && gpu.vram_total_mb > 0 ? Math.round((1 - gpu.vram_free_mb / gpu.vram_total_mb) * 100) : 0;
  const freeGb = gpu ? gpu.vram_free_mb / 1024 : 0;
  const totalGb = gpu ? gpu.vram_total_mb / 1024 : 0;

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      {/* Top Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Dashboard</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Hardware allocation, local runtime engines, and system telemetry
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <Button variant="ghost" onClick={() => setRefreshKey((k) => k + 1)}>
            <RefreshIcon />
            <span>Refresh</span>
          </Button>
          {(!env || !env.provisioned) && (
            <Button onClick={runProvision} disabled={provisioning}>
              {provisioning ? <Spinner label="Provisioning WSL…" /> : "Provision WSL"}
            </Button>
          )}
        </div>
      </div>

      {provisionErr && (
        <ErrorAlert
          source="provisioning"
          error={provisionErr}
          onDismiss={() => setProvisionErr(null)}
        />
      )}
      {llamacppErr && (
        <ErrorAlert
          source="provisioning"
          error={llamacppErr}
          title="llama.cpp Engine Setup Error"
          onDismiss={() => setLlamacppErr(null)}
        />
      )}

      {/* Unified First-Run Diagnostic & Hardware Validation */}
      <FirstRunDiagnosticCard onOpenTrustModal={() => setShowTrustModal(true)} />

      {/* VRAM Pressure Alert */}
      {gpu && (() => {
        const alloc = env?.allocation;
        const activeWeight = alloc ? alloc.total_weight_gb : (env?.running_weight_gb || 0);
        const mainWeight = alloc ? alloc.main_weight_gb : (env?.running_model_weight_gb || 0);
        const specWeight = alloc ? alloc.spec_weight_gb : (env?.running_speculative_weight_gb || 0);
        const overheadGb = 1.0;
        const usableCapacityGb = Math.max(0, totalGb - overheadGb);
        const loadedIntoRamGb = alloc
          ? alloc.ram_spill_gb
          : (env?.running_ram_weight_gb != null && env.running_ram_weight_gb > 0
              ? env.running_ram_weight_gb
              : (totalGb > 0 ? Math.max(0, activeWeight - usableCapacityGb) : 0));
        const isLoadedIntoRam = alloc ? alloc.ram_spill_gb > 0.01 : (activeWeight > 0 && loadedIntoRamGb > 0.01);
        const isLowHeadroom = alloc ? alloc.is_low_vram : (freeGb < 1.0);

        if (!isLoadedIntoRam && !isLowHeadroom) return null;

        return (
          <div className="rounded-2xl border border-warn/30 bg-warn/10 p-4 text-amber-200 shadow-sm flex items-start gap-3.5">
            <AlertIcon />
            <div>
              <div className="font-semibold text-amber-100 text-sm">
                {isLoadedIntoRam
                  ? "VRAM Limit Warning — Weights Loaded into System RAM"
                  : "Low VRAM Available Warning"}
              </div>
              <div className="text-xs text-amber-200/90 mt-1 leading-relaxed">
                {isLoadedIntoRam ? (
                  <>
                    {activeWeight > usableCapacityGb ? (
                      <>
                        Active model weights ({fmtGB(activeWeight)}
                        {specWeight > 0 ? ` [model: ${fmtGB(mainWeight)}, draft: ${fmtGB(specWeight)}]` : ""}
                        ) exceed dedicated GPU VRAM capacity ({fmtNum(totalGb, 1)} GB total, ~{fmtNum(usableCapacityGb, 1)} GB usable).{" "}
                      </>
                    ) : (
                      <>
                        Model weights are partially running from system RAM to maintain GPU context and driver headroom.{" "}
                      </>
                    )}
                    <strong>Model weights loaded into system RAM by ~{fmtGB(loadedIntoRamGb)}.</strong>{" "}
                    {alloc?.warning_message && alloc.warning_message.includes("layers on GPU") ? (
                      <span className="block mt-1 font-medium text-amber-100">{alloc.warning_message}</span>
                    ) : null}
                    Running weights from system RAM will reduce generation tokens/sec.
                  </>
                ) : (
                  `Only ${fmtNum(freeGb, 2)} GB free VRAM remaining. Current weights fit within GPU memory. Starting an additional server or expanding context length may cause CUDA out-of-memory errors.`
                )}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Primary Inset Groups (Apple/Fluent Settings Layout) */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* Left Column: Inference Engines & Servers */}
        <div>
          <InsetGroup header="Inference Engines & Runtimes">
            {/* WSL / vLLM Row */}
            <SettingsRow
              icon={<SquircleIcon color="green" icon={<TerminalIcon />} />}
              title="vLLM Inference Engine"
              subtitle={
                env ? (
                  <span>
                    {env.distro} ({env.wsl_ok ? "WSL2 running" : "standby"}) ·{" "}
                    {rep?.vllm_version ? (
                      <span className="text-emerald-400 font-mono">v{rep.vllm_version}</span>
                    ) : (
                      <span className="text-amber-400">Not provisioned</span>
                    )}
                  </span>
                ) : "Loading WSL2 environment…"
              }
              right={
                env?.provisioned ? (
                  <Badge color="emerald">Ready</Badge>
                ) : (
                  <Button variant="ghost" className="text-xs h-7 px-2" onClick={runProvision} disabled={provisioning}>
                    Provision
                  </Button>
                )
              }
            />

            {/* llama.cpp Row */}
            <SettingsRow
              icon={<SquircleIcon color="blue" icon={<ServerStackIcon />} />}
              title="llama.cpp Native Engine"
              subtitle={
                env ? (
                  <span>
                    {env.llamacpp_installed ? "Native Windows backend" : "Not installed"} ·{" "}
                    <span className={env.llamacpp_cuda_available ? "text-emerald-400" : "text-slate-400"}>
                      {env.llamacpp_cuda_available ? "CUDA acceleration active" : "CPU only"}
                    </span>
                  </span>
                ) : "Checking native backend…"
              }
              right={
                env?.llamacpp_installed ? (
                  <Badge color="emerald">Installed</Badge>
                ) : (
                  <Button variant="ghost" className="text-xs h-7 px-2" onClick={installLlamacpp} disabled={installingLlamacpp}>
                    Install
                  </Button>
                )
              }
            />

            {/* Active Servers Summary */}
            <SettingsRow
              icon={<SquircleIcon color="purple" icon={<ServerStackIcon />} />}
              title="Active Server Processes"
              subtitle={
                env && env.servers_running > 0 ? (
                  <span>
                    {env.servers_running} active {env.servers_running === 1 ? "engine" : "engines"} ·{" "}
                    <span className="font-mono text-slate-300">{fmtGB(env.running_weight_gb)}</span> weights
                  </span>
                ) : (
                  "No local inference servers currently running"
                )
              }
              right={
                <Badge color={env && env.servers_running > 0 ? "emerald" : "slate"}>
                  {env ? `${env.servers_running} Running` : "0"}
                </Badge>
              }
            />
          </InsetGroup>
        </div>

        {/* Right Column: Hardware & Host Resources */}
        <div>
          <InsetGroup header="Hardware & Memory Capacity">
            {/* GPU Row */}
            <SettingsRow
              icon={<SquircleIcon color="orange" icon={<GpuChipIcon />} />}
              title={gpu ? gpu.name : "NVIDIA Graphics Card"}
              subtitle={
                gpu ? (
                  <span>
                    {fmtNum(gpu.vram_free_mb / 1024, 1)} GB free of {fmtNum(gpu.vram_total_mb / 1024, 0)} GB ·{" "}
                    {env?.gpu_bandwidth_gbs ?? 700} GB/s bandwidth
                  </span>
                ) : (
                  "NVIDIA GPU not detected"
                )
              }
              right={
                gpu ? (
                  <span className="font-mono text-xs text-slate-300 bg-surface-3 px-2 py-0.5 rounded-md border border-edge">
                    {gpu.util_percent}% util
                  </span>
                ) : (
                  <Badge color="slate">N/A</Badge>
                )
              }
            />

            {/* CPU & RAM Row */}
            <SettingsRow
              icon={<SquircleIcon color="slate" icon={<CpuIcon />} />}
              title={env?.cpu_name ? env.cpu_name.replace("(R)", "").replace("(TM)", "") : "Host Processor & RAM"}
              subtitle={
                env ? (
                  <span>
                    {env.cpu_cores ? `${env.cpu_cores} cores · ` : ""}
                    {env.available_ram_gb ? fmtNum(env.available_ram_gb, 1) : "?"} / {env.total_ram_gb ? fmtNum(env.total_ram_gb, 1) : "?"} GB RAM available
                  </span>
                ) : (
                  "Detecting host hardware…"
                )
              }
              right={
                env?.ram_bandwidth_gbps ? (
                  <span className="font-mono text-xs text-slate-400">
                    {env.ram_bandwidth_gbps} GB/s
                  </span>
                ) : null
              }
            />
          </InsetGroup>
        </div>
      </div>

      {/* GPU & VRAM Live Telemetry Card */}
      <Card className="p-5 space-y-4">
        <CardTitle right={<span className="text-xs text-slate-400 font-mono">5-minute window</span>}>
          Real-time Hardware Telemetry
        </CardTitle>
        <div className="flex flex-col md:flex-row items-center justify-around gap-8">
          <Gauge
            pct={vramPct}
            label={gpu ? `${vramPct}%` : "—"}
            sub={gpu ? `${fmtNum(freeGb, 1)} GB free / ${fmtNum(totalGb, 0)} GB` : "No GPU"}
          />
          <div className="flex-1 w-full space-y-4">
            <Sparkline
              label="VRAM Allocation"
              data={systemSeries.map((p) => p.vram_used_mb / 1024)}
              unit="GB"
              min={0}
              max={totalGb || 24}
              color="#0a84ff"
              currentValue={gpu ? `${((gpu.vram_total_mb - gpu.vram_free_mb) / 1024).toFixed(1)} GB` : undefined}
              formatValue={(v) => `${v.toFixed(1)}`}
            />
            <Sparkline
              label="GPU Core Utilization"
              data={systemSeries.map((p) => p.gpu_util_pct)}
              unit="%"
              min={0}
              max={100}
              color="#30d158"
              currentValue={gpu ? `${gpu.util_percent}%` : undefined}
              formatValue={(v) => `${Math.round(v)}`}
            />
          </div>
        </div>
      </Card>

      {/* Setup & Diagnostics Collapsible */}
      <Collapsible
        key={provisioning ? "diag-open" : "diag-closed"}
        defaultOpen={provisioning || !!provisionErr}
        title="Environment Diagnostics & Maintenance"
        subtitle="WSL provisioning logs, FlashInfer JIT build tools, and engine updates"
      >
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {/* Provisioning Logs */}
          <Card className="p-4 space-y-3">
            <CardTitle
              right={
                provisioning ? (
                  <Spinner label="Running…" />
                ) : (
                  <span className="text-xs text-slate-500 font-mono">{logs.length} lines</span>
                )
              }
            >
              WSL Execution Log
            </CardTitle>
            <div className="h-64 overflow-y-auto rounded-xl bg-surface p-3 font-mono text-[11px] leading-relaxed border border-edge-subtle">
              {logs.length === 0 ? (
                <span className="text-slate-500">
                  {provisioning
                    ? "Waiting for output stream…"
                    : env?.provisioned
                    ? "Environment provisioned. Re-run to verify or repair packages."
                    : "Hit 'Provision WSL' to initialize apt packages, python virtualenv, and vLLM wheels."}
                </span>
              ) : (
                logs.map((l, i) => (
                  <div key={i}>
                    <span className="text-accent/80">[{l.phase}]</span>{" "}
                    <span className="text-slate-300">{l.line}</span>
                  </div>
                ))
              )}
              <div ref={logEndRef} />
            </div>
            {env?.provisioned && !provisioning && (
              <div className="flex flex-wrap gap-2 pt-1">
                <Button variant="ghost" className="text-xs h-7 px-2.5" onClick={runProvision}>
                  Verify / Repair WSL
                </Button>
                <Button variant="ghost" className="text-xs h-7 px-2.5" onClick={updateVllm} disabled={updatingVllm}>
                  {updatingVllm ? <Spinner label="Updating…" /> : "Update vLLM"}
                </Button>
                {env && (!env.llamacpp_installed || !env.llamacpp_cuda_available) && (
                  <Button variant="ghost" className="text-xs h-7 px-2.5" onClick={installLlamacpp} disabled={installingLlamacpp}>
                    {installingLlamacpp ? <Spinner label="Installing…" /> : env.llamacpp_installed ? "Rebuild CUDA" : "Install llama.cpp"}
                  </Button>
                )}
              </div>
            )}
            {(updatingVllm || vllmUpdateMsg) && (
              <div className="text-xs text-slate-400 mt-1">
                {updatingVllm ? "Upgrading vLLM in WSL — this may take a few minutes…" : vllmUpdateMsg}
              </div>
            )}
          </Card>

          {/* FlashInfer JIT Tools */}
          <Card className="p-4 space-y-3">
            <CardTitle
              right={
                <div className="flex items-center gap-1.5">
                  <Button variant="ghost" className="text-xs h-7 px-2" onClick={checkFlashInfer} disabled={checkingFlashInfer}>
                    {checkingFlashInfer ? <Spinner label="Checking…" /> : "Check"}
                  </Button>
                  {flashInferReady && (!flashInferReady.nvcc || !flashInferReady.gcc || !flashInferReady.ninja || !flashInferReady.python_dev) && (
                    <Button variant="primary" className="text-xs h-7 px-2.5" onClick={installCudaTools} disabled={checkingFlashInfer}>
                      Install CUDA Tools
                    </Button>
                  )}
                </div>
              }
            >
              FlashInfer JIT Build Tools
            </CardTitle>
            {flashInferReady ? (
              <div className="space-y-2 text-xs">
                {[
                  ["nvcc (NVIDIA Compiler)", "nvcc"],
                  ["gcc (GNU C Compiler)", "gcc"],
                  ["ninja (Build System)", "ninja"],
                  ["python3.12-dev (Headers)", "python_dev"],
                ].map(([label, key]) => {
                  const present = Boolean(flashInferReady[key as keyof typeof flashInferReady]);
                  return (
                    <div key={key} className="flex items-center justify-between py-1 border-b border-edge-subtle">
                      <span className="font-medium text-slate-300">{label}</span>
                      <Badge color={present ? "emerald" : "red"}>{present ? "Found" : "Missing"}</Badge>
                    </div>
                  );
                })}
                {flashInferReady.cuda_home && (
                  <div className="flex items-center justify-between py-1 border-b border-edge-subtle">
                    <span className="text-slate-400">CUDA_HOME</span>
                    <span className="font-mono text-[10px] text-slate-300 truncate max-w-[220px]">{flashInferReady.cuda_home}</span>
                  </div>
                )}
                {flashInferReady.torch_cuda_version && (
                  <div className="flex items-center justify-between py-1">
                    <span className="text-slate-400">PyTorch CUDA Target</span>
                    <span className="font-mono text-xs text-accent">{flashInferReady.torch_cuda_version}</span>
                  </div>
                )}
                {!flashInferReady.nvcc || !flashInferReady.gcc || !flashInferReady.ninja || !flashInferReady.python_dev ? (
                  <div className="mt-2.5 p-3 rounded-xl bg-warn/10 border border-warn/25 text-xs text-amber-200 leading-relaxed">
                    Missing build tools for JIT compilation. Click <strong>Install CUDA Tools</strong> to provision gcc, ninja, and python dev packages automatically in WSL.
                  </div>
                ) : (
                  <div className="mt-2.5 p-3 rounded-xl bg-good/10 border border-good/25 text-xs text-emerald-200">
                    All FlashInfer JIT build tools present and configured.
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-slate-400 py-6 text-center">
                Click <strong>Check</strong> to inspect nvcc, gcc, ninja, and python dev tools in WSL.
              </div>
            )}
          </Card>
        </div>
      </Collapsible>

      <SmartScreenTrustModal isOpen={showTrustModal} onClose={() => setShowTrustModal(false)} />
    </div>
  );
}