import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, CardTitle, Spinner } from "../ui";
import type { DiagnosticReport, DiagnosticStep } from "../types";

interface FirstRunDiagnosticCardProps {
  onOpenTrustModal?: () => void;
  className?: string;
}

export function FirstRunDiagnosticCard({
  onOpenTrustModal,
  className = "",
}: FirstRunDiagnosticCardProps) {
  const [report, setReport] = useState<DiagnosticReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({});

  const runDiagnostics = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.diagnosticsRun();
      setReport(res);
      // Auto-expand any failing or warning steps
      const exp: Record<string, boolean> = {};
      res.steps.forEach((s: DiagnosticStep) => {
        if (s.status === "fail" || s.status === "warning") {
          exp[s.id] = true;
        }
      });
      setExpandedSteps(exp);
    } catch (e) {
      console.error("Failed to run diagnostics:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void runDiagnostics();
  }, [runDiagnostics]);

  const toggleStep = (id: string) => {
    setExpandedSteps((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleExportBundle = async () => {
    setExporting(true);
    setExportMsg(null);
    try {
      const bundleJson = await api.diagnosticsExportBundle();
      // Copy to clipboard and trigger download
      await navigator.clipboard?.writeText(bundleJson);
      
      const blob = new Blob([bundleJson], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `local-llm-panel-diagnostics-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setExportMsg("Diagnostic bundle exported & copied (all tokens/keys redacted)");
      setTimeout(() => setExportMsg(null), 4000);
    } catch (e) {
      setExportMsg(`Export failed: ${e}`);
    } finally {
      setExporting(false);
    }
  };

  const statusBadge = (s: DiagnosticStep["status"]) => {
    switch (s) {
      case "pass":
        return <Badge color="emerald">Pass</Badge>;
      case "warning":
        return <Badge color="amber">Warning</Badge>;
      case "fail":
        return <Badge color="red">Fail</Badge>;
      default:
        return <Badge color="slate">Skipped</Badge>;
    }
  };

  const statusIcon = (s: DiagnosticStep["status"]) => {
    switch (s) {
      case "pass":
        return <span className="text-emerald-400 font-bold">✓</span>;
      case "warning":
        return <span className="text-amber-400 font-bold">⚠️</span>;
      case "fail":
        return <span className="text-red-400 font-bold">✗</span>;
      default:
        return <span className="text-slate-500">○</span>;
    }
  };

  return (
    <Card className={`p-5 space-y-4 shadow-sm border-edge ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg select-none">🩺</span>
            <CardTitle>Unified First-Run Diagnostic & System Readiness</CardTitle>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            Windows 10/11 & NVIDIA CUDA hardware validation sequence (8 ordered stages)
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onOpenTrustModal && (
            <Button
              variant="ghost"
              className="text-xs h-7 px-2.5 text-indigo-300 hover:text-indigo-200"
              onClick={onOpenTrustModal}
            >
              🛡️ SmartScreen & Trust
            </Button>
          )}
          <Button
            variant="subtle"
            className="text-xs h-7 px-2.5"
            onClick={handleExportBundle}
            disabled={exporting || loading}
            title="Export a comprehensive diagnostic bundle with all secrets redacted"
          >
            {exporting ? <Spinner label="Exporting…" /> : "📥 Export Redacted Bundle"}
          </Button>
          <Button
            variant="ghost"
            className="text-xs h-7 px-2.5"
            onClick={() => void runDiagnostics()}
            disabled={loading}
          >
            {loading ? <Spinner label="Testing…" /> : "↻ Re-run"}
          </Button>
        </div>
      </div>

      {exportMsg && (
        <div className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 p-2.5 text-xs text-cyan-200 flex items-center justify-between">
          <span>{exportMsg}</span>
          <button onClick={() => setExportMsg(null)} className="text-slate-400 hover:text-white font-bold ml-2">
            ✕
          </button>
        </div>
      )}

      {/* Primary Readiness Banner with Beginner Safety Guarantee */}
      {report && (
        <div
          className={`rounded-xl p-3.5 border flex items-center justify-between gap-3 text-xs ${
            report.passed && report.gpu_usable
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-200"
              : !report.gpu_usable
              ? "bg-red-500/10 border-red-500/40 text-red-200"
              : "bg-amber-500/10 border-amber-500/30 text-amber-200"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <span className="text-xl">
              {report.passed && report.gpu_usable ? "🚀" : !report.gpu_usable ? "🚫" : "⚠️"}
            </span>
            <div>
              <div className="font-semibold text-sm">
                {report.passed && report.gpu_usable
                  ? "System Ready for GPU Acceleration"
                  : !report.gpu_usable
                  ? "GPU Acceleration Unusable — Action Required"
                  : "System Functional with Warnings"}
              </div>
              <div className="opacity-90 mt-0.5">{report.summary}</div>
            </div>
          </div>
          <Badge
            color={
              report.passed && report.gpu_usable
                ? "emerald"
                : !report.gpu_usable
                ? "red"
                : "amber"
            }
          >
            {report.passed && report.gpu_usable
              ? "All Systems Pass"
              : !report.gpu_usable
              ? "GPU Unusable"
              : "Review Steps"}
          </Badge>
        </div>
      )}

      {/* 8-Step Ordered Checklist */}
      <div className="space-y-1.5 pt-1">
        {loading && !report && (
          <div className="py-6 text-center text-xs text-slate-400">
            <Spinner label="Evaluating system stages 1 through 8…" />
          </div>
        )}

        {report?.steps.map((step) => {
          const isExpanded = !!expandedSteps[step.id];
          return (
            <div
              key={step.id}
              className={`rounded-lg border text-xs transition-all ${
                step.status === "fail"
                  ? "border-red-500/30 bg-red-950/20"
                  : step.status === "warning"
                  ? "border-amber-500/30 bg-amber-950/20"
                  : "border-white/5 bg-surface-3/30 hover:border-white/10"
              }`}
            >
              <button
                className="w-full flex items-center justify-between p-2.5 text-left group"
                onClick={() => toggleStep(step.id)}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="w-5 text-center shrink-0">{statusIcon(step.status)}</span>
                  <div className="flex items-center gap-2 truncate">
                    <span className="font-mono text-slate-500 text-[11px] shrink-0">
                      Step {step.order}:
                    </span>
                    <span className="font-medium text-slate-200 truncate">{step.title}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0 ml-2">
                  <span className="font-mono text-[11px] text-slate-300 truncate max-w-[200px]">
                    {step.value}
                  </span>
                  {statusBadge(step.status)}
                  <span className="text-slate-500 text-xs w-3 text-center">
                    {isExpanded ? "▾" : "▸"}
                  </span>
                </div>
              </button>

              {isExpanded && (
                <div className="px-3 pb-3 pt-1 border-t border-white/5 space-y-2 text-[11px] leading-relaxed">
                  {step.details && (
                    <div className="text-slate-300 font-mono text-[11px]">{step.details}</div>
                  )}

                  {step.fix_recommendation && (
                    <div className="rounded-md bg-black/40 border border-amber-500/20 p-2 text-amber-200 space-y-1">
                      <div className="font-semibold text-amber-300">Actionable Fix Recommendation:</div>
                      <div className="whitespace-pre-wrap font-mono text-[11px]">
                        {step.fix_recommendation}
                      </div>
                      <div className="pt-1 flex gap-2">
                        <Button
                          variant="ghost"
                          className="text-[10px] h-5 px-1.5"
                          onClick={() => {
                            if (step.fix_recommendation) {
                              void navigator.clipboard?.writeText(step.fix_recommendation);
                            }
                          }}
                        >
                          Copy command / fix
                        </Button>
                        {step.doc_url && (
                          <Button
                            variant="ghost"
                            className="text-[10px] h-5 px-1.5 text-indigo-300 hover:text-indigo-200"
                            onClick={() => {
                              if (step.doc_url) void api.openUrl(step.doc_url);
                            }}
                          >
                            Open Documentation ↗
                          </Button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
