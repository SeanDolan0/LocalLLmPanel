import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, CardTitle, Spinner } from "../ui";
import type { DiagnosticReport, DiagnosticStep } from "../types";

interface FirstRunDiagnosticCardProps {
  onOpenTrustModal?: () => void;
  className?: string;
}

function CheckIcon() {
  return (
    <svg className="h-4 w-4 text-emerald-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg className="h-4 w-4 text-amber-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg className="h-4 w-4 text-red-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function SkipIcon() {
  return (
    <svg className="h-4 w-4 text-slate-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="8" y1="12" x2="16" y2="12" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg className={`h-3.5 w-3.5 text-slate-400 transition-transform duration-150 ${open ? "rotate-180" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
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
        return <CheckIcon />;
      case "warning":
        return <AlertIcon />;
      case "fail":
        return <XIcon />;
      default:
        return <SkipIcon />;
    }
  };

  return (
    <Card className={`space-y-4 shadow-sm border-edge ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge-subtle pb-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="h-6 w-6 rounded-md bg-accent/15 text-accent flex items-center justify-center">
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
              </svg>
            </div>
            <CardTitle>System Readiness & Diagnostic Sequence</CardTitle>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            Windows 10/11 & NVIDIA CUDA hardware validation sequence (8 stages)
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onOpenTrustModal && (
            <Button
              variant="ghost"
              className="text-xs h-7 px-2.5 text-accent hover:text-accent-hover"
              onClick={onOpenTrustModal}
            >
              SmartScreen & Trust
            </Button>
          )}
          <Button
            variant="subtle"
            className="text-xs h-7 px-2.5"
            onClick={handleExportBundle}
            disabled={exporting || loading}
            title="Export a comprehensive diagnostic bundle with all secrets redacted"
          >
            {exporting ? <Spinner label="Exporting…" /> : "Export Redacted Bundle"}
          </Button>
          <Button
            variant="ghost"
            className="text-xs h-7 px-2.5"
            onClick={() => void runDiagnostics()}
            disabled={loading}
          >
            {loading ? <Spinner label="Testing…" /> : "Re-run"}
          </Button>
        </div>
      </div>

      {exportMsg && (
        <div className="rounded-xl border border-accent/30 bg-accent/10 p-2.5 text-xs text-slate-100 flex items-center justify-between">
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
              ? "bg-good/10 border-good/30 text-emerald-200"
              : !report.gpu_usable
              ? "bg-bad/10 border-bad/30 text-red-200"
              : "bg-warn/10 border-warn/30 text-amber-200"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <span className="shrink-0">
              {report.passed && report.gpu_usable ? (
                <CheckIcon />
              ) : !report.gpu_usable ? (
                <XIcon />
              ) : (
                <AlertIcon />
              )}
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
              className={`rounded-xl border text-xs transition-all ${
                step.status === "fail"
                  ? "border-bad/30 bg-bad/5"
                  : step.status === "warning"
                  ? "border-warn/30 bg-warn/5"
                  : "border-edge-subtle bg-surface-3/40 hover:border-edge"
              }`}
            >
              <button
                className="w-full flex items-center justify-between p-3 text-left group"
                onClick={() => toggleStep(step.id)}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="w-5 flex justify-center shrink-0">{statusIcon(step.status)}</span>
                  <div className="flex items-center gap-2 truncate">
                    <span className="font-mono text-slate-500 text-[11px] shrink-0">
                      Step {step.order}:
                    </span>
                    <span className="font-medium text-slate-200 truncate">{step.title}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 shrink-0 ml-2">
                  <span className="font-mono text-[11px] text-slate-300 truncate max-w-[200px]">
                    {step.value}
                  </span>
                  {statusBadge(step.status)}
                  <ChevronIcon open={isExpanded} />
                </div>
              </button>

              {isExpanded && (
                <div className="px-3.5 pb-3 pt-1 border-t border-edge-subtle space-y-2 text-[11px] leading-relaxed">
                  {step.details && (
                    <div className="text-slate-300 font-mono text-[11px]">{step.details}</div>
                  )}

                  {step.fix_recommendation && (
                    <div className="rounded-lg bg-surface-2 border border-warn/25 p-2.5 text-amber-200 space-y-1.5">
                      <div className="font-semibold text-amber-300">Actionable Fix Recommendation:</div>
                      <div className="whitespace-pre-wrap font-mono text-[11px]">
                        {step.fix_recommendation}
                      </div>
                      <div className="pt-1 flex gap-2">
                        <Button
                          variant="ghost"
                          className="text-[10px] h-6 px-2"
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
                            className="text-[10px] h-6 px-2 text-accent hover:text-accent-hover"
                            onClick={() => {
                              if (step.doc_url) void api.openUrl(step.doc_url);
                            }}
                          >
                            Open Documentation →
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
