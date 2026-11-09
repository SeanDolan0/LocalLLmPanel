import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, CardTitle, Spinner } from "../ui";
import type { CompatibilityTestReport, ServerListRow } from "../types";

interface ServerCompatibilityModalProps {
  server: ServerListRow;
  isOpen: boolean;
  onClose: () => void;
}

export function ServerCompatibilityModal({
  server,
  isOpen,
  onClose,
}: ServerCompatibilityModalProps) {
  const [report, setReport] = useState<CompatibilityTestReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const runTest = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.serversCompatibilityTest(server.def.id);
      setReport(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }, [server.def.id]);

  useEffect(() => {
    if (isOpen && !report && !running) {
      void runTest();
    }
  }, [isOpen, report, running, runTest]);

  if (!isOpen) return null;

  const handleCopyReport = () => {
    if (!report) return;
    const summary = [
      `=== Server Compatibility Report: ${server.def.name} ===`,
      `Model: ${report.model_id}`,
      `Backend: ${report.backend}`,
      `Status: ${report.passed_count}/${report.total_count} Passed (${report.all_passed ? "Full Conformance" : "Partial Support"})`,
      `Timestamp: ${report.timestamp}`,
      ``,
      ...report.items.map(
        (it) =>
          `[${it.passed ? "PASS" : "FAIL"}] ${it.name} (${it.latency_ms} ms)\n  Desc: ${it.description}${
            it.details ? `\n  Details: ${it.details}` : ""
          }${it.error ? `\n  Error: ${it.error}` : ""}`
      ),
    ].join("\n");

    void navigator.clipboard?.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto">
      <Card className="w-full max-w-2xl p-5 space-y-4 shadow-2xl border-indigo-500/30 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between border-b border-edge pb-3 shrink-0">
          <div className="flex items-center gap-2.5">
            <span className="text-xl select-none">🧪</span>
            <div>
              <CardTitle>Per-Server Compatibility Test</CardTitle>
              <div className="text-xs text-slate-400 mt-0.5">
                {server.def.name} · port {server.def.port} · {server.def.backend}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg font-bold"
            title="Close"
          >
            ✕
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300">
            <strong>Test execution failed:</strong> {error}
          </div>
        )}

        {/* Results summary header */}
        {report && (
          <div className="flex items-center justify-between bg-surface-3/50 p-3 rounded-lg border border-white/5 shrink-0">
            <div className="flex items-center gap-2">
              <Badge color={report.all_passed ? "emerald" : report.passed_count > 0 ? "amber" : "red"}>
                {report.passed_count} / {report.total_count} Passed
              </Badge>
              <span className="text-xs text-slate-200 font-medium">
                {report.all_passed
                  ? "Full Agentic & OpenAI Conformance"
                  : "Partial Support (check failed features below)"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                className="text-xs h-7 px-2"
                onClick={handleCopyReport}
                disabled={running}
              >
                {copied ? "✓ Copied" : "Copy Report"}
              </Button>
              <Button
                variant="subtle"
                className="text-xs h-7 px-2"
                onClick={() => void runTest()}
                disabled={running}
              >
                {running ? <Spinner label="Testing…" /> : "↻ Re-run Test"}
              </Button>
            </div>
          </div>
        )}

        {/* Test items list */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {running && !report && (
            <div className="p-8 text-center text-sm text-slate-400 flex flex-col items-center justify-center gap-3">
              <Spinner label="Executing live test battery against server endpoint…" />
              <div className="text-xs text-slate-500">
                Testing basic chat, streaming token delivery, tool calling, structured JSON output, and system prompt…
              </div>
            </div>
          )}

          {report?.items.map((item) => (
            <div
              key={item.id}
              className={`rounded-lg border p-3 text-xs transition-all ${
                item.passed
                  ? "border-emerald-500/20 bg-emerald-500/5 hover:border-emerald-500/30"
                  : "border-red-500/30 bg-red-500/5 hover:border-red-500/40"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className={`text-sm ${item.passed ? "text-emerald-400" : "text-red-400"}`}>
                    {item.passed ? "✓" : "✗"}
                  </span>
                  <div>
                    <div className="font-semibold text-slate-200 text-xs flex items-center gap-2">
                      <span>{item.name}</span>
                      <span className="font-mono text-[11px] text-slate-400 font-normal">
                        {item.latency_ms} ms
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">{item.description}</div>
                  </div>
                </div>
                <Badge color={item.passed ? "emerald" : "red"}>
                  {item.passed ? "Passed" : "Failed"}
                </Badge>
              </div>

              {item.details && (
                <div className="mt-2 pl-5 text-[11px] text-emerald-300 font-mono truncate">
                  {item.details}
                </div>
              )}

              {item.error && (
                <div className="mt-2 pl-5 text-[11px] text-red-300 font-mono whitespace-pre-wrap bg-red-950/20 p-2 rounded border border-red-500/20">
                  {item.error}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="flex justify-between items-center pt-2 border-t border-edge shrink-0 text-xs text-slate-400">
          <span>Tests query live endpoint: <code>/v1/chat/completions</code></span>
          <Button variant="primary" onClick={onClose}>
            Close
          </Button>
        </div>
      </Card>
    </div>
  );
}
