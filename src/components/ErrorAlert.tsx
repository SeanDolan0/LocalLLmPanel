import { useState } from "react";
import { Badge, Button } from "../ui";

export type ErrorSource = "download" | "provisioning" | "server" | "system";

interface ErrorAlertProps {
  source: ErrorSource;
  error: string | Error | null;
  title?: string;
  onDismiss?: () => void;
  className?: string;
}

const SOURCE_CONFIG: Record<
  ErrorSource,
  {
    badge: string;
    badgeColor: "indigo" | "amber" | "red" | "slate";
    border: string;
    bg: string;
    text: string;
    icon: string;
    defaultTitle: string;
    hints: string[];
  }
> = {
  download: {
    badge: "Model Download Error",
    badgeColor: "indigo",
    border: "border-indigo-500/40",
    bg: "bg-indigo-950/30",
    text: "text-indigo-200",
    icon: "📥",
    defaultTitle: "Download / Transfer Failure",
    hints: [
      "Check your internet connection and Hugging Face endpoint reachability.",
      "If downloading a gated model (e.g. Llama 3), ensure your HF token has repo permissions in Settings.",
      "Verify sufficient free disk space on the storage drive (see Storage Planning).",
      "Downloads resume automatically from the last verified chunk upon retrying.",
    ],
  },
  provisioning: {
    badge: "Runtime / Provisioning Error",
    badgeColor: "amber",
    border: "border-amber-500/40",
    bg: "bg-amber-950/30",
    text: "text-amber-200",
    icon: "⚡",
    defaultTitle: "WSL / Backend Provisioning Failure",
    hints: [
      "Ensure Windows 10 build 19041+ or Windows 11 with WSL2 virtualization enabled.",
      "Verify that host NVIDIA drivers (>= 525) are installed on Windows (do not install Linux drivers in WSL).",
      "Check apt repository connectivity or run 'wsl --update' in PowerShell.",
      "Review the setup & diagnostics logs below for exact phase failure details.",
    ],
  },
  server: {
    badge: "Server Execution Error",
    badgeColor: "red",
    border: "border-red-500/40",
    bg: "bg-red-950/30",
    text: "text-red-200",
    icon: "🔥",
    defaultTitle: "Inference / Server Startup Failure",
    hints: [
      "Check GPU VRAM headroom. If model weights exceed VRAM, reduce context size or increase CPU offload / swap space.",
      "Check for port conflicts or verify that another server is not holding the requested port.",
      "Inspect server logs on the card to identify OOM (Out Of Memory) or unsupported model architecture errors.",
      "Run the Compatibility Test to check feature support (tool calling, structured output, streaming).",
    ],
  },
  system: {
    badge: "System Error",
    badgeColor: "slate",
    border: "border-slate-500/40",
    bg: "bg-slate-900/40",
    text: "text-slate-200",
    icon: "⚠️",
    defaultTitle: "Unexpected System Error",
    hints: [
      "Check system diagnostics on the Dashboard.",
      "Export the diagnostic bundle to share with maintainers.",
    ],
  },
};

export function ErrorAlert({
  source,
  error,
  title,
  onDismiss,
  className = "",
}: ErrorAlertProps) {
  const [copied, setCopied] = useState(false);
  const [showHints, setShowHints] = useState(false);

  if (!error) return null;

  const config = SOURCE_CONFIG[source];
  const errorMessage = typeof error === "string" ? error : error.message;

  const handleCopy = () => {
    const payload = `[${config.badge}] ${title || config.defaultTitle}\nError: ${errorMessage}`;
    void navigator.clipboard?.writeText(payload);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`rounded-xl border ${config.border} ${config.bg} p-4 shadow-sm transition-all ${className}`}
      role="alert"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <span className="text-xl leading-none select-none">{config.icon}</span>
          <div>
            <div className="flex items-center gap-2">
              <Badge color={config.badgeColor}>{config.badge}</Badge>
              <span className="text-xs font-semibold text-slate-100">
                {title || config.defaultTitle}
              </span>
            </div>
            <div className={`mt-2 font-mono text-xs whitespace-pre-wrap ${config.text} leading-relaxed`}>
              {errorMessage}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <Button
            variant="ghost"
            className="text-[11px] h-7 px-2"
            onClick={handleCopy}
            title="Copy error message to clipboard"
          >
            {copied ? "✓ Copied" : "Copy"}
          </Button>
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="text-slate-400 hover:text-slate-200 text-sm p-1 ml-1 leading-none"
              title="Dismiss"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Actionable contextual hints toggle */}
      <div className="mt-3 border-t border-white/5 pt-2 flex items-center justify-between">
        <button
          onClick={() => setShowHints(!showHints)}
          className="text-[11px] text-slate-400 hover:text-slate-300 flex items-center gap-1 font-medium transition-colors"
        >
          <span>{showHints ? "▾ Hide fix recommendations" : "▸ Show fix recommendations"}</span>
        </button>
      </div>

      {showHints && (
        <ul className="mt-2 space-y-1 text-[11px] text-slate-300/90 pl-4 list-disc">
          {config.hints.map((hint, i) => (
            <li key={i}>{hint}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
