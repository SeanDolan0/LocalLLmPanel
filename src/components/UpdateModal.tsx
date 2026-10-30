import { useState, useEffect } from "react";
import { api, fmtBytes, fmtTransferRate } from "../api";
import type { AppUpdateInfo, AppUpdateProgress } from "../types";
import { Button, Badge } from "../ui";

interface UpdateModalProps {
  info: AppUpdateInfo;
  isOpen: boolean;
  onClose: () => void;
}

export default function UpdateModal({ info, isOpen, onClose }: UpdateModalProps) {
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<AppUpdateProgress | null>(null);
  const [launchSuccess, setLaunchSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onAppUpdateProgress((p) => {
      setProgress(p);
      if (p.phase === "error") {
        setError(p.error || "Update download failed");
        setDownloading(false);
      }
    }).then((fn) => {
      unlisten = fn;
    });

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  if (!isOpen) return null;

  const handleStartUpdate = async () => {
    setError(null);
    setLaunchSuccess(null);
    setDownloading(true);
    setProgress({
      percent: 0,
      downloaded_bytes: 0,
      total_bytes: info.asset_size,
      speed_bps: 0,
      phase: "downloading",
    });

    try {
      const res = await api.downloadAndInstallUpdate();
      setLaunchSuccess(res.message);
      setDownloading(false);
    } catch (e) {
      setError(String(e));
      setDownloading(false);
    }
  };

  const handleViewGithub = () => {
    if (info.html_url) {
      api.openUrl(info.html_url);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-xl rounded-xl border border-edge bg-surface-1 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-6 pb-4 border-b border-edge/60 flex items-start justify-between bg-surface-2/40">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 text-black font-bold shadow-md shadow-indigo-500/20">
              <span className="text-xl">✨</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-100">
                  Update Available
                </h2>
                <Badge color="cyan">{info.latest_version}</Badge>
              </div>
              <div className="text-xs text-slate-400 mt-0.5">
                Current: <span className="text-slate-300 font-medium">{info.current_version}</span>
                {info.published_at && (
                  <span className="ml-2 opacity-80">
                    · Published {new Date(info.published_at).toLocaleDateString()}
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={downloading}
            className="text-slate-400 hover:text-slate-200 transition-colors p-1 rounded-md hover:bg-surface-3 disabled:opacity-30"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-4 overflow-y-auto flex-1 text-sm">
          {/* Release Title */}
          {info.release_name && info.release_name !== info.latest_version && (
            <div className="text-sm font-semibold text-indigo-300">
              {info.release_name}
            </div>
          )}

          {/* Release Notes */}
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Release Notes
            </div>
            <div className="rounded-lg border border-edge/60 bg-surface-2/60 p-3.5 text-xs text-slate-300 font-mono whitespace-pre-wrap max-h-48 overflow-y-auto leading-relaxed select-text">
              {info.release_notes || "No release notes provided."}
            </div>
          </div>

          {/* Asset & Size Info */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 rounded-lg bg-surface-2/30 p-2.5 border border-edge/40">
            <div>
              Package: <span className="text-slate-200 font-mono font-medium">{info.asset_name || "Executable"}</span>
              {info.is_installer && (
                <span className="ml-1.5 text-emerald-400 font-medium">(Installer)</span>
              )}
            </div>
            {info.asset_size && (
              <div>Size: <span className="text-slate-200 font-medium">{fmtBytes(info.asset_size)}</span></div>
            )}
          </div>

          {/* Progress / Downloading State */}
          {downloading && progress && (
            <div className="space-y-2 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-3.5">
              <div className="flex items-center justify-between text-xs font-medium">
                <span className="text-indigo-200 capitalize">
                  {progress.phase === "verifying"
                    ? "Verifying checksum (SHA256)..."
                    : progress.phase === "ready"
                    ? "Launching installer..."
                    : "Downloading update..."}
                </span>
                <span className="text-indigo-300">{progress.percent.toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-surface-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-200"
                  style={{ width: `${Math.min(100, Math.max(0, progress.percent))}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>
                  {fmtBytes(progress.downloaded_bytes)}
                  {progress.total_bytes ? ` / ${fmtBytes(progress.total_bytes)}` : ""}
                </span>
                {progress.speed_bps > 0 && <span>{fmtTransferRate(progress.speed_bps)}</span>}
              </div>
            </div>
          )}

          {/* Success notice */}
          {launchSuccess && (
            <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3.5 text-xs text-emerald-300 space-y-1">
              <div className="font-semibold text-emerald-200">✓ Installer Launched</div>
              <div>{launchSuccess}</div>
            </div>
          )}

          {/* Error notice */}
          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3.5 text-xs text-red-300 space-y-1">
              <div className="font-semibold text-red-200">Update Error</div>
              <div>{error}</div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="p-4 px-6 border-t border-edge/60 bg-surface-2/40 flex items-center justify-between">
          <Button variant="ghost" onClick={handleViewGithub} className="text-xs">
            View on GitHub ↗
          </Button>
          <div className="flex items-center gap-2">
            <Button
              variant="subtle"
              onClick={onClose}
              disabled={downloading}
              className="text-xs"
            >
              {launchSuccess ? "Close" : "Later"}
            </Button>
            {!launchSuccess && (
              <Button
                variant="primary"
                onClick={handleStartUpdate}
                disabled={downloading || !info.download_url}
                className="text-xs"
              >
                {downloading ? "Downloading..." : "Download & Install"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
