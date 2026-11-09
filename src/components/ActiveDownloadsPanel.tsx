import { useDownloads, type UnifiedDownloadItem } from "../context/DownloadContext";
import { fmtBytes, fmtTransferRate, fmtDuration } from "../api";

interface ActiveDownloadsPanelProps {
  className?: string;
  showCompletedRecent?: boolean;
}

export function ActiveDownloadsPanel({
  className = "",
  showCompletedRecent = true,
}: ActiveDownloadsPanelProps) {
  const {
    downloads,
    activeDownloads,
    activeCount,
    pauseDownload,
    resumeDownload,
    cancelDownload,
    dismissDownload,
    clearCompleted,
    refreshDownloads,
  } = useDownloads();

  const displayList = showCompletedRecent
    ? downloads.filter(
        (d) =>
          d.status === "Downloading" ||
          d.status === "Resolving" ||
          d.status === "Queued" ||
          d.status === "Paused" ||
          d.status === "Verifying" ||
          (d.status === "Completed" && d.finishedAt && Date.now() - d.finishedAt < 15000) ||
          d.status === "Failed"
      )
    : activeDownloads;

  if (displayList.length === 0) {
    return null;
  }

  const hasFinished = displayList.some(
    (d) => d.status === "Completed" || d.status === "Failed" || d.status === "Cancelled"
  );

  return (
    <div
      className={`rounded-xl border border-indigo-500/30 bg-surface-2/95 p-4 shadow-lg space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {activeCount > 0 ? (
            <span className="flex h-2.5 w-2.5 rounded-full bg-indigo-500 animate-pulse" />
          ) : (
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          )}
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Active Downloads {activeCount > 0 ? `(${activeCount} running)` : "(Finished)"}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {hasFinished && (
            <button
              onClick={clearCompleted}
              className="text-[11px] text-slate-400 hover:text-slate-200 transition-colors underline"
              title="Clear completed and failed downloads"
            >
              Clear Finished
            </button>
          )}
          <span className="text-[11px] text-slate-500 font-mono">
            Unified Download Manager
          </span>
        </div>
      </div>

      <div className="space-y-2.5">
        {displayList.map((job) => (
          <DownloadJobItem
            key={job.id}
            job={job}
            onPause={() => pauseDownload(job.id)}
            onResume={() => resumeDownload(job.id)}
            onCancel={() => cancelDownload(job.id)}
            onDismiss={() => dismissDownload(job.id)}
            onRepair={() => refreshDownloads().catch(() => {})}
          />
        ))}
      </div>
    </div>
  );
}

function DownloadJobItem({
  job,
  onPause,
  onResume,
  onCancel,
  onDismiss,
  onRepair,
}: {
  job: UnifiedDownloadItem;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onDismiss: () => void;
  onRepair: () => void;
}) {
  const isCompleted = job.status === "Completed";
  const isFailed = job.status === "Failed";
  const isPaused = job.status === "Paused";
  const isResolving = job.status === "Resolving";
  const isVerifying = job.status === "Verifying";
  const isDownloading = job.status === "Downloading";

  // Safe percentage calculation
  const percent = isCompleted
    ? 100
    : job.totalBytes > 0
    ? Math.min(100, Math.max(0, (job.downloadedBytes / job.totalBytes) * 100))
    : Math.min(100, Math.max(0, job.percent || 0));

  // Safe transferred string
  const sizeString =
    job.totalBytes > 0
      ? `${fmtBytes(job.downloadedBytes)} / ${fmtBytes(job.totalBytes)}`
      : job.downloadedBytes > 0
      ? `${fmtBytes(job.downloadedBytes)} downloaded`
      : isResolving
      ? "Resolving file size..."
      : isVerifying
      ? "Verifying integrity..."
      : "Starting download...";

  return (
    <div className="rounded-lg border border-edge/80 bg-surface-3/50 p-3 space-y-2 transition-all">
      {/* Header row: Model Name, quant/backend tags, verification badges, action buttons */}
      <div className="flex items-center justify-between gap-3 text-xs">
        <div className="min-w-0 flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-slate-100 truncate" title={job.modelId}>
            {job.modelId}
          </span>
          {job.quant && (
            <span className="px-1.5 py-0.5 rounded bg-surface-2 border border-edge/60 text-[10px] font-mono text-slate-300">
              {job.quant}
            </span>
          )}
          <span className="text-[10px] font-mono text-slate-400">
            {job.backend === "llamacpp" ? "llama.cpp GGUF" : "vLLM SafeTensors"}
          </span>
          {job.revision && (
            <span
              className="px-1.5 py-0.5 rounded bg-surface-2 border border-edge/60 text-[10px] font-mono text-slate-400"
              title={`Pinned Commit SHA: ${job.revision}`}
            >
              rev: {job.revision.slice(0, 7)}
            </span>
          )}
          {/* A2: Integrity Verification Status Badges */}
          {job.verificationState === "verified" && (
            <span
              className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 text-[10px] font-mono flex items-center gap-1"
              title="SHA-256 checksum verified against Hugging Face Git LFS metadata before promotion"
            >
              <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              SHA-256 Verified
            </span>
          )}
          {job.verificationState === "size_only" && (
            <span
              className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 text-[10px] font-mono"
              title="Remote SHA-256 unavailable from repository metadata. Verified by file size only."
            >
              Size Checked (no checksum)
            </span>
          )}
          {isVerifying && (
            <span className="px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 text-[10px] font-mono animate-pulse">
              Verifying Checksum...
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Status Badge */}
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
              isDownloading
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                : isVerifying
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 animate-pulse"
                : isPaused
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                : isCompleted
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                : isFailed
                ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                : "bg-surface-2 text-slate-400 border border-edge"
            }`}
          >
            {job.status}
          </span>

          {/* Action buttons (respecting backend capabilities) */}
          {job.canPause && (
            <button
              onClick={onPause}
              className="px-2 py-0.5 rounded text-xs text-amber-300 hover:text-amber-200 hover:bg-amber-500/10 transition-colors"
              title="Pause download"
            >
              Pause
            </button>
          )}

          {job.canResume && (
            <button
              onClick={onResume}
              className="px-2 py-0.5 rounded text-xs text-emerald-300 hover:text-emerald-200 hover:bg-emerald-500/10 transition-colors"
              title="Resume download"
            >
              Resume
            </button>
          )}

          {job.canCancel && (
            <button
              onClick={onCancel}
              className="px-2 py-0.5 rounded text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 transition-colors"
              title="Cancel download"
            >
              Cancel
            </button>
          )}

          {(isCompleted || isFailed) && (
            <button
              onClick={onDismiss}
              className="px-2 py-0.5 rounded text-xs text-slate-400 hover:text-slate-200 hover:bg-surface-2 transition-colors"
              title="Dismiss"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="flex items-center gap-2.5">
        <div className="flex-1 h-2 rounded-full bg-surface-2 overflow-hidden relative">
          <div
            className={`h-full rounded-full transition-all duration-300 ease-out ${
              isCompleted
                ? "bg-emerald-500"
                : isFailed
                ? "bg-rose-500"
                : isPaused
                ? "bg-amber-500"
                : isVerifying
                ? "bg-indigo-400 animate-pulse"
                : isResolving
                ? "bg-indigo-400 animate-pulse"
                : "bg-indigo-500"
            }`}
            style={{ width: `${percent}%` }}
          />
        </div>
        <span className="font-mono text-xs text-slate-300 w-12 text-right shrink-0">
          {percent.toFixed(0)}%
        </span>
      </div>

      {/* Info & Stats Row */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-slate-300">{sizeString}</span>
          {job.totalShards > 1 && (
            <span className="text-slate-500 font-mono text-[10px]">
              (file {job.currentShard} of {job.totalShards})
            </span>
          )}
          {job.currentFile && job.totalShards > 1 && (
            <span
              className="text-slate-500 truncate max-w-[180px] font-mono text-[10px]"
              title={job.currentFile}
            >
              {job.currentFile.split("/").pop()}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {job.speedBps > 0 && isDownloading && (
            <span className="flex items-center gap-1 font-mono text-slate-300">
              <svg className="w-3 h-3 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              {fmtTransferRate(job.speedBps)}
            </span>
          )}

          {job.etaSeconds != null && job.etaSeconds > 0 && isDownloading && (
            <span className="flex items-center gap-1 font-mono text-slate-300">
              <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ETA: {fmtDuration(job.etaSeconds)}
            </span>
          )}

          {isCompleted && (
            <span className="text-emerald-400 font-medium flex items-center gap-1">
              ✓ Ready in Library
            </span>
          )}
        </div>
      </div>

      {/* Error display if failed (A5 disk space preflight & A9 library registration) */}
      {isFailed && job.error && (
        <div className="space-y-1 mt-1">
          {job.error.includes("Insufficient disk space") ? (
            <div className="text-xs text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded p-2.5 space-y-1">
              <div className="font-semibold flex items-center gap-1.5 text-amber-300">
                <svg className="w-4 h-4 text-amber-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Insufficient Disk Space Preflight
              </div>
              <div className="text-slate-300 pl-5.5 text-[11px] leading-relaxed">{job.error}</div>
            </div>
          ) : job.error.toLowerCase().includes("library registration failed") ? (
            <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded p-2.5 flex items-center justify-between gap-3">
              <div className="space-y-0.5 min-w-0">
                <span className="font-semibold block text-rose-200">Library Registration Error</span>
                <span className="text-slate-300 block text-[11px] truncate">{job.error}</span>
              </div>
              <button
                onClick={onRepair}
                className="px-2.5 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/30 rounded text-xs shrink-0 font-medium transition-colors"
                title="Rescan downloads and refresh local library"
              >
                Rescan Library
              </button>
            </div>
          ) : (
            <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded p-2">
              {job.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Compact inline download indicator for use inside model cards or table rows.
 */
export function InlineDownloadProgress({
  modelIdOrRepo,
  onCancel,
}: {
  modelIdOrRepo: string;
  onCancel?: () => void;
}) {
  const { getDownload } = useDownloads();
  const job = getDownload(modelIdOrRepo);

  if (!job) return null;

  const percent = job.status === "Completed"
    ? 100
    : job.totalBytes > 0
    ? Math.min(100, Math.max(0, (job.downloadedBytes / job.totalBytes) * 100))
    : Math.min(100, Math.max(0, job.percent || 0));

  return (
    <div className="flex items-center gap-2 w-full min-w-[140px] max-w-[240px]">
      <div className="flex-1 h-1.5 rounded-full bg-surface-3 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            job.status === "Completed"
              ? "bg-emerald-500"
              : job.status === "Failed"
              ? "bg-rose-500"
              : job.status === "Paused"
              ? "bg-amber-500"
              : "bg-indigo-500"
          }`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="font-mono text-xs text-slate-300 shrink-0">
        {percent.toFixed(0)}%
      </span>
      {onCancel && job.canCancel && (
        <button
          onClick={onCancel}
          className="text-xs text-rose-400 hover:text-rose-300 shrink-0"
          title="Cancel download"
        >
          ✕
        </button>
      )}
    </div>
  );
}
