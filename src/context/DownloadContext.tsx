import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { api, events } from "../api";
import type { DownloadJob, PullStatus, DownloadProgressPayload } from "../types";

export type DownloadStatusType =
  | "Queued"
  | "Resolving"
  | "Downloading"
  | "Paused"
  | "Completed"
  | "Failed"
  | "Cancelled";

export interface UnifiedDownloadItem {
  id: string;
  modelId: string;
  repoId: string;
  backend: string;
  quant?: string;
  files: string[];
  currentFile?: string | null;
  currentShard: number;
  totalShards: number;
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  speedBps: number;
  etaSeconds: number | null;
  status: DownloadStatusType;
  error?: string | null;
  isGguf: boolean;
  canPause: boolean;
  canResume: boolean;
  canCancel: boolean;
  rawJob?: DownloadJob;
  updatedAt: number;
  finishedAt?: number;
}

interface DownloadContextValue {
  downloads: UnifiedDownloadItem[];
  activeDownloads: UnifiedDownloadItem[];
  activeCount: number;
  overallPercent: number;
  cancelDownload: (idOrModel: string, quant?: string | null) => Promise<void>;
  pauseDownload: (id: string) => Promise<void>;
  resumeDownload: (id: string) => Promise<void>;
  clearCompleted: () => Promise<void>;
  dismissDownload: (id: string) => void;
  getDownload: (idOrModelOrRepo: string, quant?: string | null) => UnifiedDownloadItem | undefined;
  refreshDownloads: () => Promise<void>;
}

const DownloadContext = createContext<DownloadContextValue | null>(null);

function normalizeStatus(statusRaw?: string | null): DownloadStatusType {
  if (!statusRaw) return "Downloading";
  const s = statusRaw.toLowerCase();
  if (s === "queued") return "Queued";
  if (s === "resolving") return "Resolving";
  if (s === "downloading") return "Downloading";
  if (s === "paused") return "Paused";
  if (s === "completed" || s === "complete") return "Completed";
  if (s === "failed") return "Failed";
  if (s === "cancelled" || s === "canceled") return "Cancelled";
  return "Downloading";
}

export function DownloadProvider({ children }: { children: React.ReactNode }) {
  const [itemsMap, setItemsMap] = useState<Record<string, UnifiedDownloadItem>>({});
  const itemsMapRef = useRef(itemsMap);
  itemsMapRef.current = itemsMap;

  // Poll download queue status
  const syncQueue = useCallback(async () => {
    try {
      const jobs = await api.downloadQueueStatus().catch(() => [] as DownloadJob[]);
      const pullState = await api.pullStatus().catch(() => ({ pulling: [] as string[] }));

      setItemsMap((prev) => {
        const next = { ...prev };
        const now = Date.now();

        // Sync GGUF jobs from backend
        for (const job of jobs) {
          const status = normalizeStatus(job.status || job.state);
          const totalBytes = job.total_bytes || 0;
          const downloadedBytes = job.downloaded_bytes || 0;
          let percent = totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;
          if (status === "Completed") percent = 100;

          const existing = next[job.id];
          const finishedAt =
            status === "Completed" || status === "Cancelled" || status === "Failed"
              ? existing?.finishedAt || now
              : undefined;

          next[job.id] = {
            id: job.id,
            modelId: job.model_id || job.repo_id,
            repoId: job.repo_id || job.model_id,
            backend: job.backend || "llamacpp",
            quant: job.quant,
            files: job.files || [],
            currentFile: job.current_file || (job.files && job.files[0]) || null,
            currentShard: job.current_shard || (job.current_file_idx != null ? job.current_file_idx + 1 : 1),
            totalShards: job.total_shards || (job.files ? job.files.length : 1),
            downloadedBytes,
            totalBytes,
            percent: Math.min(100, Math.max(0, percent)),
            speedBps: job.speed_bps || 0,
            etaSeconds: job.eta_seconds ?? null,
            status,
            error: job.error || null,
            isGguf: true,
            canPause: status === "Downloading",
            canResume: status === "Paused",
            canCancel: status === "Downloading" || status === "Resolving" || status === "Queued" || status === "Paused",
            rawJob: job,
            updatedAt: now,
            finishedAt,
          };
        }

        // Sync legacy HF pull jobs
        for (const mid of pullState.pulling || []) {
          // If already tracked as a GGUF job, don't duplicate
          const existingJob = Object.values(next).find(
            (j) => j.repoId.toLowerCase() === mid.toLowerCase() || j.modelId.toLowerCase() === mid.toLowerCase()
          );
          if (!existingJob) {
            const pullId = `pull-${mid}`;
            if (!next[pullId]) {
              next[pullId] = {
                id: pullId,
                modelId: mid,
                repoId: mid,
                backend: "vllm",
                files: [],
                currentShard: 1,
                totalShards: 1,
                downloadedBytes: 0,
                totalBytes: 0,
                percent: 0,
                speedBps: 0,
                etaSeconds: null,
                status: "Downloading",
                isGguf: false,
                canPause: false,
                canResume: false,
                canCancel: true,
                updatedAt: now,
              };
            }
          }
        }

        // Auto-clean very old completed items (after 30 seconds)
        for (const [key, item] of Object.entries(next)) {
          if (
            (item.status === "Completed" || item.status === "Cancelled") &&
            item.finishedAt &&
            now - item.finishedAt > 30000
          ) {
            delete next[key];
          }
        }

        return next;
      });
    } catch {
      // ignore transient sync errors
    }
  }, []);

  // Subscribe to Tauri events
  useEffect(() => {
    syncQueue();

    // Listen to downloadProgress (rich GGUF event)
    const unsubDl = events.downloadProgress((p: DownloadProgressPayload) => {
      const now = Date.now();
      const status = normalizeStatus(p.status || p.state);
      const totalBytes = p.total_bytes || 0;
      const downloadedBytes = p.downloaded_bytes || 0;
      let percent = p.percent;
      if (totalBytes > 0 && (!percent || percent <= 0)) {
        percent = (downloadedBytes / totalBytes) * 100;
      }
      if (status === "Completed") percent = 100;

      setItemsMap((prev) => {
        const next = { ...prev };
        const key = p.job_id || `job-${p.model_id}`;
        const existing = next[key];

        const finishedAt =
          status === "Completed" || status === "Cancelled" || status === "Failed"
            ? existing?.finishedAt || now
            : undefined;

        next[key] = {
          id: p.job_id || key,
          modelId: p.model_id || existing?.modelId || "",
          repoId: p.repo_id || existing?.repoId || p.model_id || "",
          backend: existing?.backend || "llamacpp",
          quant: p.quant || existing?.quant,
          files: existing?.files || [],
          currentFile: p.current_file || existing?.currentFile,
          currentShard: p.current_shard || (p.file_index != null ? p.file_index + 1 : existing?.currentShard || 1),
          totalShards: p.total_shards || p.total_files || existing?.totalShards || 1,
          downloadedBytes,
          totalBytes: totalBytes > 0 ? totalBytes : existing?.totalBytes || 0,
          percent: Math.min(100, Math.max(0, percent || 0)),
          speedBps: p.speed_bps || 0,
          etaSeconds: p.eta_seconds ?? null,
          status,
          error: p.error || null,
          isGguf: true,
          canPause: status === "Downloading",
          canResume: status === "Paused",
          canCancel: status === "Downloading" || status === "Resolving" || status === "Queued" || status === "Paused",
          rawJob: existing?.rawJob,
          updatedAt: now,
          finishedAt,
        };

        return next;
      });
    });

    // Listen to pullProgress (vLLM / backward compatibility event)
    const unsubPull = events.pullProgress((p: PullStatus) => {
      const now = Date.now();
      const status = normalizeStatus(p.state);

      setItemsMap((prev) => {
        const next = { ...prev };

        // Match against existing item by model or repoId, respecting quant if present
        const matchEntry = Object.entries(next).find(
          ([, item]) => {
            const idMatches =
              item.repoId.toLowerCase() === p.model.toLowerCase() ||
              item.modelId.toLowerCase() === p.model.toLowerCase();
            if (!idMatches) return false;
            if (!p.quant || !item.quant) return true;
            return item.quant.toLowerCase() === p.quant.toLowerCase();
          }
        );

        const key = matchEntry ? matchEntry[0] : (p.quant ? `pull-${p.model}:${p.quant}` : `pull-${p.model}`);
        const existing = matchEntry ? matchEntry[1] : undefined;

        const totalBytes = p.bytes_total || existing?.totalBytes || 0;
        const downloadedBytes = p.bytes_downloaded || existing?.downloadedBytes || 0;
        let percent = p.percent != null ? p.percent : existing?.percent || 0;
        if (totalBytes > 0 && percent <= 0) {
          percent = (downloadedBytes / totalBytes) * 100;
        }
        if (status === "Completed") percent = 100;

        const finishedAt =
          status === "Completed" || status === "Cancelled" || status === "Failed"
            ? existing?.finishedAt || now
            : undefined;

        next[key] = {
          id: existing?.id || key,
          modelId: existing?.modelId || p.model,
          repoId: existing?.repoId || p.model,
          backend: existing?.backend || "vllm",
          quant: p.quant || existing?.quant,
          files: existing?.files || [],
          currentFile: p.file || existing?.currentFile,
          currentShard: existing?.currentShard || 1,
          totalShards: existing?.totalShards || 1,
          downloadedBytes,
          totalBytes,
          percent: Math.min(100, Math.max(0, percent)),
          speedBps: p.speed_bps != null ? p.speed_bps : existing?.speedBps || 0,
          etaSeconds: p.eta_seconds != null ? p.eta_seconds : existing?.etaSeconds ?? null,
          status,
          error: p.error || (status === "Failed" ? p.file : null),
          isGguf: existing?.isGguf ?? false,
          canPause: existing?.canPause ?? false,
          canResume: existing?.canResume ?? false,
          canCancel: status === "Downloading" || status === "Resolving" || status === "Queued",
          rawJob: existing?.rawJob,
          updatedAt: now,
          finishedAt,
        };

        return next;
      });
    });

    // Adaptive polling: poll frequently (2.5s) if downloads active, less often (8s) when idle
    const pollTimer = setInterval(() => {
      syncQueue();
    }, 2500);

    return () => {
      clearInterval(pollTimer);
      unsubDl.then((f) => f());
      unsubPull.then((f) => f());
    };
  }, [syncQueue]);

  // Derived lists
  const downloads = useMemo(() => {
    return Object.values(itemsMap).sort((a, b) => b.updatedAt - a.updatedAt);
  }, [itemsMap]);

  const activeDownloads = useMemo(() => {
    return downloads.filter(
      (d) =>
        d.status === "Downloading" ||
        d.status === "Resolving" ||
        d.status === "Queued" ||
        d.status === "Paused"
    );
  }, [downloads]);

  const activeCount = activeDownloads.length;

  const overallPercent = useMemo(() => {
    if (activeCount === 0) return 0;
    const sum = activeDownloads.reduce((acc, curr) => acc + curr.percent, 0);
    return Math.round(sum / activeCount);
  }, [activeDownloads, activeCount]);

  // Actions
  const cancelDownload = useCallback(
    async (idOrModel: string, quant?: string | null) => {
      const qQuant = quant?.toLowerCase();
      const item = Object.values(itemsMapRef.current).find((d) => {
        if (d.id === idOrModel) return true;
        const idMatches =
          d.modelId.toLowerCase() === idOrModel.toLowerCase() ||
          d.repoId.toLowerCase() === idOrModel.toLowerCase();
        if (!idMatches) return false;
        if (!qQuant || !d.quant) return true;
        return d.quant.toLowerCase() === qQuant;
      });

      if (item && item.isGguf && item.id.startsWith("job-")) {
        await api.downloadCancel(item.id).catch(() => {});
      } else {
        const repo = item ? item.repoId : idOrModel;
        await api.pullCancel(repo).catch(() => {});
      }

      setItemsMap((prev) => {
        const next = { ...prev };
        for (const [key, d] of Object.entries(next)) {
          const isTarget =
            d.id === idOrModel ||
            ((d.modelId.toLowerCase() === idOrModel.toLowerCase() ||
              d.repoId.toLowerCase() === idOrModel.toLowerCase()) &&
              (!qQuant || !d.quant || d.quant.toLowerCase() === qQuant));
          if (isTarget) {
            next[key] = {
              ...d,
              status: "Cancelled",
              speedBps: 0,
              finishedAt: Date.now(),
            };
          }
        }
        return next;
      });

      setTimeout(syncQueue, 400);
    },
    [syncQueue]
  );

  const pauseDownload = useCallback(
    async (id: string) => {
      const item = itemsMapRef.current[id];
      if (item) {
        await api.downloadPause(item.id).catch(() => {});
        setItemsMap((prev) => {
          if (!prev[id]) return prev;
          return {
            ...prev,
            [id]: {
              ...prev[id],
              status: "Paused",
              speedBps: 0,
              canPause: false,
              canResume: true,
            },
          };
        });
        setTimeout(syncQueue, 300);
      }
    },
    [syncQueue]
  );

  const resumeDownload = useCallback(
    async (id: string) => {
      const item = itemsMapRef.current[id];
      if (item && item.rawJob) {
        const j = item.rawJob;
        await api.downloadStart(j.model_id, j.repo_id, j.quant, j.files, j.backend).catch(() => {});
        setItemsMap((prev) => {
          if (!prev[id]) return prev;
          return {
            ...prev,
            [id]: {
              ...prev[id],
              status: "Resolving",
              canPause: true,
              canResume: false,
            },
          };
        });
        setTimeout(syncQueue, 300);
      }
    },
    [syncQueue]
  );

  const clearCompleted = useCallback(async () => {
    await api.downloadClearCompleted().catch(() => {});
    setItemsMap((prev) => {
      const next = { ...prev };
      for (const [key, item] of Object.entries(next)) {
        if (item.status === "Completed" || item.status === "Cancelled" || item.status === "Failed") {
          delete next[key];
        }
      }
      return next;
    });
  }, []);

  const dismissDownload = useCallback((id: string) => {
    setItemsMap((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  const getDownload = useCallback((idOrModelOrRepo: string, quant?: string | null): UnifiedDownloadItem | undefined => {
    const q = idOrModelOrRepo.toLowerCase();
    const qQuant = quant?.toLowerCase();
    return Object.values(itemsMapRef.current).find((d) => {
      const idMatches =
        d.id.toLowerCase() === q ||
        d.modelId.toLowerCase() === q ||
        d.repoId.toLowerCase() === q ||
        (d.modelId.includes("/") && d.modelId.split("/")[1].toLowerCase() === q);
      if (!idMatches) return false;
      if (!qQuant || !d.quant) return true;
      return d.quant.toLowerCase() === qQuant;
    });
  }, []);

  const value = useMemo(
    () => ({
      downloads,
      activeDownloads,
      activeCount,
      overallPercent,
      cancelDownload,
      pauseDownload,
      resumeDownload,
      clearCompleted,
      dismissDownload,
      getDownload,
      refreshDownloads: syncQueue,
    }),
    [
      downloads,
      activeDownloads,
      activeCount,
      overallPercent,
      cancelDownload,
      pauseDownload,
      resumeDownload,
      clearCompleted,
      dismissDownload,
      getDownload,
      syncQueue,
    ]
  );

  return <DownloadContext.Provider value={value}>{children}</DownloadContext.Provider>;
}

export function useDownloads() {
  const ctx = useContext(DownloadContext);
  if (!ctx) {
    throw new Error("useDownloads must be used within a DownloadProvider");
  }
  return ctx;
}
