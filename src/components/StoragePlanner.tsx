import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge } from "../ui";
import type { DiskPreflightInfo } from "../types";

interface StoragePlannerProps {
  modelWeightGb: number;
  kvCacheGb?: number;
  stagingBufferGb?: number;
  customPath?: string;
  compact?: boolean;
  className?: string;
}

export function StoragePlanner({
  modelWeightGb,
  kvCacheGb = 0,
  stagingBufferGb,
  customPath,
  compact = false,
  className = "",
}: StoragePlannerProps) {
  const [preflight, setPreflight] = useState<DiskPreflightInfo | null>(null);
  const [loading, setLoading] = useState(true);

  // Staging space required during shard download & safe file promotion (weight + 512MB margin)
  const effectiveStagingGb = stagingBufferGb ?? (modelWeightGb > 0 ? modelWeightGb + 0.5 : 0);
  const totalRequiredGb = modelWeightGb + effectiveStagingGb + kvCacheGb;
  const totalRequiredBytes = Math.round(totalRequiredGb * 1024 * 1024 * 1024);

  useEffect(() => {
    let active = true;
    setLoading(true);
    api
      .diskPreflight(customPath, totalRequiredBytes)
      .then((info: DiskPreflightInfo) => {
        if (active) setPreflight(info);
      })
      .catch(() => {
        if (active) setPreflight(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [customPath, totalRequiredBytes]);

  const availableGb = preflight?.available_gb ?? 0;
  const headroomGb = availableGb - totalRequiredGb;
  const status =
    headroomGb < 0
      ? "insufficient"
      : headroomGb < 5.0
      ? "tight"
      : "sufficient";

  if (compact) {
    return (
      <div className={`flex items-center gap-2 text-xs ${className}`}>
        <span className="text-slate-400">Storage needed:</span>
        <span className="font-mono text-slate-200">{fmtNum(totalRequiredGb, 1)} GB</span>
        <span className="text-slate-500">/</span>
        <span
          className={`font-mono ${
            status === "insufficient"
              ? "text-red-400 font-semibold"
              : status === "tight"
              ? "text-amber-400"
              : "text-emerald-400"
          }`}
        >
          {loading ? "checking…" : `${fmtNum(availableGb, 1)} GB free`}
        </span>
        <Badge
          color={
            status === "insufficient" ? "red" : status === "tight" ? "amber" : "emerald"
          }
        >
          {status === "insufficient"
            ? `Needs +${fmtNum(Math.abs(headroomGb), 1)} GB`
            : status === "tight"
            ? "Tight"
            : "Sufficient Space"}
        </Badge>
      </div>
    );
  }

  // Calculate percentage of free space used
  const reqPct = availableGb > 0 ? Math.min(100, Math.round((totalRequiredGb / availableGb) * 100)) : 100;

  return (
    <div
      className={`rounded-xl border border-edge bg-surface-2 p-4 text-xs space-y-3 shadow-sm ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base select-none">💾</span>
          <span className="font-semibold text-slate-200">Storage Planning & Disk Preflight</span>
        </div>
        <Badge
          color={
            status === "insufficient" ? "red" : status === "tight" ? "amber" : "emerald"
          }
        >
          {status === "insufficient"
            ? "Insufficient Volume Space"
            : status === "tight"
            ? "Storage Tight"
            : "Adequate Headroom"}
        </Badge>
      </div>

      {/* Breakdown Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px]">
        <div className="rounded-lg bg-surface-3/70 p-2 border border-white/5">
          <div className="text-slate-400">Model Weights</div>
          <div className="text-sm font-semibold text-indigo-300 font-mono mt-0.5">
            {fmtNum(modelWeightGb, 1)} GB
          </div>
        </div>
        <div className="rounded-lg bg-surface-3/70 p-2 border border-white/5">
          <div className="text-slate-400" title="Temporary shard download and verified staging area">
            Temp Staging
          </div>
          <div className="text-sm font-semibold text-cyan-300 font-mono mt-0.5">
            {fmtNum(effectiveStagingGb, 1)} GB
          </div>
        </div>
        <div className="rounded-lg bg-surface-3/70 p-2 border border-white/5">
          <div className="text-slate-400" title="Disk swap buffer for context overflow">
            KV Cache / Swap
          </div>
          <div className="text-sm font-semibold text-emerald-300 font-mono mt-0.5">
            {fmtNum(kvCacheGb, 1)} GB
          </div>
        </div>
        <div className="rounded-lg bg-surface-3/70 p-2 border border-white/5">
          <div className="text-slate-400">Total Required</div>
          <div className="text-sm font-semibold text-white font-mono mt-0.5">
            {fmtNum(totalRequiredGb, 1)} GB
          </div>
        </div>
      </div>

      {/* Capacity Bar */}
      <div className="space-y-1 pt-1">
        <div className="flex justify-between text-[11px] text-slate-400">
          <span>Required Allocation vs Volume Free Space</span>
          <span className="font-mono text-slate-300">
            {loading ? "Checking volume…" : `${fmtNum(availableGb, 1)} GB free (${preflight?.target_path || "main drive"})`}
          </span>
        </div>
        <div className="h-2 w-full rounded-full bg-surface-3 overflow-hidden flex">
          <div
            className={`h-full transition-all duration-300 ${
              status === "insufficient"
                ? "bg-red-500"
                : status === "tight"
                ? "bg-amber-400"
                : "bg-indigo-500"
            }`}
            style={{ width: `${Math.min(100, reqPct)}%` }}
            title={`Required: ${fmtNum(totalRequiredGb, 1)} GB (${reqPct}%)`}
          />
        </div>
      </div>

      {/* Headroom / Recommendation text */}
      <div className="text-[11px] leading-relaxed pt-1">
        {loading ? (
          <span className="text-slate-400">Evaluating storage volume…</span>
        ) : status === "insufficient" ? (
          <div className="text-red-300 flex items-start gap-1.5 font-medium">
            <span>⚠️</span>
            <span>
              Insufficient disk space! This download requires <strong>{fmtNum(totalRequiredGb, 1)} GB</strong> total
              staging and weight storage, but only <strong>{fmtNum(availableGb, 1)} GB</strong> is available.
              Free at least <strong>{fmtNum(Math.abs(headroomGb), 1)} GB</strong> to prevent download failure.
            </span>
          </div>
        ) : status === "tight" ? (
          <div className="text-amber-300 flex items-start gap-1.5 font-medium">
            <span>⚠️</span>
            <span>
              Storage is tight. After staging and model storage, only <strong>{fmtNum(headroomGb, 1)} GB</strong> free
              headroom will remain on this volume.
            </span>
          </div>
        ) : (
          <div className="text-emerald-300 flex items-start gap-1.5">
            <span>✓</span>
            <span>
              Sufficient space available: <strong>{fmtNum(headroomGb, 1)} GB</strong> free headroom will remain after
              staging and model weight storage.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
