import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-edge bg-surface-2 p-4 ${className}`}>{children}</div>
  );
}

export function CardTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h3 className="text-sm font-semibold text-slate-200">{children}</h3>
      {right}
    </div>
  );
}

export function Badge({
  color = "slate",
  children,
  title,
}: {
  color?: "slate" | "emerald" | "amber" | "red" | "indigo" | "cyan";
  children: ReactNode;
  title?: string;
}) {
  const map = {
    slate: "bg-slate-500/10 text-slate-300 border-slate-500/30",
    emerald: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    red: "bg-red-500/10 text-red-300 border-red-500/30",
    indigo: "bg-indigo-500/10 text-indigo-300 border-indigo-500/30",
    cyan: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
  };
  return (
    <span title={title} className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${map[color]}`}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  className = "",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger" | "subtle";
  disabled?: boolean;
  className?: string;
  title?: string;
}) {
  const base =
    "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary:
      "bg-indigo-500 hover:bg-indigo-400 text-white shadow-sm shadow-indigo-500/25",
    ghost: "border border-edge bg-surface-3 hover:bg-surface-3/70 text-slate-300",
    danger: "border border-red-500/40 bg-red-500/10 hover:bg-red-500/20 text-red-300",
    subtle: "text-slate-400 hover:text-slate-200 hover:bg-surface-3",
  };
  return (
    <button className={`${base} ${variants[variant]} ${className}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <svg className="h-4 w-4 animate-spin text-indigo-400" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
      </svg>
      {label}
    </div>
  );
}

export function Gauge({ pct, label, sub }: { pct: number; label: string; sub?: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const color = clamped >= 90 ? "#f87171" : clamped >= 70 ? "#fbbf24" : "#34d399";
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-28 w-28">
        <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
          <circle cx="50" cy="50" r="42" fill="none" stroke="#1e2536" strokeWidth="10" />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${clamped * 2.638} 263.8`}
            style={{ transition: "stroke-dasharray 0.5s" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold">{label}</span>
        </div>
      </div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-400">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-slate-600">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full rounded-md border border-edge bg-surface px-2.5 py-1.5 text-sm text-slate-200 outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 placeholder:text-slate-600";