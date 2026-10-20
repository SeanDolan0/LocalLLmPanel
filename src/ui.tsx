import { useEffect, useState, type ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-edge bg-surface-2 p-4 ${className}`}>{children}</div>
  );
}

// ---------------------------------------------------------------------------
// Collapsible section — progressive disclosure for advanced controls.
// ---------------------------------------------------------------------------
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      className={`h-3.5 w-3.5 shrink-0 text-slate-500 transition-transform duration-150 ${open ? "rotate-90" : ""}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 5l7 7-7 7" />
    </svg>
  );
}

export function Collapsible({
  title,
  subtitle,
  right,
  defaultOpen = false,
  storageKey,
  open: openProp,
  onOpenChange,
  children,
  className = "",
  compact = false,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  /** Rendered on the header row (e.g. a badge or action). Kept clickable. */
  right?: ReactNode;
  defaultOpen?: boolean;
  /** When set, open/closed state persists across sessions in localStorage. */
  storageKey?: string;
  /** Controlled mode: pass the open state and a change handler. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children?: ReactNode;
  className?: string;
  compact?: boolean;
}) {
  const [internalOpen, setInternalOpen] = useState(() => {
    if (storageKey) {
      const v = localStorage.getItem(storageKey);
      if (v !== null) return v === "1";
    }
    return defaultOpen;
  });
  const open = openProp ?? internalOpen;
  const toggle = () => {
    const next = !open;
    setInternalOpen(next);
    onOpenChange?.(next);
  };

  useEffect(() => {
    if (storageKey) localStorage.setItem(storageKey, open ? "1" : "0");
  }, [open, storageKey]);

  return (
    <div className={`rounded-xl border border-edge bg-surface-2 ${className}`}>
      <div className={`flex items-center gap-2.5 ${compact ? "px-3 py-2" : "px-4 py-3"}`}>
        <button
          type="button"
          onClick={toggle}
          className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
          aria-expanded={open}
        >
          <Chevron open={open} />
          <span className="min-w-0">
            <span className={`block font-semibold text-slate-200 ${compact ? "text-xs" : "text-sm"}`}>{title}</span>
            {subtitle && (
              <span className={`block truncate text-slate-500 ${compact ? "text-[10px]" : "text-xs"}`}>{subtitle}</span>
            )}
          </span>
        </button>
        {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
      </div>
      {open && <div className={`${compact ? "px-3 pb-3" : "px-4 pb-4"} border-t border-edge/50 pt-3`}>{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overflow (⋯) menu — keeps low-frequency actions reachable without clutter.
// ---------------------------------------------------------------------------
export type MenuItem = {
  label: string;
  onClick?: () => void;
  danger?: boolean;
  disabled?: boolean;
  hint?: string;
};

export function Menu({
  items,
  label = "⋯",
  title = "More actions",
  className = "",
}: {
  items: MenuItem[];
  label?: string;
  title?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const visible = items.filter((i) => !i.disabled);
  if (visible.length === 0) return null;

  return (
    <div className={`relative ${className}`}>
      <button
        type="button"
        title={title}
        aria-label={title}
        onClick={() => setOpen((o) => !o)}
        className={`flex h-7 w-7 items-center justify-center rounded-md border text-sm leading-none transition-colors ${
          open
            ? "border-indigo-500/50 bg-indigo-500/10 text-indigo-300"
            : "border-edge bg-surface-3 text-slate-400 hover:text-slate-200"
        }`}
      >
        {label}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-50 mt-1 min-w-52 rounded-lg border border-edge bg-surface-3 py-1 shadow-xl shadow-black/40">
            {visible.map((item, i) => (
              <button
                key={`${item.label}-${i}`}
                type="button"
                onClick={() => {
                  setOpen(false);
                  item.onClick?.();
                }}
                className={`flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-xs transition-colors ${
                  item.danger
                    ? "text-red-300 hover:bg-red-500/10"
                    : "text-slate-300 hover:bg-surface-2 hover:text-slate-100"
                }`}
              >
                <span>{item.label}</span>
                {item.hint && <span className="text-[10px] text-slate-500">{item.hint}</span>}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
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