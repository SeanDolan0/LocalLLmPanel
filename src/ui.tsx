import { useEffect, useState, type ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-edge/80 bg-surface-2 p-5 shadow-sm transition-all ${className}`}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Collapsible section — progressive disclosure for advanced controls.
// ---------------------------------------------------------------------------
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      className={`h-4 w-4 shrink-0 text-slate-400 transition-transform duration-200 ease-out ${open ? "rotate-90 text-accent" : ""}`}
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
    <div className={`rounded-2xl border border-edge bg-surface-2 overflow-hidden shadow-sm transition-all ${className}`}>
      <div className={`flex items-center gap-3 ${compact ? "px-3.5 py-2.5" : "px-4 py-3.5"}`}>
        <button
          type="button"
          onClick={toggle}
          className="flex min-w-0 flex-1 items-center gap-3 text-left group"
          aria-expanded={open}
        >
          <Chevron open={open} />
          <span className="min-w-0">
            <span className={`block font-semibold text-slate-100 group-hover:text-accent transition-colors ${compact ? "text-xs" : "text-sm"}`}>{title}</span>
            {subtitle && (
              <span className={`block truncate text-slate-400 ${compact ? "text-[10px]" : "text-xs"}`}>{subtitle}</span>
            )}
          </span>
        </button>
        {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
      </div>
      {open && <div className={`${compact ? "px-3.5 pb-3.5" : "px-4 pb-4"} border-t border-edge-subtle pt-3.5`}>{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overflow (···) menu — keeps low-frequency actions reachable without clutter.
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
  label = "···",
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
        className={`flex h-8 w-8 items-center justify-center rounded-lg border text-sm leading-none transition-all duration-150 active:scale-95 ${
          open
            ? "border-accent/50 bg-accent/15 text-accent"
            : "border-edge bg-surface-3/80 text-slate-300 hover:text-white hover:bg-surface-3"
        }`}
      >
        {label}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-50 mt-1.5 min-w-52 rounded-xl border border-edge bg-surface-3/95 p-1 backdrop-blur-md shadow-2xl shadow-black/60">
            {visible.map((item, i) => (
              <button
                key={`${item.label}-${i}`}
                type="button"
                onClick={() => {
                  setOpen(false);
                  item.onClick?.();
                }}
                className={`flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-xs font-medium transition-colors ${
                  item.danger
                    ? "text-red-400 hover:bg-bad/15"
                    : "text-slate-200 hover:bg-surface-2 hover:text-white"
                }`}
              >
                <span>{item.label}</span>
                {item.hint && <span className="text-[10px] text-slate-400">{item.hint}</span>}
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
    <div className="mb-3.5 flex items-center justify-between">
      <h3 className="text-sm font-semibold text-slate-100">{children}</h3>
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
    slate: "bg-slate-500/10 text-slate-300 border-slate-500/20",
    emerald: "bg-good/10 text-emerald-400 border-good/20",
    amber: "bg-warn/10 text-amber-400 border-warn/20",
    red: "bg-bad/10 text-red-400 border-bad/20",
    indigo: "bg-accent/10 text-sky-400 border-accent/20",
    cyan: "bg-cyan-500/10 text-cyan-300 border-cyan-500/20",
  };
  return (
    <span title={title} className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium tracking-wide ${map[color]}`}>
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
    "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all duration-150 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100";
  const variants = {
    primary:
      "bg-accent hover:bg-accent-hover text-white shadow-sm shadow-accent/25",
    ghost: "border border-edge bg-surface-3 hover:bg-surface-3/80 text-slate-200 hover:text-white",
    danger: "border border-bad/30 bg-bad/10 hover:bg-bad/20 text-red-300",
    subtle: "text-slate-400 hover:text-slate-100 hover:bg-surface-3/60",
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
      <svg className="h-4 w-4 animate-spin text-accent" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
      </svg>
      {label}
    </div>
  );
}

export function Gauge({ pct, label, sub }: { pct: number; label: string; sub?: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const color = clamped >= 90 ? "#ff453a" : clamped >= 70 ? "#ff9f0a" : "#30d158";
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-28 w-28">
        <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
          <circle cx="50" cy="50" r="42" fill="none" stroke="#252a36" strokeWidth="10" />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${clamped * 2.638} 263.8`}
            style={{ transition: "stroke-dasharray 0.5s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-slate-100 tracking-tight">{label}</span>
        </div>
      </div>
      {sub && <div className="text-[11px] text-slate-400">{sub}</div>}
    </div>
  );
}

/// Small "?" bubble that shows a plain-language explainer on hover,
/// keyboard focus, or tap. Use next to a setting label; the text should say
/// what the setting is and how it affects performance / RAM.
export function HelpBubble({ text, label }: { text: string; label?: string }) {
  return (
    <span
      className="group relative inline-flex shrink-0 align-middle"
      role="tooltip"
      aria-label={label ?? text}
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <span
        tabIndex={0}
        aria-hidden="true"
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-600 bg-surface text-[10px] font-semibold leading-none text-slate-400 transition-colors group-hover:border-accent group-hover:text-accent group-focus-within:border-accent group-focus-within:text-accent cursor-help"
      >
        ?
      </span>
      <span className="pointer-events-none absolute left-1/2 top-full z-50 mt-1.5 w-64 -translate-x-1/2 rounded-xl border border-edge bg-surface-3/95 p-3 text-left text-xs font-normal leading-relaxed text-slate-200 opacity-0 shadow-2xl shadow-black/60 backdrop-blur-md transition-opacity duration-150 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100">
        {text}
      </span>
    </span>
  );
}

export function Field({ label, children, hint, help }: { label: string; children: ReactNode; hint?: string; help?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-slate-300">
        {label}
        {help && <HelpBubble text={help} />}
      </span>
      {children}
      {hint && <span className="mt-1.5 block text-[11px] text-slate-400 leading-normal">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-slate-100 outline-none transition-all duration-150 focus:border-accent focus:ring-2 focus:ring-accent/20 placeholder:text-slate-500";

// ---------------------------------------------------------------------------
// Apple / Windows Fluent Primitives
// ---------------------------------------------------------------------------

export type SquircleColor = "blue" | "green" | "orange" | "purple" | "slate" | "red";

export function SquircleIcon({
  icon,
  color = "blue",
  size = "md",
}: {
  icon: ReactNode;
  color?: SquircleColor;
  size?: "sm" | "md";
}) {
  const colors: Record<SquircleColor, string> = {
    blue: "bg-accent text-white shadow-sm shadow-accent/20",
    green: "bg-good text-white shadow-sm shadow-good/20",
    orange: "bg-warn text-white shadow-sm shadow-warn/20",
    purple: "bg-purple-500 text-white shadow-sm shadow-purple-500/20",
    slate: "bg-surface-3 text-slate-300 border border-edge",
    red: "bg-bad text-white shadow-sm shadow-bad/20",
  };

  const sizes = {
    sm: "h-6 w-6 rounded-md text-xs",
    md: "h-7 w-7 rounded-lg text-sm",
  };

  return (
    <div className={`flex shrink-0 items-center justify-center ${sizes[size]} ${colors[color]}`}>
      {icon}
    </div>
  );
}

/**
 * Apple-style Inset Group container: groups rows into a clean rounded card with hairline dividers.
 */
export function InsetGroup({
  children,
  className = "",
  header,
}: {
  children: ReactNode;
  className?: string;
  header?: ReactNode;
}) {
  return (
    <div className={`mb-4 ${className}`}>
      {header && (
        <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
          {header}
        </div>
      )}
      <div className="rounded-2xl border border-edge bg-surface-2 overflow-hidden shadow-sm divide-y divide-edge-subtle">
        {children}
      </div>
    </div>
  );
}

/**
 * Apple Inset List Row: an icon, title, description, and trailing control.
 */
export function SettingsRow({
  icon,
  title,
  subtitle,
  right,
  onClick,
  className = "",
}: {
  icon?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  onClick?: () => void;
  className?: string;
}) {
  const isClickable = Boolean(onClick);

  return (
    <div
      onClick={onClick}
      className={`flex items-center justify-between gap-3.5 px-4 py-3 transition-colors ${
        isClickable ? "cursor-pointer hover:bg-surface-3/50 active:bg-surface-3/80" : ""
      } ${className}`}
    >
      <div className="flex min-w-0 items-center gap-3">
        {icon}
        <div className="min-w-0">
          <div className="text-sm font-medium text-slate-100 truncate">{title}</div>
          {subtitle && <div className="text-xs text-slate-400 truncate mt-0.5">{subtitle}</div>}
        </div>
      </div>
      {right && <div className="flex shrink-0 items-center gap-2.5">{right}</div>}
    </div>
  );
}

/**
 * Apple-style UISwitch toggle component with smooth spring thumb.
 */
export function Toggle({
  checked,
  onChange,
  disabled = false,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:cursor-not-allowed disabled:opacity-40 ${
        checked ? "bg-accent border-accent" : "bg-surface-3 border-edge"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-md transition duration-200 ease-in-out ${
          checked ? "translate-x-[21px]" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}

/**
 * Apple-style Segmented Control (Pill switch)
 */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className = "",
}: {
  options: { value: T; label: string; count?: number }[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  return (
    <div className={`inline-flex rounded-xl bg-surface-3/80 p-1 border border-edge ${className}`}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1 text-xs font-medium transition-all duration-150 ${
              active
                ? "bg-surface-2 text-white shadow-sm border border-edge/60"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>{opt.label}</span>
            {opt.count !== undefined && (
              <span
                className={`rounded-full px-1.5 py-0.2 text-[10px] font-semibold ${
                  active ? "bg-accent text-white" : "bg-surface text-slate-400"
                }`}
              >
                {opt.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}