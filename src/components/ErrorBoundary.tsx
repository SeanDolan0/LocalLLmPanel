import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Optional compact mode for inline panels (e.g. a modal); defaults to full-page. */
  variant?: "page" | "inline";
  /** Called when the user dismisses the fallback (inline mode typically closes the overlay). */
  onDismiss?: () => void;
  dismissLabel?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Keeps a single bad render from unmounting the entire app (which presents as a
 * blank screen until restart). Logs the original error so it can be diagnosed.
 */
export default class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary] render crash:", error, info?.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const dismiss = this.props.onDismiss ?? this.reset;
    const dismissLabel = this.props.dismissLabel ?? (this.props.variant === "inline" ? "Close" : "Reset view");

    if (this.props.variant === "inline") {
      return (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-xs text-rose-200 space-y-2">
          <div className="font-semibold text-rose-300">This panel failed to render.</div>
          <pre className="whitespace-pre-wrap break-words text-[11px] text-rose-200/80 max-h-40 overflow-auto">
            {String(error?.message || error)}
          </pre>
          <button
            onClick={dismiss}
            className="rounded-md border border-rose-500/40 bg-rose-500/15 px-3 py-1.5 text-[11px] font-medium text-rose-200 hover:bg-rose-500/25 transition-colors"
          >
            {dismissLabel}
          </button>
        </div>
      );
    }

    return (
      <div className="flex h-full min-h-[40vh] w-full flex-col items-center justify-center gap-4 p-8 text-center">
        <div className="text-lg font-semibold text-rose-300">Something went wrong</div>
        <pre className="max-w-2xl whitespace-pre-wrap break-words text-[11px] text-slate-400 max-h-48 overflow-auto">
          {String(error?.message || error)}
        </pre>
        <div className="flex items-center gap-2">
          <button
            onClick={dismiss}
            className="rounded-md border border-edge bg-surface-2 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-surface-3 transition-colors"
          >
            {dismissLabel}
          </button>
          <button
            onClick={() => window.location.reload()}
            className="rounded-md border border-indigo-500/40 bg-indigo-500/15 px-4 py-2 text-xs font-medium text-indigo-200 hover:bg-indigo-500/25 transition-colors"
          >
            Reload app
          </button>
        </div>
      </div>
    );
  }
}
