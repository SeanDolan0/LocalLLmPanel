import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, events, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Spinner, inputCls } from "../ui";
import type { LibraryEntry, PullStatus } from "../types";

export default function Library() {
  const navigate = useNavigate();
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [diskUsageMb, setDiskUsageMb] = useState<number | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [taskFilter, setTaskFilter] = useState<"all" | "instruct" | "embed">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<LibraryEntry | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const refresh = useCallback(() => {
    Promise.all([
      api.libraryList().catch(() => [] as LibraryEntry[]),
      api.libraryDiskUsage().catch(() => 0),
    ])
      .then(([list, mb]) => {
        setEntries(list);
        setDiskUsageMb(mb || list.reduce((acc, e) => acc + (e.size_mb || 0), 0));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 10000);
    const unsub = events.pullProgress((p) => {
      setPulls((prev) => ({ ...prev, [p.model]: p }));
      if (p.state === "complete") refresh();
    });
    return () => {
      clearInterval(t);
      unsub.then((f) => f());
    };
  }, [refresh]);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    setErrorMsg(null);
    try {
      await api.libraryRemove(deleteTarget.model_id);
      setDeleteTarget(null);
      refresh();
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setIsDeleting(false);
    }
  };

  const filteredEntries = entries.filter((e) => {
    if (taskFilter === "instruct" && e.task && e.task !== "instruct" && e.task !== "chat") return false;
    if (taskFilter === "embed" && e.task !== "embed") return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      if (!e.model_id.toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const totalGb = diskUsageMb !== null ? (diskUsageMb / 1024).toFixed(1) : "0.0";
  const instructCount = entries.filter((e) => !e.task || e.task === "instruct" || e.task === "chat").length;
  const embedCount = entries.filter((e) => e.task === "embed").length;

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Model Library</h1>
          <p className="text-xs text-slate-400 mt-1">
            Locally downloaded HuggingFace models cached in WSL (<code>~/.cache/huggingface/hub</code>).
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="ghost" onClick={refresh}>↻ Refresh</Button>
        </div>
      </div>

      {/* Disk Usage Banner */}
      <Card className="bg-gradient-to-r from-surface-2 to-surface-3 border-indigo-500/20">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              💾
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-200">
                Local cache: <span className="text-indigo-300 font-bold">{totalGb} GB</span> used across{" "}
                <span className="text-slate-100 font-bold">{entries.length}</span> {entries.length === 1 ? "model" : "models"}
              </div>
              <div className="text-xs text-slate-400 mt-0.5">
                vLLM uses this cache directly when spinning up servers.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link to="/search">
              <Button variant="ghost" className="text-xs">
                + Browse More Models
              </Button>
            </Link>
          </div>
        </div>
      </Card>

      {/* Filter and Search Bar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setTaskFilter("all")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              taskFilter === "all"
                ? "bg-indigo-600 text-white"
                : "bg-surface-2 border border-edge text-slate-400 hover:text-slate-200"
            }`}
          >
            All ({entries.length})
          </button>
          <button
            onClick={() => setTaskFilter("instruct")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              taskFilter === "instruct"
                ? "bg-indigo-600 text-white"
                : "bg-surface-2 border border-edge text-slate-400 hover:text-slate-200"
            }`}
          >
            Instruct ({instructCount})
          </button>
          <button
            onClick={() => setTaskFilter("embed")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              taskFilter === "embed"
                ? "bg-indigo-600 text-white"
                : "bg-surface-2 border border-edge text-slate-400 hover:text-slate-200"
            }`}
          >
            Embed ({embedCount})
          </button>
        </div>

        <div className="w-full sm:w-64">
          <input
            type="text"
            className={inputCls}
            placeholder="Filter models by name…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* List */}
      {loading ? (
        <Card className="flex items-center justify-center p-8">
          <Spinner label="Scanning HF cache in WSL…" />
        </Card>
      ) : entries.length === 0 ? (
        <Card className="p-8 text-center space-y-3">
          <CardTitle>No models pulled yet</CardTitle>
          <div className="text-sm text-slate-400 max-w-md mx-auto">
            Your local cache is empty. Use <Link className="text-indigo-300 hover:underline font-medium" to="/search">Search</Link> to find compatible models and deploy or download them.
          </div>
        </Card>
      ) : filteredEntries.length === 0 ? (
        <Card className="p-6 text-center text-sm text-slate-400">
          No cached models matched your filter criteria.
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredEntries.map((e) => {
            const pull = pulls[e.model_id];
            const sizeGb = (e.size_mb / 1024).toFixed(1);
            return (
              <Card key={e.model_id} className="flex flex-col justify-between hover:border-edge/90 transition-colors">
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-semibold text-slate-200" title={e.model_id}>
                        {e.model_id}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-400">
                        <span>{sizeGb} GB ({fmtNum(e.size_mb)} MB)</span>
                        <span>·</span>
                        <span>{fmtNum(e.files)} files</span>
                      </div>
                    </div>
                    {pull ? (
                      <Badge color={pull.state === "complete" ? "emerald" : pull.state === "failed" ? "red" : "indigo"}>
                        {pull.state}
                      </Badge>
                    ) : e.in_use ? (
                      <Badge color="amber" title={`Running on server: ${e.in_use_server || "active"}`}>
                        in use
                      </Badge>
                    ) : (
                      <Badge color="emerald">cached</Badge>
                    )}
                  </div>

                  {/* Metadata Chips */}
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {e.params_b && (
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium text-slate-300 border border-slate-700">
                        {e.params_b}B
                      </span>
                    )}
                    {e.quant && (
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium text-slate-300 border border-slate-700">
                        {e.quant}
                      </span>
                    )}
                    {e.task && (
                      <span className="rounded bg-indigo-950/60 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300 border border-indigo-800/40">
                        {e.task}
                      </span>
                    )}
                    {e.in_use && e.in_use_server && (
                      <span className="rounded bg-amber-950/60 px-1.5 py-0.5 text-[10px] font-medium text-amber-300 border border-amber-800/40">
                        Server: {e.in_use_server}
                      </span>
                    )}
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-edge/60 flex items-center justify-between gap-2">
                  {e.in_use ? (
                    <Button
                      variant="danger"
                      disabled
                      title={`Model is currently in use by server "${e.in_use_server || "active"}". Stop the server before deleting.`}
                      className="text-xs opacity-40 cursor-not-allowed"
                    >
                      Delete
                    </Button>
                  ) : (
                    <Button
                      variant="danger"
                      onClick={() => setDeleteTarget(e)}
                      className="text-xs"
                    >
                      Delete
                    </Button>
                  )}

                  <Button
                    variant="ghost"
                    className="text-xs"
                    onClick={() => navigate("/servers", { state: { prefillModel: e.model_id } })}
                  >
                    Deploy →
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-xl border border-red-500/30 bg-surface-2 p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-red-400">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10 border border-red-500/20 text-xl font-bold">
                ⚠️
              </div>
              <div>
                <h3 className="text-base font-semibold text-slate-100">Delete Model from Disk?</h3>
                <p className="text-xs text-slate-400">Permanent deletion from local WSL cache</p>
              </div>
            </div>

            <div className="text-sm text-slate-300 space-y-2">
              <p>
                Are you sure you want to delete <strong className="text-slate-100">{deleteTarget.model_id}</strong>?
              </p>
              <div className="rounded-lg bg-surface-3 p-3 border border-edge text-xs space-y-1">
                <div className="flex justify-between text-slate-300">
                  <span>Freed space:</span>
                  <span className="font-semibold text-emerald-400">
                    ~{(deleteTarget.size_mb / 1024).toFixed(1)} GB ({fmtNum(deleteTarget.size_mb)} MB)
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Files removed:</span>
                  <span>{fmtNum(deleteTarget.files)}</span>
                </div>
              </div>
              <p className="text-xs text-slate-500">
                This frees disk space in your WSL virtual hard disk. If needed later, vLLM will automatically re-download it.
              </p>
            </div>

            {errorMsg && (
              <div className="rounded-md bg-red-950/50 border border-red-500/30 p-2.5 text-xs text-red-300">
                {errorMsg}
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <Button
                variant="ghost"
                onClick={() => {
                  setDeleteTarget(null);
                  setErrorMsg(null);
                }}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                onClick={handleDelete}
                disabled={isDeleting}
              >
                {isDeleting ? <Spinner label="Deleting from WSL…" /> : "Yes, Delete from Disk"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}