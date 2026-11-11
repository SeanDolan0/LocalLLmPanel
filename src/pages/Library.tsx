import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { StoragePlanner } from "../components/StoragePlanner";
import { ErrorAlert } from "../components/ErrorAlert";
import { api, events, fmtNum } from "../api";
import { ActiveDownloadsPanel } from "../components/ActiveDownloadsPanel";
import { useDownloads } from "../context/DownloadContext";
import { Badge, Button, Card, CardTitle, Spinner, inputCls, SegmentedControl, SquircleIcon } from "../ui";
import type { LibraryEntry, PullStatus } from "../types";

function RefreshIcon() {
  return (
    <svg className="h-3.5 w-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function FolderImportIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
      <path d="M12 10v6" />
      <path d="m9 13 3 3 3-3" />
    </svg>
  );
}

function HardDriveIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" x2="2" y1="12" y2="12" />
      <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
      <line x1="6" x2="6.01" y1="16" y2="16" />
      <line x1="10" x2="10.01" y1="16" y2="16" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  );
}

export default function Library() {
  const navigate = useNavigate();
  const { getDownload, cancelDownload } = useDownloads();
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [diskUsageMb, setDiskUsageMb] = useState<number | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});
  const [taskFilter, setTaskFilter] = useState<"all" | "instruct" | "embed">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<LibraryEntry | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importBusy, setImportBusy] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

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

  useEffect(() => {
    api.pullStatus().then((res) => {
      if (res?.pulling?.length) {
        const active: Record<string, PullStatus> = {};
        res.pulling.forEach((id) => {
          active[id] = { model: id, state: "downloading" };
        });
        setPulls((prev) => ({ ...prev, ...active }));
      }
    }).catch(() => {});
  }, []);

  const handleCancelPull = async (modelId: string, quant?: string | null) => {
    try {
      await cancelDownload(modelId, quant);
      setPulls((prev) => {
        const next = { ...prev };
        delete next[modelId];
        if (quant) {
          delete next[`${modelId.toLowerCase()}:${quant.toLowerCase()}`];
        }
        return next;
      });
      refresh();
    } catch (err) {
      setErrorMsg(String(err));
    }
  };

  const handleImport = async () => {
    const selected = importPath.trim()
      ? importPath.trim()
      : await open({
          multiple: false,
          directory: false,
          filters: [{ name: "GGUF model", extensions: ["gguf"] }],
        });
    const path = Array.isArray(selected) ? selected[0] : selected;
    if (!path || typeof path !== "string") return;
    setImportBusy(true);
    setImportError(null);
    try {
      await api.libraryImportLocal(path);
      setImportOpen(false);
      setImportPath("");
      refresh();
    } catch (err) {
      setImportError(String(err));
    } finally {
      setImportBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    setErrorMsg(null);
    try {
      await api.libraryRemove(deleteTarget.model_id, deleteTarget.model_path, deleteTarget.quant);
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
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Model Library</h1>
          <p className="text-xs text-slate-400 mt-1">
            Locally downloaded model weights cached in WSL (<code className="font-mono text-slate-300">~/.cache/huggingface/hub</code>) and Windows GGUF folders.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <Button variant="ghost" onClick={refresh}>
            <RefreshIcon />
            <span>Refresh</span>
          </Button>
          <Button variant="primary" onClick={() => setImportOpen(true)}>
            <FolderImportIcon />
            <span>Import GGUF</span>
          </Button>
        </div>
      </div>

      {/* Disk Usage Overview Card */}
      <Card className="p-5 border-edge/80 bg-surface-2 shadow-sm">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <SquircleIcon color="purple" icon={<HardDriveIcon />} size="md" />
            <div>
              <div className="text-sm font-semibold text-slate-100">
                Local cache: <span className="text-accent font-bold font-mono">{totalGb} GB</span> used across{" "}
                <span className="text-slate-100 font-bold font-mono">{entries.length}</span> {entries.length === 1 ? "model" : "models"}
              </div>
              <div className="text-xs text-slate-400 mt-1 flex flex-wrap items-center gap-2">
                <span>Inference engines load directly from this local cache.</span>
                <span>·</span>
                <StoragePlanner modelWeightGb={0} compact />
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link to="/search">
              <Button variant="ghost" className="text-xs">
                Browse Hugging Face →
              </Button>
            </Link>
          </div>
        </div>
      </Card>

      {/* Consolidated Active Downloads Panel */}
      <ActiveDownloadsPanel className="mb-4" />

      {/* Filter and Search Bar with Segmented Control */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <SegmentedControl
          value={taskFilter}
          onChange={setTaskFilter}
          options={[
            { value: "all", label: "All", count: entries.length },
            { value: "instruct", label: "Instruct", count: instructCount },
            { value: "embed", label: "Embeddings", count: embedCount },
          ]}
        />

        <div className="w-full sm:w-72">
          <input
            type="text"
            className={inputCls}
            placeholder="Filter models by name…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Model Cards Grid */}
      {loading ? (
        <Card className="flex items-center justify-center p-12">
          <Spinner label="Scanning HF cache and local GGUFs…" />
        </Card>
      ) : entries.length === 0 ? (
        <Card className="p-12 text-center space-y-3">
          <CardTitle>No models pulled yet</CardTitle>
          <div className="text-sm text-slate-400 max-w-md mx-auto leading-relaxed">
            Your local storage is empty. Use <Link className="text-accent hover:underline font-medium" to="/search">Search</Link> to discover hardware-compatible models and download them.
          </div>
        </Card>
      ) : filteredEntries.length === 0 ? (
        <Card className="p-8 text-center text-sm text-slate-400">
          No cached models matched your filter criteria.
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredEntries.map((e) => {
            const pullKey = e.quant ? `${e.model_id.toLowerCase()}:${e.quant.toLowerCase()}` : e.model_id.toLowerCase();
            const pull = pulls[pullKey] || pulls[e.model_id];
            const activeJob = getDownload(e.model_id, e.quant) || (e.repo_id ? getDownload(e.repo_id, e.quant) : undefined);
            const isDownloading =
              activeJob?.status === "Downloading" ||
              activeJob?.status === "Resolving" ||
              pull?.state === "downloading";
            const sizeGb = (e.size_mb / 1024).toFixed(1);
            const stableKey = `${e.model_id}:${e.quant || "default"}:${e.model_path || ""}`;

            return (
              <Card key={stableKey} className="flex flex-col justify-between hover:border-accent/50 transition-all p-5 shadow-sm active:scale-[0.99]">
                <div>
                  <div className="flex items-start justify-between gap-2.5">
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-semibold text-slate-100 text-sm leading-snug" title={e.model_id}>
                        {e.model_id}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-400 font-mono">
                        <span>{sizeGb} GB ({fmtNum(e.size_mb)} MB)</span>
                        <span>·</span>
                        <span>{fmtNum(e.files)} files</span>
                      </div>
                    </div>
                    {isDownloading ? (
                      <Badge color="indigo">Downloading</Badge>
                    ) : pull ? (
                      <Badge color={pull.state === "complete" ? "emerald" : pull.state === "failed" ? "red" : "indigo"}>
                        {pull.state}
                      </Badge>
                    ) : e.in_use ? (
                      <Badge color="amber" title={`Running on server: ${e.in_use_server || "active"}`}>
                        In Use
                      </Badge>
                    ) : e.is_local ? (
                      <Badge color="slate">Local File</Badge>
                    ) : (
                      <Badge color="emerald">Cached</Badge>
                    )}
                  </div>

                  {/* Metadata Chips */}
                  <div className="mt-3.5 flex flex-wrap gap-1.5">
                    {e.params_b && (
                      <span className="rounded-md bg-surface-3 px-2 py-0.5 text-[11px] font-mono font-medium text-slate-300 border border-edge">
                        {e.params_b}B
                      </span>
                    )}
                    {e.quant && (
                      <span className="rounded-md bg-surface-3 px-2 py-0.5 text-[11px] font-mono font-medium text-slate-300 border border-edge">
                        {e.quant}
                      </span>
                    )}
                    {e.task && (
                      <span className="rounded-md bg-accent/15 px-2 py-0.5 text-[11px] font-medium text-accent border border-accent/25">
                        {e.task}
                      </span>
                    )}
                    {e.in_use && e.in_use_server && (
                      <span className="rounded-md bg-warn/15 px-2 py-0.5 text-[11px] font-medium text-amber-300 border border-warn/30">
                        Server: {e.in_use_server}
                      </span>
                    )}
                  </div>
                </div>

                <div className="mt-4 pt-3.5 border-t border-edge-subtle flex items-center justify-between gap-2">
                  {e.is_local && !e.model_path ? (
                    <span
                      className="text-[11px] text-slate-500"
                      title="Managed on disk outside the HF cache"
                    >
                      Local folder on disk
                    </span>
                  ) : isDownloading ? (
                    <Button
                      variant="danger"
                      onClick={() => handleCancelPull(activeJob?.id || e.model_id, e.quant)}
                      className="text-xs h-7 px-2.5"
                    >
                      Cancel
                    </Button>
                  ) : e.in_use ? (
                    <Button
                      variant="danger"
                      disabled
                      title={`Model is currently in use by server "${e.in_use_server || "active"}". Stop the server before deleting.`}
                      className="text-xs h-7 px-2.5 opacity-40 cursor-not-allowed"
                    >
                      Delete
                    </Button>
                  ) : (
                    <Button
                      variant="danger"
                      onClick={() => setDeleteTarget(e)}
                      className="text-xs h-7 px-2.5"
                    >
                      Delete
                    </Button>
                  )}

                  <Button
                    variant="ghost"
                    className="text-xs h-7 px-3"
                    disabled={isDownloading}
                    onClick={() =>
                      navigate("/servers", {
                        state: {
                          prefillModel: e.model_id,
                          prefillBackend:
                            e.quant?.toUpperCase() === "GGUF" || e.model_id.toUpperCase().includes("GGUF")
                              ? "llamacpp"
                              : undefined,
                          prefillModelPath:
                            e.model_path || (e.is_local ? e.model_id : undefined) || undefined,
                          prefillQuant: e.quant || undefined,
                          prefillTask: e.task === "embed" ? "embed" : "instruct",
                        },
                      })
                    }
                  >
                    Deploy →
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Import Local GGUF Modal */}
      {importOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-2xl border border-edge bg-surface-2 p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <SquircleIcon color="blue" icon={<FolderImportIcon />} size="md" />
              <div>
                <h3 className="text-base font-semibold text-slate-100">Import Local GGUF</h3>
                <p className="text-xs text-slate-400">Choose an existing .gguf file or enter its path</p>
              </div>
            </div>

            <div className="text-sm text-slate-300 space-y-3">
              <label className="block text-xs font-medium text-slate-300" htmlFor="import-path">
                GGUF file path (optional)
              </label>
              <input
                id="import-path"
                className={inputCls}
                placeholder="C:\models\model-Q4_K_M.gguf"
                value={importPath}
                onChange={(e) => setImportPath(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleImport();
                }}
                autoFocus
              />
              <div className="rounded-xl bg-surface-3 p-3 border border-edge text-xs space-y-1.5 text-slate-300">
                <div className="flex items-center gap-2">
                  <span className="text-accent font-semibold">Windows:</span>
                  <span className="font-mono text-slate-400">D:\AI\qwen</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 font-semibold">WSL:</span>
                  <span className="font-mono text-slate-400">/mnt/d/AI/qwen</span>
                </div>
                <p className="pt-1 text-slate-500">
                  The selected file is added to the native llama.cpp library.
                </p>
              </div>
              {importError && (
                <ErrorAlert
                  source="download"
                  error={importError}
                  title="Local Model Import Error"
                  onDismiss={() => setImportError(null)}
                />
              )}
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <Button
                variant="ghost"
                onClick={() => {
                  setImportOpen(false);
                  setImportError(null);
                  setImportPath("");
                }}
                disabled={importBusy}
              >
                Cancel
              </Button>
              <Button variant="primary" onClick={handleImport} disabled={importBusy}>
                {importBusy ? <Spinner label="Checking…" /> : "Choose / Import GGUF"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-2xl border border-bad/30 bg-surface-2 p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <SquircleIcon color="red" icon={<TrashIcon />} size="md" />
              <div>
                <h3 className="text-base font-semibold text-slate-100">Delete Model from Disk?</h3>
                <p className="text-xs text-slate-400">
                  {deleteTarget.model_path ? "Permanent removal from Windows GGUF library" : "Permanent removal from local WSL cache"}
                </p>
              </div>
            </div>

            <div className="text-sm text-slate-300 space-y-2.5">
              <p>
                Are you sure you want to delete <strong className="text-slate-100">{deleteTarget.model_id}{deleteTarget.quant ? ` (${deleteTarget.quant})` : ""}</strong>?
              </p>
              <div className="rounded-xl bg-surface-3 p-3 border border-edge text-xs space-y-1.5">
                <div className="flex justify-between text-slate-300">
                  <span>Freed space:</span>
                  <span className="font-semibold text-emerald-400 font-mono">
                    ~{(deleteTarget.size_mb / 1024).toFixed(1)} GB ({fmtNum(deleteTarget.size_mb)} MB)
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Files removed:</span>
                  <span className="font-mono">{fmtNum(deleteTarget.files)}</span>
                </div>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                {deleteTarget.model_path
                  ? "This frees space on your Windows disk. If needed later, re-download the GGUF from Search."
                  : "This frees disk space in your WSL virtual hard disk. If needed later, vLLM will automatically re-download it."}
              </p>
            </div>

            {errorMsg && (
              <ErrorAlert
                source="server"
                error={errorMsg}
                title="Model Removal Error"
                onDismiss={() => setErrorMsg(null)}
              />
            )}

            <div className="flex items-center justify-end gap-2.5 pt-2">
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
                {isDeleting ? <Spinner label="Deleting…" /> : "Delete from Disk"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}