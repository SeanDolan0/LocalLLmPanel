import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, events, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle } from "../ui";
import type { LibraryEntry, PullStatus } from "../types";

export default function Library() {
  const navigate = useNavigate();
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({});

  const refresh = useCallback(() => {
    api
      .libraryList()
      .then(setEntries)
      .catch(() => setEntries([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    const unsub = events.pullProgress((p) => {
      setPulls((prev) => ({ ...prev, [p.model]: p }));
      if (p.state === "complete") refresh();
    });
    return () => {
      clearInterval(t);
      unsub.then((f) => f());
    };
  }, [refresh]);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-100">Library</h1>
        <Button variant="ghost" onClick={refresh}>↻ Refresh</Button>
      </div>

      <div className="text-xs text-slate-500">
        Models cached in WSL (<code>~/.cache/huggingface/hub</code>). Pulling is optional — vLLM auto-downloads at
        first start.
      </div>

      {loading ? (
        <Card>Scanning HF cache…</Card>
      ) : entries.length === 0 ? (
        <Card>
          <CardTitle>No models pulled yet</CardTitle>
          <div className="text-sm text-slate-500">
            Use <Link className="text-indigo-300 hover:underline" to="/search">Search</Link> to find models and pull them.
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {entries.map((e) => {
            const pull = pulls[e.model_id];
            return (
              <Card key={e.model_id} className="hover:border-indigo-500/40 transition-colors flex flex-col justify-between">
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate font-medium text-slate-200" title={e.model_id}>{e.model_id}</div>
                      <div className="mt-1 text-xs text-slate-500">
                        {fmtNum(e.size_mb)} MB · {fmtNum(e.files)} files
                      </div>
                    </div>
                    {pull ? (
                      <Badge color={pull.state === "complete" ? "emerald" : pull.state === "failed" ? "red" : "indigo"}>
                        {pull.state}
                      </Badge>
                    ) : (
                      <Badge color="emerald">cached</Badge>
                    )}
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-edge/60 flex items-center justify-end">
                  <Button
                    variant="ghost"
                    onClick={() => navigate("/servers", { state: { prefillModel: e.model_id } })}
                  >
                    Deploy Server →
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}