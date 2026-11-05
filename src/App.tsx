import { useState, useEffect } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { MemoryRouter } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Search from "./pages/Search";
import Library from "./pages/Library";
import Servers from "./pages/Servers";
import Settings from "./pages/Settings";
import UpdateModal from "./components/UpdateModal";
import ErrorBoundary from "./components/ErrorBoundary";
import { api } from "./api";
import type { AppUpdateInfo } from "./types";
import { DownloadProvider, useDownloads } from "./context/DownloadContext";

const nav = [
  { to: "/", label: "Dashboard", icon: "◧" },
  { to: "/search", label: "Search", icon: "⌕" },
  { to: "/library", label: "Library", icon: "▤" },
  { to: "/servers", label: "Servers", icon: "⧉" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

function SidebarDownloadIndicator() {
  const { activeCount, overallPercent } = useDownloads();
  if (activeCount === 0) return null;

  return (
    <div className="px-2.5 pb-2">
      <NavLink
        to="/library"
        className="block rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2 text-xs hover:border-indigo-400/50 hover:bg-indigo-500/15 transition-all group shadow-sm"
        title="View active downloads"
      >
        <div className="flex items-center justify-between text-indigo-300 font-medium">
          <span className="flex items-center gap-1.5 truncate">
            <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse shrink-0" />
            <span className="truncate">Downloading ({activeCount})</span>
          </span>
          <span className="font-mono text-[11px] text-indigo-200 shrink-0">{overallPercent}%</span>
        </div>
        <div className="w-full bg-surface-3 h-1.5 rounded-full overflow-hidden mt-1.5">
          <div
            className="bg-indigo-500 h-full rounded-full transition-all duration-300 ease-out"
            style={{ width: `${overallPercent}%` }}
          />
        </div>
      </NavLink>
    </div>
  );
}

export default function App() {
  const [updateInfo, setUpdateInfo] = useState<AppUpdateInfo | null>(null);
  const [showUpdateModal, setShowUpdateModal] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onAppUpdateAvailable((info) => {
      if (info.available) {
        setUpdateInfo(info);
      }
    }).then((fn) => {
      unlisten = fn;
    });

    // Check updates quietly on mount (respecting the 10-minute cache)
    api.checkAppUpdate(false)
      .then((info) => {
        if (info.available) {
          setUpdateInfo(info);
        }
      })
      .catch(() => {});

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  return (
    <DownloadProvider>
      <MemoryRouter>
        <div className="flex h-screen w-screen overflow-hidden">
          {/* Sidebar */}
          <aside className="w-52 shrink-0 border-r border-edge bg-surface-2 flex flex-col">
            <div className="px-4 py-5 flex items-center gap-2.5">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center text-black font-bold text-sm shadow-lg shadow-indigo-500/20">
                LLM
              </div>
              <div>
                <div className="text-sm font-semibold leading-tight">LLM Panel</div>
                <div className="text-[10px] text-slate-500 leading-tight">vLLM · WSL2</div>
              </div>
            </div>
            <nav className="flex-1 px-2 space-y-0.5">
              {nav.map((n) => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.to === "/"}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
                      isActive
                        ? "bg-indigo-500/15 text-indigo-300 font-medium"
                        : "text-slate-400 hover:bg-surface-3 hover:text-slate-200"
                    }`
                  }
                >
                  <span className="w-4 text-center text-base leading-none opacity-80">{n.icon}</span>
                  {n.label}
                </NavLink>
              ))}
            </nav>

            {/* Sidebar Active Downloads Bar */}
            <SidebarDownloadIndicator />

            {/* Update Available Banner in Sidebar */}
            {updateInfo?.available && (
              <div className="px-2.5 pb-2">
                <button
                  onClick={() => setShowUpdateModal(true)}
                  className="w-full flex items-center justify-between gap-1.5 rounded-lg border border-cyan-500/40 bg-gradient-to-r from-indigo-500/15 to-cyan-500/15 p-2 text-left hover:border-cyan-400 hover:from-indigo-500/25 hover:to-cyan-500/25 transition-all shadow-sm group"
                  title={`Update ${updateInfo.latest_version} available`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <span className="relative flex h-2 w-2 shrink-0">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                    </span>
                    <div className="truncate">
                      <div className="text-[11px] font-semibold text-cyan-200 group-hover:text-white leading-tight truncate">
                        Update Ready
                      </div>
                      <div className="text-[10px] text-slate-400 truncate">
                        {updateInfo.latest_version}
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] font-medium text-cyan-300 shrink-0 bg-cyan-500/20 rounded px-1.5 py-0.5">
                    View
                  </span>
                </button>
              </div>
            )}

            <div className="px-4 py-4 text-[10px] text-slate-600 leading-relaxed">
              Estimates are estimates.
              <br />
              Measured data wins.
            </div>
          </aside>

          {/* Main */}
          <main className="flex-1 overflow-y-auto">
            <ErrorBoundary variant="page" dismissLabel="Back to search">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/search" element={<Search />} />
                <Route path="/library" element={<Library />} />
                <Route path="/servers" element={<Servers />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </ErrorBoundary>
          </main>

          {/* Update Modal */}
          {updateInfo && (
            <UpdateModal
              info={updateInfo}
              isOpen={showUpdateModal}
              onClose={() => setShowUpdateModal(false)}
            />
          )}
        </div>
      </MemoryRouter>
    </DownloadProvider>
  );
}