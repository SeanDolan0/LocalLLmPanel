import { useState, useEffect, type ReactNode } from "react";
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

function DashboardIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="7" height="9" x="3" y="3" rx="1" />
      <rect width="7" height="5" x="14" y="3" rx="1" />
      <rect width="7" height="9" x="14" y="12" rx="1" />
      <rect width="7" height="5" x="3" y="16" rx="1" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m7.5 4.27 9 5.15" />
      <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
      <path d="m3.3 7 8.7 5 8.7-5" />
      <path d="M12 22V12" />
    </svg>
  );
}

function ServersIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="8" x="2" y="2" rx="2" />
      <rect width="20" height="8" x="2" y="14" rx="2" />
      <line x1="6" x2="6.01" y1="6" y2="6" />
      <line x1="6" x2="6.01" y1="18" y2="18" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

const nav: { to: string; label: string; icon: ReactNode }[] = [
  { to: "/", label: "Dashboard", icon: <DashboardIcon /> },
  { to: "/search", label: "Search", icon: <SearchIcon /> },
  { to: "/library", label: "Library", icon: <LibraryIcon /> },
  { to: "/servers", label: "Servers", icon: <ServersIcon /> },
  { to: "/settings", label: "Settings", icon: <SettingsIcon /> },
];

function SidebarDownloadIndicator() {
  const { activeCount, overallPercent } = useDownloads();
  if (activeCount === 0) return null;

  return (
    <div className="px-3 pb-2.5">
      <NavLink
        to="/library"
        className="block rounded-xl border border-accent/25 bg-accent/10 p-2.5 text-xs hover:border-accent/40 hover:bg-accent/15 transition-all group shadow-sm"
        title="View active downloads"
      >
        <div className="flex items-center justify-between text-accent font-medium">
          <span className="flex items-center gap-2 truncate">
            <span className="h-2 w-2 rounded-full bg-accent animate-pulse shrink-0" />
            <span className="truncate">Downloading ({activeCount})</span>
          </span>
          <span className="font-mono text-[11px] text-slate-200 shrink-0">{overallPercent}%</span>
        </div>
        <div className="w-full bg-surface-3 h-1.5 rounded-full overflow-hidden mt-2">
          <div
            className="bg-accent h-full rounded-full transition-all duration-300 ease-out"
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
  const [updateBanner, setUpdateBanner] = useState<{ type: "success" | "error"; message: string } | null>(null);

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

    // Check if an update attempt just completed or failed
    api.lastUpdateStatus()
      .then((status) => {
        if (status) {
          if (status.status === "success") {
            const ver = status.expected_version ? ` to v${status.expected_version.replace(/^v/, "")}` : "";
            setUpdateBanner({
              type: "success",
              message: `Restarted successfully with update${ver}.`,
            });
          } else if (status.status === "failed") {
            setUpdateBanner({
              type: "error",
              message: `Update installation failed: ${status.message}`,
            });
          }
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
        <div className="flex h-screen w-screen overflow-hidden bg-surface">
          {/* Sidebar */}
          <aside className="w-56 shrink-0 border-r border-edge bg-surface-2/95 backdrop-blur-md flex flex-col justify-between">
            <div>
              {/* App Identity Header */}
              <div className="px-4 py-4 flex items-center gap-3">
                <div className="h-9 w-9 rounded-xl bg-accent text-white flex items-center justify-center font-bold text-sm shadow-md shadow-accent/20 shrink-0">
                  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
                    <rect width="18" height="18" x="3" y="3" rx="2" />
                    <path d="M9 3v18" />
                    <path d="m14 9 3 3-3 3" />
                  </svg>
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-100 leading-tight truncate">Local LLM Panel</div>
                  <div className="text-[11px] text-slate-400 leading-tight truncate mt-0.5">Inference & Gateway</div>
                </div>
              </div>

              {/* Navigation Items */}
              <nav className="px-2.5 pt-1 space-y-1">
                {nav.map((n) => (
                  <NavLink
                    key={n.to}
                    to={n.to}
                    end={n.to === "/"}
                    className={({ isActive }) =>
                      `relative flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-150 ${
                        isActive
                          ? "bg-accent/15 text-accent shadow-xs"
                          : "text-slate-400 hover:bg-surface-3/70 hover:text-slate-100"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {isActive && (
                          <span className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-accent" />
                        )}
                        <span className={`shrink-0 ${isActive ? "text-accent" : "text-slate-400"}`}>
                          {n.icon}
                        </span>
                        <span>{n.label}</span>
                      </>
                    )}
                  </NavLink>
                ))}
              </nav>
            </div>

            {/* Bottom Section */}
            <div>
              {/* Sidebar Active Downloads Bar */}
              <SidebarDownloadIndicator />

              {/* Update Available Banner in Sidebar */}
              {updateInfo?.available && (
                <div className="px-3 pb-2.5">
                  <button
                    onClick={() => setShowUpdateModal(true)}
                    className="w-full flex items-center justify-between gap-2 rounded-xl border border-accent/30 bg-accent/10 p-2.5 text-left hover:border-accent hover:bg-accent/15 transition-all shadow-sm group"
                    title={`Update ${updateInfo.latest_version} available`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="relative flex h-2 w-2 shrink-0">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
                      </span>
                      <div className="truncate">
                        <div className="text-xs font-semibold text-slate-100 group-hover:text-white leading-tight truncate">
                          Update Available
                        </div>
                        <div className="text-[10px] text-slate-400 truncate">
                          {updateInfo.latest_version}
                        </div>
                      </div>
                    </div>
                    <span className="text-[10px] font-semibold text-accent shrink-0 bg-accent/20 rounded-md px-2 py-0.5">
                      View
                    </span>
                  </button>
                </div>
              )}

              <div className="px-4 py-3.5 border-t border-edge-subtle text-[11px] text-slate-500 leading-relaxed">
                Measured performance wins.
              </div>
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="flex-1 overflow-y-auto">
            {updateBanner && (
              <div
                className={`flex items-center justify-between px-5 py-2.5 text-xs border-b ${
                  updateBanner.type === "success"
                    ? "bg-good/10 border-good/20 text-emerald-300"
                    : "bg-bad/10 border-bad/20 text-red-300"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-semibold">{updateBanner.type === "success" ? "✓" : "!"}</span>
                  <span>{updateBanner.message}</span>
                </div>
                <button
                  onClick={() => setUpdateBanner(null)}
                  className="text-slate-400 hover:text-slate-200 ml-4 font-bold text-sm leading-none"
                  title="Dismiss"
                >
                  ✕
                </button>
              </div>
            )}
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