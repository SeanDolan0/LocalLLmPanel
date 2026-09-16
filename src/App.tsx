import { NavLink, Route, Routes } from "react-router-dom";
import { MemoryRouter } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Search from "./pages/Search";
import Library from "./pages/Library";
import Servers from "./pages/Servers";
import Settings from "./pages/Settings";

const nav = [
  { to: "/", label: "Dashboard", icon: "◧" },
  { to: "/search", label: "Search", icon: "⌕" },
  { to: "/library", label: "Library", icon: "▤" },
  { to: "/servers", label: "Servers", icon: "⧉" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

export default function App() {
  return (
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
          <div className="px-4 py-4 text-[10px] text-slate-600 leading-relaxed">
            Estimates are estimates.
            <br />
            Measured data wins.
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/search" element={<Search />} />
            <Route path="/library" element={<Library />} />
            <Route path="/servers" element={<Servers />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </MemoryRouter>
  );
}