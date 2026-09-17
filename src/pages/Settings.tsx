import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls } from "../ui";
import type { MemorySettings, Settings as SettingsT, SystemMemoryInfo, WslConfigInfo } from "../types";

export default function Settings() {
  const [s, setS] = useState<SettingsT | null>(null);
  const [wsl, setWsl] = useState<WslConfigInfo | null>(null);
  const [mem, setMem] = useState<MemorySettings | null>(null);
  const [sysMem, setSysMem] = useState<SystemMemoryInfo | null>(null);
  const [manualMode, setManualMode] = useState(false);
  const [saved, setSaved] = useState(false);
  const [memSaved, setMemSaved] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.settingsGet().then(setS).catch((e) => setErr(String(e)));
    api.wslconfigGet().then(setWsl).catch(() => {});
    api
      .getMemorySettings()
      .then((m) => {
        setMem(m);
        setManualMode(m.manual_ram_limit_mb !== null);
      })
      .catch((e) => setErr(String(e)));
    api.getSystemMemory().then(setSysMem).catch(() => {});
  }, []);

  const save = async () => {
    if (!s) return;
    setErr(null);
    setSaved(false);
    try {
      const updated = await api.settingsSet({
        distro: s.distro,
        llm_dir: s.llm_dir,
        venv_dir: s.venv_dir,
        hf_token: s.hf_token,
        default_quant: s.default_quant,
      });
      setS(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setErr(String(e));
    }
  };

  const saveMem = async () => {
    if (!mem) return;
    setErr(null);
    setMemSaved(false);
    try {
      const toSave: MemorySettings = {
        ...mem,
        manual_ram_limit_mb: manualMode ? mem.manual_ram_limit_mb : null,
      };
      await api.updateMemorySettings(toSave);
      setMem(toSave);
      setMemSaved(true);
      const refreshedSys = await api.getSystemMemory();
      setSysMem(refreshedSys);
      setTimeout(() => setMemSaved(false), 2000);
    } catch (e) {
      setErr(String(e));
    }
  };

  if (!s || !mem) {
    return <div className="p-6 text-sm text-slate-500">Loading settings…</div>;
  }

  return (
    <div className="mx-auto max-w-4xl p-6 space-y-5">
      <h1 className="text-xl font-bold text-slate-100">Settings</h1>
      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{err}</div>}

      <Card>
        <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>WSL & environment</CardTitle>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field label="WSL distro">
            <input className={inputCls} value={s.distro} onChange={(e) => setS({ ...s, distro: e.target.value })} />
          </Field>
          <Field label="Default quantization">
            <select className={inputCls} value={s.default_quant} onChange={(e) => setS({ ...s, default_quant: e.target.value })}>
              <option value="fp16">FP16</option>
              <option value="fp8">FP8</option>
              <option value="awq">AWQ</option>
              <option value="gptq">GPTQ</option>
            </select>
          </Field>
          <Field label="LLM dir (inside WSL)" hint="provision creates it; logs/pid files live here">
            <input className={inputCls} value={s.llm_dir} onChange={(e) => setS({ ...s, llm_dir: e.target.value })} />
          </Field>
          <Field label="Venv dir (inside WSL)">
            <input className={inputCls} value={s.venv_dir} onChange={(e) => setS({ ...s, venv_dir: e.target.value })} />
          </Field>
          <div className="sm:col-span-2">
            <Field label="Hugging Face token (for gated models)" hint="stored in %APPDATA% config, passed as HF_TOKEN to pulls and vLLM launches">
              <input className={inputCls} type="password" placeholder="hf_…" value={s.hf_token} onChange={(e) => setS({ ...s, hf_token: e.target.value })} />
            </Field>
          </div>
        </div>
        <div className="mt-4 flex justify-end">
          <Button onClick={save}>Save changes</Button>
        </div>
      </Card>

      <Card>
        <CardTitle right={memSaved ? <Badge color="emerald">saved</Badge> : undefined}>
          Hardware & Memory Tuning
        </CardTitle>

        <div className="space-y-4">
          {/* Live Telemetry Card */}
          <div className="rounded-lg border border-edge bg-surface/80 p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                WSL2 System Memory Telemetry
              </span>
              {sysMem && (
                <span className="text-[11px] text-slate-500">
                  {mem.enable_ram_overflow ? "RAM overflow active" : "RAM overflow disabled"}
                </span>
              )}
            </div>
            {sysMem ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
                <div className="rounded-md border border-edge/60 bg-surface-2/60 p-2.5">
                  <div className="text-[11px] text-slate-400">WSL2 Total RAM</div>
                  <div className="text-base font-bold text-slate-100 font-mono">
                    {(sysMem.wsl_total_mb / 1024).toFixed(1)} GB
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    {fmtNum(sysMem.wsl_total_mb)} MB
                  </div>
                </div>
                <div className="rounded-md border border-edge/60 bg-surface-2/60 p-2.5">
                  <div className="text-[11px] text-slate-400">WSL2 Available RAM</div>
                  <div className="text-base font-bold text-cyan-300 font-mono">
                    {(sysMem.wsl_available_mb / 1024).toFixed(1)} GB
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    {fmtNum(sysMem.wsl_available_mb)} MB
                  </div>
                </div>
                <div className="rounded-md border border-edge/60 bg-surface-2/60 p-2.5">
                  <div className="text-[11px] text-slate-400">Usable Overflow Budget</div>
                  <div
                    className={`text-base font-bold font-mono ${
                      mem.enable_ram_overflow ? "text-emerald-400" : "text-slate-500"
                    }`}
                  >
                    {mem.enable_ram_overflow
                      ? `${(sysMem.usable_budget_mb / 1024).toFixed(1)} GB`
                      : "0 GB (Disabled)"}
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    {mem.enable_ram_overflow
                      ? `${fmtNum(sysMem.usable_budget_mb)} MB`
                      : "toggle overflow to enable"}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 py-1">Loading system memory metrics…</div>
            )}
          </div>

          {/* GPU VRAM & VRAM Overhead */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label={`GPU VRAM Utilization (${Math.round(mem.default_gpu_mem_util * 100)}%)`}
              hint="Target fraction of GPU VRAM allocated for model weights and KV cache (50% – 98%, default: 92%)."
            >
              <div className="flex items-center gap-3 pt-1">
                <input
                  type="range"
                  min="50"
                  max="98"
                  step="1"
                  value={Math.round(mem.default_gpu_mem_util * 100)}
                  onChange={(e) =>
                    setMem({ ...mem, default_gpu_mem_util: Number(e.target.value) / 100 })
                  }
                  className="h-2 w-full cursor-pointer accent-indigo-500 rounded-lg bg-surface"
                />
                <span className="w-12 font-mono text-sm font-semibold text-indigo-400">
                  {Math.round(mem.default_gpu_mem_util * 100)}%
                </span>
              </div>
            </Field>

            <Field
              label="VRAM Overhead Buffer (MB)"
              hint="Reserved VRAM for CUDA context, runtime memory, and display output (default: 2500 MB)."
            >
              <input
                className={inputCls}
                type="number"
                min="500"
                step="100"
                value={mem.vram_overhead_mb}
                onChange={(e) =>
                  setMem({ ...mem, vram_overhead_mb: Number(e.target.value) || 0 })
                }
              />
            </Field>
          </div>

          {/* Toggles for RAM Overflow & Weight Offloading */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
              <input
                type="checkbox"
                checked={mem.enable_ram_overflow}
                onChange={(e) =>
                  setMem({ ...mem, enable_ram_overflow: e.target.checked })
                }
                className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
              />
              <div className="space-y-0.5">
                <div className="text-sm font-medium text-slate-200">
                  Enable RAM Context Overflow
                </div>
                <div className="text-xs text-slate-400">
                  Allows context window to spill into WSL2 system RAM via <code>--swap-space</code> when VRAM is full.
                </div>
              </div>
            </label>

            <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
              <input
                type="checkbox"
                checked={mem.offload_weights_allowed}
                onChange={(e) =>
                  setMem({ ...mem, offload_weights_allowed: e.target.checked })
                }
                className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
              />
              <div className="space-y-0.5">
                <div className="text-sm font-medium text-slate-200">
                  Allow Weight Offloading
                </div>
                <div className="text-xs text-slate-400">
                  Permits offloading model weights to CPU RAM via <code>--cpu-offload-gb</code> if model exceeds VRAM.
                </div>
              </div>
            </label>
          </div>

          {/* RAM Safety Reserve & Global Context Cap */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label="RAM Safety Reserve (MB)"
              hint="WSL2 system RAM kept untouched for Linux kernel and other processes (default: 4096 MB)."
            >
              <input
                className={inputCls}
                type="number"
                min="512"
                step="512"
                value={mem.safety_reserve_mb}
                onChange={(e) =>
                  setMem({ ...mem, safety_reserve_mb: Number(e.target.value) || 0 })
                }
              />
            </Field>

            <Field
              label="Global Context Cap (Tokens)"
              hint="Optional ceiling on context window lengths (e.g. 65536). Leave blank for uncapped."
            >
              <input
                className={inputCls}
                type="number"
                min="1024"
                step="1024"
                placeholder="Uncapped"
                value={mem.max_context_cap ?? ""}
                onChange={(e) => {
                  const val = e.target.value.trim();
                  setMem({ ...mem, max_context_cap: val === "" ? null : Number(val) });
                }}
              />
            </Field>

            <div className="sm:col-span-2">
              <Field
                label="RAM Budget Mode"
                hint="Auto dynamically sizes swap budget from available WSL2 RAM minus safety reserve. Manual locks a specific limit."
              >
                <div className="space-y-2 pt-1">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setManualMode(false);
                        setMem({ ...mem, manual_ram_limit_mb: null });
                      }}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                        !manualMode
                          ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 font-semibold"
                          : "bg-surface text-slate-400 border-edge hover:text-slate-200"
                      }`}
                    >
                      Auto (Dynamic)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setManualMode(true);
                        if (mem.manual_ram_limit_mb === null) {
                          setMem({ ...mem, manual_ram_limit_mb: sysMem?.usable_budget_mb || 8192 });
                        }
                      }}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                        manualMode
                          ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 font-semibold"
                          : "bg-surface text-slate-400 border-edge hover:text-slate-200"
                      }`}
                    >
                      Manual Override (MB)
                    </button>
                  </div>

                  {manualMode && (
                    <div className="flex items-center gap-2 pt-1 max-w-sm">
                      <input
                        className={inputCls}
                        type="number"
                        min="1024"
                        step="1024"
                        placeholder="e.g. 16384"
                        value={mem.manual_ram_limit_mb ?? ""}
                        onChange={(e) => {
                          const v = e.target.value === "" ? null : Number(e.target.value);
                          setMem({ ...mem, manual_ram_limit_mb: v });
                        }}
                      />
                      <span className="text-xs text-slate-400 whitespace-nowrap">
                        MB {mem.manual_ram_limit_mb ? `(~${(mem.manual_ram_limit_mb / 1024).toFixed(1)} GB)` : ""}
                      </span>
                    </div>
                  )}
                </div>
              </Field>
            </div>
          </div>

          <div className="mt-4 flex justify-end">
            <Button onClick={saveMem}>Save memory settings</Button>
          </div>
        </div>
      </Card>

      <Card>
        <CardTitle>Measured stats (from live runs)</CardTitle>
        <div className="space-y-2 text-sm">
          {Object.keys(s.measured).length === 0 ? (
            <div className="text-slate-500">No measured data yet — run a server and send traffic.</div>
          ) : (
            Object.entries(s.measured).map(([model, m]) => (
              <div key={model} className="flex flex-wrap items-center gap-2 rounded-md border border-edge bg-surface px-3 py-2">
                <span className="flex-1 truncate text-slate-200">{model}</span>
                <span className="font-mono text-xs text-cyan-300">
                  {m.tokens_per_sec != null ? `${m.tokens_per_sec.toFixed(1)} tok/s` : "—"}
                </span>
                <span className="text-xs text-slate-500">{fmtNum(m.total_generation_tokens)} gen</span>
              </div>
            ))
          )}
        </div>
      </Card>

      <Card>
        <CardTitle
          right={
            <Button
              variant="ghost"
              onClick={() => navigator.clipboard?.writeText(wsl?.content ?? "")}
              disabled={!wsl?.content}
            >
              Copy
            </Button>
          }
        >
          .wslconfig <span className="text-xs font-normal text-slate-500">(read-only — the app never overwrites it)</span>
        </CardTitle>
        <div className="text-xs text-slate-500">
          {wsl?.path ?? "no path found"} — recommended:{" "}
          <code>memory=24GB</code> for 7B+ models, <code>nestedVirtualization=true</code>.
        </div>
        <pre className="mt-2 rounded-md bg-black/30 p-3 font-mono text-xs text-slate-300">{wsl?.content ?? "(no .wslconfig yet — create one at C:\\Users\\you\\.wslconfig)"}</pre>
      </Card>
    </div>
  );
}