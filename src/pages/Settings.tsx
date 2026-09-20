import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls } from "../ui";
import type {
  AdvancedSettings,
  MemorySettings,
  Settings as SettingsT,
  SystemMemoryInfo,
  WslConfigInfo,
} from "../types";

export default function Settings() {
  const [s, setS] = useState<SettingsT | null>(null);
  const [wsl, setWsl] = useState<WslConfigInfo | null>(null);
  const [mem, setMem] = useState<MemorySettings | null>(null);
  const [sysMem, setSysMem] = useState<SystemMemoryInfo | null>(null);
  const [manualMode, setManualMode] = useState(false);
  const [saved, setSaved] = useState(false);
  const [memSaved, setMemSaved] = useState(false);
  const [distros, setDistros] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // Simple vs. Advanced mode toggle with localStorage persistence
  const [mode, setMode] = useState<"simple" | "advanced">(() => {
    return (localStorage.getItem("llm_panel_settings_mode") as "simple" | "advanced") || "simple";
  });

  const toggleMode = (newMode: "simple" | "advanced") => {
    setMode(newMode);
    localStorage.setItem("llm_panel_settings_mode", newMode);
  };

  useEffect(() => {
    api
      .settingsGet()
      .then((cfg) => {
        setS(cfg);
        api
          .autostartGet()
          .then((auto) => {
            setS((prev) => (prev ? { ...prev, launch_at_login: auto } : null));
          })
          .catch(() => {});
      })
      .catch((e) => setErr(String(e)));
    api.wslDistros().then(setDistros).catch(() => {});
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
        advanced_settings: s.advanced_settings,
        minimize_to_tray: s.minimize_to_tray,
        resume_servers_on_launch: s.resume_servers_on_launch,
        auto_restart_crashed: s.auto_restart_crashed,
        launch_at_login: s.launch_at_login,
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

  // Safe fallback for advanced settings
  const adv: AdvancedSettings = s.advanced_settings ?? {
    hf_home: null,
    hf_offline: false,
    host: "127.0.0.1",
    api_key: null,
    kv_cache_dtype: "auto",
    enable_prefix_caching: true,
    enable_chunked_prefill: false,
    max_num_seqs: null,
    disable_custom_all_reduce: false,
    log_level: "INFO",
    extra_vllm_args: null,
    custom_env_vars: null,
  };

  const updateAdv = (patch: Partial<AdvancedSettings>) => {
    setS({
      ...s,
      advanced_settings: {
        ...adv,
        ...patch,
      },
    });
  };

  const renderApplianceCard = () => (
    <Card>
      <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
        Appliance & Startup Behavior
      </CardTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
          <input
            type="checkbox"
            checked={!!s.minimize_to_tray}
            onChange={(e) => setS({ ...s, minimize_to_tray: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="text-sm font-medium text-slate-200">
              Minimize to System Tray
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Closing the window hides it into the system notification tray instead of stopping servers.
            </div>
          </div>
        </label>

        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
          <input
            type="checkbox"
            checked={!!s.launch_at_login}
            onChange={(e) => setS({ ...s, launch_at_login: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="text-sm font-medium text-slate-200">
              Launch at Windows Login
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Starts LocalLLM Panel automatically when signing into Windows (via HKCU Run registry).
            </div>
          </div>
        </label>

        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
          <input
            type="checkbox"
            checked={!!s.resume_servers_on_launch}
            onChange={(e) => setS({ ...s, resume_servers_on_launch: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="text-sm font-medium text-slate-200">
              Auto-Resume Running Servers
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Automatically boots up model servers that were actively running when the panel was closed.
            </div>
          </div>
        </label>

        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
          <input
            type="checkbox"
            checked={!!s.auto_restart_crashed}
            onChange={(e) => setS({ ...s, auto_restart_crashed: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="text-sm font-medium text-slate-200">
              Auto-Restart Crashed Servers
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Monitors background server processes and restarts them with backoff if they crash.
            </div>
          </div>
        </label>
      </div>

      <div className="mt-4 flex justify-end">
        <Button onClick={save}>Save Startup Settings</Button>
      </div>
    </Card>
  );

  return (
    <div className="mx-auto max-w-4xl p-6 space-y-6">
      {/* Header and Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-edge/60 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-bold text-slate-100">Settings</h1>
            <Badge color={mode === "advanced" ? "indigo" : "slate"}>
              {mode === "advanced" ? "Advanced Mode" : "Simple Mode"}
            </Badge>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            {mode === "simple"
              ? "Essential settings for running models smoothly with sensible automated defaults."
              : "Complete control over WSL paths, vLLM engine optimization flags, storage drives, and network security."}
          </p>
        </div>

        {/* Toggle Switch */}
        <div className="inline-flex items-center rounded-lg bg-surface-2 p-1 border border-edge shrink-0">
          <button
            type="button"
            onClick={() => toggleMode("simple")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === "simple"
                ? "bg-indigo-600 text-white shadow-sm font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>Simple</span>
          </button>
          <button
            type="button"
            onClick={() => toggleMode("advanced")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === "advanced"
                ? "bg-indigo-600 text-white shadow-sm font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>Advanced</span>
            <span className="text-[10px] px-1 py-0.2 rounded bg-indigo-950/70 text-indigo-200 font-mono">
              dev
            </span>
          </button>
        </div>
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
          {err}
        </div>
      )}

      {/* ===================================================================== */}
      {/* SIMPLE MODE VIEW                                                      */}
      {/* ===================================================================== */}
      {mode === "simple" && (
        <div className="space-y-5">
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              Core Environment
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="WSL Distribution"
                hint={
                  distros.length > 0
                    ? `Installed distros: ${distros.join(", ")}`
                    : "e.g. Ubuntu-22.04"
                }
              >
                <div className="space-y-1">
                  <input
                    list="wsl-distros-simple"
                    className={inputCls}
                    value={s.distro}
                    onChange={(e) => setS({ ...s, distro: e.target.value })}
                  />
                  <datalist id="wsl-distros-simple">
                    {distros.map((d) => (
                      <option key={d} value={d} />
                    ))}
                  </datalist>
                  {distros.length > 0 && !distros.includes(s.distro) && (
                    <div className="text-[11px] text-amber-400">
                      ⚠️ "{s.distro}" was not found in installed WSL distributions.
                    </div>
                  )}
                </div>
              </Field>

              <Field
                label="Default Quantization"
                hint="Preferred precision used when deploying models from search."
              >
                <select
                  className={inputCls}
                  value={s.default_quant}
                  onChange={(e) => setS({ ...s, default_quant: e.target.value })}
                >
                  <option value="fp16">FP16 (Full Precision / Highest Quality)</option>
                  <option value="fp8">FP8 (Fast 8-bit / Modern RTX 40 & Ada)</option>
                  <option value="awq">AWQ (4-bit GPU Optimized / High VRAM Savings)</option>
                  <option value="gptq">GPTQ (4-bit Standard Quantization)</option>
                </select>
              </Field>

              <div className="sm:col-span-2">
                <Field
                  label="Hugging Face Token"
                  hint="Optional: Required for gated model families like Llama 3, Gemma, or Mistral."
                >
                  <input
                    className={inputCls}
                    type="password"
                    placeholder="hf_…"
                    value={s.hf_token}
                    onChange={(e) => setS({ ...s, hf_token: e.target.value })}
                  />
                </Field>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Changes</Button>
            </div>
          </Card>

          <Card>
            <CardTitle right={memSaved ? <Badge color="emerald">saved</Badge> : undefined}>
              GPU & Memory Preset
            </CardTitle>
            <div className="space-y-4">
              <Field
                label={`GPU VRAM Allocation (${Math.round(mem.default_gpu_mem_util * 100)}%)`}
                hint="Fraction of GPU VRAM allocated for model weights and KV cache."
              >
                <div className="space-y-3 pt-1">
                  {/* Preset quick buttons */}
                  <div className="flex gap-2">
                    {[
                      { label: "Conservative (85%)", val: 0.85 },
                      { label: "Balanced (92%)", val: 0.92 },
                      { label: "Performance (98%)", val: 0.98 },
                    ].map((p) => {
                      const active = Math.round(mem.default_gpu_mem_util * 100) === Math.round(p.val * 100);
                      return (
                        <button
                          key={p.val}
                          type="button"
                          onClick={() => setMem({ ...mem, default_gpu_mem_util: p.val })}
                          className={`px-3 py-1.5 text-xs rounded-md border transition-colors ${
                            active
                              ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 font-semibold"
                              : "bg-surface-2 text-slate-400 border-edge hover:text-slate-200"
                          }`}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>

                  <div className="flex items-center gap-3">
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
                </div>
              </Field>

              {/* Simple RAM context overflow toggle */}
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
                    Allow RAM Context Overflow (RAM Swap)
                  </div>
                  <div className="text-xs text-slate-400 leading-relaxed">
                    Automatically uses WSL2 system RAM when model context exceeds GPU VRAM, preventing out-of-memory errors on large prompts.
                  </div>
                </div>
              </label>

              <div className="mt-4 flex justify-end">
                <Button onClick={saveMem}>Save Memory Preset</Button>
              </div>
            </div>
          </Card>

          {renderApplianceCard()}

          {/* Quick Notice to Switch to Advanced Mode */}
          <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-4 text-xs text-slate-400 flex items-start gap-3">
            <span className="text-indigo-400 text-base">💡</span>
            <div>
              <div className="font-semibold text-slate-200">Looking for advanced developer controls?</div>
              <p className="mt-0.5 leading-relaxed">
                Toggle to <strong className="text-indigo-300">Advanced Mode</strong> in the top-right to configure custom model cache directories (e.g. secondary drives like <code>/mnt/d/...</code>), network IP binding (<code>0.0.0.0</code> for LAN WebUI access), API keys, FP8 KV-caching, prefix caching, and custom environment variables.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* ADVANCED DEVELOPER MODE VIEW                                          */}
      {/* ===================================================================== */}
      {mode === "advanced" && (
        <div className="space-y-6">
          {/* Section 1: WSL & Storage Management */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              WSL & Model Storage
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="WSL Distribution"
                hint={
                  distros.length > 0
                    ? `Installed distros: ${distros.join(", ")}`
                    : "e.g. Ubuntu-22.04"
                }
              >
                <div className="space-y-1">
                  <input
                    list="wsl-distros-adv"
                    className={inputCls}
                    value={s.distro}
                    onChange={(e) => setS({ ...s, distro: e.target.value })}
                  />
                  <datalist id="wsl-distros-adv">
                    {distros.map((d) => (
                      <option key={d} value={d} />
                    ))}
                  </datalist>
                  {distros.length > 0 && !distros.includes(s.distro) && (
                    <div className="text-[11px] text-amber-400">
                      ⚠️ "{s.distro}" was not found in installed WSL distributions.
                    </div>
                  )}
                </div>
              </Field>

              <Field
                label="Custom Model Cache Path (HF_HOME)"
                hint="Store models on another Windows drive (e.g. /mnt/d/ai-models/hf) to save C: drive space."
              >
                <input
                  className={inputCls}
                  placeholder="Default: ~/.cache/huggingface (or /mnt/d/hf)"
                  value={adv.hf_home ?? ""}
                  onChange={(e) => {
                    const v = e.target.value.trim();
                    updateAdv({ hf_home: v === "" ? null : v });
                  }}
                />
              </Field>

              <Field label="LLM Working Dir (inside WSL)" hint="Base runtime dir where pid and log files reside.">
                <input
                  className={inputCls}
                  value={s.llm_dir}
                  onChange={(e) => setS({ ...s, llm_dir: e.target.value })}
                />
              </Field>

              <Field label="Python Virtual Environment" hint="Path to the provisioned vLLM venv inside WSL.">
                <input
                  className={inputCls}
                  value={s.venv_dir}
                  onChange={(e) => setS({ ...s, venv_dir: e.target.value })}
                />
              </Field>

              <div className="sm:col-span-2">
                <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
                  <input
                    type="checkbox"
                    checked={adv.hf_offline}
                    onChange={(e) => updateAdv({ hf_offline: e.target.checked })}
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-slate-200">
                      Hugging Face Offline Mode (<code>HF_HUB_OFFLINE=1</code>)
                    </div>
                    <div className="text-xs text-slate-400">
                      Prevents vLLM from attempting network requests to Hugging Face on startup; strictly requires cached model weights.
                    </div>
                  </div>
                </label>
              </div>

              <div className="sm:col-span-2">
                <Field
                  label="Hugging Face Token"
                  hint="Passed as HF_TOKEN to model downloads and vLLM server launches."
                >
                  <input
                    className={inputCls}
                    type="password"
                    placeholder="hf_…"
                    value={s.hf_token}
                    onChange={(e) => setS({ ...s, hf_token: e.target.value })}
                  />
                </Field>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Storage Settings</Button>
            </div>
          </Card>

          {/* Section 2: vLLM Engine & Optimization Flags */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              vLLM Engine & Optimization
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="KV Cache Precision (kv_cache_dtype)"
                hint="FP8 KV-cache halves memory usage for context tokens, enabling 2x larger context windows."
              >
                <select
                  className={inputCls}
                  value={adv.kv_cache_dtype}
                  onChange={(e) => updateAdv({ kv_cache_dtype: e.target.value })}
                >
                  <option value="auto">auto (Default 16-bit float / FP16/BF16)</option>
                  <option value="fp8">fp8 (Standard 8-bit Float)</option>
                  <option value="fp8_e5m2">fp8_e5m2 (Higher Dynamic Range)</option>
                  <option value="fp8_e4m3">fp8_e4m3 (Higher Precision / Recommended for Ada/Hopper)</option>
                </select>
              </Field>

              <Field
                label="Max Concurrency (max_num_seqs)"
                hint="Maximum concurrent request sequences processed in a batch (leave blank for vLLM default 256)."
              >
                <input
                  className={inputCls}
                  type="number"
                  placeholder="vLLM default (256)"
                  min="1"
                  max="1024"
                  value={adv.max_num_seqs ?? ""}
                  onChange={(e) => {
                    const v = e.target.value.trim();
                    updateAdv({ max_num_seqs: v === "" ? null : Number(v) });
                  }}
                />
              </Field>

              <div className="sm:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
                <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
                  <input
                    type="checkbox"
                    checked={adv.enable_prefix_caching}
                    onChange={(e) => updateAdv({ enable_prefix_caching: e.target.checked })}
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-slate-200">
                      Enable Prefix Caching
                    </div>
                    <div className="text-xs text-slate-400">
                      Reuses KV-cache for repeated prompt prefixes (e.g. system prompts, multi-turn chat history). Dramatically accelerates time-to-first-token.
                    </div>
                  </div>
                </label>

                <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
                  <input
                    type="checkbox"
                    checked={adv.enable_chunked_prefill}
                    onChange={(e) => updateAdv({ enable_chunked_prefill: e.target.checked })}
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-slate-200">
                      Enable Chunked Prefill
                    </div>
                    <div className="text-xs text-slate-400">
                      Chunks long user prompts so decoding requests aren't blocked, reducing generation latency spikes under load.
                    </div>
                  </div>
                </label>

                <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors sm:col-span-2">
                  <input
                    type="checkbox"
                    checked={adv.disable_custom_all_reduce}
                    onChange={(e) => updateAdv({ disable_custom_all_reduce: e.target.checked })}
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-slate-200">
                      Disable Custom All-Reduce (<code>--disable-custom-all-reduce</code>)
                    </div>
                    <div className="text-xs text-slate-400">
                      Recommended when running tensor parallelism on multi-GPU consumer GeForce setups or WSL2 where hardware NVLink peer-to-peer memory access is unavailable.
                    </div>
                  </div>
                </label>
              </div>

              <div className="sm:col-span-2">
                <Field
                  label="Extra vLLM Launch CLI Arguments"
                  hint="Arbitrary CLI flags appended directly to python -m vllm.entrypoints.openai.api_server."
                >
                  <input
                    className={inputCls}
                    placeholder="e.g. --tensor-parallel-size 2 --pipeline-parallel-size 1"
                    value={adv.extra_vllm_args ?? ""}
                    onChange={(e) => {
                      const v = e.target.value;
                      updateAdv({ extra_vllm_args: v.trim() === "" ? null : v });
                    }}
                  />
                </Field>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Engine Settings</Button>
            </div>
          </Card>

          {/* Section 3: Networking & API Security */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              Networking & API Access
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="Host Binding"
                hint="127.0.0.1 restricts access to local machine; 0.0.0.0 allows LAN devices / WebUIs."
              >
                <select
                  className={inputCls}
                  value={adv.host}
                  onChange={(e) => updateAdv({ host: e.target.value })}
                >
                  <option value="127.0.0.1">127.0.0.1 (Localhost Only / Secure)</option>
                  <option value="0.0.0.0">0.0.0.0 (Bind All Interfaces / LAN Access)</option>
                </select>
              </Field>

              <Field
                label="API Key Protection (--api-key)"
                hint="Optional bearer token required to query the OpenAI-compatible vLLM endpoints."
              >
                <input
                  className={inputCls}
                  type="password"
                  placeholder="Optional (e.g. sk-my-secret-key)"
                  value={adv.api_key ?? ""}
                  onChange={(e) => {
                    const v = e.target.value.trim();
                    updateAdv({ api_key: v === "" ? null : v });
                  }}
                />
              </Field>
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Network Settings</Button>
            </div>
          </Card>

          {/* Section 4: Hardware & Deep Memory Tuning */}
          <Card>
            <CardTitle right={memSaved ? <Badge color="emerald">saved</Badge> : undefined}>
              Hardware & Deep Memory Tuning
            </CardTitle>

            <div className="space-y-4">
              {/* Telemetry */}
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

              {/* Sliders & Overhead */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field
                  label={`GPU VRAM Utilization (${Math.round(mem.default_gpu_mem_util * 100)}%)`}
                  hint="Target fraction of GPU VRAM allocated for model weights and KV cache (50% – 98%)."
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
                  hint="Reserved VRAM buffer for CUDA context, runtime allocations, and display."
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

              {/* Toggles */}
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

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field
                  label="RAM Safety Reserve (MB)"
                  hint="WSL2 system RAM kept untouched for Linux kernel and background services."
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
                <Button onClick={saveMem}>Save Memory Settings</Button>
              </div>
            </div>
          </Card>

          {/* Section 5: Environment & Logging */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              Environment & Logging
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="vLLM Logging Level"
                hint="Controls verbosity of console and telemetry logs from the vLLM engine."
              >
                <select
                  className={inputCls}
                  value={adv.log_level}
                  onChange={(e) => updateAdv({ log_level: e.target.value })}
                >
                  <option value="INFO">INFO (Standard production logs)</option>
                  <option value="DEBUG">DEBUG (Detailed execution & kernel logging)</option>
                  <option value="WARNING">WARNING (Only warnings & errors)</option>
                  <option value="ERROR">ERROR (Errors only)</option>
                </select>
              </Field>

              <div className="sm:col-span-2">
                <Field
                  label="Custom Environment Variables"
                  hint="Exported to the WSL bash environment before starting vLLM (one KEY=VAL per line, # comments allowed)."
                >
                  <textarea
                    className={`${inputCls} font-mono text-xs h-24`}
                    placeholder={"CUDA_VISIBLE_DEVICES=0\nNCCL_DEBUG=INFO\nTRITON_CACHE_DIR=/tmp/triton"}
                    value={adv.custom_env_vars ?? ""}
                    onChange={(e) => {
                      const v = e.target.value;
                      updateAdv({ custom_env_vars: v.trim() === "" ? null : v });
                    }}
                  />
                </Field>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Environment Settings</Button>
            </div>
          </Card>

          {/* Section 6: Appliance & Startup Behavior */}
          {renderApplianceCard()}

          {/* Section 7: .wslconfig Viewer */}
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
              .wslconfig <span className="text-xs font-normal text-slate-500">(Windows-side WSL2 configuration)</span>
            </CardTitle>
            <div className="text-xs text-slate-500">
              {wsl?.path ?? "no path found"} — recommended:{" "}
              <code>memory=24GB</code> for 7B+ models, <code>nestedVirtualization=true</code>.
            </div>
            <pre className="mt-2 rounded-md bg-black/30 p-3 font-mono text-xs text-slate-300 overflow-x-auto">
              {wsl?.content ?? "(no .wslconfig yet — create one at %USERPROFILE%\\.wslconfig)"}
            </pre>
          </Card>

          {/* Section 7: Measured Stats (Live Runs) */}
          <Card>
            <CardTitle>Measured Model Benchmarks</CardTitle>
            <div className="space-y-2 text-sm">
              {Object.keys(s.measured).length === 0 ? (
                <div className="text-slate-500">No measured data yet — start a server and send requests to measure live tok/s.</div>
              ) : (
                Object.entries(s.measured).map(([model, m]) => (
                  <div
                    key={model}
                    className="flex flex-wrap items-center gap-2 rounded-md border border-edge bg-surface px-3 py-2"
                  >
                    <span className="flex-1 truncate text-slate-200 font-medium">{model}</span>
                    <span className="font-mono text-xs text-cyan-300">
                      {m.tokens_per_sec != null ? `${m.tokens_per_sec.toFixed(1)} tok/s` : "—"}
                    </span>
                    <span className="text-xs text-slate-500">{fmtNum(m.total_generation_tokens)} tokens</span>
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}