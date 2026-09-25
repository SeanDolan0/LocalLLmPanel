import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls } from "../ui";
import type {
  AdvancedSettings,
  GatewayStatus,
  MemorySettings,
  Settings as SettingsT,
  SystemMemoryInfo,
  WslConfigInfo,
} from "../types";

// ---------------------------------------------------------------------------
// Shared Settings Field Components
// ---------------------------------------------------------------------------
function WslDistroField({
  distros,
  value,
  onChange,
  idSuffix,
}: {
  distros: string[];
  value: string;
  onChange: (value: string) => void;
  idSuffix: string;
}) {
  return (
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
          list={`wsl-distros-${idSuffix}`}
          className={inputCls}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <datalist id={`wsl-distros-${idSuffix}`}>
          {distros.map((d) => (
            <option key={d} value={d} />
          ))}
        </datalist>
        {distros.length > 0 && !distros.includes(value) && (
          <div className="text-[11px] text-amber-400">
            ⚠️ "{value}" was not found in installed WSL distributions.
          </div>
        )}
      </div>
    </Field>
  );
}

function HfTokenField({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <Field
      label="Hugging Face Token"
      hint="Passed as HF_TOKEN to model downloads and vLLM server launches."
    >
      <div className="space-y-1.5">
        <input
          className={inputCls}
          type="password"
          placeholder="hf_…"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <div className="flex items-center gap-1.5 text-[11px] text-emerald-400">
          <span>🔒</span>
          <span>Encrypted at rest with Windows DPAPI (CryptProtectData). Never stored in plaintext.</span>
        </div>
      </div>
    </Field>
  );
}

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
  const [showLanModal, setShowLanModal] = useState(false);
  const [importExportMsg, setImportExportMsg] = useState<string | null>(null);
  const [gw, setGw] = useState<GatewayStatus | null>(null);
  const [gatewayCopied, setGatewayCopied] = useState(false);
  const [snippetCopied, setSnippetCopied] = useState<string | null>(null);
  const [gwInstruct, setGwInstruct] = useState<string | null>(null);
  const [gwEmbed, setGwEmbed] = useState<string | null>(null);
  const [llamacppStatus, setLlamacppStatus] = useState<import("../types").LlamacppInstallStatus | null>(null);
  const [githubMsg, setGithubMsg] = useState<string | null>(null);
  const [hfTokenDirty, setHfTokenDirty] = useState(false);
  const [githubTokenDirty, setGithubTokenDirty] = useState(false);
  const [apiKeyDirty, setApiKeyDirty] = useState(false);
  const [customEnvDirty, setCustomEnvDirty] = useState(false);

  // Simple vs. Advanced mode toggle with localStorage persistence
  const [mode, setMode] = useState<"simple" | "advanced">(() => {
    return (localStorage.getItem("llm_panel_settings_mode") as "simple" | "advanced") || "simple";
  });

  const toggleMode = (newMode: "simple" | "advanced") => {
    setMode(newMode);
    localStorage.setItem("llm_panel_settings_mode", newMode);
  };

  const handleExportConfig = async () => {
    try {
      const jsonStr = await api.configExport();
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `localllm-panel-config-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setImportExportMsg("Config exported!");
      setTimeout(() => setImportExportMsg(null), 3000);
    } catch (e) {
      setErr(`Export failed: ${e}`);
    }
  };

  const handleImportConfig = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const updated = await api.configImport(text);
      setS(updated);
      setHfTokenDirty(false);
      setGithubTokenDirty(false);
      setApiKeyDirty(false);
      setCustomEnvDirty(false);
      const refreshedMem = await api.getMemorySettings();
      setMem(refreshedMem);
      setImportExportMsg("Config imported successfully!");
      setTimeout(() => setImportExportMsg(null), 3000);
    } catch (e) {
      setErr(`Import failed: ${e}`);
    } finally {
      e.target.value = "";
    }
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
    api.llamacppStatus().then(setLlamacppStatus).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await api.gatewayStatus();
        if (!cancelled) setGw(status);
      } catch {
        /* gateway status unavailable */
      }
      try {
        const rows = await api.serversList();
        if (!cancelled) {
          const instruct = rows.find(
            (r) => r.def.task === "instruct" && r.status === "running",
          );
          const embed = rows.find(
            (r) => r.def.task === "embed" && r.status === "running",
          );
          setGwInstruct(
            instruct ? (instruct.def.served_model_name ?? instruct.def.model_id) : null,
          );
          setGwEmbed(
            embed ? (embed.def.served_model_name ?? embed.def.model_id) : null,
          );
        }
      } catch {
        /* servers list unavailable */
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const save = async () => {
    if (!s) return;
    setErr(null);
    setSaved(false);
    try {
      const patch: Parameters<typeof api.settingsSet>[0] = {
        distro: s.distro,
        llm_dir: s.llm_dir,
        venv_dir: s.venv_dir,
        llamacpp_dir: s.llamacpp_dir,
        gguf_dir: s.gguf_dir,
        llamacpp_executable: s.llamacpp_executable,
        llamacpp_channels: s.llamacpp_channels,
        default_quant: s.default_quant,
        advanced_settings: s.advanced_settings,
        minimize_to_tray: s.minimize_to_tray,
        auto_restart_crashed: s.auto_restart_crashed,
        launch_at_login: s.launch_at_login,
      };
      if (hfTokenDirty) patch.hf_token = s.hf_token;
      if (githubTokenDirty) patch.github_token = s.github_token;
      if (apiKeyDirty) patch.clear_advanced_api_key = !s.advanced_settings?.api_key;
      if (customEnvDirty) patch.clear_custom_env_vars = !s.advanced_settings?.custom_env_vars;
      const updated = await api.settingsSet(patch);
      setS(updated);
      setHfTokenDirty(false);
      setGithubTokenDirty(false);
      setApiKeyDirty(false);
      setCustomEnvDirty(false);
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
  const adv: AdvancedSettings = s.advanced_settings ?? {
    hf_home: null,
    hf_offline: false,
    host: "127.0.0.1",
    api_key: null,
    gateway_enabled: false,
    gateway_port: 11434,
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

  const renderConfigBackupCard = () => (
    <Card>
      <CardTitle right={importExportMsg ? <Badge color="emerald">{importExportMsg}</Badge> : undefined}>
        Configuration Portability & Backup
      </CardTitle>
      <p className="text-xs text-slate-400 leading-relaxed">
        Export a portable LocalLLM Panel configuration (WSL setup, memory presets, server definitions, and optimization settings). API keys, tokens, and environment values are intentionally omitted; restore keeps the secrets already stored on this machine.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button variant="subtle" onClick={handleExportConfig}>
          <span>⬇️ Export Configuration (JSON)</span>
        </Button>
        <label className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md border border-edge bg-surface-2 text-slate-300 hover:text-white cursor-pointer transition-colors">
          <span>⬆️ Import Configuration</span>
          <input
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImportConfig}
          />
        </label>
      </div>
    </Card>
  );

  return (
    <div className="mx-auto max-w-4xl p-6 space-y-6">
      {/* LAN Access Warning Consent Modal */}
      {showLanModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-xl border border-amber-500/40 bg-surface-1 p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl">⚠️</span>
              <h2 className="text-base font-bold text-amber-300">Enable Local Network (LAN) Access?</h2>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Binding vLLM to <code className="bg-black/40 px-1 py-0.5 rounded text-amber-200">0.0.0.0</code> allows any computer, phone, or device on your local Wi-Fi / subnet to query your GPU model endpoints without Windows credentials.
            </p>
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90 leading-relaxed">
              <strong>Security Recommendation:</strong> Only enable this on a trusted home/office network, and configure an <strong>API Key</strong> to authenticate incoming requests.
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button
                variant="subtle"
                onClick={() => {
                  setShowLanModal(false);
                }}
              >
                Keep Localhost (127.0.0.1)
              </Button>
              <button
                type="button"
                onClick={() => {
                  updateAdv({ host: "0.0.0.0" });
                  setShowLanModal(false);
                }}
                className="rounded-lg bg-amber-600 px-4 py-2 text-xs font-semibold text-white hover:bg-amber-500 transition-colors"
              >
                I Understand the Risks, Enable LAN
              </button>
            </div>
          </div>
        </div>
      )}

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
              <WslDistroField
                distros={distros}
                value={s.distro}
                onChange={(value) => setS({ ...s, distro: value })}
                idSuffix="simple"
              />

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
              <HfTokenField
                value={s.hf_token}
                onChange={(value) => { setS({ ...s, hf_token: value }); setHfTokenDirty(true); }}
              />
              <Field label="GitHub Token (optional)" hint="Used only for api.github.com to raise the unauthenticated rate limit. It is never sent to Hugging Face.">
                <div className="space-y-1.5">
                  <input className={inputCls} type="password" placeholder="ghp_…" value={s.github_token} onChange={(e) => { setS({ ...s, github_token: e.target.value }); setGithubTokenDirty(true); }} />
                  <div className="flex gap-2">
                    <Button variant="ghost" onClick={async () => { setGithubMsg("Testing…"); try { const r = await api.githubAccess(); setGithubMsg(`${r.message} Token used: ${r.token_used ? "yes" : "no"}.`); } catch (e) { setGithubMsg(String(e)); } }}>Test GitHub access</Button>
                    <Button variant="ghost" onClick={async () => { const updated = await api.clearGithubToken(); setS(updated); setGithubTokenDirty(false); setGithubMsg("GitHub token cleared."); }}>Clear GitHub token</Button>
                  </div>
                  {githubMsg && <div className="text-xs text-slate-400">{githubMsg}</div>}
                </div>
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

          {renderConfigBackupCard()}

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
              <WslDistroField
                distros={distros}
                value={s.distro}
                onChange={(value) => setS({ ...s, distro: value })}
                idSuffix="adv"
              />

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

              <Field
                label="llama.cpp Directory (Upstream)"
                hint="Windows folder containing the installed CUDA llama-server build (ggml-org/llama.cpp)."
              >
                <input
                  className={inputCls}
                  value={s.llamacpp_channels?.upstream?.dir ?? s.llamacpp_dir}
                  onChange={(e) => setS({ ...s, llamacpp_channels: { ...s.llamacpp_channels, upstream: { ...s.llamacpp_channels?.upstream, dir: e.target.value } } })}
                />
              </Field>

              <Field
                label="llama.cpp Directory (PrismML)"
                hint="Windows folder for PrismML fork build (PrismML-Eng/llama.cpp@prism). Supports ternary formats (PQ2_0, PTQ1_0)."
              >
                <input
                  className={inputCls}
                  value={s.llamacpp_channels?.prism?.dir ?? ""}
                  onChange={(e) => setS({ ...s, llamacpp_channels: { ...s.llamacpp_channels, prism: { ...s.llamacpp_channels?.prism, dir: e.target.value } } })}
                />
              </Field>

              <Field
                label="GGUF Model Directory"
                hint="Windows folder used by native GGUF downloads and library scanning."
              >
                <input
                  className={inputCls}
                  value={s.gguf_dir}
                  onChange={(e) => setS({ ...s, gguf_dir: e.target.value })}
                />
              </Field>

              <Field
                label="Custom llama-server.exe (Upstream)"
                hint="Optional absolute path for upstream build; leave blank to use the installed build."
              >
                <input
                  className={inputCls}
                  placeholder="C:\\path\\to\\llama-server.exe"
                  value={s.llamacpp_channels?.upstream?.executable ?? s.llamacpp_executable ?? ""}
                  onChange={(e) =>
                    setS({
                      ...s,
                      llamacpp_channels: { ...s.llamacpp_channels, upstream: { ...s.llamacpp_channels?.upstream, executable: e.target.value.trim() || null } },
                    })
                  }
                />
              </Field>

              <Field
                label="Custom llama-server.exe (PrismML)"
                hint="Optional absolute path for PrismML build; leave blank to use the installed build."
              >
                <input
                  className={inputCls}
                  placeholder="C:\\path\\to\\llama-server.exe"
                  value={s.llamacpp_channels?.prism?.executable ?? ""}
                  onChange={(e) =>
                    setS({
                      ...s,
                      llamacpp_channels: { ...s.llamacpp_channels, prism: { ...s.llamacpp_channels?.prism, executable: e.target.value.trim() || null } },
                    })
                  }
                />
              </Field>
              <div className="sm:col-span-2 rounded-lg border border-edge bg-surface/60 p-3 text-xs text-slate-400">
                <div className="flex items-center gap-2 text-sm text-slate-200">
                  <span className={`h-2 w-2 rounded-full ${llamacppStatus?.cuda_available ? "bg-emerald-400" : "bg-amber-400"}`} />
                  Native llama.cpp (Upstream): {llamacppStatus?.installed ? (llamacppStatus.cuda_available ? "CUDA ready" : "installed, CUDA device unavailable") : "not installed"}
                </div>
                {llamacppStatus?.devices?.length ? (
                  <div className="mt-1">Devices: {llamacppStatus.devices.map((d) => `${d.id} (${d.name})`).join(", ")}</div>
                ) : (
                  <div className="mt-1">Install a CUDA build; Vulkan/CPU archives are not selected by this app.</div>
                )}
              </div>
              <div className="sm:col-span-2 rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3 text-xs text-slate-400">
                <div className="flex items-center gap-2 text-sm text-indigo-200">
                  <span className="h-2 w-2 rounded-full bg-indigo-400" />
                  PrismML Channel: {s.llamacpp_channels?.prism?.installed_tag ? "installed" : "not installed"}
                </div>
                <div className="mt-1 text-slate-400">Supports ternary quant formats (PQ2_0, PTQ1_0) and Hadamard activation transform for Bonsai 2 models.</div>
              </div>

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
<HfTokenField
                value={s.hf_token}
                onChange={(value) => { setS({ ...s, hf_token: value }); setHfTokenDirty(true); }}
              />
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
                  onChange={(e) => {
                    if (e.target.value === "0.0.0.0") {
                      setShowLanModal(true);
                    } else {
                      updateAdv({ host: e.target.value });
                    }
                  }}
                >
                  <option value="127.0.0.1">127.0.0.1 (Localhost Only / Secure)</option>
                  <option value="0.0.0.0">0.0.0.0 (Bind All Interfaces / LAN Access)</option>
                </select>
              </Field>

              {adv.host === "0.0.0.0" && (
                <div className="sm:col-span-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300 flex items-center justify-between">
                  <span>⚠️ <strong>LAN Access Active:</strong> Server endpoints bind to <code>0.0.0.0</code> and accept connections from your local subnet.</span>
                  <button
                    type="button"
                    onClick={() => updateAdv({ host: "127.0.0.1" })}
                    className="ml-3 text-[11px] underline text-amber-200 hover:text-white whitespace-nowrap"
                  >
                    Revert to 127.0.0.1
                  </button>
                </div>
              )}

              <Field
                label="API Key Protection (--api-key)"
                hint={
                  s.api_key_configured
                    ? "A key is stored securely. Leave blank to keep it; enter a replacement to rotate it."
                    : "Required for the local gateway. Leave blank to keep an existing key."
                }
              >
                <input
                  className={inputCls}
                  type="password"
                  placeholder="Optional (e.g. sk-my-secret-key)"
                  value={adv.api_key ?? ""}
                  onChange={(e) => {
                    const v = e.target.value.trim();
                    updateAdv({ api_key: v === "" ? null : v });
                    setApiKeyDirty(true);
                  }}
                />
              </Field>
            </div>

            <div className="mt-4 rounded-lg border border-edge bg-surface/60 p-3.5">
              <div
                className="flex cursor-pointer select-none items-start justify-between gap-3"
                onClick={() => updateAdv({ gateway_enabled: !adv.gateway_enabled })}
              >
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                    checked={adv.gateway_enabled}
                    onChange={(e) => updateAdv({ gateway_enabled: e.target.checked })}
                  />
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-slate-200">
                      OpenAI-Compatible Gateway
                    </div>
                    <p className="text-[11px] text-slate-500">
                      One stay-put endpoint that routes chat requests to whichever running vLLM
                      server actually serves the requested model.
                    </p>
                  </div>
                </div>
                {gw && (
                  <span
                    className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                      gw.running && adv.gateway_enabled
                        ? "bg-emerald-500/15 text-emerald-300"
                        : "bg-slate-500/15 text-slate-400"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        gw.running && adv.gateway_enabled ? "bg-emerald-400" : "bg-slate-500"
                      }`}
                    />
                    {gw.running && adv.gateway_enabled ? "listening" : "off"}
                  </span>
                )}
              </div>

              {adv.gateway_enabled && !s.api_key_configured && (
                <div className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-200">
                  The gateway is enabled but has no API key. Set one above before using client routes; unauthenticated /v1 access is denied.
                </div>
              )}
              {adv.gateway_enabled && (
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Field
                    label="Gateway Port"
                    hint="Ollama-style default; point your client here."
                  >
                    <input
                      className={inputCls}
                      type="number"
                      min={1024}
                      max={65535}
                      value={adv.gateway_port}
                      onClick={(e) => e.currentTarget.select()}
                      onChange={(e) =>
                        updateAdv({ gateway_port: Number(e.target.value) || 11434 })
                      }
                    />
                  </Field>
                  <Field
                    label="Base URL for Clients"
                    hint="Use in Cursor, Continue, LibreChat, or any OpenAI-compatible client."
                  >
                    <div className="flex gap-2">
                      <input
                        className={inputCls}
                        readOnly
                        value={`http://127.0.0.1:${adv.gateway_port}/v1`}
                        onFocus={(e) => e.target.select()}
                      />
                      <Button
                        variant="subtle"
                        onClick={async () => {
                          try {
                            await navigator.clipboard.writeText(
                              `http://127.0.0.1:${adv.gateway_port}/v1`,
                            );
                            setGatewayCopied(true);
                            setTimeout(() => setGatewayCopied(false), 2000);
                          } catch {
                            /* clipboard unavailable */
                          }
                        }}
                      >
                        {gatewayCopied ? "Copied" : "Copy"}
                      </Button>
                    </div>
                  </Field>
                </div>
              )}
              {adv.gateway_enabled && (
                <ClientSnippets
                  gatewayPort={adv.gateway_port}
                  instructModel={gwInstruct}
                  embedModel={gwEmbed}
                  copied={snippetCopied}
                  setCopied={setSnippetCopied}
                />
              )}
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
                      setCustomEnvDirty(true);
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

          {/* Section 7: Configuration Portability & Backup */}
          {renderConfigBackupCard()}

          {/* Section 8: .wslconfig Viewer */}
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

function ClientSnippets({
  gatewayPort,
  instructModel,
  embedModel,
  copied,
  setCopied,
}: {
  gatewayPort: number;
  instructModel: string | null;
  embedModel: string | null;
  copied: string | null;
  setCopied: (k: string | null) => void;
}) {
  const base = `http://127.0.0.1:${gatewayPort}/v1`;
  const chat = instructModel ?? "your-instruct-model";
  const embed = embedModel ?? "your-embed-model";

  const snippets: { key: string; label: string; hint: string; text: string }[] = [
    {
      key: "continue",
      label: "Continue config.yaml",
      hint: "Paste into ~/.continue/config.yaml",
      text: `name: Local Assistant\nversion: 1.0.0\nschema: v1\nmodels:\n  - name: Local Instruct\n    provider: openai\n    model: ${chat}\n    apiBase: ${base}\n    apiKey: "REPLACE_WITH_GATEWAY_API_KEY"\n  - name: Local Autocomplete\n    provider: openai\n    model: ${chat}\n    apiBase: ${base}\n    apiKey: "REPLACE_WITH_GATEWAY_API_KEY"\n    capabilities: [autocomplete]\n  - name: Local Embed\n    provider: openai\n    model: ${embed}\n    apiBase: ${base}\n    apiKey: "REPLACE_WITH_GATEWAY_API_KEY"\n    capabilities: [embed]\n`,
    },
    {
      key: "cursor",
      label: "Cursor / Cline",
      hint: "Base URL + model override",
      text: `Base URL: ${base}\nChat model: ${chat}\nAutocomplete model: ${chat}\nEmbeddings model: ${embed}\nAPI key: REPLACE_WITH_GATEWAY_API_KEY\n`,
    },
    {
      key: "curl-chat",
      label: "curl chat",
      hint: "Chat + FIM completions smoke test",
      text: `GATEWAY_API_KEY='REPLACE_WITH_GATEWAY_API_KEY'; curl ${base}/chat/completions -H "Authorization: Bearer $GATEWAY_API_KEY" -H "Content-Type: application/json" -d '{"model":"${chat}","messages":[{"role":"user","content":"hi"}]}'\n`,
    },
    {
      key: "curl-embed",
      label: "curl embeddings",
      hint: "Embeddings smoke test",
      text: `GATEWAY_API_KEY='REPLACE_WITH_GATEWAY_API_KEY'; curl ${base}/embeddings -H "Authorization: Bearer $GATEWAY_API_KEY" -H "Content-Type: application/json" -d '{"model":"${embed}","input":"hello world"}'\n`,
    },
  ];

  const copy = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
      setTimeout(() => setCopied(null), 2000);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="mt-3 rounded-lg border border-edge bg-surface/40 p-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        One-click client configs
      </div>
      {instructModel === null && embedModel === null && (
        <div className="mt-1.5 text-[11px] text-amber-300">
          No running servers — start an instruct / embed server first; snippets use placeholders.
        </div>
      )}
      <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {snippets.map((s) => (
          <div key={s.key} className="rounded-md border border-edge/60 bg-surface-2/60 p-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-slate-200">{s.label}</span>
              <Button variant="subtle" onClick={() => copy(s.key, s.text)}>
                {copied === s.key ? "Copied" : "Copy"}
              </Button>
            </div>
            <div className="mt-0.5 text-[11px] text-slate-500">{s.hint}</div>
            <pre className="mt-1.5 max-h-24 overflow-y-auto whitespace-pre-wrap rounded bg-black/30 p-2 font-mono text-[10px] leading-relaxed text-slate-400">
              {s.text}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}