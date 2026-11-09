import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Collapsible, Field, HelpBubble, Spinner, inputCls } from "../ui";
import UpdateModal from "../components/UpdateModal";
import type {
  AdvancedSettings,
  AppUpdateInfo,
  GatewayStatus,
  MemorySettings,
  Settings as SettingsT,
  SystemMemoryInfo,
  WslConfigInfo,
} from "../types";

const ADV_TABS = [
  { id: "general", label: "General" },
  { id: "storage", label: "Storage" },
  { id: "engine", label: "Engine" },
  { id: "network", label: "Network" },
  { id: "memory", label: "Memory" },
  { id: "environment", label: "Environment" },
] as const;

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
      help="The WSL2 distro that runs the vLLM servers. It does not change GPU speed or RAM usage by itself — pick the Ubuntu distro you provisioned; switching after provisioning may require a re-provision."
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
      help="Lets the panel download gated models and raises Hugging Face rate limits for pulls. No effect on model speed or RAM — it is stored encrypted with Windows DPAPI and never sent anywhere except Hugging Face."
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
  const [llamacppUpdates, setLlamacppUpdates] = useState<import("../types").LlamaCppUpdateStatus[] | null>(null);
  const [checkingLlamacppUpdates, setCheckingLlamacppUpdates] = useState(false);
  const [installingChannel, setInstallingChannel] = useState<string | null>(null);
  const [llamacppMsg, setLlamacppMsg] = useState<string | null>(null);
  const [forkMsg, setForkMsg] = useState<string | null>(null);
  const [showAddFork, setShowAddFork] = useState(false);
  const [editingForkId, setEditingForkId] = useState<string | null>(null);
  const [forkForm, setForkForm] = useState({
    kind: "github",
    id: "",
    name: "",
    repo: "",
    local_executable: "",
    description: "",
    asset_regex: "",
    runtime_regex: "",
    main_prefix: "",
  });
  const [githubMsg, setGithubMsg] = useState<string | null>(null);
  const [hfTokenDirty, setHfTokenDirty] = useState(false);
  const [githubTokenDirty, setGithubTokenDirty] = useState(false);
  const [apiKeyDirty, setApiKeyDirty] = useState(false);
  const [apiKeyCopied, setApiKeyCopied] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [customEnvDirty, setCustomEnvDirty] = useState(false);
  const [updateInfo, setUpdateInfo] = useState<AppUpdateInfo | null>(null);
  const [checkingUpdates, setCheckingUpdates] = useState(false);
  const [showUpdateModal, setShowUpdateModal] = useState(false);

  const generateGatewayApiKey = () => {
    const bytes = new Uint8Array(24);
    crypto.getRandomValues(bytes);
    const randomHex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
    const key = `sk-${randomHex}`;
    updateAdv({ api_key: key });
    setApiKeyDirty(true);
    setShowApiKey(true);
    return key;
  };

  const copyApiKey = async (customKey?: string) => {
    const key = customKey ?? adv.api_key;
    if (!key) return;
    try {
      await navigator.clipboard.writeText(key);
      setApiKeyCopied(true);
      setTimeout(() => setApiKeyCopied(false), 2000);
    } catch {
      /* clipboard unavailable */
    }
  };

  // Simple vs. Advanced mode toggle with localStorage persistence
  const [mode, setMode] = useState<"simple" | "advanced">(() => {
    return (localStorage.getItem("llm_panel_settings_mode") as "simple" | "advanced") || "simple";
  });

  const toggleMode = (newMode: "simple" | "advanced") => {
    setMode(newMode);
    localStorage.setItem("llm_panel_settings_mode", newMode);
  };

  // Advanced-mode tab with localStorage persistence
  const [advTab, setAdvTab] = useState<string>(
    () => localStorage.getItem("llm_panel_settings_adv_tab") || "general"
  );
  const selectAdvTab = (tab: string) => {
    setAdvTab(tab);
    localStorage.setItem("llm_panel_settings_adv_tab", tab);
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

  const checkLlamacppUpdates = async () => {
    setCheckingLlamacppUpdates(true);
    setLlamacppMsg(null);
    try {
      setLlamacppUpdates(await api.llamacppCheckUpdates());
    } catch (e) {
      setLlamacppMsg(`Update check failed: ${e}`);
    } finally {
      setCheckingLlamacppUpdates(false);
    }
  };

  const installLlamacppChannel = async (channel: string) => {
    const label = s?.llamacpp_forks?.find((fork) => fork.id === channel)?.name ?? channel;
    setInstallingChannel(channel);
    setLlamacppMsg(null);
    try {
      await api.installLlamacppChannel(channel);
      const fresh = await api.llamacppStatus().catch(() => null);
      if (fresh) setLlamacppStatus(fresh);
      // The install persisted new channel config to disk; pull just that slice
      // (not the whole settings object) so unsaved edits elsewhere survive.
      const cfg = await api.settingsGet().catch(() => null);
      if (cfg) {
        setS((prev) => (prev ? { ...prev, llamacpp_channels: cfg.llamacpp_channels, llamacpp_forks: cfg.llamacpp_forks } : prev));
      }
      setLlamacppUpdates(null);
      setLlamacppMsg(`${label} installed/updated.`);
      setTimeout(() => setLlamacppMsg(null), 5000);
    } catch (e) {
      setLlamacppMsg(`${label} install failed: ${e}`);
    } finally {
      setInstallingChannel(null);
    }
  };

  const normalizeForkRepo = (raw: string): string => {
    let repo = raw.trim().replace(/\/+$/, "");
    repo = repo.replace(/^https?:\/\/github\.com\//i, "");
    repo = repo.replace(/\.git$/i, "");
    const parts = repo.split("/").filter(Boolean);
    return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : repo;
  };

  const resetForkForm = () => {
    setForkForm({ kind: "github", id: "", name: "", repo: "", local_executable: "", description: "", asset_regex: "", runtime_regex: "", main_prefix: "" });
    setEditingForkId(null);
    setShowAddFork(false);
  };

  const isLocalForkForm = forkForm.kind === "local";

  const submitFork = async () => {
    setForkMsg(null);
    // Built-in-style defaults: blank regex fields fall back to the upstream naming convention.
    const DEFAULT_MAIN = "^llama-.*-bin-win-cuda-(\\d+)\\.(\\d+)-x64\\.zip$";
    const DEFAULT_RUNTIME = "^cudart-llama-bin-win-cuda-(\\d+)\\.(\\d+)-x64\\.zip$";
    const local = forkForm.kind === "local";
    const fork = local
      ? {
        id: editingForkId ?? forkForm.id.trim().toLowerCase(),
        name: forkForm.name.trim(),
        kind: "local",
        repo: "",
        local_executable: forkForm.local_executable.trim(),
        description: forkForm.description.trim(),
        asset_regex: "",
        runtime_regex: "",
        main_prefix: "",
        builtin: false,
      }
      : {
        id: editingForkId ?? forkForm.id.trim().toLowerCase(),
        name: forkForm.name.trim(),
        kind: "github",
        repo: normalizeForkRepo(forkForm.repo),
        local_executable: null,
        description: forkForm.description.trim(),
        asset_regex: forkForm.asset_regex.trim() || DEFAULT_MAIN,
        runtime_regex: forkForm.runtime_regex.trim() || DEFAULT_RUNTIME,
        main_prefix: forkForm.main_prefix.trim() || "llama-",
        builtin: false,
      };
    try {
      const forks = await api.llamacppForkUpsert(fork);
      setS((prev) => (prev ? { ...prev, llamacpp_forks: forks } : prev));
      setLlamacppUpdates(null);
      setForkMsg(editingForkId ? `Fork "${fork.name}" updated.` : local
        ? `Local fork "${fork.name}" added. Press Verify to probe the built binary, then pick it in a server's llama.cpp Fork dropdown.`
        : `Fork "${fork.name}" added. Use it in a server's llama.cpp Fork dropdown or install it below.`);
      resetForkForm();
    } catch (e) {
      setForkMsg(String(e));
    }
  };

  const editFork = (fork: import("../types").LlamaCppFork) => {
    setEditingForkId(fork.id);
    setForkForm({
      kind: fork.kind ?? "github",
      id: fork.id,
      name: fork.name,
      repo: fork.repo,
      local_executable: fork.local_executable ?? "",
      description: fork.description ?? "",
      asset_regex: fork.asset_regex,
      runtime_regex: fork.runtime_regex,
      main_prefix: fork.main_prefix,
    });
    setShowAddFork(true);
    setForkMsg(null);
  };

  const removeFork = async (id: string) => {
    setForkMsg(null);
    try {
      const forks = await api.llamacppForkRemove(id);
      setS((prev) => (prev ? { ...prev, llamacpp_forks: forks } : prev));
      setLlamacppUpdates(null);
      setForkMsg(`Fork "${id}" removed.`);
    } catch (e) {
      setForkMsg(String(e));
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
    api.getSystemMemory().then(setSysMem).catch((e) => setErr(`System memory unavailable: ${e}`));
    api.llamacppStatus().then(setLlamacppStatus).catch((e) => setErr(`llama.cpp status unavailable: ${e}`));
    api
      .checkAppUpdate(false)
      .then((info) => setUpdateInfo(info))
      .catch((e) => setErr(`Update check failed: ${e}`));
  }, []);

  const updatesById = new Map((llamacppUpdates ?? []).map((u) => [u.channel, u]));

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await api.gatewayStatus();
        if (!cancelled) setGw(status);
      } catch (e) {
        if (!cancelled) setErr(`Gateway status unavailable: ${e}`);
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
      } catch (e) {
        if (!cancelled) setErr(`Server status unavailable: ${e}`);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const handleCheckUpdates = async () => {
    setCheckingUpdates(true);
    setErr(null);
    try {
      const info = await api.checkAppUpdate(true);
      setUpdateInfo(info);
      if (info.available) {
        setShowUpdateModal(true);
      }
    } catch (e) {
      setErr(`Update check failed: ${e}`);
    } finally {
      setCheckingUpdates(false);
    }
  };

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
        auto_shutdown_wsl: s.auto_shutdown_wsl,
        auto_check_updates: s.auto_check_updates,
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
            <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              Minimize to System Tray
              <HelpBubble text="Closing the window hides the panel into the tray instead of stopping servers. No effect on model speed or RAM." />
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
            <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              Launch at Windows Login
              <HelpBubble text="Starts the panel automatically when you sign into Windows (via the HKCU Run registry). No effect on model speed or RAM." />
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
            <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              Auto-Restart Crashed Servers
              <HelpBubble text="Restarts a server with backoff if its backend process crashes. No effect on model speed or RAM while running." />
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Monitors background server processes and restarts them with backoff if they crash.
            </div>
          </div>
        </label>
        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors sm:col-span-2">
          <input
            type="checkbox"
            checked={s.auto_shutdown_wsl !== false}
            onChange={(e) => setS({ ...s, auto_shutdown_wsl: e.target.checked })}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">Auto-shutdown WSL when idle
              <HelpBubble text="Terminates WSL when nothing has used it for a minute, freeing system RAM. Next server start pays a short WSL cold-boot time." />
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Terminates the WSL distro when no servers, pulls, or provisioning are active and nothing has used WSL for at least 60 seconds. Keeps WSL shut down after closing/idle periods to free memory.
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

  const renderAppUpdatesCard = () => (
    <Card>
      <CardTitle
        right={
          updateInfo?.available ? (
            <Badge color="cyan">Update {updateInfo.latest_version} available</Badge>
          ) : updateInfo && !updateInfo.error ? (
            <Badge color="emerald">Up to date ({updateInfo.current_version})</Badge>
          ) : undefined
        }
      >
        Application Updates
      </CardTitle>
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-lg border border-edge bg-surface/40 p-4">
          <div>
            <div className="text-sm font-medium text-slate-200 flex items-center gap-2">
              Installed Version:{" "}
              <span className="font-mono text-indigo-300 font-semibold">
                {updateInfo?.current_version || "..."}
              </span>
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Automatically checks GitHub Releases (SeanDolan0/LocalLLmPanel) for updates.
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="ghost"
              onClick={handleCheckUpdates}
              disabled={checkingUpdates}
              className="text-xs"
            >
              {checkingUpdates ? <Spinner label="Checking..." /> : "Check for Updates"}
            </Button>
            {updateInfo?.available && (
              <Button
                variant="primary"
                onClick={() => setShowUpdateModal(true)}
                className="text-xs"
              >
                View & Install {updateInfo.latest_version}
              </Button>
            )}
          </div>
        </div>

        <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
          <input
            type="checkbox"
            checked={s?.auto_check_updates !== false}
            onChange={(e) => {
              if (!s) return;
              const updated = { ...s, auto_check_updates: e.target.checked };
              setS(updated);
              api.settingsSet({ auto_check_updates: e.target.checked }).catch(() => {});
            }}
            className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
          />
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              Automatically check for updates on startup
              <HelpBubble text="Quietly checks GitHub Releases on launch and alerts you when a panel update is ready. No effect on model speed or RAM." />
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              When enabled, Local LLM Panel quietly checks GitHub Releases on launch and alerts you in the sidebar when a new update is ready.
            </div>
          </div>
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
            {!(s.api_key_configured || adv.api_key?.trim()) ? (
              <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-300 leading-relaxed">
                <strong>API Key Required:</strong> An API key must be configured and saved before enabling 0.0.0.0 LAN binding. Starting an externally-bound server without an API key is refused for security.
              </div>
            ) : (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90 leading-relaxed">
                <strong>Security Protection:</strong> Incoming requests must authenticate with your configured API key. Only enable on trusted networks.
              </div>
            )}
            <div className="rounded-lg border border-sky-500/30 bg-sky-500/10 p-3 text-xs text-sky-200/90 leading-relaxed">
              <strong>WSL2 Network Reachability Note:</strong> On default WSL2 NAT, binding <code>0.0.0.0</code> inside WSL does not automatically make the service LAN-reachable from other machines on the physical LAN. WSL2 mirrored mode (<code>networkingMode=mirrored</code> in <code>.wslconfig</code>) or a Windows port proxy (<code>netsh interface portproxy</code>) is required. Note that native llama.cpp binds to <code>127.0.0.1</code> only.
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
                disabled={!(s.api_key_configured || adv.api_key?.trim())}
                onClick={() => {
                  updateAdv({ host: "0.0.0.0" });
                  setShowLanModal(false);
                }}
                className={`rounded-lg px-4 py-2 text-xs font-semibold text-white transition-colors ${
                  s.api_key_configured || adv.api_key?.trim()
                    ? "bg-amber-600 hover:bg-amber-500 cursor-pointer"
                    : "bg-slate-700 text-slate-400 cursor-not-allowed opacity-50"
                }`}
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
                help="Which checkpoint precision Search deploys by default. FP16 is highest quality but uses the most VRAM; FP8 / AWQ / GPTQ use far less VRAM and system RAM so larger models or longer contexts fit, at some quality cost."
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
              <Collapsible
                title="GitHub token (optional)"
                subtitle="Only needed if GitHub search hits anonymous rate limits"
                className="mt-3"
              >
                <Field label="GitHub Token (optional)" help="Raises the api.github.com rate limit used for llama.cpp release checks and downloads. No effect on model speed or RAM — the token is sent only to GitHub." hint="Used only for api.github.com to raise the unauthenticated rate limit. It is never sent to Hugging Face.">
                  <div className="space-y-1.5">
                    <input className={inputCls} type="password" placeholder="ghp_…" value={s.github_token} onChange={(e) => { setS({ ...s, github_token: e.target.value }); setGithubTokenDirty(true); }} />
                    <div className="flex gap-2">
                      <Button variant="ghost" onClick={async () => { setGithubMsg("Testing…"); try { const r = await api.githubAccess(); setGithubMsg(`${r.message} Token used: ${r.token_used ? "yes" : "no"}.`); } catch (e) { setGithubMsg(String(e)); } }}>Test GitHub access</Button>
                      <Button variant="ghost" onClick={async () => { const updated = await api.clearGithubToken(); setS(updated); setGithubTokenDirty(false); setGithubMsg("GitHub token cleared."); }}>Clear GitHub token</Button>
                    </div>
                    {githubMsg && <div className="text-xs text-slate-400">{githubMsg}</div>}
                  </div>
                </Field>
              </Collapsible>
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
                help="Default share of GPU VRAM new vLLM servers may reserve for weights + KV cache. Higher fits longer contexts and more concurrency but leaves less headroom (out-of-memory risk); the overhead buffer is still reserved on top."
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
                  <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                    Allow RAM Context Overflow (RAM Swap)
                    <HelpBubble text="Lets the context spill into WSL2 system RAM when VRAM is full (--swap-space). Prevents out-of-memory crashes on long prompts but is much slower than GPU KV cache." />
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

          {renderAppUpdatesCard()}

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
        <div className="space-y-5">
          {/* Section tabs */}
          <div className="flex flex-wrap items-center gap-1 rounded-lg border border-edge bg-surface-2 p-1 w-fit">
            {ADV_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => selectAdvTab(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                  advTab === tab.id
                    ? "bg-indigo-600 text-white shadow-sm font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {advTab === "general" && (
            <div className="space-y-6">
              {renderAppUpdatesCard()}
              {renderApplianceCard()}
              {renderConfigBackupCard()}
            </div>
          )}
          {advTab === "storage" && (
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
                help="Where Hugging Face weights are cached. No effect on model speed or RAM — point it at a large, fast drive if C: is tight."
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

              <Field label="LLM Working Dir (inside WSL)" help="Where pid and log files live inside WSL. No effect on model speed or RAM." hint="Base runtime dir where pid and log files reside.">
                <input
                  className={inputCls}
                  value={s.llm_dir}
                  onChange={(e) => setS({ ...s, llm_dir: e.target.value })}
                />
              </Field>

              <Field label="Python Virtual Environment" help="The provisioned vLLM Python environment inside WSL. No effect on model speed or RAM — only change it if you intentionally provisioned elsewhere." hint="Path to the provisioned vLLM venv inside WSL.">
                <input
                  className={inputCls}
                  value={s.venv_dir}
                  onChange={(e) => setS({ ...s, venv_dir: e.target.value })}
                />
              </Field>

              <Field
                label="GGUF Model Directory"
                help="Where native GGUF model files are stored and scanned from. No effect on inference speed or RAM — use a large drive for big models."
                hint="Windows folder used by native GGUF downloads and library scanning."
              >
                <input
                  className={inputCls}
                  value={s.gguf_dir}
                  onChange={(e) => setS({ ...s, gguf_dir: e.target.value })}
                />
              </Field>
              <div className="sm:col-span-2 flex items-center justify-between gap-2 pt-1">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">llama.cpp Forks</div>
                <Button variant="ghost" className="text-[11px]" onClick={checkLlamacppUpdates} disabled={checkingLlamacppUpdates}>
                  {checkingLlamacppUpdates ? <Spinner label="checking." /> : "Check for updates"}
                </Button>
              </div>

              {(s.llamacpp_forks ?? []).map((fork) => {
                const ch = s.llamacpp_channels?.[fork.id];
                const update = updatesById.get(fork.id);
                const tag = ch?.installed_tag ?? null;
                const isLocal = (fork.kind ?? "github") === "local";
                return (
                  <div key={fork.id} className="sm:col-span-2 rounded-lg border border-edge bg-surface/60 p-3 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-sm text-slate-200 min-w-0">
                        <span className={`h-2 w-2 rounded-full shrink-0 ${tag ? "bg-emerald-400" : "bg-slate-600"}`} />
                        <span className="font-medium truncate">{fork.name}</span>
                        {fork.builtin ? (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-500/15 text-slate-400 border border-slate-600/40">built-in</span>
                        ) : (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">custom</span>
                        )}
                        {isLocal ? (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">local build</span>
                        ) : (
                          <a
                            className="text-[11px] text-indigo-400 hover:underline shrink-0"
                            href={`https://github.com/${fork.repo}/releases`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {fork.repo} ↗
                          </a>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {fork.builtin !== true && (
                          <>
                            <Button variant="ghost" className="text-[11px]" onClick={() => editFork(fork)}>Edit</Button>
                            <Button variant="ghost" className="text-[11px] text-rose-300" onClick={() => removeFork(fork.id)}>Remove</Button>
                          </>
                        )}
                        <Button variant="subtle" className="text-[11px]" onClick={() => installLlamacppChannel(fork.id)} disabled={installingChannel !== null}>
                          {installingChannel === fork.id ? <Spinner label="installing." /> : isLocal ? "Verify / Refresh" : update?.update_available && update.latest_tag && tag ? `Update to ${update.latest_tag}` : "Install / Reinstall"}
                        </Button>
                      </div>
                    </div>
                    {fork.description ? <div className="text-[11px] text-slate-500">{fork.description}</div> : null}
                    {isLocal ? (
                      <div className="text-[11px] text-slate-400">
                        Points at your build: <code className="text-amber-200/90 break-all">{fork.local_executable || "(no path set)"}</code>
                        {" "}— rebuilt binary is picked up by pressing Verify / Refresh (or restarting the server); no GitHub download is needed.
                      </div>
                    ) : (
                    <div className="grid gap-2 sm:grid-cols-2">
                      <Field label="Install directory" help="Where this fork's downloaded CUDA build is extracted. No effect on inference speed or RAM — keep it on a fast SSD." hint="Windows folder for this fork's CUDA build(s).">
                        <input
                          className={inputCls}
                          value={ch?.dir ?? s.llamacpp_dir}
                          onChange={(e) => setS({ ...s, llamacpp_channels: { ...s.llamacpp_channels, [fork.id]: { ...s.llamacpp_channels?.[fork.id], dir: e.target.value } } })}
                        />
                      </Field>
                      <Field label="Custom llama-server.exe" help="Optional binary override for this fork (e.g. a manually downloaded build). Inference speed and RAM come from that binary's flags, not from this path — blank uses the installed build." hint="Optional absolute path override; leave blank to use the installed build.">
                        <input
                          className={inputCls}
                          placeholder="C:\\path\\to\\llama-server.exe"
                          value={ch?.executable ?? ""}
                          onChange={(e) =>
                            setS({
                              ...s,
                              llamacpp_channels: { ...s.llamacpp_channels, [fork.id]: { ...s.llamacpp_channels?.[fork.id], executable: e.target.value.trim() || null } },
                            })
                          }
                        />
                      </Field>
                    </div>
                    )}
                    <div className="text-[11px] text-slate-400">
                      {tag && tag !== "local" ? `installed (tag ${tag})` : tag ? `verified local build${ch?.version ? ` · ${ch.version.split("\n")[0]}` : ""}` : "not verified yet — press Verify / Refresh after building"}
                      {fork.id === "upstream" && llamacppStatus ? ` · ${llamacppStatus.cuda_available ? "CUDA ready" : llamacppStatus.installed ? "CUDA device unavailable" : "not installed"}` : ""}
                      {fork.id === "upstream" && llamacppStatus?.devices?.length ? ` · Devices: ${llamacppStatus.devices.map((d) => `${d.id} (${d.name})`).join(", ")}` : ""}
                    </div>
                    {update ? (
                      <div className="text-[11px]">
                        {update.latest_tag ? (
                          update.update_available && tag ? (
                            <span className="text-amber-300">Update available: {tag} → {update.latest_tag}{update.cuda_version ? ` (CUDA ${update.cuda_version})` : ""}</span>
                          ) : (
                            <span className="text-emerald-300">Up to date (latest {update.latest_tag})</span>
                          )
                        ) : (
                          <span className="text-rose-300">{update.warning ?? "Update check failed"}</span>
                        )}
                        {update.warning && update.latest_tag ? <div className="text-amber-300/80 mt-0.5">{update.warning}</div> : null}
                      </div>
                    ) : null}
                  </div>
                );
              })}
              <div className="sm:col-span-2">
                {!showAddFork ? (
                  <Button variant="subtle" className="text-[11px]" onClick={() => { resetForkForm(); setShowAddFork(true); setForkMsg(null); }}>
                    + Add custom fork…
                  </Button>
                ) : (
                  <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-xs font-semibold text-indigo-200">{editingForkId ? `Edit fork "${editingForkId}"` : "Add a custom llama.cpp fork"}</div>
                      <Button variant="ghost" className="text-[11px]" onClick={resetForkForm}>Cancel</Button>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[11px] text-slate-400">Source:</span>
                      <button
                        type="button"
                        onClick={() => setForkForm({ ...forkForm, kind: "github" })}
                        className={`px-2.5 py-1 text-[11px] rounded-md border transition-colors ${!isLocalForkForm ? "bg-indigo-500/20 text-indigo-200 border-indigo-500/40 font-semibold" : "bg-surface-2 text-slate-400 border-edge hover:text-slate-200"}`}
                      >
                        GitHub releases
                      </button>
                      <button
                        type="button"
                        onClick={() => setForkForm({ ...forkForm, kind: "local" })}
                        className={`px-2.5 py-1 text-[11px] rounded-md border transition-colors ${isLocalForkForm ? "bg-amber-500/20 text-amber-200 border-amber-500/40 font-semibold" : "bg-surface-2 text-slate-400 border-edge hover:text-slate-200"}`}
                      >
                        Local build (no GitHub)
                      </button>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2">
                      <Field label="Fork id" help="Stable slug used in server definitions and install folders. No effect on model speed or RAM." hint="Lowercase slug used in servers/installs, e.g. xhtoken">
                        <input className={inputCls} placeholder="xhtoken" value={forkForm.id} disabled={editingForkId !== null} onChange={(e) => setForkForm({ ...forkForm, id: e.target.value })} />
                      </Field>
                      <Field label="Display name" help="Just a readable name shown in dropdowns. No effect on model speed or RAM.">
                        <input className={inputCls} placeholder="XHToken (Spark)" value={forkForm.name} onChange={(e) => setForkForm({ ...forkForm, name: e.target.value })} />
                      </Field>
                      {isLocalForkForm ? (
                        <Field label="Built llama-server.exe" help="Absolute path to the llama-server.exe you built yourself. The panel runs this binary in place — model speed and RAM use come from how you built it, not from a download." hint="Absolute path to the binary you built yourself, e.g. C:\llama.cpp\build\bin\llama-server.exe">
                          <input className={inputCls} placeholder="C:\llama.cpp\build\bin\llama-server.exe" value={forkForm.local_executable} onChange={(e) => setForkForm({ ...forkForm, local_executable: e.target.value })} />
                        </Field>
                      ) : (
                      <>
                        <Field label="GitHub repo" help="The fork's GitHub repo used to find release downloads. No effect on model speed or RAM." hint="'owner/name' or a github.com URL">
                          <input className={inputCls} placeholder="XHToken/llama.cpp" value={forkForm.repo} onChange={(e) => setForkForm({ ...forkForm, repo: e.target.value })} />
                        </Field>
                        <Field label="Asset prefix" help="Part of the release zip name stripped to derive the installed tag. No effect on model speed or RAM — only affects which download is picked." hint="Prefix stripped from the main asset name to derive the tag">
                          <input className={inputCls} placeholder="llama-" value={forkForm.main_prefix} onChange={(e) => setForkForm({ ...forkForm, main_prefix: e.target.value })} />
                        </Field>
                      </>
                      )}
                    </div>
                    <Field label="Description" help="A note about what this fork adds. No effect on model speed or RAM." hint="Optional">
                      <input className={inputCls} placeholder="What this fork adds (e.g. Spark DSpark speculative decoding)" value={forkForm.description} onChange={(e) => setForkForm({ ...forkForm, description: e.target.value })} />
                    </Field>
                    {!isLocalForkForm && (
                    <Collapsible title="Advanced: asset regexes" compact>
                      <div className="grid gap-2 sm:grid-cols-2 pt-2">
                        <Field label="Main asset regex" help="Matches the fork's Windows x64 CUDA zip name and extracts the CUDA version. No effect on model speed or RAM — a wrong pattern just fails to find a download." hint="Windows x64 CUDA zip; captures CUDA major.minor in groups 1-2">
                          <input className={inputCls} placeholder="^llama-.*-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$" value={forkForm.asset_regex} onChange={(e) => setForkForm({ ...forkForm, asset_regex: e.target.value })} />
                        </Field>
                        <Field label="Runtime (cudart) regex" help="Matches the CUDA runtime (cudart) zip that ships alongside the main build. No effect on model speed or RAM beyond the runtime needed to run the GPU build." hint="Blank keeps the upstream default naming">
                          <input className={inputCls} placeholder="^cudart-llama-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$" value={forkForm.runtime_regex} onChange={(e) => setForkForm({ ...forkForm, runtime_regex: e.target.value })} />
                        </Field>
                      </div>
                    </Collapsible>
                    )}
                    <div className="flex items-center gap-2">
                      <Button variant="subtle" className="text-[11px]" onClick={submitFork}>{editingForkId ? "Save fork" : "Add fork"}</Button>
                    </div>
                    {isLocalForkForm ? (
                      <p className="text-[10px] text-slate-500">
                        Build the fork yourself (e.g. with CMake/MSVC or a local checkout) and point the fork at the
                        resulting <code>llama-server.exe</code>. Nothing is downloaded: Verify probes that binary in
                        place, and servers use it directly. Rebuild + Verify (or a server restart) to pick up a new build.
                      </p>
                    ) : (
                    <p className="text-[10px] text-slate-500">
                      Install/update scans the fork's GitHub releases and picks the newest Windows x64 CUDA build your driver supports. The main asset must match the regex above
                      (default: the upstream naming convention); a matching cudart zip is expanded alongside it. Forks that publish no Windows CUDA zips can't be auto-installed —
                      add them as a Local build fork instead and point it at your own <code>llama-server.exe</code>.
                    </p>
                    )}
                  </div>
                )}
                {forkMsg ? <div className="mt-1 text-[11px] text-slate-300">{forkMsg}</div> : null}
              </div>
              {llamacppMsg && (
                <div className="sm:col-span-2 text-[11px] text-slate-300">{llamacppMsg}</div>
              )}

              <div className="sm:col-span-2">
                <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-edge bg-surface/60 p-3 hover:bg-surface/80 transition-colors">
                  <input
                    type="checkbox"
                    checked={adv.hf_offline}
                    onChange={(e) => updateAdv({ hf_offline: e.target.checked })}
                    className="mt-0.5 h-4 w-4 rounded border-edge bg-surface-2 text-indigo-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Hugging Face Offline Mode (<code>HF_HUB_OFFLINE=1</code>)
                      <HelpBubble text="Forces vLLM to never reach out to Hugging Face at startup. No effect on model speed or RAM — servers fail fast if the model is not already cached locally." />
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
            </div>
          )}
          {advTab === "engine" && (
            <div className="space-y-6">
          {/* Section 2: vLLM Engine & Optimization Flags */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              vLLM Engine & Optimization
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="KV Cache Precision (kv_cache_dtype)"
                help="vLLM KV-cache dtype. FP8 halves the KV-cache VRAM (roughly 2× usable context) at a small quality cost; auto keeps 16-bit (FP16/BF16) for best quality."
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
                help="Caps how many sequences vLLM batches at once. Higher means more throughput under concurrent load but proportionally more KV-cache VRAM (out-of-memory risk); lower is safer but slower."
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Enable Prefix Caching
                      <HelpBubble text="Reuses KV-cache blocks for repeated prompt prefixes (system prompts, multi-turn history). Faster time-to-first-token; costs some VRAM for the cached blocks." />
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Enable Chunked Prefill
                      <HelpBubble text="Splits long prompt prefills so ongoing decodes are not blocked. Smoother latency under load; negligible RAM/VRAM change." />
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Disable Custom All-Reduce (<code>--disable-custom-all-reduce</code>)
                      <HelpBubble text="Disables vLLM's custom CUDA all-reduce. Negligible effect on a single GPU; recommended on multi-GPU consumer cards / WSL2 without NVLink, where the custom path can hang." />
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
                  help="Raw vLLM flags appended at server start. Speed and RAM effects depend entirely on the flags you add — harness request parameters (temperature, max tokens, reasoning, tools) override these defaults."
                  hint="Arbitrary CLI flags appended directly to python -m vllm.entrypoints.openai.api_server. These are server defaults: request parameters from your harness (reasoning_effort, temperature, max_tokens, tools, …) take precedence."
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
          {advTab === "network" && (
            <div className="space-y-6">
          {/* Section 3: Networking & API Security */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              Networking & API Access
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="Host Binding"
                help="127.0.0.1 accepts connections from this machine only; 0.0.0.0 also accepts from your local network. No effect on model speed or RAM — LAN access exposes your endpoints to other devices."
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
                <div className="sm:col-span-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300 space-y-2">
                  <div className="flex items-center justify-between">
                    <span>⚠️ <strong>LAN Access Active:</strong> Server endpoints bind to <code>0.0.0.0</code> (API key required for start).</span>
                    <button
                      type="button"
                      onClick={() => updateAdv({ host: "127.0.0.1" })}
                      className="ml-3 text-[11px] underline text-amber-200 hover:text-white whitespace-nowrap"
                    >
                      Revert to 127.0.0.1
                    </button>
                  </div>
                  <p className="text-[11px] text-amber-200/80 leading-normal">
                    Note: Default WSL2 NAT isolates WSL from your physical LAN. Use WSL2 mirrored networking mode (<code>networkingMode=mirrored</code> in <code>.wslconfig</code>) or <code>netsh interface portproxy</code> to expose WSL ports to external LAN devices. Native llama.cpp binds to 127.0.0.1.
                  </p>
                </div>
              )}

              <Field
                label="API Key Protection (--api-key)"
                help="Key gating server and gateway endpoints. Applies to both vLLM and llama.cpp backends unless overridden per-server. Encrypted at rest with Windows DPAPI (CryptProtectData) and never stored in plaintext."
                hint={
                  s.api_key_configured
                    ? "Stored securely using Windows DPAPI (CryptProtectData). Leave blank to keep it; enter or generate a replacement to rotate it."
                    : "Required for the local gateway and external LAN access (0.0.0.0). Click 'Auto-generate Key' to create a random key."
                }
              >
                <div className="space-y-2">
                  <div className="flex flex-col sm:flex-row gap-2">
                    <div className="relative flex-1">
                      <input
                        className={`${inputCls} pr-10`}
                        type={showApiKey ? "text" : "password"}
                        placeholder={s.api_key_configured && !adv.api_key ? "•••••••••••••••••••••••• (Stored)" : "Optional (e.g. sk-my-secret-key)"}
                        value={adv.api_key ?? ""}
                        onChange={(e) => {
                          const v = e.target.value.trim();
                          updateAdv({ api_key: v === "" ? null : v });
                          setApiKeyDirty(true);
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => setShowApiKey(!showApiKey)}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-200 focus:outline-none"
                        title={showApiKey ? "Hide key" : "Show key"}
                      >
                        {showApiKey ? "🙈" : "👁️"}
                      </button>
                    </div>
                    <Button
                      variant="subtle"
                      className="shrink-0 flex items-center gap-1.5"
                      onClick={() => {
                        const newKey = generateGatewayApiKey();
                        copyApiKey(newKey);
                      }}
                      title="Generate a random secure API key for the gateway and copy to clipboard"
                    >
                      <span>⚡</span>
                      <span>Auto-generate Key</span>
                    </Button>
                    {adv.api_key && (
                      <Button
                        variant="ghost"
                        className="shrink-0"
                        onClick={() => copyApiKey()}
                        title="Copy API key to clipboard"
                      >
                        {apiKeyCopied ? "Copied" : "Copy"}
                      </Button>
                    )}
                  </div>
                  {apiKeyCopied && (
                    <div className="text-[11px] text-emerald-400">
                      ✓ API key copied to clipboard! Remember to click &quot;Save Network Settings&quot; below.
                    </div>
                  )}
                </div>
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      OpenAI-Compatible Gateway
                      <HelpBubble text="One fixed endpoint that routes chat requests to whichever running vLLM server serves the requested model. No effect on model speed or RAM; requires an API key." />
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
                        : gw.port_conflict
                        ? "bg-amber-500/15 text-amber-300"
                        : "bg-slate-500/15 text-slate-400"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        gw.running && adv.gateway_enabled
                          ? "bg-emerald-400"
                          : gw.port_conflict
                          ? "bg-amber-400"
                          : "bg-slate-500"
                      }`}
                    />
                    {gw.running && adv.gateway_enabled ? "listening" : gw.port_conflict ? "port conflict" : "off"}
                  </span>
                )}
              </div>

              {gw?.port_conflict && adv.gateway_enabled && (
                <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">
                  Port {gw.port} is already in use by another application. The gateway cannot start until you change the port or stop the conflicting application.
                </div>
              )}

              {adv.gateway_enabled && !s.api_key_configured && !adv.api_key && (
                <div className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <span>The gateway is enabled but has no API key. Set one above before using client routes; unauthenticated /v1 access is denied.</span>
                  <Button
                    variant="subtle"
                    className="shrink-0 text-xs"
                    onClick={() => {
                      const newKey = generateGatewayApiKey();
                      copyApiKey(newKey);
                    }}
                  >
                    ⚡ Auto-generate Key
                  </Button>
                </div>
              )}
              {adv.gateway_enabled && (
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Field
                    label="Gateway Port"
                    help="Port the routing gateway listens on. No effect on model speed or RAM."
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
                    help="The URL clients (Cursor, Continue, LibreChat, …) call. No effect on model speed or RAM."
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
                  apiKey={adv.api_key}
                  copied={snippetCopied}
                  setCopied={setSnippetCopied}
                />
              )}
            </div>

            <div className="mt-4 flex justify-end">
              <Button onClick={save}>Save Network Settings</Button>
            </div>
          </Card>
            </div>
          )}
          {advTab === "memory" && (
            <div className="space-y-6">
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
                  help="Default share of VRAM new vLLM servers may reserve. Higher fits longer contexts and higher concurrency but leaves less headroom (out-of-memory risk); the overhead buffer is still reserved on top."
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
                  help="VRAM held aside for the CUDA context, runtime allocations, and display. Larger is more stable but leaves less usable VRAM for weights and KV cache."
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Enable RAM Context Overflow
                      <HelpBubble text="Lets the context spill into WSL2 system RAM when VRAM is full (--swap-space). Prevents out-of-memory crashes on long prompts but is much slower than GPU KV cache." />
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
                    <div className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
                      Allow Weight Offloading
                      <HelpBubble text="Permits offloading model weights to CPU RAM (--cpu-offload-gb) when the model does not fit in VRAM. Lets oversized models run, but CPU inference is far slower than GPU." />
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
                  help="WSL2 system RAM kept untouched for the Linux kernel and background services. Larger is more stable but shrinks the usable overflow budget."
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
                  help="Optional ceiling for context lengths in fit calculations. A lower cap saves VRAM/RAM but truncates long prompts; blank means uncapped."
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
                    help="Auto sizes the swap budget from available WSL2 RAM minus the safety reserve; Manual locks a fixed MB budget. A bigger budget fits longer contexts but uses more system RAM."
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
            </div>
          )}
          {advTab === "environment" && (
            <div className="space-y-6">
          {/* Section 5: Environment & Logging */}
          <Card>
            <CardTitle right={saved ? <Badge color="emerald">saved</Badge> : undefined}>
              Environment & Logging
            </CardTitle>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field
                label="vLLM Logging Level"
                help="vLLM log verbosity. DEBUG adds CPU and disk I/O during inference; ERROR is the cheapest. No effect on model RAM."
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
                  help="Environment variables exported into WSL before vLLM starts. Some (NCCL, Triton cache) can affect speed or stability; none change model RAM directly. A bad value can prevent a server from starting."
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
            </div>
          )}
        </div>
      )}

      {/* Update Modal */}
      {updateInfo && (
        <UpdateModal
          info={updateInfo}
          isOpen={showUpdateModal}
          onClose={() => setShowUpdateModal(false)}
        />
      )}
    </div>
  );
}

function ClientSnippets({
  gatewayPort,
  instructModel,
  embedModel,
  apiKey,
  copied,
  setCopied,
}: {
  gatewayPort: number;
  instructModel: string | null;
  embedModel: string | null;
  apiKey?: string | null;
  copied: string | null;
  setCopied: (k: string | null) => void;
}) {
  const base = `http://127.0.0.1:${gatewayPort}/v1`;
  const chat = instructModel ?? "your-instruct-model";
  const embed = embedModel ?? "your-embed-model";
  const keyToUse = apiKey || "YOUR_GATEWAY_API_KEY";

  const snippets: { key: string; label: string; hint: string; text: string }[] = [
    {
      key: "continue",
      label: "Continue config.yaml",
      hint: "Paste into ~/.continue/config.yaml",
      text: `name: Local Assistant\nversion: 1.0.0\nschema: v1\nmodels:\n  - name: Local Instruct\n    provider: openai\n    model: ${chat}\n    apiBase: ${base}\n    apiKey: "${keyToUse}"\n  - name: Local Autocomplete\n    provider: openai\n    model: ${chat}\n    apiBase: ${base}\n    apiKey: "${keyToUse}"\n    capabilities: [autocomplete]\n  - name: Local Embed\n    provider: openai\n    model: ${embed}\n    apiBase: ${base}\n    apiKey: "${keyToUse}"\n    capabilities: [embed]\n`,
    },
    {
      key: "cursor",
      label: "Cursor / Cline",
      hint: "Base URL + model override",
      text: `Base URL: ${base}\nChat model: ${chat}\nAutocomplete model: ${chat}\nEmbeddings model: ${embed}\nAPI key: ${keyToUse}\n`,
    },
    {
      key: "curl-responses",
      label: "curl responses",
      hint: "Responses API (/v1/responses) smoke test",
      text: `GATEWAY_API_KEY='${keyToUse}'; curl ${base}/responses -H "Authorization: Bearer $GATEWAY_API_KEY" -H "Content-Type: application/json" -d '{"model":"${chat}","input":"hi","instructions":"You are a helpful assistant"}'\n`,
    },
    {
      key: "curl-chat",
      label: "curl chat",
      hint: "Chat + FIM completions smoke test",
      text: `GATEWAY_API_KEY='${keyToUse}'; curl ${base}/chat/completions -H "Authorization: Bearer $GATEWAY_API_KEY" -H "Content-Type: application/json" -d '{"model":"${chat}","messages":[{"role":"user","content":"hi"}]}'\n`,
    },
    {
      key: "curl-embed",
      label: "curl embeddings",
      hint: "Embeddings smoke test",
      text: `GATEWAY_API_KEY='${keyToUse}'; curl ${base}/embeddings -H "Authorization: Bearer $GATEWAY_API_KEY" -H "Content-Type: application/json" -d '{"model":"${embed}","input":"hello world"}'\n`,
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
      <div className="mt-1 text-[11px] text-slate-400">
        The gateway API key is local to LocalLLmPanel and protects your loopback endpoints from unauthorized local processes or browser pages. It is not an OpenAI hosted platform key.
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