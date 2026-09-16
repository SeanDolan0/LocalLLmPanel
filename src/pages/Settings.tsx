import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Badge, Button, Card, CardTitle, Field, inputCls } from "../ui";
import type { Settings as SettingsT, WslConfigInfo } from "../types";

export default function Settings() {
  const [s, setS] = useState<SettingsT | null>(null);
  const [wsl, setWsl] = useState<WslConfigInfo | null>(null);
  const [saved, setSaved] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.settingsGet().then(setS).catch((e) => setErr(String(e)));
    api.wslconfigGet().then(setWsl).catch(() => {});
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

  if (!s) {
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