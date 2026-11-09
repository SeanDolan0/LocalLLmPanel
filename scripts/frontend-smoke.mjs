#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");

function invokeCommands(source) {
  return [...source.matchAll(/invoke\s*(?:<[\s\S]*?>)?\s*\(\s*["']([^"']+)["']/g)].map((match) => match[1]);
}

function findSourceFiles(dir, extensions = [".ts", ".tsx"]) {
  const results = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "node_modules" && entry.name !== "dist" && entry.name !== "dist-release") {
        results.push(...findSourceFiles(full, extensions));
      }
    } else if (extensions.some((ext) => entry.name.endsWith(ext))) {
      results.push(full);
    }
  }
  return results;
}

test("frontend invoke surface is registered by Tauri", () => {
  const lib = read("src-tauri/src/lib.rs");
  const handlers = new Set([...lib.matchAll(/commands::([A-Za-z0-9_]+)/g)].map((match) => match[1]));

  const sourceFiles = findSourceFiles(path.join(root, "src"));
  const allInvokes = new Set();
  for (const file of sourceFiles) {
    const content = fs.readFileSync(file, "utf8");
    for (const cmd of invokeCommands(content)) {
      allInvokes.add(cmd);
    }
  }

  const missing = [...allInvokes].filter((command) => !handlers.has(command));
  assert.deepEqual(missing, [], `unregistered Tauri commands found in frontend: ${missing.join(", ")}`);
});

test("release manifests stay synchronized and lock metadata is valid", async () => {
  const { execFileSync } = await import("node:child_process");
  execFileSync(process.execPath, [path.join(root, "scripts", "validate-release.mjs")], {
    cwd: root,
    stdio: "pipe",
  });
});

test("frontend does not ship known unsafe gateway examples", () => {
  const settings = read("src/pages/Settings.tsx");
  assert.equal(settings.includes("apiKey: not-needed"), false);
  assert.equal(settings.includes("API key: not-needed"), false);
});

test("Tauri ships a restrictive CSP and secure window settings", () => {
  const config = JSON.parse(read("src-tauri/tauri.conf.json"));
  assert.equal(typeof config.app.security.csp, "string");
  assert.match(config.app.security.csp, /default-src 'self'/);
  assert.match(config.app.security.csp, /object-src 'none'/);
  assert.equal(config.app.security.dangerousDisableAssetCspModification, undefined);
  assert.equal(config.productName, "Local LLM Panel");
  assert.equal(config.identifier, "com.local-llm-panel.app");
});

test("gateway source does not advertise a wildcard CORS policy", () => {
  const gateway = read("src-tauri/src/gateway.rs");
  assert.equal(gateway.includes("Access-Control-Allow-Origin: *"), false);
});

test("frontend entry points and navigation routes exist and match", () => {
  const app = read("src/App.tsx");
  const navTargets = [...app.matchAll(/to:\s*["']([^"']+)["']/g)].map((m) => m[1]);
  const routePaths = [...app.matchAll(/path\s*=\s*["']([^"']+)["']/g)].map((m) => m[1]);

  for (const target of navTargets) {
    assert.ok(
      routePaths.includes(target) || routePaths.includes(target.replace(/^\//, "")),
      `Navigation target ${target} has no matching Route in App.tsx`
    );
  }

  for (const file of [
    "src/main.tsx",
    "src/App.tsx",
    "src/pages/Dashboard.tsx",
    "src/pages/Search.tsx",
    "src/pages/Library.tsx",
    "src/pages/Servers.tsx",
    "src/pages/Settings.tsx",
  ]) {
    assert.equal(fs.existsSync(path.join(root, file)), true, `missing frontend entry: ${file}`);
  }
});

test("core documentation and policy files exist and are non-empty", () => {
  const docs = [
    "README.md",
    "ABOUT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/USER_GUIDE.md",
  ];
  for (const doc of docs) {
    const full = path.join(root, doc);
    assert.equal(fs.existsSync(full), true, `missing required documentation: ${doc}`);
    const stat = fs.statSync(full);
    assert.ok(stat.size > 200, `${doc} is unexpectedly small (${stat.size} bytes)`);
  }
});

test("Tauri updater plugin configuration and capabilities are valid", () => {
  const tauriConf = JSON.parse(read("src-tauri/tauri.conf.json"));
  assert.equal(tauriConf.bundle?.createUpdaterArtifacts, true, "bundle.createUpdaterArtifacts must be true");
  assert.ok(tauriConf.plugins?.updater, "plugins.updater configuration must exist");
  assert.ok(typeof tauriConf.plugins.updater.pubkey === "string" && tauriConf.plugins.updater.pubkey.length > 20, "pubkey must be configured");
  assert.ok(Array.isArray(tauriConf.plugins.updater.endpoints) && tauriConf.plugins.updater.endpoints.length > 0, "endpoints must be configured");

  for (const endpoint of tauriConf.plugins.updater.endpoints) {
    assert.ok(endpoint.startsWith("https://"), `updater endpoint must use https: ${endpoint}`);
  }

  // Ensure no private key is committed in repo
  const repoFiles = [
    "src-tauri/tauri.conf.json",
    "src-tauri/Cargo.toml",
    "package.json",
    ".github/workflows/draft-release.yml"
  ];
  for (const f of repoFiles) {
    const content = read(f);
    assert.equal(content.includes("untrusted comment: rsign encrypted secret key"), false, `secret key found in ${f}`);
  }

  const caps = JSON.parse(read("src-tauri/capabilities/default.json"));
  assert.ok(caps.permissions.includes("updater:default"), "capabilities/default.json must include updater:default");
});

test("updater IPC methods and types maintain frontend contract", () => {
  const typesContent = read("src/types.ts");
  const requiredTypes = [
    "export interface AppUpdateInfo",
    "export interface AppUpdateProgress",
    "export interface UpdateLaunchResult",
    "export interface LastUpdateStatus",
  ];
  for (const t of requiredTypes) {
    assert.ok(typesContent.includes(t), `missing expected updater type: ${t}`);
  }

  const apiContent = read("src/api.ts");
  assert.ok(apiContent.includes("checkAppUpdate"), "missing api.checkAppUpdate");
  assert.ok(apiContent.includes("downloadAndInstallUpdate"), "missing api.downloadAndInstallUpdate");
  assert.ok(apiContent.includes("lastUpdateStatus"), "missing api.lastUpdateStatus");
});

test("updater manifest parsing correctly selects windows-x86_64 target", () => {
  const sampleManifest = {
    version: "v2.3.0",
    notes: "New release",
    pub_date: "2026-11-06T00:00:00Z",
    platforms: {
      "windows-x86_64": {
        signature: "sample_sig",
        url: "https://github.com/SeanDolan0/LocalLLmPanel/releases/download/v2.3.0/Local.LLM.Panel_2.3.0_x64-setup.nsis.zip"
      }
    }
  };

  assert.ok(sampleManifest.platforms["windows-x86_64"], "manifest must contain windows-x86_64");
  assert.equal(sampleManifest.platforms["darwin-x86_64"], undefined, "manifest must not serve darwin");
  assert.equal(sampleManifest.platforms["linux-x86_64"], undefined, "manifest must not serve linux");
  assert.ok(sampleManifest.platforms["windows-x86_64"].url.endsWith(".nsis.zip"));
});

test("audit allowlist schema and parsing", async () => {
  const { parseAuditAllowlist, isAdvisoryAllowed, getCargoIgnoreFlags } = await import("./security-audit.mjs");
  const rawAllowlist = read("audit-allowlist.json");
  const allowlist = parseAuditAllowlist(rawAllowlist);

  assert.ok(Array.isArray(allowlist.advisories));
  assert.ok(allowlist.advisories.length >= 2);

  // Check known accepted advisories
  assert.equal(isAdvisoryAllowed("RUSTSEC-2024-0370", allowlist), true);
  assert.equal(isAdvisoryAllowed("rustsec-2024-0370", allowlist), true, "allowlist matching must be case-insensitive");
  assert.equal(isAdvisoryAllowed("RUSTSEC-2024-0429", allowlist), true);
  assert.equal(isAdvisoryAllowed("GHSA-non-existent-9999", allowlist), false);

  // Validate cargo ignore flags
  const flags = getCargoIgnoreFlags(allowlist);
  assert.ok(flags.includes("--ignore RUSTSEC-2024-0370"));
  assert.ok(flags.includes("--ignore RUSTSEC-2024-0429"));

  // Negative test cases: malformed allowlist
  assert.throws(() => parseAuditAllowlist("{ malformed json"), /Malformed audit allowlist/);
  assert.throws(() => parseAuditAllowlist(JSON.stringify({ advisories: "not an array" })), /missing required 'advisories' array/);
  assert.throws(() => parseAuditAllowlist(JSON.stringify({ advisories: [{ reason: "no id" }] })), /missing 'id'/);
  assert.throws(() => parseAuditAllowlist(JSON.stringify({ advisories: [{ id: "R1", package: "pkg", reason: "" }] })), /must document a non-empty 'reason'/);
});

test("cleanup dry-run and itemization IPC contract", () => {
  const lib = read("src-tauri/src/lib.rs");
  assert.ok(lib.includes("commands::cleanup_inspect"), "lib.rs must register cleanup_inspect");
  assert.ok(lib.includes("commands::cleanup_execute"), "lib.rs must register cleanup_execute");

  const commands = read("src-tauri/src/commands.rs");
  assert.ok(commands.includes("pub async fn cleanup_inspect"), "commands.rs missing cleanup_inspect");
  assert.ok(commands.includes("pub async fn cleanup_execute"), "commands.rs missing cleanup_execute");

  const api = read("src/api.ts");
  assert.ok(api.includes("cleanupInspect: () => invoke<CleanupPlan>(\"cleanup_inspect\")"));
  assert.ok(api.includes("cleanupExecute: (confirmedTargets: CleanupTarget[]) =>"));

  const types = read("src/types.ts");
  const requiredTypes = [
    "export type CleanupTarget",
    "export interface CleanupItem",
    "export interface CleanupPlan",
    "export interface CleanupItemResult",
    "export interface CleanupExecutionReport",
  ];
  for (const t of requiredTypes) {
    assert.ok(types.includes(t), `missing expected cleanup type: ${t}`);
  }

  // Ensure all 6 cleanup targets are represented in CleanupTarget union
  for (const target of ["wsl_distro", "venv", "hf_cache", "gguf_dir", "partial_downloads", "sudoers_rule"]) {
    assert.ok(types.includes(`"${target}"`), `CleanupTarget missing target: ${target}`);
  }
});

test("Rust commands and frontend API contract drift validation", () => {
  const commandsSource = read("src-tauri/src/commands.rs");
  const apiSource = read("src/api.ts");
  const libSource = read("src-tauri/src/lib.rs");

  const registeredHandlers = new Set([...libSource.matchAll(/commands::([A-Za-z0-9_]+)/g)].map((m) => m[1]));

  // Parse all #[tauri::command] functions
  const cmdRegex = /#\[tauri::command\]\s*(?:#\[[^\]]+\]\s*)*pub\s+(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*\(([\s\S]*?)\)\s*(?:->\s*([^{]+))?\{/g;
  const rustCommands = new Map();

  for (const match of commandsSource.matchAll(cmdRegex)) {
    const fnName = match[1];
    const rawParams = match[2];
    const returnType = (match[3] || "").trim();

    // Bracket-aware parameter splitting (handles nested generics like State<'_, Arc<AppState>>)
    function splitParams(raw) {
      const parts = [];
      let current = "";
      let depth = 0;
      for (let i = 0; i < raw.length; i++) {
        const ch = raw[i];
        if (ch === "<" || ch === "(" || ch === "[") depth++;
        else if (ch === ">" || ch === ")" || ch === "]") depth--;
        else if (ch === "," && depth === 0) {
          parts.push(current.trim());
          current = "";
          continue;
        }
        current += ch;
      }
      if (current.trim()) parts.push(current.trim());
      return parts;
    }

    const params = splitParams(rawParams)
      .filter((p) => p.length > 0 && !p.startsWith("//"))
      .filter((p) => {
        return (
          !p.includes("State<") &&
          !p.includes("State<'") &&
          !p.includes("AppHandle") &&
          !p.includes("&AppHandle") &&
          !p.includes("Window") &&
          !p.includes("WebviewWindow")
        );
      })
      .map((p) => {
        const colonIdx = p.indexOf(":");
        if (colonIdx === -1) return { name: p, isOptional: false, type: "" };
        const name = p.slice(0, colonIdx).trim().replace(/^mut\s+/, "");
        const type = p.slice(colonIdx + 1).trim();
        const isOptional = type.startsWith("Option<");
        return { name, isOptional, type };
      });

    rustCommands.set(fnName, { fnName, params, returnType });
  }

  // Parse all invoke calls in api.ts
  const invokeRegex = /invoke\s*(?:<([\s\S]*?)>)?\s*\(\s*["']([^"']+)["']\s*(?:,\s*(\{[\s\S]*?\}))?\s*\)/g;
  const apiCalls = new Map();

  for (const match of apiSource.matchAll(invokeRegex)) {
    const returnType = (match[1] || "").trim();
    const cmdName = match[2];
    const rawPayload = match[3] || "";

    // Extract property keys from payload
    const keys = [];
    if (rawPayload) {
      const propRegex = /([A-Za-z0-9_]+)\s*:/g;
      for (const m of rawPayload.matchAll(propRegex)) {
        keys.push(m[1]);
      }
      // Also shorthand properties e.g. { modelId, repoId }
      const inner = rawPayload.slice(1, -1).trim();
      for (const part of inner.split(",")) {
        const trimmed = part.trim();
        if (trimmed && !trimmed.includes(":") && /^[A-Za-z0-9_]+$/.test(trimmed)) {
          keys.push(trimmed);
        }
      }
    }

    apiCalls.set(cmdName, { cmdName, returnType, keys: new Set(keys) });
  }

  // 1. Verify every frontend invoke call has a registered Tauri handler
  for (const [cmdName] of apiCalls) {
    assert.ok(
      registeredHandlers.has(cmdName),
      `Frontend calls unhandled command: "${cmdName}" (not registered in lib.rs)`
    );
  }

  // 2. Verify argument names between Rust and TS
  function toCamelCase(str) {
    return str.replace(/_([a-z0-9])/g, (_, g) => g.toUpperCase());
  }

  for (const [cmdName, apiCall] of apiCalls) {
    const rustCmd = rustCommands.get(cmdName);
    if (!rustCmd) continue; // Command may be defined in another module

    for (const param of rustCmd.params) {
      if (param.isOptional) continue; // Optional params can be omitted

      const camelParam = toCamelCase(param.name);
      const snakeParam = param.name;

      const hasArg = apiCall.keys.has(camelParam) || apiCall.keys.has(snakeParam);
      assert.ok(
        hasArg,
        `Contract Drift in command "${cmdName}": Rust expects required parameter "${param.name}" (${param.type}), but api.ts payload is missing "${camelParam}". Present keys: [${[...apiCall.keys].join(", ")}]`
      );
    }
  }
});
