#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");

function invokeCommands(source) {
  return [...source.matchAll(/invoke(?:<[^>]+>)?\(\s*["']([^"']+)["']/g)].map((match) => match[1]);
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
