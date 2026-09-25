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

test("frontend invoke surface is registered by Tauri", () => {
  const api = read("src/api.ts");
  const lib = read("src-tauri/src/lib.rs");
  const handlers = new Set([...lib.matchAll(/commands::([A-Za-z0-9_]+)/g)].map((match) => match[1]));
  const missing = [...new Set(invokeCommands(api))].filter((command) => !handlers.has(command));
  assert.deepEqual(missing, [], `unregistered Tauri commands: ${missing.join(", ")}`);
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

test("Tauri ships a restrictive CSP", () => {
  const config = JSON.parse(read("src-tauri/tauri.conf.json"));
  assert.equal(typeof config.app.security.csp, "string");
  assert.match(config.app.security.csp, /default-src 'self'/);
  assert.match(config.app.security.csp, /object-src 'none'/);
  assert.equal(config.app.security.dangerousDisableAssetCspModification, undefined);
});

test("gateway source does not advertise a wildcard CORS policy", () => {
  const gateway = read("src-tauri/src/gateway.rs");
  assert.equal(gateway.includes("Access-Control-Allow-Origin: *"), false);
});

test("frontend entry points exist for the primary user flows", () => {
  for (const file of ["src/main.tsx", "src/pages/Dashboard.tsx", "src/pages/Search.tsx", "src/pages/Library.tsx", "src/pages/Servers.tsx", "src/pages/Settings.tsx"]) {
    assert.equal(fs.existsSync(path.join(root, file)), true, `missing frontend entry: ${file}`);
  }
});
