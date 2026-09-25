#!/usr/bin/env node
/**
 * Deterministic release checks used by CI and the build runner.
 *
 * This intentionally does not build or publish anything.  It catches the two
 * classes of release mistakes that are easy to miss in a desktop release:
 * version drift between manifests and corrupt npm lock metadata, and a build
 * that claims success without producing the requested artifacts.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tauriDir = path.join(rootDir, "src-tauri");
const distDir = path.resolve(
  process.env.LLM_PANEL_DIST_DIR || path.join(rootDir, "dist-release"),
);
const args = process.argv.slice(2);
const mode = args.find((arg) => ["--fast", "--setup", "--all"].includes(arg))?.slice(2) ?? "setup";
const shouldCheckArtifacts = args.includes("--artifacts");

function readText(file) {
  return fs.readFileSync(file, "utf8");
}

function readJson(file) {
  return JSON.parse(readText(file));
}

function fail(message) {
  throw new Error(message);
}

function inferTarballVersion(packagePath, record) {
  if (typeof record.resolved !== "string" || !record.resolved.includes(".tgz")) {
    return null;
  }
  const packageName = packagePath.replace(/^.*node_modules\//, "");
  const baseName = packageName.split("/").pop();
  const filename = record.resolved.split("/").pop();
  if (!baseName || !filename) return null;
  const marker = `${baseName}-`;
  const start = filename.indexOf(marker);
  if (start < 0 || !filename.endsWith(".tgz")) return null;
  return filename.slice(start + marker.length, -4);
}

function checkManifestVersions(pkg, lock, cargoToml, tauriConf) {
  const cargoMatch = cargoToml.match(/name\s*=\s*"llm-panel"\s*\nversion\s*=\s*"([^"]+)"/);
  const version = pkg.version;
  if (!version) fail("package.json has no version");
  if (lock.version !== version) fail(`package-lock.json version ${lock.version} != ${version}`);
  if (lock.packages?.[""]?.version !== version) {
    fail(`package-lock.json root package version ${lock.packages?.[""]?.version} != ${version}`);
  }
  if (!cargoMatch || cargoMatch[1] !== version) {
    fail(`Cargo.toml version ${cargoMatch?.[1] ?? "missing"} != ${version}`);
  }
  if (tauriConf.version !== version) {
    fail(`tauri.conf.json version ${tauriConf.version} != ${version}`);
  }
}

function checkLockIntegrity(lock) {
  const badVersion = "1.7.3-rev2";
  for (const [packagePath, record] of Object.entries(lock.packages ?? {})) {
    if (record.version === badVersion) {
      fail(`corrupt lockfile version for ${packagePath}: ${badVersion}`);
    }
    const inferred = inferTarballVersion(packagePath, record);
    if (inferred && record.version && inferred !== record.version) {
      fail(`lockfile version mismatch for ${packagePath}: ${record.version} != ${inferred}`);
    }
  }
}

function expectedArtifacts(version) {
  const artifacts = [`LocalLLMPanel-v${version}.exe`];
  if (mode !== "fast") artifacts.push(`Local.LLM.Panel_${version}_x64-setup.exe`);
  if (mode === "all") artifacts.push(`Local.LLM.Panel_${version}_x64_en-US.msi`);
  return artifacts;
}

function validateArtifacts(version) {
  for (const name of expectedArtifacts(version)) {
    const file = path.join(distDir, name);
    if (!fs.existsSync(file)) fail(`missing release artifact: ${file}`);
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size === 0) fail(`empty release artifact: ${file}`);

    const checksumFile = `${file}.sha256`;
    if (!fs.existsSync(checksumFile)) fail(`missing checksum: ${checksumFile}`);
    const digest = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    const recorded = readText(checksumFile).trim().split(/\s+/)[0];
    if (recorded !== digest) fail(`checksum mismatch for ${name}`);
  }
}

const pkg = readJson(path.join(rootDir, "package.json"));
const lock = readJson(path.join(rootDir, "package-lock.json"));
const cargoToml = readText(path.join(tauriDir, "Cargo.toml"));
const tauriConf = readJson(path.join(tauriDir, "tauri.conf.json"));

checkManifestVersions(pkg, lock, cargoToml, tauriConf);
checkLockIntegrity(lock);
if (shouldCheckArtifacts) validateArtifacts(pkg.version);

console.log(`[release] validated v${pkg.version} (${mode}${shouldCheckArtifacts ? ", artifacts" : ""})`);
