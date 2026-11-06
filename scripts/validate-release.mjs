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

  // Validate official signed updater bundle and latest.json metadata
  const hasSigningKey = Boolean(process.env.TAURI_SIGNING_PRIVATE_KEY);
  const isSignedReleaseRequired = Boolean(
    (process.env.REQUIRE_SIGNED_RELEASE === "1" && hasSigningKey) || fs.existsSync(path.join(distDir, "latest.json"))
  );

  if (mode !== "fast" && isSignedReleaseRequired) {
    const zipName = `Local.LLM.Panel_${version}_x64-setup.nsis.zip`;
    const zipFile = path.join(distDir, zipName);
    if (!fs.existsSync(zipFile)) fail(`missing updater bundle artifact: ${zipFile}`);
    const zipStat = fs.statSync(zipFile);
    if (!zipStat.isFile() || zipStat.size === 0) fail(`empty updater bundle artifact: ${zipFile}`);

    // Verify zip checksum
    const zipChecksumFile = `${zipFile}.sha256`;
    if (fs.existsSync(zipChecksumFile)) {
      const digest = crypto.createHash("sha256").update(fs.readFileSync(zipFile)).digest("hex");
      const recorded = readText(zipChecksumFile).trim().split(/\s+/)[0];
      if (recorded !== digest) fail(`checksum mismatch for updater bundle ${zipName}`);
    }

    // Verify signature file
    const sigFile = `${zipFile}.sig`;
    if (!fs.existsSync(sigFile)) fail(`missing updater signature file: ${sigFile}`);
    const sigContent = readText(sigFile).trim();
    let decodedSig = "";
    try {
      decodedSig = Buffer.from(sigContent, "base64").toString("utf8");
    } catch {}
    if (!decodedSig.startsWith("untrusted comment:") && !sigContent.startsWith("untrusted comment:")) {
      fail(`invalid updater signature format in ${sigFile}`);
    }

    // Verify latest.json metadata
    const manifestFile = path.join(distDir, "latest.json");
    if (!fs.existsSync(manifestFile)) fail(`missing updater metadata manifest: ${manifestFile}`);
    let manifest;
    try {
      manifest = JSON.parse(readText(manifestFile));
    } catch (e) {
      fail(`malformed latest.json manifest: ${e.message}`);
    }

    const normManifestVersion = String(manifest.version || "").replace(/^v/, "");
    if (normManifestVersion !== version) {
      fail(`latest.json version ${manifest.version} does not match expected version ${version}`);
    }

    if (!manifest.platforms || typeof manifest.platforms !== "object") {
      fail("latest.json missing platforms object");
    }

    const winPlatform = manifest.platforms["windows-x86_64"];
    if (!winPlatform) {
      fail("latest.json does not contain platform entry for windows-x86_64");
    }

    if (!winPlatform.signature || winPlatform.signature.trim() !== sigContent) {
      fail("latest.json signature does not match updater .sig file content");
    }

    const expectedUrl = `https://github.com/SeanDolan0/LocalLLmPanel/releases/download/v${version}/${zipName}`;
    if (winPlatform.url !== expectedUrl) {
      fail(`latest.json URL ${winPlatform.url} does not match expected release URL ${expectedUrl}`);
    }

    // Ensure we do not accidentally serve foreign architectures as windows-x86_64
    for (const [key, platform] of Object.entries(manifest.platforms)) {
      if (key !== "windows-x86_64") {
        fail(`latest.json serves unexpected platform target: ${key}`);
      }
      if (!platform.url.includes("x64") || !platform.url.endsWith(".zip")) {
        fail(`latest.json served asset is not a Windows x64 zip bundle: ${platform.url}`);
      }
    }
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
