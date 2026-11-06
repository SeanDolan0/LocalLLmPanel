#!/usr/bin/env node
import { execFileSync, execSync, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const tauriDir = path.join(rootDir, "src-tauri");
const distReleaseDir = path.resolve(
  process.env.LLM_PANEL_DIST_DIR || path.join(rootDir, "dist-release"),
);

// 1. Parse arguments
function copyArtifact(source, destination) {
  // A previous release may still be open in Explorer/antivirus.  Remove the
  // old staged file before copying so the build does not fail with EBUSY on a
  // stale artifact from an earlier run.
  try {
    if (fs.existsSync(destination)) {
      fs.rmSync(destination, { force: true });
    }
    fs.copyFileSync(source, destination);
  } catch (error) {
    if (error?.code === "EPERM" || error?.code === "EBUSY") {
      throw new Error(
        `Cannot replace ${destination}; close any running Local LLM Panel instance and retry`,
        { cause: error },
      );
    }
    throw error;
  }
}

function createStoredZip(srcPath, destZipPath, entryName) {
  const content = fs.readFileSync(srcPath);
  const nameBuf = Buffer.from(entryName, "utf8");
  const crc = zlib.crc32(content);
  const size = content.length;

  const localHeader = Buffer.alloc(30 + nameBuf.length);
  localHeader.writeUInt32LE(0x04034b50, 0); // Local file header signature
  localHeader.writeUInt16LE(20, 4);         // Version needed to extract (2.0)
  localHeader.writeUInt16LE(0, 6);          // General purpose bit flag
  localHeader.writeUInt16LE(0, 8);          // Compression method: 0 = Stored (uncompressed)
  localHeader.writeUInt16LE(0, 10);         // Last mod file time
  localHeader.writeUInt16LE(0x21, 12);       // Last mod file date (1980-01-01)
  localHeader.writeUInt32LE(crc, 14);       // CRC-32
  localHeader.writeUInt32LE(size, 18);      // Compressed size
  localHeader.writeUInt32LE(size, 22);      // Uncompressed size
  localHeader.writeUInt16LE(nameBuf.length, 26); // File name length
  localHeader.writeUInt16LE(0, 28);         // Extra field length
  nameBuf.copy(localHeader, 30);

  const centralHeader = Buffer.alloc(46 + nameBuf.length);
  centralHeader.writeUInt32LE(0x02014b50, 0); // Central directory signature
  centralHeader.writeUInt16LE(20, 4);         // Version made by
  centralHeader.writeUInt16LE(20, 6);         // Version needed to extract
  centralHeader.writeUInt16LE(0, 8);          // General purpose bit flag
  centralHeader.writeUInt16LE(0, 10);         // Compression method: 0 = Stored
  centralHeader.writeUInt16LE(0, 12);         // Last mod file time
  centralHeader.writeUInt16LE(0x21, 14);       // Last mod file date
  centralHeader.writeUInt32LE(crc, 16);       // CRC-32
  centralHeader.writeUInt32LE(size, 20);      // Compressed size
  centralHeader.writeUInt32LE(size, 24);      // Uncompressed size
  centralHeader.writeUInt16LE(nameBuf.length, 28); // File name length
  centralHeader.writeUInt16LE(0, 30);         // Extra field length
  centralHeader.writeUInt16LE(0, 32);         // File comment length
  centralHeader.writeUInt16LE(0, 34);         // Disk number start
  centralHeader.writeUInt16LE(0, 36);         // Internal file attributes
  centralHeader.writeUInt32LE(0x00000020, 38); // External file attributes (archive)
  centralHeader.writeUInt32LE(0, 42);         // Relative offset of local header
  nameBuf.copy(centralHeader, 46);

  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);          // EOCD signature
  eocd.writeUInt16LE(0, 4);                   // Number of this disk
  eocd.writeUInt16LE(0, 6);                   // Disk where central directory starts
  eocd.writeUInt16LE(1, 8);                   // Number of central directory records on this disk
  eocd.writeUInt16LE(1, 10);                  // Total number of central directory records
  eocd.writeUInt32LE(centralHeader.length, 12); // Size of central directory
  eocd.writeUInt32LE(localHeader.length + size, 16); // Offset of central directory
  eocd.writeUInt16LE(0, 20);                  // Comment length

  const fd = fs.openSync(destZipPath, "w");
  try {
    fs.writeSync(fd, localHeader);
    fs.writeSync(fd, content);
    fs.writeSync(fd, centralHeader);
    fs.writeSync(fd, eocd);
  } finally {
    fs.closeSync(fd);
  }
}

const args = process.argv.slice(2);
const isFast = args.includes("--fast");
const isAll = args.includes("--all");
const isSetup = args.includes("--setup") || (!isFast && !isAll && !args.includes("--check"));
const isCheckOnly = args.includes("--check");

console.log("\n==================================================");
console.log("  Local LLM Panel - Optimized Build Runner");
console.log("==================================================\n");

// 2. Read and verify versions across configuration files
const pkgJsonPath = path.join(rootDir, "package.json");
const cargoTomlPath = path.join(tauriDir, "Cargo.toml");
const tauriConfPath = path.join(tauriDir, "tauri.conf.json");

const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf8"));
const version = pkg.version;
console.log(`Target Version: v${version}`);

const cargoToml = fs.readFileSync(cargoTomlPath, "utf8");
const cargoVersionMatch = cargoToml.match(/name\s*=\s*"llm-panel"\s*\nversion\s*=\s*"([^"]+)"/);
const tauriConf = JSON.parse(fs.readFileSync(tauriConfPath, "utf8"));

if (!cargoVersionMatch || cargoVersionMatch[1] !== version) {
  throw new Error(
    `[release] Cargo.toml version ${cargoVersionMatch?.[1] ?? "missing"} does not match package.json ${version}`,
  );
}
if (tauriConf.version !== version) {
  throw new Error(
    `[release] tauri.conf.json version ${tauriConf.version} does not match package.json ${version}`,
  );
}

const validateArgs = [path.join(rootDir, "scripts", "validate-release.mjs")];
if (isAll) validateArgs.push("--all");
else if (isFast) validateArgs.push("--fast");
else if (isSetup) validateArgs.push("--setup");
execFileSync(process.execPath, validateArgs, { cwd: rootDir, stdio: "inherit" });

// 3. Environment & Linker Resolution
function configureEnvironment() {
  const env = { ...process.env };
  const pathSeparator = path.delimiter;
  const currentPath = env.PATH || env.Path || "";

  // Check if lld-link is already reachable
  let hasLldLink = false;
  try {
    const check = spawnSync("lld-link.exe", ["--version"], { stdio: "ignore" });
    if (check.status === 0) hasLldLink = true;
  } catch {}

  if (!hasLldLink) {
    const candidatePaths = [
      "C:\\Program Files\\LLVM\\bin",
      "C:\\Program Files (x86)\\LLVM\\bin",
      "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\Llvm\\x64\\bin",
      "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Tools\\Llvm\\x64\\bin",
      "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\Llvm\\x64\\bin",
      "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\Llvm\\x64\\bin",
    ];

    for (const p of candidatePaths) {
      if (fs.existsSync(path.join(p, "lld-link.exe"))) {
        console.log(`[Toolchain] Found LLVM linker at: ${p}`);
        env.PATH = `${p}${pathSeparator}${currentPath}`;
        env.Path = env.PATH;
        hasLldLink = true;
        break;
      }
    }
  }

  if (!hasLldLink) {
    try {
      const rustcSysroot = execSync("rustc --print sysroot", { encoding: "utf8" }).trim();
      const rustLldDir = path.join(rustcSysroot, "lib", "rustlib", "x86_64-pc-windows-msvc", "bin");
      if (fs.existsSync(path.join(rustLldDir, "rust-lld.exe"))) {
        console.log(`[Toolchain] Found rust-lld at: ${rustLldDir}`);
        env.PATH = `${rustLldDir}${pathSeparator}${currentPath}`;
        env.Path = env.PATH;
        env.CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "rust-lld.exe";
        hasLldLink = true;
      }
    } catch {}
  }

  if (!hasLldLink) {
    console.warn("[WARN] lld-link.exe not found in PATH or standard LLVM directories.");
    console.warn("[WARN] Falling back to default MSVC link.exe (linking may be slower).");
    env.CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "link.exe";
  } else {
    console.log("[Toolchain] High-speed lld linker enabled.");
  }

  return env;
}

const buildEnv = configureEnvironment();
const startTime = Date.now();

// 4. Check mode
if (isCheckOnly) {
  console.log("\n[Preflight] Running TypeScript check...");
  execSync("npm run build", { cwd: rootDir, env: buildEnv, stdio: "inherit" });
  console.log("\n[Preflight] Running Cargo workspace check...");
  execSync("cargo check --workspace --locked", { cwd: tauriDir, env: buildEnv, stdio: "inherit" });
  console.log("\n[Preflight] Running frontend contract smoke tests...");
  execSync("npm run test:contract", { cwd: rootDir, env: buildEnv, stdio: "inherit" });
  console.log(`\n[OK] Preflight checks passed in ${((Date.now() - startTime) / 1000).toFixed(2)}s.`);
  process.exit(0);
}

// 5. Ensure dist-release output directory exists
if (!fs.existsSync(distReleaseDir)) {
  fs.mkdirSync(distReleaseDir, { recursive: true });
}

// 6. Assemble Tauri build command
let tauriArgs = ["tauri", "build"];
let modeLabel = "";

if (isFast) {
  tauriArgs.push("--no-bundle");
  modeLabel = "Fast Standalone (.exe only, no installers)";
} else if (isAll) {
  tauriArgs.push("--bundles", "nsis,msi");
  modeLabel = "Full Release (Standalone + NSIS Installer + WiX MSI)";
} else {
  tauriArgs.push("--bundles", "nsis");
  modeLabel = "Setup Release (Standalone + NSIS Installer)";
}

const signingKey = process.env.TAURI_SIGNING_PRIVATE_KEY || buildEnv.TAURI_SIGNING_PRIVATE_KEY;
if (signingKey) {
  tauriArgs.push("--ci");
} else if (!isFast) {
  tauriArgs.push("--no-sign");
}

console.log(`\nBuild Mode: ${modeLabel}`);
const fullCmd = `npx ${tauriArgs.join(" ")}`;
console.log(`Executing: ${fullCmd}\n`);

try {
  execSync(fullCmd, {
    cwd: rootDir,
    env: buildEnv,
    stdio: "inherit",
  });
} catch (err) {
  console.error(`\n[ERROR] Build failed: ${err.message}`);
  process.exit(err.status || 1);
}

// 7. Stage and copy artifacts to dist-release
console.log("\n[Staging] Staging release artifacts to dist-release/...");

const stagedFiles = [];
const rawExePath = path.join(tauriDir, "target", "release", "llm-panel.exe");
const stagedExeName = `LocalLLMPanel-v${version}.exe`;
const stagedExePath = path.join(distReleaseDir, stagedExeName);

if (fs.existsSync(rawExePath)) {
  copyArtifact(rawExePath, stagedExePath);
  const sizeMb = (fs.statSync(stagedExePath).size / (1024 * 1024)).toFixed(2);
  stagedFiles.push({ name: stagedExeName, type: "Standalone Executable", size: `${sizeMb} MB`, path: stagedExePath });
}

if (!isFast) {
  // Find NSIS output
  const nsisDir = path.join(tauriDir, "target", "release", "bundle", "nsis");
  if (fs.existsSync(nsisDir)) {
    const matchingNsis = fs.readdirSync(nsisDir).find(f => f.endsWith(".exe") && f.includes(version));
    if (matchingNsis) {
      const src = path.join(nsisDir, matchingNsis);
      const destName = `Local.LLM.Panel_${version}_x64-setup.exe`;
      const dest = path.join(distReleaseDir, destName);
      copyArtifact(src, dest);
      const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
      stagedFiles.push({ name: destName, type: "NSIS Installer", size: `${sizeMb} MB`, path: dest });

      // Create NSIS updater bundle zip: Local.LLM.Panel_${version}_x64-setup.nsis.zip
      // Must use compression method 0 (Stored) to ensure universal compatibility
      // with older installed app versions that lack deflate decompression in their updater.
      const zipName = `Local.LLM.Panel_${version}_x64-setup.nsis.zip`;
      const zipPath = path.join(distReleaseDir, zipName);
      if (fs.existsSync(zipPath)) {
        fs.rmSync(zipPath, { force: true });
      }
      createStoredZip(dest, zipPath, destName);
      const zipSizeMb = (fs.statSync(zipPath).size / (1024 * 1024)).toFixed(2);
      stagedFiles.push({ name: zipName, type: "NSIS Updater Bundle", size: `${zipSizeMb} MB`, path: zipPath });

      // Sign updater artifacts if TAURI_SIGNING_PRIVATE_KEY is available
      if (signingKey) {
        console.log("\n[Signing] Signing updater bundle and installer with Tauri signer...");
        const signEnv = {
          ...buildEnv,
          TAURI_SIGNING_PRIVATE_KEY: signingKey,
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD || "",
        };

        // Sign updater zip
        execSync(`npx tauri signer sign "${zipPath}"`, { cwd: rootDir, env: signEnv, stdio: "pipe" });
        const zipSigPath = `${zipPath}.sig`;
        if (fs.existsSync(zipSigPath)) {
          const sigContent = fs.readFileSync(zipSigPath, "utf8").trim();

          // Generate latest.json
          const latestJson = {
            version: `v${version}`,
            notes: `Release v${version}`,
            pub_date: new Date().toISOString(),
            platforms: {
              "windows-x86_64": {
                signature: sigContent,
                url: `https://github.com/SeanDolan0/LocalLLmPanel/releases/download/v${version}/${zipName}`,
              },
            },
          };
          const latestJsonPath = path.join(distReleaseDir, "latest.json");
          fs.writeFileSync(latestJsonPath, JSON.stringify(latestJson, null, 2) + "\n", "utf8");
          console.log(`[Staging] Generated updater manifest: latest.json`);
        }

        // Also sign setup exe
        execSync(`npx tauri signer sign "${dest}"`, { cwd: rootDir, env: signEnv, stdio: "pipe" });
      } else {
        console.warn("\n[WARN] TAURI_SIGNING_PRIVATE_KEY not set. Skipping updater bundle signing and latest.json generation.");
      }
    }
  }

  // Find MSI output if --all was requested
  if (isAll) {
    const msiDir = path.join(tauriDir, "target", "release", "bundle", "msi");
    if (fs.existsSync(msiDir)) {
      const matchingMsi = fs.readdirSync(msiDir).find(f => f.endsWith(".msi") && f.includes(version));
      if (matchingMsi) {
        const src = path.join(msiDir, matchingMsi);
        const destName = `Local.LLM.Panel_${version}_x64_en-US.msi`;
        const dest = path.join(distReleaseDir, destName);
        copyArtifact(src, dest);
        const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
        stagedFiles.push({ name: destName, type: "WiX MSI Package", size: `${sizeMb} MB`, path: dest });
      }
    }
  }
}

// Write deterministic checksums next to every staged artifact.  The release
// validator requires these files so a successful build cannot be mistaken for
// a complete, publishable release.
for (const item of stagedFiles) {
  const digest = crypto.createHash("sha256").update(fs.readFileSync(item.path)).digest("hex");
  fs.writeFileSync(`${item.path}.sha256`, `${digest}  ${item.name}\n`);
}

execFileSync(
  process.execPath,
  [path.join(rootDir, "scripts", "validate-release.mjs"), isAll ? "--all" : isFast ? "--fast" : "--setup", "--artifacts"],
  { cwd: rootDir, stdio: "inherit" },
);

const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);

console.log("\n==================================================");
console.log(`  Build Completed Successfully in ${elapsedSec}s!`);
console.log("==================================================\n");
console.log("Staged Artifacts in dist-release/:");
for (const item of stagedFiles) {
  console.log(`  - [${item.type}] ${item.name} (${item.size})`);
}
console.log("\nReady for testing or deployment.\n");
