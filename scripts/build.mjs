#!/usr/bin/env node
import { execSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const tauriDir = path.join(rootDir, "src-tauri");
const distReleaseDir = path.join(rootDir, "dist-release");

// 1. Parse arguments
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

if (cargoVersionMatch && cargoVersionMatch[1] !== version) {
  console.warn(`[WARN] Version mismatch in Cargo.toml: ${cargoVersionMatch[1]} vs package.json: ${version}`);
}
if (tauriConf.version !== version) {
  console.warn(`[WARN] Version mismatch in tauri.conf.json: ${tauriConf.version} vs package.json: ${version}`);
}

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
    console.warn("[WARN] lld-link.exe not found in PATH or standard LLVM directories.");
    console.warn("[WARN] Falling back to default MSVC link.exe (linking may be slower).");
    env.CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "link.exe";
  } else {
    console.log("[Toolchain] High-speed lld-link.exe enabled.");
  }

  return env;
}

const buildEnv = configureEnvironment();
const startTime = Date.now();

// 4. Check mode
if (isCheckOnly) {
  console.log("\n[Preflight] Running TypeScript check...");
  execSync("npm run build", { cwd: rootDir, env: buildEnv, stdio: "inherit" });
  console.log("\n[Preflight] Running Cargo check...");
  execSync("cargo check", { cwd: tauriDir, env: buildEnv, stdio: "inherit" });
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
  fs.copyFileSync(rawExePath, stagedExePath);
  const sizeMb = (fs.statSync(stagedExePath).size / (1024 * 1024)).toFixed(2);
  stagedFiles.push({ name: stagedExeName, type: "Standalone Executable", size: `${sizeMb} MB`, path: stagedExePath });

  const altExeName = `Local.LLM.Panel_${version}_x64-standalone.exe`;
  const altExePath = path.join(distReleaseDir, altExeName);
  fs.copyFileSync(rawExePath, altExePath);
  stagedFiles.push({ name: altExeName, type: "Standalone Executable (alias)", size: `${sizeMb} MB`, path: altExePath });
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
      fs.copyFileSync(src, dest);
      const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
      stagedFiles.push({ name: destName, type: "NSIS Installer", size: `${sizeMb} MB`, path: dest });
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
        fs.copyFileSync(src, dest);
        const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
        stagedFiles.push({ name: destName, type: "WiX MSI Package", size: `${sizeMb} MB`, path: dest });
      }
    }
  }
}

const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);

console.log("\n==================================================");
console.log(`  Build Completed Successfully in ${elapsedSec}s!`);
console.log("==================================================\n");
console.log("Staged Artifacts in dist-release/:");
for (const item of stagedFiles) {
  console.log(`  - [${item.type}] ${item.name} (${item.size})`);
}
console.log("\nReady for testing or deployment.\n");
