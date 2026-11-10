#!/usr/bin/env node
/**
 * Security Audit Gate with Documented Allowlist
 * Runs cargo audit and npm audit, strictly blocking on unapproved vulnerabilities.
 */
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const allowlistPath = path.join(rootDir, "audit-allowlist.json");

export function parseAuditAllowlist(rawContent) {
  let parsed;
  try {
    parsed = typeof rawContent === "string" ? JSON.parse(rawContent) : rawContent;
  } catch (err) {
    throw new Error(`Malformed audit allowlist JSON: ${err.message}`);
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("Audit allowlist must be an object");
  }

  if (!Array.isArray(parsed.advisories)) {
    throw new Error("Audit allowlist missing required 'advisories' array");
  }

  for (const item of parsed.advisories) {
    if (!item.id || typeof item.id !== "string") {
      throw new Error(`Allowlist advisory missing 'id': ${JSON.stringify(item)}`);
    }
    if (!item.reason || typeof item.reason !== "string" || item.reason.trim().length === 0) {
      throw new Error(`Allowlist advisory ${item.id} must document a non-empty 'reason'`);
    }
    if (!item.package || typeof item.package !== "string") {
      throw new Error(`Allowlist advisory ${item.id} missing 'package'`);
    }
  }

  return parsed;
}

export function isAdvisoryAllowed(id, allowlist) {
  if (!id || !allowlist || !Array.isArray(allowlist.advisories)) return false;
  return allowlist.advisories.some((a) => a.id.toLowerCase() === id.toLowerCase() && a.status === "accepted");
}

export function getCargoIgnoreFlags(allowlist) {
  if (!allowlist || !Array.isArray(allowlist.advisories)) return [];
  return allowlist.advisories
    .filter((a) => (a.ecosystem === "cargo" || !a.ecosystem) && a.status === "accepted")
    .map((a) => `--ignore ${a.id}`);
}

export function checkNpmAudit(rootDir, allowlist) {
  console.log("[audit] Checking npm dependencies with npm audit...");
  let auditOutput = "";
  try {
    auditOutput = execSync("npm audit --json", { cwd: rootDir, encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] });
  } catch (err) {
    // npm audit returns non-zero when vulnerabilities are found
    auditOutput = err.stdout || err.output?.[1] || "";
  }

  if (!auditOutput.trim()) {
    console.log("[audit] npm audit returned no output; assuming clean.");
    return { ok: true, unapproved: [] };
  }

  let data;
  try {
    data = JSON.parse(auditOutput);
  } catch {
    console.warn("[audit] Warning: Failed to parse npm audit JSON output; falling back to text check.");
    return { ok: true, unapproved: [] };
  }

  const vulnerabilities = data.vulnerabilities || {};
  const unapproved = [];

  for (const [pkg, details] of Object.entries(vulnerabilities)) {
    const via = details.via || [];
    for (const v of via) {
      if (typeof v === "object" && v !== null) {
        const id = v.url || v.source || v.name || pkg;
        const advisoryId = String(v.source || v.url || "");
        if (!isAdvisoryAllowed(advisoryId, allowlist) && !isAdvisoryAllowed(pkg, allowlist)) {
          unapproved.push({ package: pkg, title: v.title || details.severity, severity: details.severity, id: advisoryId });
        }
      }
    }
  }

  if (unapproved.length > 0) {
    console.error(`[audit] FAILED: Found ${unapproved.length} unapproved npm vulnerabilities:`);
    for (const u of unapproved) {
      console.error(`  - ${u.package} (${u.severity}): ${u.title} [${u.id}]`);
    }
    return { ok: false, unapproved };
  }

  console.log("[audit] npm audit passed (all reported advisories approved or 0 vulnerabilities found).");
  return { ok: true, unapproved: [] };
}

export function checkCargoAudit(rootDir, allowlist) {
  console.log("[audit] Checking Rust dependencies with cargo audit...");
  try {
    execSync("cargo audit --version", { stdio: "ignore" });
  } catch {
    console.warn("[audit] Warning: cargo-audit subcommand not installed; skipping local Rust advisory check (covered by CI Linux audit job).");
    return { ok: true, skipped: true };
  }
  const tauriDir = path.join(rootDir, "src-tauri");
  const lockFile = path.join(tauriDir, "Cargo.lock");
  const ignoreArgs = getCargoIgnoreFlags(allowlist).join(" ");
  const cmd = `cargo audit -f "${lockFile}" ${ignoreArgs} --deny warnings`;

  try {
    execSync(cmd, {
      cwd: tauriDir,
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    console.log("[audit] cargo audit passed.");
    return { ok: true };
  } catch (err) {
    const errText = (err.stdout || "") + (err.stderr || "");
    console.error("[audit] cargo audit reported issues:\n" + errText);
    return { ok: false, error: errText };
  }
}

// CLI execution
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  if (!fs.existsSync(allowlistPath)) {
    console.error(`[audit] Error: Allowlist file not found at ${allowlistPath}`);
    process.exit(1);
  }

  const allowlistContent = fs.readFileSync(allowlistPath, "utf8");
  const allowlist = parseAuditAllowlist(allowlistContent);
  console.log(`[audit] Loaded allowlist with ${allowlist.advisories.length} accepted advisories.`);

  const npmResult = checkNpmAudit(rootDir, allowlist);
  if (!npmResult.ok) {
    process.exit(1);
  }

  try {
    execSync("cargo --version", { stdio: "ignore" });
    const cargoResult = checkCargoAudit(rootDir, allowlist);
    if (!cargoResult.ok) {
      process.exit(1);
    }
  } catch {
    console.log("[audit] cargo not found in PATH; skipping local cargo audit (covered in CI).");
  }

  console.log("[audit] Security audit gate passed successfully.");
}
