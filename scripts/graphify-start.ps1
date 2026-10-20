<#
.SYNOPSIS
  Bring the graphify knowledge graph up to date and open it. Run at the start of a work session.

.DESCRIPTION
  Checks what actually changed since the last build and does only the work that is
  required, because the two halves of graphify have very different costs:

    * Code changes  -> tree-sitter AST pass. No LLM, no server, ~4s.
    * Doc changes   -> semantic pass. Needs a local LLM.

  The local LLM is vLLM in WSL2 holding ~10 GB of the 12 GB VRAM on this machine, so
  this script starts it ONLY when a tracked doc has actually changed. On a normal
  code-only morning it never starts, and your GPU stays free.

  Staleness is computed from graphify-out/manifest.json, which records an mtime per
  tracked file (docs additionally carry a semantic_hash). That is more reliable than
  graphify's own `check-update`, which only sees changes made through a git hook -
  an uncommitted doc edit would otherwise go unnoticed.

.PARAMETER Stop
  Shut down the local vLLM server and free the VRAM. Nothing else runs.

.PARAMETER Status
  Report graph/server state and exit. Touches nothing.

.PARAMETER Force
  Full rebuild including the semantic pass, even if nothing changed.

.PARAMETER NoOpen
  Do not launch the browser.

.EXAMPLE
  ./scripts/graphify-start.ps1
  ./scripts/graphify-start.ps1 -Status
  ./scripts/graphify-start.ps1 -Stop
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$Status,
  [switch]$Force,
  [switch]$NoOpen
)

# Deliberately NOT 'Stop'. graphify writes progress and benign warnings to stderr
# (e.g. "2 source file(s) produced zero nodes"), and under 'Stop' PowerShell turns
# any native stderr line into a terminating error - which aborted a healthy rebuild.
# Real failures are caught explicitly via exit codes in Invoke-Native and try/catch.
$ErrorActionPreference = 'Continue'

$Distro    = 'local-llm-panel-ubuntu'
$Model     = 'qwen25-coder-7b'
$BaseUrl   = 'http://localhost:8000/v1'
$OutDir    = 'graphify-out'
$GraphJson = Join-Path $OutDir 'graph.json'
$StateFile = Join-Path $OutDir '.graphify-start-state.json'
$GraphHtml = Join-Path $OutDir 'graph.html'

Set-Location (Join-Path $PSScriptRoot '..')

# --- helpers -----------------------------------------------------------------

function Invoke-Native {
  # Run a native command, capture stdout+stderr, and only surface them on failure.
  # graphify is chatty on stderr even when it succeeds, so output is suppressed by
  # default and the caller can re-run verbosely if something actually broke.
  param(
    [Parameter(Mandatory)][string]$Exe,
    [string[]]$NativeArgs = @(),
    [switch]$Quiet
  )
  $out = & $Exe @NativeArgs 2>&1
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    $tail = ($out | Select-Object -Last 12) -join "`n"
    throw "$Exe failed (exit $code).`n$tail"
  }
  if (-not $Quiet) { $out | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray } }
  return $out
}

function Test-VllmRunning {
  try { return [bool](Invoke-RestMethod "$BaseUrl/models" -TimeoutSec 3).data }
  catch { return $false }
}

function Get-VllmProcs {
  # No `bash -lc '...'` wrapper on purpose. Windows PowerShell 5.1 mangles nested
  # quotes when passing arguments to a native exe and silently returns EMPTY here,
  # which made -Stop report "not running" while the server was alive. Invoking pgrep
  # directly through wsl.exe with a single quoted argument works on both 5.1 and 7.
  # pgrep never matches its own PID, so the pattern cannot self-match.
  $out = wsl.exe -d $Distro -- pgrep -c -f 'vllm serve' 2>$null
  $n = 0
  if ($out -and ($out | Select-Object -Last 1) -match '^\s*(\d+)') { $n = [int]$Matches[1] }
  return $n
}

function Stop-Vllm {
  wsl.exe -d $Distro -- pkill -f 'vllm serve' 2>$null | Out-Null
  # Wait for the process to actually exit. vLLM shuts down asynchronously and holds
  # GPU memory plus port 8000 for several seconds after SIGTERM; returning early
  # lets a subsequent start race the dying process and fail to bind the port.
  $deadline = (Get-Date).AddSeconds(45)
  while ((Get-Date) -lt $deadline) {
    if ((Get-VllmProcs) -eq 0) { break }
    Start-Sleep -Seconds 2
  }
  Start-Sleep -Seconds 2
}

function Start-Vllm {
  # Never start on top of a dying instance.
  if ((Get-VllmProcs) -gt 0) { Stop-Vllm }

  Write-Host "  vLLM: starting $Model in $Distro (first boot loads the model, ~90s)" -ForegroundColor Cyan
  # The trailing `sleep 2` is load-bearing. wsl.exe returns as soon as bash has
  # backgrounded the job, and WSL tears the session down on exit - which kills the
  # process before setsid finishes detaching it. Holding the invocation open a
  # moment lets it genuinely detach. Symptom when missing: an empty
  # ~/vllm-graphify.log and no vllm process.
  wsl.exe -d $Distro -- bash -lc 'setsid nohup ~/vllm-graphify.sh > ~/vllm-graphify.log 2>&1 < /dev/null & disown; sleep 2' 2>$null | Out-Null
  # (no nested quotes inside the bash -lc payload, so this survives PS 5.1 too)
  $deadline = (Get-Date).AddMinutes(8)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    if (Test-VllmRunning) { Write-Host '  vLLM: ready' -ForegroundColor Green; return $true }
  }
  $tail = wsl.exe -d $Distro -- bash -lc 'tail -n 12 ~/vllm-graphify.log' 2>$null
  throw "vLLM did not start. Last log lines:`n$($tail -join "`n")"
}

function Get-GraphStats {
  if (-not (Test-Path $GraphJson)) { return $null }
  try {
    $g = Get-Content $GraphJson -Raw | ConvertFrom-Json
    # Largest communities first - that is what makes them the "hubs".
    $sizes = @{}
    foreach ($n in $g.nodes) {
      $c = $n.community_name
      if ($c) { $sizes[$c] = 1 + [int]$sizes[$c] }
    }
    $hubs = @($sizes.GetEnumerator() | Sort-Object Value -Descending |
              Select-Object -First 6 | ForEach-Object { "$($_.Key) ($($_.Value))" })
    return [pscustomobject]@{
      Nodes = $g.nodes.Count
      Edges = $g.links.Count
      Hubs  = $hubs
      Stale = (Get-Item $GraphJson).LastWriteTime
    }
  } catch { return $null }
}

function Get-StaleDocs {
  # Returns the doc files whose CONTENT differs from the last successful semantic
  # pass, plus a list of docs never seen before.
  #
  # This deliberately does NOT use graphify's manifest mtimes. When a doc's semantic
  # pass yields no usable nodes, graphify never records it in the manifest, so it
  # looks permanently changed and gets re-dispatched on every single run - the
  # 10 GB server would start forever. Content hashing against a local state file is
  # idempotent, and still notices a genuine edit.
  $seen = @{}
  if (Test-Path $StateFile) {
    try {
      $prev = Get-Content $StateFile -Raw | ConvertFrom-Json
      foreach ($p in $prev.PSObject.Properties) { $seen[$p.Name] = $p.Value }
    } catch { $seen = @{} }
  }

  $py = (Get-Content (Join-Path $OutDir '.graphify_python') -Raw).Trim()
  $docs = @(& $py (Join-Path $OutDir '_list_docs.py') 2>$null) | Where-Object { $_ -and $_ -notmatch '^\s' }
  if (-not $docs) { return @{ Changed = @(); State = $seen } }

  $changed = [System.Collections.Generic.List[string]]::new()
  $next = @{}
  foreach ($rel in $docs) {
    if (-not (Test-Path -LiteralPath $rel)) { continue }
    $h = (Get-FileHash -LiteralPath $rel -Algorithm SHA256).Hash
    $next[$rel] = $h
    if (-not $seen.ContainsKey($rel) -or $seen[$rel] -ne $h) { $changed.Add($rel) }
  }
  return @{ Changed = $changed; State = $next }
}

function Save-DocState {
  param($State)
  $json = [ordered]@{}
  foreach ($k in ($State.Keys | Sort-Object)) { $json[$k] = $State[$k] }
  $json | ConvertTo-Json | Set-Content -Path $StateFile -Encoding utf8
}

function Write-Status {
  $s = Get-GraphStats
  $vllm = if (Test-VllmRunning) { 'running' } else { 'stopped' }
  Write-Host ''
  Write-Host 'graphify' -ForegroundColor White
  Write-Host "  graph     : $(if ($s) { "$($s.Nodes) nodes, $($s.Edges) edges" } else { 'not built' })"
  if ($s) { Write-Host "  built     : $($s.Stale)" }
  Write-Host "  vLLM      : $vllm"
  Write-Host "  hooks     : $(if ((graphify hook status 2>&1 | Out-String) -match 'installed|ok|yes') { 'installed' } else { 'NOT installed' })"
  if ($s -and $s.Hubs) {
    Write-Host "  top hubs  : $(($s.Hubs | Select-Object -First 5) -join ', ')"
  }
  Write-Host ''
  Write-Host '  query     : graphify query "how does X work?"' -ForegroundColor DarkGray
  Write-Host '  path      : graphify path "A" "B"' -ForegroundColor DarkGray
  Write-Host '  explain   : graphify explain "Symbol"' -ForegroundColor DarkGray
  Write-Host ''
}

# --- modes -------------------------------------------------------------------

if ($Stop) {
  if ((Get-VllmProcs) -gt 0) {
    Stop-Vllm
    if ((Get-VllmProcs) -gt 0) { Write-Host 'vLLM did not stop cleanly - try: wsl -d ' + $Distro + ' -- bash -lc ''pkill -9 -f "vllm[ ]serve"''' -ForegroundColor Yellow }
    else { Write-Host 'vLLM stopped - VRAM released.' -ForegroundColor Green }
  } else {
    Write-Host 'vLLM was not running.'
  }
  exit 0
}

if ($Status) { Write-Status; exit 0 }

if (-not (Get-Command graphify -ErrorAction SilentlyContinue)) {
  throw "graphify not found. Install with: uv tool install 'graphifyy[openai]'"
}

# --- work out what needs doing ----------------------------------------------

$hasGraph = Test-Path $GraphJson
$needFull = $Force -or -not $hasGraph

# Computed unconditionally (when the interpreter exists) so the state file is always
# refreshed after a build, including a -Force one.
$docState = $null
$needDocs = $false
if (Test-Path (Join-Path $OutDir '.graphify_python')) {
  $docState = Get-StaleDocs
  $needDocs = $docState.Changed.Count -gt 0
}

# Code is always refreshed: `graphify update` is ~4s when nothing changed, it keeps
# the manifest authoritative, and it catches new/deleted files for free. Tracking
# code changes in the state file would duplicate that for no benefit.
$needCode = $hasGraph

if (-not ($needFull -or $needCode -or $needDocs)) {
  Write-Host "Graph is current - nothing to rebuild." -ForegroundColor Green
  Write-Status
  if (-not $NoOpen -and (Test-Path $GraphHtml)) { Start-Process $GraphHtml }
  exit 0
}

Write-Host 'Building the graph...' -ForegroundColor White

# --- 1. code: AST only, fast, no LLM ----------------------------------------

if ($needCode) {
  Write-Host '  code: AST rebuild (no LLM)' -ForegroundColor Cyan
  Invoke-Native -Exe graphify -NativeArgs @('update', '.') -Quiet | Out-Null
}

# --- 2. docs: semantic pass, needs the local LLM ---------------------------

if ($needFull -or $needDocs) {
  if ($docState -and $docState.Changed.Count -gt 0) {
    Write-Host "  docs: $($docState.Changed.Count) changed - $(($docState.Changed | Select-Object -First 3) -join ', ')" -ForegroundColor Cyan
  } else {
    Write-Host '  docs: full semantic pass over all docs' -ForegroundColor Cyan
  }

  $startedVllm = $false
  if (-not (Test-VllmRunning)) { $startedVllm = $true; Start-Vllm | Out-Null }

  $env:OPENAI_BASE_URL = $BaseUrl
  $env:OPENAI_API_KEY = 'local'        # local servers accept any non-empty value
  $env:OPENAI_MODEL   = $Model
  # graphify's openai backend hardcodes max_tokens=16384 (graphify/llm.py:116),
  # assuming that is the model's whole context. Against an equally-sized server that
  # leaves no room for input, every request 400s, and the pass silently returns ZERO
  # nodes. Capping output is what makes a self-hosted backend usable at all.
  $env:GRAPHIFY_MAX_OUTPUT_TOKENS = '8192'

  $e = @('extract', '.', '--backend', 'openai', '--model', $Model,
         '--token-budget', '8000', '--max-concurrency', '2')
  if ($Force) { $e += '--force' }
  Invoke-Native -Exe graphify -NativeArgs $e -Quiet | Out-Null

  # cluster-only re-clusters and the community set is not bit-stable across runs,
  # so it drops saved labels when the set shifts. Name first, then correct.
  Invoke-Native -Exe graphify -NativeArgs @('label', '.', "--backend=openai",
    "--model=$Model", '--max-concurrency=2', '--batch-size=40') -Quiet | Out-Null
  $py = (Get-Content (Join-Path $OutDir '.graphify_python') -Raw).Trim()
  Invoke-Native -Exe $py -NativeArgs @((Join-Path $OutDir '_finalize_labels.py')) -Quiet | Out-Null

  if ($docState) { Save-DocState $docState.State }

  if ($startedVllm) {
    Write-Host ''
    Write-Host 'vLLM was started for the doc pass and is now idle. Free the VRAM with:' -ForegroundColor DarkGray
    Write-Host '  ./scripts/graphify-start.ps1 -Stop' -ForegroundColor DarkGray
  }
}

Write-Status
if (-not $NoOpen -and (Test-Path $GraphHtml)) { Start-Process $GraphHtml }
