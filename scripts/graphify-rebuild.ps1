<#
.SYNOPSIS
  Rebuild the graphify knowledge graph for this repo using a LOCAL LLM (no cloud API).

.DESCRIPTION
  Starts (or reuses) the vLLM server in the local-llm-panel-ubuntu WSL2 distro and
  runs graphify's headless extract against it. Nothing leaves the machine: the
  semantic pass over docs is served by Qwen2.5-Coder-7B-Instruct-AWQ on localhost.

  Two env vars are required and are NOT the ones you would guess:

    GRAPHIFY_MAX_OUTPUT_TOKENS
      graphify's openai backend hardcodes max_tokens=16384 (graphify/llm.py:116),
      assuming that is the model's whole context window. Against a server whose
      context is the same size, every request is rejected with
      "you requested 16384 output tokens and your prompt contains N characters"
      and the semantic pass silently returns ZERO nodes. Capping output is what
      makes the backend usable against a self-hosted server at all.

    token-budget
      The default 60000 also overflows the context, triggering bisection down to
      "dropping" whole files. 8000 leaves room for the ~2.5k-token extraction
      spec plus the model's JSON output.

  The vLLM launch script needs two fixes of its own (CUDA_HOME pointing at the
  nvidia-cu13 pip wheel, and VLLM_USE_FLASHINFER_SAMPLER=0). Both live in
  ~/vllm-graphify.sh inside the distro.

.PARAMETER SkipServer
  Assume a server is already listening on port 8000.

.EXAMPLE
  ./scripts/graphify-rebuild.ps1
  ./scripts/graphify-rebuild.ps1 -SkipServer -Force
#>
[CmdletBinding()]
param(
  [switch]$SkipServer,
  [switch]$Force
)

# Deliberately NOT 'Stop' - graphify writes benign warnings to stderr, and under
# 'Stop' PowerShell turns any native stderr line into a terminating error, aborting
# a healthy rebuild. Failures are caught via $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$Distro = 'local-llm-panel-ubuntu'
$Model = 'qwen25-coder-7b'
$BaseUrl = 'http://localhost:8000/v1'

Set-Location (Join-Path $PSScriptRoot '..')

function Test-Server {
  try {
    $r = Invoke-RestMethod "$BaseUrl/models" -TimeoutSec 3
    return [bool]$r.data
  } catch { return $false }
}

if (-not $SkipServer -and -not (Test-Server)) {
  Write-Host "Starting vLLM in $Distro (first boot loads the model, ~90s)..." -ForegroundColor Cyan
  # The trailing `sleep 2` is load-bearing: wsl.exe returns as soon as bash has
  # backgrounded the job, and WSL tears the session down on exit, killing the
  # process before setsid finishes detaching it.
  wsl.exe -d $Distro -- bash -lc 'setsid nohup ~/vllm-graphify.sh > ~/vllm-graphify.log 2>&1 < /dev/null & disown; sleep 2' 2>$null | Out-Null

  $deadline = (Get-Date).AddMinutes(8)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    if (Test-Server) { break }
  }
  if (-not (Test-Server)) {
    throw "vLLM did not come up. Check: wsl -d $Distro -- bash -lc 'tail -40 ~/vllm-graphify.log'"
  }
  Write-Host "vLLM ready." -ForegroundColor Green
}

$env:OPENAI_BASE_URL = $BaseUrl
$env:OPENAI_API_KEY = 'local'          # local servers accept any non-empty value
$env:OPENAI_MODEL   = $Model
$env:GRAPHIFY_MAX_OUTPUT_TOKENS = '8192'

$args = @(
  'extract', '.',
  '--backend', 'openai',
  '--model', $Model,
  '--token-budget', '8000',
  '--max-concurrency', '2'
)
if ($Force) { $args += '--force' }

graphify @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# cluster-only re-clusters, and the community set is not perfectly stable across
# runs, so it discards saved labels when the set shifts. Name the communities,
# then let _finalize_labels.py correct the module-accurate names.
graphify label . --backend=openai --model=$Model --max-concurrency=2 --batch-size=40
& (Get-Content graphify-out/.graphify_python) graphify-out/_finalize_labels.py

Write-Host ""
Write-Host "Done. Open graphify-out/graph.html" -ForegroundColor Green
