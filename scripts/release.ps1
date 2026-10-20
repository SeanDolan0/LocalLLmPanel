param(
    [Parameter(Mandatory=$true)]
    [string]$Version,

    [Parameter(Mandatory=$false)]
    [ValidateSet('setup', 'all')]
    [string]$BuildType = 'setup',

    [Parameter(Mandatory=$false)]
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Error($msg) { Write-Host "!! $msg" -ForegroundColor Red; exit 1 }

Write-Step "Starting release v$Version (build: $BuildType)"

$repoRoot = "C:\Users\sedol\Documents\LocalLLmPanel"
Set-Location $repoRoot

Write-Step "Running preflight checks..."
npm run check
if (-not $SkipTests) {
    cd src-tauri; cargo test; cd ..
}

Write-Step "Updating version in 4 files..."
$files = @(
    "package.json",
    "package-lock.json",
    "src-tauri/Cargo.toml",
    "src-tauri/tauri.conf.json"
)

foreach ($file in $files) {
    $content = Get-Content $file -Raw
    if ($file -eq "package.json" -or $file -eq "package-lock.json") {
        $content = $content -replace '"version"\s*:\s*"[\d.]+ "', "`"version`": `"$Version`""
    } elseif ($file -eq "src-tauri/Cargo.toml") {
        $content = $content -replace 'version\s*=\s*"[\d.]+ "', "version = `"$Version`""
    } elseif ($file -eq "src-tauri/tauri.conf.json") {
        $content = $content -replace '"version"\s*:\s*"[\d.]+ "', "`"version`": `"$Version`""
    }
    Set-Content $file -Value $content -Encoding UTF8
    Write-Host "  Updated $file"
}

Write-Step "Building..."
npm run "build:$BuildType"

Write-Step "Committing and tagging..."
git add package.json package-lock.json src-tauri/Cargo.toml src-tauri/tauri.conf.json src-tauri/Cargo.lock dist-release/
git commit -m "release: v$Version"
git tag -a "v$Version" -m "Release v$Version"

Write-Step "Pushing to GitHub..."
git push origin main
git push origin "v$Version"

Write-Step "Creating GitHub release..."
$artifacts = Get-ChildItem "dist-release" -File
if (-not $artifacts) { Write-Error "No artifacts found in dist-release/" }
gh release create "v$Version" dist-release/* --title "v$Version" --generate-notes

Write-Host "Release v$Version complete!" -ForegroundColor Green