<#
.SYNOPSIS
    Local LLM Panel - Release Manager TUI
.DESCRIPTION
    Interactive release tool with modern TUI interface (PowerShell 7+ compatible)
#>

$ErrorActionPreference = 'Stop'

# Force UTF-8 output encoding for proper Unicode rendering
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# ===== CONFIG =====
$RepoRoot = "C:\Users\sedol\Documents\LocalLLmPanel"
Set-Location $RepoRoot

$VersionFiles = @(
    @{ Path = "package.json";           Pattern = '"version"\s*:\s*"[\d.]+ "';          Replacement = '"version": "' },
    @{ Path = "package-lock.json";      Pattern = '"version"\s*:\s*"[\d.]+ "';          Replacement = '"version": "' },
    @{ Path = "src-tauri/Cargo.toml";   Pattern = 'version\s*=\s*"[\d.]+ "';            Replacement = 'version = "' },
    @{ Path = "src-tauri/tauri.conf.json"; Pattern = '"version"\s*:\s*"[\d.]+ "';       Replacement = '"version": "' }
)

$BuildTypes = @(
    @{ Id = 'setup'; Label = 'Standard Release (NSIS installer)';      Time = '~40s' },
    @{ Id = 'all';   Label = 'Full Distribution (NSIS + MSI installers)'; Time = '~2-3m' }
)

# ===== COLORS =====
$ColPrimary   = 'Cyan'
$ColSecondary = 'DarkCyan'
$ColAccent    = 'Yellow'
$ColSuccess   = 'Green'
$ColError     = 'Red'
$ColWarning   = 'DarkYellow'
$ColMuted     = 'DarkGray'
$ColHighlight = 'White'
$ColDim       = 'Gray'

# ===== ASCII/UNICODE DETECTION =====
$UseUnicode = $true
try {
    $enc = [Console]::OutputEncoding
    if ($enc.CodePage -ne 65001) { $UseUnicode = $false }
} catch { $UseUnicode = $false }

if ($UseUnicode) {
    $Box = @{ TL='┌'; TR='┐'; BL='└'; BR='┘'; H='─'; V='│'; LR='├'; RR='┤'; }
    $Icons = @{ Pending='○'; Running='◐'; Done='✓'; Err='✗'; Skip='⊘'; Bullet='•'; Arrow='▸'; }
} else {
    $Box = @{ TL='+'; TR='+'; BL='+'; BR='+'; H='-'; V='|'; LR='+'; RR='+'; }
    $Icons = @{ Pending='[ ]'; Running='[>]'; Done='[x]'; Err='[!]'; Skip='[-]'; Bullet='*'; Arrow='>'; }
}

function Write-Header {
    param($Title, $Subtitle = '')
    $width = 70
    $line = $Box.H * $width
    Write-Host ""
    Write-Host "  $line" -ForegroundColor $ColPrimary
    Write-Host "  $Title" -ForegroundColor $ColHighlight
    if ($Subtitle) { Write-Host "  $Subtitle" -ForegroundColor $ColMuted }
    Write-Host "  $line" -ForegroundColor $ColPrimary
    Write-Host ""
}

function Write-Box {
    param($Title, $Content, $Color = 'Cyan')
    $width = 66
    $inner = $width - 4
    $tl = "  $($Box.TL)$($Box.H * $width)$($Box.TR)"
    $bl = "  $($Box.BL)$($Box.H * $width)$($Box.BR)"
    $header = "  $($Box.V) $Title$(' ' * ($inner - $Title.Length)) $($Box.V)"
    $sep  = "  $($Box.LR)$($Box.H * $width)$($Box.RR)"
    
    Write-Host $tl -ForegroundColor $Color
    Write-Host $header -ForegroundColor $Color
    Write-Host $sep -ForegroundColor $Color
    foreach ($line in $Content) {
        $padded = $line.PadRight($inner)
        Write-Host "  $($Box.V) $padded $($Box.V)" -ForegroundColor $Color
    }
    Write-Host $bl -ForegroundColor $Color
    Write-Host ""
}

function Write-Step {
    param($Index, $Total, $Title, $Status = 'pending', $Detail = '')
    $icon = $Icons.$Status
    $stepColors = @{ pending = $ColMuted; running = $ColAccent; done = $ColSuccess; error = $ColError; skip = $ColWarning }
    $color = $stepColors[$Status]
    $barWidth = 28
    $filled = [math]::Floor(($Index / $Total) * $barWidth)
    $bar = ('█' * $filled) + ('░' * ($barWidth - $filled))
    if (-not $UseUnicode) { $bar = ('#' * $filled) + ('.' * ($barWidth - $filled)) }
    $pct = [math]::Round(($Index / $Total) * 100)
    Write-Host "  $icon [$bar] $pct%  $Title" -ForegroundColor $color -NoNewline
    if ($Detail) { Write-Host "  $Detail" -ForegroundColor $ColDim } else { Write-Host "" }
}

function Write-Info { param($Msg) Write-Host "  $($Icons.Bullet) $Msg" -ForegroundColor $ColSecondary }
function Write-Success { param($Msg) Write-Host "  $($Icons.Done) $Msg" -ForegroundColor $ColSuccess }
function Write-Warn { param($Msg) Write-Host "  $($Icons.Skip) $Msg" -ForegroundColor $ColWarning }
function Write-ErrorMsg { param($Msg) Write-Host "  $($Icons.Err) $Msg" -ForegroundColor $ColError }
function Write-Dim { param($Msg) Write-Host "  $Msg" -ForegroundColor $ColDim }

function Read-Choice {
    param($Prompt, $Options, $Default = 0)
    Write-Host ""
    Write-Host "  $Prompt" -ForegroundColor $ColHighlight
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $opt = $Options[$i]
        $marker = if ($i -eq $Default) { $Icons.Arrow } else { ' ' }
        $fgColor = if ($i -eq $Default) { $ColAccent } else { $ColPrimary }
        Write-Host "  $marker [$($i + 1)] $($opt.Label)  $($ColDim)$($opt.Time)" -ForegroundColor $fgColor
    }
    Write-Host ""
    Write-Host -NoNewline "  Choice: " -ForegroundColor $ColAccent
    $input = Read-Host
    if ($input -match '^\d+$') {
        $idx = [int]$input - 1
        if ($idx -ge 0 -and $idx -lt $Options.Count) { return $Options[$idx] }
    }
    return $Options[$Default]
}

function Read-Input {
    param($Prompt, $Default = '', $Validator = $null)
    while ($true) {
        Write-Host -NoNewline "  $Prompt" -ForegroundColor $ColHighlight
        if ($Default) { Write-Host -NoNewline " [$Default]" -ForegroundColor $ColMuted }
        Write-Host -NoNewline ": " -ForegroundColor $ColAccent
        $input = Read-Host
        if (-not $input -and $Default) { $input = $Default }
        if ($Validator -and -not (& $Validator $input)) {
            Write-ErrorMsg "Invalid input, try again"
            continue
        }
        return $input
    }
}

function Confirm-Action {
    param($Message, $Default = $true)
    $suffix = if ($Default) { '[Y/n]' } else { '[y/N]' }
    Write-Host -NoNewline "  $Message ${suffix}: " -ForegroundColor $ColAccent
    $input = Read-Host
    if (-not $input) { return $Default }
    return $input -match '^[yY]'
}

function Run-Command {
    param($Command, $WorkingDir = $RepoRoot, $ShowOutput = $false)
    
    try {
        $processInfo = @{
            FilePath = $Command[0]
            ArgumentList = $Command[1..($Command.Count - 1)]
            WorkingDirectory = $WorkingDir
            RedirectStandardOutput = $true
            RedirectStandardError = $true
            UseShellExecute = $false
            CreateNoWindow = $true
        }
        
        $process = [System.Diagnostics.Process]::Start($processInfo)
        $output = $process.StandardOutput.ReadToEnd()
        $errorOutput = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        
        $exitCode = $process.ExitCode
        $result = $output + $errorOutput
        
        if ($ShowOutput -and $result) { Write-Host $result }
        
        if ($exitCode -ne 0) { throw "Exit code $exitCode`n$result" }
        return @{ Success = $true; Output = $result; ExitCode = $exitCode }
    } catch {
        return @{ Success = $false; Error = $_.ToString(); ExitCode = -1 }
    }
}

# ===== MAIN TUI =====
function Show-MainMenu {
    Clear-Host
    Write-Header "LOCAL LLM PANEL" "Release Manager v1.0"
    
    # Show current version
    try {
        $pkg = Get-Content "$RepoRoot\package.json" -Raw | ConvertFrom-Json -ErrorAction Stop
        $currentVer = $pkg.version
    } catch {
        $currentVer = "0.0.0"
        Write-Warn "Could not read current version from package.json"
    }
    Write-Box "CURRENT VERSION" @("  $currentVer"), 'DarkCyan'
    
    $steps = @(
        @{ Name = 'Preflight Checks';     Fn = { Invoke-Preflight } },
        @{ Name = 'Version Bump';         Fn = { Invoke-VersionBump } },
        @{ Name = 'Build Artifacts';      Fn = { Invoke-Build } },
        @{ Name = 'Git Commit & Tag';     Fn = { Invoke-Git } },
        @{ Name = 'Push & GitHub Release';Fn = { Invoke-Release } }
    )
    
    # Build type selection
    $buildChoice = Read-Choice "Select build type:" $BuildTypes -Default 0
    $buildType = $buildChoice.Id
    $buildLabel = $buildChoice.Label
    
    # Version input
    $version = Read-Input "New version" $currentVer {
        param($v) $v -match '^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
    }
    
    # Confirm
    Write-Box "RELEASE SUMMARY" @(
        "  Version:      $version",
        "  Build Type:   $buildLabel",
        "  Repo:         Local LLM Panel"
    ), 'Yellow'
    
    if (-not (Confirm-Action "Proceed with release?")) {
        Write-Warn "Release cancelled"
        return
    }
    
    # Execute pipeline
    Clear-Host
    Write-Header "RELEASE PIPELINE" "v$version • $buildLabel"
    
    for ($i = 0; $i -lt $steps.Count; $i++) {
        $step = $steps[$i]
        Write-Step ($i + 1) $steps.Count $step.Name -Status 'running'
        
        try {
            $result = & $step.Fn -Version $version -BuildType $buildType
            Write-Step ($i + 1) $steps.Count $step.Name -Status 'done' -Detail $result.Detail
        } catch {
            Write-Step ($i + 1) $steps.Count $step.Name -Status 'error' -Detail "Failed: $_"
            Write-ErrorMsg "Pipeline failed at step: $($step.Name)"
            Write-Dim $_.ToString()
            return
        }
    }
    
    Write-Host ""
    Write-Box "RELEASE COMPLETE" @(
        "  Version $version released successfully!",
        "  GitHub release created with artifacts."
    ), 'Green'
}

# ===== PIPELINE STEPS =====
function Invoke-Preflight {
    param($Version, $BuildType)
    $results = @()
    
    Write-Info "Running npm run check..."
    $r = Run-Command @('npm', 'run', 'check')
    if (-not $r.Success) { throw "Preflight check failed: $($r.Error)" }
    $results += "TypeScript + Cargo check passed"
    
    Write-Info "Running cargo test..."
    $r = Run-Command @('cargo', 'test'), "$RepoRoot\src-tauri"
    if (-not $r.Success) { throw "Unit tests failed: $($r.Error)" }
    $results += "All unit tests passed"
    
    return @{ Detail = ($results -join '; ') }
}

function Invoke-VersionBump {
    param($Version, $BuildType)
    $updated = @()
    foreach ($f in $VersionFiles) {
        $path = "$RepoRoot\$($f.Path)"
        if (-not (Test-Path $path)) { throw "File not found: $path" }
        $content = Get-Content $path -Raw
        $newContent = $content -replace $f.Pattern, ('{0}{1}"' -f $f.Replacement, $Version)
        Set-Content $path -Value $newContent -Encoding UTF8
        $updated += $f.Path
    }
    return @{ Detail = "Updated: $($updated -join ', ')" }
}

function Invoke-Build {
    param($Version, $BuildType)
    Write-Info "Running npm run build:$BuildType..."
    $r = Run-Command @('npm', 'run', "build:$BuildType")
    if (-not $r.Success) { throw "Build failed: $($r.Error)" }
    
    $artifacts = Get-ChildItem "$RepoRoot\dist-release" -File -ErrorAction SilentlyContinue
    $names = $artifacts.Name -join ', '
    if (-not $names) { $names = 'No artifacts found (check dist-release/)' }
    return @{ Detail = "Artifacts: $names" }
}

function Invoke-Git {
    param($Version, $BuildType)
    $files = "package.json", "package-lock.json", "src-tauri/Cargo.toml", "src-tauri/tauri.conf.json", "src-tauri/Cargo.lock", "dist-release/"
    Write-Info "Staging files..."
    Run-Command @('git', 'add') + $files
    Write-Info "Committing..."
    Run-Command @('git', 'commit', '-m', "release: v$Version")
    Write-Info "Tagging..."
    Run-Command @('git', 'tag', '-a', "v$Version", '-m', "Release v$Version")
    return @{ Detail = "Committed and tagged v$Version" }
}

function Invoke-Release {
    param($Version, $BuildType)
    Write-Info "Pushing to origin main..."
    Run-Command @('git', 'push', 'origin', 'main')
    Write-Info "Pushing tag v$Version..."
    Run-Command @('git', 'push', 'origin', "v$Version")
    
    $artifacts = Get-ChildItem "$RepoRoot\dist-release" -File
    if (-not $artifacts) { throw "No artifacts in dist-release/" }
    $artifactPaths = $artifacts.FullName
    Write-Info "Creating GitHub release..."
    $r = Run-Command @('gh', 'release', 'create', "v$Version") + $artifactPaths + @('--title', "v$Version", '--generate-notes')
    if (-not $r.Success) { throw "GitHub release failed: $($r.Error)" }
    return @{ Detail = "GitHub release created: v$Version" }
}

# ===== ENTRY =====
try {
    Show-MainMenu
} catch {
    Write-ErrorMsg "Fatal error: $_"
    Write-Dim $_.ScriptStackTrace
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}