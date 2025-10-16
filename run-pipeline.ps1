param(
  [Parameter(Mandatory = $true)]
  [string]$GitUrl,

  [Parameter(Mandatory = $false)]
  [ValidateSet("semantic", "hybrid")]
  [string]$Mode = "semantic"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# --- Configuration ---
$tempRepoDir = "temp_repos"
$orchestratorPackage = "packages.python-orchestrator.orchestrator"

# --- Pre-flight Checks: Find Virtual Environment ---
# Check common venv locations (matches bash script logic)
$venvCandidates = @(
  "packages\python-orchestrator\.venv",
  ".venv",
  "venv"
)

$venvPath = $null
foreach ($candidate in $venvCandidates) {
  $activateScript = Join-Path $candidate "Scripts\Activate.ps1"
  if (Test-Path $activateScript) {
    $venvPath = (Resolve-Path $candidate).Path
    break
  }
}

if (-not $venvPath) {
  throw "Python virtual environment not found. Please run setup as per README."
}

Write-Host "Using virtual environment: $venvPath"

# --- Main Logic ---
$repoName = [IO.Path]::GetFileNameWithoutExtension($GitUrl.TrimEnd('/').TrimEnd('.git'))
$clonePath = Join-Path $tempRepoDir $repoName

Write-Host "--- Starting RepoSynth Pipeline for: $GitUrl (Mode: $Mode) ---"

# 1) Clone Repository
Write-Host "[1/3] Cloning repository..."
if (Test-Path $clonePath) { Remove-Item $clonePath -Recurse -Force }
New-Item -ItemType Directory -Path $tempRepoDir -Force | Out-Null
git clone --depth=1 --no-tags --no-recurse-submodules $GitUrl $clonePath
if ($LASTEXITCODE -ne 0) { throw "Error: Failed to clone repository." }
Write-Host "Repository cloned to: $clonePath"

# 2) Activate Virtual Environment & Run Pipeline
Write-Host "[2/3] Running the analysis pipeline..."
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript

# Run pipeline with arguments directly (no environment variables needed)
python -m $orchestratorPackage --repo $clonePath --mode $Mode
$code = $LASTEXITCODE

# Deactivate if available
if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }

if ($code -ne 0) { throw "Error: Python pipeline failed with exit code $code" }

Write-Host "Pipeline completed successfully."

# 3) Locate Final Pack
Write-Host "[3/3] Locating final pack..."
$packDir = "pack"
if (Test-Path $packDir) {
  Write-Host ""
  Write-Host "--- ✅ Pipeline Finished Successfully! ---"
  Write-Host "The complete '$Mode' pack has been generated in the '$packDir' directory."
  Write-Host "Main summary: $packDir\repoBrief.md"
} else {
  throw "Error: Output pack directory '$packDir' not found."
}