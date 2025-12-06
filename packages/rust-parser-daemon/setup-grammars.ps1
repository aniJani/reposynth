# Setup script for tree-sitter grammars (Windows PowerShell)
# Run this from the rust-parser-daemon directory

$ErrorActionPreference = "Stop"

$GRAMMAR_DIR = "grammars"
New-Item -ItemType Directory -Force -Path $GRAMMAR_DIR | Out-Null
Set-Location $GRAMMAR_DIR

Write-Host "=== Setting up Tree-Sitter Grammars ===" -ForegroundColor Cyan
Write-Host "These versions are compatible with tree-sitter 0.22 (LANGUAGE_VERSION 14)"
Write-Host ""

# Grammar versions - UPDATE THESE when changing tree-sitter version
$GRAMMARS = @{
    "tree-sitter-python"     = @{ Version = "v0.21.0"; Url = "https://github.com/tree-sitter/tree-sitter-python.git" }
    "tree-sitter-typescript" = @{ Version = "v0.21.2"; Url = "https://github.com/tree-sitter/tree-sitter-typescript.git" }
    "tree-sitter-css"        = @{ Version = "v0.21.1"; Url = "https://github.com/tree-sitter/tree-sitter-css.git" }
    "tree-sitter-scss"       = @{ Version = "v1.0.0";  Url = "https://github.com/serenadeai/tree-sitter-scss.git" }
    "tree-sitter-html"       = @{ Version = "v0.20.4"; Url = "https://github.com/tree-sitter/tree-sitter-html.git" }
}

foreach ($grammar in $GRAMMARS.Keys) {
    $version = $GRAMMARS[$grammar].Version
    $url = $GRAMMARS[$grammar].Url
    
    Write-Host "--- Setting up $grammar @ $version ---" -ForegroundColor Yellow
    
    if (Test-Path $grammar) {
        Write-Host "  Directory exists, updating..."
        Push-Location $grammar
        git fetch --tags
        git checkout $version 2>&1 | Out-Null
        Pop-Location
    } else {
        Write-Host "  Cloning..."
        git clone $url 2>&1 | Out-Null
        Push-Location $grammar
        git checkout $version 2>&1 | Out-Null
        Pop-Location
    }
    
    Write-Host "  ✓ $grammar ready" -ForegroundColor Green
    Write-Host ""
}

Set-Location ..

Write-Host "=== All grammars set up successfully! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now build the Rust parser daemon:"
Write-Host "  cargo build --release"
