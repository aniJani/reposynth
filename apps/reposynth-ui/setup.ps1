# Week 9 Frontend - Quick Setup Script
# Run this script to install dependencies and start the frontend

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  RepoSynth Week 9 Frontend Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
$currentDir = Get-Location
$targetDir = "E:\SummerProjects\reposynth\apps\reposynth-ui"

if ($currentDir.Path -ne $targetDir) {
    Write-Host "📁 Navigating to frontend directory..." -ForegroundColor Yellow
    Set-Location $targetDir
}

# Check if node is installed
Write-Host "🔍 Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Node.js is not installed!" -ForegroundColor Red
    Write-Host "Please install Node.js 18+ from https://nodejs.org" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Node.js installed: $nodeVersion" -ForegroundColor Green

# Check if npm is installed
$npmVersion = npm --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm is not installed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ npm installed: v$npmVersion" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 2-3 minutes..." -ForegroundColor Gray
Write-Host ""

npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ npm install failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try these fixes:" -ForegroundColor Yellow
    Write-Host "  1. Clear npm cache: npm cache clean --force" -ForegroundColor Gray
    Write-Host "  2. Delete node_modules: Remove-Item -Recurse -Force node_modules" -ForegroundColor Gray
    Write-Host "  3. Try again: npm install" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

# Check if backend is running
Write-Host "🔍 Checking if backend is running..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction Stop
    Write-Host "✅ Backend is running!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend is not running!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To start the backend, run in another terminal:" -ForegroundColor Gray
    Write-Host "  cd E:\SummerProjects\reposynth" -ForegroundColor Cyan
    Write-Host "  docker-compose up" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Enter to continue anyway, or Ctrl+C to exit..." -ForegroundColor Yellow
    Read-Host
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  🚀 Starting Development Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend will be available at:" -ForegroundColor Green
Write-Host "  http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start dev server
npm run dev
