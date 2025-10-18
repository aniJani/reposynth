# RepoSynth Setup Guide

## Quick Start

### 1. Create Virtual Environment (Root Level)

```powershell
# Navigate to project root
cd C:\Users\rajka\reposynth

# Create virtual environment at root level
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your prompt:
```
(.venv) PS C:\Users\rajka\reposynth>
```

### 2. Install Dependencies

```powershell
# Install all Python dependencies (from root)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs:
- **Pipeline dependencies**: faiss, sentence-transformers, radon, ruff, etc.
- **Week 6 API dependencies**: fastapi, uvicorn, pydantic, tiktoken, pygount

### 3. Build Rust Parser Daemon

```powershell
# Navigate to Rust daemon
cd packages\rust-parser-daemon

# Build release binary
cargo build --release

# Return to root
cd ..\..
```

The daemon will be at: `packages\rust-parser-daemon\target\release\rust-parser-daemon.exe`

### 4. Verify Installation

```powershell
# Check Python dependencies
python -c "import fastapi, tiktoken, pygount; print('✓ All dependencies installed')"

# Check Rust daemon exists
if (Test-Path "packages\rust-parser-daemon\target\release\rust-parser-daemon.exe") {
    Write-Host "✓ Rust daemon built successfully"
} else {
    Write-Host "✗ Rust daemon not found"
}
```

---

## Running RepoSynth

### Option 1: Run Full Pipeline

```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Run pipeline on a GitHub repo
.\run-pipeline.ps1 --repo https://github.com/jquense/yup --mode hybrid
```

### Option 2: Run API Server (Week 6)

```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Start the API server
python api_server.py

# Or with auto-reload for development
python api_server.py --reload
```

Then open: `http://localhost:8000/docs`

---

## Project Structure

```
reposynth/
├── .venv/                     ← Virtual environment (root level)
├── requirements.txt           ← All Python dependencies (root level)
├── api_server.py              ← API server entry point
├── run-pipeline.ps1           ← Pipeline runner script
│
├── packages/
│   ├── python-orchestrator/
│   │   └── orchestrator/
│   │       ├── __main__.py    ← Pipeline CLI entry point
│   │       ├── pipeline_runner.py
│   │       ├── estimator.py   ← Week 6: Token estimator
│   │       ├── schemas.py     ← Week 6: Pydantic models
│   │       └── api.py         ← Week 6: FastAPI app
│   │
│   └── rust-parser-daemon/
│       ├── src/
│       └── target/release/
│           └── rust-parser-daemon.exe  ← Built binary
│
├── pack/                      ← Generated output artifacts
├── temp_repos/                ← Temporary cloned repos
└── .reposynth_cache/          ← Persistent cache (Week 5)
```

---

## Migrating from Old Setup

If you previously had a venv in `packages/python-orchestrator/.venv`:

### Step 1: Deactivate old venv
```powershell
deactivate
```

### Step 2: Remove old venv (optional)
```powershell
Remove-Item -Recurse -Force packages\python-orchestrator\.venv
```

### Step 3: Follow Quick Start above
Create new `.venv` at root and install dependencies.

---

## Troubleshooting

### Issue: `pip` command fails with "Unable to create process"

**Solution:** Use `python -m pip` instead of `pip`:
```powershell
python -m pip install -r requirements.txt
```

### Issue: Virtual environment not activating

**Solution:** Enable script execution (run as Administrator):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Rust daemon not found

**Solution:** Build the Rust daemon:
```powershell
cd packages\rust-parser-daemon
cargo build --release
```

### Issue: "tiktoken not found" when running estimator

**Solution:** Reinstall dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Issue: Old venv still being used

**Solution:** Completely restart your terminal and activate the new root venv:
```powershell
# Close terminal and reopen
cd C:\Users\rajka\reposynth
.\.venv\Scripts\Activate.ps1
```

---

## Development Workflow

### Daily Development

```powershell
# 1. Navigate to project
cd C:\Users\rajka\reposynth

# 2. Activate venv
.\.venv\Scripts\Activate.ps1

# 3. Work on code...

# 4. Test pipeline
.\run-pipeline.ps1 --repo https://github.com/some/repo --mode hybrid

# 5. Or test API
python api_server.py --reload
```

### Adding New Dependencies

```powershell
# Install the package
python -m pip install some-package

# Update requirements.txt
python -m pip freeze > requirements.txt
```

### Running Tests (Future)

```powershell
pytest tests/
```

---

## Summary

**TL;DR:**
1. Create `.venv` at **root level** (not in packages/)
2. Activate: `.\.venv\Scripts\Activate.ps1`
3. Install: `python -m pip install -r requirements.txt`
4. Run: `python api_server.py` or `.\run-pipeline.ps1`

Everything runs from the **project root** now! 🎉
