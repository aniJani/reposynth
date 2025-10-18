# Week 7 Testing Guide

> Complete testing guide for RepoSynth Week 7 features: Security Gates & Final CLI Polish

## Overview

Week 7 adds the final core engine features:
- 🔒 **Security Scanning** with detect-secrets
- 📦 **Pack Format Perfection** (semantic/hybrid/full modes)
- 🎨 **CLI Polish** with comprehensive help
- ✅ **Bug Fixes** (CRLF, import resolution)

---

## Prerequisites

1. **Virtual environment activated:**
   ```powershell
   .venv\Scripts\activate
   ```

2. **Dependencies installed:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Rust parser daemon built:**
   ```powershell
   cd packages/rust-parser-daemon
   cargo build --release
   cd ../..
   ```

---

## Test 1: CLI Help Messages

**Purpose:** Verify comprehensive help documentation

```powershell
python -m packages.python-orchestrator.orchestrator --help
```

**Expected Output:**
- ✅ Clear description of all modes (semantic, hybrid, full)
- ✅ All feature flags documented (--with-parsing, --with-security-scans, etc.)
- ✅ Usage examples section
- ✅ Mode comparison table

**Pass Criteria:** Help output is clear, comprehensive, and well-formatted

---

## Test 2: Semantic Mode (Default - Lightweight)

**Purpose:** Test lightweight analysis with loose files

```powershell
# Clean up
Remove-Item -Recurse -Force pack -ErrorAction SilentlyContinue

# Run semantic mode
python -m packages.python-orchestrator.orchestrator
```

**Expected Behavior:**
- ✅ Parses repository files
- ✅ Builds import graph
- ✅ Runs static analysis
- ✅ Generates embeddings
- ✅ Creates repoBrief.md

**Expected Output Files in `pack/`:**
```
pack/
├── name_registry.json
├── import_graph.json
├── analysis.sqlite
├── vectors.faiss
├── vector_ids.json
├── repoBrief.md
└── manifest.json
```

**Verify:**
```powershell
Get-ChildItem pack

# Check the brief
cat pack/repoBrief.md | Select-String "Key Architectural Modules" -Context 3

# Verify function names are correct (not fragments)
cat pack/name_registry.json | Select-String "api_server.py:main"
```

**Pass Criteria:**
- ✅ All files created
- ✅ RepoBrief shows correct function names (e.g., "main" not "poin")
- ✅ Import graph shows connections (api.py imports schemas.py, etc.)
- ✅ No .zip archive created

---

## Test 3: Hybrid Mode (Recommended - Complete Analysis)

**Purpose:** Test full analysis with security scanning and .zip packaging

```powershell
# Clean up
Remove-Item -Recurse -Force pack -ErrorAction SilentlyContinue
Remove-Item *.zip -ErrorAction SilentlyContinue

# Run hybrid mode
python -m packages.python-orchestrator.orchestrator --mode hybrid
```

**Expected Behavior:**
- ✅ All semantic features
- ✅ Security scanning runs (detect-secrets)
- ✅ Variable registry built
- ✅ Source spans stored
- ✅ Creates .zip archive

**Expected Output:**
```
reposynth_reposynth_hybrid.zip (~1-2 MB)

pack/
├── name_registry.json
├── import_graph.json
├── variable_registry.json          ← NEW
├── analysis.sqlite
├── vectors.faiss
├── vector_ids.json
├── security_report.json            ← NEW
├── spans.zip                       ← NEW
├── repoBrief.md
└── manifest.json
```

**Verify:**
```powershell
# Check .zip was created
Get-ChildItem *.zip

# Check security report
cat pack/security_report.json | ConvertFrom-Json | Select-Object scan_timestamp, summary

# Check variable registry
cat pack/variable_registry.json | Select-String "api_server.py" -Context 2

# Extract and verify archive
Expand-Archive -Path reposynth_reposynth_hybrid.zip -DestinationPath test_hybrid -Force
cat test_hybrid/README.md
Get-ChildItem test_hybrid/pack
```

**Pass Criteria:**
- ✅ .zip archive created (~1-2 MB)
- ✅ security_report.json contains scan results
- ✅ variable_registry.json tracks variables
- ✅ spans.zip contains source code spans
- ✅ Archive includes README.md

---

## Test 4: Full Mode (Maximum Detail - With Raw AST)

**Purpose:** Test complete analysis with raw AST files

```powershell
# Clean up
Remove-Item -Recurse -Force pack -ErrorAction SilentlyContinue
Remove-Item *.zip -ErrorAction SilentlyContinue

# Run full mode
python -m packages.python-orchestrator.orchestrator --mode full
```

**Expected Behavior:**
- ✅ All hybrid features
- ✅ Includes raw AST files in archive
- ✅ Larger archive size (~4-5 MB)

**Expected Output:**
```
reposynth_reposynth_full.zip (~4 MB)

pack/
├── ast_raw/                        ← NEW (42 .jsonl files)
│   ├── api_server.py.jsonl
│   ├── packages_python-orchestrator_orchestrator_api.py.jsonl
│   └── ... (40+ more files)
├── name_registry.json
├── import_graph.json
├── variable_registry.json
├── analysis.sqlite
├── vectors.faiss
├── vector_ids.json
├── security_report.json
├── spans.zip
├── repoBrief.md
└── manifest.json
```

**Verify:**
```powershell
# Check archive size
Get-ChildItem reposynth_reposynth_full.zip | Select-Object Name, Length

# Count AST files
(Get-ChildItem pack/ast_raw/*.jsonl).Count

# Extract and verify
Expand-Archive -Path reposynth_reposynth_full.zip -DestinationPath test_full -Force
Get-ChildItem test_full/pack/ast_raw | Measure-Object
```

**Pass Criteria:**
- ✅ .zip archive created (~4 MB)
- ✅ ast_raw/ directory contains 42+ .jsonl files
- ✅ Archive is significantly larger than hybrid mode
- ✅ All hybrid features included

---

## Test 5: Caching System

**Purpose:** Verify intelligent caching works across runs

```powershell
# First run (no cache)
Remove-Item -Recurse -Force .reposynth_cache -ErrorAction SilentlyContinue
Measure-Command { python -m packages.python-orchestrator.orchestrator }

# Second run (with cache)
Measure-Command { python -m packages.python-orchestrator.orchestrator }
```

**Expected Behavior:**
- ✅ First run: ~30-60 seconds (builds everything)
- ✅ Second run: ~2-5 seconds (cache hits)
- ✅ Output shows "⚡ CACHE HIT" messages

**Verify:**
```powershell
# Check cache directory exists
Get-ChildItem .reposynth_cache/reposynth
```

**Pass Criteria:**
- ✅ Second run is 10-20x faster
- ✅ All stages show cache hits
- ✅ Output is identical despite using cache

---

## Test 6: Feature Toggles

**Purpose:** Verify individual features can be toggled

```powershell
# Disable embeddings
python -m packages.python-orchestrator.orchestrator --no-with-embeddings

# Enable security scanning in semantic mode
python -m packages.python-orchestrator.orchestrator --mode semantic --with-security-scans

# Disable cache
python -m packages.python-orchestrator.orchestrator --no-cache
```

**Expected Behavior:**
- ✅ `--no-with-embeddings` skips embedding generation
- ✅ `--with-security-scans` adds security_report.json even in semantic mode
- ✅ `--no-cache` forces fresh analysis

**Pass Criteria:**
- ✅ Flags override mode defaults correctly
- ✅ Pipeline adapts to configuration
- ✅ No errors when features disabled

---

## Test 7: Import Graph & RepoBrief Quality

**Purpose:** Verify all critical bug fixes are working

```powershell
# Run analysis
python -m packages.python-orchestrator.orchestrator

# Check function names
cat pack/name_registry.json | Select-String "api_server.py"
cat pack/name_registry.json | Select-String "main"

# Check import graph
cat pack/import_graph.json | ConvertFrom-Json | Select-Object -ExpandProperty "packages\python-orchestrator\orchestrator\api.py"

# Check brief
cat pack/repoBrief.md
```

**Expected Results:**
- ✅ Function names: `"api_server.py:main"` (not `"poin"` or other fragments)
- ✅ API signatures: `async def value_error_handler(request, exc)` (not broken text)
- ✅ Import graph: `api.py` imports `schemas.py`, `estimator.py`, `git_utils.py`
- ✅ Brief: "Imports 3 other local modules" for api.py

**Pass Criteria:**
- ✅ No broken function names or fragments
- ✅ Import graph shows internal dependencies
- ✅ RepoBrief is human-readable and accurate

---

## Test 8: External Repository (Optional)

**Purpose:** Test on a real open-source project

```powershell
# Analyze Flask (small, well-known project)
python -m packages.python-orchestrator.orchestrator --repo https://github.com/pallets/click --mode hybrid
```

**Expected Behavior:**
- ✅ Clones repository to temp directory
- ✅ Runs full analysis
- ✅ Creates `reposynth_click_hybrid.zip`
- ✅ Brief shows meaningful architecture insights

**Pass Criteria:**
- ✅ Analysis completes successfully
- ✅ Archive created with correct name
- ✅ RepoBrief shows Flask's architecture

---

## Bug Fixes Verification

### CRLF Line Ending Bug (CRITICAL FIX)
**Issue:** On Windows, Python's `open()` normalizes `\r\n` → `\n`, shifting byte positions

**Fix Applied:** All `open()` calls for source code now use `newline=''`

**Verification:**
```powershell
# Should show "main" not "poin"
cat pack/name_registry.json | Select-String "api_server.py:main"
```

### Relative Import Resolution (CRITICAL FIX)
**Issue:** Imports like `from .schemas import (...)` weren't being resolved

**Fix Applied:** Properly strip leading dots and navigate directories

**Verification:**
```powershell
# Should show api.py imports schemas.py, estimator.py, git_utils.py
cat pack/import_graph.json | Select-String "api.py" -Context 2
```

---

## Success Criteria Summary

All tests must pass:
- [x] CLI help is comprehensive and clear
- [x] Semantic mode creates loose files
- [x] Hybrid mode creates .zip with security scanning
- [x] Full mode includes raw AST files
- [x] Caching provides 10-20x speedup
- [x] Feature toggles work correctly
- [x] Function names are correct (CRLF fix)
- [x] Import graph shows connections (relative imports fix)
- [x] RepoBrief is readable and accurate

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'faiss'"
**Solution:** Ensure virtual environment is activated and dependencies installed
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "FATAL: Daemon executable not found"
**Solution:** Build the Rust parser daemon
```powershell
cd packages/rust-parser-daemon
cargo build --release
cd ../..
```

### Issue: "detect-secrets not found"
**Solution:** Install security scanning dependencies
```powershell
pip install detect-secrets
```

### Issue: Slow performance
**Solution:** Ensure cache is working. Delete `.reposynth_cache` and run twice to verify caching

---

## Next Steps After Testing

1. **Commit Changes:**
   ```powershell
   git add .
   git commit -m "Week 7: Security scanning, pack modes, CRLF & import fixes"
   ```

2. **Create Documentation:**
   - Update main README.md with Week 7 features
   - Document the three modes
   - Add troubleshooting guide

3. **Optional Enhancements:**
   - Test on more external repositories
   - Profile performance bottlenecks
   - Add integration tests

---

## Questions?

- Check `--help` for all available options
- Review `pack/repoBrief.md` for architectural insights
- Examine `pack/manifest.json` for artifact details

**RepoSynth Week 7 is complete!** 🎉
