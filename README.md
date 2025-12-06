# RepoSynth

> A powerful, configurable repository analysis and synthesis pipeline for extracting semantic insights from codebases.

RepoSynth is a comprehensive code analysis tool that parses repositories into structured artifacts including AST representations, dependency graphs, complexity metrics, semantic embeddings, and security reports. It's designed to help developers understand, document, and analyze codebases at scale!

## Features

### Core Analysis Pipeline

- **🌳 AST Parsing**: Multi-language parsing (Python, TypeScript, JavaScript) using Tree-sitter
- **🔗 Dependency Graphs**: Automatic import relationship mapping and module centrality analysis
- **📊 Static Analysis**: Cyclomatic complexity metrics using Ruff
- **🧠 Semantic Embeddings**: ML-powered embeddings for public APIs using SentenceTransformers
- **🔒 Security Scanning**: Detect hardcoded secrets and vulnerabilities (Week 7)
- **📝 Architectural Briefs**: Auto-generated repository summaries with hotspots and key modules
- **⚡ Intelligent Caching**: Commit-based caching for lightning-fast re-analysis

### Analysis Modes

RepoSynth supports three analysis modes with different trade-offs:

| Mode | Output Format | Features | Use Case |
|------|--------------|----------|----------|
| **Semantic** | Loose files (.md) | AST, graphs, metrics, embeddings | Quick insights, human-readable |
| **Hybrid** | .zip archive | + Variable registry, source spans | Complete analysis for tooling |
| **Full** | .zip archive | + Raw AST files | Maximum detail, archival |

## Quick Start

### Prerequisites

- Python 3.10+
- Rust (for building the parser daemon)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/aniJani/reposynth.git
cd reposynth

# Build the Rust parser daemon
cd packages/rust-parser-daemon
cargo build --release
cd ../..

# Create and activate virtual environment
cd packages/python-orchestrator
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r ../../requirements.txt

# Return to project root
cd ../..
```

### Usage

#### Using the Bash Script (Recommended)

```bash
# Make the script executable
chmod +x run-pipeline.sh

# Analyze a repository with semantic mode
./run-pipeline.sh --repo https://github.com/expressjs/express --mode semantic

# Analyze with hybrid mode (includes .zip packaging)
./run-pipeline.sh --repo https://github.com/expressjs/express --mode hybrid

# Full analysis with security scanning
./run-pipeline.sh --repo https://github.com/expressjs/express --mode full --with-security-scans
```

#### Using Python Directly

```bash
# Analyze current directory
python -m packages.python-orchestrator.orchestrator

# Analyze specific repository
python -m packages.python-orchestrator.orchestrator --repo /path/to/repo --mode hybrid

# Enable security scanning
python -m packages.python-orchestrator.orchestrator --repo /path/to/repo --with-security-scans

# Disable caching for fresh analysis
python -m packages.python-orchestrator.orchestrator --repo /path/to/repo --no-cache

# Get help
python -m packages.python-orchestrator.orchestrator --help
```

## Output Artifacts

After running the pipeline, you'll find the following artifacts in the `pack/` directory:

### Core Files (All Modes)

- **`repoBrief.md`**: Architectural summary with key modules, complexity hotspots, and public APIs
- **`manifest.json`**: Artifact metadata with SHA256 checksums for integrity verification
- **`name_registry.json`**: Symbol definitions with file locations and byte ranges
- **`import_graph.json`**: Module dependency relationships
- **`analysis.sqlite`**: SQLite database with complexity metrics
- **`vectors.faiss`**: FAISS vector index for semantic search
- **`vector_ids.json`**: Mapping between vector IDs and fully qualified names

### Hybrid/Full Mode Additional Files

- **`variable_registry.json`**: Variable declarations and scope tracking
- **`source_spans.json`**: Complete source code for all public APIs with byte ranges
- **`security_report.json`**: Security scan results (if enabled)
- **`ast_raw/`**: Raw AST files in JSONL format (full mode only)

**Note:** The `repoBrief.md` file includes a complete "Source Code for Public APIs" section with all public API source code embedded directly in the markdown for easy viewing.

### Archive Output (Hybrid/Full Modes)

Hybrid and full modes create a self-contained `.zip` archive:
- `reposynth_<repo-name>_hybrid.zip`
- `reposynth_<repo-name>_full.zip`

## Configuration Options

### Command-Line Flags

```bash
# Primary arguments
--repo PATH              # Path to repository (default: current directory)
--mode MODE              # semantic|hybrid|full (default: semantic)

# Feature toggles (override mode defaults)
--with-parsing           # Enable AST parsing (default: true)
--with-graphs            # Enable dependency graphs (default: true)
--with-analysis          # Enable complexity analysis (default: true)
--with-embeddings        # Enable semantic embeddings (default: true)
--with-security-scans    # Enable security scanning (new in Week 7)
--with-variable-registry # Enable variable tracking (hybrid/full)
--with-spans             # Enable source span storage (hybrid/full)

# Performance options
--no-cache               # Disable all caching (slower, ensures clean state)
```

### Mode Details

#### Semantic Mode (Default)
- **Best for**: Quick insights, documentation, human review
- **Output**: Loose files in `pack/` directory
- **Features**: AST parsing, graphs, complexity metrics, embeddings
- **Cache**: Full caching support
- **Speed**: ~2-5 minutes for medium repositories

#### Hybrid Mode
- **Best for**: Tool integration, detailed analysis
- **Output**: `.zip` archive with all artifacts
- **Features**: All semantic features + variable registry + source spans
- **Security**: Includes security scanning by default
- **Cache**: Full caching support
- **Speed**: ~3-7 minutes for medium repositories

#### Full Mode
- **Best for**: Archival, maximum detail, research
- **Output**: `.zip` archive with raw AST files
- **Features**: Everything + raw AST files
- **Security**: Includes security scanning by default
- **Cache**: Full caching support (AST files cached separately)
- **Speed**: ~4-10 minutes for medium repositories

## Week 7 Features (Latest)

### Security Gates

RepoSynth now includes comprehensive security scanning powered by `detect-secrets`:

- **Automatic Secret Detection**: Scans for hardcoded API keys, tokens, passwords
- **Severity Classification**: Categorizes findings by severity (high/medium)
- **Detailed Reports**: JSON output with file paths, line numbers, and issue types
- **Smart Caching**: Commit-based caching prevents redundant scans
- **Non-blocking**: Security issues are reported but don't halt the pipeline

Example security report:
```json
{
  "scan_timestamp": "2025-10-18T12:34:56.789Z",
  "repository": "/path/to/repo",
  "commit_hash": "abc123...",
  "summary": {
    "total_findings": 3,
    "high_severity": 1,
    "medium_severity": 2,
    "files_scanned": 2
  },
  "findings": [...]
}
```

### Pack Format Perfection

- **Semantic Mode**: Lightweight markdown and JSON files for human readability
- **Hybrid/Full Modes**: Complete `.zip` archives with README and all artifacts
- **Integrity Verification**: SHA256 checksums for all artifacts in manifest
- **Smart Packaging**: Only includes relevant files per mode (no bloat)

### CLI Polish

- **Rich Help Messages**: Comprehensive `--help` output with examples
- **Clear Descriptions**: Every flag has detailed explanation
- **Smart Defaults**: Mode-based configurations with override capability
- **Error Messages**: Helpful error messages with suggestions

## Advanced Usage

### API Server (Week 6 Feature)

RepoSynth includes a FastAPI-based token estimation API:

```bash
# Start the API server
python api_server.py

# Estimate tokens for a local repository
curl -X POST "http://localhost:8000/api/estimate" \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/repo"}'

# Estimate tokens for a GitHub repository
curl -X POST "http://localhost:8000/api/estimate-github" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/expressjs/express"}'
```

See [docs/WEEK6_API_USAGE.md](docs/WEEK6_API_USAGE.md) for full API documentation.

### Caching System

RepoSynth uses intelligent multi-stage caching stored in `.reposynth_cache/`:

- **Commit-based**: Cache invalidates automatically when code changes
- **Stage-level**: Each pipeline stage caches independently
- **Cross-repository**: Cache is shared across analysis runs
- **Persistent**: Survives repository deletion (useful for temp clones)

Cache hit example output:
```
⚡ CACHE HIT: Loading graphs from cache...
✓ Graphs loaded from cache (instant)
```

Disable caching when needed:
```bash
python -m orchestrator --repo /path/to/repo --no-cache
```

## Performance

Typical performance on a medium-sized repository (~100 files, ~10K LoC):

| Stage | Duration | Cacheable |
|-------|----------|-----------|
| Parsing | ~5 seconds | ✓ |
| Graphs | ~2 seconds | ✓ |
| Analysis | ~10 seconds | ✓ |
| Embeddings | ~15 seconds | ✓ |
| Security Scans | ~5 seconds | ✓ |
| Assembly | ~1 second | - |
| **Total (first run)** | **~40 seconds** | |
| **Total (cached)** | **~2 seconds** | |

## Architecture

```
reposynth/
├── packages/
│   ├── rust-parser-daemon/      # Tree-sitter-based AST parser (Rust)
│   └── python-orchestrator/      # Main pipeline orchestrator (Python)
│       └── orchestrator/
│           ├── __main__.py       # CLI entry point
│           ├── pipeline_runner.py # Core pipeline logic
│           ├── api.py            # FastAPI server
│           ├── estimator.py      # Token estimation
│           ├── language_adapter.py # Language-specific handlers
│           └── schemas.py        # Pydantic models
├── pack/                         # Output artifacts
├── .reposynth_cache/            # Persistent cache
├── run-pipeline.sh              # Bash runner script
├── run-pipeline.ps1             # PowerShell runner script
└── requirements.txt             # Python dependencies
```

## Documentation

- **[SETUP.md](SETUP.md)**: Comprehensive setup guide
- **[docs/WEEK6_API_USAGE.md](docs/WEEK6_API_USAGE.md)**: API documentation and examples

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Your License Here]

## Roadmap

### Completed
- ✅ Week 1-5: Core pipeline with AST, graphs, analysis, embeddings
- ✅ Week 6: FastAPI server with token estimation
- ✅ Week 7: Security scanning and pack format perfection

### Upcoming
- 🔄 Web interface for pipeline visualization
- 🔄 Additional language support (Java, Go, Ruby)
- 🔄 Advanced security rules and custom scanners
- 🔄 Cloud deployment support
- 🔄 Integration with popular CI/CD platforms

## Support

For issues, questions, or feature requests, please visit:
- [GitHub Issues](https://github.com/aniJani/reposynth/issues)
- [Documentation](https://github.com/aniJani/reposynth/wiki)

---

**RepoSynth** - Understand your codebase at a glance. 🚀
