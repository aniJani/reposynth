# Rust Parser Daemon

A high-performance AST parser daemon for RepoSynth. Parses source files using tree-sitter and outputs AST nodes in JSONL format.

## Supported Languages

| Language | Extensions | Grammar |
|----------|------------|---------|
| Python | `.py` | tree-sitter-python |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| JavaScript | `.js`, `.jsx` | tree-sitter-typescript |
| CSS | `.css` | tree-sitter-css |
| SCSS | `.scss` | tree-sitter-scss |
| HTML | `.html`, `.htm` | tree-sitter-html |

## Setup

### 1. Install Tree-Sitter Grammars

**⚠️ IMPORTANT**: Grammar versions must be compatible with tree-sitter 0.22.

```bash
# Run the setup script (recommended)
./setup-grammars.sh      # Linux/macOS
.\setup-grammars.ps1     # Windows PowerShell
```

Or see `grammars/README.md` for manual installation.

### 2. Build

```bash
cargo build --release
```

The binary will be at `target/release/rust-parser-daemon`.

## Usage

The daemon communicates via stdin/stdout using JSONL (JSON Lines) protocol.

### Request Format

```json
{"id": "unique-id", "path": "/absolute/path/to/file.py"}
```

### Response Format

Success:
```json
{"id": "unique-id", "path": "/path/to/file.py", "ast": [...]}
```

Error:
```json
{"id": "unique-id", "path": "/path/to/file.py", "error": "Error message"}
```

## Docker Build

When building with Docker, ensure grammars are installed locally first:

```bash
./setup-grammars.sh
docker-compose build
```

The grammars are copied into the Docker image during build.

## Adding New Language Support

1. Add grammar to `grammars/README.md` with compatible version
2. Update `setup-grammars.sh` and `setup-grammars.ps1`
3. Add grammar compilation in `build.rs`
4. Add extern declaration and extension mapping in `src/main.rs`
5. Create a language adapter in `packages/python-orchestrator/orchestrator/language_adapter.py`

## Troubleshooting

### `LanguageError { version: 15 }`

Grammar version mismatch. Re-run `setup-grammars.sh` to get compatible versions.

### Daemon crashes on specific file types

Check that the grammar for that file type is installed and at a compatible version.
