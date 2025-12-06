# Tree-Sitter Grammars for RepoSynth Parser

This directory contains tree-sitter grammar submodules used by the Rust parser daemon.

## ⚠️ IMPORTANT: Version Compatibility

The Rust parser uses `tree-sitter = "0.22"`. Grammar versions **MUST** be compatible with this version.

Tree-sitter 0.22 uses **LANGUAGE_VERSION 14**. Grammars compiled for tree-sitter 0.23+ (LANGUAGE_VERSION 15) will cause runtime panics.

## Required Grammar Versions

| Grammar | Compatible Version | Repository |
|---------|-------------------|------------|
| tree-sitter-python | `v0.21.0` | https://github.com/tree-sitter/tree-sitter-python |
| tree-sitter-typescript | `v0.21.2` | https://github.com/tree-sitter/tree-sitter-typescript |
| tree-sitter-css | `v0.21.1` | https://github.com/tree-sitter/tree-sitter-css |
| tree-sitter-scss | `v1.0.0` | https://github.com/serenadeai/tree-sitter-scss |
| tree-sitter-html | `v0.20.4` | https://github.com/tree-sitter/tree-sitter-html |

## Installation

### Option 1: Run the setup script (Recommended)

```bash
# From the rust-parser-daemon directory
./setup-grammars.sh

# Or on Windows PowerShell
.\setup-grammars.ps1
```

### Option 2: Manual installation

```bash
cd packages/rust-parser-daemon/grammars

# Python
git clone https://github.com/tree-sitter/tree-sitter-python.git
cd tree-sitter-python && git checkout v0.21.0 && cd ..

# TypeScript (also handles JS/JSX/TSX)
git clone https://github.com/tree-sitter/tree-sitter-typescript.git
cd tree-sitter-typescript && git checkout v0.21.2 && cd ..

# CSS
git clone https://github.com/tree-sitter/tree-sitter-css.git
cd tree-sitter-css && git checkout v0.21.1 && cd ..

# SCSS
git clone https://github.com/serenadeai/tree-sitter-scss.git
cd tree-sitter-scss && git checkout v1.0.0 && cd ..

# HTML
git clone https://github.com/tree-sitter/tree-sitter-html.git
cd tree-sitter-html && git checkout v0.20.4 && cd ..
```

## Adding New Grammars

When adding support for a new language:

1. **Find a compatible version**: Check the grammar's releases for versions that support tree-sitter 0.22 (LANGUAGE_VERSION 14)
   - Look for tags like `v0.20.x` or `v0.21.x`
   - Check the grammar's `src/parser.c` for `LANGUAGE_VERSION` if unsure

2. **Add to setup script**: Update `setup-grammars.sh` and `setup-grammars.ps1`

3. **Update build.rs**: Add the grammar compilation in `build.rs`

4. **Update main.rs**: Add the extern declaration and file extension mapping

5. **Update this README**: Document the version in the table above

## Troubleshooting

### Error: `LanguageError { version: 15 }`

This means the grammar was compiled for tree-sitter 0.23+ but we're using 0.22.

**Fix**: Checkout an older, compatible version of the grammar (see table above).

### Error: `No such file or directory: parser.c`

The grammar wasn't cloned or is missing files.

**Fix**: Re-run the setup script or manually clone the grammar.

## Upgrading Tree-Sitter

If upgrading the `tree-sitter` crate version in `Cargo.toml`:

1. Update all grammars to versions compatible with the new tree-sitter version
2. Update this README with the new compatible versions
3. Update the setup scripts
4. Rebuild and test all supported file types
