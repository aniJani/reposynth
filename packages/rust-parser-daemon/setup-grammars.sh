#!/bin/bash
# Setup script for tree-sitter grammars
# Run this from the rust-parser-daemon directory

set -e

GRAMMAR_DIR="grammars"
mkdir -p "$GRAMMAR_DIR"
cd "$GRAMMAR_DIR"

echo "=== Setting up Tree-Sitter Grammars ==="
echo "These versions are compatible with tree-sitter 0.22 (LANGUAGE_VERSION 14)"
echo ""

# Grammar versions - UPDATE THESE when changing tree-sitter version
declare -A GRAMMARS=(
    ["tree-sitter-python"]="v0.21.0|https://github.com/tree-sitter/tree-sitter-python.git"
    ["tree-sitter-typescript"]="v0.21.2|https://github.com/tree-sitter/tree-sitter-typescript.git"
    ["tree-sitter-css"]="v0.21.1|https://github.com/tree-sitter/tree-sitter-css.git"
    ["tree-sitter-scss"]="v1.0.0|https://github.com/serenadeai/tree-sitter-scss.git"
    ["tree-sitter-html"]="v0.20.4|https://github.com/tree-sitter/tree-sitter-html.git"
)

for grammar in "${!GRAMMARS[@]}"; do
    IFS='|' read -r version url <<< "${GRAMMARS[$grammar]}"
    
    echo "--- Setting up $grammar @ $version ---"
    
    if [ -d "$grammar" ]; then
        echo "  Directory exists, updating..."
        cd "$grammar"
        git fetch --tags
        git checkout "$version"
        cd ..
    else
        echo "  Cloning..."
        git clone "$url"
        cd "$grammar"
        git checkout "$version"
        cd ..
    fi
    
    echo "  ✓ $grammar ready"
    echo ""
done

echo "=== All grammars set up successfully! ==="
echo ""
echo "You can now build the Rust parser daemon:"
echo "  cargo build --release"
