#!/bin/bash

# --- run-pipeline.sh ---
#
# A script to run the full RepoSynth pipeline on a remote Git repository.
#
# Usage: ./run-pipeline.sh --repo <git-url> [--mode <semantic|hybrid|full>] [--with-security-scans]
#
# Modes:
#   semantic: Lightweight analysis with AST, graphs, metrics, and embeddings (loose files)
#   hybrid:   Adds variable registry and source spans, packaged as .zip archive
#   full:     Complete analysis with all features and raw AST files in .zip archive
#
# Examples:
#   ./run-pipeline.sh --repo https://github.com/expressjs/express --mode semantic
#   ./run-pipeline.sh --repo https://github.com/expressjs/express --mode hybrid
#   ./run-pipeline.sh --repo https://github.com/expressjs/express --mode full --with-security-scans

set -e # Exit immediately if a command exits with a non-zero status.

# --- Default Configuration ---
MODE="semantic"
GIT_URL=""
SECURITY_SCANS=""
TEMP_REPO_DIR="temp_repos"
ORCHESTRATOR_PACKAGE="packages.python-orchestrator.orchestrator"
VENV_PATH=".venv"

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --repo) GIT_URL="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --with-security-scans) SECURITY_SCANS="--with-security-scans" ;;
        --no-security-scans) SECURITY_SCANS="--no-with-security-scans" ;;
        -h|--help)
            echo "Usage: $0 --repo <git-url> [--mode <semantic|hybrid|full>] [--with-security-scans]"
            echo ""
            echo "Modes:"
            echo "  semantic: Lightweight analysis with AST, graphs, metrics, and embeddings"
            echo "  hybrid:   Adds variable registry and source spans (packaged as .zip)"
            echo "  full:     Complete analysis with all features including raw AST files"
            echo ""
            echo "Options:"
            echo "  --with-security-scans    Enable security vulnerability scanning"
            echo "  --no-security-scans      Disable security scanning (overrides mode default)"
            echo ""
            echo "Examples:"
            echo "  $0 --repo https://github.com/expressjs/express --mode semantic"
            echo "  $0 --repo https://github.com/expressjs/express --mode hybrid"
            echo "  $0 --repo https://github.com/expressjs/express --mode full"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$GIT_URL" ]; then
    echo "Error: --repo <git-url> is a required argument."
    echo "Usage: $0 --repo <git-url> [--mode <semantic|hybrid|full>]"
    echo "Use --help for more information."
    exit 1
fi

# Validate the mode
if [[ "$MODE" != "semantic" && "$MODE" != "hybrid" && "$MODE" != "full" ]]; then
    echo "Error: Invalid mode '$MODE'. Please use 'semantic', 'hybrid', or 'full'."
    exit 1
fi

# --- Pre-flight Checks ---
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Python virtual environment not found at '$VENV_PATH'."
    echo "Please run the setup instructions in the README."
    exit 1
fi

# --- Main Logic ---
REPO_NAME=$(basename "$GIT_URL" .git)
CLONE_PATH="$TEMP_REPO_DIR/$REPO_NAME"

echo "--- Starting RepoSynth Pipeline for: $GIT_URL (Mode: $MODE) ---"

# 1. Clean and Clone Repository
echo "[1/3] Cloning repository..."
rm -rf "$CLONE_PATH"
mkdir -p "$TEMP_REPO_DIR"
git clone --depth=1 --no-tags --no-recurse-submodules "$GIT_URL" "$CLONE_PATH"
echo "Repository cloned to: $CLONE_PATH"

# 2. Activate Virtual Environment and Run Python Pipeline
echo "[2/3] Running the analysis pipeline (mode: $MODE)..."
source "$VENV_PATH/bin/activate"

# Run the pipeline as a module, passing arguments directly
python3 -m "$ORCHESTRATOR_PACKAGE" --repo "$CLONE_PATH" --mode "$MODE" $SECURITY_SCANS

deactivate
echo "Pipeline completed successfully."

# 3. Final Output
echo "[3/3] Locating final pack..."
PACK_DIR="pack"

if [ -d "$PACK_DIR" ]; then
    echo ""
    echo "--- ✅ Pipeline Finished Successfully! ---"
    echo "The complete '$MODE' pack has been generated."

    # Check for zip archive (hybrid/full modes)
    if [ "$MODE" = "hybrid" ] || [ "$MODE" = "full" ]; then
        ZIP_FILE="reposynth_${REPO_NAME}_${MODE}.zip"
        if [ -f "$ZIP_FILE" ]; then
            echo ""
            echo "📦 Archive created: $ZIP_FILE"
            echo "   Size: $(du -h "$ZIP_FILE" | cut -f1)"
            echo ""
            echo "Extract the archive and review 'repoBrief.md' for a summary."
        fi
    else
        echo "   Location: $PACK_DIR/"
        echo "   Main summary: '$PACK_DIR/repoBrief.md'"
    fi

    # Display security scan results if available
    if [ -f "$PACK_DIR/security_report.json" ]; then
        echo ""
        echo "🔒 Security scan completed. Review '$PACK_DIR/security_report.json' for findings."
    fi
else
    echo "Error: Output pack directory '$PACK_DIR' not found." >&2
    exit 1
fi