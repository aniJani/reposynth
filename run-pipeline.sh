#!/bin/bash

# --- run-pipeline.sh ---
#
# A script to run the full RepoSynth pipeline on a remote Git repository.
#
# Usage: ./run-pipeline.sh --repo <git-url> [--mode <semantic|hybrid>]
#
# Example: ./run-pipeline.sh --repo https://github.com/expressjs/express --mode hybrid

set -e # Exit immediately if a command exits with a non-zero status.

# --- Default Configuration ---
MODE="semantic"
GIT_URL=""
TEMP_REPO_DIR="temp_repos"
ORCHESTRATOR_PACKAGE="packages.python-orchestrator.orchestrator"
VENV_PATH="packages/python-orchestrator/.venv"

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --repo) GIT_URL="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$GIT_URL" ]; then
    echo "Error: --repo <git-url> is a required argument."
    echo "Usage: $0 --repo <git-url> [--mode <semantic|hybrid>]"
    exit 1
fi

# Validate the mode
if [[ "$MODE" != "semantic" && "$MODE" != "hybrid" ]]; then
    echo "Error: Invalid mode '$MODE'. Please use 'semantic' or 'hybrid'."
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
echo "[2/3] Running the analysis pipeline..."
source "$VENV_PATH/bin/activate"

# Run the pipeline as a module, passing arguments directly
python3 -m "$ORCHESTRATOR_PACKAGE" --repo "$CLONE_PATH" --mode "$MODE"

deactivate
echo "Pipeline completed successfully."

# 3. Final Output
echo "[3/3] Locating final pack..."
PACK_DIR="pack"
if [ -d "$PACK_DIR" ]; then
    echo ""
    echo "--- ✅ Pipeline Finished Successfully! ---"
    echo "The complete '$MODE' pack has been generated in the '$PACK_DIR' directory."
    echo "The main summary can be found at: '$PACK_DIR/repoBrief.md'"
else
    echo "Error: Output pack directory '$PACK_DIR' not found." >&2
    exit 1
fi