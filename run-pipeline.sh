#!/bin/bash

# --- run-pipeline.sh ---
#
# A script to run the full RepoSynth pipeline on a remote Git repository.
#
# Usage: ./run-pipeline.sh <git-url>
# Example: ./run-pipeline.sh https://github.com/expressjs/express

# --- Configuration ---
# Directory to clone temporary repos into
TEMP_REPO_DIR="temp_repos"
# Main python orchestrator package path
ORCHESTRATOR_PACKAGE="packages.python-orchestrator.orchestrator"
# Python virtual environment path
VENV_PATH="packages/python-orchestrator/.venv"

# --- Pre-flight Checks ---
# Check if a Git URL was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a Git repository URL."
    echo "Usage: $0 <git-url>"
    exit 1
fi

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Python virtual environment not found at '$VENV_PATH'."
    echo "Please run the setup instructions in the README."
    exit 1
fi

# --- Main Logic ---

GIT_URL=$1
REPO_NAME=$(basename "$GIT_URL" .git)
CLONE_PATH="$TEMP_REPO_DIR/$REPO_NAME"

echo "--- Starting RepoSynth Pipeline for: $GIT_URL ---"

# 1. Clean and Clone the Repository (Sandboxed)
echo "[1/3] Cloning repository..."
rm -rf "$CLONE_PATH"
mkdir -p "$TEMP_REPO_DIR"
git clone --depth=1 --no-tags --no-recurse-submodules "$GIT_URL" "$CLONE_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone repository."
    exit 1
fi
echo "Repository cloned to: $CLONE_PATH"

# 2. Activate Virtual Environment and Run the Python Pipeline
echo "[2/3] Running the analysis pipeline..."
source "$VENV_PATH/bin/activate"

# Export the repo path so the Python script can use it
export REPOSYNTH_TARGET_REPO="$CLONE_PATH"

# Run the pipeline as a module
python3 -m "$ORCHESTRATOR_PACKAGE"
if [ $? -ne 0 ]; then
    echo "Error: Python pipeline failed."
    deactivate
    exit 1
fi
deactivate
echo "Pipeline completed successfully."

# 3. Final Output
echo "[3/3] Locating final pack..."
PACK_DIR="pack" # This is the output dir configured in the python script
if [ -d "$PACK_DIR" ]; then
    echo ""
    echo "--- ✅ Pipeline Finished Successfully! ---"
    echo "The complete Semantic Pack has been generated in the '$PACK_DIR' directory."
    echo "The main summary can be found at: '$PACK_DIR/repoBrief.md'"
else
    echo "Error: Output pack directory '$PACK_DIR' not found."
    exit 1
fi