cd packages/rust-parser-daemon
cargo build --release
cd ../..
cd packages/python-orchestrator

# Create a virtual environment using your modern Python 3.10+
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install required packages
pip install radon sentence-transformers faiss-cpu numpy

# Go back to the project root
cd ../..
python3 -m packages.python-orchestrator.orchestrator