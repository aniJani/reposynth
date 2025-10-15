# --- FILE: packages/python-orchestrator/orchestrator/__main__.py ---

import sys
from pathlib import Path

# When running as a module, Python adds the project root to the path,
# but to be safe, especially in IDEs, let's ensure our local modules are found.
# This makes `from orchestrator.pipeline_runner import Pipeline` work.
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.pipeline_runner import Pipeline

def main():
    """The main entrypoint for running the pipeline."""
    
    # --- This logic is moved from the bottom of pipeline_runner.py ---
    
    # Cleanly resolve the project's root directory
    root_dir = Path(__file__).parent.parent.parent.parent.resolve()
    daemon_path = root_dir / "packages/rust-parser-daemon/target/release/rust-parser-daemon"
    
    if sys.platform == "darwin": # Use darwin for macOS
        # On macOS, the executable won't have an extension
        pass
    elif sys.platform == "win32":
        daemon_path = daemon_path.with_suffix('.exe')

    if not daemon_path.exists():
        print(f"FATAL: Daemon executable not found at path: {daemon_path}", file=sys.stderr)
        print("Please run 'cargo build --release' in 'packages/rust-parser-daemon'", file=sys.stderr)
        sys.exit(1)

    # We create a dedicated output directory for the pack
    output_pack_dir = root_dir / "pack"
    
    print("--- Initializing Pipeline ---")
    pipeline = Pipeline(
        repo_path=str(root_dir),
        output_path=str(output_pack_dir),
        daemon_path=str(daemon_path)
    )
    pipeline.run()

if __name__ == '__main__':
    main()