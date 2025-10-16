# --- FILE: packages/python-orchestrator/orchestrator/__main__.py ---

import sys
import os
from pathlib import Path
import argparse

# When running as a module, Python adds the project root to the path,
# but to be safe, especially in IDEs, let's ensure our local modules are found.
# This makes `from orchestrator.pipeline_runner import Pipeline` work.
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.pipeline_runner import Pipeline


def main():
    """The main entrypoint for running the pipeline."""

    # --- This logic is moved from the bottom of pipeline_runner.py ---

    # Cleanly resolve the project's root directory
    parser = argparse.ArgumentParser(description="Run the RepoSynth analysis pipeline.")
    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        choices=["semantic", "hybrid"],
        help="The packaging mode to run.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to a local repository to analyze. Overrides environment variable.",
    )
    args = parser.parse_args()

    # target_repo_path_str = os.environ.get("REPOSYNTH_TARGET_REPO")

    root_dir = Path(__file__).parent.parent.parent.parent.resolve()
    repo_path_source = args.repo or os.environ.get("REPOSYNTH_TARGET_REPO")

    if repo_path_source:
        repo_to_parse = Path(repo_path_source).resolve()
        source_type = "command line" if args.repo else "environment variable"
        print(f"--- Target repository set from {source_type}: {repo_to_parse} ---")
    else:
        repo_to_parse = root_dir
        print(
            f"--- No target repository specified. Parsing the RepoSynth project itself. ---"
        )
    daemon_path = (
        root_dir / "packages/rust-parser-daemon/target/release/rust-parser-daemon"
    )

    if sys.platform == "darwin":  # Use darwin for macOS
        # On macOS, the executable won't have an extension
        pass
    elif sys.platform == "win32":
        daemon_path = daemon_path.with_suffix(".exe")

    if not daemon_path.exists():
        print(
            f"FATAL: Daemon executable not found at path: {daemon_path}",
            file=sys.stderr,
        )
        print(
            "Please run 'cargo build --release' in 'packages/rust-parser-daemon'",
            file=sys.stderr,
        )
        sys.exit(1)

    # We create a dedicated output directory for the pack
    output_pack_dir = root_dir / "pack"

    print("--- Initializing Pipeline ---")
    config = {"mode": args.mode}
    pipeline = Pipeline(
        repo_path=str(repo_to_parse),
        output_path=str(output_pack_dir),
        daemon_path=str(daemon_path),
        config=config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
