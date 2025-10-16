import sys
from pathlib import Path
import argparse

# Ensure the orchestrator package is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.pipeline_runner import Pipeline

def main():
    """The main entrypoint for running the configurable RepoSynth pipeline."""
    
    parser = argparse.ArgumentParser(description="Run the RepoSynth analysis pipeline.")
    
    # --- Primary Arguments ---
    parser.add_argument(
        "--repo",
        type=str,
        help="Path to the local repository to analyze. If not provided, analyzes the RepoSynth project itself."
    )
    
    # --- Mode-based Configuration ---
    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        choices=["semantic", "hybrid", "full"],
        help="The high-level packaging mode to use (default: semantic)."
    )
    
    # --- Fine-grained Feature Toggles ---
    # These allow overriding the defaults set by the --mode flag.
    parser.add_argument("--with-parsing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-variable-registry", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--with-spans", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching for this run.")

    args = parser.parse_args()
    
    # --- Construct the Configuration Dictionary ---
    # Start with mode-based defaults
    if args.mode == "semantic":
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "build_variable_registry": False,
            "store_spans": False,
        }
    elif args.mode == "hybrid":
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "build_variable_registry": True,
            "store_spans": True,
        }
    else: # Default to semantic for safety
        config = {"run_parsing": True, "build_graphs": True, "run_analysis": True, "run_embeddings": True, "build_variable_registry": False, "store_spans": False}

    # Override defaults with specific command-line toggles
    config["run_parsing"] = args.with_parsing
    config["build_graphs"] = args.with_graphs
    config["run_analysis"] = args.with_analysis
    config["run_embeddings"] = args.with_embeddings
    config["no_cache"] = args.no_cache
    
    if args.with_variable_registry is not None:
        config["build_variable_registry"] = args.with_variable_registry
    if args.with_spans is not None:
        config["store_spans"] = args.with_spans
    
    # --- Path Resolution Logic ---
    root_dir = Path(__file__).parent.parent.parent.parent.resolve()
    
    if args.repo:
        repo_to_parse = Path(args.repo).resolve()
        print(f"--- Target repository: {repo_to_parse} ---")
    else:
        repo_to_parse = root_dir
        print(f"--- No --repo specified. Parsing the RepoSynth project itself. ---")

    daemon_path = root_dir / "packages/rust-parser-daemon/target/release/rust-parser-daemon"
    
    if sys.platform == "darwin": pass
    elif sys.platform == "win32": daemon_path = daemon_path.with_suffix('.exe')

    if not daemon_path.exists():
        print(f"FATAL: Daemon executable not found at path: {daemon_path}", file=sys.stderr)
        sys.exit(1)

    output_pack_dir = root_dir / "pack"
    
    print(f"--- Initializing Pipeline with Config: {config} ---")
    
    pipeline = Pipeline(
        repo_path=str(repo_to_parse),
        output_path=str(output_pack_dir),
        daemon_path=str(daemon_path),
    )
    pipeline.run(config=config)

if __name__ == '__main__':
    main()